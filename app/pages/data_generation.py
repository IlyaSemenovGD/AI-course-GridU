"""
Data Generation page.

Layout (mirrors UI_interface_sample.png):
  ┌─────────────────────────────────────────────────────────────┐
  │ Prompt / instructions  (text_area)                          │
  │ Upload DDL Schema      (file_uploader)                      │
  │ Advanced Parameters    (temperature slider, rows input)     │
  │ [Generate]                                                  │
  ├─────────────────────────────────────────────────────────────┤
  │ Data Preview           (tab per table)                      │
  │   ┌── table dataframe ────────────────────────────────────┐ │
  │   │ quick-edit instructions  [Submit]                     │ │
  │   └───────────────────────────────────────────────────────┘ │
  │ [Download CSV / ZIP]   [Save to Database]                   │
  └─────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from google import genai

from app.core.data_generator import DataGenerator
from app.core.database import db_manager_from_env
from app.core.ddl_parser import TableDef, parse_ddl, topological_sort
from app.utils.observability import get_langfuse

# ── constants ────────────────────────────────────────────────────────────────
_DEFAULT_MODEL = "gemini-2.5-flash"
_SAMPLE_SCHEMAS_DIR = Path(__file__).parent.parent.parent  # project root


# ── cached resources ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_gemini_client() -> genai.Client:
    # Prefer AI Studio API key if provided (no GCP project / Vertex AI needed)
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        return genai.Client(api_key=api_key)

    # Fall back to Vertex AI ADC
    project = os.environ.get("GCP_PROJECT", "")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    if not project:
        st.error(
            "Set either GOOGLE_API_KEY (AI Studio) or GCP_PROJECT (Vertex AI) "
            "in your .env file and restart."
        )
        st.stop()
    return genai.Client(vertexai=True, project=project, location=location)


@st.cache_resource(show_spinner=False)
def _get_db():
    db = db_manager_from_env()
    try:
        db.initialise()
    except Exception:
        pass  # DB might not be available; handled gracefully
    return db


# ── session-state initialiser ────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "ddl_text": "",
        "ddl_filename": "",
        "parsed_tables": [],        # list[TableDef]
        "sorted_tables": [],        # topological order
        "circular_fks": set(),
        "generated_data": {},       # table_name → pd.DataFrame
        "generation_done": False,
        "saved_dataset_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── download helpers ──────────────────────────────────────────────────────────

def _df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _all_tables_to_zip(data: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for tbl_name, df in data.items():
            csv_bytes = _df_to_csv_bytes(df)
            zf.writestr(f"{tbl_name}.csv", csv_bytes)
    buf.seek(0)
    return buf.read()


# ── generation logic (runs synchronously, updates progress bar) ──────────────

def _run_generation(
    sorted_tables: list[TableDef],
    circular_fks: set,
    instructions: str,
    rows_per_table: int,
    temperature: float,
    model: str,
) -> dict[str, pd.DataFrame]:
    client = _get_gemini_client()
    langfuse = get_langfuse()
    generator = DataGenerator(client=client, model=model, langfuse=langfuse)

    total = len(sorted_tables)
    progress_bar = st.progress(0, text="Starting generation…")
    status_text = st.empty()

    generated: dict[str, pd.DataFrame] = {}

    for idx, table in enumerate(sorted_tables):
        status_text.info(f"Generating table **{table.name}** ({idx + 1}/{total})…")
        progress_bar.progress(idx / total, text=f"Table {idx + 1}/{total}: {table.name}")

        # Collect parent IDs from already-generated tables
        all_data_lower = {k.lower(): v for k, v in generated.items()}
        parent_ids: dict[str, list] = {}
        for fk in table.foreign_keys:
            if (table.name, fk.column) in circular_fks:
                continue
            ref_key = fk.references_table.lower()
            if ref_key in all_data_lower:
                df_ref = all_data_lower[ref_key]
                if fk.references_column in df_ref.columns:
                    parent_ids[ref_key] = df_ref[fk.references_column].dropna().tolist()

        # Stream generation with a live token counter
        stream_container = st.empty()
        accumulated = ""
        chunks_received = 0
        for chunk in generator.stream_table_generation(
            table=table,
            parent_ids=parent_ids,
            instructions=instructions,
            num_rows=min(rows_per_table, DataGenerator.BATCH_SIZE),
            temperature=temperature,
        ):
            accumulated += chunk
            chunks_received += 1
            if chunks_received % 10 == 0:
                stream_container.caption(f"Receiving data… {len(accumulated)} chars")

        stream_container.empty()

        # Parse streamed JSON
        raw = accumulated.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            rows_data = json.loads(raw)
            if not isinstance(rows_data, list):
                rows_data = [rows_data]
        except Exception:
            rows_data = DataGenerator._fallback_rows(table, min(rows_per_table, 10), 0)

        # If more rows needed, generate remaining batches
        batch_size = DataGenerator.BATCH_SIZE
        if rows_per_table > batch_size:
            remaining = rows_per_table - len(rows_data)
            extra_batches = (remaining + batch_size - 1) // batch_size
            for b in range(extra_batches):
                offset = len(rows_data)
                batch_n = min(batch_size, rows_per_table - offset)
                if batch_n <= 0:
                    break
                extra_accumulated = ""
                for chunk in generator.stream_table_generation(
                    table=table,
                    parent_ids=parent_ids,
                    instructions=instructions,
                    num_rows=batch_n,
                    temperature=temperature,
                    id_offset=offset,
                ):
                    extra_accumulated += chunk
                try:
                    extra_raw = extra_accumulated.strip()
                    extra_raw = re.sub(r"^```(?:json)?\s*", "", extra_raw)
                    extra_raw = re.sub(r"\s*```$", "", extra_raw)
                    extra_rows = json.loads(extra_raw)
                    if isinstance(extra_rows, list):
                        rows_data.extend(extra_rows)
                except Exception:
                    pass

        df = pd.DataFrame(rows_data)

        # Fix auto-increment PKs and FK columns
        all_gen_lower = {k.lower(): v for k, v in generated.items()}
        for col in table.columns:
            if col.is_auto_increment:
                df[col.name] = range(1, len(df) + 1)
        for fk in table.foreign_keys:
            if (table.name, fk.column) in circular_fks:
                if fk.column in df.columns:
                    df[fk.column] = None
            else:
                ref_key = fk.references_table.lower()
                if ref_key in all_gen_lower:
                    ref_df = all_gen_lower[ref_key]
                    if fk.references_column in ref_df.columns:
                        valid_ids = ref_df[fk.references_column].dropna().tolist()
                        if valid_ids and fk.column in df.columns:
                            df[fk.column] = [random.choice(valid_ids) for _ in range(len(df))]

        generated[table.name] = df

    # Resolve circular FKs now that all tables exist
    for tbl_name, col_name in circular_fks:
        tbl_obj = next((t for t in sorted_tables if t.name == tbl_name), None)
        if not tbl_obj:
            continue
        fk = next((f for f in tbl_obj.foreign_keys if f.column == col_name), None)
        if not fk:
            continue
        ref_df = generated.get(fk.references_table)
        if ref_df is not None and fk.references_column in ref_df.columns:
            valid_ids = ref_df[fk.references_column].dropna().tolist()
            df = generated[tbl_name]
            if valid_ids and col_name in df.columns:
                generated[tbl_name][col_name] = [
                    random.choice(valid_ids) if random.random() < 0.7 else None
                    for _ in range(len(df))
                ]

    progress_bar.progress(1.0, text="Generation complete!")
    status_text.success(f"Generated data for {total} table(s).")
    return generated


# ── main render function ──────────────────────────────────────────────────────

def render() -> None:
    _init_state()

    st.header("Data Generation")

    # ── 1. Instructions ──────────────────────────────────────────────────────
    instructions = st.text_area(
        "Prompt",
        placeholder="Describe the kind of data you want…  "
        "e.g. 'US-based tech companies, employees earning $60k–$200k, "
        "projects started after 2020'",
        height=80,
        key="instructions_input",
    )

    # ── 2. DDL Upload ────────────────────────────────────────────────────────
    col_upload, col_sample = st.columns([3, 2])
    with col_upload:
        uploaded = st.file_uploader(
            "Upload DDL Schema",
            type=["sql", "ddl", "txt"],
            help="Supported formats: .sql  .ddl  .txt",
            key="ddl_uploader",
        )
    with col_sample:
        st.markdown("**Or use a sample schema:**")
        sample_schemas = {
            "— select —": "",
            "Company & Employees": "company_employee_schema.ddl",
            "Restaurants": "restrurants_schema.ddl",
            "Library Management": "library_mgm_schema.ddl",
        }
        sample_choice = st.selectbox(
            "Sample schemas",
            list(sample_schemas.keys()),
            label_visibility="collapsed",
            key="sample_schema_select",
        )

    # Resolve DDL text source
    if uploaded is not None:
        ddl_text = uploaded.read().decode("utf-8", errors="replace")
        ddl_filename = uploaded.name
    elif sample_choice and sample_schemas[sample_choice]:
        schema_path = _SAMPLE_SCHEMAS_DIR / sample_schemas[sample_choice]
        ddl_text = schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""
        ddl_filename = sample_schemas[sample_choice]
    else:
        ddl_text = st.session_state.ddl_text
        ddl_filename = st.session_state.ddl_filename

    # Parse DDL when it changes
    if ddl_text and ddl_text != st.session_state.ddl_text:
        st.session_state.ddl_text = ddl_text
        st.session_state.ddl_filename = ddl_filename
        try:
            tables = parse_ddl(ddl_text)
            sorted_tbls, circ_fks = topological_sort(tables)
            st.session_state.parsed_tables = tables
            st.session_state.sorted_tables = sorted_tbls
            st.session_state.circular_fks = circ_fks
            st.session_state.generation_done = False
            st.session_state.generated_data = {}
        except Exception as exc:
            st.error(f"DDL parse error: {exc}")

    if st.session_state.ddl_text:
        n_tables = len(st.session_state.parsed_tables)
        st.caption(
            f"Loaded **{st.session_state.ddl_filename}** — "
            f"{n_tables} table(s) detected: "
            + ", ".join(f"`{t.name}`" for t in st.session_state.parsed_tables)
        )
        if st.session_state.circular_fks:
            circ_list = ", ".join(
                f"`{t}.{c}`" for t, c in st.session_state.circular_fks
            )
            st.info(
                f"Circular FK columns detected ({circ_list}) — "
                "they will be filled after all tables are generated."
            )

    # ── 3. Advanced parameters ───────────────────────────────────────────────
    with st.expander("Advanced Parameters", expanded=True):
        param_col1, param_col2, param_col3 = st.columns([3, 1, 1])
        with param_col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.9,
                step=0.05,
                help="Higher = more creative / diverse data",
            )
        with param_col2:
            rows_per_table = st.number_input(
                "Rows per table",
                min_value=1,
                max_value=1000,
                value=50,
                step=10,
            )
        with param_col3:
            model = st.selectbox(
                "Model",
                ["gemini-2.5-flash", "gemini-3.1-flash-lite-preview"],
                index=0,
                key="model_select",
            )

    # ── 4. Generate button ───────────────────────────────────────────────────
    generate_disabled = not st.session_state.ddl_text
    if st.button("Generate", type="primary", disabled=generate_disabled, use_container_width=False):
        if not st.session_state.sorted_tables:
            st.warning("No tables found in the DDL. Please upload a valid schema.")
        else:
            with st.spinner(""):
                generated = _run_generation(
                    sorted_tables=st.session_state.sorted_tables,
                    circular_fks=st.session_state.circular_fks,
                    instructions=instructions,
                    rows_per_table=rows_per_table,
                    temperature=temperature,
                    model=model,
                )
            st.session_state.generated_data = generated
            st.session_state.generation_done = True
            st.session_state.saved_dataset_id = None
            st.rerun()

    # ── 5. Data preview & per-table edit ────────────────────────────────────
    if st.session_state.generation_done and st.session_state.generated_data:
        _render_data_preview()


# ---------------------------------------------------------------------------
# Data preview section (extracted for readability)
# ---------------------------------------------------------------------------

def _render_data_preview() -> None:
    data: dict[str, pd.DataFrame] = st.session_state.generated_data
    table_names = list(data.keys())

    st.divider()
    st.subheader("Data Preview")

    tabs = st.tabs(table_names)
    for tab, tbl_name in zip(tabs, table_names):
        with tab:
            df = data[tbl_name]
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.caption(f"{len(df)} rows × {len(df.columns)} columns")

            # ── Per-table modification ──────────────────────────────────────
            mod_col, btn_col = st.columns([5, 1])
            with mod_col:
                mod_instruction = st.text_input(
                    "Quick edit instructions",
                    placeholder="e.g. 'Set all salaries between $70k–$180k' or "
                    "'Change status distribution to 60% Active, 40% Inactive'",
                    key=f"mod_{tbl_name}",
                    label_visibility="collapsed",
                )
            with btn_col:
                submit = st.button("Submit", key=f"submit_{tbl_name}", use_container_width=True)

            if submit and mod_instruction.strip():
                table_def = next(
                    (t for t in st.session_state.parsed_tables if t.name == tbl_name),
                    None,
                )
                if table_def is None:
                    st.error("Table definition not found.")
                else:
                    client = _get_gemini_client()
                    langfuse = get_langfuse()
                    generator = DataGenerator(
                        client=client,
                        model=st.session_state.get("model_select", _DEFAULT_MODEL),
                        langfuse=langfuse,
                    )
                    with st.spinner(f"Applying changes to {tbl_name}…"):
                        new_df, msgs = generator.modify_table_data(
                            df=df,
                            table=table_def,
                            instruction=mod_instruction,
                            temperature=0.5,
                        )
                    st.session_state.generated_data[tbl_name] = new_df
                    for msg in msgs:
                        st.success(msg)
                    st.rerun()

    # ── Download & Save ─────────────────────────────────────────────────────
    st.divider()
    dl_col1, dl_col2, save_col = st.columns([2, 2, 2])

    with dl_col1:
        zip_bytes = _all_tables_to_zip(data)
        st.download_button(
            label="Download all as ZIP",
            data=zip_bytes,
            file_name="generated_data.zip",
            mime="application/zip",
            use_container_width=True,
        )

    with dl_col2:
        # Selectbox to choose which table to download individually
        chosen = st.selectbox(
            "Download single table",
            table_names,
            key="single_dl_select",
            label_visibility="collapsed",
        )
        if chosen:
            st.download_button(
                label=f"Download {chosen}.csv",
                data=_df_to_csv_bytes(data[chosen]),
                file_name=f"{chosen}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with save_col:
        if st.session_state.saved_dataset_id:
            st.success(f"Saved (dataset id={st.session_state.saved_dataset_id})")
        else:
            dataset_name = st.text_input(
                "Dataset name",
                value=Path(st.session_state.ddl_filename).stem or "dataset",
                key="dataset_name_input",
                label_visibility="collapsed",
                placeholder="Dataset name for DB",
            )
            if st.button("Save to Database", use_container_width=True):
                db = _get_db()
                if not db.ping():
                    st.error("Cannot reach PostgreSQL. Check POSTGRES_* env vars.")
                else:
                    with st.spinner("Saving to database…"):
                        try:
                            dataset_id = db.save_dataset(
                                name=dataset_name,
                                tables_data=data,
                                ddl_text=st.session_state.ddl_text,
                                rows_per_table=len(next(iter(data.values()))),
                            )
                            st.session_state.saved_dataset_id = dataset_id
                            st.success(f"Saved as dataset id={dataset_id}.")
                        except Exception as exc:
                            st.error(f"Save failed: {exc}")
