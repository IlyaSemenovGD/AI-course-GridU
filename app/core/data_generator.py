"""
DataGenerator – uses Gemini 2.0 Flash (via Vertex AI) to create synthetic data.

Key design decisions:
  - Streaming   : generate_content_stream is used so the caller can relay
                  incremental progress to the UI.
  - Structured output: response_mime_type="application/json" ensures valid JSON.
  - Function calling  : used in modify_table_data so Gemini decides which
                        transformation to apply (update range, replace values, etc.)
  - FK integrity : tables are generated in topological order; circular FK
                   columns are filled after all tables exist.
"""

from __future__ import annotations

import json
import random
import re
from typing import Callable, Generator, Optional

import pandas as pd
from google import genai
from google.genai import types

from .ddl_parser import ColumnDef, ForeignKey, TableDef


# ---------------------------------------------------------------------------
# Type-mapping helpers
# ---------------------------------------------------------------------------

_BASE_TYPE_MAP: dict[str, str] = {
    "int": "integer",
    "integer": "integer",
    "bigint": "integer",
    "smallint": "integer",
    "tinyint": "integer",
    "serial": "integer",
    "bigserial": "integer",
    "float": "number",
    "double": "number",
    "real": "number",
    "decimal": "number",
    "numeric": "number",
    "boolean": "boolean",
    "bool": "boolean",
    # everything else → string
}


def _base_type(data_type: str) -> str:
    """Return the normalised base type keyword from e.g. VARCHAR(255) → varchar."""
    m = re.match(r"(\w+)", data_type.lower())
    return m.group(1) if m else "varchar"


def _json_type(col: ColumnDef) -> str:
    if col.enum_values:
        return "string"
    return _BASE_TYPE_MAP.get(_base_type(col.data_type), "string")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _generation_prompt(
    table: TableDef,
    parent_ids: dict[str, list],   # fk_ref_table_lower → list of id values
    instructions: str,
    num_rows: int,
    id_offset: int,
    batch_seed: int,
) -> str:
    schema = json.dumps(table.to_schema_dict(), indent=2)

    fk_hints = ""
    for fk in table.foreign_keys:
        ids = parent_ids.get(fk.references_table.lower(), [])
        if ids:
            sample = ids[: min(30, len(ids))]
            fk_hints += (
                f"\n- Column `{fk.column}` MUST be one of these existing IDs "
                f"from {fk.references_table}: {sample}"
            )

    auto_inc_cols = [c.name for c in table.columns if c.is_auto_increment]
    if auto_inc_cols:
        pk_hint = (
            f"\n- Auto-increment columns {auto_inc_cols} must be sequential "
            f"integers starting at {id_offset + 1}."
        )
    else:
        pk_hint = ""

    return f"""Generate exactly {num_rows} realistic synthetic data rows for the SQL table below.

TABLE SCHEMA:
{schema}

RULES:
- Return ONLY a valid JSON array of {num_rows} objects – no markdown, no prose.
- Every object must include all non-nullable columns.
- Respect ENUM values exactly as listed; do NOT invent new values.
- Dates → "YYYY-MM-DD", datetimes/timestamps → "YYYY-MM-DD HH:MM:SS".
- Decimal/numeric values must include decimal places where appropriate.
- Data must be realistic and contextually coherent (not placeholder text like "value_1").\
{pk_hint}{fk_hints}

USER INSTRUCTIONS:
{instructions or "Generate diverse, realistic data representative of a production database."}

(Internal seed: {batch_seed})"""


# ---------------------------------------------------------------------------
# Modification tools (for function-calling flow)
# ---------------------------------------------------------------------------

_MODIFY_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="set_column_range",
            description=(
                "Set all values of a numeric column to random numbers within "
                "[min_val, max_val]. Use for salary ranges, ratings, ages, etc."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "column":  {"type": "STRING",  "description": "Column name"},
                    "min_val": {"type": "NUMBER",   "description": "Minimum value (inclusive)"},
                    "max_val": {"type": "NUMBER",   "description": "Maximum value (inclusive)"},
                    "decimals": {"type": "INTEGER", "description": "Decimal places, 0 for integer"},
                },
                "required": ["column", "min_val", "max_val"],
            },
        ),
        types.FunctionDeclaration(
            name="replace_enum_values",
            description=(
                "Replace values in an ENUM or categorical column. "
                "Provide distribution as a list of (value, weight) pairs."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "column": {"type": "STRING"},
                    "values": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "Allowed values to use",
                    },
                    "weights": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"},
                        "description": "Optional probability weights (same length as values)",
                    },
                },
                "required": ["column", "values"],
            },
        ),
        types.FunctionDeclaration(
            name="conditional_update",
            description=(
                "Update a column's value for rows where another column matches "
                "a given condition. E.g. set status='Completed' for all rows where "
                "end_date is not null."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "target_column":  {"type": "STRING"},
                    "new_value":      {"type": "STRING"},
                    "filter_column":  {"type": "STRING", "description": "Column to evaluate"},
                    "filter_op":      {
                        "type": "STRING",
                        "enum": ["equals", "not_null", "is_null", "greater_than", "less_than"],
                    },
                    "filter_value":   {"type": "STRING", "description": "Value for equals/gt/lt"},
                },
                "required": ["target_column", "new_value", "filter_column", "filter_op"],
            },
        ),
        types.FunctionDeclaration(
            name="regenerate_column",
            description=(
                "Regenerate ALL values for a column using the LLM, given a "
                "textual description of what the values should look like."
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "column":       {"type": "STRING"},
                    "description":  {"type": "STRING", "description": "What the new values should look like"},
                },
                "required": ["column", "description"],
            },
        ),
    ]
)


# ---------------------------------------------------------------------------
# DataGenerator
# ---------------------------------------------------------------------------

class DataGenerator:
    """Generates and modifies synthetic tabular data using Gemini 2.0 Flash."""

    BATCH_SIZE = 50   # max rows requested per LLM call

    def __init__(
        self,
        client: genai.Client,
        model: str = "gemini-2.5-flash",
        langfuse=None,
    ) -> None:
        self.client = client
        self.model = model
        self.langfuse = langfuse

    # ------------------------------------------------------------------
    # Public: generate all tables
    # ------------------------------------------------------------------

    def generate_all_tables(
        self,
        sorted_tables: list[TableDef],
        circular_fks: set[tuple[str, str]],
        instructions: str,
        rows_per_table: int,
        temperature: float,
        progress_callback: Optional[Callable[[int, int, str, str], None]] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Generate data for every table in dependency order.

        progress_callback(current, total, table_name, status)
        """
        all_data: dict[str, pd.DataFrame] = {}  # lower(table_name) → df

        for idx, table in enumerate(sorted_tables):
            if progress_callback:
                progress_callback(idx, len(sorted_tables), table.name, "generating")

            df = self._generate_table(
                table=table,
                all_data=all_data,
                circular_fks=circular_fks,
                instructions=instructions,
                rows=rows_per_table,
                temperature=temperature,
            )
            all_data[table.name.lower()] = df

            if progress_callback:
                progress_callback(idx + 1, len(sorted_tables), table.name, "done")

        # Resolve circular FK columns after all tables are populated
        all_data = self._resolve_circular_fks(sorted_tables, all_data, circular_fks)

        return {tbl.name: all_data[tbl.name.lower()] for tbl in sorted_tables}

    # ------------------------------------------------------------------
    # Public: modify one table
    # ------------------------------------------------------------------

    def modify_table_data(
        self,
        df: pd.DataFrame,
        table: TableDef,
        instruction: str,
        temperature: float,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Apply user modification to *df* via Gemini function-calling.

        Returns (modified_df, list_of_applied_change_messages).
        """
        schema_str = json.dumps(table.to_schema_dict(), indent=2)
        sample = df.head(5).to_dict(orient="records")

        prompt = (
            f"You are helping modify a synthetic dataset for the table '{table.name}'.\n\n"
            f"SCHEMA:\n{schema_str}\n\n"
            f"SAMPLE ROWS (first 5 of {len(df)}):\n"
            f"{json.dumps(sample, indent=2, default=str)}\n\n"
            f"USER REQUEST: {instruction}\n\n"
            "Call one or more tools to fulfil the request. "
            "If multiple columns need changes, call multiple tools."
        )

        messages: list = [{"role": "user", "parts": [{"text": prompt}]}]
        modifications: list[str] = []
        trace = self._start_trace("modify_table", {"table": table.name, "instruction": instruction})

        for _turn in range(6):  # max 6 agentic turns
            response = self.client.models.generate_content(
                model=self.model,
                contents=messages,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    tools=[_MODIFY_TOOLS],
                ),
            )

            candidate = response.candidates[0]
            func_calls = [
                p.function_call
                for p in candidate.content.parts
                if hasattr(p, "function_call") and p.function_call
            ]

            if not func_calls:
                break  # model is done

            # Execute each function call
            fn_responses = []
            for fc in func_calls:
                result = self._apply_modification(df, fc.name, dict(fc.args), table)
                df = result["df"]
                modifications.append(result["message"])
                fn_responses.append(
                    types.Part.from_function_response(
                        name=fc.name, response={"result": result["message"]}
                    )
                )

            # Append model turn + function results back into conversation
            messages.append({"role": "model", "parts": candidate.content.parts})
            messages.append({"role": "user", "parts": fn_responses})

        self._end_trace(trace, {"modifications": modifications})
        return df, modifications

    # ------------------------------------------------------------------
    # Streaming generator (public, for UI progress)
    # ------------------------------------------------------------------

    def stream_table_generation(
        self,
        table: TableDef,
        parent_ids: dict[str, list],
        instructions: str,
        num_rows: int,
        temperature: float,
        id_offset: int = 0,
    ) -> Generator[str, None, None]:
        """
        Yield raw text chunks while generating rows for *table*.
        Caller collects chunks and parses the final JSON.
        """
        prompt = _generation_prompt(
            table=table,
            parent_ids=parent_ids,
            instructions=instructions,
            num_rows=num_rows,
            id_offset=id_offset,
            batch_seed=random.randint(1000, 99999),
        )

        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        ):
            if chunk.text:
                yield chunk.text

    # ------------------------------------------------------------------
    # Private: single-table generation
    # ------------------------------------------------------------------

    def _generate_table(
        self,
        table: TableDef,
        all_data: dict[str, pd.DataFrame],
        circular_fks: set[tuple[str, str]],
        instructions: str,
        rows: int,
        temperature: float,
    ) -> pd.DataFrame:
        parent_ids = self._collect_parent_ids(table, all_data, circular_fks)
        all_rows: list[dict] = []
        num_batches = max(1, (rows + self.BATCH_SIZE - 1) // self.BATCH_SIZE)
        trace = self._start_trace("generate_table", {"table": table.name, "rows": rows})

        for batch_idx in range(num_batches):
            remaining = rows - len(all_rows)
            if remaining <= 0:
                break
            batch_size = min(self.BATCH_SIZE, remaining)

            try:
                batch = self._stream_and_parse(
                    table=table,
                    parent_ids=parent_ids,
                    instructions=instructions,
                    num_rows=batch_size,
                    temperature=temperature,
                    id_offset=len(all_rows),
                )
                batch = self._fix_rows(batch, table, all_data, circular_fks, len(all_rows))
                all_rows.extend(batch)
            except Exception as exc:
                # Fallback: deterministic placeholder rows rather than crashing
                print(f"[DataGenerator] Batch {batch_idx} failed for {table.name}: {exc}")
                fallback = self._fallback_rows(table, batch_size, len(all_rows))
                all_rows.extend(fallback)

        self._end_trace(trace, {"rows_generated": len(all_rows)})
        return pd.DataFrame(all_rows)

    def _stream_and_parse(
        self,
        table: TableDef,
        parent_ids: dict[str, list],
        instructions: str,
        num_rows: int,
        temperature: float,
        id_offset: int,
    ) -> list[dict]:
        """Call Gemini with streaming and return parsed rows."""
        chunks: list[str] = []
        for chunk in self.stream_table_generation(
            table=table,
            parent_ids=parent_ids,
            instructions=instructions,
            num_rows=num_rows,
            temperature=temperature,
            id_offset=id_offset,
        ):
            chunks.append(chunk)

        raw = "".join(chunks).strip()
        # Strip markdown code fences if Gemini adds them despite the mime type
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        rows = json.loads(raw)
        return rows if isinstance(rows, list) else [rows]

    def _fix_rows(
        self,
        rows: list[dict],
        table: TableDef,
        all_data: dict[str, pd.DataFrame],
        circular_fks: set[tuple[str, str]],
        offset: int,
    ) -> list[dict]:
        """Post-process LLM rows: set auto-increment PKs and enforce FK values."""
        fixed = []
        for i, row in enumerate(rows):
            row = dict(row)
            seq = offset + i + 1

            for col in table.columns:
                if col.is_auto_increment:
                    row[col.name] = seq

            for fk in table.foreign_keys:
                if (table.name, fk.column) in circular_fks:
                    row[fk.column] = None  # filled in later
                    continue
                ref_df = all_data.get(fk.references_table.lower())
                if ref_df is not None and not ref_df.empty:
                    valid_ids = ref_df[fk.references_column].dropna().tolist()
                    if valid_ids:
                        row[fk.column] = random.choice(valid_ids)

            fixed.append(row)
        return fixed

    def _resolve_circular_fks(
        self,
        tables: list[TableDef],
        all_data: dict[str, pd.DataFrame],
        circular_fks: set[tuple[str, str]],
    ) -> dict[str, pd.DataFrame]:
        """Fill circular FK columns now that all tables are available."""
        for tbl_name, col_name in circular_fks:
            tbl_key = tbl_name.lower()
            tbl_obj = next((t for t in tables if t.name.lower() == tbl_key), None)
            if not tbl_obj or tbl_key not in all_data:
                continue
            fk = next(
                (f for f in tbl_obj.foreign_keys if f.column.lower() == col_name.lower()),
                None,
            )
            if not fk:
                continue
            ref_key = fk.references_table.lower()
            ref_df = all_data.get(ref_key)
            if ref_df is None or ref_df.empty:
                continue
            valid_ids = ref_df[fk.references_column].dropna().tolist()
            if not valid_ids:
                continue
            df = all_data[tbl_key]
            # ~70 % chance of having a non-NULL manager/reviewer/etc.
            all_data[tbl_key][col_name] = [
                random.choice(valid_ids) if random.random() < 0.7 else None
                for _ in range(len(df))
            ]
        return all_data

    # ------------------------------------------------------------------
    # Private: collect parent IDs
    # ------------------------------------------------------------------

    def _collect_parent_ids(
        self,
        table: TableDef,
        all_data: dict[str, pd.DataFrame],
        circular_fks: set[tuple[str, str]],
    ) -> dict[str, list]:
        result: dict[str, list] = {}
        for fk in table.foreign_keys:
            if (table.name, fk.column) in circular_fks:
                continue
            ref_key = fk.references_table.lower()
            df = all_data.get(ref_key)
            if df is not None and not df.empty and fk.references_column in df.columns:
                result[ref_key] = df[fk.references_column].dropna().tolist()
        return result

    # ------------------------------------------------------------------
    # Private: apply a single function-call modification
    # ------------------------------------------------------------------

    def _apply_modification(
        self,
        df: pd.DataFrame,
        func_name: str,
        args: dict,
        table: TableDef,
    ) -> dict:
        df = df.copy()

        if func_name == "set_column_range":
            col = args.get("column", "")
            min_v = float(args.get("min_val", 0))
            max_v = float(args.get("max_val", 100))
            decimals = int(args.get("decimals", 2))
            if col not in df.columns:
                return {"df": df, "message": f"Column '{col}' not found – skipped."}
            if decimals == 0:
                df[col] = [random.randint(int(min_v), int(max_v)) for _ in range(len(df))]
            else:
                df[col] = [round(random.uniform(min_v, max_v), decimals) for _ in range(len(df))]
            return {"df": df, "message": f"Set '{col}' to random values in [{min_v}, {max_v}]."}

        if func_name == "replace_enum_values":
            col = args.get("column", "")
            values = args.get("values", [])
            weights = args.get("weights", None)
            if col not in df.columns:
                return {"df": df, "message": f"Column '{col}' not found – skipped."}
            if not values:
                return {"df": df, "message": "No values provided – skipped."}
            if weights and len(weights) == len(values):
                total = sum(weights)
                norm = [w / total for w in weights]
                df[col] = random.choices(values, weights=norm, k=len(df))
            else:
                df[col] = [values[i % len(values)] for i in range(len(df))]
            return {"df": df, "message": f"Replaced values in '{col}' with {values}."}

        if func_name == "conditional_update":
            target = args.get("target_column", "")
            new_val = args.get("new_value", "")
            fcol = args.get("filter_column", "")
            fop = args.get("filter_op", "equals")
            fval = args.get("filter_value", "")
            if target not in df.columns:
                return {"df": df, "message": f"Target column '{target}' not found – skipped."}
            if fcol not in df.columns:
                return {"df": df, "message": f"Filter column '{fcol}' not found – skipped."}

            if fop == "equals":
                mask = df[fcol].astype(str) == str(fval)
            elif fop == "not_null":
                mask = df[fcol].notna()
            elif fop == "is_null":
                mask = df[fcol].isna()
            elif fop == "greater_than":
                try:
                    mask = pd.to_numeric(df[fcol], errors="coerce") > float(fval)
                except Exception:
                    mask = df[fcol].astype(str) > str(fval)
            elif fop == "less_than":
                try:
                    mask = pd.to_numeric(df[fcol], errors="coerce") < float(fval)
                except Exception:
                    mask = df[fcol].astype(str) < str(fval)
            else:
                mask = pd.Series([True] * len(df), index=df.index)

            # Try to coerce new_val to the column's existing dtype
            coerced: object = new_val
            try:
                if df[target].dtype in ("int64", "int32", "float64"):
                    coerced = float(new_val)
                    if df[target].dtype in ("int64", "int32"):
                        coerced = int(coerced)
            except (ValueError, TypeError):
                pass

            count = int(mask.sum())
            df.loc[mask, target] = coerced
            return {
                "df": df,
                "message": f"Updated {count} rows: set '{target}' = '{new_val}' where {fcol} {fop} {fval}.",
            }

        if func_name == "regenerate_column":
            col = args.get("column", "")
            description = args.get("description", "")
            if col not in df.columns:
                return {"df": df, "message": f"Column '{col}' not found – skipped."}
            # Ask Gemini for N new values as a JSON array
            col_def = table.get_column(col)
            schema = json.dumps(col_def.data_type if col_def else "TEXT")
            regen_prompt = (
                f"Generate exactly {len(df)} values for a database column named '{col}' "
                f"(type: {schema}).\n"
                f"Requirements: {description}\n"
                "Return ONLY a JSON array of the values (strings or numbers). No prose."
            )
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=regen_prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.7,
                        response_mime_type="application/json",
                    ),
                )
                new_vals = json.loads(resp.text)
                if isinstance(new_vals, list) and len(new_vals) == len(df):
                    df[col] = new_vals
                    return {"df": df, "message": f"Regenerated column '{col}'."}
            except Exception as exc:
                return {"df": df, "message": f"Regeneration of '{col}' failed: {exc}"}

        return {"df": df, "message": f"Unknown function '{func_name}' – skipped."}

    # ------------------------------------------------------------------
    # Private: fallback rows (no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_rows(table: TableDef, n: int, offset: int) -> list[dict]:
        rows = []
        for i in range(n):
            idx = offset + i + 1
            row: dict = {}
            for col in table.columns:
                if col.is_auto_increment:
                    row[col.name] = idx
                elif col.enum_values:
                    row[col.name] = col.enum_values[idx % len(col.enum_values)]
                elif _base_type(col.data_type) in ("int", "integer", "bigint", "smallint", "serial"):
                    row[col.name] = idx
                elif _base_type(col.data_type) in ("float", "double", "decimal", "numeric", "real"):
                    row[col.name] = round(idx * 10.5, 2)
                elif "date" in col.data_type.lower() and "time" not in col.data_type.lower():
                    row[col.name] = f"2024-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d}"
                elif "time" in col.data_type.lower():
                    row[col.name] = f"2024-{(idx % 12) + 1:02d}-{(idx % 28) + 1:02d} 10:00:00"
                elif col.nullable:
                    row[col.name] = None
                else:
                    row[col.name] = f"placeholder_{idx}"
            rows.append(row)
        return rows

    # ------------------------------------------------------------------
    # Private: Langfuse tracing helpers
    # ------------------------------------------------------------------

    def _start_trace(self, name: str, metadata: dict):
        if not self.langfuse:
            return None
        try:
            return self.langfuse.trace(name=name, metadata=metadata)
        except Exception:
            return None

    def _end_trace(self, trace, output: dict) -> None:
        if trace is None:
            return
        try:
            trace.update(output=output)
            self.langfuse.flush()
        except Exception:
            pass
