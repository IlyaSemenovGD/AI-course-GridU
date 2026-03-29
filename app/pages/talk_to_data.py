"""
Talk to Your Data – Phase 2 chat interface.

Layout (mirrors UI_interface_chat_sample.png):
  ┌─────────────────────────────────────────┐
  │ Dataset selector (dropdown)             │
  ├─────────────────────────────────────────┤
  │ Chat history                            │
  │   user bubble   → plain text           │
  │   assistant     → text + SQL + table   │
  │                   + optional chart      │
  ├─────────────────────────────────────────┤
  │ [ Ask a question about your data... ]  │
  └─────────────────────────────────────────┘
"""

from __future__ import annotations

import os

import streamlit as st
from google import genai

from app.core.database import db_manager_from_env
from app.core.sql_agent import SQLAgent
from app.utils.observability import get_langfuse

# ── State defaults ────────────────────────────────────────────────────────────
_STATE_DEFAULTS: dict = {
    "ttd_dataset_id":          None,
    "ttd_schema_name":         None,
    "ttd_dataset_name":        "",
    "ttd_messages":            [],   # display messages (role, content, sql, df, chart_fig)
    "ttd_gemini_history":      [],   # raw {role, parts} for Gemini API
    "ttd_turn_index":          0,
    "ttd_last_jailbreak_score": 0.0,
    "ttd_guardrail_warnings":  [],
}


def _init_state() -> None:
    for k, v in _STATE_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_gemini_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if api_key:
        return genai.Client(api_key=api_key)
    project = os.environ.get("GCP_PROJECT", "")
    location = os.environ.get("GCP_LOCATION", "us-central1")
    if not project:
        st.error("Set GOOGLE_API_KEY or GCP_PROJECT in .env and restart.")
        st.stop()
    return genai.Client(vertexai=True, project=project, location=location)


@st.cache_resource(show_spinner=False)
def _get_db():
    db = db_manager_from_env()
    try:
        db.initialise()
    except Exception:
        pass
    return db


# ── Message rendering ─────────────────────────────────────────────────────────

def _render_message(msg: dict) -> None:
    """Render a single message in the chat. Works for both history and new messages."""
    role = msg["role"]
    with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
        # Main text
        if msg.get("content"):
            st.markdown(msg["content"])

        # SQL block
        if msg.get("sql"):
            with st.expander("View SQL", expanded=True):
                st.code(msg["sql"], language="sql")

        # Query result table
        if msg.get("df") is not None and not msg["df"].empty:
            st.dataframe(msg["df"], use_container_width=True, hide_index=True)
            st.caption(f"{len(msg['df'])} rows returned")

        # Seaborn chart
        if msg.get("chart_fig") is not None:
            st.pyplot(msg["chart_fig"], clear_figure=False)


# ── Main render ───────────────────────────────────────────────────────────────

def render() -> None:
    _init_state()

    st.header("Talk to Your Data")

    db = _get_db()

    # ── Dataset selector ──────────────────────────────────────────────────
    if not db.ping():
        st.warning(
            "PostgreSQL is not reachable. Start the database and refresh. "
            "See README for setup instructions."
        )
        return

    datasets = db.list_datasets()
    if not datasets:
        st.info(
            "No datasets found. Go to **Data Generation** tab, generate data, "
            "and click *Save to Database* first."
        )
        return

    dataset_options = {f"{d['name']} (id={d['id']})": d for d in datasets}
    selected_label = st.selectbox(
        "Dataset",
        list(dataset_options.keys()),
        label_visibility="collapsed",
        key="ttd_dataset_select",
    )
    selected_ds = dataset_options[selected_label]

    # Reset conversation when user switches dataset
    if st.session_state.ttd_dataset_id != selected_ds["id"]:
        st.session_state.ttd_dataset_id    = selected_ds["id"]
        st.session_state.ttd_schema_name   = selected_ds["schema_name"]
        st.session_state.ttd_dataset_name  = selected_ds["name"]
        st.session_state.ttd_messages      = []
        st.session_state.ttd_gemini_history = []
        st.session_state.ttd_turn_index    = 0
        st.session_state.ttd_guardrail_warnings = []

    # Show table list as a subtle hint
    tables = selected_ds.get("tables", [])
    if tables:
        st.caption(f"Tables: `{'`, `'.join(tables)}`")

    st.divider()

    # ── Conversation history ──────────────────────────────────────────────
    for msg in st.session_state.ttd_messages:
        _render_message(msg)

    # ── Guardrail warnings (sidebar) ──────────────────────────────────────
    if st.session_state.ttd_guardrail_warnings:
        with st.sidebar:
            st.warning("⚠️ Guardrail events", icon="🛡️")
            for w in st.session_state.ttd_guardrail_warnings[-5:]:
                st.caption(w)

    # ── Chat input ────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about your data…")

    if not user_input:
        return

    # Append user message to display history
    user_msg = {"role": "user", "content": user_input, "sql": None, "df": None, "chart_fig": None}
    st.session_state.ttd_messages.append(user_msg)
    _render_message(user_msg)

    # ── Run agent ─────────────────────────────────────────────────────────
    client   = _get_gemini_client()
    langfuse = get_langfuse()
    model    = os.environ.get("CHAT_MODEL", "gemini-2.5-flash")

    agent = SQLAgent(
        client=client,
        db=db,
        model=model,
        langfuse=langfuse,
    )

    # Containers for live streaming
    with st.chat_message("assistant", avatar="🤖"):
        text_placeholder = st.empty()
        accumulated_text = ""

        # Mutable containers filled by the callback
        result_state: dict = {
            "sql": None, "df": None,
            "chart_fig": None, "chart_type": None,
            "guardrail": None, "errors": [],
        }

        def stream_callback(event: str, payload) -> None:
            nonlocal accumulated_text

            if event == "token":
                accumulated_text += payload
                text_placeholder.markdown(accumulated_text + "▌")

            elif event == "sql_result":
                result_state["sql"]        = payload["sql"]
                result_state["df"]         = payload["df"]
                result_state["chart_fig"]  = payload["chart_fig"]
                result_state["chart_type"] = payload["chart_type"]
                # Render SQL + table immediately (before final text)
                if payload["sql"]:
                    with st.expander("View SQL", expanded=True):
                        st.code(payload["sql"], language="sql")
                if payload["df"] is not None and not payload["df"].empty:
                    st.dataframe(payload["df"], use_container_width=True, hide_index=True)
                    st.caption(f"{len(payload['df'])} rows returned")
                if payload["chart_fig"] is not None:
                    st.pyplot(payload["chart_fig"], clear_figure=False)

            elif event == "guardrail":
                result_state["guardrail"] = payload
                text_placeholder.warning(
                    f"🛡️ **Request blocked** ({payload.category}): {payload.reason}"
                )
                st.session_state.ttd_guardrail_warnings.append(
                    f"[Turn {st.session_state.ttd_turn_index + 1}] "
                    f"{payload.category}: {payload.reason}"
                )

            elif event == "error":
                result_state["errors"].append(payload)
                st.error(payload)

            elif event == "done":
                # Finalise the streamed text (remove cursor)
                if accumulated_text:
                    text_placeholder.markdown(accumulated_text)

        result = agent.run(
            user_message=user_input,
            dataset_id=st.session_state.ttd_dataset_id,
            schema_name=st.session_state.ttd_schema_name,
            gemini_history=list(st.session_state.ttd_gemini_history),
            stream_callback=stream_callback,
        )

    # ── Persist to session state ──────────────────────────────────────────
    assistant_msg = {
        "role":       "assistant",
        "content":    result["content"],
        "sql":        result["sql"],
        "df":         result["df"],
        "chart_fig":  result["chart_fig"],
        "chart_type": result["chart_type"],
    }
    st.session_state.ttd_messages.append(assistant_msg)

    # Update Gemini history (only serialisable text/tool parts)
    st.session_state.ttd_gemini_history.append(
        {"role": "user", "parts": [{"text": user_input}]}
    )
    if result["content"] and not result.get("guardrail_blocked"):
        st.session_state.ttd_gemini_history.append(
            {"role": "model", "parts": [{"text": result["content"]}]}
        )

    st.session_state.ttd_turn_index += 1

    # ── Clear conversation button ─────────────────────────────────────────
    if st.session_state.ttd_messages:
        if st.button("🗑️ Clear conversation", key="clear_chat"):
            st.session_state.ttd_messages       = []
            st.session_state.ttd_gemini_history  = []
            st.session_state.ttd_turn_index      = 0
            st.rerun()
