"""
Streamlit entry point.

Run locally:
    streamlit run app/main.py

Run via Docker Compose:
    docker compose up --build
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root (phase1/) is on sys.path so that `app.*` imports
# work regardless of which directory Streamlit was launched from.
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env if running locally (docker-compose injects env vars directly)
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJECT_ROOT / ".env")
except ImportError:
    pass

import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🗄️ Data Assistant")
    st.divider()

    page = st.radio(
        "Navigation",
        options=["Data Generation", "Talk to your data"],
        label_visibility="collapsed",
        key="nav_page",
    )

    st.divider()

    # Connection status indicators
    with st.expander("Status", expanded=False):
        # Gemini
        gcp_project = os.environ.get("GCP_PROJECT", "")
        if gcp_project:
            st.success(f"Vertex AI: `{gcp_project}`")
        else:
            st.error("Vertex AI: GCP_PROJECT not set")

        # PostgreSQL
        from app.core.database import db_manager_from_env
        db = db_manager_from_env()
        if db.ping():
            st.success("PostgreSQL: connected")
        else:
            st.warning("PostgreSQL: not reachable")

        # Langfuse
        from app.utils.observability import get_langfuse
        lf = get_langfuse()
        if lf is not None:
            st.success("Langfuse: enabled")
        else:
            st.info("Langfuse: disabled (no credentials)")

# ── Route to page ─────────────────────────────────────────────────────────────
if page == "Data Generation":
    from app.pages.data_generation import render
    render()
else:
    from app.pages.talk_to_data import render
    render()
