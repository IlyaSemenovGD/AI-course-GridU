"""
Talk to Your Data page  –  Phase 2 (placeholder).

Shows stored datasets and provides a query interface.
Phase 1 only stores data; the full NL-to-SQL chat is implemented in Phase 2.
"""

from __future__ import annotations

import streamlit as st

from app.core.database import db_manager_from_env


@st.cache_resource(show_spinner=False)
def _get_db():
    db = db_manager_from_env()
    try:
        db.initialise()
    except Exception:
        pass
    return db


def render() -> None:
    st.header("Talk to Your Data")
    st.info(
        "This tab is implemented in **Phase 2**.  "
        "For now you can browse the datasets that were saved from the "
        "Data Generation tab.",
        icon="ℹ️",
    )

    db = _get_db()
    if not db.ping():
        st.warning(
            "Cannot reach PostgreSQL — stored datasets are unavailable. "
            "Make sure the database is running and POSTGRES_* env vars are set."
        )
        return

    datasets = db.list_datasets()
    if not datasets:
        st.write("No datasets saved yet.  Go to the **Data Generation** tab and click *Save to Database*.")
        return

    st.subheader("Saved Datasets")
    for ds in datasets:
        with st.expander(
            f"**{ds['name']}**  (id={ds['id']}, "
            f"{len(ds['tables'])} tables, "
            f"created {str(ds['created_at'])[:19]})"
        ):
            st.write(f"Tables: `{'`, `'.join(ds['tables'])}`")
            st.write(f"Schema: `{ds['schema_name']}`")
            st.write(f"Rows per table (approx): {ds['rows_per_table']}")

            col_load, col_del = st.columns([1, 1])
            with col_load:
                if st.button("Preview", key=f"preview_{ds['id']}"):
                    with st.spinner("Loading…"):
                        data = db.load_dataset(ds["id"])
                    if data:
                        for tbl, df in data.items():
                            st.markdown(f"**{tbl}**")
                            st.dataframe(df.head(20), use_container_width=True, hide_index=True)
                    else:
                        st.error("Could not load dataset.")
            with col_del:
                if st.button("Delete", key=f"del_{ds['id']}", type="secondary"):
                    db.delete_dataset(ds["id"])
                    st.success("Dataset deleted.")
                    st.rerun()
