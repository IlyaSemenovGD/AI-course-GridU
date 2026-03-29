"""
DatabaseManager – stores generated datasets in PostgreSQL.

Each dataset is saved in its own schema (namespaced by a slugified dataset name
plus a short hash).  A meta-table "_datasets" in the public schema tracks all
datasets so the Talk-to-your-data tab can discover them.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import textwrap
from datetime import datetime
from typing import Optional

import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(name: str) -> str:
    """Turn an arbitrary string into a safe PostgreSQL identifier fragment."""
    name = name.lower().strip()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = name.strip("_")
    return name[:40] or "dataset"


def _schema_name(dataset_name: str) -> str:
    h = hashlib.md5(dataset_name.encode()).hexdigest()[:6]
    return f"{_slugify(dataset_name)}_{h}"


def _pg_col_type(series: pd.Series) -> str:
    """Map a pandas Series dtype to a PostgreSQL type."""
    dtype_str = str(series.dtype)
    if "int" in dtype_str:
        return "BIGINT"
    if "float" in dtype_str:
        return "DOUBLE PRECISION"
    if "bool" in dtype_str:
        return "BOOLEAN"
    if "datetime" in dtype_str:
        return "TIMESTAMP"
    return "TEXT"


# ---------------------------------------------------------------------------
# DatabaseManager
# ---------------------------------------------------------------------------

class DatabaseManager:
    META_TABLE = "public._datasets"

    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
    ) -> None:
        self._dsn = {
            "host": host,
            "port": port,
            "dbname": dbname,
            "user": user,
            "password": password,
        }

    # ------------------------------------------------------------------
    # Connection helper
    # ------------------------------------------------------------------

    def _connect(self):
        return psycopg2.connect(**self._dsn)

    # ------------------------------------------------------------------
    # Initialise meta table (idempotent)
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        """Create the _datasets meta table if it does not exist."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                textwrap.dedent("""\
                    CREATE TABLE IF NOT EXISTS public._datasets (
                        id            SERIAL PRIMARY KEY,
                        name          TEXT    NOT NULL,
                        schema_name   TEXT    NOT NULL UNIQUE,
                        tables        TEXT[]  NOT NULL,
                        ddl_text      TEXT,
                        rows_per_table INTEGER,
                        created_at    TIMESTAMP DEFAULT NOW()
                    );
                """)
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Save a generated dataset
    # ------------------------------------------------------------------

    def save_dataset(
        self,
        name: str,
        tables_data: dict[str, pd.DataFrame],
        ddl_text: str = "",
        rows_per_table: int = 0,
    ) -> int:
        """
        Persist every DataFrame as a table inside a dedicated PostgreSQL schema.
        Returns the new dataset id.
        """
        schema = _schema_name(name)
        table_names = list(tables_data.keys())

        with self._connect() as conn, conn.cursor() as cur:
            # Create isolated schema
            cur.execute(
                sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
            )

            for tbl_name, df in tables_data.items():
                safe_name = _slugify(tbl_name)
                # Build CREATE TABLE DDL from DataFrame dtypes
                col_defs = ", ".join(
                    f'"{col}" {_pg_col_type(df[col])}'
                    for col in df.columns
                )
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}.{}").format(
                        sql.Identifier(schema), sql.Identifier(safe_name)
                    )
                )
                cur.execute(
                    sql.SQL("CREATE TABLE {}.{} ({})").format(
                        sql.Identifier(schema),
                        sql.Identifier(safe_name),
                        sql.SQL(col_defs),
                    )
                )
                # Bulk insert using execute_values
                rows = [tuple(row) for row in df.itertuples(index=False, name=None)]
                if rows:
                    cols_sql = sql.SQL(", ").join(sql.Identifier(c) for c in df.columns)
                    insert_sql = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s").format(
                        sql.Identifier(schema),
                        sql.Identifier(safe_name),
                        cols_sql,
                    )
                    execute_values(cur, insert_sql, rows)

            # Register in meta table
            cur.execute(
                textwrap.dedent("""\
                    INSERT INTO public._datasets
                        (name, schema_name, tables, ddl_text, rows_per_table)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (schema_name) DO UPDATE
                        SET name           = EXCLUDED.name,
                            tables         = EXCLUDED.tables,
                            ddl_text       = EXCLUDED.ddl_text,
                            rows_per_table = EXCLUDED.rows_per_table,
                            created_at     = NOW()
                    RETURNING id;
                """),
                (name, schema, table_names, ddl_text, rows_per_table),
            )
            row = cur.fetchone()
            dataset_id: int = row[0] if row else -1
            conn.commit()

        return dataset_id

    # ------------------------------------------------------------------
    # List all stored datasets
    # ------------------------------------------------------------------

    def list_datasets(self) -> list[dict]:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT id, name, schema_name, tables, rows_per_table, created_at "
                    "FROM public._datasets ORDER BY created_at DESC;"
                )
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Load a dataset back as DataFrames
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_id: int) -> Optional[dict[str, pd.DataFrame]]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT schema_name, tables FROM public._datasets WHERE id = %s;",
                (dataset_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            schema, table_names = row

        result: dict[str, pd.DataFrame] = {}
        with self._connect() as conn:
            for tbl in table_names:
                safe = _slugify(tbl)
                try:
                    df = pd.read_sql(
                        f'SELECT * FROM "{schema}"."{safe}"',
                        conn,
                    )
                    result[tbl] = df
                except Exception:
                    pass
        return result

    # ------------------------------------------------------------------
    # Delete a dataset (schema + meta row)
    # ------------------------------------------------------------------

    def delete_dataset(self, dataset_id: int) -> bool:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT schema_name FROM public._datasets WHERE id = %s;",
                (dataset_id,),
            )
            row = cur.fetchone()
            if not row:
                return False
            schema = row[0]
            cur.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
            )
            cur.execute("DELETE FROM public._datasets WHERE id = %s;", (dataset_id,))
            conn.commit()
        return True

    # ------------------------------------------------------------------
    # Quick connectivity check
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        try:
            with self._connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1;")
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Factory from environment variables
# ---------------------------------------------------------------------------

def db_manager_from_env() -> DatabaseManager:
    return DatabaseManager(
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "datagen"),
        user=os.environ.get("POSTGRES_USER", "datagen"),
        password=os.environ.get("POSTGRES_PASSWORD", "datagen"),
    )
