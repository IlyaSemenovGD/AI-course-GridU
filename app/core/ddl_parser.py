"""
DDL Parser – converts CREATE TABLE / ALTER TABLE SQL into Python dataclasses
that the data-generator can consume.

Supports:
  - MySQL/PostgreSQL CREATE TABLE syntax
  - AUTO_INCREMENT / SERIAL primary keys
  - Inline and table-level PRIMARY KEY
  - Inline REFERENCES and FOREIGN KEY … REFERENCES constraints
  - ALTER TABLE … ADD CONSTRAINT … FOREIGN KEY
  - ENUM(…) column types
  - CHECK constraints for numeric ranges
  - NOT NULL, UNIQUE, DEFAULT
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ColumnDef:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False
    is_unique: bool = False
    is_auto_increment: bool = False
    default: Optional[str] = None
    enum_values: Optional[list[str]] = None
    check_min: Optional[float] = None
    check_max: Optional[float] = None
    references_table: Optional[str] = None
    references_column: Optional[str] = None


@dataclass
class ForeignKey:
    column: str
    references_table: str
    references_column: str


@dataclass
class TableDef:
    name: str
    columns: list[ColumnDef] = field(default_factory=list)
    primary_keys: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def get_column(self, name: str) -> Optional[ColumnDef]:
        for col in self.columns:
            if col.name.lower() == name.lower():
                return col
        return None

    def to_schema_dict(self) -> dict:
        """Human-readable dict representation for LLM prompts."""
        cols = []
        for c in self.columns:
            entry: dict = {
                "name": c.name,
                "type": c.data_type,
                "nullable": c.nullable,
            }
            if c.is_primary_key:
                entry["primary_key"] = True
            if c.is_auto_increment:
                entry["auto_increment"] = True
            if c.is_unique:
                entry["unique"] = True
            if c.enum_values:
                entry["enum_values"] = c.enum_values
            if c.check_min is not None:
                entry["check_min"] = c.check_min
            if c.check_max is not None:
                entry["check_max"] = c.check_max
            if c.references_table:
                entry["references"] = f"{c.references_table}({c.references_column})"
            if c.default is not None:
                entry["default"] = c.default
            cols.append(entry)

        return {
            "table": self.name,
            "columns": cols,
            "foreign_keys": [
                {
                    "column": fk.column,
                    "references": f"{fk.references_table}({fk.references_column})",
                }
                for fk in self.foreign_keys
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_comments(ddl: str) -> str:
    ddl = re.sub(r"--[^\n]*", "", ddl)
    ddl = re.sub(r"/\*.*?\*/", "", ddl, flags=re.DOTALL)
    return ddl


def _split_top_level(text: str, delimiter: str = ",") -> list[str]:
    """Split *text* on *delimiter* ignoring content inside parentheses / quotes."""
    parts: list[str] = []
    depth = 0
    in_str = False
    str_char = ""
    buf: list[str] = []

    for ch in text:
        if in_str:
            buf.append(ch)
            if ch == str_char:
                in_str = False
        elif ch in ("'", '"'):
            in_str = True
            str_char = ch
            buf.append(ch)
        elif ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth -= 1
            buf.append(ch)
        elif ch == delimiter and depth == 0:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


# ---------------------------------------------------------------------------
# Column / constraint parsers
# ---------------------------------------------------------------------------

_SKIP_KEYWORDS = (
    "PRIMARY KEY",
    "FOREIGN KEY",
    "UNIQUE KEY",
    "UNIQUE (",
    "UNIQUE(",
    "INDEX ",
    "KEY ",
    "CONSTRAINT ",
    "CHECK (",
    "CHECK(",
)


def _parse_column_def(part: str) -> Optional[ColumnDef]:
    """Parse a single column-definition fragment into ColumnDef, or None if it
    is a table-level constraint."""
    part = part.strip()
    upper = part.upper().lstrip()

    for kw in _SKIP_KEYWORDS:
        if upper.startswith(kw):
            return None

    # Must start with a column name
    m = re.match(r"[`\"]?(\w+)[`\"]?\s+(.*)", part, re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    col_name = m.group(1)
    rest = m.group(2).strip()

    col = ColumnDef(name=col_name, data_type="TEXT")

    # ENUM type  (must be parsed before generic type, as it contains commas)
    enum_m = re.match(r"ENUM\s*\(([^)]+)\)\s*", rest, re.IGNORECASE)
    if enum_m:
        col.data_type = "ENUM"
        col.enum_values = [
            v.strip().strip("'\"") for v in enum_m.group(1).split(",")
        ]
        rest = rest[enum_m.end():]
    else:
        # Generic type, potentially with size args like VARCHAR(255) or DECIMAL(10,2)
        type_m = re.match(r"(\w+\s*(?:\([^)]*\))?)", rest)
        if type_m:
            col.data_type = type_m.group(1).strip()
            rest = rest[type_m.end():].strip()

    rest_upper = rest.upper()

    if re.search(r"\bAUTO_INCREMENT\b", rest_upper):
        col.is_auto_increment = True

    if re.search(r"\bNOT\s+NULL\b", rest_upper):
        col.nullable = False

    if re.search(r"\bPRIMARY\s+KEY\b", rest_upper):
        col.is_primary_key = True
        col.nullable = False

    if re.search(r"\bUNIQUE\b", rest_upper):
        col.is_unique = True

    # DEFAULT value (string or bare token)
    def_m = re.search(r"\bDEFAULT\s+(?:'([^']*)'|(\S+))", rest, re.IGNORECASE)
    if def_m:
        col.default = def_m.group(1) if def_m.group(1) is not None else def_m.group(2)

    # Inline REFERENCES
    ref_m = re.search(
        r"\bREFERENCES\s+[`\"]?(\w+)[`\"]?\s*\(\s*[`\"]?(\w+)[`\"]?\s*\)",
        rest,
        re.IGNORECASE,
    )
    if ref_m:
        col.references_table = ref_m.group(1)
        col.references_column = ref_m.group(2)

    # Inline CHECK for numeric range
    chk_m = re.search(r"\bCHECK\s*\(([^)]+)\)", rest, re.IGNORECASE)
    if chk_m:
        expr = chk_m.group(1)
        mn = re.search(r">=\s*(\d+(?:\.\d+)?)", expr)
        mx = re.search(r"<=\s*(\d+(?:\.\d+)?)", expr)
        if mn:
            col.check_min = float(mn.group(1))
        if mx:
            col.check_max = float(mx.group(1))

    return col


def _parse_table_pk(part: str) -> Optional[list[str]]:
    m = re.match(r"PRIMARY\s+KEY\s*\(([^)]+)\)", part.strip(), re.IGNORECASE)
    if m:
        return [c.strip().strip("`\"'") for c in m.group(1).split(",")]
    return None


def _parse_table_fk(part: str) -> Optional[ForeignKey]:
    """Parse table-level FOREIGN KEY … REFERENCES … (with optional CONSTRAINT name)."""
    m = re.search(
        r"FOREIGN\s+KEY\s*\(\s*[`\"]?(\w+)[`\"]?\s*\)\s*"
        r"REFERENCES\s+[`\"]?(\w+)[`\"]?\s*\(\s*[`\"]?(\w+)[`\"]?\s*\)",
        part,
        re.IGNORECASE,
    )
    if m:
        return ForeignKey(
            column=m.group(1),
            references_table=m.group(2),
            references_column=m.group(3),
        )
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_ddl(ddl_text: str) -> list[TableDef]:
    """Parse a DDL string and return a list of TableDef objects."""
    ddl_text = _strip_comments(ddl_text)
    tables: dict[str, TableDef] = {}

    # ── CREATE TABLE blocks ────────────────────────────────────────────────
    create_re = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[`\"]?(\w+)[`\"]?\s*\((.*?)\)\s*;",
        re.IGNORECASE | re.DOTALL,
    )
    for m in create_re.finditer(ddl_text):
        tbl_name = m.group(1)
        body = m.group(2)
        table = TableDef(name=tbl_name)

        for part in _split_top_level(body):
            part = part.strip()
            if not part:
                continue
            upper = part.upper().lstrip()

            if upper.startswith("PRIMARY KEY"):
                pks = _parse_table_pk(part)
                if pks:
                    table.primary_keys.extend(pks)
                    for pk in pks:
                        c = table.get_column(pk)
                        if c:
                            c.is_primary_key = True
                            c.nullable = False

            elif "FOREIGN KEY" in upper:
                fk = _parse_table_fk(part)
                if fk:
                    table.foreign_keys.append(fk)
                    c = table.get_column(fk.column)
                    if c:
                        c.references_table = fk.references_table
                        c.references_column = fk.references_column

            else:
                col = _parse_column_def(part)
                if col:
                    table.columns.append(col)
                    if col.references_table:
                        if not any(
                            f.column.lower() == col.name.lower()
                            for f in table.foreign_keys
                        ):
                            table.foreign_keys.append(
                                ForeignKey(
                                    column=col.name,
                                    references_table=col.references_table,
                                    references_column=col.references_column,
                                )
                            )
                    if col.is_primary_key and col.name not in table.primary_keys:
                        table.primary_keys.append(col.name)

        tables[tbl_name.lower()] = table

    # ── ALTER TABLE … ADD CONSTRAINT … FOREIGN KEY … ──────────────────────
    alter_re = re.compile(
        r"ALTER\s+TABLE\s+[`\"]?(\w+)[`\"]?\s+"
        r"ADD\s+(?:CONSTRAINT\s+\w+\s+)?"
        r"FOREIGN\s+KEY\s*\(\s*[`\"]?(\w+)[`\"]?\s*\)\s+"
        r"REFERENCES\s+[`\"]?(\w+)[`\"]?\s*\(\s*[`\"]?(\w+)[`\"]?\s*\)",
        re.IGNORECASE | re.DOTALL,
    )
    for m in alter_re.finditer(ddl_text):
        tbl_key = m.group(1).lower()
        col_name = m.group(2)
        ref_table = m.group(3)
        ref_col = m.group(4)
        if tbl_key in tables:
            tbl = tables[tbl_key]
            if not any(f.column.lower() == col_name.lower() for f in tbl.foreign_keys):
                tbl.foreign_keys.append(
                    ForeignKey(column=col_name, references_table=ref_table, references_column=ref_col)
                )
            c = tbl.get_column(col_name)
            if c:
                c.references_table = ref_table
                c.references_column = ref_col

    return list(tables.values())


def topological_sort(tables: list[TableDef]) -> tuple[list[TableDef], set[tuple[str, str]]]:
    """
    Return (ordered_tables, circular_fks).

    ordered_tables  – parent tables before children.
    circular_fks    – set of (table_name, column_name) pairs that form cycles
                      and therefore cannot be filled until all tables exist.
    """
    name_map: dict[str, TableDef] = {t.name.lower(): t for t in tables}
    visited: set[str] = set()
    in_stack: set[str] = set()
    result: list[TableDef] = []
    cycle_edges: set[tuple[str, str]] = set()

    def visit(tbl: TableDef) -> None:
        key = tbl.name.lower()
        if key in in_stack:
            return  # cycle – will be resolved later
        if key in visited:
            return
        in_stack.add(key)
        for fk in tbl.foreign_keys:
            ref = fk.references_table.lower()
            if ref != key and ref in name_map:
                visit(name_map[ref])
        in_stack.discard(key)
        visited.add(key)
        result.append(tbl)

    for tbl in tables:
        if tbl.name.lower() not in visited:
            visit(tbl)

    # Detect which FK columns still reference a table that appears *after* the
    # current table in the sorted order (i.e. are truly circular).
    order_index: dict[str, int] = {t.name.lower(): i for i, t in enumerate(result)}
    for tbl in result:
        for fk in tbl.foreign_keys:
            ref = fk.references_table.lower()
            if ref == tbl.name.lower():
                continue  # self-reference – handled like circular
            if ref not in order_index:
                continue
            if order_index[ref] > order_index[tbl.name.lower()]:
                cycle_edges.add((tbl.name, fk.column))

    return result, cycle_edges
