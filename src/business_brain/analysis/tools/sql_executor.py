"""Safe SQL query builder and executor for analysis operations.

Builds parameterized SQL from structured QueryIntent objects, supporting:
- Multi-table JOINs (Gap #2: cross-table analysis)
- Time range filtering (Gap #5: time-scoped windows)
- GROUP BY with aggregations (Gap #1: N-ary segmented analysis)
- Safe identifier quoting (reuses _safe() pattern from precompute_engine.py)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Identifier safety (from precompute_engine.py)
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe(name: str) -> str:
    """Sanitize a table/column name to prevent SQL injection."""
    return _SAFE_NAME_RE.sub("", name)


def _q(name: str) -> str:
    """Quote a sanitized identifier."""
    return f'"{_safe(name)}"'


# ---------------------------------------------------------------------------
# Dataclasses — structured query building blocks
# ---------------------------------------------------------------------------


@dataclass
class FilterSpec:
    """A WHERE condition."""

    column: str
    operator: str  # =, !=, >, <, >=, <=, IN, NOT IN, LIKE, IS NULL, IS NOT NULL
    value: Any = None  # None for IS NULL / IS NOT NULL


@dataclass
class TimeRange:
    """Time-scoped filter for analysis windows (Gap #5)."""

    column: str
    start: str | None = None  # ISO timestamp or relative like "now() - interval '30 days'"
    end: str | None = None
    window: str | None = None  # "7d", "30d", "90d", "ytd", "all"
    granularity: str | None = None  # day/week/month — for GROUP BY time bucketing


@dataclass
class AggregationSpec:
    """A SELECT aggregation."""

    column: str
    function: str  # AVG, SUM, COUNT, MIN, MAX, STDDEV, VARIANCE
    alias: str | None = None

    @property
    def sql_alias(self) -> str:
        return self.alias or f"{self.function.lower()}_{self.column}"


@dataclass
class JoinSpec:
    """Cross-table JOIN specification (Gap #2)."""

    table: str
    local_col: str
    remote_col: str
    join_type: str = "INNER"  # INNER, LEFT, RIGHT


@dataclass
class QueryIntent:
    """Structured description of what to query."""

    tables: list[str]
    select_columns: list[str] = field(default_factory=list)
    aggregations: list[AggregationSpec] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    filters: list[FilterSpec] = field(default_factory=list)
    time_range: TimeRange | None = None
    join: JoinSpec | None = None
    order_by: list[str] = field(default_factory=list)  # ["col ASC", "col DESC"]
    limit: int | None = None


@dataclass
class QueryResult:
    """Result of an executed query."""

    rows: list[dict[str, Any]]
    columns: list[str]
    row_count: int
    query: str
    duration_ms: int

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.columns)

    def to_series(self, column: str) -> pd.Series:
        return pd.Series([r[column] for r in self.rows if r.get(column) is not None])


# ---------------------------------------------------------------------------
# SQL Builder
# ---------------------------------------------------------------------------

_WINDOW_TO_INTERVAL = {
    "7d": "7 days",
    "30d": "30 days",
    "90d": "90 days",
    "365d": "365 days",
}


def _build_time_filter(tr: TimeRange, table_alias: str | None = None) -> str:
    """Build a WHERE clause fragment for time scoping."""
    prefix = f"{_q(table_alias)}." if table_alias else ""
    col = f"{prefix}{_q(tr.column)}"

    if tr.window and tr.window != "all":
        if tr.window == "ytd":
            return f"{col} >= date_trunc('year', CURRENT_DATE)"
        interval = _WINDOW_TO_INTERVAL.get(tr.window, tr.window)
        return f"{col} >= NOW() - INTERVAL '{interval}'"

    parts = []
    if tr.start:
        parts.append(f"{col} >= '{tr.start}'")
    if tr.end:
        parts.append(f"{col} <= '{tr.end}'")
    return " AND ".join(parts) if parts else ""


def _build_filter(f: FilterSpec, table_alias: str | None = None) -> str:
    """Build a single WHERE clause fragment."""
    prefix = f"{_q(table_alias)}." if table_alias else ""
    col = f"{prefix}{_q(f.column)}"

    op = f.operator.upper()
    if op in ("IS NULL", "IS NOT NULL"):
        return f"{col} {op}"
    if op == "IN":
        vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in f.value)
        return f"{col} IN ({vals})"
    if op == "NOT IN":
        vals = ", ".join(f"'{v}'" if isinstance(v, str) else str(v) for v in f.value)
        return f"{col} NOT IN ({vals})"
    if isinstance(f.value, str):
        return f"{col} {op} '{f.value}'"
    return f"{col} {op} {f.value}"


def build_sql(intent: QueryIntent) -> str:
    """Build a SQL string from a QueryIntent."""
    primary = intent.tables[0]
    primary_q = _q(primary)

    # SELECT clause
    select_parts: list[str] = []
    for col in intent.select_columns:
        select_parts.append(_q(col))
    for agg in intent.aggregations:
        select_parts.append(
            f'{agg.function}(CAST({_q(agg.column)} AS DOUBLE PRECISION)) AS "{_safe(agg.sql_alias)}"'
        )
    if not select_parts:
        select_parts = ["*"]

    sql = f"SELECT {', '.join(select_parts)} FROM {primary_q}"

    # JOIN clause (Gap #2)
    if intent.join:
        j = intent.join
        sql += (
            f" {j.join_type} JOIN {_q(j.table)}"
            f" ON {primary_q}.{_q(j.local_col)} = {_q(j.table)}.{_q(j.remote_col)}"
        )

    # WHERE clause
    where_parts: list[str] = []
    for f in intent.filters:
        clause = _build_filter(f)
        if clause:
            where_parts.append(clause)
    if intent.time_range:
        tc = _build_time_filter(intent.time_range)
        if tc:
            where_parts.append(tc)
    if where_parts:
        sql += f" WHERE {' AND '.join(where_parts)}"

    # GROUP BY
    if intent.group_by:
        sql += f" GROUP BY {', '.join(_q(c) for c in intent.group_by)}"

    # Time bucketing in GROUP BY
    if intent.time_range and intent.time_range.granularity:
        bucket = f"date_trunc('{intent.time_range.granularity}', {_q(intent.time_range.column)})"
        if intent.group_by:
            sql += f", {bucket}"
        else:
            sql += f" GROUP BY {bucket}"

    # ORDER BY
    if intent.order_by:
        order_parts = []
        for ob in intent.order_by:
            parts = ob.strip().split()
            col = _q(parts[0])
            direction = parts[1].upper() if len(parts) > 1 else "ASC"
            order_parts.append(f"{col} {direction}")
        sql += f" ORDER BY {', '.join(order_parts)}"

    # LIMIT
    if intent.limit:
        sql += f" LIMIT {int(intent.limit)}"

    return sql


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


async def execute(session: AsyncSession, intent: QueryIntent) -> QueryResult:
    """Execute a QueryIntent and return structured results."""
    query = build_sql(intent)
    start = time.monotonic()

    result = await session.execute(sql_text(query))
    raw_rows = result.fetchall()

    duration_ms = int((time.monotonic() - start) * 1000)

    columns = list(result.keys()) if raw_rows else []
    rows = [dict(r._mapping) for r in raw_rows]

    return QueryResult(
        rows=rows,
        columns=columns,
        row_count=len(rows),
        query=query,
        duration_ms=duration_ms,
    )


async def execute_raw(session: AsyncSession, sql: str) -> QueryResult:
    """Execute a raw SQL string. Use sparingly — prefer QueryIntent."""
    start = time.monotonic()
    result = await session.execute(sql_text(sql))
    raw_rows = result.fetchall()
    duration_ms = int((time.monotonic() - start) * 1000)
    columns = list(result.keys()) if raw_rows else []
    rows = [dict(r._mapping) for r in raw_rows]
    return QueryResult(
        rows=rows,
        columns=columns,
        row_count=len(rows),
        query=sql,
        duration_ms=duration_ms,
    )
