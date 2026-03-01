"""Tests for analysis/tools/sql_executor.py — SQL building + execution."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from business_brain.analysis.tools.sql_executor import (
    AggregationSpec,
    FilterSpec,
    JoinSpec,
    QueryIntent,
    QueryResult,
    TimeRange,
    _safe,
    _q,
    build_sql,
    execute,
    execute_raw,
)


# ---------------------------------------------------------------------------
# Identifier safety
# ---------------------------------------------------------------------------


class TestSafety:
    def test_safe_simple(self):
        assert _safe("table_name") == "table_name"

    def test_safe_strips_injection(self):
        assert _safe("table; DROP TABLE users") == "tableDROPTABLEusers"

    def test_safe_strips_special_chars(self):
        assert _safe("col-name.with spaces!") == "colnamewithspaces"

    def test_q_quotes(self):
        assert _q("table_name") == '"table_name"'


# ---------------------------------------------------------------------------
# SQL Building
# ---------------------------------------------------------------------------


class TestBuildSql:
    def test_simple_select(self):
        intent = QueryIntent(tables=["orders"], select_columns=["id", "amount"])
        sql = build_sql(intent)
        assert '"id"' in sql
        assert '"amount"' in sql
        assert '"orders"' in sql

    def test_select_star(self):
        intent = QueryIntent(tables=["orders"])
        sql = build_sql(intent)
        assert "SELECT *" in sql

    def test_aggregation(self):
        intent = QueryIntent(
            tables=["orders"],
            aggregations=[
                AggregationSpec(column="amount", function="AVG", alias="avg_amount"),
                AggregationSpec(column="amount", function="COUNT", alias="cnt"),
            ],
            group_by=["category"],
        )
        sql = build_sql(intent)
        assert "AVG(" in sql
        assert "COUNT(" in sql
        assert "GROUP BY" in sql
        assert '"category"' in sql

    def test_where_clause(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            filters=[
                FilterSpec(column="status", operator="=", value="active"),
                FilterSpec(column="amount", operator=">", value=100),
            ],
        )
        sql = build_sql(intent)
        assert "WHERE" in sql
        assert "'active'" in sql
        assert "> 100" in sql

    def test_is_null_filter(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            filters=[FilterSpec(column="deleted_at", operator="IS NULL")],
        )
        sql = build_sql(intent)
        assert "IS NULL" in sql

    def test_in_filter(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            filters=[FilterSpec(column="status", operator="IN", value=["active", "pending"])],
        )
        sql = build_sql(intent)
        assert "IN (" in sql
        assert "'active'" in sql
        assert "'pending'" in sql

    def test_join(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["orders.id"],
            join=JoinSpec(
                table="customers",
                local_col="customer_id",
                remote_col="id",
                join_type="LEFT",
            ),
        )
        sql = build_sql(intent)
        assert "LEFT JOIN" in sql
        assert '"customers"' in sql
        assert '"customer_id"' in sql

    def test_time_range_window(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            time_range=TimeRange(column="created_at", window="30d"),
        )
        sql = build_sql(intent)
        assert "30 days" in sql
        assert "NOW()" in sql

    def test_time_range_ytd(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            time_range=TimeRange(column="created_at", window="ytd"),
        )
        sql = build_sql(intent)
        assert "date_trunc" in sql
        assert "year" in sql

    def test_time_range_all(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            time_range=TimeRange(column="created_at", window="all"),
        )
        sql = build_sql(intent)
        # "all" should not add any time filter
        assert "WHERE" not in sql

    def test_order_by(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            order_by=["amount DESC"],
        )
        sql = build_sql(intent)
        assert "ORDER BY" in sql
        assert "DESC" in sql

    def test_limit(self):
        intent = QueryIntent(
            tables=["orders"],
            select_columns=["id"],
            limit=10,
        )
        sql = build_sql(intent)
        assert "LIMIT 10" in sql

    def test_complex_query(self):
        """Test a full N-ary segmented query with time scope and join."""
        intent = QueryIntent(
            tables=["production"],
            select_columns=["shift", "furnace"],
            aggregations=[
                AggregationSpec(column="output_kg", function="AVG", alias="avg_output"),
            ],
            group_by=["shift", "furnace"],
            filters=[
                FilterSpec(column="output_kg", operator="IS NOT NULL"),
            ],
            time_range=TimeRange(column="production_date", window="30d"),
            join=JoinSpec(
                table="furnace_config",
                local_col="furnace_id",
                remote_col="id",
            ),
            order_by=["avg_output DESC"],
            limit=50,
        )
        sql = build_sql(intent)
        assert "INNER JOIN" in sql
        assert "GROUP BY" in sql
        assert "ORDER BY" in sql
        assert "LIMIT 50" in sql
        assert "30 days" in sql


# ---------------------------------------------------------------------------
# QueryResult
# ---------------------------------------------------------------------------


class TestQueryResult:
    def test_to_dataframe(self):
        qr = QueryResult(
            rows=[{"a": 1, "b": 2}, {"a": 3, "b": 4}],
            columns=["a", "b"],
            row_count=2,
            query="SELECT ...",
            duration_ms=5,
        )
        df = qr.to_dataframe()
        assert len(df) == 2
        assert list(df.columns) == ["a", "b"]

    def test_to_series(self):
        qr = QueryResult(
            rows=[{"val": 10}, {"val": 20}, {"val": None}],
            columns=["val"],
            row_count=3,
            query="SELECT ...",
            duration_ms=5,
        )
        s = qr.to_series("val")
        assert len(s) == 2  # None excluded
        assert s.iloc[0] == 10


# ---------------------------------------------------------------------------
# Async execution
# ---------------------------------------------------------------------------


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_returns_result(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_row = MagicMock()
        mock_row._mapping = {"id": 1, "name": "test"}
        mock_result.fetchall.return_value = [mock_row]
        mock_result.keys.return_value = ["id", "name"]
        mock_session.execute.return_value = mock_result

        intent = QueryIntent(tables=["test_table"], select_columns=["id", "name"])
        qr = await execute(mock_session, intent)

        assert qr.row_count == 1
        assert qr.rows[0]["id"] == 1
        assert qr.duration_ms >= 0
        assert '"test_table"' in qr.query

    @pytest.mark.asyncio
    async def test_execute_raw(self):
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_result.keys.return_value = []
        mock_session.execute.return_value = mock_result

        qr = await execute_raw(mock_session, "SELECT 1")
        assert qr.row_count == 0
        assert qr.query == "SELECT 1"
