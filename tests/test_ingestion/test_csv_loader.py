"""Unit tests for csv_loader — DataFrame → SQL generation with mocked DB."""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, call

import pandas as pd
import pytest

from business_brain.ingestion.csv_loader import (
    _pg_type,
    load_csv,
    upsert_dataframe,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_session() -> AsyncMock:
    """Return an AsyncMock that behaves like an AsyncSession."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    return session


# ── _pg_type ─────────────────────────────────────────────────────────────────


class TestPgType:
    def test_int(self):
        assert _pg_type(pd.Series([1, 2]).dtype) == "BIGINT"

    def test_float(self):
        assert _pg_type(pd.Series([1.5]).dtype) == "DOUBLE PRECISION"

    def test_bool(self):
        assert _pg_type(pd.Series([True]).dtype) == "BOOLEAN"

    def test_string_fallback(self):
        assert _pg_type(pd.Series(["a", "b"]).dtype) == "TEXT"


# ── upsert_dataframe ────────────────────────────────────────────────────────


class TestUpsertDataframe:
    @pytest.mark.asyncio
    async def test_empty_dataframe_returns_zero(self):
        session = _make_session()
        result = await upsert_dataframe(pd.DataFrame(), session, "t")
        assert result == 0
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_table_and_upserts(self):
        df = pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})
        session = _make_session()

        count = await upsert_dataframe(df, session, "test_tbl")

        assert count == 2
        # Two execute calls: CREATE TABLE + INSERT
        assert session.execute.call_count == 2

        # First call is the DDL
        ddl_sql = str(session.execute.call_args_list[0].args[0].text)
        assert "CREATE TABLE IF NOT EXISTS" in ddl_sql
        assert '"test_tbl"' in ddl_sql
        assert '"id" BIGINT' in ddl_sql
        assert '"name" TEXT' in ddl_sql

        # Second call is the upsert
        upsert_sql = str(session.execute.call_args_list[1].args[0].text)
        assert "INSERT INTO" in upsert_sql
        assert "ON CONFLICT" in upsert_sql

    @pytest.mark.asyncio
    async def test_custom_pk_column(self):
        df = pd.DataFrame({"code": ["X"], "val": [10]})
        session = _make_session()

        await upsert_dataframe(df, session, "t", pk_column="code")

        ddl_sql = str(session.execute.call_args_list[0].args[0].text)
        assert 'PRIMARY KEY ("code")' in ddl_sql

    @pytest.mark.asyncio
    async def test_single_column_uses_do_nothing(self):
        df = pd.DataFrame({"id": [1]})
        session = _make_session()

        await upsert_dataframe(df, session, "t")

        upsert_sql = str(session.execute.call_args_list[1].args[0].text)
        assert "DO NOTHING" in upsert_sql


# ── load_csv ─────────────────────────────────────────────────────────────────


class TestLoadCsv:
    @pytest.mark.asyncio
    async def test_load_csv_reads_file(self, tmp_path):
        csv_file = tmp_path / "sales.csv"
        csv_file.write_text("id,product,amount\n1,Widget,9.99\n2,Gadget,19.99\n")

        session = _make_session()
        count = await load_csv(csv_file, session)

        assert count == 2

    @pytest.mark.asyncio
    async def test_table_name_from_filename(self, tmp_path):
        csv_file = tmp_path / "My Sales Data.csv"
        csv_file.write_text("id,val\n1,10\n")

        session = _make_session()
        await load_csv(csv_file, session)

        ddl_sql = str(session.execute.call_args_list[0].args[0].text)
        assert '"my_sales_data"' in ddl_sql
