"""Tests for schema_rag.retrieve_relevant_tables — Focus Mode allowed_tables filtering."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.memory.schema_rag import retrieve_relevant_tables


@patch("business_brain.memory.schema_rag.vector_store")
@patch("business_brain.memory.schema_rag.embed_text")
@patch("business_brain.memory.schema_rag.metadata_store")
class TestRetrieveRelevantTablesFiltering:
    @pytest.mark.asyncio
    async def test_retrieve_with_allowed_tables_calls_get_filtered(
        self, mock_metadata_store, mock_embed, mock_vector_store
    ):
        """Verify get_filtered is called with the allowed_tables argument."""
        session = AsyncMock()

        mock_embed.return_value = [0.1] * 384
        mock_vector_store.search = AsyncMock(return_value=[])

        entry = MagicMock(
            table_name="sales",
            description="Sales transactions",
            columns_metadata=[{"name": "amount", "type": "numeric", "description": "sale amount"}],
        )
        mock_metadata_store.get_filtered = AsyncMock(return_value=[entry])

        await retrieve_relevant_tables(
            session, "show sales", allowed_tables=["sales"]
        )

        mock_metadata_store.get_filtered.assert_called_once_with(session, ["sales"])

    @pytest.mark.asyncio
    async def test_retrieve_with_none_returns_all(
        self, mock_metadata_store, mock_embed, mock_vector_store
    ):
        """When allowed_tables is None, get_filtered is called with None (all tables)."""
        session = AsyncMock()

        mock_embed.return_value = [0.1] * 384
        mock_vector_store.search = AsyncMock(return_value=[])

        entry_a = MagicMock(
            table_name="sales",
            description="Sales data",
            columns_metadata=[{"name": "revenue", "type": "numeric", "description": "revenue"}],
        )
        entry_b = MagicMock(
            table_name="customers",
            description="Customer data",
            columns_metadata=[{"name": "name", "type": "text", "description": "customer name"}],
        )
        mock_metadata_store.get_filtered = AsyncMock(return_value=[entry_a, entry_b])

        tables, contexts = await retrieve_relevant_tables(
            session, "show everything", allowed_tables=None
        )

        mock_metadata_store.get_filtered.assert_called_once_with(session, None)
        # With None, all tables are considered — both should appear in fallback
        assert len(tables) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_single_table(
        self, mock_metadata_store, mock_embed, mock_vector_store
    ):
        """Single table in allowed_tables returns only that table's results."""
        session = AsyncMock()

        mock_embed.return_value = [0.1] * 384
        mock_vector_store.search = AsyncMock(return_value=[])

        entry = MagicMock(
            table_name="orders",
            description="Order transactions",
            columns_metadata=[{"name": "order_id", "type": "integer", "description": "order identifier"}],
        )
        mock_metadata_store.get_filtered = AsyncMock(return_value=[entry])

        tables, _ = await retrieve_relevant_tables(
            session, "show orders", allowed_tables=["orders"]
        )

        mock_metadata_store.get_filtered.assert_called_once_with(session, ["orders"])
        # The single table should appear in results (either via scoring or fallback)
        table_names = [t["table_name"] for t in tables]
        assert "orders" in table_names

    @pytest.mark.asyncio
    async def test_scoring_works_within_filtered_set(
        self, mock_metadata_store, mock_embed, mock_vector_store
    ):
        """Results are properly scored — higher relevance tables rank first."""
        session = AsyncMock()

        mock_embed.return_value = [0.1] * 384
        mock_vector_store.search = AsyncMock(return_value=[])

        # "sales" should score higher for query "sales revenue" than "customers"
        entry_sales = MagicMock(
            table_name="sales",
            description="Sales transactions and revenue data",
            columns_metadata=[
                {"name": "revenue", "type": "numeric", "description": "total revenue"},
                {"name": "date", "type": "date", "description": "sale date"},
            ],
        )
        entry_customers = MagicMock(
            table_name="customers",
            description="Customer information",
            columns_metadata=[
                {"name": "name", "type": "text", "description": "customer name"},
            ],
        )
        mock_metadata_store.get_filtered = AsyncMock(
            return_value=[entry_sales, entry_customers]
        )

        tables, _ = await retrieve_relevant_tables(
            session, "sales revenue", allowed_tables=["sales", "customers"]
        )

        # Sales should be ranked first because the query mentions "sales"
        assert len(tables) >= 1
        assert tables[0]["table_name"] == "sales"
        # Score is stripped from final output (line 187 of schema_rag.py),
        # but correct ordering proves scoring worked
        assert "table_name" in tables[0]

    @pytest.mark.asyncio
    async def test_empty_allowed_tables_returns_empty(
        self, mock_metadata_store, mock_embed, mock_vector_store
    ):
        """Empty allowed_tables list means get_filtered returns [], so no results."""
        session = AsyncMock()

        mock_embed.return_value = [0.1] * 384
        mock_vector_store.search = AsyncMock(return_value=[])

        mock_metadata_store.get_filtered = AsyncMock(return_value=[])

        tables, _ = await retrieve_relevant_tables(
            session, "show sales", allowed_tables=[]
        )

        mock_metadata_store.get_filtered.assert_called_once_with(session, [])
        # No entries to score, so results should be empty
        assert tables == []
