"""Tests for the Data Engineer Agent."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.cognitive.data_engineer_agent import (
    DataEngineerAgent,
    _drop_empty_columns,
    _sanitize_col_name,
    _sanitize_table_name,
    _try_parse_type,
    check_hygiene,
    clean_rows,
    infer_column_types,
    parse_csv,
)


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------

class TestParseCSV:
    def test_basic_csv(self):
        raw = b"id,name,value\n1,alpha,10\n2,beta,20"
        rows = parse_csv(raw)
        assert len(rows) == 2
        assert rows[0] == {"id": "1", "name": "alpha", "value": "10"}

    def test_csv_with_bom(self):
        raw = b"\xef\xbb\xbfid,name\n1,hello"
        rows = parse_csv(raw)
        assert "id" in rows[0]

    def test_empty_csv(self):
        raw = b"id,name\n"
        rows = parse_csv(raw)
        assert rows == []


class TestParseExcel:
    @patch("openpyxl.load_workbook")
    def test_basic_excel(self, mock_load):
        # Simulate openpyxl worksheet
        mock_ws = MagicMock()
        mock_ws.iter_rows.return_value = iter([
            ("id", "name", "value"),
            (1, "alpha", 10),
            (2, "beta", 20),
        ])
        mock_wb = MagicMock()
        mock_wb.active = mock_ws
        mock_load.return_value = mock_wb

        from business_brain.cognitive.data_engineer_agent import parse_excel
        rows = parse_excel(b"fake_excel")
        assert len(rows) == 2
        assert rows[0]["id"] == 1
        assert rows[1]["name"] == "beta"
        mock_wb.close.assert_called_once()


class TestParsePDF:
    @patch("PyPDF2.PdfReader")
    def test_basic_pdf(self, mock_reader_cls):
        page1 = MagicMock()
        page1.extract_text.return_value = "Revenue report Q1"
        page2 = MagicMock()
        page2.extract_text.return_value = "Total: $1M"
        mock_reader = MagicMock()
        mock_reader.pages = [page1, page2]
        mock_reader_cls.return_value = mock_reader

        from business_brain.cognitive.data_engineer_agent import parse_pdf
        result = parse_pdf(b"fake_pdf")
        assert "Revenue report Q1" in result
        assert "Total: $1M" in result


# ---------------------------------------------------------------------------
# Type inference tests
# ---------------------------------------------------------------------------

class TestTypeInference:
    def test_integer(self):
        assert _try_parse_type("42") == "integer"
        assert _try_parse_type("-7") == "integer"

    def test_float(self):
        assert _try_parse_type("3.14") == "float"
        assert _try_parse_type("-0.5") == "float"

    def test_boolean(self):
        assert _try_parse_type("true") == "boolean"
        assert _try_parse_type("False") == "boolean"
        assert _try_parse_type("yes") == "boolean"

    def test_date(self):
        assert _try_parse_type("2024-01-15") == "date"
        assert _try_parse_type("01/15/2024") == "date"

    def test_text(self):
        assert _try_parse_type("hello world") == "text"

    def test_empty(self):
        assert _try_parse_type("") == "empty"
        assert _try_parse_type("  ") == "empty"

    def test_infer_column_types(self):
        rows = [
            {"id": "1", "amount": "10.5", "active": "true", "name": "Alice"},
            {"id": "2", "amount": "20.3", "active": "false", "name": "Bob"},
        ]
        types = infer_column_types(rows)
        assert types["id"] == "BIGINT"
        assert types["amount"] == "DOUBLE PRECISION"
        assert types["active"] == "BOOLEAN"
        assert types["name"] == "TEXT"

    def test_infer_empty_rows(self):
        assert infer_column_types([]) == {}


# ---------------------------------------------------------------------------
# Hygiene checks tests
# ---------------------------------------------------------------------------

class TestHygieneChecks:
    def test_empty_values(self):
        rows = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": ""},
        ]
        issues = check_hygiene(rows)
        name_issues = [i for i in issues if i["column"] == "name"]
        assert any("empty" in i["issue"] for i in name_issues)

    def test_whitespace(self):
        rows = [
            {"id": "1", "name": "  Alice "},
            {"id": "2", "name": "Bob"},
        ]
        issues = check_hygiene(rows)
        ws_issues = [i for i in issues if "whitespace" in i["issue"]]
        assert len(ws_issues) == 1

    def test_duplicates(self):
        rows = [
            {"id": "1", "name": "Alice"},
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        issues = check_hygiene(rows)
        dup_issues = [i for i in issues if "duplicate" in i["issue"]]
        assert len(dup_issues) == 1
        assert "1" in dup_issues[0]["issue"]

    def test_no_issues(self):
        rows = [
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        issues = check_hygiene(rows)
        # Only possible issue would be if there's nothing wrong
        assert not any("duplicate" in i["issue"] for i in issues)

    def test_empty_rows_input(self):
        assert check_hygiene([]) == []


# ---------------------------------------------------------------------------
# Cleaning tests
# ---------------------------------------------------------------------------

class TestCleanRows:
    def test_strip_whitespace(self):
        rows = [{"id": "1", "name": "  Alice  "}]
        col_types = {"id": "BIGINT", "name": "TEXT"}
        cleaned, dropped, dups = clean_rows(rows, col_types)
        assert cleaned[0]["name"] == "Alice"

    def test_drop_empty_rows(self):
        rows = [
            {"id": "1", "name": "Alice"},
            {"id": "", "name": ""},
        ]
        col_types = {"id": "BIGINT", "name": "TEXT"}
        cleaned, dropped, dups = clean_rows(rows, col_types)
        assert len(cleaned) == 1
        assert dropped == 1

    def test_remove_duplicates(self):
        rows = [
            {"id": "1", "name": "Alice"},
            {"id": "1", "name": "Alice"},
            {"id": "2", "name": "Bob"},
        ]
        col_types = {"id": "BIGINT", "name": "TEXT"}
        cleaned, dropped, dups = clean_rows(rows, col_types)
        assert len(cleaned) == 2
        assert dups == 1

    def test_coerce_type_mismatch(self):
        rows = [
            {"id": "1", "value": "100"},
            {"id": "2", "value": "N/A"},
        ]
        col_types = {"id": "BIGINT", "value": "BIGINT"}
        cleaned, _, _ = clean_rows(rows, col_types)
        assert cleaned[0]["value"] == "100"
        assert cleaned[1]["value"] is None

    def test_integer_in_float_column(self):
        rows = [{"id": "1", "price": "10"}]
        col_types = {"id": "BIGINT", "price": "DOUBLE PRECISION"}
        cleaned, _, _ = clean_rows(rows, col_types)
        # Integer should be allowed in float column
        assert cleaned[0]["price"] == "10"

    def test_empty_input(self):
        cleaned, dropped, dups = clean_rows([], {})
        assert cleaned == []
        assert dropped == 0
        assert dups == 0

    def test_drop_null_pk_rows(self):
        rows = [
            {"id": "1", "name": "Alice"},
            {"id": "", "name": "Bob"},
            {"id": "3", "name": "Carol"},
        ]
        col_types = {"id": "BIGINT", "name": "TEXT"}
        cleaned, dropped, dups = clean_rows(rows, col_types)
        assert len(cleaned) == 2
        assert dropped == 1
        assert all(row["id"] is not None for row in cleaned)


# ---------------------------------------------------------------------------
# Drop empty columns tests
# ---------------------------------------------------------------------------

class TestDropEmptyColumns:
    def test_drops_fully_empty_columns(self):
        rows = [
            {"id": "1", "name": "Alice", "extra": ""},
            {"id": "2", "name": "Bob", "extra": ""},
        ]
        cleaned, dropped = _drop_empty_columns(rows)
        assert "extra" not in cleaned[0]
        assert dropped == ["extra"]

    def test_keeps_partially_filled_columns(self):
        rows = [
            {"id": "1", "name": "Alice", "notes": ""},
            {"id": "2", "name": "Bob", "notes": "important"},
        ]
        cleaned, dropped = _drop_empty_columns(rows)
        assert "notes" in cleaned[0]
        assert dropped == []

    def test_empty_rows_input(self):
        cleaned, dropped = _drop_empty_columns([])
        assert cleaned == []
        assert dropped == []

    def test_drops_none_columns(self):
        rows = [
            {"id": "1", "name": "Alice", "col_5": None},
            {"id": "2", "name": "Bob", "col_5": None},
        ]
        cleaned, dropped = _drop_empty_columns(rows)
        assert "col_5" not in cleaned[0]
        assert dropped == ["col_5"]


# ---------------------------------------------------------------------------
# Sanitization tests
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_table_name(self):
        assert _sanitize_table_name("My Sales-Data") == "my_sales_data"
        assert _sanitize_table_name("test.csv") == "test_csv"

    def test_col_name(self):
        assert _sanitize_col_name("Order ID") == "order_id"
        assert _sanitize_col_name("price$") == "price"


# ---------------------------------------------------------------------------
# Agent integration tests (mocked DB + Gemini)
# ---------------------------------------------------------------------------

class TestDataEngineerAgentInvoke:
    @pytest.mark.asyncio
    @patch("business_brain.cognitive.data_engineer_agent.ingest_context", new_callable=AsyncMock)
    @patch("business_brain.cognitive.data_engineer_agent.metadata_store")
    @patch("business_brain.cognitive.data_engineer_agent._get_client")
    async def test_csv_upload(self, mock_client, mock_meta, mock_ingest):
        # Mock Gemini response
        mock_response = MagicMock()
        mock_response.text = '{"table_description": "Sales data", "column_descriptions": {"id": "Row ID", "name": "Product name", "value": "Sale amount"}, "business_context": "Sales tracking data."}'
        mock_client.return_value.models.generate_content.return_value = mock_response

        mock_meta.upsert = AsyncMock()
        mock_ingest.return_value = 1

        session = AsyncMock()

        csv_bytes = b"id,name,value\n1,alpha,10\n2,beta,20\n3,gamma,30"
        agent = DataEngineerAgent()
        result = await agent.invoke({
            "file_bytes": csv_bytes,
            "file_name": "sales.csv",
            "db_session": session,
        })

        assert result["table_name"] == "sales"
        assert result["file_type"] == "csv"
        assert result["rows_total"] == 3
        assert result["rows_inserted"] == 3
        assert result["metadata"]["description"] == "Sales data"
        assert result["context_generated"] == "Sales tracking data."

        # Verify DB calls
        assert session.execute.call_count >= 1
        assert session.commit.call_count >= 1
        mock_meta.upsert.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.data_engineer_agent.ingest_context", new_callable=AsyncMock)
    @patch("business_brain.cognitive.data_engineer_agent._get_client")
    async def test_pdf_upload(self, mock_client, mock_ingest):
        mock_response = MagicMock()
        mock_response.text = '{"table_description": "Revenue report", "column_descriptions": {}, "business_context": "Quarterly revenue data."}'
        mock_client.return_value.models.generate_content.return_value = mock_response
        mock_ingest.return_value = 42

        session = AsyncMock()

        agent = DataEngineerAgent()
        with patch("business_brain.cognitive.data_engineer_agent.parse_pdf", return_value="Revenue Q1: $1M"):
            result = await agent.invoke({
                "file_bytes": b"fake_pdf",
                "file_name": "report.pdf",
                "db_session": session,
            })

        assert result["file_type"] == "pdf"
        assert result["rows_total"] == 0
        assert result["context_id"] == 42
        mock_ingest.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsupported_file(self):
        session = AsyncMock()
        agent = DataEngineerAgent()
        with pytest.raises(ValueError, match="Unsupported file type"):
            await agent.invoke({
                "file_bytes": b"data",
                "file_name": "file.docx",
                "db_session": session,
            })

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.data_engineer_agent._get_client")
    async def test_empty_csv(self, mock_client):
        session = AsyncMock()
        agent = DataEngineerAgent()
        result = await agent.invoke({
            "file_bytes": b"id,name\n",
            "file_name": "empty.csv",
            "db_session": session,
        })

        assert result["rows_total"] == 0
        assert any("empty" in i["issue"].lower() for i in result["issues"])

    @pytest.mark.asyncio
    @patch("business_brain.cognitive.data_engineer_agent.ingest_context", new_callable=AsyncMock)
    @patch("business_brain.cognitive.data_engineer_agent.metadata_store")
    @patch("business_brain.cognitive.data_engineer_agent._get_client")
    async def test_cleaning_applied(self, mock_client, mock_meta, mock_ingest):
        mock_response = MagicMock()
        mock_response.text = '{"table_description": "Test", "column_descriptions": {}, "business_context": ""}'
        mock_client.return_value.models.generate_content.return_value = mock_response
        mock_meta.upsert = AsyncMock()
        mock_ingest.return_value = 1

        session = AsyncMock()

        csv_bytes = b"id,name\n1,Alice\n1,Alice\n2,Bob\n,"
        agent = DataEngineerAgent()
        result = await agent.invoke({
            "file_bytes": csv_bytes,
            "file_name": "test.csv",
            "db_session": session,
        })

        assert result["rows_total"] == 4
        assert result["duplicates_removed"] >= 1
        assert result["rows_dropped"] >= 1
        # Should have inserted only clean unique rows
        assert result["rows_inserted"] == 2
