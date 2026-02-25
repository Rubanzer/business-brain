"""Tests verifying the /csv upload endpoint correctly calls metadata_store.upsert().

The /csv endpoint was fixed to register uploaded tables in the metadata store
after loading data via upsert_dataframe(), ensuring tables are visible in the UI.
"""

from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from business_brain.db.connection import get_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _mock_session_override():
    session = AsyncMock()
    yield session


@pytest.fixture()
def client():
    """Create a TestClient that skips real DB startup events and background discovery."""
    with patch("business_brain.action.routers.data.run_discovery_background", new_callable=AsyncMock):
        from business_brain.action.api import app

        original_startup = list(app.router.on_startup)
        original_shutdown = list(app.router.on_shutdown)
        app.router.on_startup.clear()
        app.router.on_shutdown.clear()

        app.dependency_overrides[get_session] = _mock_session_override

        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            yield c

        app.dependency_overrides.clear()
        app.router.on_startup = original_startup
        app.router.on_shutdown = original_shutdown


@pytest.fixture()
def client_with_auth():
    """TestClient with an authenticated user injected via dependency override."""
    with patch("business_brain.action.routers.data.run_discovery_background", new_callable=AsyncMock):
        from business_brain.action.api import app, get_current_user

        original_startup = list(app.router.on_startup)
        original_shutdown = list(app.router.on_shutdown)
        app.router.on_startup.clear()
        app.router.on_shutdown.clear()

        app.dependency_overrides[get_session] = _mock_session_override

        async def _fake_user():
            return {"sub": "user-42", "role": "admin"}

        app.dependency_overrides[get_current_user] = _fake_user

        from fastapi.testclient import TestClient

        with TestClient(app) as c:
            yield c

        app.dependency_overrides.clear()
        app.router.on_startup = original_startup
        app.router.on_shutdown = original_shutdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_csv(*rows: str) -> bytes:
    """Build a CSV byte string from header + data rows."""
    return "\n".join(rows).encode("utf-8")


SIMPLE_CSV = _make_csv("id,name,value", "1,alpha,10", "2,beta,20", "3,gamma,30")
HEADERS_ONLY_CSV = _make_csv("id,name,value")


# ---------------------------------------------------------------------------
# Test class: Core metadata_store.upsert behaviour
# ---------------------------------------------------------------------------

class TestCsvUploadMetadataUpsert:
    """Verify that /csv calls metadata_store.upsert() with correct arguments."""

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_calls_metadata_upsert(self, mock_upsert_df, mock_meta, client):
        """metadata_store.upsert is called exactly once with the correct table_name."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post("/csv", files={"file": ("sales.csv", SIMPLE_CSV, "text/csv")})

        assert resp.status_code == 200
        mock_meta.upsert.assert_called_once()
        _, kwargs = mock_meta.upsert.call_args
        assert kwargs["table_name"] == "sales"

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_metadata_columns_match_df(self, mock_upsert_df, mock_meta, client):
        """columns_metadata matches the DataFrame columns and their dtypes."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post("/csv", files={"file": ("data.csv", SIMPLE_CSV, "text/csv")})

        assert resp.status_code == 200
        _, kwargs = mock_meta.upsert.call_args
        cols = kwargs["columns_metadata"]
        col_names = [c["name"] for c in cols]
        assert col_names == ["id", "name", "value"]
        # Each entry must have a 'type' key that is a non-empty string
        for c in cols:
            assert "type" in c
            assert isinstance(c["type"], str)
            assert len(c["type"]) > 0

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_metadata_description_generated(self, mock_upsert_df, mock_meta, client):
        """description is non-empty and contains the table name."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        client.post("/csv", files={"file": ("revenue.csv", SIMPLE_CSV, "text/csv")})

        _, kwargs = mock_meta.upsert.call_args
        desc = kwargs["description"]
        assert len(desc) > 0
        assert "revenue" in desc

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_metadata_upsert_receives_session(self, mock_upsert_df, mock_meta, client):
        """The first positional argument to metadata_store.upsert is the DB session."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        client.post("/csv", files={"file": ("t.csv", SIMPLE_CSV, "text/csv")})

        args, _ = mock_meta.upsert.call_args
        # First positional arg should be the AsyncMock session from our override
        assert isinstance(args[0], AsyncMock)

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_response_format(self, mock_upsert_df, mock_meta, client):
        """Response JSON has status, table, and rows keys with correct values."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post("/csv", files={"file": ("sales.csv", SIMPLE_CSV, "text/csv")})

        data = resp.json()
        assert data["status"] == "loaded"
        assert data["table"] == "sales"
        assert data["rows"] == 3


# ---------------------------------------------------------------------------
# Test class: Authentication / uploaded_by handling
# ---------------------------------------------------------------------------

class TestCsvUploadAuthMetadata:
    """Verify uploaded_by and uploaded_by_role are passed correctly based on auth."""

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_with_auth_passes_uploaded_by(
        self, mock_upsert_df, mock_meta, client_with_auth
    ):
        """When user is authenticated, uploaded_by=user['sub'] and uploaded_by_role=user['role']."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client_with_auth.post(
            "/csv", files={"file": ("sales.csv", SIMPLE_CSV, "text/csv")}
        )

        assert resp.status_code == 200
        _, kwargs = mock_meta.upsert.call_args
        assert kwargs["uploaded_by"] == "user-42"
        assert kwargs["uploaded_by_role"] == "admin"

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_without_auth_passes_none(self, mock_upsert_df, mock_meta, client):
        """When user is None (no auth), uploaded_by=None and uploaded_by_role=None."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post("/csv", files={"file": ("sales.csv", SIMPLE_CSV, "text/csv")})

        assert resp.status_code == 200
        _, kwargs = mock_meta.upsert.call_args
        assert kwargs["uploaded_by"] is None
        assert kwargs["uploaded_by_role"] is None


# ---------------------------------------------------------------------------
# Test class: Description formatting (column preview, "+N more")
# ---------------------------------------------------------------------------

class TestCsvUploadDescriptionFormatting:
    """Verify the description string formats column previews correctly."""

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_metadata_description_with_many_columns(
        self, mock_upsert_df, mock_meta, client
    ):
        """More than 5 columns results in '(+N more)' in the description."""
        mock_upsert_df.return_value = 1
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        cols = [f"col{i}" for i in range(8)]
        csv_data = _make_csv(",".join(cols), ",".join(["1"] * 8))

        client.post("/csv", files={"file": ("wide.csv", csv_data, "text/csv")})

        _, kwargs = mock_meta.upsert.call_args
        desc = kwargs["description"]
        assert "(+3 more)" in desc

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_exactly_five_columns_no_plus_more(self, mock_upsert_df, mock_meta, client):
        """Exactly 5 columns should NOT have '(+N more)' in the description."""
        mock_upsert_df.return_value = 1
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        cols = ["a", "b", "c", "d", "e"]
        csv_data = _make_csv(",".join(cols), ",".join(["1"] * 5))

        client.post("/csv", files={"file": ("five.csv", csv_data, "text/csv")})

        _, kwargs = mock_meta.upsert.call_args
        desc = kwargs["description"]
        assert "(+" not in desc
        assert "5 columns" in desc

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_six_columns_shows_plus_more(self, mock_upsert_df, mock_meta, client):
        """6 columns should show '(+1 more)' in the description."""
        mock_upsert_df.return_value = 1
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        cols = ["a", "b", "c", "d", "e", "f"]
        csv_data = _make_csv(",".join(cols), ",".join(["1"] * 6))

        client.post("/csv", files={"file": ("six.csv", csv_data, "text/csv")})

        _, kwargs = mock_meta.upsert.call_args
        desc = kwargs["description"]
        assert "(+1 more)" in desc
        assert "6 columns" in desc


# ---------------------------------------------------------------------------
# Test class: Table name sanitisation
# ---------------------------------------------------------------------------

class TestCsvUploadTableNameSanitisation:
    """Verify filename-to-table-name conversion handles special characters."""

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_table_name_sanitized(self, mock_upsert_df, mock_meta, client):
        """Hyphens and spaces in filename are replaced with underscores."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post(
            "/csv", files={"file": ("my-data file.csv", SIMPLE_CSV, "text/csv")}
        )

        data = resp.json()
        assert data["table"] == "my_data_file"
        _, kwargs = mock_meta.upsert.call_args
        assert kwargs["table_name"] == "my_data_file"

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_special_chars_in_filename_sanitized(self, mock_upsert_df, mock_meta, client):
        """Multiple hyphens and spaces are each replaced with underscores."""
        mock_upsert_df.return_value = 3
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post(
            "/csv", files={"file": ("Q1-sales report-2024.csv", SIMPLE_CSV, "text/csv")}
        )

        data = resp.json()
        assert data["table"] == "Q1_sales_report_2024"
        _, kwargs = mock_meta.upsert.call_args
        assert kwargs["table_name"] == "Q1_sales_report_2024"


# ---------------------------------------------------------------------------
# Test class: Column type mapping
# ---------------------------------------------------------------------------

class TestCsvUploadColumnTypes:
    """Verify that column dtype strings are passed correctly in columns_metadata."""

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_metadata_column_types_correct(self, mock_upsert_df, mock_meta, client):
        """int64, float64, and object columns are mapped to correct dtype strings."""
        mock_upsert_df.return_value = 2
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        csv_data = _make_csv("int_col,float_col,str_col", "1,1.5,hello", "2,2.5,world")

        client.post("/csv", files={"file": ("types.csv", csv_data, "text/csv")})

        _, kwargs = mock_meta.upsert.call_args
        type_map = {c["name"]: c["type"] for c in kwargs["columns_metadata"]}
        assert type_map["int_col"] == "int64"
        assert type_map["float_col"] == "float64"
        assert type_map["str_col"] in ("object", "str")  # pandas 2.x+ may return "str"


# ---------------------------------------------------------------------------
# Test class: Call ordering
# ---------------------------------------------------------------------------

class TestCsvUploadCallOrdering:
    """Verify metadata_store.upsert is called after upsert_dataframe."""

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_metadata_upsert_called_after_data_upsert(
        self, mock_upsert_df, mock_meta, client
    ):
        """metadata_store.upsert must be called after upsert_dataframe succeeds."""
        call_order = []

        async def _track_upsert_df(*args, **kwargs):
            call_order.append("upsert_dataframe")
            return 3

        async def _track_meta_upsert(*args, **kwargs):
            call_order.append("metadata_upsert")
            return MagicMock()

        mock_upsert_df.side_effect = _track_upsert_df
        mock_meta.upsert = AsyncMock(side_effect=_track_meta_upsert)

        client.post("/csv", files={"file": ("t.csv", SIMPLE_CSV, "text/csv")})

        assert call_order == ["upsert_dataframe", "metadata_upsert"]


# ---------------------------------------------------------------------------
# Test class: Edge cases (empty, single column, large, corrupt)
# ---------------------------------------------------------------------------

class TestCsvUploadEdgeCases:
    """Edge cases: empty files, single columns, many rows, corrupt data."""

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_upload_empty_df(self, mock_upsert_df, mock_meta, client):
        """CSV with headers and no data rows results in rows=0."""
        mock_upsert_df.return_value = 0
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post(
            "/csv", files={"file": ("empty.csv", HEADERS_ONLY_CSV, "text/csv")}
        )

        data = resp.json()
        assert data["status"] == "loaded"
        assert data["rows"] == 0
        mock_meta.upsert.assert_called_once()

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_corrupt_binary_returns_error(self, mock_upsert_df, mock_meta, client):
        """Uploading random binary data returns an error response."""
        resp = client.post(
            "/csv", files={"file": ("bad.csv", b"\x80\x81\x82\xff\xfe", "text/csv")}
        )

        data = resp.json()
        assert "error" in data
        # metadata_store.upsert should NOT have been called since parsing failed
        mock_meta.upsert.assert_not_called()

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_empty_file_returns_error(self, mock_upsert_df, mock_meta, client):
        """Uploading a zero-byte file returns an error response."""
        resp = client.post("/csv", files={"file": ("empty.csv", b"", "text/csv")})

        data = resp.json()
        assert "error" in data
        mock_meta.upsert.assert_not_called()

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_headers_only_zero_rows(self, mock_upsert_df, mock_meta, client):
        """CSV with only headers results in rows=0 and metadata is still registered."""
        mock_upsert_df.return_value = 0
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        resp = client.post(
            "/csv", files={"file": ("heads.csv", HEADERS_ONLY_CSV, "text/csv")}
        )

        data = resp.json()
        assert data["rows"] == 0
        mock_meta.upsert.assert_called_once()
        _, kwargs = mock_meta.upsert.call_args
        col_names = [c["name"] for c in kwargs["columns_metadata"]]
        assert col_names == ["id", "name", "value"]

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_single_column_file(self, mock_upsert_df, mock_meta, client):
        """A single-column CSV is handled correctly."""
        mock_upsert_df.return_value = 2
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        csv_data = _make_csv("score", "100", "200")

        resp = client.post("/csv", files={"file": ("scores.csv", csv_data, "text/csv")})

        data = resp.json()
        assert data["status"] == "loaded"
        assert data["rows"] == 2
        _, kwargs = mock_meta.upsert.call_args
        assert len(kwargs["columns_metadata"]) == 1
        assert kwargs["columns_metadata"][0]["name"] == "score"
        assert "(+" not in kwargs["description"]

    @patch("business_brain.action.routers.data.metadata_store")
    @patch("business_brain.ingestion.csv_loader.upsert_dataframe")
    def test_csv_many_rows(self, mock_upsert_df, mock_meta, client):
        """A CSV with many rows returns the correct row count."""
        mock_upsert_df.return_value = 1000
        mock_meta.upsert = AsyncMock(return_value=MagicMock())

        rows = ["id,value"] + [f"{i},{i * 10}" for i in range(1000)]
        csv_data = "\n".join(rows).encode("utf-8")

        resp = client.post("/csv", files={"file": ("big.csv", csv_data, "text/csv")})

        data = resp.json()
        assert data["status"] == "loaded"
        assert data["rows"] == 1000
        mock_meta.upsert.assert_called_once()
