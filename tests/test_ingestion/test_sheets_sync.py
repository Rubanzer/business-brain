"""Tests for sheets_sync pure-logic helpers (no Google API needed)."""

from business_brain.ingestion.sheets_sync import _extract_sheet_id


class TestExtractSheetId:
    """Test Google Sheets URL/ID extraction."""

    def test_full_url(self):
        url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms/edit#gid=0"
        assert _extract_sheet_id(url) == "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"

    def test_bare_id(self):
        sheet_id = "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms"
        assert _extract_sheet_id(sheet_id) == sheet_id

    def test_url_without_edit(self):
        url = "https://docs.google.com/spreadsheets/d/ABC123"
        assert _extract_sheet_id(url) == "ABC123"

    def test_url_with_gid(self):
        url = "https://docs.google.com/spreadsheets/d/XYZ789/edit?gid=12345"
        assert _extract_sheet_id(url) == "XYZ789"

    def test_url_with_htmlview(self):
        url = "https://docs.google.com/spreadsheets/d/SHEET_ID/htmlview"
        assert _extract_sheet_id(url) == "SHEET_ID"

    def test_url_with_export(self):
        url = "https://docs.google.com/spreadsheets/d/MY_ID/export?format=csv"
        assert _extract_sheet_id(url) == "MY_ID"

    def test_short_url_no_d_marker(self):
        # If URL has slashes but no /d/ pattern, returns last part
        url = "https://example.com/sheets/something"
        result = _extract_sheet_id(url)
        # Should return the original since no /d/ pattern found
        assert result == url

    def test_empty_string(self):
        assert _extract_sheet_id("") == ""

    def test_url_trailing_slash(self):
        url = "https://docs.google.com/spreadsheets/d/ABC123/"
        assert _extract_sheet_id(url) == "ABC123"
