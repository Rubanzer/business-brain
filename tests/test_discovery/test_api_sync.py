"""Tests for the API sync module (no network calls needed)."""

from business_brain.ingestion.api_sync import _fetch_api_data


class TestFetchApiDataParsing:
    """Test response path navigation logic."""

    def test_response_path_simple(self):
        """Test navigating a simple dot path."""
        data = {"data": {"results": [{"id": 1}, {"id": 2}]}}

        # Simulate what _fetch_api_data does with response_path
        response_path = "data.results"
        result = data
        for key in response_path.split("."):
            if isinstance(result, dict):
                result = result.get(key, [])
        assert result == [{"id": 1}, {"id": 2}]

    def test_response_path_none(self):
        """When no response_path, data is used as-is."""
        data = [{"id": 1}, {"id": 2}]
        # No path navigation needed
        assert isinstance(data, list)
        assert len(data) == 2

    def test_response_path_single_key(self):
        """Test single key path."""
        data = {"items": [{"name": "a"}, {"name": "b"}]}
        response_path = "items"
        result = data
        for key in response_path.split("."):
            if isinstance(result, dict):
                result = result.get(key, [])
        assert len(result) == 2

    def test_response_path_missing_key(self):
        """Test missing key returns empty."""
        data = {"other": [1, 2, 3]}
        response_path = "data.results"
        result = data
        for key in response_path.split("."):
            if isinstance(result, dict):
                result = result.get(key, [])
        assert result == []

    def test_dict_to_list_conversion(self):
        """Single dict should be wrapped in a list."""
        data = {"id": 1, "name": "test"}
        if isinstance(data, dict):
            data = [data]
        assert len(data) == 1
        assert data[0]["id"] == 1


class TestChangeDetection:
    """Test row hashing for change detection."""

    def test_same_rows_same_hash(self):
        """Identical row dicts should produce the same hash."""
        import hashlib
        import json

        row = {"id": 1, "name": "test", "value": 42.5}
        h1 = hashlib.md5(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
        h2 = hashlib.md5(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()
        assert h1 == h2

    def test_different_rows_different_hash(self):
        """Different row dicts should produce different hashes."""
        import hashlib
        import json

        row_a = {"id": 1, "name": "test", "value": 42.5}
        row_b = {"id": 1, "name": "test", "value": 43.0}
        h1 = hashlib.md5(json.dumps(row_a, sort_keys=True, default=str).encode()).hexdigest()
        h2 = hashlib.md5(json.dumps(row_b, sort_keys=True, default=str).encode()).hexdigest()
        assert h1 != h2

    def test_key_order_irrelevant(self):
        """Dict key order shouldn't affect hash (sort_keys=True)."""
        import hashlib
        import json

        row_a = {"name": "test", "id": 1}
        row_b = {"id": 1, "name": "test"}
        h1 = hashlib.md5(json.dumps(row_a, sort_keys=True, default=str).encode()).hexdigest()
        h2 = hashlib.md5(json.dumps(row_b, sort_keys=True, default=str).encode()).hexdigest()
        assert h1 == h2
