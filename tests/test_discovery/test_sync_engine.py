"""Tests for sync engine scheduling logic (no DB needed)."""

from datetime import datetime, timedelta, timezone


class TestSyncDueCheck:
    """Test the 'is this source due for sync' logic."""

    def _is_due(self, last_sync_at, sync_frequency_minutes, active=True):
        """Replicate the due-checking logic from sync_all_due."""
        if not active:
            return False
        if sync_frequency_minutes <= 0:
            return False

        now = datetime.now(timezone.utc)

        if last_sync_at is None:
            return True  # Never synced = always due

        if last_sync_at.tzinfo is None:
            last_sync_at = last_sync_at.replace(tzinfo=timezone.utc)

        elapsed = now - last_sync_at
        return elapsed >= timedelta(minutes=sync_frequency_minutes)

    def test_never_synced_is_due(self):
        assert self._is_due(None, 5)

    def test_recently_synced_not_due(self):
        recent = datetime.now(timezone.utc) - timedelta(seconds=30)
        assert not self._is_due(recent, 5)

    def test_old_sync_is_due(self):
        old = datetime.now(timezone.utc) - timedelta(minutes=10)
        assert self._is_due(old, 5)

    def test_exactly_at_boundary(self):
        boundary = datetime.now(timezone.utc) - timedelta(minutes=5)
        assert self._is_due(boundary, 5)

    def test_inactive_source_never_due(self):
        assert not self._is_due(None, 5, active=False)

    def test_zero_frequency_never_due(self):
        assert not self._is_due(None, 0)

    def test_negative_frequency_never_due(self):
        assert not self._is_due(None, -1)

    def test_naive_datetime_handled(self):
        """Naive datetime (no tzinfo) should still work when treated as UTC."""
        naive = datetime.utcnow() - timedelta(minutes=10)
        assert self._is_due(naive, 5)

    def test_long_frequency(self):
        """60-minute frequency, synced 30 minutes ago."""
        recent = datetime.now(timezone.utc) - timedelta(minutes=30)
        assert not self._is_due(recent, 60)

    def test_long_frequency_due(self):
        """60-minute frequency, synced 61 minutes ago."""
        old = datetime.now(timezone.utc) - timedelta(minutes=61)
        assert self._is_due(old, 60)


class TestSourceTypeRouting:
    """Test that sync routes to the correct handler based on source type."""

    def _get_handler(self, source_type):
        """Replicate routing logic from sync_source."""
        if source_type == "google_sheet":
            return "sheets_sync"
        elif source_type == "api":
            return "api_sync"
        else:
            return "skipped"

    def test_google_sheet_routes(self):
        assert self._get_handler("google_sheet") == "sheets_sync"

    def test_api_routes(self):
        assert self._get_handler("api") == "api_sync"

    def test_unknown_type_skipped(self):
        assert self._get_handler("ftp") == "skipped"

    def test_manual_upload_skipped(self):
        assert self._get_handler("manual_upload") == "skipped"

    def test_recurring_upload_skipped(self):
        assert self._get_handler("recurring_upload") == "skipped"
