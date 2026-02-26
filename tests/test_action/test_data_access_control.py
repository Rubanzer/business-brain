"""Comprehensive tests for data-level access control.

Tests the _get_accessible_tables() function and metadata_store.upsert()
tracking of uploaded_by / uploaded_by_role for role-based data isolation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from business_brain.action.api import _get_accessible_tables, ROLE_LEVELS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(table_name, uploaded_by=None, uploaded_by_role=None):
    """Create a mock MetadataEntry with the given attributes."""
    entry = MagicMock()
    entry.table_name = table_name
    entry.uploaded_by = uploaded_by
    entry.uploaded_by_role = uploaded_by_role
    return entry


def _user(sub, role):
    """Shorthand to build a user dict matching JWT payload shape."""
    return {"sub": sub, "role": role}


# ---------------------------------------------------------------------------
# TestGetAccessibleTables
# ---------------------------------------------------------------------------


class TestGetAccessibleTables:
    """Tests for _get_accessible_tables role-hierarchy filtering."""

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_none_user_returns_none(self, mock_store):
        """No auth (user=None) means all tables visible (backward compat)."""
        session = AsyncMock()
        result = await _get_accessible_tables(session, None)
        assert result is None
        # metadata_store should never be called when there is no auth
        mock_store.get_all.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_owner_returns_none(self, mock_store):
        """Owner role gets None (full access to all tables)."""
        session = AsyncMock()
        result = await _get_accessible_tables(session, _user("owner1", "owner"))
        assert result is None
        mock_store.get_all.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_admin_returns_none(self, mock_store):
        """Admin role gets None (full access to all tables)."""
        session = AsyncMock()
        result = await _get_accessible_tables(session, _user("admin1", "admin"))
        assert result is None
        mock_store.get_all.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_viewer_sees_own_uploads(self, mock_store):
        """Viewer 'v1' uploaded 'sales' -> viewer sees 'sales'."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("sales", uploaded_by="v1", uploaded_by_role="viewer"),
        ])

        result = await _get_accessible_tables(session, _user("v1", "viewer"))
        assert result == ["sales"]

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_viewer_cannot_see_admin_uploads(self, mock_store):
        """Admin uploaded 'secrets' -> viewer cannot see it."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("secrets", uploaded_by="admin1", uploaded_by_role="admin"),
        ])

        result = await _get_accessible_tables(session, _user("v1", "viewer"))
        assert result == []

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_viewer_cannot_see_operator_uploads(self, mock_store):
        """Operator uploaded 'ops_data' -> viewer cannot see it."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("ops_data", uploaded_by="op1", uploaded_by_role="operator"),
        ])

        result = await _get_accessible_tables(session, _user("v1", "viewer"))
        assert result == []

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_viewer_sees_legacy_tables(self, mock_store):
        """Table with uploaded_by=None (legacy) is visible to viewer."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("old_report", uploaded_by=None, uploaded_by_role=None),
        ])

        result = await _get_accessible_tables(session, _user("v1", "viewer"))
        assert result == ["old_report"]

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_operator_sees_own_uploads(self, mock_store):
        """Operator sees tables they uploaded themselves."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("ops_table", uploaded_by="op1", uploaded_by_role="operator"),
        ])

        result = await _get_accessible_tables(session, _user("op1", "operator"))
        assert result == ["ops_table"]

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_operator_sees_viewer_uploads(self, mock_store):
        """Operator can see viewer-uploaded tables (lower role)."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("viewer_data", uploaded_by="v1", uploaded_by_role="viewer"),
        ])

        result = await _get_accessible_tables(session, _user("op1", "operator"))
        assert result == ["viewer_data"]

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_operator_sees_other_operator_uploads(self, mock_store):
        """Operator can see tables uploaded by other operators (equal role level)."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("peer_ops", uploaded_by="op2", uploaded_by_role="operator"),
        ])

        result = await _get_accessible_tables(session, _user("op1", "operator"))
        assert result == ["peer_ops"]

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_operator_cannot_see_admin_uploads(self, mock_store):
        """Operator cannot see admin-uploaded tables."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("admin_only", uploaded_by="admin1", uploaded_by_role="admin"),
        ])

        result = await _get_accessible_tables(session, _user("op1", "operator"))
        assert result == []

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_manager_sees_operator_and_viewer(self, mock_store):
        """Manager sees uploads from operator and viewer (both lower roles)."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("ops_report", uploaded_by="op1", uploaded_by_role="operator"),
            _make_entry("viewer_csv", uploaded_by="v1", uploaded_by_role="viewer"),
        ])

        result = await _get_accessible_tables(session, _user("mgr1", "manager"))
        assert "ops_report" in result
        assert "viewer_csv" in result
        assert len(result) == 2

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_manager_sees_other_manager_uploads(self, mock_store):
        """Manager can see tables uploaded by other managers (equal role level)."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("peer_mgr_data", uploaded_by="mgr2", uploaded_by_role="manager"),
        ])

        result = await _get_accessible_tables(session, _user("mgr1", "manager"))
        assert result == ["peer_mgr_data"]

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_manager_cannot_see_admin_uploads(self, mock_store):
        """Manager cannot see admin-uploaded tables."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("admin_secret", uploaded_by="admin1", uploaded_by_role="admin"),
        ])

        result = await _get_accessible_tables(session, _user("mgr1", "manager"))
        assert result == []

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_mixed_tables(self, mock_store):
        """Mix of legacy, own, lower-role, and higher-role uploads.

        Scenario for operator 'op1':
        - legacy_tbl (no uploader)        -> accessible
        - own_tbl (op1, operator)          -> accessible (own upload)
        - viewer_tbl (v1, viewer)          -> accessible (lower role)
        - other_op_tbl (op2, operator)     -> accessible (equal role)
        - mgr_tbl (mgr1, manager)          -> NOT accessible (higher role)
        - admin_tbl (admin1, admin)         -> NOT accessible (higher role)
        - owner_tbl (owner1, owner)         -> NOT accessible (higher role)
        """
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("legacy_tbl", uploaded_by=None, uploaded_by_role=None),
            _make_entry("own_tbl", uploaded_by="op1", uploaded_by_role="operator"),
            _make_entry("viewer_tbl", uploaded_by="v1", uploaded_by_role="viewer"),
            _make_entry("other_op_tbl", uploaded_by="op2", uploaded_by_role="operator"),
            _make_entry("mgr_tbl", uploaded_by="mgr1", uploaded_by_role="manager"),
            _make_entry("admin_tbl", uploaded_by="admin1", uploaded_by_role="admin"),
            _make_entry("owner_tbl", uploaded_by="owner1", uploaded_by_role="owner"),
        ])

        result = await _get_accessible_tables(session, _user("op1", "operator"))

        assert "legacy_tbl" in result
        assert "own_tbl" in result
        assert "viewer_tbl" in result
        assert "other_op_tbl" in result
        assert "mgr_tbl" not in result
        assert "admin_tbl" not in result
        assert "owner_tbl" not in result
        assert len(result) == 4

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_empty_metadata_returns_empty_list(self, mock_store):
        """No tables at all returns empty list (not None)."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[])

        result = await _get_accessible_tables(session, _user("v1", "viewer"))
        assert result == []
        assert isinstance(result, list)

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_db_error_returns_none_failopen(self, mock_store):
        """metadata_store.get_all raises exception -> returns None (fail-open)."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(side_effect=RuntimeError("DB connection lost"))

        result = await _get_accessible_tables(session, _user("v1", "viewer"))
        assert result is None

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_unknown_user_role_treated_as_viewer(self, mock_store):
        """User with role='intern' (not in ROLE_LEVELS) gets level 0 (same as viewer).

        Should only see own uploads and legacy tables, not operator/manager uploads.
        """
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("legacy", uploaded_by=None),
            _make_entry("own_data", uploaded_by="intern1", uploaded_by_role="intern"),
            _make_entry("op_data", uploaded_by="op1", uploaded_by_role="operator"),
            _make_entry("viewer_data", uploaded_by="v1", uploaded_by_role="viewer"),
        ])

        result = await _get_accessible_tables(session, _user("intern1", "intern"))

        assert "legacy" in result
        assert "own_data" in result
        # viewer level == 0, intern level == 0, so viewer uploads ARE visible (equal level)
        assert "viewer_data" in result
        # operator level == 1 > 0, so NOT visible
        assert "op_data" not in result
        assert len(result) == 3


# ---------------------------------------------------------------------------
# TestMetadataStoreUpsertTracking
# ---------------------------------------------------------------------------


class TestMetadataStoreUpsertTracking:
    """Tests that metadata_store.upsert correctly tracks uploaded_by and
    uploaded_by_role on creation but preserves them on update."""

    @pytest.mark.asyncio
    @patch("business_brain.memory.metadata_store.get_by_table")
    async def test_upsert_new_entry_sets_uploaded_by(self, mock_get_by_table):
        """New entry gets uploaded_by and uploaded_by_role set."""
        from business_brain.memory.metadata_store import upsert

        mock_get_by_table.return_value = None  # no existing entry

        session = AsyncMock()
        # Track what gets added to the session
        added_entries = []
        session.add = MagicMock(side_effect=lambda e: added_entries.append(e))
        session.refresh = AsyncMock()

        result = await upsert(
            session,
            table_name="new_table",
            description="Fresh upload",
            columns_metadata=[{"name": "col1", "type": "text"}],
            uploaded_by="user42",
            uploaded_by_role="operator",
        )

        # Verify session.add was called with a MetadataEntry
        assert len(added_entries) == 1
        new_entry = added_entries[0]
        assert new_entry.table_name == "new_table"
        assert new_entry.uploaded_by == "user42"
        assert new_entry.uploaded_by_role == "operator"
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    @patch("business_brain.memory.metadata_store.get_by_table")
    async def test_upsert_existing_does_not_overwrite_uploader(self, mock_get_by_table):
        """Updating an existing entry preserves original uploaded_by and uploaded_by_role."""
        from business_brain.memory.metadata_store import upsert

        existing = MagicMock()
        existing.table_name = "existing_table"
        existing.description = "Old description"
        existing.columns_metadata = None
        existing.uploaded_by = "original_user"
        existing.uploaded_by_role = "viewer"
        mock_get_by_table.return_value = existing

        session = AsyncMock()
        session.refresh = AsyncMock()

        result = await upsert(
            session,
            table_name="existing_table",
            description="Updated description",
            columns_metadata=[{"name": "col1", "type": "int"}],
            uploaded_by="different_user",
            uploaded_by_role="admin",
        )

        # Description and columns get updated
        assert existing.description == "Updated description"
        assert existing.columns_metadata == [{"name": "col1", "type": "int"}]
        # uploaded_by and uploaded_by_role should NOT be overwritten
        assert existing.uploaded_by == "original_user"
        assert existing.uploaded_by_role == "viewer"
        # session.add should NOT be called (entry already exists)
        session.add.assert_not_called()

    @pytest.mark.asyncio
    @patch("business_brain.memory.metadata_store.get_by_table")
    async def test_upsert_without_uploader_leaves_null(self, mock_get_by_table):
        """When uploaded_by=None, field stays null on new entry."""
        from business_brain.memory.metadata_store import upsert

        mock_get_by_table.return_value = None  # no existing entry

        session = AsyncMock()
        added_entries = []
        session.add = MagicMock(side_effect=lambda e: added_entries.append(e))
        session.refresh = AsyncMock()

        result = await upsert(
            session,
            table_name="legacy_import",
            description="Imported without auth",
            columns_metadata=None,
            uploaded_by=None,
            uploaded_by_role=None,
        )

        assert len(added_entries) == 1
        new_entry = added_entries[0]
        assert new_entry.uploaded_by is None
        assert new_entry.uploaded_by_role is None


# ---------------------------------------------------------------------------
# TestAccessControlIntegration
# ---------------------------------------------------------------------------


class TestAccessControlIntegration:
    """Integration-style tests combining access control logic with
    realistic multi-table scenarios."""

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_viewer_with_no_uploads_sees_only_legacy(self, mock_store):
        """A viewer who uploaded nothing sees only legacy tables (no uploader recorded)."""
        session = AsyncMock()
        mock_store.get_all = AsyncMock(return_value=[
            _make_entry("legacy_report", uploaded_by=None, uploaded_by_role=None),
            _make_entry("legacy_dashboard", uploaded_by=None, uploaded_by_role=None),
            _make_entry("op_upload", uploaded_by="op1", uploaded_by_role="operator"),
            _make_entry("mgr_upload", uploaded_by="mgr1", uploaded_by_role="manager"),
            _make_entry("admin_upload", uploaded_by="admin1", uploaded_by_role="admin"),
        ])

        result = await _get_accessible_tables(session, _user("new_viewer", "viewer"))

        assert "legacy_report" in result
        assert "legacy_dashboard" in result
        assert "op_upload" not in result
        assert "mgr_upload" not in result
        assert "admin_upload" not in result
        assert len(result) == 2

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_all_roles_see_legacy_tables(self, mock_store):
        """Every restricted role (viewer, operator, manager) sees tables with no uploader."""
        session = AsyncMock()
        legacy_entries = [
            _make_entry("shared_data", uploaded_by=None, uploaded_by_role=None),
            _make_entry("historical", uploaded_by=None, uploaded_by_role=None),
        ]

        for role in ("viewer", "operator", "manager"):
            mock_store.get_all = AsyncMock(return_value=list(legacy_entries))
            result = await _get_accessible_tables(
                session, _user(f"{role}_user", role)
            )
            assert "shared_data" in result, f"{role} should see 'shared_data'"
            assert "historical" in result, f"{role} should see 'historical'"
            assert len(result) == 2, f"{role} should see exactly 2 legacy tables"

    @pytest.mark.asyncio
    @patch("business_brain.action.dependencies.metadata_store")
    async def test_promoted_user_sees_more(self, mock_store):
        """If a user's role changes from viewer to manager (new JWT),
        they immediately see more tables."""
        session = AsyncMock()
        entries = [
            _make_entry("legacy", uploaded_by=None, uploaded_by_role=None),
            _make_entry("own_table", uploaded_by="user1", uploaded_by_role="viewer"),
            _make_entry("op_table", uploaded_by="op1", uploaded_by_role="operator"),
            _make_entry("mgr_table", uploaded_by="mgr2", uploaded_by_role="manager"),
            _make_entry("admin_table", uploaded_by="admin1", uploaded_by_role="admin"),
        ]

        # As viewer: sees legacy + own only
        mock_store.get_all = AsyncMock(return_value=list(entries))
        viewer_result = await _get_accessible_tables(
            session, _user("user1", "viewer")
        )
        assert "legacy" in viewer_result
        assert "own_table" in viewer_result
        assert "op_table" not in viewer_result
        assert "mgr_table" not in viewer_result
        assert "admin_table" not in viewer_result
        assert len(viewer_result) == 2

        # Promoted to manager: sees legacy + own + operator + other managers
        mock_store.get_all = AsyncMock(return_value=list(entries))
        manager_result = await _get_accessible_tables(
            session, _user("user1", "manager")
        )
        assert "legacy" in manager_result
        assert "own_table" in manager_result
        assert "op_table" in manager_result
        assert "mgr_table" in manager_result
        assert "admin_table" not in manager_result
        assert len(manager_result) == 4

        # The manager sees strictly more tables than the viewer did
        assert len(manager_result) > len(viewer_result)
        assert set(viewer_result).issubset(set(manager_result))


# ---------------------------------------------------------------------------
# TestRoleLevelsConstant
# ---------------------------------------------------------------------------


class TestRoleLevelsConstant:
    """Sanity checks on the ROLE_LEVELS mapping itself."""

    def test_all_expected_roles_present(self):
        assert set(ROLE_LEVELS.keys()) == {"viewer", "operator", "manager", "admin", "owner"}

    def test_hierarchy_ordering(self):
        assert ROLE_LEVELS["viewer"] < ROLE_LEVELS["operator"]
        assert ROLE_LEVELS["operator"] < ROLE_LEVELS["manager"]
        assert ROLE_LEVELS["manager"] < ROLE_LEVELS["admin"]
        assert ROLE_LEVELS["admin"] < ROLE_LEVELS["owner"]

    def test_levels_are_integers(self):
        for role, level in ROLE_LEVELS.items():
            assert isinstance(level, int), f"Level for {role} should be int"
