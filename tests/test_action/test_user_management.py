"""Tests for user management — listing, role updates, deactivation, and role hierarchy."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from business_brain.action.api import (
    ROLE_LEVELS,
    _create_jwt,
    deactivate_user,
    list_users,
    require_role,
    update_user_role,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_user(**overrides):
    """Create a MagicMock that behaves like a User ORM object."""
    defaults = {
        "id": "user-1",
        "email": "u@x.com",
        "name": "Test User",
        "role": "viewer",
        "plan": "free",
        "is_active": True,
        "upload_count": 0,
        "created_at": None,
        "last_login_at": None,
    }
    defaults.update(overrides)
    user = MagicMock(**defaults)
    # Ensure attribute access works (MagicMock already does this, but let's be explicit)
    for k, v in defaults.items():
        setattr(user, k, v)
    return user


def _make_session_returning(users):
    """Build an AsyncMock session whose execute().scalars().all() returns *users*."""
    session = AsyncMock()
    result = MagicMock()
    result.scalars.return_value.all.return_value = users
    result.scalars.return_value.first.return_value = users[0] if users else None
    session.execute = AsyncMock(return_value=result)
    return session


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


class TestListUsers:
    @pytest.mark.asyncio
    async def test_list_users_returns_all_active(self):
        """list_users returns all users from the database, both active and inactive."""
        u1 = _mock_user(id="u1", name="Alice", is_active=True)
        u2 = _mock_user(id="u2", name="Bob", is_active=False)
        session = _make_session_returning([u1, u2])

        admin_user = {"sub": "admin-1", "role": "admin"}
        users = await list_users(session=session, user=admin_user)

        assert len(users) == 2
        assert users[0]["id"] == "u1"
        assert users[1]["id"] == "u2"


# ---------------------------------------------------------------------------
# update_user_role
# ---------------------------------------------------------------------------


class TestUpdateRole:
    @pytest.mark.asyncio
    async def test_update_role_valid_role(self):
        """Changing a user's role to a valid role should succeed."""
        target = _mock_user(id="target-1", role="viewer")
        session = _make_session_returning([target])

        admin_user = {"sub": "admin-1", "role": "admin"}
        req = MagicMock()
        req.role = "operator"

        result = await update_user_role(
            user_id="target-1", req=req, session=session, user=admin_user,
        )
        assert result["status"] == "updated"
        assert result["new_role"] == "operator"
        assert target.role == "operator"
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_role_invalid_role_raises(self):
        """An invalid role name should raise HTTP 400."""
        session = AsyncMock()
        admin_user = {"sub": "admin-1", "role": "admin"}
        req = MagicMock()
        req.role = "superadmin"  # not in ROLE_LEVELS

        with pytest.raises(HTTPException) as exc_info:
            await update_user_role(
                user_id="u1", req=req, session=session, user=admin_user,
            )
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_update_role_prevents_owner_change_by_non_owner(self):
        """A non-owner admin should not be able to change an owner's role."""
        target = _mock_user(id="owner-1", role="owner")
        session = _make_session_returning([target])

        admin_user = {"sub": "admin-1", "role": "admin"}  # not an owner
        req = MagicMock()
        req.role = "viewer"

        with pytest.raises(HTTPException) as exc_info:
            await update_user_role(
                user_id="owner-1", req=req, session=session, user=admin_user,
            )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_update_role_user_not_found(self):
        """Updating a nonexistent user should raise HTTP 404."""
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result)

        admin_user = {"sub": "admin-1", "role": "admin"}
        req = MagicMock()
        req.role = "operator"

        with pytest.raises(HTTPException) as exc_info:
            await update_user_role(
                user_id="ghost", req=req, session=session, user=admin_user,
            )
        assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# deactivate_user
# ---------------------------------------------------------------------------


class TestDeactivateUser:
    @pytest.mark.asyncio
    async def test_deactivate_user_sets_inactive(self):
        """Deactivating a user should set is_active = False."""
        target = _mock_user(id="target-1", is_active=True)
        session = _make_session_returning([target])

        owner_user = {"sub": "owner-1", "role": "owner"}
        result = await deactivate_user(
            user_id="target-1", session=session, user=owner_user,
        )
        assert result["status"] == "deactivated"
        assert target.is_active is False
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_deactivate_self_raises_error(self):
        """An owner should not be able to deactivate their own account."""
        target = _mock_user(id="owner-1")
        session = _make_session_returning([target])

        owner_user = {"sub": "owner-1", "role": "owner"}

        with pytest.raises(HTTPException) as exc_info:
            await deactivate_user(
                user_id="owner-1", session=session, user=owner_user,
            )
        assert exc_info.value.status_code == 400
        assert "own account" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# Role hierarchy and require_role
# ---------------------------------------------------------------------------


class TestRoleHierarchy:
    def test_role_hierarchy_owner_is_highest(self):
        """Owner should have the highest level in ROLE_LEVELS."""
        max_role = max(ROLE_LEVELS, key=ROLE_LEVELS.get)
        assert max_role == "owner"

    def test_role_hierarchy_viewer_is_lowest(self):
        """Viewer should have the lowest level in ROLE_LEVELS."""
        min_role = min(ROLE_LEVELS, key=ROLE_LEVELS.get)
        assert min_role == "viewer"

    def test_role_hierarchy_ordering(self):
        """Roles should be ordered: viewer < operator < manager < admin < owner."""
        assert ROLE_LEVELS["viewer"] < ROLE_LEVELS["operator"]
        assert ROLE_LEVELS["operator"] < ROLE_LEVELS["manager"]
        assert ROLE_LEVELS["manager"] < ROLE_LEVELS["admin"]
        assert ROLE_LEVELS["admin"] < ROLE_LEVELS["owner"]

    @pytest.mark.asyncio
    async def test_require_role_rejects_insufficient(self):
        """require_role('admin') should reject a viewer-level token."""
        token = _create_jwt("u1", "e@x.com", "viewer", "free")
        checker = require_role("admin")

        with pytest.raises(HTTPException) as exc_info:
            await checker(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_require_role_accepts_sufficient(self):
        """require_role('operator') should accept an admin-level token."""
        token = _create_jwt("u1", "e@x.com", "admin", "pro")
        checker = require_role("operator")

        user = await checker(authorization=f"Bearer {token}")
        assert user["role"] == "admin"

    @pytest.mark.asyncio
    async def test_require_role_rejects_no_auth(self):
        """require_role should raise 401 when no auth header is provided."""
        checker = require_role("viewer")

        with pytest.raises(HTTPException) as exc_info:
            await checker(authorization="")
        assert exc_info.value.status_code == 401
