"""Comprehensive tests for role hierarchy and require_role access control."""

import pytest
from fastapi import HTTPException

from business_brain.action.api import (
    ROLE_LEVELS,
    _create_jwt,
    require_role,
)


# ---------------------------------------------------------------------------
# Full 5x5 role matrix (25 combinations)
# ---------------------------------------------------------------------------


class TestRoleHierarchyMatrix:
    """Every user-role vs. min-role combination must behave correctly."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "user_role,min_role,should_pass",
        [
            # viewer (level 0) — can only access viewer-level endpoints
            ("viewer", "viewer", True),
            ("viewer", "operator", False),
            ("viewer", "manager", False),
            ("viewer", "admin", False),
            ("viewer", "owner", False),
            # operator (level 1) — can access viewer + operator
            ("operator", "viewer", True),
            ("operator", "operator", True),
            ("operator", "manager", False),
            ("operator", "admin", False),
            ("operator", "owner", False),
            # manager (level 2) — can access viewer + operator + manager
            ("manager", "viewer", True),
            ("manager", "operator", True),
            ("manager", "manager", True),
            ("manager", "admin", False),
            ("manager", "owner", False),
            # admin (level 3) — can access everything except owner
            ("admin", "viewer", True),
            ("admin", "operator", True),
            ("admin", "manager", True),
            ("admin", "admin", True),
            ("admin", "owner", False),
            # owner (level 4) — full access
            ("owner", "viewer", True),
            ("owner", "operator", True),
            ("owner", "manager", True),
            ("owner", "admin", True),
            ("owner", "owner", True),
        ],
    )
    async def test_role_matrix(self, user_role, min_role, should_pass):
        """Verify that role level comparison works for every pair."""
        token = _create_jwt("u1", "u@x.com", user_role, "free")
        checker = require_role(min_role)

        if should_pass:
            result = await checker(authorization=f"Bearer {token}")
            assert isinstance(result, dict)
            assert result["sub"] == "u1"
            assert result["email"] == "u@x.com"
            assert result["role"] == user_role
        else:
            with pytest.raises(HTTPException) as exc_info:
                await checker(authorization=f"Bearer {token}")
            assert exc_info.value.status_code == 403
            assert min_role in exc_info.value.detail
            assert user_role in exc_info.value.detail


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestRoleEdgeCases:
    """Boundary conditions and unusual inputs for the role system."""

    @pytest.mark.asyncio
    async def test_unknown_role_treated_as_level_zero(self):
        """A JWT with a role not in ROLE_LEVELS should resolve to level 0.

        This means it can pass a 'viewer' check (level 0) but fail an
        'operator' check (level 1).
        """
        token = _create_jwt("u2", "super@x.com", "superadmin", "free")

        # Should pass viewer (level 0 >= 0)
        viewer_checker = require_role("viewer")
        result = await viewer_checker(authorization=f"Bearer {token}")
        assert result["role"] == "superadmin"

        # Should fail operator (level 0 < 1)
        operator_checker = require_role("operator")
        with pytest.raises(HTTPException) as exc_info:
            await operator_checker(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_empty_role_treated_as_viewer(self):
        """A JWT with an empty-string role should resolve to level 0.

        Empty string is not in ROLE_LEVELS, so ROLE_LEVELS.get("", 0) -> 0.
        """
        token = _create_jwt("u3", "empty@x.com", "", "free")

        # Should pass viewer (level 0 >= 0)
        viewer_checker = require_role("viewer")
        result = await viewer_checker(authorization=f"Bearer {token}")
        assert result["role"] == ""

        # Should fail operator (level 0 < 1)
        operator_checker = require_role("operator")
        with pytest.raises(HTTPException) as exc_info:
            await operator_checker(authorization=f"Bearer {token}")
        assert exc_info.value.status_code == 403

    def test_role_levels_ordering(self):
        """The five roles must have strictly ascending levels."""
        assert ROLE_LEVELS["viewer"] < ROLE_LEVELS["operator"]
        assert ROLE_LEVELS["operator"] < ROLE_LEVELS["manager"]
        assert ROLE_LEVELS["manager"] < ROLE_LEVELS["admin"]
        assert ROLE_LEVELS["admin"] < ROLE_LEVELS["owner"]

        # Also verify exact values for completeness
        assert ROLE_LEVELS == {
            "viewer": 0,
            "operator": 1,
            "manager": 2,
            "admin": 3,
            "owner": 4,
        }

    @pytest.mark.asyncio
    async def test_require_role_returns_user_dict(self):
        """On success, the returned dict must contain sub, email, role, plan."""
        token = _create_jwt("uid-42", "alice@example.com", "admin", "pro")
        checker = require_role("viewer")
        result = await checker(authorization=f"Bearer {token}")

        assert result["sub"] == "uid-42"
        assert result["email"] == "alice@example.com"
        assert result["role"] == "admin"
        assert result["plan"] == "pro"
        # JWT standard claims should also be present
        assert "iat" in result
        assert "exp" in result

    @pytest.mark.asyncio
    async def test_require_role_no_auth_raises_401(self):
        """An empty Authorization header must raise 401, not 403."""
        checker = require_role("viewer")
        with pytest.raises(HTTPException) as exc_info:
            await checker(authorization="")
        assert exc_info.value.status_code == 401
        assert "Authentication required" in exc_info.value.detail
