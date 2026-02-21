"""Tests for invite flow — create, accept, list, and revoke invites."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from business_brain.action.api import (
    accept_invite,
    create_invite,
    list_invites,
    revoke_invite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_invite(**overrides):
    """Create a MagicMock that behaves like an InviteToken ORM object."""
    defaults = {
        "id": "inv-1",
        "email": "invitee@x.com",
        "role": "viewer",
        "plan": "free",
        "company_id": None,
        "token": "abc123tokenxyz",
        "used": False,
        "expires_at": datetime.utcnow() + timedelta(days=7),
        "created_by": "admin-1",
    }
    defaults.update(overrides)
    inv = MagicMock(**defaults)
    for k, v in defaults.items():
        setattr(inv, k, v)
    return inv


def _mock_user(**overrides):
    """Create a MagicMock that behaves like a User ORM object."""
    defaults = {
        "id": "user-1",
        "email": "u@x.com",
        "name": "Test User",
        "password_hash": "salt:hash",
        "role": "viewer",
        "plan": "free",
        "company_id": None,
        "is_active": True,
    }
    defaults.update(overrides)
    user = MagicMock(**defaults)
    for k, v in defaults.items():
        setattr(user, k, v)
    return user


# ---------------------------------------------------------------------------
# create_invite
# ---------------------------------------------------------------------------


class TestCreateInvite:
    @pytest.mark.asyncio
    async def test_create_invite_generates_token(self):
        """create_invite should add an InviteToken to the session and return a token string."""
        session = AsyncMock()
        admin_user = {"sub": "admin-1", "role": "admin"}

        req = MagicMock()
        req.email = "new@company.com"
        req.role = "operator"
        req.plan = "free"

        with patch("business_brain.db.v3_models.InviteToken") as MockInvite:
            instance = MagicMock()
            MockInvite.return_value = instance

            result = await create_invite(req=req, session=session, user=admin_user)

        assert result["status"] == "created"
        assert result["email"] == "new@company.com"
        assert result["role"] == "operator"
        assert "token" in result
        session.add.assert_called_once()
        session.commit.assert_called_once()


# ---------------------------------------------------------------------------
# accept_invite
# ---------------------------------------------------------------------------


class TestAcceptInvite:
    @pytest.mark.asyncio
    async def test_accept_invite_creates_user_with_role(self):
        """Accepting a valid invite should create a user with the invite's role."""
        invite = _mock_invite(role="operator", plan="basic")
        session = AsyncMock()

        # First execute: find invite by token (returns invite)
        invite_result = MagicMock()
        invite_result.scalars.return_value.first.return_value = invite

        # Second execute: check if email exists (returns None — no existing user)
        email_result = MagicMock()
        email_result.scalars.return_value.first.return_value = None

        session.execute = AsyncMock(side_effect=[invite_result, email_result])

        # After commit + refresh, the new user should have attributes set
        async def fake_refresh(user):
            user.id = "new-user-id"
            user.email = invite.email
            user.name = "Invited User"
            user.role = invite.role
            user.plan = invite.plan

        session.refresh = fake_refresh

        req = MagicMock()
        req.token = invite.token
        req.name = "Invited User"
        req.password = "securepass"

        with patch("business_brain.db.v3_models.User") as MockUser:
            new_user = MagicMock()
            new_user.id = "new-user-id"
            new_user.email = invite.email
            new_user.name = "Invited User"
            new_user.role = "operator"
            new_user.plan = "basic"
            MockUser.return_value = new_user

            result = await accept_invite(req=req, session=session)

        assert result["status"] == "registered"
        assert result["user"]["role"] == "operator"
        assert result["user"]["plan"] == "basic"
        assert invite.used is True
        session.add.assert_called_once()
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_accept_expired_invite_fails(self):
        """An invite that has expired should raise HTTP 400."""
        invite = _mock_invite(
            expires_at=datetime.utcnow() - timedelta(days=1),  # expired
        )
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.first.return_value = invite
        session.execute = AsyncMock(return_value=result)

        req = MagicMock()
        req.token = invite.token
        req.name = "Late User"
        req.password = "pass"

        with pytest.raises(HTTPException) as exc_info:
            await accept_invite(req=req, session=session)
        assert exc_info.value.status_code == 400
        assert "expired" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_accept_used_invite_fails(self):
        """An invite that has already been used should raise HTTP 400."""
        session = AsyncMock()
        # When used=True, the query filters it out, so scalars().first() returns None
        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result)

        req = MagicMock()
        req.token = "used-token"
        req.name = "User"
        req.password = "pass"

        with pytest.raises(HTTPException) as exc_info:
            await accept_invite(req=req, session=session)
        assert exc_info.value.status_code == 400
        assert "invalid" in exc_info.value.detail.lower() or "expired" in exc_info.value.detail.lower()


# ---------------------------------------------------------------------------
# list_invites
# ---------------------------------------------------------------------------


class TestListInvites:
    @pytest.mark.asyncio
    async def test_list_pending_invites_excludes_used(self):
        """list_invites should return only unused invites (the query filters used=False)."""
        pending = _mock_invite(id="inv-1", used=False)
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = [pending]
        session.execute = AsyncMock(return_value=result)

        admin_user = {"sub": "admin-1", "role": "admin"}
        invites = await list_invites(session=session, user=admin_user)

        assert len(invites) == 1
        assert invites[0]["id"] == "inv-1"
        assert "email" in invites[0]
        assert "token" in invites[0]

    @pytest.mark.asyncio
    async def test_list_invites_empty(self):
        """When there are no pending invites, list_invites returns an empty list."""
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.all.return_value = []
        session.execute = AsyncMock(return_value=result)

        admin_user = {"sub": "admin-1", "role": "admin"}
        invites = await list_invites(session=session, user=admin_user)
        assert invites == []

    @pytest.mark.asyncio
    async def test_list_invites_handles_db_error(self):
        """list_invites should return [] on database errors (it catches Exception)."""
        session = AsyncMock()
        session.execute = AsyncMock(side_effect=Exception("DB down"))

        admin_user = {"sub": "admin-1", "role": "admin"}
        invites = await list_invites(session=session, user=admin_user)
        assert invites == []


# ---------------------------------------------------------------------------
# revoke_invite
# ---------------------------------------------------------------------------


class TestRevokeInvite:
    @pytest.mark.asyncio
    async def test_revoke_invite_deletes(self):
        """Revoking a pending invite should delete it from the session."""
        invite = _mock_invite(id="inv-1", used=False)
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.first.return_value = invite
        session.execute = AsyncMock(return_value=result)

        admin_user = {"sub": "admin-1", "role": "admin"}
        resp = await revoke_invite(invite_id="inv-1", session=session, user=admin_user)

        assert resp["status"] == "revoked"
        assert resp["invite_id"] == "inv-1"
        session.delete.assert_called_once_with(invite)
        session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_invite_404(self):
        """Revoking an invite that doesn't exist should raise HTTP 404."""
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.first.return_value = None
        session.execute = AsyncMock(return_value=result)

        admin_user = {"sub": "admin-1", "role": "admin"}
        with pytest.raises(HTTPException) as exc_info:
            await revoke_invite(invite_id="ghost", session=session, user=admin_user)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_revoke_used_invite_400(self):
        """Revoking an already-used invite should raise HTTP 400."""
        invite = _mock_invite(id="inv-1", used=True)
        session = AsyncMock()
        result = MagicMock()
        result.scalars.return_value.first.return_value = invite
        session.execute = AsyncMock(return_value=result)

        admin_user = {"sub": "admin-1", "role": "admin"}
        with pytest.raises(HTTPException) as exc_info:
            await revoke_invite(invite_id="inv-1", session=session, user=admin_user)
        assert exc_info.value.status_code == 400
        assert "already used" in exc_info.value.detail.lower()
