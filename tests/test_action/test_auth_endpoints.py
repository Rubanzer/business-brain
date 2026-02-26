"""Tests for core auth functions — password hashing, JWT, and user extraction."""

from unittest.mock import patch

import pytest

from business_brain.action.api import (
    ROLE_LEVELS,
    _create_jwt,
    _decode_jwt,
    _hash_password,
    _verify_password,
    get_current_user,
)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------


class TestPasswordHashing:
    def test_hash_password_produces_salt_and_hash(self):
        """_hash_password returns 'salt:hash' format with hex strings."""
        result = _hash_password("mysecret")
        parts = result.split(":")
        assert len(parts) == 2, "Expected 'salt:hash' format"
        salt, hashed = parts
        # salt is 16 bytes hex-encoded = 32 chars
        assert len(salt) == 32
        # SHA-256 hex digest = 64 chars
        assert len(hashed) == 64

    def test_verify_password_correct(self):
        """A correct password should verify against its own hash."""
        password = "correcthorse"
        pw_hash = _hash_password(password)
        assert _verify_password(password, pw_hash) is True

    def test_verify_password_incorrect(self):
        """A wrong password should not verify."""
        pw_hash = _hash_password("rightpassword")
        assert _verify_password("wrongpassword", pw_hash) is False

    def test_hash_password_different_salts(self):
        """Two hashes of the same password should differ (random salt)."""
        h1 = _hash_password("samepass")
        h2 = _hash_password("samepass")
        assert h1 != h2

    def test_verify_password_malformed_hash(self):
        """Malformed hash string should return False, not raise."""
        assert _verify_password("anything", "nocolon") is False
        assert _verify_password("anything", "") is False


# ---------------------------------------------------------------------------
# JWT create / decode
# ---------------------------------------------------------------------------


class TestJWT:
    def test_create_jwt_returns_token(self):
        """_create_jwt returns a three-part dot-separated string."""
        token = _create_jwt("user-1", "a@b.com", "admin", "pro")
        parts = token.split(".")
        assert len(parts) == 3
        assert all(len(p) > 0 for p in parts)

    def test_decode_jwt_valid_token(self):
        """Roundtrip: create then decode should return the original claims."""
        token = _create_jwt("user-42", "test@example.com", "owner", "enterprise")
        data = _decode_jwt(token)
        assert data is not None
        assert data["sub"] == "user-42"
        assert data["email"] == "test@example.com"
        assert data["role"] == "owner"
        assert data["plan"] == "enterprise"
        assert "iat" in data
        assert "exp" in data

    def test_decode_jwt_expired_token(self):
        """A token with an expiration in the past should return None."""
        # Patch time.time so the token is created in the distant past
        with patch("business_brain.action.dependencies.time") as mock_time:
            mock_time.time.return_value = 1_000_000  # epoch year ~2001
            token = _create_jwt("u1", "e@x.com", "viewer", "free")

        # Now decode with real time — the token should be expired
        assert _decode_jwt(token) is None

    def test_decode_jwt_invalid_token(self):
        """Garbage token strings should return None."""
        assert _decode_jwt("not.a.valid-token") is None
        assert _decode_jwt("") is None
        assert _decode_jwt("only-one-part") is None

    def test_decode_jwt_tampered_payload(self):
        """Changing the payload should invalidate the signature."""
        token = _create_jwt("u1", "e@x.com", "viewer", "free")
        parts = token.split(".")
        # Tamper with the payload
        tampered = parts[0] + "." + parts[1] + "X" + "." + parts[2]
        assert _decode_jwt(tampered) is None


# ---------------------------------------------------------------------------
# get_current_user (async)
# ---------------------------------------------------------------------------


class TestGetCurrentUser:
    @pytest.mark.asyncio
    async def test_get_current_user_valid_token(self):
        """With a valid Bearer token, get_current_user returns the user dict."""
        token = _create_jwt("uid-99", "user@co.com", "manager", "pro")
        user = await get_current_user(f"Bearer {token}")
        assert user is not None
        assert user["sub"] == "uid-99"
        assert user["email"] == "user@co.com"
        assert user["role"] == "manager"

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self):
        """Without an Authorization header, returns None."""
        assert await get_current_user("") is None
        assert await get_current_user("Token xyz") is None

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_bearer(self):
        """A Bearer header with an invalid token should return None."""
        assert await get_current_user("Bearer garbage.token.here") is None
