"""Comprehensive edge-case tests for the auth system in business-brain.

Covers password hashing, JWT creation/decoding, get_current_user,
and require_role with boundary conditions and malformed inputs.
"""

import base64
import json

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from business_brain.action.api import (
    _hash_password,
    _verify_password,
    _create_jwt,
    _decode_jwt,
    get_current_user,
    require_role,
    ROLE_LEVELS,
)


# ---------------------------------------------------------------------------
# Password edge cases
# ---------------------------------------------------------------------------


class TestPasswordEdgeCases:
    """Edge cases for _hash_password and _verify_password."""

    def test_hash_empty_password(self):
        """An empty string should still produce a valid 'salt:hash' result."""
        result = _hash_password("")
        parts = result.split(":")
        assert len(parts) == 2, "Expected 'salt:hash' format"
        salt, hashed = parts
        # salt = 16 random bytes hex-encoded = 32 hex chars
        assert len(salt) == 32
        # SHA-256 hex digest = 64 chars
        assert len(hashed) == 64

    def test_verify_empty_password_match(self):
        """hash('') then verify('') should return True."""
        pw_hash = _hash_password("")
        assert _verify_password("", pw_hash) is True

    def test_hash_unicode_password(self):
        """Unicode passwords (emoji, CJK characters) should hash and verify."""
        passwords = [
            "\U0001f525\U0001f4a9\U0001f680",  # fire, poop, rocket emojis
            "\u4f60\u597d\u4e16\u754c",          # Chinese: "hello world"
            "\u00e9\u00e0\u00fc\u00f1\u00df",    # accented Latin chars
            "\u0410\u0411\u0412\u0413",          # Cyrillic
        ]
        for pw in passwords:
            pw_hash = _hash_password(pw)
            assert _verify_password(pw, pw_hash) is True, (
                f"Unicode password {pw!r} should verify against its own hash"
            )
            # Wrong password should still fail
            assert _verify_password(pw + "x", pw_hash) is False

    def test_verify_wrong_type_returns_false(self):
        """Passing None or int as password_hash should return False, not raise."""
        # _verify_password calls password_hash.split(":") which will raise
        # on non-string types, but the except clause catches it.
        assert _verify_password("anything", None) is False
        assert _verify_password("anything", 12345) is False
        assert _verify_password("anything", True) is False

    def test_verify_no_colon_returns_false(self):
        """A password_hash without a colon separator should return False."""
        assert _verify_password("test", "nocolonseparator") is False

    def test_hash_very_long_password(self):
        """A 10,000-character password should still hash and verify correctly."""
        long_pw = "a" * 10_000
        pw_hash = _hash_password(long_pw)
        parts = pw_hash.split(":")
        assert len(parts) == 2
        assert len(parts[0]) == 32  # salt
        assert len(parts[1]) == 64  # SHA-256 digest
        assert _verify_password(long_pw, pw_hash) is True
        assert _verify_password(long_pw[:-1], pw_hash) is False  # off-by-one


# ---------------------------------------------------------------------------
# JWT edge cases
# ---------------------------------------------------------------------------


class TestJWTEdgeCases:
    """Edge cases for _create_jwt and _decode_jwt."""

    def test_decode_empty_string_returns_none(self):
        """_decode_jwt('') should return None."""
        assert _decode_jwt("") is None

    def test_decode_none_returns_none(self):
        """_decode_jwt(None) should return None (catches TypeError from .split)."""
        assert _decode_jwt(None) is None

    def test_decode_single_part_returns_none(self):
        """A token with no dots should return None."""
        assert _decode_jwt("justonepart") is None

    def test_decode_two_parts_returns_none(self):
        """A token with only two dot-separated parts should return None."""
        assert _decode_jwt("a.b") is None

    def test_expired_token_returns_none(self):
        """A token created far in the past should be expired and return None."""
        with patch("business_brain.action.api.time") as mock_time:
            # Create token as if current time is epoch + 1000 seconds (Jan 1970)
            mock_time.time.return_value = 1000
            token = _create_jwt("u1", "e@x.com", "viewer", "free")

        # Now decode with real time -- the token's exp is 1000 + 7*86400 = 605400
        # which is still in 1970, long before the actual current time.
        assert _decode_jwt(token) is None

    def test_token_at_exact_expiry_boundary(self):
        """A token whose exp == now should return None (strict less-than check).

        The implementation checks: if data['exp'] < int(time.time()) -> None
        So exp == now means exp < now is False, which means the token is
        still valid. But if exp == now exactly and we advance time by 1 second,
        it should become expired.
        """
        creation_time = 1_000_000

        # Create token at a known time
        with patch("business_brain.action.api.time") as mock_time:
            mock_time.time.return_value = creation_time
            token = _create_jwt("u1", "e@x.com", "viewer", "free")

        exp_time = creation_time + (7 * 86400)

        # At exactly exp, exp < exp is False => still valid
        with patch("business_brain.action.api.time") as mock_time:
            mock_time.time.return_value = exp_time
            result = _decode_jwt(token)
            assert result is not None, "Token at exact exp boundary should still be valid"

        # One second after exp, exp < (exp+1) is True => expired
        with patch("business_brain.action.api.time") as mock_time:
            mock_time.time.return_value = exp_time + 1
            result = _decode_jwt(token)
            assert result is None, "Token 1 second past exp should be expired"

    def test_tampered_payload_different_role(self):
        """Changing the role in the payload invalidates the HMAC signature."""
        token = _create_jwt("user-1", "test@example.com", "viewer", "free")
        header, payload, signature = token.split(".")

        # Decode payload, change role, re-encode
        padded_payload = payload + "=" * (4 - len(payload) % 4)
        data = json.loads(base64.urlsafe_b64decode(padded_payload))
        assert data["role"] == "viewer"
        data["role"] = "owner"
        tampered_payload = (
            base64.urlsafe_b64encode(json.dumps(data).encode())
            .decode()
            .rstrip("=")
        )

        tampered_token = f"{header}.{tampered_payload}.{signature}"
        assert _decode_jwt(tampered_token) is None

    def test_tampered_signature_returns_none(self):
        """Modifying the last character of the signature invalidates the token."""
        token = _create_jwt("user-1", "test@example.com", "admin", "pro")
        header, payload, signature = token.split(".")

        # Flip the last character of the hex signature
        last_char = signature[-1]
        replacement = "0" if last_char != "0" else "1"
        tampered_sig = signature[:-1] + replacement

        tampered_token = f"{header}.{payload}.{tampered_sig}"
        assert _decode_jwt(tampered_token) is None

    def test_roundtrip_preserves_all_fields(self):
        """create -> decode roundtrip should preserve sub, email, role, plan."""
        token = _create_jwt(
            user_id="uuid-abc-123",
            email="deep@example.org",
            role="manager",
            plan="enterprise",
        )
        data = _decode_jwt(token)
        assert data is not None
        assert data["sub"] == "uuid-abc-123"
        assert data["email"] == "deep@example.org"
        assert data["role"] == "manager"
        assert data["plan"] == "enterprise"
        assert "iat" in data
        assert "exp" in data
        assert data["exp"] == data["iat"] + (7 * 86400)


# ---------------------------------------------------------------------------
# get_current_user edge cases (async)
# ---------------------------------------------------------------------------


class TestGetCurrentUserEdgeCases:
    """Edge cases for the async get_current_user function."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_none(self):
        """get_current_user('') should return None."""
        result = await get_current_user("")
        assert result is None

    @pytest.mark.asyncio
    async def test_bearer_only_returns_none(self):
        """'Bearer ' with no token after it should produce an empty string
        token, which _decode_jwt returns None for."""
        result = await get_current_user("Bearer ")
        assert result is None

    @pytest.mark.asyncio
    async def test_lowercase_bearer_returns_none(self):
        """'bearer ...' (lowercase) should return None because the code
        checks startswith('Bearer ') with a capital B."""
        token = _create_jwt("u1", "e@x.com", "admin", "pro")
        result = await get_current_user(f"bearer {token}")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_bearer_prefix_returns_none(self):
        """An authorization header without 'Bearer ' prefix returns None."""
        token = _create_jwt("u1", "e@x.com", "admin", "pro")
        # "Token" scheme instead of "Bearer"
        result = await get_current_user(f"Token {token}")
        assert result is None

    @pytest.mark.asyncio
    async def test_valid_bearer_returns_user(self):
        """A properly formatted 'Bearer <valid-token>' should return a
        user dict with all expected fields."""
        token = _create_jwt("uid-77", "valid@test.com", "operator", "growth")
        result = await get_current_user(f"Bearer {token}")
        assert result is not None
        assert result["sub"] == "uid-77"
        assert result["email"] == "valid@test.com"
        assert result["role"] == "operator"
        assert result["plan"] == "growth"


# ---------------------------------------------------------------------------
# require_role edge cases (async)
# ---------------------------------------------------------------------------


class TestRequireRoleEdgeCases:
    """Edge cases for the require_role dependency factory."""

    @pytest.mark.asyncio
    async def test_unknown_role_treated_as_viewer(self):
        """A JWT with an unknown role like 'superadmin' should get ROLE_LEVELS
        default of 0 (viewer level), and thus fail an admin-level check."""
        from fastapi import HTTPException

        token = _create_jwt("u-unknown", "super@admin.com", "superadmin", "free")
        checker = require_role("admin")  # min_level = 3

        with pytest.raises(HTTPException) as exc_info:
            await checker(authorization=f"Bearer {token}")

        assert exc_info.value.status_code == 403
        assert "admin" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_no_auth_raises_401(self):
        """Missing authorization header should raise 401."""
        from fastapi import HTTPException

        checker = require_role("viewer")

        with pytest.raises(HTTPException) as exc_info:
            await checker(authorization="")

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_exact_role_level_passes(self):
        """A user with exactly the required role level should pass."""
        token = _create_jwt("uid-mgr", "mgr@co.com", "manager", "pro")
        checker = require_role("manager")  # min_level = 2
        user = await checker(authorization=f"Bearer {token}")
        assert user["sub"] == "uid-mgr"
        assert user["role"] == "manager"

    @pytest.mark.asyncio
    async def test_higher_role_passes(self):
        """A user with a higher role than required should pass."""
        token = _create_jwt("uid-owner", "boss@co.com", "owner", "enterprise")
        checker = require_role("operator")  # min_level = 1
        user = await checker(authorization=f"Bearer {token}")
        assert user["sub"] == "uid-owner"
        assert user["role"] == "owner"

    @pytest.mark.asyncio
    async def test_lower_role_raises_403(self):
        """A user with a lower role than required should get 403."""
        from fastapi import HTTPException

        token = _create_jwt("uid-view", "view@co.com", "viewer", "free")
        checker = require_role("admin")  # min_level = 3

        with pytest.raises(HTTPException) as exc_info:
            await checker(authorization=f"Bearer {token}")

        assert exc_info.value.status_code == 403
