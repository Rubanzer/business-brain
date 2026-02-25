"""Shared dependencies for API routers — auth, helpers, constants."""

import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JWT Configuration
# ---------------------------------------------------------------------------

JWT_SECRET = os.environ.get("JWT_SECRET", "")
if not JWT_SECRET:
    JWT_SECRET = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET not set in environment — generated ephemeral secret. "
        "Tokens will NOT survive serverless cold starts. Set JWT_SECRET env var in production."
    )
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_DAYS = 7

# Role hierarchy (higher index = more permissions)
ROLE_LEVELS = {"viewer": 0, "operator": 1, "manager": 2, "admin": 3, "owner": 4}


# ---------------------------------------------------------------------------
# Password Hashing (bcrypt with SHA-256 legacy fallback)
# ---------------------------------------------------------------------------


def hash_password(password: str) -> str:
    """Hash password using bcrypt (secure, GPU-resistant).

    Falls back to SHA-256 with salt if bcrypt is not installed.
    """
    try:
        import bcrypt

        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        logger.warning("bcrypt not installed — falling back to SHA-256 hashing")
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}:{hashed}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash.

    Supports both bcrypt (new) and legacy SHA-256 salt:hash format.
    """
    try:
        if password_hash.startswith("$2b$") or password_hash.startswith("$2a$"):
            import bcrypt

            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

        salt, hashed = password_hash.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


async def migrate_password_to_bcrypt(user, password: str, session) -> None:
    """Re-hash a legacy SHA-256 password with bcrypt on successful login."""
    try:
        import bcrypt

        new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user.password_hash = new_hash
        await session.commit()
        logger.info("Migrated password to bcrypt for user %s", user.email)
    except ImportError:
        pass
    except Exception:
        logger.debug("bcrypt migration failed for user %s — non-critical", user.email, exc_info=True)


# ---------------------------------------------------------------------------
# JWT Token Management
# ---------------------------------------------------------------------------


def create_jwt(user_id: str, email: str, role: str, plan: str) -> str:
    """Create a simple JWT token."""
    import base64
    import json as _json

    header = base64.urlsafe_b64encode(
        _json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).decode().rstrip("=")
    now = int(time.time())
    payload_data = {
        "sub": user_id,
        "email": email,
        "role": role,
        "plan": plan,
        "iat": now,
        "exp": now + (JWT_EXPIRE_DAYS * 86400),
    }
    payload = base64.urlsafe_b64encode(
        _json.dumps(payload_data).encode()
    ).decode().rstrip("=")
    signature = hmac.new(
        JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256
    ).hexdigest()
    return f"{header}.{payload}.{signature}"


def decode_jwt(token: str) -> Optional[dict]:
    """Decode and verify a JWT token."""
    import base64
    import json as _json

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, signature = parts

        expected_sig = hmac.new(
            JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None

        payload += "=" * (4 - len(payload) % 4)
        data = _json.loads(base64.urlsafe_b64decode(payload))

        if data.get("exp", 0) < int(time.time()):
            return None

        return data
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FastAPI Dependencies
# ---------------------------------------------------------------------------


async def get_current_user(authorization: str = Header(default="")) -> Optional[dict]:
    """Extract user from JWT token in Authorization header. Returns None if no auth."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    return decode_jwt(token)


def require_role(min_role: str):
    """Dependency that checks the user has at least the specified role level."""
    min_level = ROLE_LEVELS.get(min_role, 0)

    async def check(authorization: str = Header(default="")) -> dict:
        user = await get_current_user(authorization)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        user_level = ROLE_LEVELS.get(user.get("role", "viewer"), 0)
        if user_level < min_level:
            raise HTTPException(
                status_code=403,
                detail=f"Requires '{min_role}' role or higher. You have '{user.get('role')}'.",
            )
        return user

    return check


# ---------------------------------------------------------------------------
# Access Control Helpers
# ---------------------------------------------------------------------------


async def get_accessible_tables(
    session: AsyncSession, user: Optional[dict] = None
) -> Optional[list]:
    """Return table names the user can access based on role hierarchy.

    - owner/admin or no auth: None (all tables — backward compat)
    - manager: own uploads + uploads by operator/viewer + legacy (no uploader)
    - operator: own uploads + uploads by viewer + legacy
    - viewer: own uploads + legacy (no uploader recorded)
    """
    if user is None:
        return None

    role = user.get("role", "viewer")
    if role in ("owner", "admin"):
        return None

    user_id = user.get("sub")
    user_level = ROLE_LEVELS.get(role, 0)

    try:
        entries = await metadata_store.get_all(session)
    except Exception:
        logger.exception("Failed to fetch metadata for access control")
        return None

    accessible = []
    for entry in entries:
        uploaded_by = getattr(entry, "uploaded_by", None)
        uploaded_role = getattr(entry, "uploaded_by_role", None)

        if uploaded_by is None:
            accessible.append(entry.table_name)
        elif uploaded_by == user_id:
            accessible.append(entry.table_name)
        elif uploaded_role is not None:
            uploader_level = ROLE_LEVELS.get(uploaded_role, 0)
            if uploader_level <= user_level:
                accessible.append(entry.table_name)
    return accessible


async def get_focus_tables(session: AsyncSession) -> Optional[list]:
    """Return list of included table names if focus mode is active, else None."""
    try:
        from sqlalchemy import select

        from business_brain.db.v3_models import FocusScope

        result = await session.execute(
            select(FocusScope.table_name).where(FocusScope.is_included == True)  # noqa: E712
        )
        tables = [row[0] for row in result.fetchall()]
        return tables if tables else None
    except Exception:
        logger.debug("Focus scope query failed, defaulting to all tables")
        await session.rollback()
        return None


async def enrich_column_descriptions(session: AsyncSession, table_name: str) -> None:
    """Auto-generate column descriptions using pattern matching."""
    from business_brain.discovery.data_dictionary import auto_describe_column

    entry = await metadata_store.get_by_table(session, table_name)
    if not entry or not entry.columns_metadata:
        return

    changed = False
    for col in entry.columns_metadata:
        if not col.get("description"):
            desc = auto_describe_column(
                col.get("name", ""),
                col.get("type", ""),
                {},
            )
            if desc:
                col["description"] = desc
                changed = True

    if changed:
        await metadata_store.upsert(
            session,
            table_name=table_name,
            description=entry.description,
            columns_metadata=entry.columns_metadata,
        )


async def run_discovery_background(trigger: str) -> None:
    """Run discovery engine in background with a fresh session."""
    try:
        from business_brain.discovery.engine import run_discovery

        async with async_session() as session:
            await run_discovery(session, trigger=trigger)
    except Exception:
        logger.exception("Background discovery failed for trigger: %s", trigger)


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

_rate_limit_store: dict = {}


def check_rate_limit(
    client_ip: str, endpoint: str, max_requests: int, window_seconds: int
) -> bool:
    """Return True if within rate limits, False if exceeded."""
    now = time.time()
    key = f"{client_ip}:{endpoint}"
    timestamps = _rate_limit_store.get(key, [])
    timestamps = [t for t in timestamps if now - t < window_seconds]
    if len(timestamps) >= max_requests:
        _rate_limit_store[key] = timestamps
        return False
    timestamps.append(now)
    _rate_limit_store[key] = timestamps
    return True


# ---------------------------------------------------------------------------
# Plan Limits
# ---------------------------------------------------------------------------

PLAN_LIMITS = {
    "free": {
        "max_uploads": 3,
        "google_sheets": False,
        "api_connections": False,
        "analyze_per_day": 5,
        "reports": False,
        "alerts": False,
        "setup": False,
        "whatif": False,
        "deploy": False,
        "export": False,
        "max_users": 1,
    },
    "basic": {
        "max_uploads": 10,
        "google_sheets": True,
        "api_connections": True,
        "analyze_per_day": 50,
        "reports": True,
        "alerts": True,
        "setup": True,
        "whatif": True,
        "deploy": True,
        "export": True,
        "max_users": 3,
    },
    "pro": {
        "max_uploads": 999999,
        "google_sheets": True,
        "api_connections": True,
        "analyze_per_day": 999999,
        "reports": True,
        "alerts": True,
        "setup": True,
        "whatif": True,
        "deploy": True,
        "export": True,
        "max_users": 10,
    },
    "enterprise": {
        "max_uploads": 999999,
        "google_sheets": True,
        "api_connections": True,
        "analyze_per_day": 999999,
        "reports": True,
        "alerts": True,
        "setup": True,
        "whatif": True,
        "deploy": True,
        "export": True,
        "max_users": 999999,
    },
}


# Import async_session at module level for background task usage
from business_brain.db.connection import async_session  # noqa: E402
