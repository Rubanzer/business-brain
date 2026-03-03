"""Authentication & user management routes."""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.action.dependencies import (
    ROLE_LEVELS,
    check_rate_limit,
    create_jwt,
    get_current_user,
    hash_password,
    log_audit,
    migrate_password_to_bcrypt,
    require_role,
    verify_password,
)
from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class InviteRequest(BaseModel):
    email: str
    role: str = "viewer"
    plan: str = "free"


class AcceptInviteRequest(BaseModel):
    token: str
    name: str
    password: str


class UpdateRoleRequest(BaseModel):
    role: str


class GoogleCallbackRequest(BaseModel):
    code: str


class TableAccessRequest(BaseModel):
    roles: list[str]  # ["viewer", "operator", "manager"] or [] for unrestricted


# ---------------------------------------------------------------------------
# Auth Routes
# ---------------------------------------------------------------------------


@router.post("/auth/register")
async def register_user(
    req: RegisterRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Register a new user. First user auto-becomes owner."""
    from sqlalchemy import func, select

    from business_brain.db.v3_models import User

    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, "register", max_requests=3, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Too many registration attempts. Try again later.")

    existing = await session.execute(select(User).where(User.email == req.email))
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    count_result = await session.execute(select(func.count()).select_from(User))
    user_count = count_result.scalar() or 0

    user = User(
        email=req.email,
        name=req.name,
        password_hash=hash_password(req.password),
        role="owner" if user_count == 0 else "viewer",
        plan="pro" if user_count == 0 else "free",
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    token = create_jwt(user.id, user.email, user.role, user.plan, company_id=user.company_id)
    return {
        "status": "registered",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "plan": user.plan,
        },
    }


@router.post("/auth/login")
async def login_user(
    req: LoginRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Login and receive a JWT token."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, "login", max_requests=5, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again in a minute.")

    result = await session.execute(select(User).where(User.email == req.email))
    user = result.scalars().first()

    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # Migrate legacy SHA-256 password to bcrypt on successful login
    if user.password_hash and ":" in user.password_hash and not user.password_hash.startswith("$2"):
        await migrate_password_to_bcrypt(user, req.password, session)

    user.last_login_at = datetime.utcnow()
    await log_audit(
        session, "login", user_id=user.id,
        resource_type="user", resource_id=user.id,
        ip_address=client_ip,
    )
    await session.commit()

    token = create_jwt(user.id, user.email, user.role, user.plan, company_id=user.company_id)
    return {
        "status": "logged_in",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "plan": user.plan,
            "upload_count": user.upload_count,
        },
    }


@router.get("/auth/me")
async def get_me(
    user: Optional[dict] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get current authenticated user info."""
    if not user:
        return {"authenticated": False}

    from sqlalchemy import select

    from business_brain.db.v3_models import User

    result = await session.execute(select(User).where(User.id == user["sub"]))
    db_user = result.scalars().first()
    if not db_user:
        return {"authenticated": False}

    return {
        "authenticated": True,
        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.name,
            "role": db_user.role,
            "plan": db_user.plan,
            "upload_count": db_user.upload_count,
            "is_active": db_user.is_active,
            "avatar_url": getattr(db_user, "avatar_url", None),
            "auth_provider": getattr(db_user, "auth_provider", "email"),
        },
    }


@router.post("/auth/invite")
async def create_invite(
    req: InviteRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> dict:
    """Create an invite token for a new user (admin/owner only)."""
    from business_brain.db.v3_models import InviteToken

    token_str = secrets.token_urlsafe(32)
    invite = InviteToken(
        email=req.email,
        role=req.role,
        plan=req.plan,
        token=token_str,
        expires_at=datetime.utcnow() + timedelta(days=7),
        created_by=user.get("sub"),
    )
    session.add(invite)
    await session.commit()

    return {"status": "created", "token": token_str, "email": req.email, "role": req.role}


@router.post("/auth/accept-invite")
async def accept_invite(
    req: AcceptInviteRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Accept an invite and create a new user account."""
    from sqlalchemy import select

    from business_brain.db.v3_models import InviteToken, User

    result = await session.execute(
        select(InviteToken).where(InviteToken.token == req.token, InviteToken.used == False)  # noqa: E712
    )
    invite = result.scalars().first()
    if not invite:
        raise HTTPException(status_code=400, detail="Invalid or expired invite token")

    if invite.expires_at and invite.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invite token has expired")

    existing = await session.execute(select(User).where(User.email == invite.email))
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=invite.email,
        name=req.name,
        password_hash=hash_password(req.password),
        role=invite.role,
        plan=invite.plan,
        company_id=invite.company_id,
        is_active=True,
    )
    session.add(user)

    invite.used = True
    await session.commit()
    await session.refresh(user)

    token = create_jwt(user.id, user.email, user.role, user.plan, company_id=user.company_id)
    return {
        "status": "registered",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "plan": user.plan,
        },
    }


# ---------------------------------------------------------------------------
# User Management Routes
# ---------------------------------------------------------------------------


@router.get("/users")
async def list_users(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> list[dict]:
    """List all users (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    result = await session.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "name": u.name,
            "role": u.role,
            "plan": u.plan,
            "is_active": u.is_active,
            "upload_count": u.upload_count,
            "created_at": str(u.created_at) if u.created_at else None,
            "last_login_at": str(u.last_login_at) if u.last_login_at else None,
        }
        for u in users
    ]


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    req: UpdateRoleRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> dict:
    """Change a user's role (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    if req.role not in ROLE_LEVELS:
        raise HTTPException(status_code=400, detail=f"Invalid role. Valid: {list(ROLE_LEVELS.keys())}")

    result = await session.execute(select(User).where(User.id == user_id))
    target = result.scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if target.role == "owner" and user.get("role") != "owner":
        raise HTTPException(status_code=403, detail="Only the owner can change another owner's role")

    old_role = target.role
    target.role = req.role
    await log_audit(
        session, "role_change", user_id=user.get("sub"),
        resource_type="user", resource_id=user_id,
        details={"old_role": old_role, "new_role": req.role},
    )
    await session.commit()
    return {"status": "updated", "user_id": user_id, "new_role": req.role}


@router.delete("/users/{user_id}")
async def deactivate_user(
    user_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("owner")),
) -> dict:
    """Deactivate a user account (owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    result = await session.execute(select(User).where(User.id == user_id))
    target = result.scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if target.id == user.get("sub"):
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    target.is_active = False
    await session.commit()
    return {"status": "deactivated", "user_id": user_id}


@router.get("/auth/invites")
async def list_invites(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> list[dict]:
    """List all pending (unused) invite tokens (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import InviteToken

    try:
        result = await session.execute(
            select(InviteToken)
            .where(InviteToken.used == False)  # noqa: E712
            .order_by(InviteToken.expires_at.desc())
        )
        invites = list(result.scalars().all())
        return [
            {
                "id": inv.id,
                "email": inv.email,
                "role": inv.role,
                "plan": inv.plan,
                "token": inv.token,
                "expires_at": str(inv.expires_at) if inv.expires_at else None,
                "created_by": inv.created_by,
                "expired": inv.expires_at is not None and inv.expires_at < datetime.utcnow(),
            }
            for inv in invites
        ]
    except Exception:
        logger.exception("Error listing invites")
        return []


@router.delete("/auth/invites/{invite_id}")
async def revoke_invite(
    invite_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> dict:
    """Revoke (delete) a pending invite token (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import InviteToken

    result = await session.execute(select(InviteToken).where(InviteToken.id == invite_id))
    invite = result.scalars().first()
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.used:
        raise HTTPException(status_code=400, detail="Invite already used, cannot revoke")

    await session.delete(invite)
    await session.commit()
    return {"status": "revoked", "invite_id": invite_id}


# ---------------------------------------------------------------------------
# Google OAuth Routes
# ---------------------------------------------------------------------------


@router.get("/auth/google/url")
async def google_auth_url() -> dict:
    """Return Google OAuth consent URL for frontend redirect."""
    from urllib.parse import urlencode

    from config.settings import settings

    if not settings.google_client_id:
        raise HTTPException(status_code=400, detail="Google OAuth not configured")

    params = urlencode({
        "client_id": settings.google_client_id,
        "redirect_uri": settings.google_redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "offline",
        "prompt": "consent",
    })
    return {"url": f"https://accounts.google.com/o/oauth2/v2/auth?{params}"}


@router.post("/auth/google/callback")
async def google_callback(
    req: GoogleCallbackRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Exchange Google auth code for tokens, create/link user account."""
    import base64
    import json as _json

    import httpx
    from sqlalchemy import func, select

    from business_brain.db.v3_models import User
    from config.settings import settings

    if not settings.google_client_id or not settings.google_client_secret:
        raise HTTPException(status_code=400, detail="Google OAuth not configured")

    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip, "google_callback", max_requests=10, window_seconds=60):
        raise HTTPException(status_code=429, detail="Too many attempts. Try again later.")

    # 1. Exchange authorization code for tokens
    async with httpx.AsyncClient(timeout=10) as client:
        token_resp = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": req.code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
    if token_resp.status_code != 200:
        logger.error("Google token exchange failed: %s", token_resp.text)
        raise HTTPException(status_code=400, detail="Google authentication failed")

    token_data = token_resp.json()
    id_token = token_data.get("id_token", "")

    # 2. Decode id_token payload (base64, no crypto — we trust Google's direct response)
    try:
        payload_part = id_token.split(".")[1]
        payload_part += "=" * (4 - len(payload_part) % 4)
        google_user = _json.loads(base64.urlsafe_b64decode(payload_part))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Google token")

    google_id = google_user.get("sub")
    email = google_user.get("email")
    name = google_user.get("name", email.split("@")[0] if email else "User")
    picture = google_user.get("picture")

    if not email:
        raise HTTPException(status_code=400, detail="Google account has no email")

    # 3. Find or create user
    # Try google_id first (returning user)
    result = await session.execute(select(User).where(User.google_id == google_id))
    user = result.scalars().first()

    if user:
        # Returning Google user — update avatar and login time
        user.avatar_url = picture
        user.last_login_at = datetime.utcnow()
        await session.commit()
    else:
        # Try by email (link existing email account to Google)
        result = await session.execute(select(User).where(User.email == email))
        user = result.scalars().first()

        if user:
            # Link Google to existing email account
            user.google_id = google_id
            user.avatar_url = picture
            user.auth_provider = "google"
            user.last_login_at = datetime.utcnow()
            await session.commit()
        else:
            # Brand new user via Google
            count_result = await session.execute(select(func.count()).select_from(User))
            user_count = count_result.scalar() or 0

            user = User(
                email=email,
                name=name,
                password_hash="oauth:google",  # cannot match verify_password()
                role="owner" if user_count == 0 else "viewer",
                plan="pro" if user_count == 0 else "free",
                google_id=google_id,
                avatar_url=picture,
                auth_provider="google",
                is_active=True,
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    token = create_jwt(user.id, user.email, user.role, user.plan, company_id=user.company_id)
    return {
        "status": "logged_in",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "plan": user.plan,
            "avatar_url": user.avatar_url,
            "auth_provider": user.auth_provider,
        },
    }


# ---------------------------------------------------------------------------
# Table-Level Access Control Routes
# ---------------------------------------------------------------------------


@router.get("/tables/access/summary")
async def table_access_summary(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> list[dict]:
    """List all tables with their role assignments (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import TableRoleAccess
    from business_brain.memory import metadata_store

    try:
        entries = await metadata_store.get_all(session)
        all_tables = [e.table_name for e in entries]

        result = await session.execute(select(TableRoleAccess))
        rules = result.scalars().all()

        # Group roles by table
        table_roles: dict[str, list[str]] = {}
        for rule in rules:
            table_roles.setdefault(rule.table_name, []).append(rule.role)

        return [
            {
                "table_name": t,
                "roles": sorted(table_roles.get(t, [])),
                "restricted": t in table_roles,
            }
            for t in all_tables
        ]
    except Exception:
        logger.exception("Error fetching table access summary")
        return []


@router.put("/tables/{table_name}/access")
async def set_table_access(
    table_name: str,
    req: TableAccessRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> dict:
    """Set role access for a table (admin/owner only).

    Pass roles=[] to make the table unrestricted (all roles can see it).
    Pass roles=["viewer", "operator"] to restrict to those roles only.
    admin/owner always have access regardless.
    """
    from sqlalchemy import delete, select

    from business_brain.db.v3_models import AuditLog, TableRoleAccess

    valid_roles = {"viewer", "operator", "manager"}
    for r in req.roles:
        if r not in valid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role '{r}'. Valid: {sorted(valid_roles)}",
            )

    # Delete existing rules for this table
    await session.execute(
        delete(TableRoleAccess).where(TableRoleAccess.table_name == table_name)
    )

    # Insert new rules
    for role in req.roles:
        session.add(TableRoleAccess(
            table_name=table_name,
            role=role,
            created_by=user.get("sub"),
        ))

    # Audit log
    session.add(AuditLog(
        user_id=user.get("sub"),
        action="access_change",
        resource_type="table",
        resource_id=table_name,
        details={"roles": req.roles},
    ))

    await session.commit()
    return {
        "status": "updated",
        "table_name": table_name,
        "roles": req.roles,
        "restricted": len(req.roles) > 0,
    }
