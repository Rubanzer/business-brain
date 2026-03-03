"""Async SQLAlchemy engine and session factory.

Supports both single-tenant (default) and multi-tenant modes.
When `settings.multi_tenant` is False, everything uses the single `engine`.
When True, tenant databases are routed via `get_tenant_session()`.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings

logger = logging.getLogger(__name__)


def _normalise_url(url: str) -> str:
    """Ensure the URL uses the async psycopg driver and has SSL for cloud DBs."""
    # Normalise scheme for async psycopg
    if url.startswith("postgres://"):
        url = "postgresql+psycopg://" + url[len("postgres://"):]
    elif url.startswith("postgresql://") and "+psycopg" not in url:
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    # Append sslmode=require for cloud databases (non-localhost)
    host = url.split("@")[-1].split("/")[0].split(":")[0] if "@" in url else ""
    if host and host != "localhost" and host != "127.0.0.1" and "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url += sep + "sslmode=require"
    return url


# ---------------------------------------------------------------------------
# Default engine (single-tenant, or master DB in multi-tenant mode)
# ---------------------------------------------------------------------------

engine = create_async_engine(
    _normalise_url(settings.database_url),
    echo=False,
    pool_timeout=5,
    connect_args={"connect_timeout": 5},
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Yield an async database session (default/master DB)."""
    async with async_session() as session:
        yield session


# ---------------------------------------------------------------------------
# Multi-tenant engine pool (only active when settings.multi_tenant=True)
# ---------------------------------------------------------------------------

_tenant_engines: dict[str, AsyncEngine] = {}
_tenant_sessions: dict[str, async_sessionmaker] = {}


def get_tenant_engine(database_url: str) -> AsyncEngine:
    """Get or create an engine for a tenant database URL."""
    if database_url not in _tenant_engines:
        _tenant_engines[database_url] = create_async_engine(
            _normalise_url(database_url),
            echo=False,
            pool_size=5,
            pool_timeout=5,
            connect_args={"connect_timeout": 5},
        )
        _tenant_sessions[database_url] = async_sessionmaker(
            _tenant_engines[database_url],
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("Created tenant engine for: %s", database_url[:40] + "...")
    return _tenant_engines[database_url]


async def get_tenant_session(company_id: str) -> AsyncGenerator[AsyncSession, None]:
    """Look up company's database_url from registry, yield a session for it.

    Only used when multi_tenant=True. Queries the master DB's company_registry
    to resolve the tenant's connection string.
    """
    from sqlalchemy import select

    from business_brain.db.v3_models import CompanyRegistry

    # Query master DB for the company's database URL
    async with async_session() as master_session:
        result = await master_session.execute(
            select(CompanyRegistry.database_url).where(
                CompanyRegistry.id == company_id,
                CompanyRegistry.is_active == True,  # noqa: E712
            )
        )
        row = result.first()

    if not row:
        raise ValueError(f"Company '{company_id}' not found or inactive")

    db_url = row[0]
    get_tenant_engine(db_url)  # ensure engine exists
    session_factory = _tenant_sessions[db_url]

    async with session_factory() as session:
        yield session
