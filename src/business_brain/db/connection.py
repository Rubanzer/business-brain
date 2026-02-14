"""Async SQLAlchemy engine and session factory."""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from config.settings import settings


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


engine = create_async_engine(_normalise_url(settings.database_url), echo=False)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Yield an async database session."""
    async with async_session() as session:
        yield session
