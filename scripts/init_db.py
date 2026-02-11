"""Create all tables and enable the pgvector extension."""

import asyncio

from sqlalchemy import text

from business_brain.db.connection import engine
from business_brain.db.models import Base


async def init() -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
    print("[init_db] Tables created successfully.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(init())
