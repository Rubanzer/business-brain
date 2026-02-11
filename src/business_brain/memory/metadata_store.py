"""CRUD operations for schema metadata entries."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.models import MetadataEntry


async def get_all(session: AsyncSession) -> list[MetadataEntry]:
    result = await session.execute(select(MetadataEntry))
    return list(result.scalars().all())


async def get_by_table(session: AsyncSession, table_name: str) -> MetadataEntry | None:
    result = await session.execute(
        select(MetadataEntry).where(MetadataEntry.table_name == table_name)
    )
    return result.scalar_one_or_none()


async def upsert(
    session: AsyncSession,
    table_name: str,
    description: str,
    columns_metadata: list[dict] | None = None,
) -> MetadataEntry:
    existing = await get_by_table(session, table_name)
    if existing:
        existing.description = description
        existing.columns_metadata = columns_metadata
    else:
        existing = MetadataEntry(
            table_name=table_name,
            description=description,
            columns_metadata=columns_metadata,
        )
        session.add(existing)
    await session.commit()
    await session.refresh(existing)
    return existing


async def delete(session: AsyncSession, table_name: str) -> bool:
    entry = await get_by_table(session, table_name)
    if entry:
        await session.delete(entry)
        await session.commit()
        return True
    return False
