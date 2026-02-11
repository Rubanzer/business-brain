"""Generic async REST API fetcher that normalises JSON → SQL rows."""

from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession


async def pull_api(
    url: str,
    session: AsyncSession,
    table_name: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> int:
    """Fetch JSON from *url*, flatten, and upsert into *table_name*.

    Returns:
        Number of rows written.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(url, headers=headers or {}, params=params or {})
        resp.raise_for_status()
        data = resp.json()

    rows = data if isinstance(data, list) else [data]

    # TODO: flatten nested JSON, create table if needed, upsert rows
    print(f"[api_puller] Fetched {len(rows)} records from {url} → {table_name}")
    return len(rows)
