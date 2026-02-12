"""Generic async REST API fetcher that normalises JSON → SQL rows."""

import logging
from typing import Any

import httpx
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.ingestion.csv_loader import upsert_dataframe

logger = logging.getLogger(__name__)


async def pull_api(
    url: str,
    session: AsyncSession,
    table_name: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    pk_column: str | None = None,
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
    df = pd.json_normalize(rows)

    if df.empty:
        logger.warning("No data returned from %s", url)
        return 0

    return await upsert_dataframe(df, session, table_name, pk_column)
