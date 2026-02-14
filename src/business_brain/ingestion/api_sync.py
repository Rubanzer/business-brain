"""API data source sync — fetches JSON from REST APIs with change detection."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import httpx
import pandas as pd
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.v3_models import DataChangeLog, DataSource
from business_brain.ingestion.csv_loader import upsert_dataframe

logger = logging.getLogger(__name__)


async def connect_api_source(
    session: AsyncSession,
    *,
    name: str,
    api_url: str,
    table_name: str,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    response_path: str | None = None,
    sync_frequency_minutes: int = 0,
) -> DataSource:
    """Connect a new REST API data source.

    Args:
        session: DB session.
        name: Human-readable name for this source.
        api_url: The URL to fetch JSON from.
        table_name: Target PostgreSQL table.
        headers: Optional HTTP headers (e.g. auth tokens).
        params: Optional query parameters.
        response_path: JSONPath-like dot path to the data array in the response.
            e.g. "data.results" means resp["data"]["results"]
        sync_frequency_minutes: Auto-sync interval (0 = manual only).

    Returns:
        The created DataSource record.
    """
    # Initial fetch
    rows_data = await _fetch_api_data(api_url, headers, params, response_path)
    df = pd.json_normalize(rows_data)

    if df.empty:
        raise ValueError(f"API returned no data from {api_url}")

    row_count = await upsert_dataframe(df, session, table_name)

    source = DataSource(
        name=name,
        source_type="api",
        connection_config={
            "api_url": api_url,
            "headers": headers or {},
            "params": params or {},
            "response_path": response_path,
        },
        table_name=table_name,
        sync_frequency_minutes=sync_frequency_minutes,
        rows_total=row_count,
        last_sync_at=datetime.now(timezone.utc),
        last_sync_status="success",
        active=True,
    )
    session.add(source)
    await session.commit()
    await session.refresh(source)

    logger.info("Connected API source '%s' → table '%s' (%d rows)", name, table_name, row_count)
    return source


async def sync_api_source(session: AsyncSession, source: DataSource) -> dict:
    """Sync an API data source with change detection.

    Fetches fresh data, compares against existing rows, logs changes,
    and updates the table.

    Returns:
        Dict with sync results.
    """
    config = source.connection_config or {}
    api_url = config.get("api_url")
    headers = config.get("headers", {})
    params = config.get("params", {})
    response_path = config.get("response_path")

    if not api_url:
        raise ValueError("Missing api_url in connection config")

    # Fetch fresh data
    rows_data = await _fetch_api_data(api_url, headers, params, response_path)
    df_new = pd.json_normalize(rows_data)

    if df_new.empty:
        source.last_sync_at = datetime.now(timezone.utc)
        source.last_sync_status = "success"
        source.last_sync_error = None
        await session.commit()
        return {"status": "success", "rows_total": 0, "changes": 0}

    # Get existing data for comparison
    changes_logged = 0
    try:
        changes_logged = await _detect_and_log_changes(session, source, df_new)
    except Exception:
        logger.debug("Change detection failed for %s — will do full replace", source.name)

    # Upsert new data
    row_count = await upsert_dataframe(df_new, session, source.table_name)

    source.last_sync_at = datetime.now(timezone.utc)
    source.last_sync_status = "success"
    source.last_sync_error = None
    source.rows_total = row_count
    await session.commit()

    return {"status": "success", "rows_total": row_count, "changes": changes_logged}


async def _fetch_api_data(
    url: str,
    headers: dict | None = None,
    params: dict | None = None,
    response_path: str | None = None,
) -> list[dict]:
    """Fetch JSON data from an API endpoint.

    Args:
        url: API URL.
        headers: HTTP headers.
        params: Query parameters.
        response_path: Dot-separated path to data array in response.

    Returns:
        List of row dicts.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers or {}, params=params or {})
        resp.raise_for_status()
        data = resp.json()

    # Navigate to the data array using response_path
    if response_path:
        for key in response_path.split("."):
            if isinstance(data, dict):
                data = data.get(key, [])
            else:
                break

    # Ensure we have a list of dicts
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        data = []

    return data


async def _detect_and_log_changes(
    session: AsyncSession,
    source: DataSource,
    df_new: pd.DataFrame,
) -> int:
    """Compare new data against existing table and log changes.

    Returns number of changes logged.
    """
    import re

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", source.table_name)
    columns = list(df_new.columns)
    col_list = ", ".join(f'"{c}"' for c in columns)

    try:
        result = await session.execute(sql_text(f'SELECT {col_list} FROM "{safe_table}"'))
        existing_rows = [dict(r._mapping) for r in result.fetchall()]
    except Exception:
        return 0

    if not existing_rows:
        return 0

    # Hash each row for quick comparison
    def row_hash(row: dict) -> str:
        return hashlib.md5(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()

    existing_hashes = {row_hash(r) for r in existing_rows}
    new_rows = df_new.to_dict(orient="records")

    changes = 0
    now = datetime.now(timezone.utc)

    for row in new_rows:
        row_str = {k: str(v) if v is not None else None for k, v in row.items()}
        h = hashlib.md5(json.dumps(row_str, sort_keys=True, default=str).encode()).hexdigest()
        if h not in existing_hashes:
            # This is a new or modified row
            change = DataChangeLog(
                data_source_id=source.id,
                change_type="row_added",
                table_name=source.table_name,
                row_identifier=h[:16],
                detected_at=now,
            )
            session.add(change)
            changes += 1

    if changes:
        await session.flush()

    return changes
