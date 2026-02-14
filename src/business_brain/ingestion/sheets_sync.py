"""Google Sheets connector — reads sheet data and syncs to PostgreSQL."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.v3_models import DataChangeLog, DataSource
from business_brain.ingestion.csv_loader import upsert_dataframe

logger = logging.getLogger(__name__)


def _get_gspread_client():
    """Build a gspread client from the service account JSON env var."""
    import gspread
    from config.settings import settings

    if not settings.google_service_account_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON not configured")

    creds_dict = json.loads(settings.google_service_account_json)
    return gspread.service_account_from_dict(creds_dict)


def _extract_sheet_id(url_or_id: str) -> str:
    """Extract the spreadsheet ID from a URL or return as-is if already an ID."""
    if "/" in url_or_id:
        # URL format: https://docs.google.com/spreadsheets/d/SHEET_ID/edit...
        parts = url_or_id.split("/")
        for i, part in enumerate(parts):
            if part == "d" and i + 1 < len(parts):
                return parts[i + 1]
    return url_or_id


async def connect_google_sheet(
    session: AsyncSession,
    sheet_url: str,
    name: str | None = None,
    tab_name: str | None = None,
    table_name: str | None = None,
    sync_frequency_minutes: int = 5,
) -> DataSource:
    """Connect a Google Sheet as a data source and perform initial sync.

    Args:
        session: DB session.
        sheet_url: Google Sheets URL or spreadsheet ID.
        name: Human-readable name for this source.
        tab_name: Specific tab to sync (None = first tab).
        table_name: Target PostgreSQL table name.
        sync_frequency_minutes: How often to poll for changes.

    Returns:
        The created DataSource record.
    """
    sheet_id = _extract_sheet_id(sheet_url)

    gc = _get_gspread_client()
    spreadsheet = gc.open_by_key(sheet_id)

    if tab_name:
        worksheet = spreadsheet.worksheet(tab_name)
    else:
        worksheet = spreadsheet.sheet1
        tab_name = worksheet.title

    if not name:
        name = f"{spreadsheet.title} - {tab_name}"

    if not table_name:
        # Sanitize: lowercase, replace spaces/hyphens with underscores
        table_name = (
            spreadsheet.title.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        import re
        table_name = re.sub(r"[^a-z0-9_]", "", table_name)

    # Read all data
    records = worksheet.get_all_records()
    if not records:
        raise ValueError("Sheet is empty or has no headers")

    df = pd.DataFrame(records)

    # Create fingerprint from column structure
    col_sig = "|".join(sorted(df.columns.tolist()))
    fingerprint = hashlib.sha256(col_sig.encode()).hexdigest()[:16]

    # Upsert into PostgreSQL
    rows_loaded = await upsert_dataframe(df, session, table_name)

    # Create DataSource record
    source = DataSource(
        name=name,
        source_type="google_sheet",
        connection_config={
            "sheet_id": sheet_id,
            "tab_name": tab_name,
            "spreadsheet_title": spreadsheet.title,
        },
        table_name=table_name,
        format_fingerprint=fingerprint,
        sync_frequency_minutes=sync_frequency_minutes,
        last_sync_at=datetime.now(timezone.utc),
        last_sync_status="success",
        rows_total=len(df),
        active=True,
    )
    session.add(source)
    await session.commit()
    await session.refresh(source)

    logger.info(
        "Connected Google Sheet '%s' tab '%s' → table '%s' (%d rows)",
        spreadsheet.title, tab_name, table_name, rows_loaded,
    )
    return source


async def sync_google_sheet(session: AsyncSession, source: DataSource) -> dict[str, Any]:
    """Sync a connected Google Sheet, detecting changes.

    Returns:
        Dict with sync results: rows_added, rows_modified, rows_total.
    """
    config = source.connection_config or {}
    sheet_id = config.get("sheet_id")
    tab_name = config.get("tab_name")

    if not sheet_id:
        raise ValueError("Missing sheet_id in connection config")

    gc = _get_gspread_client()
    spreadsheet = gc.open_by_key(sheet_id)

    if tab_name:
        worksheet = spreadsheet.worksheet(tab_name)
    else:
        worksheet = spreadsheet.sheet1

    records = worksheet.get_all_records()
    if not records:
        source.last_sync_at = datetime.now(timezone.utc)
        source.last_sync_status = "success"
        await session.commit()
        return {"rows_added": 0, "rows_modified": 0, "rows_total": 0}

    new_df = pd.DataFrame(records)

    # Fetch existing data for diff
    import re
    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", source.table_name)
    changes = {"rows_added": 0, "rows_modified": 0, "rows_total": len(new_df)}

    try:
        existing_res = await session.execute(sql_text(f'SELECT * FROM "{safe_table}"'))
        existing_rows = [dict(r._mapping) for r in existing_res.fetchall()]
        old_df = pd.DataFrame(existing_rows) if existing_rows else pd.DataFrame()

        # Detect changes
        if not old_df.empty:
            # Compare row counts
            changes["rows_added"] = max(0, len(new_df) - len(old_df))

            # Simple cell-level diff on overlapping rows
            min_rows = min(len(old_df), len(new_df))
            common_cols = [c for c in new_df.columns if c in old_df.columns]

            for i in range(min_rows):
                for col in common_cols:
                    old_val = str(old_df.iloc[i].get(col, ""))
                    new_val = str(new_df.iloc[i].get(col, ""))
                    if old_val != new_val:
                        changes["rows_modified"] += 1
                        # Log the change
                        change_log = DataChangeLog(
                            data_source_id=source.id,
                            change_type="row_modified",
                            table_name=source.table_name,
                            row_identifier=str(i),
                            column_name=col,
                            old_value=old_val,
                            new_value=new_val,
                        )
                        session.add(change_log)
                        break  # one log per row

            # Log added rows
            for i in range(len(old_df), len(new_df)):
                change_log = DataChangeLog(
                    data_source_id=source.id,
                    change_type="row_added",
                    table_name=source.table_name,
                    row_identifier=str(i),
                )
                session.add(change_log)
        else:
            changes["rows_added"] = len(new_df)

    except Exception:
        logger.exception("Failed to diff existing data for %s", source.table_name)

    # Drop and re-create (simpler than complex diffing for sheets)
    try:
        await session.execute(sql_text(f'DROP TABLE IF EXISTS "{safe_table}"'))
        await session.commit()
    except Exception:
        await session.rollback()

    rows_loaded = await upsert_dataframe(new_df, session, source.table_name)

    # Update source record
    source.last_sync_at = datetime.now(timezone.utc)
    source.last_sync_status = "success"
    source.last_sync_error = None
    source.rows_total = rows_loaded
    await session.commit()

    logger.info(
        "Synced sheet '%s': +%d rows, ~%d modified, %d total",
        source.name, changes["rows_added"], changes["rows_modified"], rows_loaded,
    )
    return changes
