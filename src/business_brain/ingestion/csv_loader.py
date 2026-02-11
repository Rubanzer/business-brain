"""CSV → PostgreSQL upsert logic."""

from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


async def load_csv(path: Path, session: AsyncSession, table_name: str | None = None) -> int:
    """Read a CSV file and upsert its rows into a PostgreSQL table.

    Args:
        path: Path to the CSV file.
        session: Async SQLAlchemy session.
        table_name: Target table name. Defaults to the file stem.

    Returns:
        Number of rows loaded.
    """
    table_name = table_name or path.stem.lower().replace(" ", "_")
    df = pd.read_csv(path)

    # TODO: implement actual upsert via SQLAlchemy or COPY
    # Placeholder — just counts rows
    print(f"[csv_loader] Loaded {len(df)} rows from {path.name} → {table_name}")
    return len(df)
