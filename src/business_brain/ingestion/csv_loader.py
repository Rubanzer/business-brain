"""CSV → PostgreSQL upsert logic."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# pandas dtype → PostgreSQL type
_PG_TYPE_MAP: dict[str, str] = {
    "int64": "BIGINT",
    "float64": "DOUBLE PRECISION",
    "bool": "BOOLEAN",
    "datetime64[ns]": "TIMESTAMP",
    "object": "TEXT",
}


def _pg_type(dtype: np.dtype) -> str:
    return _PG_TYPE_MAP.get(str(dtype), "TEXT")


async def _ensure_table(
    session: AsyncSession, table_name: str, df: pd.DataFrame, pk_column: str | None
) -> None:
    """Create the target table if it doesn't already exist.

    If *pk_column* is None, a synthetic ``_row_id SERIAL PRIMARY KEY`` is used.
    """
    col_defs_parts = []
    if pk_column is None:
        col_defs_parts.append('"_row_id" SERIAL PRIMARY KEY')
        for col in df.columns:
            col_defs_parts.append(f'"{col}" {_pg_type(df[col].dtype)}')
    else:
        for col in df.columns:
            col_defs_parts.append(f'"{col}" {_pg_type(df[col].dtype)}')
        col_defs_parts.append(f'PRIMARY KEY ("{pk_column}")')

    ddl = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({", ".join(col_defs_parts)})'
    await session.execute(text(ddl))
    await session.commit()


async def upsert_dataframe(
    df: pd.DataFrame,
    session: AsyncSession,
    table_name: str,
    pk_column: str | None = None,
) -> int:
    """Upsert a DataFrame into *table_name*, creating the table if needed.

    Args:
        df: Data to upsert.
        session: Async SQLAlchemy session.
        table_name: Target table name.
        pk_column: Primary-key column. Defaults to the first column.

    Returns:
        Number of rows upserted.
    """
    if df.empty:
        return 0

    pk_column = pk_column or df.columns[0]

    # Check if the chosen PK column actually has unique values
    if df[pk_column].nunique() < len(df):
        logger.info(
            "Column '%s' is not unique (%d unique / %d rows) — using serial PK for %s",
            pk_column, df[pk_column].nunique(), len(df), table_name,
        )
        use_serial_pk = True
    else:
        use_serial_pk = False

    await _ensure_table(session, table_name, df, None if use_serial_pk else pk_column)

    columns = list(df.columns)
    col_list = ", ".join(f'"{c}"' for c in columns)
    param_list = ", ".join(f":{c}" for c in columns)

    if use_serial_pk:
        stmt = f'INSERT INTO "{table_name}" ({col_list}) VALUES ({param_list})'
    else:
        update_set = ", ".join(
            f'"{c}" = EXCLUDED."{c}"' for c in columns if c != pk_column
        )
        if update_set:
            stmt = (
                f'INSERT INTO "{table_name}" ({col_list}) VALUES ({param_list}) '
                f'ON CONFLICT ("{pk_column}") DO UPDATE SET {update_set}'
            )
        else:
            stmt = (
                f'INSERT INTO "{table_name}" ({col_list}) VALUES ({param_list}) '
                f'ON CONFLICT ("{pk_column}") DO NOTHING'
            )

    # Replace NaN/NaT with None for SQL compatibility
    rows = df.replace({np.nan: None}).to_dict(orient="records")

    # Batch inserts for performance with large files
    batch_size = 200
    for batch_start in range(0, len(rows), batch_size):
        batch = rows[batch_start:batch_start + batch_size]
        await session.execute(text(stmt), batch)

    await session.commit()

    logger.info("Inserted %d rows into %s", len(rows), table_name)
    return len(rows)


async def load_csv(
    path: Path,
    session: AsyncSession,
    table_name: str | None = None,
    pk_column: str | None = None,
) -> int:
    """Read a CSV file and upsert its rows into a PostgreSQL table.

    Args:
        path: Path to the CSV file.
        session: Async SQLAlchemy session.
        table_name: Target table name. Defaults to the file stem.
        pk_column: Primary-key column. Defaults to the first column.

    Returns:
        Number of rows loaded.
    """
    table_name = table_name or path.stem.lower().replace(" ", "_")
    df = pd.read_csv(path)
    return await upsert_dataframe(df, session, table_name, pk_column)
