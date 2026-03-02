"""Data Upload, Metadata, & Table Data Access routes."""

import json
import logging
from io import StringIO
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.action.dependencies import (
    enrich_column_descriptions,
    get_accessible_tables,
    get_current_user,
    run_discovery_background,
)
from business_brain.db.connection import get_session
from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data"])


# ---------------------------------------------------------------------------
# File Upload Routes
# ---------------------------------------------------------------------------


@router.post("/csv")
async def upload_csv(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Upload a CSV or Excel file for quick ingestion into the database."""
    import gzip
    from io import BytesIO

    import pandas as pd
    from business_brain.ingestion.csv_loader import upsert_dataframe

    try:
        contents = await file.read()
        file_name = file.filename or "upload.csv"

        if file_name.endswith(".gz"):
            contents = gzip.decompress(contents)
            file_name = file_name[:-3]

        table_name = file_name.rsplit(".", 1)[0].replace("-", "_").replace(" ", "_")
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "csv"

        if ext in ("xlsx", "xls"):
            df = pd.read_excel(BytesIO(contents))
        else:
            df = pd.read_csv(StringIO(contents.decode("utf-8")))

        rows = await upsert_dataframe(df, session, table_name)

        columns_metadata = [
            {"name": col, "type": str(df[col].dtype)} for col in df.columns
        ]
        col_preview = ", ".join(str(c) for c in df.columns[:5])
        if len(df.columns) > 5:
            col_preview += f" (+{len(df.columns) - 5} more)"
        description = f"Uploaded table '{table_name}' with {len(df.columns)} columns: {col_preview}"

        await metadata_store.upsert(
            session,
            table_name=table_name,
            description=description,
            columns_metadata=columns_metadata,
            uploaded_by=user.get("sub") if user else None,
            uploaded_by_role=user.get("role") if user else None,
        )

        return {"status": "loaded", "table": table_name, "rows": rows}
    except Exception as exc:
        logger.exception("CSV upload failed")
        return {"error": str(exc)}


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Smart file upload — parse, clean, load, and auto-generate metadata."""
    import gzip
    from io import BytesIO

    import pandas as pd
    from business_brain.cognitive.data_engineer_agent import DataEngineerAgent
    from business_brain.ingestion.format_matcher import (
        find_matching_fingerprint,
        register_fingerprint,
    )

    file_bytes = await file.read()
    file_name = file.filename or "upload.csv"

    if file_name.endswith(".gz"):
        file_bytes = gzip.decompress(file_bytes)
        file_name = file_name[:-3]

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    def _read_df_preview(raw: bytes, extension: str) -> pd.DataFrame:
        if extension == "csv":
            return pd.read_csv(StringIO(raw.decode("utf-8")), nrows=0)
        return pd.read_excel(BytesIO(raw), nrows=0)

    def _read_df_full(raw: bytes, extension: str) -> pd.DataFrame:
        if extension == "csv":
            return pd.read_csv(StringIO(raw.decode("utf-8")))
        return pd.read_excel(BytesIO(raw))

    # --- Recurring format detection ---
    try:
        if ext in ("csv", "xlsx", "xls"):
            df_preview = _read_df_preview(file_bytes, ext)
            columns = list(df_preview.columns)

            if columns:
                match = await find_matching_fingerprint(session, columns)
                if match:
                    from business_brain.ingestion.csv_loader import upsert_dataframe

                    df = _read_df_full(file_bytes, ext)

                    if match.column_mapping:
                        rename_map = {}
                        for src_col in df.columns:
                            if src_col in match.column_mapping:
                                rename_map[src_col] = match.column_mapping[src_col]
                        if rename_map:
                            df = df.rename(columns=rename_map)

                    rows = await upsert_dataframe(df, session, match.table_name)
                    match.match_count += 1
                    await session.commit()

                    try:
                        columns_metadata = [
                            {"name": col, "type": str(df[col].dtype)} for col in df.columns
                        ]
                        await metadata_store.upsert(
                            session,
                            table_name=match.table_name,
                            description=f"Recurring upload to '{match.table_name}' ({rows} rows appended)",
                            columns_metadata=columns_metadata,
                            uploaded_by=user.get("sub") if user else None,
                            uploaded_by_role=user.get("role") if user else None,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to update metadata for recurring upload to '%s'",
                            match.table_name,
                            exc_info=True,
                        )

                    background_tasks.add_task(run_discovery_background, f"recurring:{match.table_name}")

                    return {
                        "status": "loaded",
                        "table_name": match.table_name,
                        "rows": rows,
                        "recurring": True,
                        "fingerprint_id": match.id,
                        "message": f"Recognized recurring format — appended {rows} rows to '{match.table_name}'",
                    }
    except Exception:
        logger.debug("Format fingerprint check failed — proceeding with normal upload")

    # --- Normal upload via DataEngineerAgent ---
    try:
        agent = DataEngineerAgent()
        report = await agent.invoke({
            "file_bytes": file_bytes,
            "file_name": file_name,
            "db_session": session,
            "uploaded_by": user.get("sub") if user else None,
            "uploaded_by_role": user.get("role") if user else None,
        })

        table_name = report.get("table_name", "unknown")
        try:
            if ext in ("csv", "xlsx", "xls"):
                df_cols = _read_df_preview(file_bytes, ext)
                columns = list(df_cols.columns)
                if columns:
                    await register_fingerprint(session, columns, table_name)
        except Exception:
            logger.debug("Failed to register fingerprint — non-critical")

        try:
            await enrich_column_descriptions(session, table_name)
        except Exception:
            logger.debug("Column description enrichment failed — non-critical")

        background_tasks.add_task(run_discovery_background, f"upload:{table_name}")

        return report
    except Exception as exc:
        logger.exception("Upload failed")
        return {"error": str(exc)}


@router.post("/context/file")
async def upload_context_file(
    file: UploadFile = File(...), session: AsyncSession = Depends(get_session)
) -> dict:
    """Upload a document (.txt, .md, .pdf) as business context."""
    import gzip

    from business_brain.cognitive.data_engineer_agent import parse_pdf
    from business_brain.ingestion.context_ingestor import chunk_text, ingest_context

    file_bytes = await file.read()
    file_name = file.filename or "context.txt"

    if file_name.endswith(".gz"):
        file_bytes = gzip.decompress(file_bytes)
        file_name = file_name[:-3]

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext == "pdf":
        text_content = parse_pdf(file_bytes)
    elif ext in ("txt", "md"):
        text_content = file_bytes.decode("utf-8")
    else:
        return {"error": f"Unsupported context file type: .{ext}. Use .txt, .md, or .pdf"}

    if not text_content.strip():
        return {"error": "File is empty or could not be parsed"}

    ids = await ingest_context(text_content, session, source=f"file:{file_name}")
    chunks = chunk_text(text_content)
    return {
        "status": "created",
        "ids": ids,
        "chunks": len(ids),
        "source": f"file:{file_name}",
        "text_length": len(text_content),
    }


# ---------------------------------------------------------------------------
# Metadata Routes
# ---------------------------------------------------------------------------


@router.get("/metadata")
async def list_metadata(
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> list[dict]:
    """List table metadata filtered by user's access level."""
    try:
        accessible = await get_accessible_tables(session, user)
        if accessible is None:
            entries = await metadata_store.get_all(session)
        else:
            entries = await metadata_store.get_filtered(session, accessible)
        return [
            {
                "table_name": e.table_name,
                "description": e.description,
                "columns": e.columns_metadata,
                "business_notes": e.business_notes,
            }
            for e in entries
        ]
    except Exception:
        logger.exception("Error listing metadata")
        return []


@router.get("/metadata/{table}")
async def get_table_metadata(
    table: str,
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Get metadata for a single table (access-controlled)."""
    try:
        accessible = await get_accessible_tables(session, user)
        if accessible is not None and table not in accessible:
            return {"error": "Table not found"}

        entry = await metadata_store.get_by_table(session, table)
        if entry is None:
            return {"error": "Table not found"}
        return {
            "table_name": entry.table_name,
            "description": entry.description,
            "columns": entry.columns_metadata,
            "business_notes": entry.business_notes,
        }
    except Exception:
        logger.exception("Error fetching metadata for table: %s", table)
        return {"error": "Failed to fetch table metadata"}


@router.put("/metadata/{table}")
async def update_table_metadata(
    table: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Update user-editable table context: description, business_notes, column descriptions."""
    try:
        accessible = await get_accessible_tables(session, user)
        if accessible is not None and table not in accessible:
            return {"error": "Table not found"}

        entry = await metadata_store.update_context(
            session,
            table_name=table,
            description=body.get("description"),
            business_notes=body.get("business_notes"),
            column_descriptions=body.get("columns"),  # [{name, description}]
        )
        if not entry:
            return {"error": "Table not found"}
        return {
            "status": "updated",
            "table_name": entry.table_name,
            "description": entry.description,
            "business_notes": entry.business_notes,
            "columns": entry.columns_metadata,
        }
    except Exception:
        logger.exception("Error updating metadata for table: %s", table)
        return {"error": "Failed to update table metadata"}


@router.delete("/metadata/{table}")
async def delete_table_metadata(table: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete metadata for a table."""
    try:
        deleted = await metadata_store.delete(session, table)
        if not deleted:
            return {"error": "Table not found"}
        return {"status": "deleted", "table": table}
    except Exception:
        logger.exception("Error deleting metadata for table: %s", table)
        return {"error": "Failed to delete table metadata"}


@router.delete("/tables/{table_name}")
async def drop_table_cascade(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Drop a table and remove all dependent metadata, vectors, profiles, insights, etc."""
    import re
    from sqlalchemy import delete as sa_delete, text as sql_text

    from business_brain.db.discovery_models import (
        DiscoveredRelationship,
        Insight,
        DeployedReport,
        TableProfile,
    )
    from business_brain.db.models import BusinessContext, MetadataEntry
    from business_brain.db.v3_models import (
        DataChangeLog,
        DataSource,
        FormatFingerprint,
        MetricThreshold,
        SanctityIssue,
        SourceMapping,
    )

    safe = re.sub(r"[^a-zA-Z0-9_]", "", table_name)
    if not safe:
        return {"error": "Invalid table name"}

    removed = {}

    # 1. Drop the actual data table
    try:
        await session.execute(sql_text(f'DROP TABLE IF EXISTS "{safe}" CASCADE'))
        removed["data_table"] = True
    except Exception as exc:
        logger.warning("Could not drop table %s: %s", safe, exc)
        removed["data_table"] = False

    # 2. Metadata
    r = await session.execute(sa_delete(MetadataEntry).where(MetadataEntry.table_name == table_name))
    removed["metadata"] = r.rowcount

    # 3. Business context vectors
    r = await session.execute(
        sa_delete(BusinessContext).where(BusinessContext.source.ilike(f"%{safe}%"))
    )
    removed["business_context"] = r.rowcount

    # 4. Table profile
    r = await session.execute(sa_delete(TableProfile).where(TableProfile.table_name == table_name))
    removed["table_profile"] = r.rowcount

    # 5. Relationships
    r = await session.execute(
        sa_delete(DiscoveredRelationship).where(
            (DiscoveredRelationship.table_a == table_name)
            | (DiscoveredRelationship.table_b == table_name)
        )
    )
    removed["relationships"] = r.rowcount

    # 6. Insights + deployed reports
    from sqlalchemy import select as sa_select, cast, String as SAString

    insight_rows = (
        await session.execute(sa_select(Insight.id).where(Insight.source_tables.cast(SAString).contains(table_name)))
    ).scalars().all()
    if insight_rows:
        r = await session.execute(sa_delete(DeployedReport).where(DeployedReport.insight_id.in_(insight_rows)))
        removed["deployed_reports"] = r.rowcount
        r = await session.execute(sa_delete(Insight).where(Insight.id.in_(insight_rows)))
        removed["insights"] = r.rowcount
    else:
        removed["deployed_reports"] = 0
        removed["insights"] = 0

    # 7-12. Other dependent data
    r = await session.execute(sa_delete(DataSource).where(DataSource.table_name == table_name))
    removed["data_sources"] = r.rowcount
    r = await session.execute(sa_delete(DataChangeLog).where(DataChangeLog.table_name == table_name))
    removed["change_log"] = r.rowcount
    r = await session.execute(sa_delete(SanctityIssue).where(SanctityIssue.table_name == table_name))
    removed["sanctity_issues"] = r.rowcount
    r = await session.execute(sa_delete(FormatFingerprint).where(FormatFingerprint.table_name == table_name))
    removed["fingerprints"] = r.rowcount
    r = await session.execute(
        sa_delete(SourceMapping).where(
            (SourceMapping.source_a_table == table_name) | (SourceMapping.source_b_table == table_name)
        )
    )
    removed["source_mappings"] = r.rowcount
    r = await session.execute(sa_delete(MetricThreshold).where(MetricThreshold.table_name == table_name))
    removed["thresholds"] = r.rowcount

    # 13-16. Analysis engine tables (if they exist)
    try:
        from business_brain.analysis.models import (
            AnalysisResult,
            AgentOutput,
            AnalysisDelta,
            AnalysisFeedback,
        )

        # Find all analysis results for this table
        result_ids = (
            await session.execute(
                sa_select(AnalysisResult.id).where(AnalysisResult.table_name == table_name)
            )
        ).scalars().all()

        if result_ids:
            # Clean up dependent rows that reference these results
            r = await session.execute(
                sa_delete(AgentOutput).where(AgentOutput.result_id.in_(result_ids))
            )
            removed["agent_outputs"] = r.rowcount
            r = await session.execute(
                sa_delete(AnalysisDelta).where(AnalysisDelta.result_id.in_(result_ids))
            )
            removed["analysis_deltas"] = r.rowcount
            r = await session.execute(
                sa_delete(AnalysisFeedback).where(AnalysisFeedback.result_id.in_(result_ids))
            )
            removed["analysis_feedback"] = r.rowcount
            # Delete the results themselves
            r = await session.execute(
                sa_delete(AnalysisResult).where(AnalysisResult.table_name == table_name)
            )
            removed["analysis_results"] = r.rowcount
        else:
            removed["analysis_results"] = 0
    except Exception:
        pass  # Analysis module not yet deployed — skip gracefully

    await session.commit()

    return {"status": "deleted", "table": table_name, "removed": removed}


@router.post("/tables/cleanup")
async def cleanup_orphaned_tables(session: AsyncSession = Depends(get_session)) -> dict:
    """Drop PostgreSQL tables that have no metadata entry (orphaned)."""
    from sqlalchemy import text as sql_text

    from business_brain.db.models import Base as ModelsBase
    from business_brain.db.discovery_models import Base as DiscoveryBase
    from business_brain.db.v3_models import Base as V3Base

    system_tables: set[str] = set()
    for base in (ModelsBase, DiscoveryBase, V3Base):
        for table_obj in base.metadata.tables.values():
            system_tables.add(table_obj.name)
    # Analysis engine tables (shared Base, already registered)
    # Explicitly list them in case the import hasn't happened yet
    system_tables.update([
        "analysis_runs", "analysis_results", "agent_outputs",
        "analysis_deltas", "analysis_feedback", "analysis_learning_state",
        "analysis_history_embeddings",
    ])
    system_tables.add("alembic_version")

    try:
        result = await session.execute(
            sql_text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        actual_tables = {row[0] for row in result.fetchall()}

        all_metadata = await metadata_store.get_all(session)
        metadata_tables = {e.table_name for e in all_metadata}

        orphaned = actual_tables - metadata_tables - system_tables
        dropped: list[str] = []
        for tbl in sorted(orphaned):
            try:
                await session.execute(sql_text(f'DROP TABLE IF EXISTS "{tbl}" CASCADE'))
                dropped.append(tbl)
                logger.info("Dropped orphaned table: %s", tbl)
            except Exception:
                logger.warning("Failed to drop orphaned table: %s", tbl)

        stale_removed = await metadata_store.validate_tables(session)

        if dropped or stale_removed:
            await session.commit()

        return {
            "status": "cleanup_complete",
            "orphaned_tables_dropped": dropped,
            "stale_metadata_removed": stale_removed,
            "system_tables_preserved": len(system_tables & actual_tables),
        }
    except Exception as exc:
        logger.exception("Table cleanup failed")
        await session.rollback()
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Data Access Routes
# ---------------------------------------------------------------------------


@router.get("/data/{table}")
async def get_table_data(
    table: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    total_hint: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Paginated read-only access to any uploaded table (access-controlled)."""
    from sqlalchemy import text as sql_text
    import re

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table)
    if not safe_table:
        return {"error": "Invalid table name"}

    accessible = await get_accessible_tables(session, user)
    if accessible is not None and safe_table not in accessible:
        return {"error": f"Access denied to table '{safe_table}'"}

    try:
        # Fast row count: use pg_class estimate for page 1, client cache for page 2+
        total_exact = True
        if total_hint is not None and total_hint > 0 and page > 1:
            total = total_hint
            total_exact = False
        else:
            # Try pg_class estimated count first (instant)
            est_count = 0
            try:
                est_result = await session.execute(sql_text(
                    "SELECT reltuples::bigint FROM pg_class WHERE relname = :tbl"
                ), {"tbl": safe_table})
                est_row = est_result.fetchone()
                if est_row and isinstance(est_row[0], (int, float)) and est_row[0] >= 0:
                    est_count = int(est_row[0])
            except Exception:
                pass  # Fall through to exact count

            if est_count >= 500:
                total = est_count
                total_exact = False
            else:
                # Small table or stale stats — exact count is fast anyway
                count_result = await session.execute(sql_text(f'SELECT COUNT(*) FROM "{safe_table}"'))
                total = count_result.scalar()

        order_clause = ""
        if sort_by:
            safe_col = re.sub(r"[^a-zA-Z0-9_]", "", sort_by)
            direction = "DESC" if sort_dir.lower() == "desc" else "ASC"
            order_clause = f'ORDER BY "{safe_col}" {direction}'

        offset = (max(page, 1) - 1) * page_size
        query = f'SELECT * FROM "{safe_table}" {order_clause} LIMIT :limit OFFSET :offset'
        result = await session.execute(sql_text(query), {"limit": page_size, "offset": offset})
        rows = [dict(row._mapping) for row in result.fetchall()]

        columns = list(rows[0].keys()) if rows else []

        return {
            "rows": rows,
            "total": total,
            "total_exact": total_exact,
            "page": page,
            "page_size": page_size,
            "columns": columns,
        }
    except Exception as exc:
        logger.exception("Failed to fetch table data")
        await session.rollback()
        return {"error": str(exc)}


@router.put("/data/{table}")
async def update_cell(
    table: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update a single cell value in an uploaded table."""
    from sqlalchemy import text as sql_text
    import re

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table)
    row_id = body.get("row_id")
    column = body.get("column")
    value = body.get("value")

    if not safe_table or row_id is None or not column:
        return {"error": "Missing required fields: row_id, column, value"}

    safe_col = re.sub(r"[^a-zA-Z0-9_]", "", column)

    try:
        pk_result = await session.execute(sql_text(
            "SELECT a.attname FROM pg_index i "
            "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
            "WHERE i.indrelid = :tbl::regclass AND i.indisprimary"
        ), {"tbl": safe_table})
        pk_row = pk_result.fetchone()
        pk_col = pk_row[0] if pk_row else None
    except Exception:
        pk_col = None

    if not pk_col:
        entry = await metadata_store.get_by_table(session, safe_table)
        if entry and entry.columns_metadata:
            pk_col = entry.columns_metadata[0]["name"]
        else:
            pk_col = "id"

    safe_pk = re.sub(r"[^a-zA-Z0-9_]", "", pk_col)

    try:
        old_value = None
        try:
            old_result = await session.execute(
                sql_text(f'SELECT "{safe_col}" FROM "{safe_table}" WHERE "{safe_pk}" = :pk'),
                {"pk": row_id},
            )
            old_row = old_result.fetchone()
            if old_row:
                old_value = str(old_row[0]) if old_row[0] is not None else None
        except Exception:
            pass

        query = f'UPDATE "{safe_table}" SET "{safe_col}" = :val WHERE "{safe_pk}" = :pk'
        await session.execute(sql_text(query), {"val": value, "pk": row_id})

        try:
            from business_brain.db.v3_models import DataChangeLog
            change_entry = DataChangeLog(
                change_type="row_modified",
                table_name=safe_table,
                row_identifier=str(row_id),
                column_name=safe_col,
                old_value=old_value,
                new_value=str(value) if value is not None else None,
            )
            session.add(change_entry)
        except Exception:
            logger.debug("Failed to log cell change — non-critical")

        await session.commit()
        return {"status": "updated", "table": safe_table, "row_id": row_id, "column": safe_col}
    except Exception as exc:
        logger.exception("Failed to update cell")
        await session.rollback()
        return {"error": str(exc)}


@router.post("/chart")
async def generate_chart(body: dict) -> dict:
    """Return chart configuration JSON for frontend rendering."""
    rows = body.get("rows", [])
    chart_type = body.get("chart_type", "bar")
    x_column = body.get("x_column")
    y_columns = body.get("y_columns", [])
    title = body.get("title", "Chart")

    if not rows or not x_column or not y_columns:
        return {"error": "Missing required fields: rows, x_column, y_columns"}

    labels = [str(row.get(x_column, "")) for row in rows]

    datasets = []
    colors = ["#6c5ce7", "#00cec9", "#ff6b6b", "#feca57", "#a29bfe", "#55efc4"]
    for i, y_col in enumerate(y_columns):
        data = []
        for row in rows:
            val = row.get(y_col)
            try:
                data.append(float(val) if val is not None else 0)
            except (ValueError, TypeError):
                data.append(0)
        datasets.append({
            "label": y_col,
            "data": data,
            "backgroundColor": colors[i % len(colors)],
            "borderColor": colors[i % len(colors)],
        })

    return {
        "type": chart_type,
        "data": {
            "labels": labels,
            "datasets": datasets,
        },
        "options": {
            "responsive": True,
            "plugins": {
                "title": {"display": True, "text": title},
            },
        },
    }


# ---------------------------------------------------------------------------
# Qualitative Document Upload & Analysis
# ---------------------------------------------------------------------------


@router.post("/upload/qualitative")
async def upload_qualitative(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Upload a qualitative document (PDF, Excel with text, TXT) for analysis."""
    import gzip

    from business_brain.cognitive.data_engineer_agent import parse_pdf
    from business_brain.db.discovery_models import QualitativeDocument

    file_bytes = await file.read()
    file_name = file.filename or "document.pdf"

    if file_name.endswith(".gz"):
        file_bytes = gzip.decompress(file_bytes)
        file_name = file_name[:-3]

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    # Extract text based on file type
    raw_text = ""
    page_count = 0

    if ext == "pdf":
        raw_text = parse_pdf(file_bytes)
        try:
            from PyPDF2 import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(file_bytes))
            page_count = len(reader.pages)
        except Exception:
            page_count = 0
    elif ext in ("xlsx", "xls"):
        import pandas as pd
        from io import BytesIO
        try:
            df = pd.read_excel(BytesIO(file_bytes))
            # Concatenate all text columns
            text_cols = df.select_dtypes(include=["object"]).columns
            text_parts = []
            for col in text_cols:
                values = df[col].dropna().astype(str).tolist()
                if values:
                    text_parts.append(f"=== {col} ===")
                    text_parts.extend(values)
            raw_text = "\n".join(text_parts)
        except Exception as exc:
            return {"error": f"Failed to read Excel file: {exc}"}
    elif ext in ("txt", "md"):
        raw_text = file_bytes.decode("utf-8")
    else:
        return {"error": f"Unsupported file type: .{ext}. Use .pdf, .xlsx, .txt, or .md"}

    if not raw_text.strip():
        return {"error": "File is empty or could not be parsed"}

    word_count = len(raw_text.split())

    # Create document record
    doc = QualitativeDocument(
        file_name=file_name,
        file_type=ext,
        raw_text=raw_text,
        page_count=page_count,
        word_count=word_count,
        uploaded_by=user.get("sub") if user else None,
        status="pending",
    )
    session.add(doc)
    await session.commit()
    await session.refresh(doc)

    # Kick off background analysis
    background_tasks.add_task(_run_qualitative_analysis, doc.id)

    return {
        "status": "uploaded",
        "doc_id": doc.id,
        "file_name": file_name,
        "word_count": word_count,
        "page_count": page_count,
        "message": "Document uploaded. Analysis running in background.",
    }


async def _run_qualitative_analysis(doc_id: str):
    """Background task: run qualitative analysis pipeline on a document."""
    from business_brain.cognitive.qualitative_analyzer import (
        QualitativeAnalyzer,
        find_linked_tables,
    )
    from business_brain.db.connection import get_session as session_factory
    from business_brain.db.discovery_models import QualitativeDocument, Insight

    try:
        async for session in session_factory():
            # Load the document
            from sqlalchemy import select
            result = await session.execute(
                select(QualitativeDocument).where(QualitativeDocument.id == doc_id)
            )
            doc = result.scalar_one_or_none()
            if not doc:
                logger.error("Qualitative doc %s not found", doc_id)
                return

            doc.status = "analyzing"
            await session.commit()

            # Run analysis
            analyzer = QualitativeAnalyzer()
            results = await analyzer.analyze(doc.raw_text or "", doc.file_name)

            # Update document with results
            doc.summary = results.get("summary")
            doc.sentiment = results.get("sentiment")
            doc.themes = results.get("themes")
            doc.key_findings = results.get("key_findings")
            doc.entities = results.get("entities")
            doc.status = "complete"

            # Find linked tables
            try:
                all_metadata = await metadata_store.get_all(session)
                table_names = [m.table_name for m in all_metadata]
                linked = await find_linked_tables(
                    results.get("entities", []),
                    table_names,
                )
                doc.linked_tables = linked
            except Exception:
                logger.debug("Table linking failed — non-critical")

            # Create a qualitative insight for the feed
            try:
                severity = "info"
                sentiment = results.get("sentiment", {})
                if isinstance(sentiment, dict):
                    score = sentiment.get("score", 0)
                    if score < -0.3:
                        severity = "critical"
                    elif score < 0:
                        severity = "warning"

                findings_summary = ""
                findings = results.get("key_findings", [])
                if isinstance(findings, list) and findings:
                    findings_summary = "; ".join(
                        f.get("finding", "") for f in findings[:3] if isinstance(f, dict)
                    )

                insight = Insight(
                    insight_type="qualitative",
                    severity=severity,
                    title=f"Qualitative Analysis: {doc.file_name}",
                    description=doc.summary or findings_summary or f"Analysis of {doc.file_name}",
                    narrative=doc.summary,
                    source_tables=doc.linked_tables or [],
                    source_columns=[],
                    impact_score=min(80, 30 + (doc.word_count or 0) // 100),
                    evidence={
                        "source_doc": doc.id,
                        "sentiment": doc.sentiment,
                        "themes": (doc.themes or [])[:5],
                        "key_findings": (doc.key_findings or [])[:5],
                    },
                    suggested_actions=[
                        f.get("finding", "") for f in (findings[:3] if isinstance(findings, list) else [])
                        if isinstance(f, dict) and f.get("severity") in ("critical", "warning")
                    ] or None,
                    quality_score=50,
                )
                session.add(insight)
                doc.linked_insights = [insight.id]
            except Exception:
                logger.debug("Qualitative insight creation failed — non-critical")

            await session.commit()
            logger.info("Qualitative analysis complete for %s (doc_id=%s)", doc.file_name, doc_id)

    except Exception as exc:
        logger.exception("Qualitative analysis failed for doc %s", doc_id)
        try:
            async for session in session_factory():
                from sqlalchemy import select
                result = await session.execute(
                    select(QualitativeDocument).where(QualitativeDocument.id == doc_id)
                )
                doc = result.scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    doc.error = str(exc)
                    await session.commit()
        except Exception:
            pass


@router.get("/qualitative")
async def list_qualitative_docs(
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """List all qualitative documents."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import QualitativeDocument

    try:
        result = await session.execute(
            select(QualitativeDocument)
            .order_by(QualitativeDocument.uploaded_at.desc())
            .limit(50)
        )
        docs = result.scalars().all()
        return [
            {
                "id": d.id,
                "file_name": d.file_name,
                "file_type": d.file_type,
                "word_count": d.word_count,
                "page_count": d.page_count,
                "status": d.status,
                "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
                "summary": d.summary,
                "sentiment": d.sentiment,
                "themes": d.themes,
                "key_findings": d.key_findings,
                "entities": d.entities,
                "linked_tables": d.linked_tables,
                "linked_insights": d.linked_insights,
                "error": d.error,
            }
            for d in docs
        ]
    except Exception:
        logger.exception("Failed to list qualitative documents")
        return []


@router.get("/qualitative/{doc_id}")
async def get_qualitative_doc(
    doc_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get a single qualitative document with full analysis results."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import QualitativeDocument

    try:
        result = await session.execute(
            select(QualitativeDocument).where(QualitativeDocument.id == doc_id)
        )
        d = result.scalar_one_or_none()
        if not d:
            return {"error": "Document not found"}
        return {
            "id": d.id,
            "file_name": d.file_name,
            "file_type": d.file_type,
            "word_count": d.word_count,
            "page_count": d.page_count,
            "status": d.status,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
            "summary": d.summary,
            "sentiment": d.sentiment,
            "themes": d.themes,
            "key_findings": d.key_findings,
            "entities": d.entities,
            "linked_tables": d.linked_tables,
            "linked_insights": d.linked_insights,
            "error": d.error,
        }
    except Exception:
        logger.exception("Failed to fetch qualitative doc %s", doc_id)
        return {"error": "Failed to fetch document"}


# ---------------------------------------------------------------------------
# Workspace — SQL-only fast query
# ---------------------------------------------------------------------------


@router.post("/workspace/query")
async def workspace_query(
    body: dict,
    session: AsyncSession = Depends(get_session),
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """Execute a natural language query in workspace context.

    Uses SQL agent directly — no CFO, no Python analyst, no deep tier.
    Fast, sub-second responses for iterative data exploration.
    """
    from sqlalchemy import text as sql_text

    question = body.get("question", "").strip()
    tables = body.get("tables", [])
    history = body.get("history", [])

    if not question:
        return {"error": "No question provided"}

    try:
        from business_brain.memory.schema_rag import retrieve_relevant_tables
        from business_brain.cognitive.sql_agent import SQLAgent

        # Get RAG context (schema + relationships + business notes)
        rag_tables, rag_contexts = await retrieve_relevant_tables(
            session, question, top_k=6,
            allowed_tables=tables if tables else None,
        )

        # Build SQL agent state
        agent = SQLAgent()
        state = {
            "question": question,
            "db_session": session,
            "rag_tables": rag_tables,
            "rag_contexts": rag_contexts,
            "chat_history": history,
        }

        result = await agent.invoke(state)
        sql_result = result.get("sql_result", {})

        rows = sql_result.get("rows", [])
        columns = list(rows[0].keys()) if rows else []

        return {
            "query": sql_result.get("query", ""),
            "rows": rows[:500],  # Cap at 500 rows for workspace
            "columns": columns,
            "row_count": len(rows),
            "error": sql_result.get("error"),
        }
    except Exception as exc:
        logger.exception("Workspace query failed")
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@router.get("/export/{table}")
async def export_table(
    table: str,
    format: str = "csv",
    session: AsyncSession = Depends(get_session),
):
    """Export a table's data as CSV or JSON."""
    from fastapi.responses import Response
    from sqlalchemy import text as sql_text
    import re

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table)
    if not safe_table:
        return {"error": "Invalid table name"}

    try:
        result = await session.execute(sql_text(f'SELECT * FROM "{safe_table}" LIMIT 10000'))
        rows = [dict(r._mapping) for r in result.fetchall()]

        if not rows:
            return {"error": "Table is empty or does not exist"}

        if format == "json":
            return Response(
                content=json.dumps(rows, default=str, indent=2),
                media_type="application/json",
                headers={"Content-Disposition": f'attachment; filename="{safe_table}.json"'},
            )

        import csv
        from io import StringIO as _StringIO

        output = _StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_table}.csv"'},
        )
    except Exception as exc:
        logger.exception("Export failed for table %s", table)
        return {"error": str(exc)}
