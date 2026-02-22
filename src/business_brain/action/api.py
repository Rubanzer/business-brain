"""FastAPI application with external trigger endpoints."""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
import uuid
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.cognitive.graph import build_graph
from business_brain.db.connection import async_session, engine, get_session
from business_brain.db.models import Base
from business_brain.ingestion.context_ingestor import ingest_context
from business_brain.memory import chat_store, metadata_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Business Brain API", version="3.0.0")

# Background sync task handle
_sync_task: asyncio.Task | None = None


@app.on_event("startup")
async def _ensure_tables():
    """Create missing tables and add any columns missing from existing tables.

    Base.metadata.create_all only creates NEW tables — it never ALTERs existing
    ones to add columns.  We run idempotent ALTER TABLE ADD COLUMN IF NOT EXISTS
    statements so that every deployment picks up schema changes automatically.
    """
    from sqlalchemy import text as sql_text

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        # Migrate columns added to ORM models after initial table creation.
        # Each statement is idempotent — IF NOT EXISTS makes it a no-op when
        # the column already exists.
        _migrations = [
            # business_contexts — versioning / lifecycle columns
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1",
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ",
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS last_validated_at TIMESTAMPTZ",
            # metadata_entries — access control columns
            "ALTER TABLE metadata_entries ADD COLUMN IF NOT EXISTS uploaded_by VARCHAR(36)",
            "ALTER TABLE metadata_entries ADD COLUMN IF NOT EXISTS uploaded_by_role VARCHAR(20)",
            # process_steps — multi-metric support
            "ALTER TABLE process_steps ADD COLUMN IF NOT EXISTS key_metrics JSON",
            "ALTER TABLE process_steps ADD COLUMN IF NOT EXISTS target_ranges JSON",
            # insights — newer feature columns
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS quality_score INTEGER DEFAULT 0",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS narrative TEXT",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS related_insights JSON",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS suggested_actions JSON",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS composite_template VARCHAR(100)",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'new'",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS session_id VARCHAR(64)",
            # deployed_reports — session + lifecycle columns
            "ALTER TABLE deployed_reports ADD COLUMN IF NOT EXISTS session_id VARCHAR(64)",
            "ALTER TABLE deployed_reports ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE deployed_reports ADD COLUMN IF NOT EXISTS refresh_frequency VARCHAR(20) DEFAULT 'manual'",
            # data_sources — access control columns
            "ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS session_id VARCHAR(64)",
            "ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS uploaded_by VARCHAR(36)",
            "ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS uploaded_by_role VARCHAR(20)",
            # metric_thresholds — derived metric columns
            "ALTER TABLE metric_thresholds ADD COLUMN IF NOT EXISTS is_derived BOOLEAN DEFAULT FALSE",
            "ALTER TABLE metric_thresholds ADD COLUMN IF NOT EXISTS formula TEXT",
            "ALTER TABLE metric_thresholds ADD COLUMN IF NOT EXISTS source_columns JSON",
            "ALTER TABLE metric_thresholds ADD COLUMN IF NOT EXISTS auto_linked BOOLEAN DEFAULT FALSE",
            "ALTER TABLE metric_thresholds ADD COLUMN IF NOT EXISTS confidence FLOAT",
        ]

        async with engine.begin() as conn:
            for stmt in _migrations:
                await conn.execute(sql_text(stmt))

        logger.info("Database tables and columns ensured (%d migrations)", len(_migrations))
    except Exception:
        logger.exception("Failed to ensure database schema")


@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception):
    """Catch-all: ensure every unhandled error returns JSON, not raw HTML."""
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again or check server logs."},
    )


@app.on_event("startup")
async def _start_sync_scheduler():
    """Start the background sync loop that polls data sources on schedule."""
    global _sync_task
    _sync_task = asyncio.create_task(_sync_loop())
    logger.info("Background sync scheduler started")


@app.on_event("shutdown")
async def _stop_sync_scheduler():
    """Stop the background sync loop."""
    global _sync_task
    if _sync_task and not _sync_task.done():
        _sync_task.cancel()
        try:
            await _sync_task
        except asyncio.CancelledError:
            pass
    logger.info("Background sync scheduler stopped")


async def _sync_loop():
    """Poll data sources every 60 seconds for any that are due for sync."""
    from business_brain.ingestion.sync_engine import sync_all_due

    # Wait a bit on startup to let DB init finish
    await asyncio.sleep(5)

    while True:
        try:
            async with async_session() as session:
                results = await sync_all_due(session)
                if results:
                    synced = [r for r in results if r.get("status") != "skipped" and "error" not in r]
                    if synced:
                        logger.info("Background sync completed: %d sources synced", len(synced))
                        # Trigger discovery for synced sources
                        for r in synced:
                            try:
                                await _run_discovery_background(f"auto_sync:{r.get('name', 'unknown')}")
                            except Exception:
                                logger.exception("Discovery after auto-sync failed for %s", r.get("name"))
        except Exception:
            logger.exception("Background sync loop error")

        await asyncio.sleep(60)  # Check every 60 seconds

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    parent_finding: Optional[dict] = None


class ContextRequest(BaseModel):
    text: str
    source: str = "api"


@app.get("/health")
async def health() -> dict:
    try:
        from business_brain.cognitive.python_analyst_agent import execute_sandboxed
        test = execute_sandboxed("print('ok')", [])
        sandbox = test["stdout"]
    except Exception as e:
        sandbox = f"ERROR: {e}"
    return {
        "status": "ok",
        "version": "3.0.0",
        "sandbox": sandbox,
    }


@app.post("/analyze")
async def analyze(req: AnalyzeRequest, session: AsyncSession = Depends(get_session)) -> dict:
    """Trigger an analysis run for the given business question."""
    # Resolve or create session_id
    session_id = req.session_id or str(uuid.uuid4())

    # Load recent chat history for context
    chat_history: list[dict] = []
    try:
        messages = await chat_store.get_history(session, session_id, limit=10)
        chat_history = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
    except Exception:
        logger.exception("Failed to load chat history")
        await session.rollback()

    # Resolve focus scope (table filtering)
    focus_tables = await _get_focus_tables(session)

    graph = build_graph()
    invoke_state = {
        "question": req.question,
        "db_session": session,
        "session_id": session_id,
        "chat_history": chat_history,
        "allowed_tables": focus_tables,
    }
    if req.parent_finding:
        invoke_state["parent_finding"] = req.parent_finding
    result = await graph.ainvoke(invoke_state)

    # Save Q+A pair to chat history
    try:
        await chat_store.append(session, session_id, "user", req.question)
        # Build assistant summary from analysis
        summary_parts = []
        analysis = result.get("analysis", {})
        if analysis.get("summary"):
            summary_parts.append(analysis["summary"])
        python_analysis = result.get("python_analysis", {})
        if python_analysis.get("narrative"):
            summary_parts.append(python_analysis["narrative"])
        assistant_content = " ".join(summary_parts) or "Analysis completed."
        # Store metadata about the result for future reference
        result_meta = {}
        sql_result = result.get("sql_result", {})
        if sql_result.get("query"):
            result_meta["sql_query"] = sql_result["query"]
            result_meta["row_count"] = len(sql_result.get("rows", []))
        await chat_store.append(session, session_id, "assistant", assistant_content, result_meta)
    except Exception:
        logger.exception("Failed to save chat history")
        await session.rollback()

    # Strip non-serializable and internal state from response
    result.pop("db_session", None)
    result.pop("chat_history", None)
    result.pop("column_classification", None)  # internal, don't expose
    result.pop("_rag_tables", None)             # internal RAG state
    result.pop("_rag_contexts", None)           # internal RAG state
    result.setdefault("cfo_key_metrics", [])
    result.setdefault("cfo_chart_suggestions", [])
    result["session_id"] = session_id
    return result


@app.post("/context")
async def submit_context(req: ContextRequest, session: AsyncSession = Depends(get_session)) -> dict:
    """Submit natural-language business context for embedding."""
    ids = await ingest_context(req.text, session, source=req.source)
    return {"status": "created", "ids": ids, "chunks": len(ids), "source": req.source}


# ---------------------------------------------------------------------------
# Access Control — JWT Authentication & Role Helpers (moved before endpoints)
# ---------------------------------------------------------------------------

# Simple JWT implementation (no dependency on python-jose)
_JWT_SECRET = secrets.token_hex(32)
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_DAYS = 7

# Role hierarchy (higher index = more permissions)
ROLE_LEVELS = {"viewer": 0, "operator": 1, "manager": 2, "admin": 3, "owner": 4}


async def _get_accessible_tables(
    session: AsyncSession, user: dict | None
) -> list[str] | None:
    """Return table names the user can access based on role hierarchy.

    - owner/admin or no auth: None (all tables — backward compat)
    - manager: own uploads + uploads by operator/viewer + legacy (no uploader)
    - operator: own uploads + uploads by viewer + legacy
    - viewer: own uploads + legacy (no uploader recorded)
    """
    if user is None:
        return None  # no auth → all tables (backward compat)

    role = user.get("role", "viewer")
    if role in ("owner", "admin"):
        return None  # full access

    user_id = user.get("sub")
    user_level = ROLE_LEVELS.get(role, 0)

    try:
        entries = await metadata_store.get_all(session)
    except Exception:
        logger.exception("Failed to fetch metadata for access control")
        return None  # fail-open

    accessible = []
    for entry in entries:
        uploaded_by = getattr(entry, "uploaded_by", None)
        uploaded_role = getattr(entry, "uploaded_by_role", None)

        if uploaded_by is None:
            accessible.append(entry.table_name)  # legacy data, allow
        elif uploaded_by == user_id:
            accessible.append(entry.table_name)  # own upload
        elif uploaded_role is not None:
            uploader_level = ROLE_LEVELS.get(uploaded_role, 0)
            if uploader_level <= user_level:
                accessible.append(entry.table_name)  # equal or lower role
    return accessible


def _hash_password(password: str) -> str:
    """Hash password using SHA-256 with a salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    try:
        salt, hashed = password_hash.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


def _create_jwt(user_id: str, email: str, role: str, plan: str) -> str:
    """Create a simple JWT token."""
    import base64
    import json as _json

    header = base64.urlsafe_b64encode(_json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).decode().rstrip("=")
    now = int(time.time())
    payload_data = {
        "sub": user_id,
        "email": email,
        "role": role,
        "plan": plan,
        "iat": now,
        "exp": now + (_JWT_EXPIRE_DAYS * 86400),
    }
    payload = base64.urlsafe_b64encode(_json.dumps(payload_data).encode()).decode().rstrip("=")
    signature = hmac.new(_JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256).hexdigest()
    return f"{header}.{payload}.{signature}"


def _decode_jwt(token: str) -> dict | None:
    """Decode and verify a JWT token."""
    import base64
    import json as _json

    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        header, payload, signature = parts

        # Verify signature
        expected_sig = hmac.new(
            _JWT_SECRET.encode(), f"{header}.{payload}".encode(), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(signature, expected_sig):
            return None

        # Decode payload (add padding)
        payload += "=" * (4 - len(payload) % 4)
        data = _json.loads(base64.urlsafe_b64decode(payload))

        # Check expiration
        if data.get("exp", 0) < int(time.time()):
            return None

        return data
    except Exception:
        return None


async def get_current_user(authorization: str = Header(default="")) -> dict | None:
    """Extract user from JWT token in Authorization header. Returns None if no auth."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    token = authorization[7:]
    user_data = _decode_jwt(token)
    return user_data


def require_role(min_role: str):
    """Dependency that checks the user has at least the specified role level."""
    min_level = ROLE_LEVELS.get(min_role, 0)

    async def check(authorization: str = Header(default="")) -> dict:
        user = await get_current_user(authorization)
        if user is None:
            raise HTTPException(status_code=401, detail="Authentication required")
        user_level = ROLE_LEVELS.get(user.get("role", "viewer"), 0)
        if user_level < min_level:
            raise HTTPException(
                status_code=403,
                detail=f"Requires '{min_role}' role or higher. You have '{user.get('role')}'.",
            )
        return user

    return check


# ---------------------------------------------------------------------------
# File Upload Endpoints
# ---------------------------------------------------------------------------

@app.post("/csv")
async def upload_csv(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
    user: dict | None = Depends(get_current_user),
) -> dict:
    """Upload a CSV or Excel file for quick ingestion into the database."""
    import gzip
    from io import BytesIO

    import pandas as pd
    from business_brain.ingestion.csv_loader import upsert_dataframe

    try:
        contents = await file.read()
        file_name = file.filename or "upload.csv"

        # Decompress if client sent gzipped file
        if file_name.endswith(".gz"):
            contents = gzip.decompress(contents)
            file_name = file_name[:-3]  # strip .gz suffix

        table_name = file_name.rsplit(".", 1)[0].replace("-", "_").replace(" ", "_")
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "csv"

        if ext in ("xlsx", "xls"):
            df = pd.read_excel(BytesIO(contents))
        else:
            df = pd.read_csv(StringIO(contents.decode("utf-8")))

        rows = await upsert_dataframe(df, session, table_name)

        # Register in metadata store for UI visibility + access control
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


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
    user: dict | None = Depends(get_current_user),
) -> dict:
    """Smart file upload — parse, clean, load, and auto-generate metadata.

    Checks for recurring format fingerprints first. If the uploaded file
    matches a known format, data is routed to the existing table instead
    of creating a new one.
    """
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

    # Decompress if client sent gzipped file
    if file_name.endswith(".gz"):
        file_bytes = gzip.decompress(file_bytes)
        file_name = file_name[:-3]  # strip .gz suffix

    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    def _read_df_preview(raw: bytes, extension: str) -> pd.DataFrame:
        """Read just column headers from uploaded file."""
        if extension == "csv":
            return pd.read_csv(StringIO(raw.decode("utf-8")), nrows=0)
        return pd.read_excel(BytesIO(raw), nrows=0)

    def _read_df_full(raw: bytes, extension: str) -> pd.DataFrame:
        """Read full data from uploaded file."""
        if extension == "csv":
            return pd.read_csv(StringIO(raw.decode("utf-8")))
        return pd.read_excel(BytesIO(raw))

    # --- Recurring format detection ---
    # Try to detect if this file matches a known recurring format
    try:
        if ext in ("csv", "xlsx", "xls"):
            df_preview = _read_df_preview(file_bytes, ext)
            columns = list(df_preview.columns)

            if columns:
                match = await find_matching_fingerprint(session, columns)
                if match:
                    # Recurring upload detected — route to existing table
                    from business_brain.ingestion.csv_loader import upsert_dataframe

                    df = _read_df_full(file_bytes, ext)

                    # Apply column mapping if columns differ
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

                    # Update metadata on recurring upload (preserves original uploader)
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

                    background_tasks.add_task(_run_discovery_background, f"recurring:{match.table_name}")

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

        # Register fingerprint for future recurring detection
        table_name = report.get("table_name", "unknown")
        try:
            if ext in ("csv", "xlsx", "xls"):
                df_cols = _read_df_preview(file_bytes, ext)
                columns = list(df_cols.columns)
                if columns:
                    await register_fingerprint(session, columns, table_name)
        except Exception:
            logger.debug("Failed to register fingerprint — non-critical")

        # Trigger discovery in background after successful upload
        background_tasks.add_task(_run_discovery_background, f"upload:{table_name}")

        return report
    except Exception as exc:
        logger.exception("Upload failed")
        return {"error": str(exc)}


@app.post("/context/file")
async def upload_context_file(
    file: UploadFile = File(...), session: AsyncSession = Depends(get_session)
) -> dict:
    """Upload a document (.txt, .md, .pdf) as business context."""
    import gzip

    from business_brain.cognitive.data_engineer_agent import parse_pdf
    from business_brain.ingestion.context_ingestor import chunk_text

    file_bytes = await file.read()
    file_name = file.filename or "context.txt"

    # Decompress if client sent gzipped file
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


@app.get("/metadata")
async def list_metadata(
    session: AsyncSession = Depends(get_session),
    user: dict | None = Depends(get_current_user),
) -> list[dict]:
    """List table metadata filtered by user's access level."""
    try:
        accessible = await _get_accessible_tables(session, user)
        if accessible is None:
            entries = await metadata_store.get_all(session)
        else:
            entries = await metadata_store.get_filtered(session, accessible)
        return [
            {
                "table_name": e.table_name,
                "description": e.description,
                "columns": e.columns_metadata,
            }
            for e in entries
        ]
    except Exception:
        logger.exception("Error listing metadata")
        return []


@app.get("/metadata/{table}")
async def get_table_metadata(
    table: str,
    session: AsyncSession = Depends(get_session),
    user: dict | None = Depends(get_current_user),
) -> dict:
    """Get metadata for a single table (access-controlled)."""
    try:
        # Check access
        accessible = await _get_accessible_tables(session, user)
        if accessible is not None and table not in accessible:
            return {"error": "Table not found"}

        entry = await metadata_store.get_by_table(session, table)
        if entry is None:
            return {"error": "Table not found"}
        return {
            "table_name": entry.table_name,
            "description": entry.description,
            "columns": entry.columns_metadata,
        }
    except Exception:
        logger.exception("Error fetching metadata for table: %s", table)
        return {"error": "Failed to fetch table metadata"}


@app.delete("/metadata/{table}")
async def delete_table_metadata(table: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete metadata for a table (e.g. after dropping the table)."""
    try:
        deleted = await metadata_store.delete(session, table)
        if not deleted:
            return {"error": "Table not found"}
        return {"status": "deleted", "table": table}
    except Exception:
        logger.exception("Error deleting metadata for table: %s", table)
        return {"error": "Failed to delete table metadata"}


@app.delete("/tables/{table_name}")
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

    # 3. Business context vectors that reference this table
    r = await session.execute(
        sa_delete(BusinessContext).where(BusinessContext.source.ilike(f"%{safe}%"))
    )
    removed["business_context"] = r.rowcount

    # 4. Table profile
    r = await session.execute(sa_delete(TableProfile).where(TableProfile.table_name == table_name))
    removed["table_profile"] = r.rowcount

    # 5. Relationships (either side)
    r = await session.execute(
        sa_delete(DiscoveredRelationship).where(
            (DiscoveredRelationship.table_a == table_name)
            | (DiscoveredRelationship.table_b == table_name)
        )
    )
    removed["relationships"] = r.rowcount

    # 6. Insights referencing this table + their deployed reports
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

    # 7. Data sources
    r = await session.execute(sa_delete(DataSource).where(DataSource.table_name == table_name))
    removed["data_sources"] = r.rowcount

    # 8. Change log
    r = await session.execute(sa_delete(DataChangeLog).where(DataChangeLog.table_name == table_name))
    removed["change_log"] = r.rowcount

    # 9. Sanctity issues
    r = await session.execute(sa_delete(SanctityIssue).where(SanctityIssue.table_name == table_name))
    removed["sanctity_issues"] = r.rowcount

    # 10. Format fingerprints
    r = await session.execute(sa_delete(FormatFingerprint).where(FormatFingerprint.table_name == table_name))
    removed["fingerprints"] = r.rowcount

    # 11. Source mappings (either side)
    r = await session.execute(
        sa_delete(SourceMapping).where(
            (SourceMapping.source_a_table == table_name) | (SourceMapping.source_b_table == table_name)
        )
    )
    removed["source_mappings"] = r.rowcount

    # 12. Metric thresholds
    r = await session.execute(sa_delete(MetricThreshold).where(MetricThreshold.table_name == table_name))
    removed["thresholds"] = r.rowcount

    await session.commit()

    return {"status": "deleted", "table": table_name, "removed": removed}


@app.post("/tables/cleanup")
async def cleanup_orphaned_tables(session: AsyncSession = Depends(get_session)) -> dict:
    """Drop PostgreSQL tables that have no metadata entry (orphaned/unnecessary tables).

    Also cleans up metadata entries for tables that no longer exist in the DB.
    Skips known system tables (ORM-managed models, alembic, pg_*).
    """
    from sqlalchemy import text as sql_text

    from business_brain.db.models import Base as ModelsBase
    from business_brain.db.discovery_models import Base as DiscoveryBase
    from business_brain.db.v3_models import Base as V3Base

    # Collect all ORM-managed table names (system tables we must not drop)
    system_tables: set[str] = set()
    for base in (ModelsBase, DiscoveryBase, V3Base):
        for table_obj in base.metadata.tables.values():
            system_tables.add(table_obj.name)
    system_tables.add("alembic_version")

    try:
        # 1. Get actual PostgreSQL tables in public schema
        result = await session.execute(
            sql_text("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        )
        actual_tables = {row[0] for row in result.fetchall()}

        # 2. Get all metadata entries
        all_metadata = await metadata_store.get_all(session)
        metadata_tables = {e.table_name for e in all_metadata}

        # 3. Find orphaned data tables (exist in DB but no metadata, not system)
        orphaned = actual_tables - metadata_tables - system_tables
        dropped: list[str] = []
        for tbl in sorted(orphaned):
            try:
                await session.execute(sql_text(f'DROP TABLE IF EXISTS "{tbl}" CASCADE'))
                dropped.append(tbl)
                logger.info("Dropped orphaned table: %s", tbl)
            except Exception:
                logger.warning("Failed to drop orphaned table: %s", tbl)

        # 4. Clean up stale metadata (metadata exists but table doesn't)
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


@app.get("/data/{table}")
async def get_table_data(
    table: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    session: AsyncSession = Depends(get_session),
    user: dict | None = Depends(get_current_user),
) -> dict:
    """Paginated read-only access to any uploaded table (access-controlled)."""
    from sqlalchemy import text as sql_text
    import re

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table)
    if not safe_table:
        return {"error": "Invalid table name"}

    # Access control check
    accessible = await _get_accessible_tables(session, user)
    if accessible is not None and safe_table not in accessible:
        return {"error": f"Access denied to table '{safe_table}'"}

    try:
        # Get total count
        count_result = await session.execute(sql_text(f'SELECT COUNT(*) FROM "{safe_table}"'))
        total = count_result.scalar()

        # Build query with optional sorting
        order_clause = ""
        if sort_by:
            safe_col = re.sub(r"[^a-zA-Z0-9_]", "", sort_by)
            direction = "DESC" if sort_dir.lower() == "desc" else "ASC"
            order_clause = f'ORDER BY "{safe_col}" {direction}'

        offset = (max(page, 1) - 1) * page_size
        query = f'SELECT * FROM "{safe_table}" {order_clause} LIMIT :limit OFFSET :offset'
        result = await session.execute(sql_text(query), {"limit": page_size, "offset": offset})
        rows = [dict(row._mapping) for row in result.fetchall()]

        # Get column names
        columns = list(rows[0].keys()) if rows else []

        return {
            "rows": rows,
            "total": total,
            "page": page,
            "page_size": page_size,
            "columns": columns,
        }
    except Exception as exc:
        logger.exception("Failed to fetch table data")
        await session.rollback()
        return {"error": str(exc)}


@app.put("/data/{table}")
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

    # Detect the actual primary key column from the database
    # Tables with non-unique first columns use a synthetic _row_id SERIAL PRIMARY KEY
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
        # Fallback: check metadata, then default to "id"
        entry = await metadata_store.get_by_table(session, safe_table)
        if entry and entry.columns_metadata:
            pk_col = entry.columns_metadata[0]["name"]
        else:
            pk_col = "id"

    safe_pk = re.sub(r"[^a-zA-Z0-9_]", "", pk_col)

    try:
        # Fetch old value for change log
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
            pass  # non-critical — proceed with update even if old value fetch fails

        query = f'UPDATE "{safe_table}" SET "{safe_col}" = :val WHERE "{safe_pk}" = :pk'
        await session.execute(sql_text(query), {"val": value, "pk": row_id})

        # Log the change for audit trail
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


@app.post("/chart")
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
# Background discovery helper
# ---------------------------------------------------------------------------

async def _get_focus_tables(session: AsyncSession) -> list[str] | None:
    """Return list of included table names if focus mode is active, else None.

    If no FocusScope rows exist for the current context, returns None (= all tables).
    """
    try:
        from sqlalchemy import select
        from business_brain.db.v3_models import FocusScope

        result = await session.execute(
            select(FocusScope.table_name).where(FocusScope.is_included == True)  # noqa: E712
        )
        tables = [row[0] for row in result.fetchall()]
        return tables if tables else None  # None means focus mode is off
    except Exception:
        logger.debug("Focus scope query failed, defaulting to all tables")
        return None


async def _run_discovery_background(trigger: str = "manual", table_filter: list[str] | None = None) -> None:
    """Run discovery engine in a background task with its own session."""
    from business_brain.discovery.engine import run_discovery

    try:
        async with async_session() as session:
            await run_discovery(session, trigger=trigger, table_filter=table_filter)
    except Exception:
        logger.exception("Background discovery failed")


# ---------------------------------------------------------------------------
# Discovery & Feed endpoints
# ---------------------------------------------------------------------------


class DeployRequest(BaseModel):
    name: str


class StatusRequest(BaseModel):
    status: str  # seen/dismissed


@app.get("/feed")
async def get_feed(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get ranked insight feed, filtered by focus scope if active."""
    from business_brain.discovery.feed_store import get_feed as _get_feed

    insights = await _get_feed(session)

    # Filter by focus scope — only show insights from focused tables
    focus_tables = await _get_focus_tables(session)
    if focus_tables:
        focus_set = set(focus_tables)
        insights = [
            i for i in insights
            if not i.source_tables or any(t in focus_set for t in (i.source_tables or []))
        ]

    return [
        {
            "id": i.id,
            "insight_type": i.insight_type,
            "severity": i.severity,
            "impact_score": i.impact_score,
            "title": i.title,
            "description": i.description,
            "narrative": i.narrative,
            "source_tables": i.source_tables,
            "source_columns": i.source_columns,
            "evidence": i.evidence,
            "related_insights": i.related_insights,
            "suggested_actions": i.suggested_actions,
            "composite_template": i.composite_template,
            "discovered_at": i.discovered_at.isoformat() if i.discovered_at else None,
            "status": i.status,
        }
        for i in insights
    ]


@app.post("/feed/dismiss-all")
async def dismiss_all_insights(session: AsyncSession = Depends(get_session)) -> dict:
    """Dismiss all active insights."""
    from business_brain.discovery.feed_store import dismiss_all

    count = await dismiss_all(session)
    return {"status": "ok", "dismissed": count}


@app.post("/feed/{insight_id}/status")
async def update_insight_status(
    insight_id: str,
    req: StatusRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update insight status (seen/dismissed)."""
    from business_brain.discovery.feed_store import update_status

    await update_status(session, insight_id, req.status)
    return {"status": "updated", "insight_id": insight_id, "new_status": req.status}


@app.post("/feed/{insight_id}/deploy")
async def deploy_insight_as_report(
    insight_id: str,
    req: DeployRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Deploy an insight as a persistent report."""
    from business_brain.discovery.feed_store import deploy_insight

    try:
        report = await deploy_insight(session, insight_id, req.name)
        return {
            "status": "deployed",
            "report_id": report.id,
            "name": report.name,
            "insight_id": report.insight_id,
        }
    except ValueError as exc:
        return {"error": str(exc)}


@app.get("/reports")
async def list_reports(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all deployed reports."""
    from business_brain.discovery.feed_store import get_reports

    reports = await get_reports(session)
    return [
        {
            "id": r.id,
            "name": r.name,
            "insight_id": r.insight_id,
            "query": r.query,
            "chart_spec": r.chart_spec,
            "last_result": r.last_result,
            "last_run_at": r.last_run_at.isoformat() if r.last_run_at else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "active": r.active,
        }
        for r in reports
    ]


@app.get("/reports/{report_id}")
async def get_report(
    report_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get a single report with latest data."""
    from business_brain.discovery.feed_store import get_report as _get_report

    report = await _get_report(session, report_id)
    if not report:
        return {"error": "Report not found"}
    return {
        "id": report.id,
        "name": report.name,
        "insight_id": report.insight_id,
        "query": report.query,
        "chart_spec": report.chart_spec,
        "last_result": report.last_result,
        "last_run_at": report.last_run_at.isoformat() if report.last_run_at else None,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "active": report.active,
    }


@app.post("/reports/{report_id}/refresh")
async def refresh_report(
    report_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Re-run a report's query and update results."""
    from business_brain.discovery.feed_store import refresh_report as _refresh

    report = await _refresh(session, report_id)
    if not report:
        return {"error": "Report not found"}
    return {
        "status": "refreshed",
        "report_id": report.id,
        "last_run_at": report.last_run_at.isoformat() if report.last_run_at else None,
        "row_count": len(report.last_result) if isinstance(report.last_result, list) else 0,
    }


@app.delete("/reports/{report_id}")
async def delete_report(
    report_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Remove a deployed report."""
    from business_brain.discovery.feed_store import delete_report as _delete

    deleted = await _delete(session, report_id)
    if not deleted:
        return {"error": "Report not found"}
    return {"status": "deleted", "report_id": report_id}


@app.post("/discovery/trigger")
async def trigger_discovery(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Manually trigger a discovery sweep, respecting focus scope."""
    focus_tables = await _get_focus_tables(session)
    background_tasks.add_task(_run_discovery_background, "manual", table_filter=focus_tables)
    msg = "Discovery sweep triggered in background"
    if focus_tables:
        msg += f" (focused on {len(focus_tables)} tables)"
    return {"status": "started", "message": msg}


@app.get("/discovery/status")
async def discovery_status(session: AsyncSession = Depends(get_session)) -> dict:
    """Get the last discovery run info."""
    from business_brain.discovery.feed_store import get_last_run

    run = await get_last_run(session)
    if not run:
        return {"status": "no_runs", "message": "No discovery runs yet"}
    return {
        "id": run.id,
        "status": run.status,
        "trigger": run.trigger,
        "tables_scanned": run.tables_scanned,
        "insights_found": run.insights_found,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
        "error": run.error,
    }


@app.get("/suggestions")
async def get_suggestions(session: AsyncSession = Depends(get_session)) -> dict:
    """Get smart question suggestions based on profiled tables."""
    from sqlalchemy import select

    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.profiler import generate_suggestions

    result = await session.execute(select(TableProfile))
    profiles = list(result.scalars().all())

    if not profiles:
        return {"suggestions": []}

    suggestions = generate_suggestions(profiles)
    return {"suggestions": suggestions}


# ---------------------------------------------------------------------------
# v3: Data Sources
# ---------------------------------------------------------------------------


@app.get("/sources")
async def list_sources(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all connected data sources."""
    from business_brain.ingestion.sync_engine import get_all_sources

    sources = await get_all_sources(session)
    return [
        {
            "id": s.id,
            "name": s.name,
            "source_type": s.source_type,
            "table_name": s.table_name,
            "sync_frequency_minutes": s.sync_frequency_minutes,
            "last_sync_at": s.last_sync_at.isoformat() if s.last_sync_at else None,
            "last_sync_status": s.last_sync_status,
            "last_sync_error": s.last_sync_error,
            "rows_total": s.rows_total,
            "active": s.active,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sources
    ]


class GoogleSheetRequest(BaseModel):
    sheet_url: str
    name: Optional[str] = None
    tab_name: Optional[str] = None
    table_name: Optional[str] = None
    sync_frequency_minutes: int = 5


@app.post("/sources/google-sheet")
async def connect_google_sheet_endpoint(
    req: GoogleSheetRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Connect a Google Sheet as a data source."""
    from business_brain.ingestion.sheets_sync import connect_google_sheet

    try:
        source = await connect_google_sheet(
            session,
            sheet_url=req.sheet_url,
            name=req.name,
            tab_name=req.tab_name,
            table_name=req.table_name,
            sync_frequency_minutes=req.sync_frequency_minutes,
        )
        # Trigger discovery after connecting new source
        background_tasks.add_task(_run_discovery_background, f"sheet:{source.table_name}")
        return {
            "status": "connected",
            "source_id": source.id,
            "table_name": source.table_name,
            "rows": source.rows_total,
        }
    except Exception as exc:
        logger.exception("Failed to connect Google Sheet")
        return {"error": str(exc)}


class ApiSourceRequest(BaseModel):
    name: str
    api_url: str
    table_name: str
    headers: Optional[dict] = None
    params: Optional[dict] = None
    sync_frequency_minutes: int = 0


@app.post("/sources/api")
async def connect_api_source(
    req: ApiSourceRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Connect an API endpoint as a data source."""
    from business_brain.db.v3_models import DataSource
    from business_brain.ingestion.api_puller import pull_api

    try:
        rows = await pull_api(req.api_url, session, req.table_name, headers=req.headers, params=req.params)
        source = DataSource(
            name=req.name,
            source_type="api",
            connection_config={"api_url": req.api_url, "headers": req.headers, "params": req.params},
            table_name=req.table_name,
            sync_frequency_minutes=req.sync_frequency_minutes,
            rows_total=rows,
            active=True,
        )
        from datetime import datetime, timezone
        source.last_sync_at = datetime.now(timezone.utc)
        source.last_sync_status = "success"
        session.add(source)
        await session.commit()
        await session.refresh(source)
        return {"status": "connected", "source_id": source.id, "rows": rows}
    except Exception as exc:
        logger.exception("Failed to connect API source")
        return {"error": str(exc)}


@app.post("/sources/{source_id}/sync")
async def sync_source_endpoint(
    source_id: str,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Manually trigger sync for a data source."""
    from business_brain.ingestion.sync_engine import get_source, sync_source

    source = await get_source(session, source_id)
    if not source:
        return {"error": "Source not found"}

    try:
        result = await sync_source(session, source)
        background_tasks.add_task(_run_discovery_background, f"sync:{source.table_name}")
        return {"status": "synced", **result}
    except Exception as exc:
        logger.exception("Sync failed for source %s", source_id)
        return {"error": str(exc)}


class SourceUpdateRequest(BaseModel):
    name: Optional[str] = None
    sync_frequency_minutes: Optional[int] = None


@app.put("/sources/{source_id}")
async def update_source_endpoint(
    source_id: str,
    req: SourceUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update a data source's configuration."""
    from business_brain.ingestion.sync_engine import update_source

    updates = req.model_dump(exclude_none=True)
    source = await update_source(session, source_id, updates)
    if not source:
        return {"error": "Source not found"}
    return {"status": "updated", "source_id": source.id}


@app.delete("/sources/{source_id}")
async def delete_source_endpoint(
    source_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Disconnect a data source."""
    from business_brain.ingestion.sync_engine import delete_source

    deleted = await delete_source(session, source_id)
    if not deleted:
        return {"error": "Source not found"}
    return {"status": "deleted", "source_id": source_id}


@app.get("/sources/{source_id}/changes")
async def get_source_changes(
    source_id: str,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get recent change log for a data source."""
    from sqlalchemy import select
    from business_brain.db.v3_models import DataChangeLog

    result = await session.execute(
        select(DataChangeLog)
        .where(DataChangeLog.data_source_id == source_id)
        .order_by(DataChangeLog.detected_at.desc())
        .limit(50)
    )
    changes = list(result.scalars().all())
    return [
        {
            "id": c.id,
            "change_type": c.change_type,
            "table_name": c.table_name,
            "row_identifier": c.row_identifier,
            "column_name": c.column_name,
            "old_value": c.old_value,
            "new_value": c.new_value,
            "detected_at": c.detected_at.isoformat() if c.detected_at else None,
        }
        for c in changes
    ]


@app.post("/sources/{source_id}/pause")
async def pause_source_endpoint(source_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Pause auto-sync for a data source."""
    from business_brain.ingestion.sync_engine import pause_source
    ok = await pause_source(session, source_id)
    return {"status": "paused"} if ok else {"error": "Source not found"}


@app.post("/sources/{source_id}/resume")
async def resume_source_endpoint(source_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Resume auto-sync for a data source."""
    from business_brain.ingestion.sync_engine import resume_source
    ok = await resume_source(session, source_id)
    return {"status": "resumed"} if ok else {"error": "Source not found"}


# ---------------------------------------------------------------------------
# v3: Sanctity Engine
# ---------------------------------------------------------------------------


@app.get("/sanctity")
async def get_sanctity_issues(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all open sanctity issues."""
    from business_brain.discovery.sanctity_engine import get_open_issues

    issues = await get_open_issues(session)
    return [
        {
            "id": i.id,
            "table_name": i.table_name,
            "column_name": i.column_name,
            "row_identifier": i.row_identifier,
            "issue_type": i.issue_type,
            "severity": i.severity,
            "description": i.description,
            "current_value": i.current_value,
            "expected_range": i.expected_range,
            "conflicting_source": i.conflicting_source,
            "conflicting_value": i.conflicting_value,
            "detected_at": i.detected_at.isoformat() if i.detected_at else None,
            "resolved": i.resolved,
        }
        for i in issues
    ]


@app.get("/sanctity/summary")
async def get_sanctity_summary(session: AsyncSession = Depends(get_session)) -> dict:
    """Get summary counts of sanctity issues."""
    from business_brain.discovery.sanctity_engine import run_sanctity_check
    return await run_sanctity_check(session)


class ResolveRequest(BaseModel):
    resolved_by: str
    note: Optional[str] = None


@app.post("/sanctity/{issue_id}/resolve")
async def resolve_sanctity_issue(
    issue_id: int,
    req: ResolveRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Mark a sanctity issue as resolved."""
    from business_brain.discovery.sanctity_engine import resolve_issue
    issue = await resolve_issue(session, issue_id, req.resolved_by, req.note)
    if not issue:
        return {"error": "Issue not found"}
    return {"status": "resolved", "issue_id": issue.id}


@app.get("/changes")
async def get_all_changes(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get recent data changes across all sources."""
    from business_brain.discovery.sanctity_engine import get_recent_changes

    changes = await get_recent_changes(session, limit=100)
    return [
        {
            "id": c.id,
            "change_type": c.change_type,
            "table_name": c.table_name,
            "column_name": c.column_name,
            "old_value": c.old_value,
            "new_value": c.new_value,
            "detected_at": c.detected_at.isoformat() if c.detected_at else None,
        }
        for c in changes
    ]


# ---------------------------------------------------------------------------
# v3: Alert System
# ---------------------------------------------------------------------------


class AlertParseRequest(BaseModel):
    text: str


class AlertDeployRequest(BaseModel):
    text: str
    parsed_rule: Optional[dict] = None
    notification_config: Optional[dict] = None


@app.post("/alerts/parse")
async def parse_alert_endpoint(
    req: AlertParseRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Parse natural language into a structured alert rule (without deploying).

    Returns the parsed rule and a human-readable confirmation message
    for the user to review before confirming deployment.
    """
    from business_brain.action.alert_parser import build_confirmation_message, parse_alert_natural_language

    try:
        parsed = await parse_alert_natural_language(session, req.text)
        confirmation = build_confirmation_message(parsed)
        return {
            "status": "parsed",
            "parsed_rule": parsed,
            "confirmation": confirmation,
        }
    except Exception as exc:
        logger.exception("Failed to parse alert")
        return {"error": str(exc)}


@app.post("/alerts/deploy")
async def deploy_alert_endpoint(
    req: AlertDeployRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Deploy an alert rule. If parsed_rule is provided, uses it directly.
    Otherwise, parses from natural language text first."""
    from business_brain.action.alert_parser import build_confirmation_message, parse_alert_natural_language
    from business_brain.action.alert_engine import deploy_alert

    try:
        if req.parsed_rule:
            parsed = req.parsed_rule
        else:
            parsed = await parse_alert_natural_language(session, req.text)

        rule = await deploy_alert(session, parsed, req.text, req.notification_config)
        confirmation = build_confirmation_message(parsed)
        return {
            "status": "deployed",
            "alert_id": rule.id,
            "parsed_rule": parsed,
            "confirmation": confirmation,
        }
    except Exception as exc:
        logger.exception("Failed to deploy alert")
        return {"error": str(exc)}


@app.get("/alerts")
async def list_alerts(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all alert rules."""
    from business_brain.action.alert_engine import get_all_alerts

    rules = await get_all_alerts(session)
    return [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "rule_type": r.rule_type,
            "rule_config": r.rule_config,
            "notification_channel": r.notification_channel,
            "active": r.active,
            "paused_until": r.paused_until.isoformat() if r.paused_until else None,
            "last_triggered_at": r.last_triggered_at.isoformat() if r.last_triggered_at else None,
            "trigger_count": r.trigger_count,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rules
    ]


@app.get("/alerts/{alert_id}")
async def get_alert_detail(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get alert rule details."""
    from business_brain.action.alert_engine import get_alert
    rule = await get_alert(session, alert_id)
    if not rule:
        return {"error": "Alert not found"}
    return {
        "id": rule.id,
        "name": rule.name,
        "description": rule.description,
        "rule_type": rule.rule_type,
        "rule_config": rule.rule_config,
        "notification_channel": rule.notification_channel,
        "message_template": rule.message_template,
        "active": rule.active,
        "paused_until": rule.paused_until.isoformat() if rule.paused_until else None,
        "last_triggered_at": rule.last_triggered_at.isoformat() if rule.last_triggered_at else None,
        "trigger_count": rule.trigger_count,
    }


class AlertUpdateRequest(BaseModel):
    name: Optional[str] = None
    threshold: Optional[float] = None
    notification_channel: Optional[str] = None
    message_template: Optional[str] = None


@app.put("/alerts/{alert_id}")
async def update_alert_endpoint(
    alert_id: str,
    req: AlertUpdateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update an alert rule."""
    from business_brain.action.alert_engine import update_alert
    updates = req.model_dump(exclude_none=True)
    rule = await update_alert(session, alert_id, updates)
    if not rule:
        return {"error": "Alert not found"}
    return {"status": "updated", "alert_id": rule.id}


@app.post("/alerts/{alert_id}/pause")
async def pause_alert_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Pause an alert rule."""
    from business_brain.action.alert_engine import pause_alert
    rule = await pause_alert(session, alert_id)
    return {"status": "paused"} if rule else {"error": "Alert not found"}


@app.post("/alerts/{alert_id}/resume")
async def resume_alert_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Resume a paused alert rule."""
    from business_brain.action.alert_engine import resume_alert
    rule = await resume_alert(session, alert_id)
    return {"status": "resumed"} if rule else {"error": "Alert not found"}


@app.delete("/alerts/{alert_id}")
async def delete_alert_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete an alert rule."""
    from business_brain.action.alert_engine import delete_alert
    deleted = await delete_alert(session, alert_id)
    return {"status": "deleted"} if deleted else {"error": "Alert not found"}


@app.get("/alerts/{alert_id}/events")
async def get_alert_events_endpoint(alert_id: str, session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get trigger history for an alert rule."""
    from business_brain.action.alert_engine import get_alert_events
    events = await get_alert_events(session, alert_id)
    return [
        {
            "id": e.id,
            "triggered_at": e.triggered_at.isoformat() if e.triggered_at else None,
            "trigger_value": e.trigger_value,
            "threshold_value": e.threshold_value,
            "notification_sent": e.notification_sent,
            "notification_error": e.notification_error,
        }
        for e in events
    ]


@app.post("/alerts/evaluate")
async def evaluate_alerts_endpoint(
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Manually trigger evaluation of all alert rules."""
    from business_brain.action.alert_engine import evaluate_all_alerts
    events = await evaluate_all_alerts(session)
    return {"status": "evaluated", "triggered": len(events)}


# ---------------------------------------------------------------------------
# v3: Telegram
# ---------------------------------------------------------------------------


class TelegramRegisterRequest(BaseModel):
    chat_id: str


@app.post("/telegram/register")
async def register_telegram(req: TelegramRegisterRequest) -> dict:
    """Register a Telegram chat ID for receiving alerts."""
    return {"status": "registered", "chat_id": req.chat_id}


@app.get("/telegram/status")
async def telegram_status() -> dict:
    """Check Telegram bot connection status."""
    from business_brain.action.telegram_bot import get_bot_info
    try:
        info = await get_bot_info()
        return {"status": "connected", "bot_username": info.get("username"), "bot_name": info.get("first_name")}
    except Exception as exc:
        return {"status": "disconnected", "error": str(exc)}


# ---------------------------------------------------------------------------
# v3: Pattern Memory
# ---------------------------------------------------------------------------


class PatternCreateRequest(BaseModel):
    name: str
    source_table: str
    conditions: list[dict]
    time_window_minutes: int = 15
    description: str = ""


@app.post("/patterns")
async def create_pattern_endpoint(
    req: PatternCreateRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Create a new pattern from user labeling."""
    from business_brain.discovery.pattern_memory import learn_pattern
    pattern = await learn_pattern(
        session, req.name, req.source_table, req.conditions,
        req.time_window_minutes, req.description,
    )
    return {"status": "created", "pattern_id": pattern.id, "name": pattern.name}


@app.get("/patterns")
async def list_patterns(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all patterns."""
    from business_brain.discovery.pattern_memory import get_all_patterns
    patterns = await get_all_patterns(session)
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "source_tables": p.source_tables,
            "conditions": p.conditions,
            "time_window_minutes": p.time_window_minutes,
            "confidence": p.confidence,
            "match_count": p.match_count,
            "false_positive_count": p.false_positive_count,
            "active": p.active,
            "created_at": p.created_at.isoformat() if p.created_at else None,
            "last_matched_at": p.last_matched_at.isoformat() if p.last_matched_at else None,
        }
        for p in patterns
    ]


@app.get("/patterns/{pattern_id}")
async def get_pattern_detail(pattern_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get pattern details including match history."""
    from business_brain.discovery.pattern_memory import get_pattern, get_pattern_matches
    pattern = await get_pattern(session, pattern_id)
    if not pattern:
        return {"error": "Pattern not found"}
    matches = await get_pattern_matches(session, pattern_id)
    return {
        "id": pattern.id,
        "name": pattern.name,
        "description": pattern.description,
        "source_tables": pattern.source_tables,
        "conditions": pattern.conditions,
        "confidence": pattern.confidence,
        "historical_occurrences": pattern.historical_occurrences,
        "match_count": pattern.match_count,
        "matches": [
            {
                "id": m.id,
                "matched_at": m.matched_at.isoformat() if m.matched_at else None,
                "similarity_score": m.similarity_score,
                "outcome": m.outcome,
            }
            for m in matches
        ],
    }


class PatternFeedbackRequest(BaseModel):
    outcome: str  # confirmed_breakdown / false_positive


@app.post("/patterns/matches/{match_id}/feedback")
async def pattern_feedback_endpoint(
    match_id: int,
    req: PatternFeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Confirm or reject a pattern match."""
    from business_brain.discovery.pattern_memory import confirm_match
    match = await confirm_match(session, match_id, req.outcome)
    if not match:
        return {"error": "Match not found"}
    return {"status": "updated", "outcome": match.outcome}


@app.delete("/patterns/{pattern_id}")
async def delete_pattern_endpoint(pattern_id: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete a pattern."""
    from business_brain.discovery.pattern_memory import delete_pattern
    deleted = await delete_pattern(session, pattern_id)
    return {"status": "deleted"} if deleted else {"error": "Pattern not found"}


# ---------------------------------------------------------------------------
# v3: Company Onboarding
# ---------------------------------------------------------------------------


@app.get("/company")
async def get_company(session: AsyncSession = Depends(get_session)) -> dict:
    """Get company profile."""
    from business_brain.action.onboarding import compute_profile_completeness, get_company_profile
    profile = await get_company_profile(session)
    if not profile:
        return {"exists": False, "completeness": 0}
    return {
        "exists": True,
        "id": profile.id,
        "name": profile.name,
        "industry": profile.industry,
        "products": profile.products,
        "departments": profile.departments,
        "process_flow": profile.process_flow,
        "systems": profile.systems,
        "known_relationships": profile.known_relationships,
        "completeness": compute_profile_completeness(profile),
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


@app.put("/company")
async def update_company(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Update company profile."""
    from business_brain.action.onboarding import save_company_profile
    profile = await save_company_profile(session, body)
    return {"status": "updated", "id": profile.id}


@app.post("/company/onboard")
async def full_onboard(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Submit full onboarding data."""
    from business_brain.action.onboarding import compute_profile_completeness, save_company_profile
    profile = await save_company_profile(session, body)
    return {
        "status": "onboarded",
        "id": profile.id,
        "completeness": compute_profile_completeness(profile),
    }


# ---------------------------------------------------------------------------
# v3: Metric Thresholds
# ---------------------------------------------------------------------------


@app.get("/thresholds")
async def list_thresholds(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all metric thresholds."""
    from business_brain.action.onboarding import get_all_thresholds
    thresholds = await get_all_thresholds(session)
    return [
        {
            "id": t.id,
            "metric_name": t.metric_name,
            "table_name": t.table_name,
            "column_name": t.column_name,
            "unit": t.unit,
            "normal_min": t.normal_min,
            "normal_max": t.normal_max,
            "warning_min": t.warning_min,
            "warning_max": t.warning_max,
            "critical_min": t.critical_min,
            "critical_max": t.critical_max,
        }
        for t in thresholds
    ]


@app.post("/thresholds")
async def create_threshold_endpoint(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Create a metric threshold."""
    from business_brain.action.onboarding import create_threshold
    threshold = await create_threshold(session, body)
    return {"status": "created", "id": threshold.id}


@app.put("/thresholds/{threshold_id}")
async def update_threshold_endpoint(
    threshold_id: int,
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update a metric threshold."""
    from business_brain.action.onboarding import update_threshold
    t = await update_threshold(session, threshold_id, body)
    if not t:
        return {"error": "Threshold not found"}
    return {"status": "updated", "id": t.id}


@app.delete("/thresholds/{threshold_id}")
async def delete_threshold_endpoint(threshold_id: int, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete a metric threshold."""
    from business_brain.action.onboarding import delete_threshold
    deleted = await delete_threshold(session, threshold_id)
    return {"status": "deleted"} if deleted else {"error": "Threshold not found"}


# ---------------------------------------------------------------------------
# v3: Format Detection
# ---------------------------------------------------------------------------


@app.get("/duplicates")
async def detect_duplicates(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Detect potential duplicate data sources."""
    from business_brain.discovery.format_detector import detect_duplicate_sources
    return await detect_duplicate_sources(session)


class ConfirmMappingRequest(BaseModel):
    table_a: str
    table_b: str
    column_mappings: list[dict]
    entity_type: str = ""
    authoritative_source: str = ""


@app.post("/source-mappings")
async def confirm_mapping_endpoint(
    req: ConfirmMappingRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Confirm a source mapping between two tables."""
    from business_brain.discovery.format_detector import confirm_source_mapping
    mapping = await confirm_source_mapping(
        session, req.table_a, req.table_b, req.column_mappings,
        req.entity_type, req.authoritative_source,
    )
    return {"status": "confirmed", "id": mapping.id}


@app.get("/source-mappings")
async def list_mappings(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all confirmed source mappings."""
    from business_brain.discovery.format_detector import get_source_mappings
    mappings = await get_source_mappings(session)
    return [
        {
            "id": m.id,
            "source_a_table": m.source_a_table,
            "source_b_table": m.source_b_table,
            "column_mappings": m.column_mappings,
            "entity_type": m.entity_type,
            "authoritative_source": m.authoritative_source,
            "confirmed_by_user": m.confirmed_by_user,
        }
        for m in mappings
    ]


# ---------------------------------------------------------------------------
# Export endpoints
# ---------------------------------------------------------------------------


@app.get("/export/{table}")
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

        # CSV format
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


@app.get("/feed/export")
async def export_feed(session: AsyncSession = Depends(get_session)):
    """Export the insight feed as JSON."""
    from fastapi.responses import Response
    from business_brain.discovery.feed_store import get_feed as _get_feed

    insights = await _get_feed(session, limit=500)
    data = [
        {
            "id": i.id,
            "insight_type": i.insight_type,
            "severity": i.severity,
            "impact_score": i.impact_score,
            "title": i.title,
            "description": i.description,
            "source_tables": i.source_tables,
            "source_columns": i.source_columns,
            "composite_template": i.composite_template,
            "suggested_actions": i.suggested_actions,
            "discovered_at": i.discovered_at.isoformat() if i.discovered_at else None,
            "status": i.status,
        }
        for i in insights
    ]
    return Response(
        content=json.dumps(data, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="insights_feed.json"'},
    )


# ---------------------------------------------------------------------------
# Sync status
# ---------------------------------------------------------------------------

# Track sync loop health
_last_sync_check: str | None = None
_sync_sources_count: int = 0


@app.get("/reports/{report_id}/export")
async def export_report(
    report_id: str,
    format: str = "json",
    session: AsyncSession = Depends(get_session),
):
    """Export a single deployed report's data as CSV or JSON."""
    from fastapi.responses import Response
    from business_brain.discovery.feed_store import get_report as _get_report

    report = await _get_report(session, report_id)
    if not report:
        return {"error": "Report not found"}

    rows = report.last_result if isinstance(report.last_result, list) else []
    export_data = {
        "report_name": report.name,
        "insight_id": report.insight_id,
        "query": report.query,
        "chart_spec": report.chart_spec,
        "last_run_at": report.last_run_at.isoformat() if report.last_run_at else None,
        "data": rows,
    }

    if format == "csv" and rows:
        import csv
        from io import StringIO as _StringIO

        output = _StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) if v is not None else "" for k, v in row.items()})

        safe_name = report.name.replace(" ", "_").replace("/", "_")[:50]
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.csv"'},
        )

    return Response(
        content=json.dumps(export_data, default=str, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id[:8]}.json"'},
    )


@app.get("/discovery/history")
async def discovery_history(
    limit: int = 10,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get recent discovery run history."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import DiscoveryRun

    result = await session.execute(
        select(DiscoveryRun).order_by(DiscoveryRun.started_at.desc()).limit(limit)
    )
    runs = list(result.scalars().all())
    return [
        {
            "id": r.id,
            "status": r.status,
            "trigger": r.trigger,
            "tables_scanned": r.tables_scanned,
            "insights_found": r.insights_found,
            "started_at": r.started_at.isoformat() if r.started_at else None,
            "completed_at": r.completed_at.isoformat() if r.completed_at else None,
            "duration_seconds": (
                (r.completed_at - r.started_at).total_seconds()
                if r.completed_at and r.started_at else None
            ),
            "error": r.error,
        }
        for r in runs
    ]


@app.get("/alerts/{alert_id}/preview")
async def preview_alert(
    alert_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Dry-run an alert rule — shows what it would match right now without triggering."""
    from business_brain.action.alert_engine import get_alert
    from sqlalchemy import text as sql_text
    import re

    rule = await get_alert(session, alert_id)
    if not rule:
        return {"error": "Alert not found"}

    config = rule.rule_config or {}
    table = config.get("table", "")
    column = config.get("column", "")
    condition = config.get("condition", "")
    threshold = config.get("threshold")

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table)
    safe_col = re.sub(r"[^a-zA-Z0-9_]", "", column)

    if not safe_table or not safe_col:
        return {"error": "Alert rule missing table or column", "rule_config": config}

    try:
        # Get current value
        query = f'SELECT AVG("{safe_col}") as avg_val, MIN("{safe_col}") as min_val, MAX("{safe_col}") as max_val, COUNT(*) as row_count FROM "{safe_table}"'
        result = await session.execute(sql_text(query))
        row = dict(result.fetchone()._mapping)

        # Build WHERE clause for condition matching
        op_map = {
            "greater_than": ">", "less_than": "<",
            "equals": "=", "not_equals": "!=",
        }
        op = op_map.get(condition, ">")
        match_query = f'SELECT COUNT(*) as matches FROM "{safe_table}" WHERE "{safe_col}" {op} :threshold'
        match_result = await session.execute(sql_text(match_query), {"threshold": threshold})
        match_count = match_result.scalar() or 0

        would_trigger = match_count > 0

        return {
            "alert_id": alert_id,
            "alert_name": rule.name,
            "would_trigger": would_trigger,
            "current_stats": {
                "avg": row.get("avg_val"),
                "min": row.get("min_val"),
                "max": row.get("max_val"),
                "row_count": row.get("row_count"),
            },
            "threshold": threshold,
            "condition": condition,
            "matching_rows": match_count,
            "source": f"{safe_table}.{safe_col}",
        }
    except Exception as exc:
        logger.exception("Alert preview failed")
        return {"error": str(exc), "alert_id": alert_id}


@app.get("/data-quality")
async def get_data_quality(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get data quality scores for all profiled tables."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.profiler import compute_data_quality_score

    result = await session.execute(select(TableProfile))
    profiles = list(result.scalars().all())

    scores = []
    for p in profiles:
        quality = compute_data_quality_score(p)
        scores.append({
            "table_name": p.table_name,
            "row_count": p.row_count,
            "domain_hint": p.domain_hint,
            "score": quality["score"],
            "breakdown": quality["breakdown"],
            "issues": quality["issues"],
            "profiled_at": p.profiled_at.isoformat() if p.profiled_at else None,
        })

    return sorted(scores, key=lambda s: s["score"])


@app.get("/sync/status")
async def get_sync_status(session: AsyncSession = Depends(get_session)) -> dict:
    """Get background sync loop status."""
    from sqlalchemy import select, func
    from business_brain.db.v3_models import DataSource

    # Count active sources
    result = await session.execute(
        select(func.count()).select_from(DataSource).where(
            DataSource.active == True,  # noqa: E712
            DataSource.sync_frequency_minutes > 0,
        )
    )
    active_count = result.scalar() or 0

    loop_running = _sync_task is not None and not _sync_task.done() if _sync_task else False

    return {
        "loop_running": loop_running,
        "active_sources": active_count,
        "poll_interval_seconds": 60,
    }


@app.get("/tables/{table_name}/columns")
async def get_table_columns(
    table_name: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get column-level stats for a profiled table."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile

    result = await session.execute(
        select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"Table '{table_name}' not profiled"})

    cls = profile.column_classification or {}
    columns_info = cls.get("columns", {})

    columns = []
    for col_name, info in columns_info.items():
        columns.append({
            "name": col_name,
            "semantic_type": info.get("semantic_type"),
            "cardinality": info.get("cardinality"),
            "sample_values": info.get("sample_values", [])[:5],
            **{k: v for k, v in info.items() if k not in ("semantic_type", "cardinality", "sample_values")},
        })

    return {
        "table_name": table_name,
        "row_count": profile.row_count,
        "domain_hint": profile.domain_hint or cls.get("domain_hint"),
        "profiled_at": profile.profiled_at.isoformat() if profile.profiled_at else None,
        "column_count": len(columns),
        "columns": columns,
    }


@app.post("/tables/{table_name}/time-analysis")
async def time_analysis(
    table_name: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Run time intelligence analysis on a table's numeric series data.

    Returns trend direction, period-over-period changes, and changepoints
    for each numeric column that has a temporal companion.
    """
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.time_intelligence import (
        compute_period_change,
        detect_changepoints,
        detect_trend,
        find_min_max_periods,
    )

    result = await session.execute(
        select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"Table '{table_name}' not profiled"})

    cls = profile.column_classification or {}
    columns = cls.get("columns", {})

    temp_cols = [c for c, i in columns.items() if i.get("semantic_type") == "temporal"]
    num_cols = [
        c for c, i in columns.items()
        if i.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ]

    if not temp_cols or not num_cols:
        return {"table_name": table_name, "analysis": [], "message": "No temporal+numeric column pairs found"}

    # For each numeric column, compute trend from sample values
    analyses = []
    for col_name in num_cols[:5]:
        info = columns.get(col_name, {})
        samples = info.get("sample_values", [])

        # Convert to floats
        values = []
        for s in samples:
            try:
                values.append(float(str(s).replace(",", "")))
            except (ValueError, TypeError):
                pass

        if len(values) < 2:
            continue

        trend = detect_trend(values)
        min_max = find_min_max_periods(values)
        changepoints = detect_changepoints(values)

        analysis = {
            "column": col_name,
            "sample_count": len(values),
            "trend": {
                "direction": trend.direction,
                "magnitude_pct_per_period": trend.magnitude,
                "r_squared": trend.r_squared,
            },
            "changepoints": changepoints,
        }

        if min_max:
            analysis["min_max"] = {
                "max_value": min_max.max_value,
                "max_index": min_max.max_index,
                "min_value": min_max.min_value,
                "min_index": min_max.min_index,
            }

        if len(values) >= 2:
            pop = compute_period_change(values[-1], values[-2])
            analysis["latest_period_change"] = {
                "current": pop.current,
                "previous": pop.previous,
                "absolute_change": pop.absolute_change,
                "pct_change": pop.pct_change,
            }

        analyses.append(analysis)

    return {
        "table_name": table_name,
        "temporal_columns": temp_cols,
        "analysis": analyses,
    }


@app.post("/tables/{table_name}/forecast")
async def forecast_table(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Forecast numeric columns using linear and exponential smoothing."""
    from sqlalchemy import select as sa_select, text as sql_text
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.time_intelligence import forecast_exponential, forecast_linear

    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile or not profile.column_classification:
        return {"error": f"No profile for table '{table_name}'"}

    cls = profile.column_classification
    if "columns" not in cls:
        return {"error": "No column data"}

    cols = cls["columns"]
    num_cols = [
        c for c, info in cols.items()
        if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ][:5]

    if not num_cols:
        return {"error": "No numeric columns found"}

    forecasts = []
    for col_name in num_cols:
        try:
            query = f'SELECT "{col_name}" FROM "{table_name}" ORDER BY ctid LIMIT 200'
            rows = await session.execute(sql_text(query))
            values = []
            for row in rows.fetchall():
                try:
                    values.append(float(str(row[0]).replace(",", "")))
                except (ValueError, TypeError):
                    pass

            if len(values) < 3:
                continue

            linear = forecast_linear(values, 5)
            exponential = forecast_exponential(values, 5)

            forecasts.append({
                "column": col_name,
                "data_points": len(values),
                "linear": {
                    "predicted": linear.predicted_values,
                    "confidence": linear.confidence,
                },
                "exponential": {
                    "predicted": exponential.predicted_values,
                    "confidence": exponential.confidence,
                },
            })
        except Exception:
            logger.exception("Forecast failed for %s.%s", table_name, col_name)

    return {"table_name": table_name, "forecasts": forecasts}


@app.post("/tables/{table_name}/correlations")
async def compute_correlations(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Compute pairwise correlations between numeric columns in a table."""
    from sqlalchemy import select as sa_select, text as sql_text
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.correlation_engine import (
        compute_correlation_matrix,
        correlation_summary,
        find_strong_correlations,
        find_surprising_correlations,
    )

    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile or not profile.column_classification:
        return {"error": f"No profile for table '{table_name}'"}

    cls = profile.column_classification
    if "columns" not in cls:
        return {"error": "No column data"}

    cols = cls["columns"]
    num_cols = [
        c for c, info in cols.items()
        if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ][:10]

    if len(num_cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis"}

    # Fetch data for all numeric columns
    col_list = ", ".join(f'"{c}"' for c in num_cols)
    try:
        rows = await session.execute(sql_text(f'SELECT {col_list} FROM "{table_name}" LIMIT 500'))
        all_rows = rows.fetchall()
    except Exception:
        return {"error": "Failed to query table data"}

    data: dict[str, list[float]] = {c: [] for c in num_cols}
    for row in all_rows:
        for i, col in enumerate(num_cols):
            try:
                data[col].append(float(str(row[i]).replace(",", "")))
            except (ValueError, TypeError):
                data[col].append(float("nan"))

    # Remove nan rows (paired deletion)
    clean_data: dict[str, list[float]] = {c: [] for c in num_cols}
    for idx in range(len(all_rows)):
        if all(not (data[c][idx] != data[c][idx]) for c in num_cols):  # nan check
            for c in num_cols:
                clean_data[c].append(data[c][idx])

    pairs = compute_correlation_matrix(clean_data)
    strong = find_strong_correlations(pairs)
    surprising = find_surprising_correlations(pairs)
    summary = correlation_summary(pairs)

    return {
        "table_name": table_name,
        "columns_analyzed": num_cols,
        "summary": summary,
        "strong_correlations": [
            {"columns": [p.column_a, p.column_b], "correlation": p.correlation, "strength": p.strength, "direction": p.direction}
            for p in strong
        ],
        "surprising_correlations": [
            {"columns": [p.column_a, p.column_b], "correlation": p.correlation}
            for p in surprising
        ],
        "all_pairs": [
            {"columns": [p.column_a, p.column_b], "correlation": p.correlation, "strength": p.strength, "direction": p.direction, "sample_size": p.sample_size}
            for p in pairs
        ],
    }


@app.post("/tables/{table_name}/distribution")
async def get_distribution(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get distribution profile for numeric columns in a table."""
    from sqlalchemy import select as sa_select, text as sql_text
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.distribution_profiler import profile_distribution

    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile or not profile.column_classification:
        return {"error": f"No profile for table '{table_name}'"}

    cls = profile.column_classification
    if "columns" not in cls:
        return {"error": "No column data"}

    cols = cls["columns"]
    num_cols = [
        c for c, info in cols.items()
        if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ][:8]

    if not num_cols:
        return {"error": "No numeric columns found"}

    distributions = []
    for col_name in num_cols:
        try:
            query = f'SELECT "{col_name}" FROM "{table_name}" LIMIT 500'
            rows = await session.execute(sql_text(query))
            values = []
            for row in rows.fetchall():
                try:
                    values.append(float(str(row[0]).replace(",", "")))
                except (ValueError, TypeError):
                    pass

            dp = profile_distribution(values)
            if dp:
                distributions.append({
                    "column": col_name,
                    "count": dp.count,
                    "mean": dp.mean,
                    "median": dp.median,
                    "stdev": dp.stdev,
                    "min": dp.min_val,
                    "max": dp.max_val,
                    "q1": dp.q1,
                    "q3": dp.q3,
                    "iqr": dp.iqr,
                    "skewness": dp.skewness,
                    "kurtosis": dp.kurtosis,
                    "shape": dp.shape,
                    "histogram": dp.histogram,
                })
        except Exception:
            logger.exception("Distribution profiling failed for %s.%s", table_name, col_name)

    return {"table_name": table_name, "distributions": distributions}


@app.get("/dashboard/summary")
async def get_dashboard_summary(session: AsyncSession = Depends(get_session)) -> dict:
    """Get high-level dashboard KPIs."""
    from sqlalchemy import select as sa_select
    from business_brain.db.discovery_models import (
        DeployedReport,
        DiscoveryRun,
        Insight,
        TableProfile,
    )
    from business_brain.discovery.dashboard_summary import compute_dashboard_summary

    profiles = list((await session.execute(sa_select(TableProfile))).scalars().all())
    insights = list((await session.execute(sa_select(Insight))).scalars().all())
    reports = list((await session.execute(sa_select(DeployedReport))).scalars().all())
    runs = list((await session.execute(sa_select(DiscoveryRun))).scalars().all())

    summary = compute_dashboard_summary(profiles, insights, reports, runs)

    return {
        "total_tables": summary.total_tables,
        "total_rows": summary.total_rows,
        "total_columns": summary.total_columns,
        "total_insights": summary.total_insights,
        "total_reports": summary.total_reports,
        "avg_quality_score": summary.avg_quality_score,
        "data_freshness_pct": summary.data_freshness_pct,
        "insight_breakdown": summary.insight_breakdown,
        "severity_breakdown": summary.severity_breakdown,
        "top_tables": summary.top_tables,
        "last_discovery_at": summary.last_discovery_at,
    }


@app.get("/data-freshness")
async def get_data_freshness(session: AsyncSession = Depends(get_session)) -> dict:
    """Get data freshness scores comparing current vs previous profiles."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.data_freshness import compute_freshness_score, detect_stale_tables

    result = await session.execute(select(TableProfile))
    profiles = list(result.scalars().all())

    if not profiles:
        return {"score": 100, "stale_count": 0, "fresh_count": 0, "unknown_count": 0, "total_tables": 0, "stale_tables": []}

    # Use profiles as both current and previous (since we only have one snapshot)
    # In production, you'd compare against a saved snapshot from the previous run
    freshness = compute_freshness_score(profiles, profiles)
    stale_insights = detect_stale_tables(profiles, profiles)

    return {
        **freshness,
        "stale_tables": [i.source_tables[0] for i in stale_insights],
    }


@app.get("/relationships")
async def get_relationships(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get all discovered cross-table relationships."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import DiscoveredRelationship

    result = await session.execute(
        select(DiscoveredRelationship).order_by(DiscoveredRelationship.confidence.desc())
    )
    rels = list(result.scalars().all())

    return [
        {
            "id": r.id,
            "table_a": r.table_a,
            "column_a": r.column_a,
            "table_b": r.table_b,
            "column_b": r.column_b,
            "relationship_type": r.relationship_type,
            "confidence": round(r.confidence, 2) if r.confidence else 0,
            "overlap_count": r.overlap_count,
            "discovered_at": r.discovered_at.isoformat() if r.discovered_at else None,
        }
        for r in rels
    ]


@app.get("/lineage/{table_name}")
async def get_lineage(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get data lineage for a specific table — what depends on it."""
    from business_brain.discovery.lineage_tracker import get_lineage_for_table

    return await get_lineage_for_table(session, table_name)


@app.get("/lineage")
async def get_full_lineage(session: AsyncSession = Depends(get_session)) -> dict:
    """Get full lineage graph with impact rankings and orphaned tables."""
    from sqlalchemy import select as sa_select
    from business_brain.db.discovery_models import (
        DeployedReport,
        DiscoveredRelationship,
        Insight,
        TableProfile,
    )
    from business_brain.discovery.lineage_tracker import (
        build_lineage_graph,
        find_orphaned_tables,
        get_impact_ranking,
    )

    profiles = list((await session.execute(sa_select(TableProfile))).scalars().all())
    relationships = list((await session.execute(sa_select(DiscoveredRelationship))).scalars().all())
    insights = list((await session.execute(sa_select(Insight))).scalars().all())
    reports = list((await session.execute(sa_select(DeployedReport))).scalars().all())

    graph = build_lineage_graph(profiles, relationships, insights, reports)

    return {
        "impact_ranking": get_impact_ranking(graph),
        "orphaned_tables": find_orphaned_tables(graph),
        "table_count": len(graph["tables"]),
        "insight_count": len(graph["insights"]),
        "report_count": len(graph["reports"]),
        "edge_count": len(graph["edges"]),
    }


# ---------------------------------------------------------------------------
# Data Comparison
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/compare")
async def compare_table_data(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare two snapshots of a table to find changes.

    Body: {"key_columns": ["id"], "old_rows": [...], "new_rows": [...]}
    If old_rows/new_rows not provided, compares current data vs previous profile.
    """
    from business_brain.discovery.data_comparator import (
        classify_change,
        compare_snapshots,
        compute_change_rate,
    )

    key_columns = body.get("key_columns", [])

    if "old_rows" in body and "new_rows" in body:
        old_rows = body["old_rows"]
        new_rows = body["new_rows"]
    else:
        # Compare current table data with itself (useful with offset/limit)
        meta_store = MetadataStore(session)
        tables = await meta_store.list_tables()
        if table_name not in tables:
            return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)
        result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 500'))
        rows = [dict(r._mapping) for r in result.fetchall()]
        return {"message": "Provide old_rows and new_rows in request body", "current_row_count": len(rows)}

    diff = compare_snapshots(old_rows, new_rows, key_columns, table_name)
    change_type = classify_change(diff)
    rate = compute_change_rate(diff)

    return {
        "table_name": diff.table_name,
        "added_rows": diff.added_rows,
        "removed_rows": diff.removed_rows,
        "changed_rows": diff.changed_rows,
        "unchanged_rows": diff.unchanged_rows,
        "total_old": diff.total_old,
        "total_new": diff.total_new,
        "column_changes": diff.column_changes,
        "change_rate": round(rate, 1),
        "change_type": change_type,
        "summary": diff.summary,
        "sample_additions": diff.sample_additions[:5],
        "sample_removals": diff.sample_removals[:5],
        "sample_changes": [
            {
                "key": sc.key_values,
                "changes": [{"column": c.column, "old": c.old_value, "new": c.new_value} for c in sc.changes],
            }
            for sc in diff.sample_changes[:5]
        ],
    }


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


@app.post("/goals/evaluate")
async def evaluate_goals_endpoint(body: dict):
    """Evaluate metric goals against current values.

    Body: {
        "goals": [{"metric_name": "revenue", "target_value": 1000, "direction": "above", "baseline": 0}],
        "current_values": {"revenue": 800}
    }
    """
    from business_brain.discovery.goal_tracker import (
        Goal,
        compute_overall_health,
        evaluate_goals,
    )

    raw_goals = body.get("goals", [])
    current_values = body.get("current_values", {})

    goals = [
        Goal(
            metric_name=g["metric_name"],
            target_value=g["target_value"],
            direction=g.get("direction", "above"),
            target_min=g.get("target_min"),
            baseline=g.get("baseline"),
            deadline=g.get("deadline"),
        )
        for g in raw_goals
    ]

    progress = evaluate_goals(goals, current_values)
    health = compute_overall_health(progress)

    return {
        "goals": [
            {
                "metric_name": p.metric_name,
                "current_value": p.current_value,
                "target_value": p.target_value,
                "direction": p.direction,
                "progress_pct": p.progress_pct,
                "status": p.status,
                "remaining": p.remaining,
                "summary": p.summary,
            }
            for p in progress
        ],
        "health": health,
    }


# ---------------------------------------------------------------------------
# Anomaly Classification
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/classify-anomalies")
async def classify_anomalies_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Classify anomalies in a numeric column.

    Body: {"column": "amount", "threshold_std": 2.0}
    """
    from business_brain.discovery.anomaly_classifier import (
        classify_series,
        compute_anomaly_score,
        summarize_anomalies,
    )

    column = body.get("column")
    threshold = body.get("threshold_std", 2.0)

    if not column:
        return JSONResponse({"error": "column is required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 1000'))
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < 5:
        return {"classifications": [], "summary": {"total": 0, "summary": "Not enough numeric data."}}

    classifications = classify_series(values, threshold_std=threshold)
    summary = summarize_anomalies(classifications)

    return {
        "table": table_name,
        "column": column,
        "total_values": len(values),
        "classifications": [
            {
                "pattern": c.pattern,
                "confidence": c.confidence,
                "description": c.description,
                "severity": c.severity,
                "affected_indices": c.affected_indices[:20],
                "score": compute_anomaly_score(c),
            }
            for c in classifications
        ],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Profile Report
# ---------------------------------------------------------------------------


@app.get("/tables/{table_name}/profile-report")
async def get_profile_report(
    table_name: str,
    session: AsyncSession = Depends(get_session),
):
    """Generate a comprehensive profile report for a table."""
    from business_brain.discovery.profile_report import (
        compute_report_priority,
        format_report_text,
        generate_profile_report,
    )

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    # Get profile from DB
    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()

    columns = {}
    row_count = 0
    domain = "general"
    if profile:
        row_count = profile.row_count or 0
        domain = profile.domain_hint or "general"
        if profile.column_classification and "columns" in profile.column_classification:
            columns = profile.column_classification["columns"]
    else:
        # Fallback: count rows
        count_result = await session.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        row_count = count_result.scalar() or 0

    # Get relationships
    rels_result = await session.execute(
        sa_select(DiscoveredRelationship).where(
            (DiscoveredRelationship.table_a == table_name) | (DiscoveredRelationship.table_b == table_name)
        )
    )
    rels = [
        {"table_a": r.table_a, "column_a": r.column_a, "table_b": r.table_b, "column_b": r.column_b,
         "relationship_type": r.relationship_type, "confidence": r.confidence}
        for r in rels_result.scalars().all()
    ]

    report = generate_profile_report(table_name, row_count, columns, domain, rels)

    return {
        "table_name": report.table_name,
        "row_count": report.row_count,
        "column_count": report.column_count,
        "domain": report.domain,
        "quality_score": report.quality_score,
        "priority": compute_report_priority(report),
        "summary": report.summary,
        "sections": [
            {"title": s.title, "content": s.content, "severity": s.severity}
            for s in report.sections
        ],
        "recommendations": report.recommendations,
        "text_report": format_report_text(report),
    }


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/benchmark")
async def benchmark_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare a metric across groups in a table.

    Body: {"group_column": "supplier", "metric_column": "cost", "metric_name": "Unit Cost"}
    """
    from business_brain.discovery.benchmarking import benchmark_groups

    group_col = body.get("group_column")
    metric_col = body.get("metric_column")
    if not group_col or not metric_col:
        return JSONResponse({"error": "group_column and metric_column required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 5000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    benchmark = benchmark_groups(rows, group_col, metric_col, body.get("metric_name"))
    if not benchmark:
        return {"error": "Insufficient data for benchmarking (need 2+ groups with numeric values)"}

    return {
        "metric_name": benchmark.metric_name,
        "group_column": benchmark.group_column,
        "best_group": benchmark.best_group,
        "worst_group": benchmark.worst_group,
        "spread": benchmark.spread,
        "spread_pct": benchmark.spread_pct,
        "ranking": benchmark.ranking,
        "significant_gaps": benchmark.significant_gaps,
        "summary": benchmark.summary,
        "groups": [
            {"name": g.group_name, "count": g.count, "mean": g.mean, "median": g.median,
             "min": g.min_val, "max": g.max_val, "std": g.std, "total": g.total}
            for g in benchmark.groups
        ],
    }


# ---------------------------------------------------------------------------
# Cohort Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/cohort")
async def cohort_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run cohort analysis on a table.

    Body: {"cohort_column": "supplier", "time_column": "month", "metric_column": "revenue"}
    """
    from business_brain.discovery.cohort_analysis import (
        build_cohorts,
        compute_cohort_health,
        find_declining_cohorts,
        pivot_cohort_table,
    )

    cohort_col = body.get("cohort_column")
    time_col = body.get("time_column")
    metric_col = body.get("metric_column")
    if not cohort_col or not time_col or not metric_col:
        return JSONResponse({"error": "cohort_column, time_column, and metric_column required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cohort_result = build_cohorts(rows, cohort_col, time_col, metric_col)
    if not cohort_result:
        return {"error": "Insufficient data for cohort analysis (need 2+ cohort-period combinations)"}

    health = compute_cohort_health(cohort_result)
    declining = find_declining_cohorts(cohort_result)
    pivot = pivot_cohort_table(cohort_result)

    return {
        "cohort_column": cohort_result.cohort_column,
        "time_column": cohort_result.time_column,
        "metric_column": cohort_result.metric_column,
        "cohorts": cohort_result.cohorts,
        "periods": cohort_result.periods,
        "summary": cohort_result.summary,
        "health": health,
        "declining_cohorts": declining,
        "pivot_table": pivot,
        "retention_matrix": cohort_result.retention_matrix,
        "growth_matrix": cohort_result.growth_matrix,
    }


# ---------------------------------------------------------------------------
# Validation Rules
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/validate")
async def validate_table(
    table_name: str,
    body: dict | None = None,
    session: AsyncSession = Depends(get_session),
):
    """Validate table data against auto-generated or custom rules.

    Body (optional): {"rules": [{"name": "...", "column": "...", "rule_type": "not_null"}]}
    If no rules provided, auto-generates from column profile.
    """
    from business_brain.discovery.validation_rules import (
        auto_generate_rules,
        create_rule,
        evaluate_rules,
    )

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    # Get rules
    rules = []
    if body and "rules" in body:
        for r in body["rules"]:
            rules.append(create_rule(
                r.get("name", r.get("column", "rule")),
                r["column"],
                r["rule_type"],
                severity=r.get("severity", "warning"),
                **{k: v for k, v in r.items() if k not in ("name", "column", "rule_type", "severity")},
            ))
    else:
        # Auto-generate from profile
        profile_result = await session.execute(
            sa_select(TableProfile).where(TableProfile.table_name == table_name)
        )
        profile = profile_result.scalar_one_or_none()
        if profile and profile.column_classification and "columns" in profile.column_classification:
            rules = auto_generate_rules(profile.column_classification["columns"])
        else:
            return {"message": "No profile found. Run discovery first to auto-generate validation rules."}

    # Get data
    result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 5000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    report = evaluate_rules(rows, rules)

    return {
        "table": table_name,
        "total_rules": report.total_rules,
        "total_rows": report.total_rows,
        "rules_passed": report.rules_passed,
        "rules_failed": report.rules_failed,
        "pass_rate": report.pass_rate,
        "total_violations": report.total_violations,
        "summary": report.summary,
        "rule_results": report.rule_results,
        "violations": [
            {"rule": v.rule_name, "column": v.column, "row": v.row_index,
             "value": str(v.value) if v.value is not None else None,
             "message": v.message, "severity": v.severity}
            for v in report.violations[:50]
        ],
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@app.get("/recommendations")
async def get_recommendations(session: AsyncSession = Depends(get_session)):
    """Get analysis recommendations based on current data state."""
    from business_brain.discovery.insight_recommender import (
        compute_coverage,
        recommend_analyses,
    )

    profiles = list((await session.execute(sa_select(TableProfile))).scalars().all())
    insights = list((await session.execute(sa_select(Insight))).scalars().all())
    relationships = list((await session.execute(sa_select(DiscoveredRelationship))).scalars().all())

    # Convert ORM objects to dicts
    profile_dicts = [
        {"table_name": p.table_name, "row_count": p.row_count,
         "column_classification": p.column_classification}
        for p in profiles
    ]
    insight_dicts = [
        {"insight_type": i.insight_type, "source_tables": i.source_tables}
        for i in insights
    ]
    rel_dicts = [
        {"table_a": r.table_a, "table_b": r.table_b}
        for r in relationships
    ]

    recs = recommend_analyses(profile_dicts, insight_dicts, rel_dicts)
    coverage = compute_coverage(profile_dicts, insight_dicts)

    return {
        "recommendations": [
            {"title": r.title, "description": r.description, "analysis_type": r.analysis_type,
             "target_table": r.target_table, "columns": r.columns, "priority": r.priority, "reason": r.reason}
            for r in recs
        ],
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# KPI Calculator
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/kpis")
async def compute_table_kpis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute KPIs for a numeric column.

    Body: {"column": "revenue", "target": 1000, "capacity": 5000}
    """
    from business_brain.discovery.kpi_calculator import compute_all_kpis, moving_average, rate_of_change

    column = body.get("column")
    if not column:
        return JSONResponse({"error": "column required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 2000'))
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if not values:
        return {"error": "No numeric values found"}

    target = body.get("target")
    capacity = body.get("capacity")
    kpis = compute_all_kpis(values, target=target, capacity=capacity)
    ma = moving_average(values, window=min(5, len(values)))
    roc = rate_of_change(values)

    return {
        "table": table_name,
        "column": column,
        "total_values": len(values),
        "kpis": [
            {"name": k.name, "value": k.value, "unit": k.unit,
             "interpretation": k.interpretation, "trend": k.trend, "status": k.status}
            for k in kpis
        ],
        "moving_average": ma[-10:] if len(ma) > 10 else ma,
        "rate_of_change": roc[-10:] if len(roc) > 10 else roc,
    }


# ---------------------------------------------------------------------------
# Pareto Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/pareto")
async def pareto_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run Pareto (80/20) analysis on a table.

    Body: {"group_column": "supplier", "metric_column": "cost", "threshold": 80.0}
    """
    from business_brain.discovery.pareto_analysis import (
        compare_pareto,
        find_concentration_risk,
        pareto_analysis,
    )

    group_col = body.get("group_column")
    metric_col = body.get("metric_column")
    if not group_col or not metric_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "group_column and metric_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found in table"}

    threshold = body.get("threshold", 80.0)
    pareto = pareto_analysis(rows, group_col, metric_col, threshold)
    if not pareto:
        return {"error": "Insufficient data for Pareto analysis (need 2+ groups)"}

    risks = find_concentration_risk(pareto)

    return {
        "total": pareto.total,
        "is_pareto": pareto.is_pareto,
        "pareto_ratio": pareto.pareto_ratio,
        "vital_few_count": pareto.vital_few_count,
        "vital_few_pct": pareto.vital_few_pct,
        "vital_few_contribution": pareto.vital_few_contribution,
        "trivial_many_count": pareto.trivial_many_count,
        "summary": pareto.summary,
        "items": [
            {
                "name": item.name,
                "value": item.value,
                "pct": item.pct_of_total,
                "cumulative_pct": item.cumulative_pct,
                "rank": item.rank,
                "is_vital": item.is_vital,
            }
            for item in pareto.items
        ],
        "concentration_risks": risks,
    }


# ---------------------------------------------------------------------------
# What-If Scenario Engine
# ---------------------------------------------------------------------------


class ScenarioRequest(BaseModel):
    name: str
    parameters: dict[str, float]
    description: str = ""


class WhatIfRequest(BaseModel):
    scenarios: list[ScenarioRequest]
    formula: str
    base_values: dict[str, float] | None = None


@app.post("/scenarios/evaluate")
async def evaluate_scenarios(body: WhatIfRequest):
    """Evaluate what-if scenarios against a formula.

    Body: {
        "scenarios": [{"name": "base", "parameters": {"price": 100, "qty": 50}}, ...],
        "formula": "price * qty",
        "base_values": {"price": 100, "qty": 50}
    }
    """
    from business_brain.discovery.whatif_engine import (
        Scenario,
        compare_scenarios,
        evaluate_scenario,
    )

    scenarios = [Scenario(s.name, s.parameters, s.description) for s in body.scenarios]
    base = body.base_values

    if len(scenarios) == 1:
        outcome = evaluate_scenario(scenarios[0], body.formula, base)
        return {
            "scenario_name": outcome.scenario_name,
            "result": outcome.result,
            "parameters": outcome.parameters,
            "interpretation": outcome.interpretation,
        }

    comp = compare_scenarios(scenarios, body.formula, base)
    return {
        "best_scenario": comp.best_scenario,
        "worst_scenario": comp.worst_scenario,
        "range_min": comp.range_min,
        "range_max": comp.range_max,
        "sensitivity": comp.sensitivity,
        "summary": comp.summary,
        "outcomes": [
            {
                "scenario_name": o.scenario_name,
                "result": o.result,
                "parameters": o.parameters,
                "interpretation": o.interpretation,
            }
            for o in comp.outcomes
        ],
    }


@app.post("/scenarios/breakeven")
async def breakeven(body: dict):
    """Find breakeven value for a variable.

    Body: {"formula": "revenue - cost", "variable": "revenue",
           "base_values": {"revenue": 0, "cost": 500}, "target": 0,
           "search_range": [0, 1000]}
    """
    from business_brain.discovery.whatif_engine import breakeven_analysis

    formula = body.get("formula", "")
    variable = body.get("variable", "")
    base_values = body.get("base_values", {})
    target = body.get("target", 0.0)
    sr = body.get("search_range", [-1000, 1000])

    result = breakeven_analysis(formula, variable, base_values, target, tuple(sr))
    return result


@app.post("/scenarios/sensitivity")
async def sensitivity(body: dict):
    """Generate sensitivity table for a variable.

    Body: {"formula": "price * qty", "variable": "price",
           "base_values": {"price": 100, "qty": 50},
           "variations": [0.8, 0.9, 1.0, 1.1, 1.2]}
    """
    from business_brain.discovery.whatif_engine import sensitivity_table

    formula = body.get("formula", "")
    variable = body.get("variable", "")
    base_values = body.get("base_values", {})
    variations = body.get("variations")

    table = sensitivity_table(formula, variable, base_values, variations)
    return {"variable": variable, "base_value": base_values.get(variable, 0), "rows": table}


# ---------------------------------------------------------------------------
# Statistical Summary
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/stats")
async def stat_summary(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute comprehensive statistical summary for a numeric column.

    Body: {"column": "revenue"}
    """
    from business_brain.discovery.stat_summary import (
        compute_stat_summary,
        format_stat_table,
    )

    column = body.get("column")
    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(
        sql_text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 5000')
    )
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < 3:
        return {"error": f"Need at least 3 numeric values, found {len(values)}"}

    summary = compute_stat_summary(values, column)
    if summary is None:
        return {"error": "Could not compute summary"}

    text_table = format_stat_table(summary)

    return {
        "column": summary.column,
        "count": summary.count,
        "mean": summary.mean,
        "median": summary.median,
        "mode": summary.mode,
        "std": summary.std,
        "variance": summary.variance,
        "min": summary.min_val,
        "max": summary.max_val,
        "range": summary.range_val,
        "q1": summary.q1,
        "q3": summary.q3,
        "iqr": summary.iqr,
        "skewness": summary.skewness,
        "kurtosis": summary.kurtosis,
        "cv": summary.cv,
        "percentiles": summary.percentiles,
        "ci_95_lower": summary.ci_95_lower,
        "ci_95_upper": summary.ci_95_upper,
        "normality": summary.normality,
        "outlier_count": summary.outlier_count,
        "interpretation": summary.interpretation,
        "formatted_table": text_table,
    }


@app.post("/tables/{table_name}/compare-distributions")
async def compare_distributions_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare distributions of two numeric columns.

    Body: {"column_a": "cost_q1", "column_b": "cost_q2"}
    """
    from business_brain.discovery.stat_summary import (
        compare_distributions,
        compute_stat_summary,
    )

    col_a = body.get("column_a")
    col_b = body.get("column_b")
    if not col_a or not col_b:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column_a and column_b required"}, 400)

    from sqlalchemy import text as sql_text

    async def _get_values(col):
        res = await session.execute(
            sql_text(f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 5000')
        )
        vals = []
        for r in res.fetchall():
            try:
                vals.append(float(r[0]))
            except (TypeError, ValueError):
                continue
        return vals

    vals_a = await _get_values(col_a)
    vals_b = await _get_values(col_b)

    if len(vals_a) < 3 or len(vals_b) < 3:
        return {"error": "Need at least 3 values in each column"}

    summary_a = compute_stat_summary(vals_a, col_a)
    summary_b = compute_stat_summary(vals_b, col_b)
    if not summary_a or not summary_b:
        return {"error": "Could not compute summaries"}

    comp = compare_distributions(summary_a, summary_b)
    return comp


# ---------------------------------------------------------------------------
# Segmentation Engine
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/segment")
async def segment_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Cluster table rows into segments based on numeric features.

    Body: {"features": ["revenue", "cost"], "n_segments": 3}
    """
    from business_brain.discovery.segmentation_engine import (
        find_segment_drivers,
        label_segments,
        segment_data,
    )

    features = body.get("features", [])
    n_segments = body.get("n_segments", 3)
    if not features:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "features list required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found"}

    seg_result = segment_data(rows, features, n_segments)
    if seg_result is None:
        return {"error": "Insufficient data for segmentation"}

    seg_result.segments = label_segments(seg_result.segments, features)
    drivers = find_segment_drivers(seg_result.segments, features)

    return {
        "n_segments": seg_result.n_segments,
        "total_rows": seg_result.total_rows,
        "features": seg_result.features,
        "quality_score": seg_result.quality_score,
        "summary": seg_result.summary,
        "segments": [
            {
                "segment_id": s.segment_id,
                "label": s.label,
                "size": s.size,
                "center": s.center,
                "spread": s.spread,
            }
            for s in seg_result.segments
        ],
        "drivers": drivers,
    }


# ---------------------------------------------------------------------------
# Trend Decomposition
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/decompose")
async def decompose_trend(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Decompose a numeric column into trend + seasonal + residual.

    Body: {"column": "revenue", "period": null}
    """
    from business_brain.discovery.trend_decomposer import (
        decompose,
        find_anomalous_residuals,
    )

    column = body.get("column")
    period = body.get("period")
    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(
        sql_text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 5000')
    )
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < 6:
        return {"error": f"Need at least 6 values, found {len(values)}"}

    dec = decompose(values, period=period)
    if dec is None:
        return {"error": "Could not decompose series"}

    anomalies = find_anomalous_residuals(dec.residual)

    return {
        "column": column,
        "period": dec.period,
        "trend_direction": dec.trend_direction,
        "trend_strength": dec.trend_strength,
        "seasonal_strength": dec.seasonal_strength,
        "summary": dec.summary,
        "trend": dec.trend[-50:] if len(dec.trend) > 50 else dec.trend,
        "seasonal": dec.seasonal[-50:] if len(dec.seasonal) > 50 else dec.seasonal,
        "residual": dec.residual[-50:] if len(dec.residual) > 50 else dec.residual,
        "anomalous_residuals": anomalies[:10],
    }


# ---------------------------------------------------------------------------
# Pivot Tables
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/pivot")
async def pivot_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Create a pivot table from table data.

    Body: {"row_field": "supplier", "col_field": "quarter", "value_field": "amount", "agg_func": "sum"}
    """
    from business_brain.discovery.pivot_engine import (
        create_pivot,
        find_pivot_outliers,
        format_pivot_text,
    )

    row_field = body.get("row_field")
    col_field = body.get("col_field")
    value_field = body.get("value_field")
    agg_func = body.get("agg_func", "sum")

    if not row_field or not col_field or not value_field:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "row_field, col_field, and value_field required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    pivot = create_pivot(rows, row_field, col_field, value_field, agg_func)
    if pivot is None:
        return {"error": "Insufficient data for pivot table"}

    outliers = find_pivot_outliers(pivot)
    text = format_pivot_text(pivot)

    # Build cells grid
    cells = {}
    for (rk, ck), cell in pivot.cells.items():
        if rk not in cells:
            cells[rk] = {}
        cells[rk][ck] = cell.value

    return {
        "row_keys": pivot.row_keys,
        "col_keys": pivot.col_keys,
        "cells": cells,
        "row_totals": pivot.row_totals,
        "col_totals": pivot.col_totals,
        "grand_total": pivot.grand_total,
        "agg_func": pivot.agg_func,
        "summary": pivot.summary,
        "formatted_text": text,
        "outliers": outliers[:10],
    }


# ---------------------------------------------------------------------------
# Variance Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/variance")
async def variance_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute budget vs actual variance analysis.

    Body: {"category_column": "department", "planned_column": "budget",
           "actual_column": "actual", "favorable_direction": "higher"}
    """
    from business_brain.discovery.variance_analysis import (
        compute_variance,
        find_root_causes,
        waterfall_breakdown,
    )

    cat_col = body.get("category_column")
    planned_col = body.get("planned_column")
    actual_col = body.get("actual_column")
    fav_dir = body.get("favorable_direction", "higher")

    if not cat_col or not planned_col or not actual_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column, planned_column, actual_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    report = compute_variance(rows, cat_col, planned_col, actual_col, fav_dir)
    if report is None:
        return {"error": "Insufficient data for variance analysis"}

    waterfall = waterfall_breakdown(report)
    causes = find_root_causes(report)

    return {
        "total_planned": report.total_planned,
        "total_actual": report.total_actual,
        "total_variance": report.total_variance,
        "total_variance_pct": report.total_variance_pct,
        "favorable_count": report.favorable_count,
        "unfavorable_count": report.unfavorable_count,
        "summary": report.summary,
        "items": [
            {
                "category": it.category,
                "planned": it.planned,
                "actual": it.actual,
                "variance": it.variance,
                "variance_pct": it.variance_pct,
                "is_favorable": it.is_favorable,
                "severity": it.severity,
            }
            for it in report.items
        ],
        "waterfall": waterfall,
        "root_causes": causes,
    }


# ---------------------------------------------------------------------------
# Data Dictionary
# ---------------------------------------------------------------------------


@app.get("/tables/{table_name}/dictionary")
async def get_data_dictionary(
    table_name: str,
    session: AsyncSession = Depends(get_session),
):
    """Auto-generate a data dictionary for a table."""
    from business_brain.discovery.data_dictionary import (
        format_dictionary_markdown,
        generate_data_dictionary,
    )

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 500'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found"}

    dd = generate_data_dictionary(rows, table_name)
    if dd is None:
        return {"error": "Could not generate dictionary"}

    markdown = format_dictionary_markdown(dd)

    return {
        "table_name": dd.table_name,
        "row_count": dd.row_count,
        "column_count": dd.column_count,
        "summary": dd.summary,
        "markdown": markdown,
        "columns": [
            {
                "name": c.name,
                "type": c.inferred_type,
                "description": c.description,
                "null_pct": c.null_pct,
                "unique_pct": c.unique_pct,
                "min": c.min_value,
                "max": c.max_value,
                "mean": c.mean_value,
                "sample_values": c.sample_values,
                "tags": c.tags,
            }
            for c in dd.columns
        ],
        "relationships": dd.relationships_hint,
    }


# ---------------------------------------------------------------------------
# Quality Score
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/quality-score")
async def compute_quality_score(
    table_name: str,
    session: AsyncSession = Depends(get_session),
):
    """Compute a unified data quality score for a table."""
    from business_brain.discovery.quality_scorer import compute_quality_report

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 5000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found"}

    report = compute_quality_report(rows)

    return {
        "overall_score": report.overall_score,
        "grade": report.grade,
        "summary": report.summary,
        "dimensions": [
            {
                "dimension": d.dimension,
                "score": d.score,
                "weight": d.weight,
                "issues": d.issues,
            }
            for d in report.dimensions
        ],
        "critical_issues": report.critical_issues,
        "recommendations": report.recommendations,
    }


# ---------------------------------------------------------------------------
# ABC Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/abc")
async def abc_analysis_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run ABC classification on table data.

    Body: {"name_column": "supplier", "value_column": "cost", "a_threshold": 80, "b_threshold": 95}
    """
    from business_brain.discovery.abc_analysis import abc_analysis, format_abc_table

    name_col = body.get("name_column")
    value_col = body.get("value_column")
    if not name_col or not value_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "name_column and value_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    a_thresh = body.get("a_threshold", 80.0)
    b_thresh = body.get("b_threshold", 95.0)
    abc = abc_analysis(rows, name_col, value_col, a_thresh, b_thresh)
    if abc is None:
        return {"error": "Insufficient data for ABC analysis"}

    return {
        "total_value": abc.total_value,
        "a_count": abc.a_count,
        "b_count": abc.b_count,
        "c_count": abc.c_count,
        "a_value_pct": abc.a_value_pct,
        "b_value_pct": abc.b_value_pct,
        "c_value_pct": abc.c_value_pct,
        "summary": abc.summary,
        "formatted_table": format_abc_table(abc),
        "items": [
            {
                "name": it.name,
                "value": it.value,
                "pct": it.pct_of_total,
                "cumulative_pct": it.cumulative_pct,
                "rank": it.rank,
                "category": it.category,
            }
            for it in abc.items
        ],
    }


# ---------------------------------------------------------------------------
# Funnel Analysis
# ---------------------------------------------------------------------------


@app.post("/funnel/analyze")
async def funnel_analyze(body: dict):
    """Analyze a conversion funnel.

    Body: {"stages": [["Visits", 1000], ["Signups", 200], ["Purchases", 50]]}
    """
    from business_brain.discovery.funnel_analysis import analyze_funnel, format_funnel_text

    stages_raw = body.get("stages", [])
    if len(stages_raw) < 2:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Need at least 2 stages"}, 400)

    stages = [(s[0], int(s[1])) for s in stages_raw]
    result = analyze_funnel(stages)
    if result is None:
        return {"error": "Could not analyze funnel"}

    return {
        "initial_count": result.initial_count,
        "final_count": result.final_count,
        "overall_conversion": result.overall_conversion,
        "biggest_drop_stage": result.biggest_drop_stage,
        "biggest_drop_pct": result.biggest_drop_pct,
        "summary": result.summary,
        "formatted_text": format_funnel_text(result),
        "stages": [
            {
                "name": s.name,
                "count": s.count,
                "pct_of_total": s.pct_of_total,
                "conversion_rate": s.conversion_rate,
                "drop_off": s.drop_off,
                "drop_off_pct": s.drop_off_pct,
            }
            for s in result.stages
        ],
    }


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/rolling")
async def rolling_stats_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute rolling statistics for a numeric column.

    Body: {"column": "revenue", "window": 5}
    """
    from business_brain.discovery.rolling_stats import (
        detect_regime_changes,
        rolling_statistics,
    )

    column = body.get("column")
    window = body.get("window", 5)
    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(
        sql_text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 5000')
    )
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < window:
        return {"error": f"Need at least {window} values, found {len(values)}"}

    stats = rolling_statistics(values, window)
    if stats is None:
        return {"error": "Could not compute rolling statistics"}

    regime_changes = detect_regime_changes(values, window)

    # Only return last 100 values to keep response size manageable
    n = min(100, len(values))

    return {
        "column": column,
        "window": window,
        "total_values": len(values),
        "summary": stats.summary,
        "rolling_mean": stats.rolling_mean[-n:],
        "rolling_std": stats.rolling_std[-n:],
        "z_scores": stats.z_scores[-n:],
        "regime_changes": regime_changes[:20],
    }


# ---------------------------------------------------------------------------
# Contribution Analysis
# ---------------------------------------------------------------------------


@app.post("/contribution/analyze")
async def contribution_analyze(body: dict):
    """Analyze what's driving change between two periods.

    Body: {"before": {"Sales": 100, "Services": 50}, "after": {"Sales": 120, "Services": 60}}
    """
    from business_brain.discovery.contribution_analysis import (
        analyze_contributions,
        waterfall_data,
    )

    before = body.get("before", {})
    after = body.get("after", {})
    result = analyze_contributions(before, after)
    if result is None:
        return {"error": "No data provided"}

    wf = waterfall_data(result)

    return {
        "total_before": result.total_before,
        "total_after": result.total_after,
        "total_change": result.total_change,
        "total_change_pct": result.total_change_pct,
        "top_positive_driver": result.top_positive_driver,
        "top_negative_driver": result.top_negative_driver,
        "concentration": result.concentration,
        "summary": result.summary,
        "items": [
            {
                "name": it.name,
                "value_before": it.value_before,
                "value_after": it.value_after,
                "absolute_change": it.absolute_change,
                "pct_change": it.pct_change,
                "contribution_pct": it.contribution_pct,
                "direction": it.direction,
            }
            for it in result.items
        ],
        "waterfall": wf,
    }


# ---------------------------------------------------------------------------
# Composite Index Calculator
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/index")
async def compute_index_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute a composite index score for entities in a table.

    Body: {
        "entity_column": "supplier",
        "metrics": [
            {"column": "quality", "weight": 0.5, "direction": "higher_is_better"},
            {"column": "cost", "weight": 0.5, "direction": "lower_is_better"}
        ],
        "index_name": "Supplier Score"
    }
    """
    from business_brain.discovery.index_calculator import compute_index, format_index_table

    entity_col = body.get("entity_column")
    metrics = body.get("metrics", [])
    index_name = body.get("index_name", "Index")

    if not entity_col or not metrics:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column and metrics required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    idx = compute_index(rows, entity_col, metrics, index_name)
    if idx is None:
        return {"error": "Insufficient data for index computation"}

    return {
        "name": idx.name,
        "entity_count": idx.entity_count,
        "mean_score": idx.mean_score,
        "median_score": idx.median_score,
        "std_score": idx.std_score,
        "top_entity": idx.top_entity,
        "bottom_entity": idx.bottom_entity,
        "summary": idx.summary,
        "formatted_table": format_index_table(idx),
        "scores": [
            {
                "entity": s.entity,
                "score": s.score,
                "grade": s.grade,
                "rank": s.rank,
                "components": [
                    {"name": c.name, "raw": c.raw_value, "normalized": c.normalized_value,
                     "weight": c.weight, "contribution": c.weighted_contribution}
                    for c in s.components
                ],
            }
            for s in idx.scores
        ],
    }


# ---------------------------------------------------------------------------
# Capacity Planning
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/capacity")
async def capacity_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute capacity utilization for entities in a table.

    Body: {
        "entity_column": "machine",
        "actual_column": "output",
        "capacity_column": "max_output",
        "time_column": null  (optional)
    }
    """
    from business_brain.discovery.capacity_planning import (
        compute_utilization,
        detect_bottlenecks,
        forecast_capacity_exhaustion,
        capacity_summary,
    )

    entity_col = body.get("entity_column")
    actual_col = body.get("actual_column")
    capacity_col = body.get("capacity_column")
    time_col = body.get("time_column")

    if not entity_col or not actual_col or not capacity_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, actual_column, capacity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    util = compute_utilization(rows, entity_col, actual_col, capacity_col, time_col)
    if util is None:
        return {"error": "Insufficient data for capacity analysis"}

    # Bottleneck detection if a stage-like entity is present
    bottlenecks = detect_bottlenecks(rows, entity_col, actual_col, time_col)

    # Exhaustion forecast
    forecasts = []
    if time_col:
        forecasts = forecast_capacity_exhaustion(rows, entity_col, actual_col, capacity_col, time_col)

    return {
        "entity_count": util.entity_count,
        "mean_utilization": util.mean_utilization,
        "summary": util.summary,
        "over_utilized": util.over_utilized,
        "under_utilized": util.under_utilized,
        "bottlenecks": util.bottlenecks,
        "entities": [
            {
                "entity": e.entity,
                "actual": e.actual,
                "capacity": e.capacity,
                "utilization_pct": e.utilization_pct,
                "status": e.status,
            }
            for e in util.entities
        ],
        "stage_bottlenecks": [
            {
                "stage": b.stage,
                "throughput": b.throughput,
                "throughput_pct_of_max": b.throughput_pct_of_max,
                "is_bottleneck": b.is_bottleneck,
                "constraint_ratio": b.constraint_ratio,
            }
            for b in bottlenecks
        ],
        "exhaustion_forecasts": [
            {
                "entity": f.entity,
                "current_utilization": f.current_utilization,
                "trend_per_period": f.trend_per_period,
                "periods_to_exhaustion": f.periods_to_exhaustion,
                "urgency": f.urgency,
            }
            for f in forecasts
        ],
    }


# ---------------------------------------------------------------------------
# Efficiency Metrics (OEE, Yield, Energy, Waste)
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/oee")
async def oee_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute OEE (Overall Equipment Effectiveness) for entities.

    Body: {
        "entity_column": "machine",
        "availability_column": "availability",
        "performance_column": "performance",
        "quality_column": "quality"
    }
    """
    from business_brain.discovery.efficiency_metrics import compute_oee

    entity_col = body.get("entity_column")
    avail_col = body.get("availability_column")
    perf_col = body.get("performance_column")
    qual_col = body.get("quality_column")

    if not all([entity_col, avail_col, perf_col, qual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, availability_column, performance_column, quality_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    oee = compute_oee(rows, entity_col, avail_col, perf_col, qual_col)
    if oee is None:
        return {"error": "Insufficient data for OEE computation"}

    return {
        "mean_oee": oee.mean_oee,
        "best_entity": oee.best_entity,
        "worst_entity": oee.worst_entity,
        "world_class_count": oee.world_class_count,
        "summary": oee.summary,
        "entities": [
            {
                "entity": e.entity,
                "availability": e.availability,
                "performance": e.performance,
                "quality": e.quality,
                "oee": e.oee,
                "oee_grade": e.oee_grade,
                "limiting_factor": e.limiting_factor,
            }
            for e in oee.entities
        ],
    }


@app.post("/tables/{table_name}/yield")
async def yield_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute yield analysis for entities.

    Body: {
        "entity_column": "line",
        "input_column": "raw_material",
        "output_column": "finished_goods",
        "defect_column": "defects"  (optional)
    }
    """
    from business_brain.discovery.efficiency_metrics import compute_yield_analysis

    entity_col = body.get("entity_column")
    input_col = body.get("input_column")
    output_col = body.get("output_column")
    defect_col = body.get("defect_column")

    if not all([entity_col, input_col, output_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, input_column, output_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    yld = compute_yield_analysis(rows, entity_col, input_col, output_col, defect_col)
    if yld is None:
        return {"error": "Insufficient data for yield analysis"}

    return {
        "mean_yield": yld.mean_yield,
        "best_entity": yld.best_entity,
        "worst_entity": yld.worst_entity,
        "summary": yld.summary,
        "entities": [
            {
                "entity": e.entity,
                "input_total": e.input_total,
                "output_total": e.output_total,
                "yield_pct": e.yield_pct,
                "defect_rate": e.defect_rate,
                "waste_pct": e.waste_pct,
            }
            for e in yld.entities
        ],
    }


@app.post("/tables/{table_name}/energy-efficiency")
async def energy_efficiency(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute energy efficiency (specific energy consumption).

    Body: {
        "entity_column": "machine",
        "output_column": "production_tons",
        "energy_column": "kwh_consumed"
    }
    """
    from business_brain.discovery.efficiency_metrics import compute_energy_efficiency

    entity_col = body.get("entity_column")
    output_col = body.get("output_column")
    energy_col = body.get("energy_column")

    if not all([entity_col, output_col, energy_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, output_column, energy_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    eng = compute_energy_efficiency(rows, entity_col, output_col, energy_col)
    if eng is None:
        return {"error": "Insufficient data for energy efficiency analysis"}

    return {
        "mean_sec": eng.mean_sec,
        "best_entity": eng.best_entity,
        "worst_entity": eng.worst_entity,
        "potential_savings_pct": eng.potential_savings_pct,
        "summary": eng.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_output": e.total_output,
                "total_energy": e.total_energy,
                "specific_energy": e.specific_energy,
                "efficiency_grade": e.efficiency_grade,
            }
            for e in eng.entities
        ],
    }


# ---------------------------------------------------------------------------
# Downtime Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/downtime")
async def downtime_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze equipment downtime patterns.

    Body: {
        "machine_column": "machine_id",
        "duration_column": "downtime_hours",
        "reason_column": "failure_reason",  (optional)
        "time_column": "date"  (optional)
    }
    """
    from business_brain.discovery.downtime_analyzer import (
        analyze_downtime,
        downtime_pareto,
        format_downtime_report,
    )

    machine_col = body.get("machine_column")
    duration_col = body.get("duration_column")
    reason_col = body.get("reason_column")
    time_col = body.get("time_column")

    if not machine_col or not duration_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "machine_column and duration_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    dt = analyze_downtime(rows, machine_col, duration_col, reason_col, time_col)
    if dt is None:
        return {"error": "Insufficient data for downtime analysis"}

    pareto = []
    if reason_col:
        pareto = downtime_pareto(rows, reason_col, duration_col)

    return {
        "total_downtime": dt.total_downtime,
        "total_events": dt.total_events,
        "worst_machine": dt.worst_machine,
        "best_machine": dt.best_machine,
        "summary": dt.summary,
        "report": format_downtime_report(dt, pareto or None),
        "machines": [
            {
                "machine": m.machine,
                "total_downtime": m.total_downtime,
                "event_count": m.event_count,
                "mttr": m.mttr,
                "availability_pct": m.availability_pct,
                "top_reason": m.top_reason,
            }
            for m in dt.machines
        ],
        "top_reasons": [
            {
                "reason": r.reason,
                "total_duration": r.total_duration,
                "event_count": r.event_count,
                "pct_of_total": r.pct_of_total,
            }
            for r in dt.top_reasons
        ],
        "pareto": [
            {
                "reason": p.reason,
                "total_duration": p.total_duration,
                "pct_of_total": p.pct_of_total,
                "cumulative_pct": p.cumulative_pct,
                "category": p.category,
            }
            for p in pareto
        ],
    }


# ---------------------------------------------------------------------------
# Supplier Scorecard
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/supplier-scorecard")
async def supplier_scorecard(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Build supplier scorecards.

    Body: {
        "supplier_column": "supplier_name",
        "metrics": [
            {"column": "quality", "weight": 0.4, "direction": "higher_is_better"},
            {"column": "delivery_time", "weight": 0.3, "direction": "lower_is_better"},
            {"column": "cost", "weight": 0.3, "direction": "lower_is_better"}
        ]
    }
    """
    from business_brain.discovery.supplier_scorecard import (
        build_scorecard,
        format_scorecard,
    )

    supplier_col = body.get("supplier_column")
    metrics = body.get("metrics", [])

    if not supplier_col or not metrics:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and metrics required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    sc = build_scorecard(rows, supplier_col, metrics)
    if sc is None:
        return {"error": "Insufficient data for scorecard"}

    return {
        "supplier_count": sc.supplier_count,
        "mean_score": sc.mean_score,
        "best_supplier": sc.best_supplier,
        "worst_supplier": sc.worst_supplier,
        "grade_distribution": sc.grade_distribution,
        "summary": sc.summary,
        "formatted_table": format_scorecard(sc),
        "suppliers": [
            {
                "supplier": s.supplier,
                "score": s.score,
                "grade": s.grade,
                "rank": s.rank,
                "strengths": s.strengths,
                "weaknesses": s.weaknesses,
                "metrics": [
                    {"name": m.metric_name, "raw": m.raw_value, "normalized": m.normalized, "weight": m.weight}
                    for m in s.metric_scores
                ],
            }
            for s in sc.suppliers
        ],
    }


@app.post("/tables/{table_name}/supplier-concentration")
async def supplier_concentration_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze supplier concentration (HHI).

    Body: {"supplier_column": "supplier", "value_column": "spend"}
    """
    from business_brain.discovery.supplier_scorecard import supplier_concentration

    supplier_col = body.get("supplier_column")
    value_col = body.get("value_column")

    if not supplier_col or not value_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and value_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    conc = supplier_concentration(rows, supplier_col, value_col)
    if conc is None:
        return {"error": "Insufficient data"}

    return {
        "hhi": conc.hhi,
        "concentration_level": conc.concentration_level,
        "top_supplier_share": conc.top_supplier_share,
        "summary": conc.summary,
        "suppliers": conc.suppliers,
    }


# ---------------------------------------------------------------------------
# Production Scheduling
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/shift-performance")
async def shift_performance(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze shift-wise production performance.

    Body: {
        "shift_column": "shift",
        "output_column": "production_tons",
        "target_column": "target_tons"  (optional)
    }
    """
    from business_brain.discovery.production_scheduler import analyze_shift_performance

    shift_col = body.get("shift_column")
    output_col = body.get("output_column")
    target_col = body.get("target_column")

    if not shift_col or not output_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "shift_column and output_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    sp = analyze_shift_performance(rows, shift_col, output_col, target_col)
    if sp is None:
        return {"error": "Insufficient data for shift analysis"}

    return {
        "best_shift": sp.best_shift,
        "worst_shift": sp.worst_shift,
        "variance_pct": sp.variance_pct,
        "total_output": sp.total_output,
        "summary": sp.summary,
        "shifts": [
            {
                "shift": s.shift,
                "total_output": s.total_output,
                "avg_output": s.avg_output,
                "event_count": s.event_count,
                "achievement_pct": s.achievement_pct,
                "std_dev": s.std_dev,
                "consistency_grade": s.consistency_grade,
            }
            for s in sp.shifts
        ],
    }


@app.post("/tables/{table_name}/plan-vs-actual")
async def plan_vs_actual_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare planned vs actual production.

    Body: {"entity_column": "product", "plan_column": "planned_qty", "actual_column": "actual_qty"}
    """
    from business_brain.discovery.production_scheduler import plan_vs_actual

    entity_col = body.get("entity_column")
    plan_col = body.get("plan_column")
    actual_col = body.get("actual_column")

    if not all([entity_col, plan_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, plan_column, actual_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    pva = plan_vs_actual(rows, entity_col, plan_col, actual_col)
    if pva is None:
        return {"error": "Insufficient data"}

    return {
        "overall_achievement_pct": pva.overall_achievement_pct,
        "over_achievers": pva.over_achievers,
        "under_achievers": pva.under_achievers,
        "summary": pva.summary,
        "entities": [
            {
                "entity": e.entity,
                "planned": e.planned,
                "actual": e.actual,
                "achievement_pct": e.achievement_pct,
                "variance": e.variance,
                "status": e.status,
            }
            for e in pva.entities
        ],
    }


# ---------------------------------------------------------------------------
# Inventory Optimization
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/inventory-turnover")
async def inventory_turnover(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute inventory turnover ratios.

    Body: {"item_column": "product", "cost_of_goods_column": "cogs", "avg_inventory_column": "avg_inventory"}
    """
    from business_brain.discovery.inventory_optimizer import compute_inventory_turnover

    item_col = body.get("item_column")
    cogs_col = body.get("cost_of_goods_column")
    inv_col = body.get("avg_inventory_column")

    if not all([item_col, cogs_col, inv_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "item_column, cost_of_goods_column, avg_inventory_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    tr = compute_inventory_turnover(rows, item_col, cogs_col, inv_col)
    if tr is None:
        return {"error": "Insufficient data"}

    return {
        "mean_turnover": tr.mean_turnover,
        "best_item": tr.best_item,
        "worst_item": tr.worst_item,
        "slow_movers": tr.slow_movers,
        "fast_movers": tr.fast_movers,
        "summary": tr.summary,
        "items": [
            {
                "item": i.item,
                "cogs": i.cogs,
                "avg_inventory": i.avg_inventory,
                "turnover_ratio": i.turnover_ratio,
                "days_of_inventory": i.days_of_inventory,
                "category": i.category,
            }
            for i in tr.items
        ],
    }


@app.post("/tables/{table_name}/inventory-health")
async def inventory_health(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze inventory health (overstocked, understocked, at reorder point).

    Body: {
        "item_column": "product",
        "quantity_column": "current_qty",
        "min_column": "min_qty",  (optional)
        "max_column": "max_qty",  (optional)
        "reorder_column": "reorder_point"  (optional)
    }
    """
    from business_brain.discovery.inventory_optimizer import analyze_inventory_health

    item_col = body.get("item_column")
    qty_col = body.get("quantity_column")
    min_col = body.get("min_column")
    max_col = body.get("max_column")
    reorder_col = body.get("reorder_column")

    if not item_col or not qty_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "item_column and quantity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    health = analyze_inventory_health(rows, item_col, qty_col, min_col, max_col, reorder_col)
    if health is None:
        return {"error": "Insufficient data"}

    return {
        "overstocked_count": health.overstocked_count,
        "understocked_count": health.understocked_count,
        "at_reorder_count": health.at_reorder_count,
        "healthy_count": health.healthy_count,
        "summary": health.summary,
        "items": [
            {
                "item": i.item,
                "current_qty": i.current_qty,
                "min_qty": i.min_qty,
                "max_qty": i.max_qty,
                "reorder_point": i.reorder_point,
                "status": i.status,
            }
            for i in health.items
        ],
    }


# ---------------------------------------------------------------------------
# Cost Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/cost-breakdown")
async def cost_breakdown_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Break down costs by category.

    Body: {"category_column": "dept", "amount_column": "cost"}
    """
    from business_brain.discovery.cost_analyzer import cost_breakdown

    cat_col = body.get("category_column")
    amt_col = body.get("amount_column")

    if not cat_col or not amt_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column and amount_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cb = cost_breakdown(rows, cat_col, amt_col)
    if cb is None:
        return {"error": "Insufficient data"}

    return {
        "total_cost": cb.total_cost,
        "top_category": cb.top_category,
        "top_3_share_pct": cb.top_3_share_pct,
        "summary": cb.summary,
        "categories": [
            {
                "name": c.name,
                "amount": c.amount,
                "share_pct": c.share_pct,
                "cumulative_pct": c.cumulative_pct,
                "rank": c.rank,
            }
            for c in cb.categories
        ],
    }


@app.post("/tables/{table_name}/cost-per-unit")
async def cost_per_unit_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute cost per unit for entities.

    Body: {"entity_column": "product", "cost_column": "total_cost", "quantity_column": "units_produced"}
    """
    from business_brain.discovery.cost_analyzer import cost_per_unit

    entity_col = body.get("entity_column")
    cost_col = body.get("cost_column")
    qty_col = body.get("quantity_column")

    if not all([entity_col, cost_col, qty_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, cost_column, quantity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cpu = cost_per_unit(rows, entity_col, cost_col, qty_col)
    if cpu is None:
        return {"error": "Insufficient data"}

    return {
        "mean_cpu": cpu.mean_cpu,
        "median_cpu": cpu.median_cpu,
        "best_entity": cpu.best_entity,
        "worst_entity": cpu.worst_entity,
        "spread_pct": cpu.spread_pct,
        "summary": cpu.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_cost": e.total_cost,
                "total_quantity": e.total_quantity,
                "cost_per_unit": e.cost_per_unit,
                "deviation_from_mean_pct": e.deviation_from_mean_pct,
            }
            for e in cpu.entities
        ],
    }


@app.post("/tables/{table_name}/cost-trend")
async def cost_trend_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Track cost trends over time.

    Body: {"time_column": "month", "cost_column": "total_cost", "entity_column": null (optional)}
    """
    from business_brain.discovery.cost_analyzer import cost_trend

    time_col = body.get("time_column")
    cost_col = body.get("cost_column")
    entity_col = body.get("entity_column")

    if not time_col or not cost_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "time_column and cost_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    ct = cost_trend(rows, time_col, cost_col, entity_col)
    if ct is None:
        return {"error": "Insufficient data"}

    return {
        "trend_direction": ct.trend_direction,
        "trend_pct_per_period": ct.trend_pct_per_period,
        "total_change_pct": ct.total_change_pct,
        "volatility": ct.volatility,
        "summary": ct.summary,
        "periods": [
            {
                "period": p.period,
                "total_cost": p.total_cost,
                "change_from_prev": p.change_from_prev,
                "change_pct": p.change_pct,
            }
            for p in ct.periods
        ],
    }


@app.post("/tables/{table_name}/cost-variance")
async def cost_variance_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze cost variance (actual vs budget).

    Body: {"entity_column": "dept", "actual_column": "actual_cost", "budget_column": "budget_cost"}
    """
    from business_brain.discovery.cost_analyzer import cost_variance

    entity_col = body.get("entity_column")
    actual_col = body.get("actual_column")
    budget_col = body.get("budget_column")

    if not all([entity_col, actual_col, budget_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, actual_column, budget_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cv = cost_variance(rows, entity_col, actual_col, budget_col)
    if cv is None:
        return {"error": "Insufficient data"}

    return {
        "total_actual": cv.total_actual,
        "total_budget": cv.total_budget,
        "total_variance": cv.total_variance,
        "total_variance_pct": cv.total_variance_pct,
        "favorable_count": cv.favorable_count,
        "unfavorable_count": cv.unfavorable_count,
        "summary": cv.summary,
        "entities": [
            {
                "entity": e.entity,
                "actual": e.actual,
                "budget": e.budget,
                "variance": e.variance,
                "variance_pct": e.variance_pct,
                "status": e.status,
            }
            for e in cv.entities
        ],
    }


# ---------------------------------------------------------------------------
# Material Balance
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/material-balance")
async def material_balance_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute material balance (input vs output, recovery rate).

    Body: {"entity_column": "furnace", "input_column": "raw_material_tons", "output_column": "product_tons"}
    """
    from business_brain.discovery.material_balance import (
        compute_material_balance,
        detect_material_leakage,
        format_balance_report,
    )

    entity_col = body.get("entity_column")
    input_col = body.get("input_column")
    output_col = body.get("output_column")
    loss_col = body.get("loss_column")

    if not all([entity_col, input_col, output_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, input_column, output_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    mb = compute_material_balance(rows, entity_col, input_col, output_col, loss_col)
    if mb is None:
        return {"error": "Insufficient data"}

    leakage = detect_material_leakage(rows, entity_col, output_col)

    return {
        "total_input": mb.total_input,
        "total_output": mb.total_output,
        "total_loss": mb.total_loss,
        "overall_recovery_pct": mb.overall_recovery_pct,
        "best_recovery_entity": mb.best_recovery_entity,
        "worst_recovery_entity": mb.worst_recovery_entity,
        "summary": mb.summary,
        "report": format_balance_report(mb, leakage or None),
        "entities": [
            {
                "entity": e.entity,
                "total_input": e.total_input,
                "total_output": e.total_output,
                "loss": e.loss,
                "recovery_pct": e.recovery_pct,
                "loss_pct": e.loss_pct,
            }
            for e in mb.entities
        ],
        "leakage_points": [
            {
                "from_stage": lp.from_stage,
                "to_stage": lp.to_stage,
                "input_qty": lp.input_qty,
                "output_qty": lp.output_qty,
                "loss": lp.loss,
                "loss_pct": lp.loss_pct,
                "severity": lp.severity,
            }
            for lp in (leakage or [])
        ],
    }


# ---------------------------------------------------------------------------
# Workforce Analytics
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/attendance")
async def attendance_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze employee attendance.

    Body: {"employee_column": "employee_name", "status_column": "attendance_status", "date_column": "date"}
    """
    from business_brain.discovery.workforce_analytics import analyze_attendance

    emp_col = body.get("employee_column")
    status_col = body.get("status_column")
    date_col = body.get("date_column")

    if not emp_col or not status_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column and status_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    att = analyze_attendance(rows, emp_col, status_col, date_col)
    if att is None:
        return {"error": "Insufficient data"}

    return {
        "total_employees": att.total_employees,
        "avg_attendance_rate": att.avg_attendance_rate,
        "chronic_absentees": att.chronic_absentees,
        "perfect_attendance": att.perfect_attendance,
        "summary": att.summary,
        "employees": [
            {
                "employee": e.employee,
                "total_days": e.total_days,
                "present_days": e.present_days,
                "absent_days": e.absent_days,
                "leave_days": e.leave_days,
                "attendance_rate": e.attendance_rate,
            }
            for e in att.employees
        ],
    }


@app.post("/tables/{table_name}/labor-productivity")
async def labor_productivity(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute labor productivity (output per hour).

    Body: {"entity_column": "department", "output_column": "units_produced", "hours_column": "labor_hours"}
    """
    from business_brain.discovery.workforce_analytics import compute_labor_productivity

    entity_col = body.get("entity_column")
    output_col = body.get("output_column")
    hours_col = body.get("hours_column")

    if not all([entity_col, output_col, hours_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, output_column, hours_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    prod = compute_labor_productivity(rows, entity_col, output_col, hours_col)
    if prod is None:
        return {"error": "Insufficient data"}

    return {
        "mean_productivity": prod.mean_productivity,
        "best_entity": prod.best_entity,
        "worst_entity": prod.worst_entity,
        "spread_ratio": prod.spread_ratio,
        "summary": prod.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_output": e.total_output,
                "total_hours": e.total_hours,
                "productivity": e.productivity,
                "productivity_index": e.productivity_index,
            }
            for e in prod.entities
        ],
    }


# ---------------------------------------------------------------------------
# Logistics Tracking
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/delivery-performance")
async def delivery_performance(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze delivery performance (on-time rate).

    Body: {"entity_column": "supplier", "promised_column": "promised_date", "actual_column": "delivery_date"}
    """
    from business_brain.discovery.logistics_tracker import analyze_delivery_performance

    entity_col = body.get("entity_column")
    promised_col = body.get("promised_column")
    actual_col = body.get("actual_column")

    if not all([entity_col, promised_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, promised_column, actual_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    dp = analyze_delivery_performance(rows, entity_col, promised_col, actual_col)
    if dp is None:
        return {"error": "Insufficient data"}

    return {
        "total_deliveries": dp.total_deliveries,
        "on_time_count": dp.on_time_count,
        "on_time_rate": dp.on_time_rate,
        "avg_delay": dp.avg_delay,
        "worst_entity": dp.worst_entity,
        "best_entity": dp.best_entity,
        "summary": dp.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_deliveries": e.total_deliveries,
                "on_time_count": e.on_time_count,
                "late_count": e.late_count,
                "early_count": e.early_count,
                "on_time_rate": e.on_time_rate,
                "avg_delay": e.avg_delay,
            }
            for e in dp.entities
        ],
    }


@app.post("/tables/{table_name}/vehicle-utilization")
async def vehicle_utilization(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute vehicle load utilization.

    Body: {"vehicle_column": "truck_id", "capacity_column": "capacity_tons", "load_column": "actual_load_tons"}
    """
    from business_brain.discovery.logistics_tracker import compute_vehicle_utilization

    vehicle_col = body.get("vehicle_column")
    capacity_col = body.get("capacity_column")
    load_col = body.get("load_column")

    if not all([vehicle_col, capacity_col, load_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vehicle_column, capacity_column, load_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    vu = compute_vehicle_utilization(rows, vehicle_col, capacity_col, load_col)
    if vu is None:
        return {"error": "Insufficient data"}

    return {
        "mean_utilization": vu.mean_utilization,
        "underloaded_count": vu.underloaded_count,
        "overloaded_count": vu.overloaded_count,
        "summary": vu.summary,
        "vehicles": [
            {
                "vehicle": v.vehicle,
                "total_trips": v.total_trips,
                "avg_load": v.avg_load,
                "avg_capacity": v.avg_capacity,
                "utilization_pct": v.utilization_pct,
                "status": v.status,
            }
            for v in vu.vehicles
        ],
    }


# ---------------------------------------------------------------------------
# Power & Energy Monitoring
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/load-profile")
async def load_profile(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze electrical load profile.

    Body: {"time_column": "timestamp", "power_column": "kw_demand"}
    """
    from business_brain.discovery.power_monitor import analyze_load_profile

    time_col = body.get("time_column")
    power_col = body.get("power_column")

    if not time_col or not power_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "time_column and power_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    lp = analyze_load_profile(rows, time_col, power_col)
    if lp is None:
        return {"error": "Insufficient data"}

    return {
        "peak_demand": lp.peak_demand,
        "avg_demand": lp.avg_demand,
        "min_demand": lp.min_demand,
        "load_factor": lp.load_factor,
        "peak_period": lp.peak_period,
        "off_peak_period": lp.off_peak_period,
        "summary": lp.summary,
        "periods": [
            {
                "period": p.period,
                "demand": p.demand,
                "pct_of_peak": p.pct_of_peak,
                "classification": p.classification,
            }
            for p in lp.periods
        ],
    }


@app.post("/tables/{table_name}/power-factor")
async def power_factor_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze power factor across entities.

    Body: {"entity_column": "transformer", "kw_column": "kw", "kva_column": "kva"}
    """
    from business_brain.discovery.power_monitor import analyze_power_factor

    entity_col = body.get("entity_column")
    kw_col = body.get("kw_column")
    kva_col = body.get("kva_column")

    if not all([entity_col, kw_col, kva_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, kw_column, kva_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    pf = analyze_power_factor(rows, entity_col, kw_col, kva_col)
    if pf is None:
        return {"error": "Insufficient data"}

    return {
        "mean_pf": pf.mean_pf,
        "penalty_risk_count": pf.penalty_risk_count,
        "excellent_count": pf.excellent_count,
        "summary": pf.summary,
        "entities": [
            {
                "entity": e.entity,
                "avg_kw": e.avg_kw,
                "avg_kva": e.avg_kva,
                "power_factor": e.power_factor,
                "status": e.status,
                "estimated_loss_pct": e.estimated_loss_pct,
            }
            for e in pf.entities
        ],
    }


# ---------------------------------------------------------------------------
# Rate Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/rate-comparison")
async def rate_comparison(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare rates across suppliers.

    Body: {"supplier_column": "supplier", "rate_column": "rate", "item_column": "material"}
    """
    from business_brain.discovery.rate_analysis import compare_rates

    supplier_col = body.get("supplier_column")
    rate_col = body.get("rate_column")
    item_col = body.get("item_column")

    if not supplier_col or not rate_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and rate_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rc = compare_rates(rows, supplier_col, rate_col, item_col)
    if rc is None:
        return {"error": "Insufficient data"}

    return {
        "overall_savings_potential": rc.overall_savings_potential,
        "best_rate_supplier": rc.best_rate_supplier,
        "worst_rate_supplier": rc.worst_rate_supplier,
        "rate_spread_pct": rc.rate_spread_pct,
        "summary": rc.summary,
        "comparisons": [
            {
                "item": c.item,
                "best_supplier": c.best_supplier,
                "worst_supplier": c.worst_supplier,
                "spread": c.spread,
                "spread_pct": c.spread_pct,
                "suppliers": [
                    {"supplier": s.supplier, "avg_rate": s.avg_rate, "volume": s.volume, "total_value": s.total_value}
                    for s in c.suppliers
                ],
            }
            for c in rc.comparisons
        ],
    }


@app.post("/tables/{table_name}/rate-anomalies")
async def rate_anomalies(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Detect rate anomalies.

    Body: {"supplier_column": "supplier", "rate_column": "rate", "item_column": null, "threshold_pct": 20}
    """
    from business_brain.discovery.rate_analysis import detect_rate_anomalies

    supplier_col = body.get("supplier_column")
    rate_col = body.get("rate_column")
    item_col = body.get("item_column")
    threshold = body.get("threshold_pct", 20)

    if not supplier_col or not rate_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and rate_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    anomalies = detect_rate_anomalies(rows, supplier_col, rate_col, item_col, threshold)

    return {
        "anomaly_count": len(anomalies),
        "anomalies": [
            {
                "supplier": a.supplier,
                "item": a.item,
                "rate": a.rate,
                "avg_rate": a.avg_rate,
                "deviation_pct": a.deviation_pct,
                "anomaly_type": a.anomaly_type,
                "severity": a.severity,
            }
            for a in anomalies
        ],
    }


# ---------------------------------------------------------------------------
# Safety & Compliance
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/incidents")
async def incident_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze safety incidents.

    Body: {"type_column": "incident_type", "severity_column": "severity", "date_column": "date", "location_column": "area"}
    """
    from business_brain.discovery.safety_compliance import analyze_incidents

    type_col = body.get("type_column")
    severity_col = body.get("severity_column")
    date_col = body.get("date_column")
    location_col = body.get("location_column")

    if not type_col or not severity_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "type_column and severity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    inc = analyze_incidents(rows, type_col, severity_col, date_col, location_col)
    if inc is None:
        return {"error": "Insufficient data"}

    return {
        "total_incidents": inc.total_incidents,
        "by_type": inc.by_type,
        "by_severity": inc.by_severity,
        "by_location": inc.by_location,
        "trend": inc.trend,
        "most_common_type": inc.most_common_type,
        "most_common_location": inc.most_common_location,
        "summary": inc.summary,
    }


@app.post("/tables/{table_name}/compliance")
async def compliance_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute compliance rates.

    Body: {"entity_column": "department", "total_checks_column": "total_audits", "passed_checks_column": "passed_audits"}
    """
    from business_brain.discovery.safety_compliance import compliance_rate

    entity_col = body.get("entity_column")
    total_col = body.get("total_checks_column")
    passed_col = body.get("passed_checks_column")

    if not all([entity_col, total_col, passed_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, total_checks_column, passed_checks_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cr = compliance_rate(rows, entity_col, total_col, passed_col)
    if cr is None:
        return {"error": "Insufficient data"}

    return {
        "mean_compliance": cr.mean_compliance,
        "fully_compliant_count": cr.fully_compliant_count,
        "non_compliant_count": cr.non_compliant_count,
        "summary": cr.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_checks": e.total_checks,
                "passed_checks": e.passed_checks,
                "compliance_pct": e.compliance_pct,
                "status": e.status,
            }
            for e in cr.entities
        ],
    }


@app.post("/tables/{table_name}/risk-matrix")
async def risk_matrix_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute risk matrix.

    Body: {"likelihood_column": "probability", "impact_column": "impact", "entity_column": "risk_item"}
    """
    from business_brain.discovery.safety_compliance import risk_matrix

    likelihood_col = body.get("likelihood_column")
    impact_col = body.get("impact_column")
    entity_col = body.get("entity_column")

    if not likelihood_col or not impact_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "likelihood_column and impact_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rm = risk_matrix(rows, likelihood_col, impact_col, entity_col)
    if rm is None:
        return {"error": "Insufficient data"}

    return {
        "critical_count": rm.critical_count,
        "high_count": rm.high_count,
        "medium_count": rm.medium_count,
        "low_count": rm.low_count,
        "summary": rm.summary,
        "items": [
            {
                "entity": i.entity,
                "likelihood": i.likelihood,
                "impact": i.impact,
                "risk_score": i.risk_score,
                "risk_level": i.risk_level,
            }
            for i in rm.items
        ],
    }


# ---------------------------------------------------------------------------
# Quality Control (SPC)
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/process-capability")
async def process_capability(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute process capability indices (Cp/Cpk).

    Body: {"column": "measurement", "lsl": 9.5, "usl": 10.5}
    """
    from business_brain.discovery.quality_control import compute_process_capability

    column = body.get("column")
    lsl = body.get("lsl")
    usl = body.get("usl")

    if not column or lsl is None or usl is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column, lsl, usl required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT "{column}" FROM "{table_name}" LIMIT 10000'))
    values = []
    for row in result.fetchall():
        try:
            values.append(float(str(row[0]).replace(",", "")))
        except (ValueError, TypeError):
            pass

    cap = compute_process_capability(values, float(lsl), float(usl))
    if cap is None:
        return {"error": "Insufficient data (need >=2 values)"}

    return {
        "cp": cap.cp,
        "cpk": cap.cpk,
        "mean": cap.mean,
        "std": cap.std,
        "lsl": cap.lsl,
        "usl": cap.usl,
        "ppm_out_of_spec": cap.ppm_out_of_spec,
        "process_grade": cap.process_grade,
        "centered": cap.centered,
        "summary": cap.summary,
    }


@app.post("/tables/{table_name}/control-chart")
async def control_chart(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Generate control chart data.

    Body: {"column": "measurement", "subgroup_size": 1}
    """
    from business_brain.discovery.quality_control import control_chart_data

    column = body.get("column")
    subgroup_size = body.get("subgroup_size", 1)

    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT "{column}" FROM "{table_name}" LIMIT 10000'))
    values = []
    for row in result.fetchall():
        try:
            values.append(float(str(row[0]).replace(",", "")))
        except (ValueError, TypeError):
            pass

    cc = control_chart_data(values, subgroup_size)
    if cc is None:
        return {"error": "Insufficient data"}

    return {
        "mean": cc.mean,
        "ucl": cc.ucl,
        "lcl": cc.lcl,
        "out_of_control_count": cc.out_of_control_count,
        "out_of_control_indices": cc.out_of_control_indices,
        "in_control_pct": cc.in_control_pct,
        "summary": cc.summary,
        "values": cc.values[-100:],  # limit to last 100
    }


@app.post("/tables/{table_name}/defects")
async def defect_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze defect rates per entity.

    Body: {"entity_column": "line", "defect_column": "defect_count", "quantity_column": "total_produced"}
    """
    from business_brain.discovery.quality_control import analyze_defects

    entity_col = body.get("entity_column")
    defect_col = body.get("defect_column")
    quantity_col = body.get("quantity_column")

    if not entity_col or not defect_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column and defect_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    df = analyze_defects(rows, entity_col, defect_col, quantity_col)
    if df is None:
        return {"error": "Insufficient data"}

    return {
        "total_defects": df.total_defects,
        "total_quantity": df.total_quantity,
        "overall_defect_rate": df.overall_defect_rate,
        "worst_entity": df.worst_entity,
        "best_entity": df.best_entity,
        "summary": df.summary,
        "entities": [
            {
                "entity": e.entity,
                "defect_count": e.defect_count,
                "quantity": e.quantity,
                "defect_rate": e.defect_rate,
                "dpmo": e.dpmo,
            }
            for e in df.entities
        ],
    }


@app.post("/tables/{table_name}/rejections")
async def rejection_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze rejection rates.

    Body: {"entity_column": "supplier", "accepted_column": "accepted_qty", "rejected_column": "rejected_qty"}
    """
    from business_brain.discovery.quality_control import analyze_rejections

    entity_col = body.get("entity_column")
    accepted_col = body.get("accepted_column")
    rejected_col = body.get("rejected_column")

    if not all([entity_col, accepted_col, rejected_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, accepted_column, rejected_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rj = analyze_rejections(rows, entity_col, accepted_col, rejected_col)
    if rj is None:
        return {"error": "Insufficient data"}

    return {
        "total_accepted": rj.total_accepted,
        "total_rejected": rj.total_rejected,
        "overall_rejection_rate": rj.overall_rejection_rate,
        "worst_entity": rj.worst_entity,
        "best_entity": rj.best_entity,
        "summary": rj.summary,
        "entities": [
            {
                "entity": e.entity,
                "accepted": e.accepted,
                "rejected": e.rejected,
                "total": e.total,
                "rejection_rate": e.rejection_rate,
            }
            for e in rj.entities
        ],
    }


# ---------------------------------------------------------------------------
# RFM Customer Analysis
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/rfm")
async def rfm_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute RFM segmentation.

    Body: {"customer_column": "customer", "date_column": "order_date", "amount_column": "amount"}
    """
    from business_brain.discovery.rfm_analysis import compute_rfm, segment_customers

    customer_col = body.get("customer_column")
    date_col = body.get("date_column")
    amount_col = body.get("amount_column")

    if not all([customer_col, date_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, date_column, amount_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rfm = compute_rfm(rows, customer_col, date_col, amount_col)
    if rfm is None:
        return {"error": "Insufficient data"}

    segments = segment_customers(rfm)

    return {
        "total_customers": rfm.total_customers,
        "avg_recency": rfm.avg_recency,
        "avg_frequency": rfm.avg_frequency,
        "avg_monetary": rfm.avg_monetary,
        "summary": rfm.summary,
        "segment_distribution": rfm.segment_distribution,
        "segments": {k: {"count": len(v), "customers": v[:10]} for k, v in segments.items()},
        "customers": [
            {
                "customer": c.customer,
                "recency_days": c.recency_days,
                "frequency": c.frequency,
                "monetary": c.monetary,
                "r_score": c.r_score,
                "f_score": c.f_score,
                "m_score": c.m_score,
                "rfm_score": c.rfm_score,
                "segment": c.segment,
            }
            for c in rfm.customers[:50]  # limit to 50
        ],
    }


@app.post("/tables/{table_name}/clv")
async def clv_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute Customer Lifetime Value.

    Body: {"customer_column": "customer", "amount_column": "amount", "date_column": "order_date"}
    """
    from business_brain.discovery.rfm_analysis import customer_lifetime_value

    customer_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")

    if not all([customer_col, amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, date_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    clv = customer_lifetime_value(rows, customer_col, amount_col, date_col)
    if clv is None:
        return {"error": "Insufficient data"}

    return {
        "avg_clv": clv.avg_clv,
        "top_customers": clv.top_customers,
        "summary": clv.summary,
        "customers": [
            {
                "customer": c.customer,
                "total_spent": c.total_spent,
                "purchase_count": c.purchase_count,
                "avg_purchase": c.avg_purchase,
                "lifespan_days": c.lifespan_days,
                "estimated_clv": c.estimated_clv,
            }
            for c in clv.customers[:50]
        ],
    }


@app.post("/tables/{table_name}/heat-analysis")
async def heat_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze heats: count, weight stats, grade distribution.
    Body: {"heat_column": "...", "weight_column": "...", "grade_column": null, "time_column": null}
    """
    from business_brain.discovery.heat_analysis import analyze_heats
    heat_col = body.get("heat_column")
    weight_col = body.get("weight_column")
    if not all([heat_col, weight_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "heat_column and weight_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_heats(rows, heat_col, weight_col, body.get("grade_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_heats": res.total_heats,
        "total_weight": res.total_weight,
        "avg_weight": res.avg_weight,
        "min_weight": res.min_weight,
        "max_weight": res.max_weight,
        "std_weight": res.std_weight,
        "grade_distribution": res.grade_distribution,
        "period_breakdown": res.period_breakdown,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/chemistry-analysis")
async def chemistry_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze chemical composition per heat.
    Body: {"heat_column": "...", "element_columns": ["C","Mn","Si"], "specs": null}
    """
    from business_brain.discovery.heat_analysis import analyze_chemistry
    heat_col = body.get("heat_column")
    elem_cols = body.get("element_columns")
    if not heat_col or not elem_cols:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "heat_column and element_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    specs_raw = body.get("specs")
    specs = None
    if specs_raw and isinstance(specs_raw, dict):
        specs = {k: tuple(v) for k, v in specs_raw.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    res = analyze_chemistry(rows, heat_col, elem_cols, specs)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_heats": res.total_heats,
        "elements": [
            {
                "element": e.element,
                "mean": e.mean,
                "std": e.std,
                "min": e.min_val,
                "max": e.max_val,
                "in_spec_pct": e.in_spec_pct,
                "out_of_spec_count": e.out_of_spec_count,
            }
            for e in res.elements
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/grade-analysis")
async def grade_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze production by steel grade.
    Body: {"grade_column": "...", "weight_column": "...", "value_column": null}
    """
    from business_brain.discovery.heat_analysis import grade_wise_analysis
    grade_col = body.get("grade_column")
    weight_col = body.get("weight_column")
    if not all([grade_col, weight_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "grade_column and weight_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = grade_wise_analysis(rows, grade_col, weight_col, body.get("value_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_weight": res.total_weight,
        "grades": [
            {
                "grade": g.grade,
                "heat_count": g.heat_count,
                "total_weight": g.total_weight,
                "pct_of_total": g.pct_of_total,
                "avg_weight": g.avg_weight,
                "total_value": g.total_value,
            }
            for g in res.grades
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/grade-anomalies")
async def grade_anomalies(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect heats where chemistry doesn't match grade spec.
    Body: {"heat_column": "...", "grade_column": "...", "element_columns": ["C","Mn"], "specs": null}
    """
    from business_brain.discovery.heat_analysis import detect_grade_anomalies
    heat_col = body.get("heat_column")
    grade_col = body.get("grade_column")
    elem_cols = body.get("element_columns")
    if not all([heat_col, grade_col, elem_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "heat_column, grade_column, element_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    specs_raw = body.get("specs")
    specs = None
    if specs_raw and isinstance(specs_raw, dict):
        specs = {}
        for grade, elems in specs_raw.items():
            if isinstance(elems, dict):
                specs[grade] = {e: tuple(v) for e, v in elems.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    anomalies = detect_grade_anomalies(rows, heat_col, grade_col, elem_cols, specs)
    return {
        "anomaly_count": len(anomalies),
        "anomalies": [
            {
                "heat": a.heat,
                "grade": a.grade,
                "element": a.element,
                "value": a.value,
                "spec_range": list(a.spec_range),
                "deviation_pct": a.deviation_pct,
            }
            for a in anomalies[:100]
        ],
    }


@app.post("/tables/{table_name}/gate-traffic")
async def gate_traffic(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze gate traffic patterns.
    Body: {"time_column": "...", "vehicle_column": null, "direction_column": null}
    """
    from business_brain.discovery.dispatch_gate import analyze_gate_traffic
    time_col = body.get("time_column")
    if not time_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "time_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_gate_traffic(rows, time_col, body.get("vehicle_column"), body.get("direction_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_vehicles": res.total_vehicles,
        "periods": [{"period": p.period, "vehicle_count": p.vehicle_count, "pct_of_total": p.pct_of_total} for p in res.periods],
        "peak_period": res.peak_period,
        "off_peak_period": res.off_peak_period,
        "avg_per_period": res.avg_per_period,
        "direction_split": res.direction_split,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/weighbridge")
async def weighbridge(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze weighbridge data.
    Body: {"vehicle_column": "...", "gross_column": "...", "tare_column": "...", "material_column": null}
    """
    from business_brain.discovery.dispatch_gate import weighbridge_analysis
    vehicle_col = body.get("vehicle_column")
    gross_col = body.get("gross_column")
    tare_col = body.get("tare_column")
    if not all([vehicle_col, gross_col, tare_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vehicle_column, gross_column, tare_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = weighbridge_analysis(rows, vehicle_col, gross_col, tare_col, body.get("material_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_net_weight": res.total_net_weight,
        "avg_net_weight": res.avg_net_weight,
        "total_vehicles": res.total_vehicles,
        "by_material": res.by_material,
        "entries": [
            {"vehicle": e.vehicle, "gross_weight": e.gross_weight, "tare_weight": e.tare_weight, "net_weight": e.net_weight, "material": e.material}
            for e in res.entries[:50]
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/material-movement")
async def material_movement(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Track inward/outward material movement.
    Body: {"material_column": "...", "quantity_column": "...", "direction_column": "..."}
    """
    from business_brain.discovery.dispatch_gate import track_material_movement
    mat_col = body.get("material_column")
    qty_col = body.get("quantity_column")
    dir_col = body.get("direction_column")
    if not all([mat_col, qty_col, dir_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "material_column, quantity_column, direction_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = track_material_movement(rows, mat_col, qty_col, dir_col, body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_inward": res.total_inward,
        "total_outward": res.total_outward,
        "net_movement": res.net_movement,
        "materials": [
            {"material": m.material, "inward_qty": m.inward_qty, "outward_qty": m.outward_qty, "net_qty": m.net_qty, "movement_count": m.movement_count}
            for m in res.materials
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/dispatch-anomalies")
async def dispatch_anomalies(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Flag vehicles with unusual dispatch weights.
    Body: {"vehicle_column": "...", "weight_column": "...", "expected_min": null, "expected_max": null}
    """
    from business_brain.discovery.dispatch_gate import detect_dispatch_anomalies
    vehicle_col = body.get("vehicle_column")
    weight_col = body.get("weight_column")
    if not all([vehicle_col, weight_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vehicle_column and weight_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    anomalies = detect_dispatch_anomalies(rows, vehicle_col, weight_col, body.get("expected_min"), body.get("expected_max"))
    return {
        "anomaly_count": len(anomalies),
        "anomalies": [
            {"vehicle": a.vehicle, "weight": a.weight, "expected_range": list(a.expected_range), "deviation_pct": a.deviation_pct, "anomaly_type": a.anomaly_type}
            for a in anomalies[:100]
        ],
    }


@app.post("/tables/{table_name}/emissions")
async def emissions_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze emissions data.
    Body: {"source_column": "...", "pollutant_column": "...", "value_column": "...", "limit_column": null, "time_column": null}
    """
    from business_brain.discovery.environmental_monitor import analyze_emissions
    src_col = body.get("source_column")
    poll_col = body.get("pollutant_column")
    val_col = body.get("value_column")
    if not all([src_col, poll_col, val_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "source_column, pollutant_column, value_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_emissions(rows, src_col, poll_col, val_col, body.get("limit_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_readings": res.total_readings,
        "sources": [
            {"source": s.source, "pollutant": s.pollutant, "total": s.total, "average": s.average, "max_value": s.max_value, "count": s.count, "compliance_pct": s.compliance_pct, "trend": s.trend}
            for s in res.sources
        ],
        "exceedances": [
            {"source": e.source, "pollutant": e.pollutant, "value": e.value, "limit": e.limit, "excess_pct": e.excess_pct}
            for e in res.exceedances[:50]
        ],
        "overall_compliance_pct": res.overall_compliance_pct,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/waste-analysis")
async def waste_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze waste generation.
    Body: {"waste_type_column": "...", "quantity_column": "...", "disposal_column": null}
    """
    from business_brain.discovery.environmental_monitor import analyze_waste_generation
    wtype_col = body.get("waste_type_column")
    qty_col = body.get("quantity_column")
    if not all([wtype_col, qty_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "waste_type_column and quantity_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_waste_generation(rows, wtype_col, qty_col, body.get("disposal_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_waste": res.total_waste,
        "types": [
            {"waste_type": t.waste_type, "quantity": t.quantity, "pct_of_total": t.pct_of_total, "disposal_breakdown": t.disposal_breakdown}
            for t in res.types
        ],
        "recycling_rate": res.recycling_rate,
        "diversion_rate": res.diversion_rate,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/water-usage")
async def water_usage(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze water usage.
    Body: {"source_column": "...", "consumption_column": "...", "discharge_column": null}
    """
    from business_brain.discovery.environmental_monitor import analyze_water_usage
    src_col = body.get("source_column")
    cons_col = body.get("consumption_column")
    if not all([src_col, cons_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "source_column and consumption_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_water_usage(rows, src_col, cons_col, body.get("discharge_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_consumption": res.total_consumption,
        "total_discharge": res.total_discharge,
        "net_consumption": res.net_consumption,
        "recycling_ratio": res.recycling_ratio,
        "sources": [
            {"source": s.source, "consumption": s.consumption, "discharge": s.discharge, "pct_of_total": s.pct_of_total}
            for s in res.sources
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/maintenance-history")
async def maintenance_history(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze equipment maintenance history.
    Body: {"equipment_column": "...", "date_column": "...", "type_column": "...", "duration_column": null, "cost_column": null}
    """
    from business_brain.discovery.maintenance_scheduler import analyze_maintenance_history
    equip_col = body.get("equipment_column")
    date_col = body.get("date_column")
    type_col = body.get("type_column")
    if not all([equip_col, date_col, type_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "equipment_column, date_column, type_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_maintenance_history(rows, equip_col, date_col, type_col, body.get("duration_column"), body.get("cost_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_events": res.total_events,
        "equipment": [
            {"equipment": e.equipment, "event_count": e.event_count, "type_breakdown": e.type_breakdown, "corrective_ratio": e.corrective_ratio, "total_downtime": e.total_downtime, "total_cost": e.total_cost, "avg_cost": e.avg_cost}
            for e in res.equipment
        ],
        "overall_corrective_ratio": res.overall_corrective_ratio,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/mtbf-mttr")
async def mtbf_mttr(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute MTBF and MTTR for equipment.
    Body: {"equipment_column": "...", "date_column": "...", "type_column": "...", "duration_column": "..."}
    """
    from business_brain.discovery.maintenance_scheduler import compute_mtbf_mttr
    equip_col = body.get("equipment_column")
    date_col = body.get("date_column")
    type_col = body.get("type_column")
    dur_col = body.get("duration_column")
    if not all([equip_col, date_col, type_col, dur_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "equipment_column, date_column, type_column, duration_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    metrics = compute_mtbf_mttr(rows, equip_col, date_col, type_col, dur_col)
    return {
        "metrics": [
            {"equipment": m.equipment, "mtbf_days": m.mtbf_days, "mttr_hours": m.mttr_hours, "failure_count": m.failure_count, "availability_pct": m.availability_pct}
            for m in metrics
        ],
    }


@app.post("/tables/{table_name}/spare-parts")
async def spare_parts(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze spare parts consumption.
    Body: {"part_column": "...", "quantity_column": "...", "cost_column": null, "equipment_column": null}
    """
    from business_brain.discovery.maintenance_scheduler import analyze_spare_parts
    part_col = body.get("part_column")
    qty_col = body.get("quantity_column")
    if not all([part_col, qty_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "part_column and quantity_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_spare_parts(rows, part_col, qty_col, body.get("cost_column"), body.get("equipment_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_parts": res.total_parts,
        "total_quantity": res.total_quantity,
        "total_cost": res.total_cost,
        "parts": [
            {"part": p.part, "total_quantity": p.total_quantity, "total_cost": p.total_cost, "abc_class": p.abc_class, "equipment_list": p.equipment_list}
            for p in res.parts[:50]
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/maintenance-schedule")
async def maintenance_schedule(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Generate predicted maintenance schedule.
    Body: {"equipment_column": "...", "date_column": "...", "type_column": "...", "interval_days": null}
    """
    from business_brain.discovery.maintenance_scheduler import generate_maintenance_schedule
    equip_col = body.get("equipment_column")
    date_col = body.get("date_column")
    type_col = body.get("type_column")
    if not all([equip_col, date_col, type_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "equipment_column, date_column, type_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    schedule = generate_maintenance_schedule(rows, equip_col, date_col, type_col, body.get("interval_days"))
    return {
        "schedule": [
            {"equipment": s.equipment, "last_maintenance": s.last_maintenance, "avg_interval_days": s.avg_interval_days, "next_maintenance_date": s.next_maintenance_date, "days_until_next": s.days_until_next, "overdue": s.overdue, "priority": s.priority}
            for s in schedule
        ],
    }


@app.post("/tables/{table_name}/contracts")
async def contracts_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze contract portfolio.
    Body: {"contract_column": "...", "vendor_column": "...", "value_column": "...", "start_column": null, "end_column": null, "status_column": null}
    """
    from business_brain.discovery.contract_analyzer import analyze_contracts
    contract_col = body.get("contract_column")
    vendor_col = body.get("vendor_column")
    value_col = body.get("value_column")
    if not all([contract_col, vendor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "contract_column, vendor_column, value_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_contracts(rows, contract_col, vendor_col, value_col, body.get("start_column"), body.get("end_column"), body.get("status_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_contracts": res.total_contracts,
        "total_value": res.total_value,
        "avg_value": res.avg_value,
        "vendor_count": res.vendor_count,
        "vendors": [
            {"vendor": v.vendor, "contract_count": v.contract_count, "total_value": v.total_value, "avg_value": v.avg_value}
            for v in res.vendors[:50]
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/expiring-contracts")
async def expiring_contracts(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Find contracts expiring soon.
    Body: {"contract_column": "...", "end_column": "...", "vendor_column": null, "value_column": null, "horizon_days": 90}
    """
    from business_brain.discovery.contract_analyzer import detect_expiring_contracts
    contract_col = body.get("contract_column")
    end_col = body.get("end_column")
    if not all([contract_col, end_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "contract_column and end_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    horizon = body.get("horizon_days", 90)
    expiring = detect_expiring_contracts(rows, contract_col, end_col, body.get("vendor_column"), body.get("value_column"), horizon_days=horizon)
    return {
        "expiring_count": len(expiring),
        "contracts": [
            {"contract": c.contract, "vendor": c.vendor, "end_date": c.end_date, "days_remaining": c.days_remaining, "value": c.value, "urgency": c.urgency}
            for c in expiring
        ],
    }


@app.post("/tables/{table_name}/vendor-concentration")
async def vendor_concentration(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Measure vendor concentration (HHI).
    Body: {"vendor_column": "...", "value_column": "..."}
    """
    from business_brain.discovery.contract_analyzer import vendor_contract_concentration
    vendor_col = body.get("vendor_column")
    value_col = body.get("value_column")
    if not all([vendor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vendor_column and value_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = vendor_contract_concentration(rows, vendor_col, value_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "hhi": res.hhi,
        "risk_rating": res.risk_rating,
        "vendor_count": res.vendor_count,
        "shares": [
            {"vendor": s.vendor, "share_pct": s.share_pct, "total_value": s.total_value}
            for s in res.shares
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/renewal-patterns")
async def renewal_patterns(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect contract renewal patterns.
    Body: {"contract_column": "...", "vendor_column": "...", "start_column": "...", "end_column": "...", "value_column": null}
    """
    from business_brain.discovery.contract_analyzer import analyze_renewal_patterns
    contract_col = body.get("contract_column")
    vendor_col = body.get("vendor_column")
    start_col = body.get("start_column")
    end_col = body.get("end_column")
    if not all([contract_col, vendor_col, start_col, end_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "contract_column, vendor_column, start_column, end_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_renewal_patterns(rows, contract_col, vendor_col, start_col, end_col, body.get("value_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_vendors": res.total_vendors,
        "vendors_with_renewals": res.vendors_with_renewals,
        "renewal_rate": res.renewal_rate,
        "renewals": [
            {"vendor": r.vendor, "renewal_count": r.renewal_count, "avg_contract_duration_days": r.avg_contract_duration_days, "value_trend": r.value_trend}
            for r in res.renewals
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/budget-vs-actual")
async def budget_vs_actual_endpoint(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Budget vs actual analysis.
    Body: {"category_column": "...", "budget_column": "...", "actual_column": "...", "period_column": null}
    """
    from business_brain.discovery.budget_tracker import budget_vs_actual
    cat_col = body.get("category_column")
    bud_col = body.get("budget_column")
    act_col = body.get("actual_column")
    if not all([cat_col, bud_col, act_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column, budget_column, actual_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = budget_vs_actual(rows, cat_col, bud_col, act_col, body.get("period_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_budget": res.total_budget,
        "total_actual": res.total_actual,
        "overall_variance_pct": res.overall_variance_pct,
        "categories": [
            {"category": c.category, "budget": c.budget, "actual": c.actual, "variance": c.variance, "variance_pct": c.variance_pct, "over_budget": c.over_budget}
            for c in res.categories
        ],
        "over_budget_count": res.over_budget_count,
        "periods": [
            {"period": p.period, "budget": p.budget, "actual": p.actual, "variance_pct": p.variance_pct}
            for p in (res.periods or [])
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/burn-rate")
async def burn_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute budget burn rate.
    Body: {"amount_column": "...", "date_column": "...", "total_budget": null}
    """
    from business_brain.discovery.budget_tracker import compute_burn_rate
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_burn_rate(rows, amount_col, date_col, body.get("total_budget"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_spent": res.total_spent,
        "days_elapsed": res.days_elapsed,
        "daily_burn": res.daily_burn,
        "monthly_burn": res.monthly_burn,
        "remaining_budget": res.remaining_budget,
        "days_until_exhaustion": res.days_until_exhaustion,
        "projected_end_date": res.projected_end_date,
        "trend": res.trend,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/spending-patterns")
async def spending_patterns(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze spending patterns.
    Body: {"category_column": "...", "amount_column": "...", "date_column": null, "vendor_column": null}
    """
    from business_brain.discovery.budget_tracker import analyze_spending_patterns
    cat_col = body.get("category_column")
    amt_col = body.get("amount_column")
    if not all([cat_col, amt_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column and amount_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_spending_patterns(rows, cat_col, amt_col, body.get("date_column"), body.get("vendor_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_spend": res.total_spend,
        "categories": [
            {"category": c.category, "amount": c.amount, "pct_of_total": c.pct_of_total}
            for c in res.categories
        ],
        "top_categories": res.top_categories,
        "vendors": [
            {"vendor": v.vendor, "amount": v.amount, "pct_of_total": v.pct_of_total}
            for v in (res.vendors or [])
        ] if res.vendors else None,
        "month_changes": [
            {"category": m.category, "month": m.month, "amount": m.amount, "prev_amount": m.prev_amount, "change_pct": m.change_pct}
            for m in (res.month_changes or [])
        ] if res.month_changes else None,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/budget-forecast")
async def budget_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast future budget periods.
    Body: {"amount_column": "...", "date_column": "...", "periods_ahead": 3}
    """
    from business_brain.discovery.budget_tracker import forecast_budget
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    periods = body.get("periods_ahead", 3)
    forecasts = forecast_budget(rows, amount_col, date_col, periods)
    return {
        "forecasts": [
            {"period": f.period, "projected_amount": f.projected_amount, "cumulative": f.cumulative, "confidence": f.confidence}
            for f in forecasts
        ],
    }


@app.post("/tables/{table_name}/demand-pattern")
async def demand_pattern(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze demand patterns.
    Body: {"product_column": "...", "quantity_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.demand_forecaster import analyze_demand_pattern
    prod_col = body.get("product_column")
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([prod_col, qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "product_column, quantity_column, date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_demand_pattern(rows, prod_col, qty_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "products": [
            {"product": p.product, "total_demand": p.total_demand, "avg_demand": p.avg_demand, "max_demand": p.max_demand, "min_demand": p.min_demand, "cv": p.cv, "pattern": p.pattern, "trend": p.trend, "adi": p.adi, "periods": p.periods}
            for p in res.products
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/demand-moving-avg")
async def demand_moving_avg(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute moving average on demand data.
    Body: {"quantity_column": "...", "date_column": "...", "window": 3}
    """
    from business_brain.discovery.demand_forecaster import compute_moving_average
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "quantity_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    window = body.get("window", 3)
    res = compute_moving_average(rows, qty_col, date_col, window)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "window": res.window,
        "points": [
            {"period": p.period, "actual": p.actual, "moving_avg": p.moving_avg}
            for p in res.points
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/demand-smoothing")
async def demand_smoothing(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Exponential smoothing on demand data.
    Body: {"quantity_column": "...", "date_column": "...", "alpha": 0.3}
    """
    from business_brain.discovery.demand_forecaster import exponential_smoothing
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "quantity_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    alpha = body.get("alpha", 0.3)
    res = exponential_smoothing(rows, qty_col, date_col, alpha)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "alpha_used": res.alpha_used,
        "optimal_alpha": res.optimal_alpha,
        "mad": res.mad,
        "points": [
            {"period": p.period, "actual": p.actual, "forecast": p.forecast}
            for p in res.points
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/demand-seasonality")
async def demand_seasonality(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect seasonal patterns in demand.
    Body: {"quantity_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.demand_forecaster import detect_demand_seasonality
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "quantity_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = detect_demand_seasonality(rows, qty_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "is_seasonal": res.is_seasonal,
        "peak_seasons": res.peak_seasons,
        "low_seasons": res.low_seasons,
        "periods": [
            {"period": p.period, "avg_demand": p.avg_demand, "seasonal_index": p.seasonal_index}
            for p in res.periods
        ],
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Procurement Analytics endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/purchase-orders")
async def purchase_orders(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze purchase orders by vendor with monthly trends.
    Body: {"po_column": "...", "vendor_column": "...", "amount_column": "...", "date_column": "...", "status_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import analyze_purchase_orders
    po_col = body.get("po_column")
    vendor_col = body.get("vendor_column")
    amount_col = body.get("amount_column")
    if not all([po_col, vendor_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "po_column, vendor_column, and amount_column required"}, 400)
    date_col = body.get("date_column")
    status_col = body.get("status_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_purchase_orders(rows, po_col, vendor_col, amount_col, date_col, status_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_orders": res.total_orders,
        "total_value": res.total_value,
        "vendors": [
            {"vendor": v.vendor, "order_count": v.order_count, "total_value": v.total_value, "avg_value": v.avg_value}
            for v in res.vendors
        ],
        "monthly_trends": [
            {"month": m.month, "order_count": m.order_count, "total_value": m.total_value}
            for m in res.monthly_trends
        ],
        "status_breakdown": res.status_breakdown,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/purchase-price-variance")
async def purchase_price_variance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute purchase price variance per item.
    Body: {"item_column": "...", "quantity_column": "...", "actual_price_column": "...", "standard_price_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import compute_purchase_price_variance
    item_col = body.get("item_column")
    qty_col = body.get("quantity_column")
    actual_col = body.get("actual_price_column")
    if not all([item_col, qty_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "item_column, quantity_column, and actual_price_column required"}, 400)
    std_col = body.get("standard_price_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_purchase_price_variance(rows, item_col, qty_col, actual_col, std_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "items": [
            {"item": it.item, "quantity": it.quantity, "avg_actual_price": it.avg_actual_price, "standard_price": it.standard_price, "total_variance": it.total_variance, "variance_type": it.variance_type}
            for it in res.items
        ],
        "total_variance": res.total_variance,
        "favorable_count": res.favorable_count,
        "unfavorable_count": res.unfavorable_count,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/vendor-performance")
async def vendor_performance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze vendor delivery performance.
    Body: {"vendor_column": "...", "delivery_date_column": "...", "promised_date_column": "...", "quality_column": "...", "quantity_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import analyze_vendor_performance
    vendor_col = body.get("vendor_column")
    delivery_col = body.get("delivery_date_column")
    promised_col = body.get("promised_date_column")
    if not all([vendor_col, delivery_col, promised_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vendor_column, delivery_date_column, and promised_date_column required"}, 400)
    quality_col = body.get("quality_column")
    qty_col = body.get("quantity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_vendor_performance(rows, vendor_col, delivery_col, promised_col, quality_col, qty_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "vendors": [
            {"vendor": v.vendor, "total_deliveries": v.total_deliveries, "on_time_count": v.on_time_count, "late_count": v.late_count, "on_time_pct": v.on_time_pct, "avg_days_late": v.avg_days_late, "avg_quality": v.avg_quality, "total_quantity": v.total_quantity}
            for v in res.vendors
        ],
        "overall_on_time_pct": res.overall_on_time_pct,
        "avg_quality": res.avg_quality,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/spend-by-category")
async def spend_by_category(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze procurement spend by category.
    Body: {"category_column": "...", "amount_column": "...", "vendor_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import analyze_spend_by_category
    cat_col = body.get("category_column")
    amount_col = body.get("amount_column")
    if not all([cat_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column and amount_column required"}, 400)
    vendor_col = body.get("vendor_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_spend_by_category(rows, cat_col, amount_col, vendor_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "categories": [
            {"category": c.category, "total_spend": c.total_spend, "pct_of_total": c.pct_of_total, "transaction_count": c.transaction_count, "vendor_count": c.vendor_count}
            for c in res.categories
        ],
        "total_spend": res.total_spend,
        "hhi": res.hhi,
        "concentration_risk": res.concentration_risk,
        "top_categories": res.top_categories,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Financial Ratios endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/profitability-ratios")
async def profitability_ratios(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute profitability ratios from revenue and cost data.
    Body: {"revenue_column": "...", "cost_column": "...", "entity_column": "..."}
    """
    from business_brain.discovery.financial_ratios import compute_profitability_ratios
    rev_col = body.get("revenue_column")
    cost_col = body.get("cost_column")
    if not all([rev_col, cost_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and cost_column required"}, 400)
    entity_col = body.get("entity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_profitability_ratios(rows, rev_col, cost_col, entity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "entities": [
            {"entity": e.entity, "revenue": e.revenue, "cost": e.cost, "gross_profit": e.gross_profit, "gross_margin_pct": e.gross_margin_pct, "cost_ratio": e.cost_ratio}
            for e in res.entities
        ],
        "total_revenue": res.total_revenue,
        "total_cost": res.total_cost,
        "overall_margin": res.overall_margin,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/liquidity-ratios")
async def liquidity_ratios(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute liquidity ratios from balance sheet data.
    Body: {"current_assets_column": "...", "current_liabilities_column": "...", "cash_column": "...", "inventory_column": "...", "entity_column": "..."}
    """
    from business_brain.discovery.financial_ratios import compute_liquidity_ratios
    ca_col = body.get("current_assets_column")
    cl_col = body.get("current_liabilities_column")
    if not all([ca_col, cl_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "current_assets_column and current_liabilities_column required"}, 400)
    cash_col = body.get("cash_column")
    inv_col = body.get("inventory_column")
    entity_col = body.get("entity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_liquidity_ratios(rows, ca_col, cl_col, cash_col, inv_col, entity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "entities": [
            {"entity": e.entity, "current_assets": e.current_assets, "current_liabilities": e.current_liabilities, "current_ratio": e.current_ratio, "quick_ratio": e.quick_ratio, "cash_ratio": e.cash_ratio, "rating": e.rating}
            for e in res.entities
        ],
        "avg_current_ratio": res.avg_current_ratio,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/efficiency-ratios")
async def efficiency_ratios(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute efficiency ratios from financial data.
    Body: {"revenue_column": "...", "assets_column": "...", "receivables_column": "...", "payables_column": "...", "cogs_column": "..."}
    """
    from business_brain.discovery.financial_ratios import compute_efficiency_ratios
    rev_col = body.get("revenue_column")
    assets_col = body.get("assets_column")
    if not all([rev_col, assets_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and assets_column required"}, 400)
    recv_col = body.get("receivables_column")
    pay_col = body.get("payables_column")
    cogs_col = body.get("cogs_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_efficiency_ratios(rows, rev_col, assets_col, recv_col, pay_col, cogs_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "ratios": {
            "asset_turnover": res.ratios.asset_turnover,
            "receivables_turnover": res.ratios.receivables_turnover,
            "dso": res.ratios.dso,
            "payables_turnover": res.ratios.payables_turnover,
            "dpo": res.ratios.dpo,
            "cash_conversion_cycle": res.ratios.cash_conversion_cycle,
        },
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/financial-trends")
async def financial_trends(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze financial trends over periods.
    Body: {"metric_column": "...", "period_column": "...", "entity_column": "..."}
    """
    from business_brain.discovery.financial_ratios import analyze_financial_trends
    metric_col = body.get("metric_column")
    period_col = body.get("period_column")
    if not all([metric_col, period_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "metric_column and period_column required"}, 400)
    entity_col = body.get("entity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_financial_trends(rows, metric_col, period_col, entity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "entities": [
            {
                "entity": e.entity,
                "periods": [
                    {"period": p.period, "value": p.value, "change": p.change, "change_pct": p.change_pct}
                    for p in e.periods
                ],
                "trend": e.trend,
                "cagr": e.cagr,
                "best_period": e.best_period,
                "worst_period": e.worst_period,
            }
            for e in res.entities
        ],
        "overall_trend": res.overall_trend,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Project Tracker endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/project-status")
async def project_status(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze project status, dates, and budget.
    Body: {"project_column": "...", "status_column": "...", "start_column": "...", "end_column": "...", "budget_column": "..."}
    """
    from business_brain.discovery.project_tracker import analyze_project_status
    proj_col = body.get("project_column")
    status_col = body.get("status_column")
    if not all([proj_col, status_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "project_column and status_column required"}, 400)
    start_col = body.get("start_column")
    end_col = body.get("end_column")
    budget_col = body.get("budget_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_project_status(rows, proj_col, status_col, start_col, end_col, budget_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "projects": [
            {"project": p.project, "status": p.status, "start_date": p.start_date, "end_date": p.end_date, "budget": p.budget, "duration_days": p.duration_days}
            for p in res.projects
        ],
        "status_distribution": res.status_distribution,
        "completion_rate": res.completion_rate,
        "avg_duration_days": res.avg_duration_days,
        "total_budget": res.total_budget,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/milestones")
async def milestones(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze milestone progress and health.
    Body: {"project_column": "...", "milestone_column": "...", "due_date_column": "...", "completion_date_column": "...", "status_column": "..."}
    """
    from business_brain.discovery.project_tracker import analyze_milestones
    proj_col = body.get("project_column")
    ms_col = body.get("milestone_column")
    due_col = body.get("due_date_column")
    if not all([proj_col, ms_col, due_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "project_column, milestone_column, and due_date_column required"}, 400)
    comp_col = body.get("completion_date_column")
    status_col = body.get("status_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_milestones(rows, proj_col, ms_col, due_col, comp_col, status_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "projects": [
            {"project": p.project, "total": p.total, "completed": p.completed, "overdue": p.overdue, "on_time_pct": p.on_time_pct, "avg_delay_days": p.avg_delay_days}
            for p in res.projects
        ],
        "total_milestones": res.total_milestones,
        "overall_on_time_pct": res.overall_on_time_pct,
        "health": res.health,
        "upcoming_count": res.upcoming_count,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/resource-allocation")
async def resource_allocation(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze resource allocation across projects.
    Body: {"resource_column": "...", "project_column": "...", "hours_column": "...", "role_column": "..."}
    """
    from business_brain.discovery.project_tracker import analyze_resource_allocation
    resource_col = body.get("resource_column")
    proj_col = body.get("project_column")
    hours_col = body.get("hours_column")
    if not all([resource_col, proj_col, hours_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "resource_column, project_column, and hours_column required"}, 400)
    role_col = body.get("role_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_resource_allocation(rows, resource_col, proj_col, hours_col, role_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "resources": [
            {"resource": r.resource, "total_hours": r.total_hours, "project_count": r.project_count, "avg_hours_per_project": r.avg_hours_per_project, "utilization_status": r.utilization_status}
            for r in res.resources
        ],
        "by_role": [
            {"role": rh.role, "total_hours": rh.total_hours, "resource_count": rh.resource_count}
            for rh in res.by_role
        ] if res.by_role is not None else None,
        "over_allocated": res.over_allocated,
        "under_utilized": res.under_utilized,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/project-health")
async def project_health(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute project health by comparing planned vs actual.
    Body: {"project_column": "...", "planned_column": "...", "actual_column": "...", "metric_type": "cost"}
    """
    from business_brain.discovery.project_tracker import compute_project_health
    proj_col = body.get("project_column")
    planned_col = body.get("planned_column")
    actual_col = body.get("actual_column")
    if not all([proj_col, planned_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "project_column, planned_column, and actual_column required"}, 400)
    metric_type = body.get("metric_type", "cost")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_project_health(rows, proj_col, planned_col, actual_col, metric_type)
    if not res:
        return {"error": "Insufficient data"}
    return {
        "projects": [
            {"project": p.project, "planned": p.planned, "actual": p.actual, "variance": p.variance, "variance_pct": p.variance_pct, "performance_index": p.performance_index, "health": p.health}
            for p in res
        ],
    }


# ---------------------------------------------------------------------------
# SCADA Analyzer endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/sensor-readings")
async def sensor_readings(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze sensor readings with per-sensor statistics.
    Body: {"sensor_column": "...", "value_column": "...", "timestamp_column": "...", "unit_column": "..."}
    """
    from business_brain.discovery.scada_analyzer import analyze_sensor_readings
    sensor_col = body.get("sensor_column")
    value_col = body.get("value_column")
    if not all([sensor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "sensor_column and value_column required"}, 400)
    ts_col = body.get("timestamp_column")
    unit_col = body.get("unit_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sensor_readings(rows, sensor_col, value_col, ts_col, unit_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "sensors": [
            {"sensor": s.sensor, "min_val": s.min_val, "max_val": s.max_val, "mean": s.mean, "std": s.std, "reading_count": s.reading_count, "stability_index": s.stability_index, "unit": s.unit}
            for s in res.sensors
        ],
        "stable_count": res.stable_count,
        "unstable_count": res.unstable_count,
        "total_readings": res.total_readings,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/sensor-anomalies")
async def sensor_anomalies(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect anomalies in sensor readings.
    Body: {"sensor_column": "...", "value_column": "...", "low_limit": 0.0, "high_limit": 100.0}
    """
    from business_brain.discovery.scada_analyzer import detect_sensor_anomalies
    sensor_col = body.get("sensor_column")
    value_col = body.get("value_column")
    if not all([sensor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "sensor_column and value_column required"}, 400)
    low_limit = body.get("low_limit")
    high_limit = body.get("high_limit")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = detect_sensor_anomalies(rows, sensor_col, value_col, low_limit, high_limit)
    return {
        "anomalies": [
            {"sensor": a.sensor, "value": a.value, "expected_range": list(a.expected_range), "anomaly_type": a.anomaly_type, "index": a.index}
            for a in res
        ],
        "total_anomalies": len(res),
    }


@app.post("/tables/{table_name}/process-stability")
async def process_stability(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute process capability (Cp, Cpk) per sensor.
    Body: {"sensor_column": "...", "value_column": "...", "target_column": "..."}
    """
    from business_brain.discovery.scada_analyzer import compute_process_stability
    sensor_col = body.get("sensor_column")
    value_col = body.get("value_column")
    if not all([sensor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "sensor_column and value_column required"}, 400)
    target_col = body.get("target_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_process_stability(rows, sensor_col, value_col, target_col)
    if not res:
        return {"error": "Insufficient data"}
    return {
        "sensors": [
            {"sensor": s.sensor, "mean": s.mean, "std": s.std, "usl": s.usl, "lsl": s.lsl, "cp": s.cp, "cpk": s.cpk, "rating": s.rating}
            for s in res
        ],
    }


@app.post("/tables/{table_name}/alarm-frequency")
async def alarm_frequency(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze alarm frequency and chattering detection.
    Body: {"alarm_column": "...", "severity_column": "...", "timestamp_column": "...", "equipment_column": "..."}
    """
    from business_brain.discovery.scada_analyzer import analyze_alarm_frequency
    alarm_col = body.get("alarm_column")
    if not alarm_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "alarm_column required"}, 400)
    severity_col = body.get("severity_column")
    ts_col = body.get("timestamp_column")
    equip_col = body.get("equipment_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_alarm_frequency(rows, alarm_col, severity_col, ts_col, equip_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_alarms": res.total_alarms,
        "by_severity": res.by_severity,
        "by_equipment": [
            {"equipment": e.equipment, "alarm_count": e.alarm_count, "critical_count": e.critical_count}
            for e in res.by_equipment
        ],
        "top_alarms": [
            {"alarm": a.alarm, "count": a.count, "pct_of_total": a.pct_of_total}
            for a in res.top_alarms
        ],
        "chattering_alarms": res.chattering_alarms,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Customer Analytics endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/customer-segments")
async def customer_segments(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Segment customers into revenue tiers.
    Body: {"customer_column": "...", "revenue_column": "...", "frequency_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.customer_analytics import analyze_customer_segments
    cust_col = body.get("customer_column")
    rev_col = body.get("revenue_column")
    if not all([cust_col, rev_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column and revenue_column required"}, 400)
    freq_col = body.get("frequency_column")
    cat_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_customer_segments(rows, cust_col, rev_col, freq_col, cat_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "tiers": [
            {"tier": t.tier, "customer_count": t.customer_count, "total_revenue": t.total_revenue, "avg_revenue": t.avg_revenue, "pct_of_total": t.pct_of_total, "avg_frequency": t.avg_frequency}
            for t in res.tiers
        ],
        "total_customers": res.total_customers,
        "total_revenue": res.total_revenue,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/churn-risk")
async def churn_risk(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze customer churn risk based on recency.
    Body: {"customer_column": "...", "date_column": "...", "amount_column": "..."}
    """
    from business_brain.discovery.customer_analytics import analyze_churn_risk
    cust_col = body.get("customer_column")
    date_col = body.get("date_column")
    if not all([cust_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column and date_column required"}, 400)
    amount_col = body.get("amount_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_churn_risk(rows, cust_col, date_col, amount_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "statuses": [
            {"status": s.status, "customer_count": s.customer_count, "pct": s.pct, "avg_spend": s.avg_spend}
            for s in res.statuses
        ],
        "total_customers": res.total_customers,
        "churn_rate": res.churn_rate,
        "at_risk_rate": res.at_risk_rate,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/customer-concentration")
async def customer_concentration(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute customer revenue concentration and Pareto analysis.
    Body: {"customer_column": "...", "amount_column": "..."}
    """
    from business_brain.discovery.customer_analytics import compute_customer_concentration
    cust_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    if not all([cust_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column and amount_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_customer_concentration(rows, cust_col, amount_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "top_customers": [
            {"customer": c.customer, "total_spend": c.total_spend, "share_pct": c.share_pct, "cumulative_pct": c.cumulative_pct}
            for c in res.top_customers
        ],
        "hhi": res.hhi,
        "concentration_risk": res.concentration_risk,
        "customers_for_80pct": res.customers_for_80pct,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/purchase-behavior")
async def purchase_behavior(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze customer purchase behavior patterns.
    Body: {"customer_column": "...", "amount_column": "...", "date_column": "...", "product_column": "..."}
    """
    from business_brain.discovery.customer_analytics import analyze_purchase_behavior
    cust_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([cust_col, amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, and date_column required"}, 400)
    product_col = body.get("product_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_purchase_behavior(rows, cust_col, amount_col, date_col, product_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "customers": [
            {"customer": c.customer, "total_orders": c.total_orders, "total_spend": c.total_spend, "avg_order_value": c.avg_order_value, "first_purchase": c.first_purchase, "last_purchase": c.last_purchase, "lifespan_days": c.lifespan_days}
            for c in res.customers
        ],
        "avg_orders": res.avg_orders,
        "avg_aov": res.avg_aov,
        "repeat_purchase_rate": res.repeat_purchase_rate,
        "avg_lifespan_days": res.avg_lifespan_days,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Sales Analytics endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/sales-performance")
async def sales_performance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze overall sales performance with optional rep and region breakdown.
    Body: {"amount_column": "...", "date_column": "...", "rep_column": "...", "region_column": "..."}
    """
    from business_brain.discovery.sales_analytics import analyze_sales_performance
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    rep_col = body.get("rep_column")
    region_col = body.get("region_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sales_performance(rows, amount_col, date_col, rep_col, region_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_sales": res.total_sales,
        "period_count": res.period_count,
        "avg_per_period": res.avg_per_period,
        "growth_rate": res.growth_rate,
        "best_period": {"period": res.best_period.period, "amount": res.best_period.amount} if res.best_period else None,
        "worst_period": {"period": res.worst_period.period, "amount": res.worst_period.amount} if res.worst_period else None,
        "by_rep": [
            {"rep": r.rep, "total": r.total, "rank": r.rank}
            for r in res.by_rep
        ],
        "by_region": [
            {"region": r.region, "total": r.total, "pct": r.pct}
            for r in res.by_region
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/product-mix")
async def product_mix(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze revenue distribution across products.
    Body: {"product_column": "...", "amount_column": "...", "quantity_column": "..."}
    """
    from business_brain.discovery.sales_analytics import analyze_product_mix
    product_col = body.get("product_column")
    amount_col = body.get("amount_column")
    if not all([product_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "product_column and amount_column required"}, 400)
    qty_col = body.get("quantity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_product_mix(rows, product_col, amount_col, qty_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "products": [
            {"product": p.product, "revenue": p.revenue, "pct_of_total": p.pct_of_total, "transaction_count": p.transaction_count, "avg_price": p.avg_price}
            for p in res.products
        ],
        "total_revenue": res.total_revenue,
        "hhi": res.hhi,
        "concentration_risk": res.concentration_risk,
        "top_products": [
            {"product": p.product, "revenue": p.revenue, "pct_of_total": p.pct_of_total, "transaction_count": p.transaction_count, "avg_price": p.avg_price}
            for p in res.top_products
        ],
        "bottom_products": [
            {"product": p.product, "revenue": p.revenue, "pct_of_total": p.pct_of_total, "transaction_count": p.transaction_count, "avg_price": p.avg_price}
            for p in res.bottom_products
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/sales-velocity")
async def sales_velocity(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute sales pipeline velocity metrics.
    Body: {"deal_column": "...", "amount_column": "...", "date_column": "...", "stage_column": "..."}
    """
    from business_brain.discovery.sales_analytics import compute_sales_velocity
    deal_col = body.get("deal_column")
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([deal_col, amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, amount_column, and date_column required"}, 400)
    stage_col = body.get("stage_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_sales_velocity(rows, deal_col, amount_col, date_col, stage_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_deals": res.total_deals,
        "total_value": res.total_value,
        "avg_deal_size": res.avg_deal_size,
        "win_rate": res.win_rate,
        "avg_days_to_close": res.avg_days_to_close,
        "pipeline_velocity": res.pipeline_velocity,
        "funnel": [
            {"stage": stage, "count": count}
            for stage, count in res.funnel
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/discount-impact")
async def discount_impact(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze the impact of discounts on revenue.
    Body: {"amount_column": "...", "discount_column": "...", "quantity_column": "...", "product_column": "..."}
    """
    from business_brain.discovery.sales_analytics import analyze_discount_impact
    amount_col = body.get("amount_column")
    discount_col = body.get("discount_column")
    if not all([amount_col, discount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and discount_column required"}, 400)
    qty_col = body.get("quantity_column")
    product_col = body.get("product_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_discount_impact(rows, amount_col, discount_col, qty_col, product_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_discount": res.avg_discount,
        "max_discount": res.max_discount,
        "revenue_impact": res.revenue_impact,
        "distribution": [
            {"range_label": b.range_label, "count": b.count, "pct": b.pct}
            for b in res.distribution
        ],
        "by_product": [
            {"product": p.product, "avg_discount": p.avg_discount, "deal_count": p.deal_count}
            for p in res.by_product
        ],
        "discount_volume_correlation": res.discount_volume_correlation,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Cash Flow Analyzer endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/cash-flow")
async def cash_flow(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze cash inflows and outflows.
    Body: {"inflow_column": "...", "outflow_column": "...", "date_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.cash_flow_analyzer import analyze_cash_flow
    inflow_col = body.get("inflow_column")
    outflow_col = body.get("outflow_column")
    if not all([inflow_col, outflow_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "inflow_column and outflow_column required"}, 400)
    date_col = body.get("date_column")
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_cash_flow(rows, inflow_col, outflow_col, date_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_inflow": res.total_inflow,
        "total_outflow": res.total_outflow,
        "net_flow": res.net_flow,
        "cash_flow_ratio": res.cash_flow_ratio,
        "negative_periods_count": res.negative_periods_count,
        "period_flows": [
            {"period": p.period, "inflow": p.inflow, "outflow": p.outflow, "net_flow": p.net_flow}
            for p in res.period_flows
        ] if res.period_flows else None,
        "category_flows": [
            {"category": c.category, "total_inflow": c.total_inflow, "total_outflow": c.total_outflow, "net_flow": c.net_flow}
            for c in res.category_flows
        ] if res.category_flows else None,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/working-capital")
async def working_capital(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute working capital metrics.
    Body: {"receivables_column": "...", "payables_column": "...", "inventory_column": "...", "period_column": "..."}
    """
    from business_brain.discovery.cash_flow_analyzer import compute_working_capital
    receivables_col = body.get("receivables_column")
    payables_col = body.get("payables_column")
    if not all([receivables_col, payables_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "receivables_column and payables_column required"}, 400)
    inventory_col = body.get("inventory_column")
    period_col = body.get("period_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_working_capital(rows, receivables_col, payables_col, inventory_col, period_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_working_capital": res.avg_working_capital,
        "min_wc": res.min_wc,
        "max_wc": res.max_wc,
        "health": res.health,
        "periods": [
            {"period": p.period, "receivables": p.receivables, "payables": p.payables, "inventory": p.inventory, "working_capital": p.working_capital}
            for p in res.periods
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/burn-rate")
async def burn_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze monthly burn rate from expense data.
    Body: {"expense_column": "...", "date_column": "...", "revenue_column": "..."}
    """
    from business_brain.discovery.cash_flow_analyzer import analyze_burn_rate
    expense_col = body.get("expense_column")
    date_col = body.get("date_column")
    if not all([expense_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "expense_column and date_column required"}, 400)
    revenue_col = body.get("revenue_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_burn_rate(rows, expense_col, date_col, revenue_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "gross_burn_rate": res.gross_burn_rate,
        "net_burn_rate": res.net_burn_rate,
        "months_analyzed": res.months_analyzed,
        "total_expenses": res.total_expenses,
        "monthly_expenses": [
            {"month": m.month, "amount": m.amount, "is_highest": m.is_highest}
            for m in res.monthly_expenses
        ],
        "trend": res.trend,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/cash-forecast")
async def cash_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast future cash position using linear projection.
    Body: {"amount_column": "...", "date_column": "...", "type_column": "...", "periods_ahead": 3}
    """
    from business_brain.discovery.cash_flow_analyzer import forecast_cash_position
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    type_col = body.get("type_column")
    periods_ahead = body.get("periods_ahead", 3)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = forecast_cash_position(rows, amount_col, date_col, type_col, periods_ahead)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "current_net_monthly": res.current_net_monthly,
        "trend_direction": res.trend_direction,
        "trend_magnitude": res.trend_magnitude,
        "projections": [
            {"period_label": p.period_label, "projected_inflow": p.projected_inflow, "projected_outflow": p.projected_outflow, "projected_net": p.projected_net}
            for p in res.projections
        ],
        "confidence": res.confidence,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Risk Matrix endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/risk-assessment")
async def risk_assessment(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Assess risks by computing likelihood x impact scores.
    Body: {"risk_column": "...", "likelihood_column": "...", "impact_column": "...", "category_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.risk_matrix import assess_risks
    risk_col = body.get("risk_column")
    likelihood_col = body.get("likelihood_column")
    impact_col = body.get("impact_column")
    if not all([risk_col, likelihood_col, impact_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "risk_column, likelihood_column, and impact_column required"}, 400)
    category_col = body.get("category_column")
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = assess_risks(rows, risk_col, likelihood_col, impact_col, category_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_count": res.total_count,
        "critical_count": res.critical_count,
        "high_count": res.high_count,
        "medium_count": res.medium_count,
        "low_count": res.low_count,
        "risks": [
            {"name": r.name, "likelihood": r.likelihood, "impact": r.impact, "risk_score": r.risk_score, "risk_level": r.risk_level, "category": r.category, "owner": r.owner}
            for r in res.risks
        ],
        "by_category": [
            {"category": c.category, "count": c.count, "avg_score": c.avg_score}
            for c in res.by_category
        ],
        "by_owner": [
            {"owner": o.owner, "count": o.count, "critical_count": o.critical_count}
            for o in res.by_owner
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/risk-heatmap")
async def risk_heatmap(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Create a 5x5 likelihood-by-impact risk heatmap.
    Body: {"likelihood_column": "...", "impact_column": "..."}
    """
    from business_brain.discovery.risk_matrix import compute_risk_heatmap
    likelihood_col = body.get("likelihood_column")
    impact_col = body.get("impact_column")
    if not all([likelihood_col, impact_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "likelihood_column and impact_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_risk_heatmap(rows, likelihood_col, impact_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "matrix": res.matrix,
        "hotspots": [
            {"likelihood": h.likelihood, "impact": h.impact, "count": h.count, "risk_score": h.risk_score}
            for h in res.hotspots
        ],
        "total_risks": res.total_risks,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/risk-trends")
async def risk_trends(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze risk score trends over time.
    Body: {"risk_column": "...", "score_column": "...", "date_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.risk_matrix import analyze_risk_trends
    risk_col = body.get("risk_column")
    score_col = body.get("score_column")
    date_col = body.get("date_column")
    if not all([risk_col, score_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "risk_column, score_column, and date_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_risk_trends(rows, risk_col, score_col, date_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "avg_score": p.avg_score, "max_score": p.max_score, "risk_count": p.risk_count}
            for p in res.periods
        ],
        "trend_direction": res.trend_direction,
        "avg_score_change": res.avg_score_change,
        "by_category_trend": res.by_category_trend,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/risk-exposure")
async def risk_exposure(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute financial risk exposure (expected loss) for each risk.
    Body: {"risk_column": "...", "impact_value_column": "...", "probability_column": "..."}
    """
    from business_brain.discovery.risk_matrix import compute_risk_exposure
    risk_col = body.get("risk_column")
    impact_value_col = body.get("impact_value_column")
    probability_col = body.get("probability_column")
    if not all([risk_col, impact_value_col, probability_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "risk_column, impact_value_column, and probability_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_risk_exposure(rows, risk_col, impact_value_col, probability_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_expected_loss": res.total_expected_loss,
        "top_risks": [
            {"name": r.name, "impact_value": r.impact_value, "probability": r.probability, "expected_loss": r.expected_loss}
            for r in res.top_risks
        ],
        "risk_concentration_pct": res.risk_concentration_pct,
        "avg_expected_loss": res.avg_expected_loss,
        "max_expected_loss": res.max_expected_loss,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Compliance Tracker endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/compliance-status")
async def compliance_status(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Audit compliance status across requirements.
    Body: {"requirement_column": "...", "status_column": "...", "category_column": "...", "due_date_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import audit_compliance_status
    requirement_col = body.get("requirement_column")
    status_col = body.get("status_column")
    if not all([requirement_col, status_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "requirement_column and status_column required"}, 400)
    category_col = body.get("category_column")
    due_date_col = body.get("due_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = audit_compliance_status(rows, requirement_col, status_col, category_col, due_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_requirements": res.total_requirements,
        "compliant_count": res.compliant_count,
        "non_compliant_count": res.non_compliant_count,
        "compliance_rate": res.compliance_rate,
        "overdue_count": res.overdue_count,
        "by_category": [
            {"category": c.category, "total": c.total, "compliant": c.compliant, "compliance_rate": c.compliance_rate}
            for c in res.by_category
        ] if res.by_category else None,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/audit-findings")
async def audit_findings(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze audit findings by severity, area, and status.
    Body: {"finding_column": "...", "severity_column": "...", "date_column": "...", "area_column": "...", "status_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import analyze_audit_findings
    finding_col = body.get("finding_column")
    severity_col = body.get("severity_column")
    if not all([finding_col, severity_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "finding_column and severity_column required"}, 400)
    date_col = body.get("date_column")
    area_col = body.get("area_column")
    status_col = body.get("status_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_audit_findings(rows, finding_col, severity_col, date_col, area_col, status_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_findings": res.total_findings,
        "by_severity": [
            {"severity": s.severity, "count": s.count, "pct": s.pct}
            for s in res.by_severity
        ],
        "by_area": [
            {"area": a.area, "count": a.count, "critical_count": a.critical_count}
            for a in res.by_area
        ] if res.by_area else None,
        "open_count": res.open_count,
        "closed_count": res.closed_count,
        "closure_rate": res.closure_rate,
        "monthly_trend": [
            {"month": m.month, "count": m.count}
            for m in res.monthly_trend
        ] if res.monthly_trend else None,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/compliance-score")
async def compliance_score(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute a weighted compliance score.
    Body: {"requirement_column": "...", "weight_column": "...", "score_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import compute_compliance_score
    requirement_col = body.get("requirement_column")
    weight_col = body.get("weight_column")
    score_col = body.get("score_column")
    if not all([requirement_col, weight_col, score_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "requirement_column, weight_column, and score_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_compliance_score(rows, requirement_col, weight_col, score_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_score": res.overall_score,
        "rating": res.rating,
        "weighted_scores": [
            {"category": w.category, "score": w.score, "weight": w.weight}
            for w in res.weighted_scores
        ],
        "weakest_areas": [
            {"category": w.category, "score": w.score, "weight": w.weight}
            for w in res.weakest_areas
        ],
        "by_category": res.by_category,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/regulatory-deadlines")
async def regulatory_deadlines(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Track regulatory deadlines and classify urgency.
    Body: {"regulation_column": "...", "deadline_column": "...", "status_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import track_regulatory_deadlines
    regulation_col = body.get("regulation_column")
    deadline_col = body.get("deadline_column")
    if not all([regulation_col, deadline_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "regulation_column and deadline_column required"}, 400)
    status_col = body.get("status_column")
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = track_regulatory_deadlines(rows, regulation_col, deadline_col, status_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_items": res.total_items,
        "overdue_count": res.overdue_count,
        "due_this_week": res.due_this_week,
        "due_this_month": res.due_this_month,
        "upcoming_count": res.upcoming_count,
        "items": [
            {"regulation": i.regulation, "deadline": i.deadline, "days_until": i.days_until, "urgency": i.urgency, "status": i.status, "owner": i.owner}
            for i in res.items
        ],
        "by_owner": [
            {"owner": o.owner, "total": o.total, "overdue": o.overdue}
            for o in res.by_owner
        ] if res.by_owner else None,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Asset Depreciation endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/depreciation-schedule")
async def depreciation_schedule(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute annual depreciation schedule for each asset.
    Body: {"asset_column": "...", "cost_column": "...", "useful_life_column": "...", "method": "straight_line", "salvage_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import compute_depreciation_schedule
    asset_col = body.get("asset_column")
    cost_col = body.get("cost_column")
    useful_life_col = body.get("useful_life_column")
    if not all([asset_col, cost_col, useful_life_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column, cost_column, and useful_life_column required"}, 400)
    method = body.get("method", "straight_line")
    salvage_col = body.get("salvage_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_depreciation_schedule(rows, asset_col, cost_col, useful_life_col, method, salvage_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "assets": [
            {
                "asset": a.asset, "cost": a.cost, "salvage": a.salvage, "useful_life": a.useful_life,
                "annual_depreciation": a.annual_depreciation,
                "schedule": [
                    {"year": y.year, "depreciation_amount": y.depreciation_amount, "book_value": y.book_value}
                    for y in a.schedule
                ],
            }
            for a in res.assets
        ],
        "total_cost": res.total_cost,
        "total_annual_depreciation": res.total_annual_depreciation,
        "method": res.method,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/asset-age")
async def asset_age(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze asset age distribution.
    Body: {"asset_column": "...", "purchase_date_column": "...", "category_column": "...", "cost_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import analyze_asset_age
    asset_col = body.get("asset_column")
    purchase_date_col = body.get("purchase_date_column")
    if not all([asset_col, purchase_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column and purchase_date_column required"}, 400)
    category_col = body.get("category_column")
    cost_col = body.get("cost_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_asset_age(rows, asset_col, purchase_date_col, category_col, cost_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_assets": res.total_assets,
        "avg_age": res.avg_age,
        "by_lifecycle_stage": [
            {"stage": s.stage, "count": s.count, "pct": s.pct}
            for s in res.by_lifecycle_stage
        ],
        "by_category": [
            {"category": c.category, "count": c.count, "avg_age": c.avg_age}
            for c in res.by_category
        ] if res.by_category else None,
        "weighted_avg_age": res.weighted_avg_age,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/book-values")
async def book_values(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute current book value for each asset.
    Body: {"asset_column": "...", "cost_column": "...", "purchase_date_column": "...", "useful_life_column": "...", "salvage_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import compute_book_values
    asset_col = body.get("asset_column")
    cost_col = body.get("cost_column")
    purchase_date_col = body.get("purchase_date_column")
    useful_life_col = body.get("useful_life_column")
    if not all([asset_col, cost_col, purchase_date_col, useful_life_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column, cost_column, purchase_date_column, and useful_life_column required"}, 400)
    salvage_col = body.get("salvage_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_book_values(rows, asset_col, cost_col, purchase_date_col, useful_life_col, salvage_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "assets": [
            {"asset": a.asset, "original_cost": a.original_cost, "salvage_value": a.salvage_value, "age_years": a.age_years, "book_value": a.book_value, "depreciation_pct": a.depreciation_pct}
            for a in res.assets
        ],
        "total_original_cost": res.total_original_cost,
        "total_book_value": res.total_book_value,
        "depreciation_pct": res.depreciation_pct,
        "fully_depreciated_count": res.fully_depreciated_count,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/maintenance-cost-ratio")
async def maintenance_cost_ratio(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute maintenance cost as percentage of asset value.
    Body: {"asset_column": "...", "maintenance_cost_column": "...", "asset_value_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import analyze_maintenance_cost_ratio
    asset_col = body.get("asset_column")
    maintenance_cost_col = body.get("maintenance_cost_column")
    asset_value_col = body.get("asset_value_column")
    if not all([asset_col, maintenance_cost_col, asset_value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column, maintenance_cost_column, and asset_value_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_maintenance_cost_ratio(rows, asset_col, maintenance_cost_col, asset_value_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "assets": [
            {"asset": a.asset, "maintenance_cost": a.maintenance_cost, "asset_value": a.asset_value, "ratio_pct": a.ratio_pct, "is_replacement_candidate": a.is_replacement_candidate}
            for a in res.assets
        ],
        "avg_ratio": res.avg_ratio,
        "replacement_candidates": res.replacement_candidates,
        "total_maintenance": res.total_maintenance,
        "total_asset_value": res.total_asset_value,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Pricing Optimizer endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/price-distribution")
async def price_distribution(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze the distribution of prices.
    Body: {"price_column": "...", "product_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import analyze_price_distribution
    price_col = body.get("price_column")
    if not price_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "price_column required"}, 400)
    product_col = body.get("product_column")
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_price_distribution(rows, price_col, product_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "mean_price": res.mean_price,
        "median_price": res.median_price,
        "std_price": res.std_price,
        "min_price": res.min_price,
        "max_price": res.max_price,
        "outlier_count": res.outlier_count,
        "outlier_pct": res.outlier_pct,
        "by_product": [
            {"product": p.product, "min_price": p.min_price, "max_price": p.max_price, "avg_price": p.avg_price, "count": p.count}
            for p in res.by_product
        ],
        "by_category": [
            {"category": c.category, "avg_price": c.avg_price, "count": c.count}
            for c in res.by_category
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/price-elasticity")
async def price_elasticity(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute price elasticity of demand.
    Body: {"price_column": "...", "quantity_column": "...", "product_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import compute_price_elasticity
    price_col = body.get("price_column")
    quantity_col = body.get("quantity_column")
    if not all([price_col, quantity_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "price_column and quantity_column required"}, 400)
    product_col = body.get("product_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_price_elasticity(rows, price_col, quantity_col, product_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_elasticity": res.overall_elasticity,
        "elasticity_type": res.elasticity_type,
        "by_product": [
            {"product": p.product, "elasticity": p.elasticity, "elasticity_type": p.elasticity_type}
            for p in res.by_product
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/competitive-pricing")
async def competitive_pricing(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze competitive pricing gaps.
    Body: {"product_column": "...", "our_price_column": "...", "competitor_price_column": "...", "competitor_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import analyze_competitive_pricing
    product_col = body.get("product_column")
    our_price_col = body.get("our_price_column")
    competitor_price_col = body.get("competitor_price_column")
    if not all([product_col, our_price_col, competitor_price_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "product_column, our_price_column, and competitor_price_column required"}, 400)
    competitor_col = body.get("competitor_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_competitive_pricing(rows, product_col, our_price_col, competitor_price_col, competitor_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_price_gap": res.avg_price_gap,
        "premium_count": res.premium_count,
        "competitive_count": res.competitive_count,
        "discount_count": res.discount_count,
        "by_product": [
            {"product": p.product, "our_price": p.our_price, "competitor_avg_price": p.competitor_avg_price, "price_gap_pct": p.price_gap_pct, "position": p.position}
            for p in res.by_product
        ],
        "by_competitor": [
            {"competitor": c.competitor, "avg_gap_pct": c.avg_gap_pct, "product_count": c.product_count}
            for c in res.by_competitor
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/price-margins")
async def price_margins(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute gross margin analysis.
    Body: {"price_column": "...", "cost_column": "...", "product_column": "...", "quantity_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import compute_price_margin_analysis
    price_col = body.get("price_column")
    cost_col = body.get("cost_column")
    if not all([price_col, cost_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "price_column and cost_column required"}, 400)
    product_col = body.get("product_column")
    quantity_col = body.get("quantity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_price_margin_analysis(rows, price_col, cost_col, product_col, quantity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_margin": res.avg_margin,
        "weighted_margin": res.weighted_margin,
        "min_margin": res.min_margin,
        "max_margin": res.max_margin,
        "negative_margin_count": res.negative_margin_count,
        "by_product": [
            {"product": p.product, "avg_price": p.avg_price, "avg_cost": p.avg_cost, "margin_pct": p.margin_pct, "volume": p.volume}
            for p in res.by_product
        ],
        "margin_distribution": [
            {"range_label": b.range_label, "count": b.count, "pct": b.pct}
            for b in res.margin_distribution
        ],
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Employee Attrition endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/attrition-rate")
async def attrition_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze employee attrition rate.
    Body: {"employee_column": "...", "status_column": "...", "date_column": "...", "department_column": "..."}
    """
    from business_brain.discovery.employee_attrition import analyze_attrition_rate
    employee_col = body.get("employee_column")
    status_col = body.get("status_column")
    if not all([employee_col, status_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column and status_column required"}, 400)
    date_col = body.get("date_column")
    department_col = body.get("department_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_attrition_rate(rows, employee_col, status_col, date_col, department_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_employees": res.total_employees,
        "active_count": res.active_count,
        "left_count": res.left_count,
        "attrition_rate": res.attrition_rate,
        "monthly_trends": [
            {"month": m.month, "active": m.active, "left": m.left, "rate": m.rate}
            for m in res.monthly_trends
        ],
        "by_department": [
            {"department": d.department, "total": d.total, "left": d.left, "rate": d.rate}
            for d in res.by_department
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/tenure-distribution")
async def tenure_distribution(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze employee tenure distribution.
    Body: {"employee_column": "...", "hire_date_column": "...", "termination_date_column": "...", "department_column": "..."}
    """
    from business_brain.discovery.employee_attrition import analyze_tenure_distribution
    employee_col = body.get("employee_column")
    hire_date_col = body.get("hire_date_column")
    if not all([employee_col, hire_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column and hire_date_column required"}, 400)
    termination_date_col = body.get("termination_date_column")
    department_col = body.get("department_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_tenure_distribution(rows, employee_col, hire_date_col, termination_date_col, department_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_tenure": res.avg_tenure,
        "median_tenure": res.median_tenure,
        "buckets": [
            {"range_label": b.range_label, "count": b.count, "pct": b.pct}
            for b in res.buckets
        ],
        "leaver_avg_tenure": res.leaver_avg_tenure,
        "stayer_avg_tenure": res.stayer_avg_tenure,
        "by_department": [
            {"department": d.department, "avg_tenure": d.avg_tenure, "count": d.count}
            for d in res.by_department
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/retention-cohorts")
async def retention_cohorts(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Group employees by hire quarter and compute retention rates.
    Body: {"employee_column": "...", "hire_date_column": "...", "termination_date_column": "..."}
    """
    from business_brain.discovery.employee_attrition import compute_retention_cohorts
    employee_col = body.get("employee_column")
    hire_date_col = body.get("hire_date_column")
    termination_date_col = body.get("termination_date_column")
    if not all([employee_col, hire_date_col, termination_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column, hire_date_column, and termination_date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_retention_cohorts(rows, employee_col, hire_date_col, termination_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "cohorts": [
            {
                "cohort_label": c.cohort_label, "starting_count": c.starting_count,
                "retained_count": c.retained_count, "retention_rate": c.retention_rate,
                "retention_milestones": [
                    {"period": m.period, "retained_pct": m.retained_pct}
                    for m in c.retention_milestones
                ],
            }
            for c in res.cohorts
        ],
        "overall_1yr_retention": res.overall_1yr_retention,
        "best_cohort": res.best_cohort,
        "worst_cohort": res.worst_cohort,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/attrition-drivers")
async def attrition_drivers(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze factors that drive employee attrition.
    Body: {"employee_column": "...", "status_column": "...", "factor_columns": ["col1", "col2", ...]}
    """
    from business_brain.discovery.employee_attrition import analyze_attrition_drivers
    employee_col = body.get("employee_column")
    status_col = body.get("status_column")
    factor_cols = body.get("factor_columns")
    if not all([employee_col, status_col, factor_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column, status_column, and factor_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_attrition_drivers(rows, employee_col, status_col, factor_cols)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "factors": [
            {"factor_name": f.factor_name, "factor_type": f.factor_type, "leaver_value": f.leaver_value, "stayer_value": f.stayer_value, "impact": f.impact, "direction": f.direction}
            for f in res.factors
        ],
        "top_driver": res.top_driver,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Revenue Forecaster endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/revenue-forecast")
async def revenue_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast future revenue using linear extrapolation.
    Body: {"revenue_column": "...", "date_column": "...", "periods_ahead": 3}
    """
    from business_brain.discovery.revenue_forecaster import forecast_revenue
    revenue_col = body.get("revenue_column")
    date_col = body.get("date_column")
    if not all([revenue_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and date_column required"}, 400)
    periods_ahead = body.get("periods_ahead", 3)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = forecast_revenue(rows, revenue_col, date_col, periods_ahead)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "revenue": p.revenue, "growth_rate": p.growth_rate}
            for p in res.periods
        ],
        "forecasts": [
            {"period": f.period, "revenue": f.revenue, "growth_rate": f.growth_rate}
            for f in res.forecasts
        ],
        "avg_growth_rate": res.avg_growth_rate,
        "trend": res.trend,
        "total_historical": res.total_historical,
        "total_forecast": res.total_forecast,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/revenue-segments")
async def revenue_segments(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze revenue breakdown by segment.
    Body: {"revenue_column": "...", "segment_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.revenue_forecaster import analyze_revenue_segments
    revenue_col = body.get("revenue_column")
    segment_col = body.get("segment_column")
    if not all([revenue_col, segment_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and segment_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_revenue_segments(rows, revenue_col, segment_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "segments": [
            {"segment": s.segment, "revenue": s.revenue, "share_pct": s.share_pct, "rank": s.rank, "transaction_count": s.transaction_count}
            for s in res.segments
        ],
        "total_revenue": res.total_revenue,
        "top_segment": res.top_segment,
        "concentration_index": res.concentration_index,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/revenue-growth")
async def revenue_growth(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute period-over-period revenue growth rates.
    Body: {"revenue_column": "...", "date_column": "...", "comparison": "period_over_period"}
    """
    from business_brain.discovery.revenue_forecaster import compute_revenue_growth
    revenue_col = body.get("revenue_column")
    date_col = body.get("date_column")
    if not all([revenue_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and date_column required"}, 400)
    comparison = body.get("comparison", "period_over_period")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_revenue_growth(rows, revenue_col, date_col, comparison)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "revenue": p.revenue, "growth_rate": p.growth_rate, "growth_absolute": p.growth_absolute}
            for p in res.periods
        ],
        "cagr": res.cagr,
        "avg_growth": res.avg_growth,
        "best_period": res.best_period,
        "worst_period": res.worst_period,
        "volatility": res.volatility,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/revenue-drivers")
async def revenue_drivers(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze which driver columns correlate with revenue.
    Body: {"revenue_column": "...", "driver_columns": ["col1", "col2", ...]}
    """
    from business_brain.discovery.revenue_forecaster import analyze_revenue_drivers
    revenue_col = body.get("revenue_column")
    driver_cols = body.get("driver_columns")
    if not all([revenue_col, driver_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and driver_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_revenue_drivers(rows, revenue_col, driver_cols)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "drivers": [
            {"driver": d.driver, "correlation": d.correlation, "direction": d.direction, "avg_when_high_revenue": d.avg_when_high_revenue, "avg_when_low_revenue": d.avg_when_low_revenue}
            for d in res.drivers
        ],
        "top_driver": res.top_driver,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Accounts Aging endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/receivables-aging")
async def receivables_aging(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze receivables aging by customer.
    Body: {"customer_column": "...", "amount_column": "...", "invoice_date_column": "...", "due_date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import analyze_receivables_aging
    customer_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    invoice_date_col = body.get("invoice_date_column")
    if not all([customer_col, amount_col, invoice_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, and invoice_date_column required"}, 400)
    due_date_col = body.get("due_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_receivables_aging(rows, customer_col, amount_col, invoice_date_col, due_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "buckets": [
            {"label": b.label, "min_days": b.min_days, "max_days": b.max_days, "count": b.count, "amount": b.amount, "pct_of_total": b.pct_of_total}
            for b in res.buckets
        ],
        "by_customer": [
            {"customer": c.customer, "total": c.total, "current": c.current, "days_31_60": c.days_31_60, "days_61_90": c.days_61_90, "days_91_120": c.days_91_120, "over_120": c.over_120}
            for c in res.by_customer
        ],
        "total_outstanding": res.total_outstanding,
        "total_overdue": res.total_overdue,
        "avg_days_outstanding": res.avg_days_outstanding,
        "worst_customers": res.worst_customers,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/payables-aging")
async def payables_aging(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze payables aging by vendor.
    Body: {"vendor_column": "...", "amount_column": "...", "invoice_date_column": "...", "due_date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import analyze_payables_aging
    vendor_col = body.get("vendor_column")
    amount_col = body.get("amount_column")
    invoice_date_col = body.get("invoice_date_column")
    if not all([vendor_col, amount_col, invoice_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vendor_column, amount_column, and invoice_date_column required"}, 400)
    due_date_col = body.get("due_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_payables_aging(rows, vendor_col, amount_col, invoice_date_col, due_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "buckets": [
            {"label": b.label, "min_days": b.min_days, "max_days": b.max_days, "count": b.count, "amount": b.amount, "pct_of_total": b.pct_of_total}
            for b in res.buckets
        ],
        "by_vendor": [
            {"vendor": v.vendor, "total": v.total, "current": v.current, "days_31_60": v.days_31_60, "days_61_90": v.days_61_90, "days_91_120": v.days_91_120, "over_120": v.over_120}
            for v in res.by_vendor
        ],
        "total_outstanding": res.total_outstanding,
        "total_overdue": res.total_overdue,
        "avg_days_outstanding": res.avg_days_outstanding,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/dso")
async def dso_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute Days Sales Outstanding (DSO).
    Body: {"revenue_column": "...", "receivables_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import compute_dso
    revenue_col = body.get("revenue_column")
    receivables_col = body.get("receivables_column")
    if not all([revenue_col, receivables_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and receivables_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_dso(rows, revenue_col, receivables_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_dso": res.overall_dso,
        "periods": [
            {"period": p.period, "dso": p.dso, "revenue": p.revenue, "receivables": p.receivables}
            for p in res.periods
        ],
        "trend": res.trend,
        "benchmark_status": res.benchmark_status,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/collection-effectiveness")
async def collection_effectiveness(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze collection effectiveness by customer.
    Body: {"customer_column": "...", "amount_column": "...", "paid_amount_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import analyze_collection_effectiveness
    customer_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    paid_amount_col = body.get("paid_amount_column")
    if not all([customer_col, amount_col, paid_amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, and paid_amount_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_collection_effectiveness(rows, customer_col, amount_col, paid_amount_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_rate": res.overall_rate,
        "total_invoiced": res.total_invoiced,
        "total_collected": res.total_collected,
        "total_outstanding": res.total_outstanding,
        "by_customer": [
            {"customer": c.customer, "invoiced": c.invoiced, "collected": c.collected, "collection_rate": c.collection_rate, "outstanding": c.outstanding}
            for c in res.by_customer
        ],
        "best_collectors": res.best_collectors,
        "worst_collectors": res.worst_collectors,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Pipeline Analyzer endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/pipeline-stages")
async def pipeline_stages(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze pipeline by stage, computing counts, values, and conversions.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import analyze_pipeline_stages
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    value_col = body.get("value_column")
    if not all([deal_col, stage_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, stage_column, and value_column required"}, 400)
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_pipeline_stages(rows, deal_col, stage_col, value_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "stages": [
            {"stage": s.stage, "deal_count": s.deal_count, "total_value": s.total_value, "avg_value": s.avg_value, "pct_of_deals": s.pct_of_deals, "pct_of_value": s.pct_of_value}
            for s in res.stages
        ],
        "conversions": [
            {"from_stage": c.from_stage, "to_stage": c.to_stage, "conversion_rate": c.conversion_rate}
            for c in res.conversions
        ],
        "total_deals": res.total_deals,
        "total_value": res.total_value,
        "weighted_value": res.weighted_value,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/pipeline-velocity")
async def pipeline_velocity(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Calculate pipeline velocity and identify bottleneck stages.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "date_column": "...", "close_date_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import analyze_pipeline_velocity
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    value_col = body.get("value_column")
    date_col = body.get("date_column")
    if not all([deal_col, stage_col, value_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, stage_column, value_column, and date_column required"}, 400)
    close_date_col = body.get("close_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_pipeline_velocity(rows, deal_col, stage_col, value_col, date_col, close_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "stage_velocities": [
            {"stage": sv.stage, "avg_days": sv.avg_days, "deal_count": sv.deal_count}
            for sv in res.stage_velocities
        ],
        "avg_cycle_days": res.avg_cycle_days,
        "fastest_deal_days": res.fastest_deal_days,
        "slowest_deal_days": res.slowest_deal_days,
        "bottleneck_stage": res.bottleneck_stage,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/win-rate")
async def win_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute win/loss rates overall and by owner.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import compute_win_rate
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    if not all([deal_col, stage_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column and stage_column required"}, 400)
    value_col = body.get("value_column")
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_win_rate(rows, deal_col, stage_col, value_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_win_rate": res.overall_win_rate,
        "total_won": res.total_won,
        "total_lost": res.total_lost,
        "won_value": res.won_value,
        "lost_value": res.lost_value,
        "by_owner": [
            {"owner": o.owner, "won": o.won, "lost": o.lost, "win_rate": o.win_rate, "won_value": o.won_value}
            for o in res.by_owner
        ],
        "best_performer": res.best_performer,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/pipeline-forecast")
async def pipeline_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast weighted pipeline value by stage.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "probability_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import forecast_pipeline
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    value_col = body.get("value_column")
    if not all([deal_col, stage_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, stage_column, and value_column required"}, 400)
    probability_col = body.get("probability_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = forecast_pipeline(rows, deal_col, stage_col, value_col, probability_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "stages": [
            {"stage": s.stage, "deal_count": s.deal_count, "raw_value": s.raw_value, "probability": s.probability, "weighted_value": s.weighted_value}
            for s in res.stages
        ],
        "total_raw": res.total_raw,
        "total_weighted": res.total_weighted,
        "expected_close_value": res.expected_close_value,
        "coverage_ratio": res.coverage_ratio,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Market Basket endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/product-associations")
async def product_associations(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Find pairs of products that frequently appear together in transactions.
    Body: {"transaction_column": "...", "product_column": "...", "min_support": 0.01}
    """
    from business_brain.discovery.market_basket import find_product_associations
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    if not all([transaction_col, product_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column and product_column required"}, 400)
    min_support = body.get("min_support", 0.01)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = find_product_associations(rows, transaction_col, product_col, min_support)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "pairs": [
            {"product_a": p.product_a, "product_b": p.product_b, "support": p.support, "confidence_a_to_b": p.confidence_a_to_b, "confidence_b_to_a": p.confidence_b_to_a, "lift": p.lift, "co_occurrence_count": p.co_occurrence_count}
            for p in res.pairs
        ],
        "total_transactions": res.total_transactions,
        "unique_products": res.unique_products,
        "avg_basket_size": res.avg_basket_size,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/basket-size")
async def basket_size(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze the distribution of basket sizes (items per transaction).
    Body: {"transaction_column": "...", "product_column": "...", "value_column": "..."}
    """
    from business_brain.discovery.market_basket import analyze_basket_size
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    if not all([transaction_col, product_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column and product_column required"}, 400)
    value_col = body.get("value_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_basket_size(rows, transaction_col, product_col, value_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_size": res.avg_size,
        "median_size": res.median_size,
        "max_size": res.max_size,
        "min_size": res.min_size,
        "distribution": [
            {"size": b.size, "count": b.count, "pct": b.pct}
            for b in res.distribution
        ],
        "avg_value": res.avg_value,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/cross-sell")
async def cross_sell(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Find products most frequently purchased alongside a target product.
    Body: {"transaction_column": "...", "product_column": "...", "target_product": "..."}
    """
    from business_brain.discovery.market_basket import find_cross_sell_opportunities
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    target_product = body.get("target_product")
    if not all([transaction_col, product_col, target_product]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column, product_column, and target_product required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = find_cross_sell_opportunities(rows, transaction_col, product_col, target_product)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "target_product": res.target_product,
        "target_transactions": res.target_transactions,
        "recommendations": [
            {"product": r.product, "co_purchase_rate": r.co_purchase_rate, "lift": r.lift, "co_occurrence_count": r.co_occurrence_count}
            for r in res.recommendations
        ],
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/product-frequency")
async def product_frequency(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze how often each product appears across transactions.
    Body: {"transaction_column": "...", "product_column": "...", "customer_column": "..."}
    """
    from business_brain.discovery.market_basket import analyze_product_frequency
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    if not all([transaction_col, product_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column and product_column required"}, 400)
    customer_col = body.get("customer_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_product_frequency(rows, transaction_col, product_col, customer_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "products": [
            {"product": p.product, "frequency": p.frequency, "pct_of_transactions": p.pct_of_transactions, "unique_customers": p.unique_customers, "rank": p.rank}
            for p in res.products
        ],
        "total_transactions": res.total_transactions,
        "total_products": res.total_products,
        "most_popular": res.most_popular,
        "least_popular": res.least_popular,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# SLA Monitor endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/sla-compliance")
async def sla_compliance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze SLA compliance by comparing actual vs target for each ticket.
    Body: {"ticket_column": "...", "sla_target_column": "...", "actual_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.sla_monitor import analyze_sla_compliance
    ticket_col = body.get("ticket_column")
    sla_target_col = body.get("sla_target_column")
    actual_col = body.get("actual_column")
    if not all([ticket_col, sla_target_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column, sla_target_column, and actual_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sla_compliance(rows, ticket_col, sla_target_col, actual_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_tickets": res.total_tickets,
        "met_count": res.met_count,
        "breached_count": res.breached_count,
        "compliance_rate": res.compliance_rate,
        "by_category": [
            {"category": c.category, "total": c.total, "met": c.met, "breached": c.breached, "compliance_rate": c.compliance_rate, "avg_actual": c.avg_actual, "avg_target": c.avg_target}
            for c in res.by_category
        ],
        "worst_category": res.worst_category,
        "avg_performance_ratio": res.avg_performance_ratio,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/response-times")
async def response_times(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze response time distribution across tickets.
    Body: {"ticket_column": "...", "response_time_column": "...", "priority_column": "...", "agent_column": "..."}
    """
    from business_brain.discovery.sla_monitor import analyze_response_times
    ticket_col = body.get("ticket_column")
    response_time_col = body.get("response_time_column")
    if not all([ticket_col, response_time_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column and response_time_column required"}, 400)
    priority_col = body.get("priority_column")
    agent_col = body.get("agent_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_response_times(rows, ticket_col, response_time_col, priority_col, agent_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_response_time": res.avg_response_time,
        "median_response_time": res.median_response_time,
        "p95_response_time": res.p95_response_time,
        "min_time": res.min_time,
        "max_time": res.max_time,
        "by_priority": [
            {"priority": p.priority, "count": p.count, "avg_time": p.avg_time, "median_time": p.median_time, "p95_time": p.p95_time}
            for p in res.by_priority
        ],
        "by_agent": [
            {"agent": a.agent, "count": a.count, "avg_time": a.avg_time, "compliance_rate": a.compliance_rate}
            for a in res.by_agent
        ],
        "outlier_count": res.outlier_count,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/resolution-metrics")
async def resolution_metrics(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute ticket resolution metrics.
    Body: {"ticket_column": "...", "created_column": "...", "resolved_column": "...", "status_column": "...", "priority_column": "..."}
    """
    from business_brain.discovery.sla_monitor import compute_resolution_metrics
    ticket_col = body.get("ticket_column")
    created_col = body.get("created_column")
    resolved_col = body.get("resolved_column")
    if not all([ticket_col, created_col, resolved_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column, created_column, and resolved_column required"}, 400)
    status_col = body.get("status_column")
    priority_col = body.get("priority_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_resolution_metrics(rows, ticket_col, created_col, resolved_col, status_col, priority_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_tickets": res.total_tickets,
        "resolved_count": res.resolved_count,
        "open_count": res.open_count,
        "resolution_rate": res.resolution_rate,
        "avg_resolution_hours": res.avg_resolution_hours,
        "median_resolution_hours": res.median_resolution_hours,
        "by_priority": [
            {"priority": p.priority, "total": p.total, "resolved": p.resolved, "avg_resolution_hours": p.avg_resolution_hours, "resolution_rate": p.resolution_rate}
            for p in res.by_priority
        ],
        "backlog_age_avg_hours": res.backlog_age_avg_hours,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/sla-trends")
async def sla_trends(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Track SLA compliance rate over time periods.
    Body: {"ticket_column": "...", "sla_met_column": "...", "date_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.sla_monitor import analyze_sla_trends
    ticket_col = body.get("ticket_column")
    sla_met_col = body.get("sla_met_column")
    date_col = body.get("date_column")
    if not all([ticket_col, sla_met_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column, sla_met_column, and date_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sla_trends(rows, ticket_col, sla_met_col, date_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "total": p.total, "met": p.met, "compliance_rate": p.compliance_rate}
            for p in res.periods
        ],
        "trend_direction": res.trend_direction,
        "overall_compliance": res.overall_compliance,
        "best_period": res.best_period,
        "worst_period": res.worst_period,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Geo Analytics endpoints
# ---------------------------------------------------------------------------


@app.post("/tables/{table_name}/regional-distribution")
async def regional_distribution(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Breakdown of value by region with concentration analysis.
    Body: {"region_column": "...", "value_column": "...", "count_column": "..."}
    """
    from business_brain.discovery.geo_analytics import analyze_regional_distribution
    region_col = body.get("region_column")
    value_col = body.get("value_column")
    if not all([region_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column and value_column required"}, 400)
    count_col = body.get("count_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_regional_distribution(rows, region_col, value_col, count_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "regions": [
            {"region": r.region, "total_value": r.total_value, "share_pct": r.share_pct, "count": r.count, "avg_value": r.avg_value, "rank": r.rank}
            for r in res.regions
        ],
        "total_value": res.total_value,
        "total_count": res.total_count,
        "top_region": res.top_region,
        "concentration_ratio": res.concentration_ratio,
        "hhi_index": res.hhi_index,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/region-comparison")
async def region_comparison(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compare multiple metrics across regions.
    Body: {"region_column": "...", "metric_columns": ["col1", "col2", ...]}
    """
    from business_brain.discovery.geo_analytics import compare_regions
    region_col = body.get("region_column")
    metric_cols = body.get("metric_columns")
    if not all([region_col, metric_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column and metric_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compare_regions(rows, region_col, metric_cols)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "comparisons": [
            {"metric": c.metric, "best_region": c.best_region, "best_value": c.best_value, "worst_region": c.worst_region, "worst_value": c.worst_value, "avg_value": c.avg_value, "std_dev": c.std_dev}
            for c in res.comparisons
        ],
        "region_scores": [
            {"region": rs.region, "metrics": rs.metrics, "overall_score": rs.overall_score}
            for rs in res.region_scores
        ],
        "best_overall": res.best_overall,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/geographic-growth")
async def geographic_growth(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Growth rates per region over time.
    Body: {"region_column": "...", "value_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.geo_analytics import analyze_geographic_growth
    region_col = body.get("region_column")
    value_col = body.get("value_column")
    date_col = body.get("date_column")
    if not all([region_col, value_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column, value_column, and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_geographic_growth(rows, region_col, value_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "regions": [
            {"region": r.region, "first_period_value": r.first_period_value, "last_period_value": r.last_period_value, "growth_rate": r.growth_rate, "periods": r.periods}
            for r in res.regions
        ],
        "fastest_growing": res.fastest_growing,
        "slowest_growing": res.slowest_growing,
        "avg_growth": res.avg_growth,
        "summary": res.summary,
    }


@app.post("/tables/{table_name}/market-penetration")
async def market_penetration(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Unique customers per region and market penetration.
    Body: {"region_column": "...", "customer_column": "...", "potential_column": "..."}
    """
    from business_brain.discovery.geo_analytics import compute_market_penetration
    region_col = body.get("region_column")
    customer_col = body.get("customer_column")
    if not all([region_col, customer_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column and customer_column required"}, 400)
    potential_col = body.get("potential_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_market_penetration(rows, region_col, customer_col, potential_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "regions": [
            {"region": r.region, "customer_count": r.customer_count, "potential": r.potential, "penetration_pct": r.penetration_pct, "rank": r.rank}
            for r in res.regions
        ],
        "total_customers": res.total_customers,
        "total_regions": res.total_regions,
        "best_penetration": res.best_penetration,
        "untapped_regions": res.untapped_regions,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Context Management Endpoints (for structured context visibility & editing)
# ---------------------------------------------------------------------------


@app.get("/context/entries")
async def list_context_entries(session: AsyncSession = Depends(get_session)) -> dict:
    """List all business context entries grouped by source."""
    from business_brain.memory.vector_store import list_all_contexts
    entries = await list_all_contexts(session, active_only=True)

    # Group by source
    grouped: dict[str, list] = {}
    for entry in entries:
        source = entry["source"]
        if source not in grouped:
            grouped[source] = []
        grouped[source].append(entry)

    return {"sources": grouped, "total": len(entries)}


@app.put("/context/entries/{entry_id}")
async def update_context_entry(
    entry_id: int,
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Edit a single context entry's text. Re-embeds the content."""
    from sqlalchemy import select
    from business_brain.db.models import BusinessContext
    from business_brain.ingestion.embeddings import embed_text

    result = await session.execute(
        select(BusinessContext).where(BusinessContext.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    if not entry:
        return {"error": "Context entry not found"}

    new_content = body.get("content", "").strip()
    if not new_content:
        return {"error": "Content cannot be empty"}

    # Update content and re-embed
    entry.content = new_content
    entry.embedding = embed_text(new_content)
    await session.commit()

    return {"status": "updated", "id": entry_id}


@app.delete("/context/entries/{entry_id}")
async def delete_context_entry(
    entry_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Soft-delete a context entry (mark as inactive)."""
    from datetime import datetime, timezone
    from sqlalchemy import select
    from business_brain.db.models import BusinessContext

    result = await session.execute(
        select(BusinessContext).where(BusinessContext.id == entry_id)
    )
    entry = result.scalar_one_or_none()
    if not entry:
        return {"error": "Context entry not found"}

    entry.active = False
    entry.superseded_at = datetime.now(timezone.utc)
    await session.commit()

    return {"status": "deleted", "id": entry_id}


# ---------------------------------------------------------------------------
# Process Steps Endpoints (structured process map)
# ---------------------------------------------------------------------------


class ProcessStepRequest(BaseModel):
    step_order: int = 0
    process_name: str
    inputs: str = ""
    outputs: str = ""
    key_metric: str = ""
    key_metrics: list[str] = []    # multiple metrics
    target_range: str = ""
    target_ranges: dict[str, str] = {}  # per-metric ranges: {"SEC": "500-625 kWh/ton"}
    linked_table: str = ""


@app.get("/process-steps")
async def list_process_steps(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all process steps ordered by step_order."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessStep

    result = await session.execute(
        select(ProcessStep).order_by(ProcessStep.step_order)
    )
    steps = result.scalars().all()
    return [
        {
            "id": s.id,
            "step_order": s.step_order,
            "process_name": s.process_name,
            "inputs": s.inputs,
            "outputs": s.outputs,
            "key_metric": s.key_metric,
            "key_metrics": s.key_metrics or ([s.key_metric] if s.key_metric else []),
            "target_range": s.target_range,
            "target_ranges": s.target_ranges or {},
            "linked_table": s.linked_table,
        }
        for s in steps
    ]


@app.post("/process-steps")
async def create_process_step(
    req: ProcessStepRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Add a new process step."""
    from business_brain.db.v3_models import ProcessStep

    # Support both key_metric (singular) and key_metrics (plural)
    key_metrics = req.key_metrics if req.key_metrics else ([req.key_metric] if req.key_metric else [])
    key_metric_single = key_metrics[0] if key_metrics else req.key_metric

    # Per-metric ranges with backward compat
    target_ranges = req.target_ranges or {}
    if req.target_range and not target_ranges and key_metrics:
        target_ranges = {key_metrics[0]: req.target_range}

    step = ProcessStep(
        step_order=req.step_order,
        process_name=req.process_name,
        inputs=req.inputs,
        outputs=req.outputs,
        key_metric=key_metric_single,
        key_metrics=key_metrics,
        target_range=req.target_range or (target_ranges.get(key_metrics[0], "") if key_metrics else ""),
        target_ranges=target_ranges,
        linked_table=req.linked_table,
    )
    session.add(step)
    await session.commit()
    await session.refresh(step)

    # Auto-regenerate RAG context from process steps
    await _regenerate_process_context(session)

    return {"status": "created", "id": step.id}


@app.put("/process-steps/{step_id}")
async def update_process_step(
    step_id: int,
    req: ProcessStepRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update an existing process step."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessStep

    result = await session.execute(
        select(ProcessStep).where(ProcessStep.id == step_id)
    )
    step = result.scalar_one_or_none()
    if not step:
        return {"error": "Process step not found"}

    key_metrics = req.key_metrics if req.key_metrics else ([req.key_metric] if req.key_metric else [])
    key_metric_single = key_metrics[0] if key_metrics else req.key_metric

    # Per-metric ranges with backward compat
    target_ranges = req.target_ranges or {}
    if req.target_range and not target_ranges and key_metrics:
        target_ranges = {key_metrics[0]: req.target_range}

    step.step_order = req.step_order
    step.process_name = req.process_name
    step.inputs = req.inputs
    step.outputs = req.outputs
    step.key_metric = key_metric_single
    step.key_metrics = key_metrics
    step.target_range = req.target_range or (target_ranges.get(key_metrics[0], "") if key_metrics else "")
    step.target_ranges = target_ranges
    step.linked_table = req.linked_table
    await session.commit()

    await _regenerate_process_context(session)

    return {"status": "updated", "id": step.id}


@app.delete("/process-steps/{step_id}")
async def delete_process_step(
    step_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Delete a process step."""
    from sqlalchemy import select, delete
    from business_brain.db.v3_models import ProcessStep

    result = await session.execute(
        select(ProcessStep).where(ProcessStep.id == step_id)
    )
    step = result.scalar_one_or_none()
    if not step:
        return {"error": "Process step not found"}

    await session.execute(delete(ProcessStep).where(ProcessStep.id == step_id))
    await session.commit()

    await _regenerate_process_context(session)

    return {"status": "deleted", "id": step_id}


# ---------------------------------------------------------------------------
# Process I/O Endpoints (inputs & outputs map)
# ---------------------------------------------------------------------------


class ProcessIORequest(BaseModel):
    io_type: str  # "input" or "output"
    name: str
    source_or_destination: str = ""
    unit: str = ""
    typical_range: str = ""
    linked_table: str = ""


@app.get("/process-io")
async def list_process_io(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all process inputs and outputs."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessIO

    result = await session.execute(
        select(ProcessIO).order_by(ProcessIO.io_type, ProcessIO.name)
    )
    ios = result.scalars().all()
    return [
        {
            "id": io.id,
            "io_type": io.io_type,
            "name": io.name,
            "source_or_destination": io.source_or_destination,
            "unit": io.unit,
            "typical_range": io.typical_range,
            "linked_table": io.linked_table,
        }
        for io in ios
    ]


@app.post("/process-io")
async def create_process_io(
    req: ProcessIORequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Add a new process input or output."""
    from business_brain.db.v3_models import ProcessIO

    io = ProcessIO(
        io_type=req.io_type,
        name=req.name,
        source_or_destination=req.source_or_destination,
        unit=req.unit,
        typical_range=req.typical_range,
        linked_table=req.linked_table,
    )
    session.add(io)
    await session.commit()
    await session.refresh(io)

    await _regenerate_io_context(session)

    return {"status": "created", "id": io.id}


@app.put("/process-io/{io_id}")
async def update_process_io(
    io_id: int,
    req: ProcessIORequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Update an existing process I/O entry."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessIO

    result = await session.execute(
        select(ProcessIO).where(ProcessIO.id == io_id)
    )
    io = result.scalar_one_or_none()
    if not io:
        return {"error": "Process I/O entry not found"}

    io.io_type = req.io_type
    io.name = req.name
    io.source_or_destination = req.source_or_destination
    io.unit = req.unit
    io.typical_range = req.typical_range
    io.linked_table = req.linked_table
    await session.commit()

    await _regenerate_io_context(session)

    return {"status": "updated", "id": io_id}


@app.delete("/process-io/{io_id}")
async def delete_process_io(
    io_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Delete a process I/O entry."""
    from sqlalchemy import select, delete
    from business_brain.db.v3_models import ProcessIO

    result = await session.execute(
        select(ProcessIO).where(ProcessIO.id == io_id)
    )
    io = result.scalar_one_or_none()
    if not io:
        return {"error": "Process I/O entry not found"}

    await session.execute(delete(ProcessIO).where(ProcessIO.id == io_id))
    await session.commit()

    await _regenerate_io_context(session)

    return {"status": "deleted", "id": io_id}


# ---------------------------------------------------------------------------
# Setup Intelligence — Auto-link, Suggest, Derived Metrics, Process-Metric Links
# ---------------------------------------------------------------------------


@app.post("/setup/auto-link-metrics")
async def auto_link_metrics(session: AsyncSession = Depends(get_session)) -> dict:
    """Auto-suggest table/column mappings for all configured metrics.

    Uses fuzzy matching of metric names against column names across all tables.
    High-confidence matches (>=0.8) are auto-linked; lower ones returned as suggestions.
    """
    from sqlalchemy import select
    from business_brain.db.v3_models import MetricThreshold

    # Get unlinked metrics
    result = await session.execute(
        select(MetricThreshold).where(
            (MetricThreshold.table_name == None) | (MetricThreshold.table_name == "")  # noqa: E711
        )
    )
    unlinked = list(result.scalars().all())

    # Get all table metadata
    all_entries = await metadata_store.get_all(session)

    auto_linked = []
    suggestions = []
    unmatched = []

    for metric in unlinked:
        metric_lower = metric.metric_name.lower().replace(" ", "_").replace("-", "_")
        metric_words = set(metric.metric_name.lower().replace("_", " ").replace("-", " ").split())
        best_candidates = []

        for entry in all_entries:
            if not entry.columns_metadata:
                continue
            for col in entry.columns_metadata:
                col_name = col.get("name", "")
                col_lower = col_name.lower().replace(" ", "_").replace("-", "_")

                # Scoring
                confidence = 0.0
                if col_lower == metric_lower:
                    confidence = 1.0
                elif col_lower.replace("_", "") == metric_lower.replace("_", ""):
                    confidence = 0.95
                elif metric_lower in col_lower or col_lower in metric_lower:
                    confidence = 0.7
                else:
                    col_words = set(col_lower.replace("_", " ").split())
                    overlap = metric_words & col_words
                    if overlap and len(overlap) >= max(1, len(metric_words) // 2):
                        confidence = 0.3 + 0.2 * len(overlap)

                if confidence > 0.2:
                    best_candidates.append({
                        "table_name": entry.table_name,
                        "column_name": col_name,
                        "confidence": round(confidence, 2),
                    })

        best_candidates.sort(key=lambda x: x["confidence"], reverse=True)

        if best_candidates and best_candidates[0]["confidence"] >= 0.8:
            best = best_candidates[0]
            metric.table_name = best["table_name"]
            metric.column_name = best["column_name"]
            metric.auto_linked = True
            metric.confidence = best["confidence"]
            auto_linked.append({
                "metric_id": metric.id,
                "metric_name": metric.metric_name,
                **best,
            })
        elif best_candidates:
            suggestions.append({
                "metric_id": metric.id,
                "metric_name": metric.metric_name,
                "candidates": best_candidates[:5],
            })
        else:
            unmatched.append({
                "metric_id": metric.id,
                "metric_name": metric.metric_name,
            })

    if auto_linked:
        await session.commit()

    return {
        "auto_linked": auto_linked,
        "suggestions": suggestions,
        "unmatched": unmatched,
    }


@app.post("/setup/suggest-metrics")
async def suggest_metrics(session: AsyncSession = Depends(get_session)) -> dict:
    """Analyze uploaded data and suggest trackable metrics from numeric columns."""
    from sqlalchemy import select
    from business_brain.db.v3_models import MetricThreshold

    all_entries = await metadata_store.get_all(session)
    if not all_entries:
        return {"suggestions": [], "message": "No data uploaded yet"}

    # Get already-configured metric names
    result = await session.execute(select(MetricThreshold.metric_name))
    existing = {row[0].lower() for row in result.fetchall()}

    suggestions_by_table = []
    for entry in all_entries:
        if not entry.columns_metadata:
            continue
        table_suggestions = []
        for col in entry.columns_metadata:
            col_name = col.get("name", "")
            col_type = col.get("type", "").lower()
            # Only suggest numeric columns
            if any(t in col_type for t in ("int", "float", "numeric", "decimal", "double", "real", "bigint")):
                # Skip if already configured
                if col_name.lower() in existing:
                    continue
                # Skip ID-like columns
                if col_name.lower().endswith("_id") or col_name.lower() == "id":
                    continue
                table_suggestions.append({
                    "column_name": col_name,
                    "column_type": col_type,
                    "suggested_metric_name": col_name.replace("_", " ").title(),
                })
        if table_suggestions:
            suggestions_by_table.append({
                "table_name": entry.table_name,
                "columns": table_suggestions,
            })

    return {"suggestions": suggestions_by_table}


@app.post("/metrics/derived")
async def create_derived_metric(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Create a derived/calculated metric with a formula."""
    from business_brain.db.v3_models import MetricThreshold

    metric_name = body.get("metric_name", "").strip()
    formula = body.get("formula", "").strip()
    if not metric_name:
        raise HTTPException(status_code=400, detail="metric_name is required")
    if not formula:
        raise HTTPException(status_code=400, detail="formula is required")

    # Parse source columns from formula (look for table.column references)
    import re
    source_refs = re.findall(r'(\w+)\.(\w+)', formula)
    source_columns = [f"{t}.{c}" for t, c in source_refs]

    # Validate that referenced columns exist
    all_entries = await metadata_store.get_all(session)
    table_columns = {}
    for entry in all_entries:
        if entry.columns_metadata:
            table_columns[entry.table_name] = {c.get("name", "") for c in entry.columns_metadata}

    for ref in source_refs:
        table, col = ref
        if table not in table_columns:
            raise HTTPException(status_code=400, detail=f"Table '{table}' not found")
        if col not in table_columns[table]:
            raise HTTPException(status_code=400, detail=f"Column '{col}' not found in table '{table}'")

    metric = MetricThreshold(
        metric_name=metric_name,
        is_derived=True,
        formula=formula,
        source_columns=source_columns,
        unit=body.get("unit", ""),
        normal_min=body.get("normal_min"),
        normal_max=body.get("normal_max"),
        warning_min=body.get("warning_min"),
        warning_max=body.get("warning_max"),
        critical_min=body.get("critical_min"),
        critical_max=body.get("critical_max"),
    )
    session.add(metric)
    await session.commit()
    await session.refresh(metric)

    return {"status": "created", "id": metric.id, "metric_name": metric_name, "is_derived": True}


@app.post("/process-steps/{step_id}/metrics")
async def link_metrics_to_step(
    step_id: int,
    body: dict,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Link one or more metrics to a process step."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessStep, ProcessMetricLink, MetricThreshold

    # Validate step exists
    result = await session.execute(select(ProcessStep).where(ProcessStep.id == step_id))
    step = result.scalar_one_or_none()
    if not step:
        raise HTTPException(status_code=404, detail="Process step not found")

    metric_ids = body.get("metric_ids", [])
    if not metric_ids:
        raise HTTPException(status_code=400, detail="metric_ids array is required")

    linked = []
    for mid in metric_ids:
        # Validate metric exists
        m_result = await session.execute(select(MetricThreshold).where(MetricThreshold.id == mid))
        metric = m_result.scalar_one_or_none()
        if not metric:
            continue

        # Check for duplicate link
        existing = await session.execute(
            select(ProcessMetricLink).where(
                ProcessMetricLink.process_step_id == step_id,
                ProcessMetricLink.metric_id == mid,
            )
        )
        if existing.scalar_one_or_none():
            continue

        link = ProcessMetricLink(
            process_step_id=step_id,
            metric_id=mid,
            is_primary=len(linked) == 0,  # first one is primary
        )
        session.add(link)
        linked.append(mid)

    await session.commit()
    return {"status": "linked", "step_id": step_id, "metrics_linked": linked}


@app.get("/process-steps/{step_id}/metrics")
async def get_step_metrics(
    step_id: int,
    session: AsyncSession = Depends(get_session),
) -> list[dict]:
    """Get all metrics linked to a process step."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessMetricLink, MetricThreshold

    result = await session.execute(
        select(ProcessMetricLink).where(ProcessMetricLink.process_step_id == step_id)
    )
    links = result.scalars().all()

    metrics = []
    for link in links:
        m_result = await session.execute(
            select(MetricThreshold).where(MetricThreshold.id == link.metric_id)
        )
        metric = m_result.scalar_one_or_none()
        if metric:
            metrics.append({
                "link_id": link.id,
                "metric_id": metric.id,
                "metric_name": metric.metric_name,
                "table_name": metric.table_name,
                "column_name": metric.column_name,
                "unit": metric.unit,
                "is_primary": link.is_primary,
                "is_derived": metric.is_derived or False,
                "auto_linked": metric.auto_linked or False,
                "confidence": metric.confidence,
            })

    return metrics


@app.delete("/process-steps/{step_id}/metrics/{metric_id}")
async def unlink_metric_from_step(
    step_id: int,
    metric_id: int,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Unlink a metric from a process step."""
    from sqlalchemy import delete as sa_delete
    from business_brain.db.v3_models import ProcessMetricLink

    result = await session.execute(
        sa_delete(ProcessMetricLink).where(
            ProcessMetricLink.process_step_id == step_id,
            ProcessMetricLink.metric_id == metric_id,
        )
    )
    await session.commit()
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Link not found")
    return {"status": "unlinked", "step_id": step_id, "metric_id": metric_id}


# ---------------------------------------------------------------------------
# Helpers: Auto-regenerate RAG context from structured data
# ---------------------------------------------------------------------------


async def _regenerate_process_context(session: AsyncSession) -> None:
    """Generate natural language context from process steps and reingest."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessStep
    from business_brain.ingestion.context_ingestor import reingest_context

    result = await session.execute(
        select(ProcessStep).order_by(ProcessStep.step_order)
    )
    steps = result.scalars().all()
    if not steps:
        return

    lines = ["PRODUCTION PROCESS MAP:"]
    for s in steps:
        line = f"Step {s.step_order}: {s.process_name}"
        if s.inputs:
            line += f" | Inputs: {s.inputs}"
        if s.outputs:
            line += f" | Outputs: {s.outputs}"
        if s.key_metrics:
            line += f" | Metrics: {', '.join(s.key_metrics)}"
        elif s.key_metric:
            line += f" | Key Metric: {s.key_metric}"
        if s.target_range:
            line += f" (Target: {s.target_range})"
        if s.linked_table:
            line += f" | Data: {s.linked_table}"
        lines.append(line)

    text = "\n".join(lines)
    await reingest_context(text, session, source="structured:process_map")


async def _regenerate_io_context(session: AsyncSession) -> None:
    """Generate natural language context from process I/O and reingest."""
    from sqlalchemy import select
    from business_brain.db.v3_models import ProcessIO
    from business_brain.ingestion.context_ingestor import reingest_context

    result = await session.execute(
        select(ProcessIO).order_by(ProcessIO.io_type, ProcessIO.name)
    )
    ios = result.scalars().all()
    if not ios:
        return

    inputs = [io for io in ios if io.io_type == "input"]
    outputs = [io for io in ios if io.io_type == "output"]

    lines = ["PROCESS INPUTS AND OUTPUTS:"]

    if inputs:
        lines.append("INPUTS:")
        for io in inputs:
            line = f"  - {io.name}"
            if io.source_or_destination:
                line += f" (from {io.source_or_destination})"
            if io.unit:
                line += f" [{io.unit}]"
            if io.typical_range:
                line += f" typical: {io.typical_range}"
            if io.linked_table:
                line += f" → table: {io.linked_table}"
            lines.append(line)

    if outputs:
        lines.append("OUTPUTS:")
        for io in outputs:
            line = f"  - {io.name}"
            if io.source_or_destination:
                line += f" (to {io.source_or_destination})"
            if io.unit:
                line += f" [{io.unit}]"
            if io.typical_range:
                line += f" typical: {io.typical_range}"
            if io.linked_table:
                line += f" → table: {io.linked_table}"
            lines.append(line)

    text = "\n".join(lines)
    await reingest_context(text, session, source="structured:process_io")


# ---------------------------------------------------------------------------
# Industry Setup Templates
# ---------------------------------------------------------------------------


@app.get("/setup/template/{industry}")
async def get_industry_setup_template(industry: str) -> dict:
    """Get pre-built setup template for a given industry."""
    from business_brain.cognitive.domain_knowledge import (
        get_industry_template,
        get_whatif_templates,
    )

    template = get_industry_template(industry)
    if not template:
        return {"error": f"No template available for industry: {industry}"}

    return {
        "industry": industry,
        "process_steps": template.get("process_steps", []),
        "metrics": template.get("metrics", []),
        "inputs": template.get("inputs", []),
        "outputs": template.get("outputs", []),
        "whatif_templates": get_whatif_templates(industry),
    }


@app.post("/setup/apply-template/{industry}")
async def apply_industry_template(
    industry: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Apply industry template — bulk-create process steps, metrics, and I/O."""
    from business_brain.cognitive.domain_knowledge import get_industry_template
    from business_brain.db.v3_models import MetricThreshold, ProcessIO, ProcessStep

    template = get_industry_template(industry)
    if not template:
        return {"error": f"No template available for industry: {industry}"}

    counts = {"process_steps": 0, "metrics": 0, "inputs": 0, "outputs": 0}

    for step_data in template.get("process_steps", []):
        session.add(ProcessStep(**step_data))
        counts["process_steps"] += 1

    for metric_data in template.get("metrics", []):
        session.add(MetricThreshold(**metric_data))
        counts["metrics"] += 1

    for io_data in template.get("inputs", []) + template.get("outputs", []):
        session.add(ProcessIO(**io_data))
        counts["inputs" if io_data["io_type"] == "input" else "outputs"] += 1

    await session.commit()

    # Regenerate RAG context
    await _regenerate_process_context(session)
    await _regenerate_io_context(session)

    return {"status": "applied", "industry": industry, "counts": counts}


# ---------------------------------------------------------------------------
# What-If Templates
# ---------------------------------------------------------------------------


@app.get("/whatif/templates")
async def get_whatif_scenario_templates(industry: str = "steel") -> list[dict]:
    """Get pre-built What-If scenario templates for the given industry."""
    from business_brain.cognitive.domain_knowledge import get_whatif_templates

    return get_whatif_templates(industry)


# ---------------------------------------------------------------------------
# Save Analysis to Feed (Analyze → Feed parity)
# ---------------------------------------------------------------------------


class SaveToFeedRequest(BaseModel):
    title: str
    description: str
    insight_type: str = "analysis"
    severity: str = "info"
    evidence: Optional[dict] = None
    suggested_actions: Optional[list[str]] = None
    source_tables: Optional[list[str]] = None


@app.post("/feed/from-analysis")
async def save_analysis_to_feed(
    req: SaveToFeedRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Save an analysis result as a manual insight in the Feed."""
    from business_brain.db.discovery_models import Insight

    insight = Insight(
        insight_type=req.insight_type,
        severity=req.severity,
        impact_score=75,  # Manual = presumed high quality
        quality_score=75,
        title=req.title,
        description=req.description,
        source_tables=req.source_tables or [],
        evidence=req.evidence or {},
        suggested_actions=req.suggested_actions or [],
        status="new",
    )
    session.add(insight)
    await session.commit()
    await session.refresh(insight)

    return {"status": "created", "insight_id": insight.id, "title": req.title}


# ---------------------------------------------------------------------------
# Plan Limits — Free Tier Enforcement
# ---------------------------------------------------------------------------

PLAN_LIMITS = {
    "free": {
        "max_uploads": 3,
        "google_sheets": False,
        "api_connections": False,
        "analyze_per_day": 5,
        "reports": False,
        "alerts": False,
        "setup": False,
        "whatif": False,
        "deploy": False,
        "export": False,
        "max_users": 1,
    },
    "basic": {
        "max_uploads": 10,
        "google_sheets": True,
        "api_connections": True,
        "analyze_per_day": 50,
        "reports": True,
        "alerts": True,
        "setup": True,
        "whatif": True,
        "deploy": True,
        "export": True,
        "max_users": 3,
    },
    "pro": {
        "max_uploads": 999999,
        "google_sheets": True,
        "api_connections": True,
        "analyze_per_day": 999999,
        "reports": True,
        "alerts": True,
        "setup": True,
        "whatif": True,
        "deploy": True,
        "export": True,
        "max_users": 10,
    },
    "enterprise": {
        "max_uploads": 999999,
        "google_sheets": True,
        "api_connections": True,
        "analyze_per_day": 999999,
        "reports": True,
        "alerts": True,
        "setup": True,
        "whatif": True,
        "deploy": True,
        "export": True,
        "max_users": 999999,
    },
}


@app.get("/plan/limits")
async def get_plan_limits(
    authorization: str = Header(default=""),
) -> dict:
    """Get current user's plan limits and usage."""
    user_data = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        user_data = _decode_jwt(token)

    plan = user_data.get("plan", "free") if user_data else "pro"  # No auth = full access
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

    return {
        "plan": plan,
        "limits": limits,
        "upload_count": 0,  # Would need DB lookup for real count
    }


# ---------------------------------------------------------------------------
# Focus Mode — Table Scoping
# ---------------------------------------------------------------------------


@app.get("/focus")
async def get_focus(session: AsyncSession = Depends(get_session)) -> dict:
    """Get current focus scope (which tables are included/excluded)."""
    from sqlalchemy import select
    from business_brain.db.v3_models import FocusScope

    try:
        result = await session.execute(select(FocusScope))
        rows = list(result.scalars().all())
        all_entries = await metadata_store.get_all(session)
        all_table_names = [e.table_name for e in all_entries]

        focused = {r.table_name: r.is_included for r in rows}
        tables = []
        for t in all_table_names:
            tables.append({
                "table_name": t,
                "is_included": focused.get(t, True),  # default to included
            })

        active = any(not v for v in focused.values()) or (len(focused) > 0 and len(focused) < len(all_table_names))
        return {"active": bool(rows) and active, "tables": tables, "total": len(all_table_names)}
    except Exception:
        logger.exception("Error fetching focus scope")
        return {"active": False, "tables": [], "total": 0}


@app.put("/focus")
async def update_focus(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Update focus scope — set which tables are included/excluded."""
    from sqlalchemy import select
    from business_brain.db.v3_models import FocusScope

    tables = body.get("tables", [])
    if not tables:
        raise HTTPException(status_code=400, detail="'tables' array is required")

    # Validate that all referenced tables exist
    all_entries = await metadata_store.get_all(session)
    valid_names = {e.table_name for e in all_entries}
    for t in tables:
        if t.get("table_name") not in valid_names:
            raise HTTPException(
                status_code=400,
                detail=f"Table '{t.get('table_name')}' not found in metadata"
            )

    # Clear existing scope and recreate
    await session.execute(
        select(FocusScope)  # Just to init; actual delete below
    )
    from sqlalchemy import delete as sa_delete
    await session.execute(sa_delete(FocusScope))

    for t in tables:
        scope = FocusScope(
            table_name=t["table_name"],
            is_included=t.get("is_included", True),
        )
        session.add(scope)

    await session.commit()

    included = sum(1 for t in tables if t.get("is_included", True))
    return {"status": "updated", "total": len(tables), "included": included}


@app.delete("/focus")
async def clear_focus(session: AsyncSession = Depends(get_session)) -> dict:
    """Clear focus scope — disable focus mode (analyze all tables)."""
    from business_brain.db.v3_models import FocusScope
    from sqlalchemy import delete as sa_delete

    result = await session.execute(sa_delete(FocusScope))
    await session.commit()
    return {"status": "cleared", "rows_removed": result.rowcount}


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class InviteRequest(BaseModel):
    email: str
    role: str = "viewer"
    plan: str = "free"


class AcceptInviteRequest(BaseModel):
    token: str
    name: str
    password: str


class UpdateRoleRequest(BaseModel):
    role: str


@app.post("/auth/register")
async def register_user(
    req: RegisterRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Register a new user. First user auto-becomes owner."""
    from sqlalchemy import func, select

    from business_brain.db.v3_models import User

    # Check if email already exists
    existing = await session.execute(select(User).where(User.email == req.email))
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    # Count existing users to determine if this is the first user
    count_result = await session.execute(select(func.count()).select_from(User))
    user_count = count_result.scalar() or 0

    user = User(
        email=req.email,
        name=req.name,
        password_hash=_hash_password(req.password),
        role="owner" if user_count == 0 else "viewer",
        plan="pro" if user_count == 0 else "free",
        is_active=True,
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)

    token = _create_jwt(user.id, user.email, user.role, user.plan)
    return {
        "status": "registered",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "plan": user.plan,
        },
    }


@app.post("/auth/login")
async def login_user(
    req: LoginRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Login and receive a JWT token."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    result = await session.execute(select(User).where(User.email == req.email))
    user = result.scalars().first()

    if not user or not _verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # Update last login
    user.last_login_at = datetime.utcnow()
    await session.commit()

    token = _create_jwt(user.id, user.email, user.role, user.plan)
    return {
        "status": "logged_in",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "plan": user.plan,
            "upload_count": user.upload_count,
        },
    }


@app.get("/auth/me")
async def get_me(
    user: dict = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get current authenticated user info."""
    if not user:
        return {"authenticated": False}

    from sqlalchemy import select

    from business_brain.db.v3_models import User

    result = await session.execute(select(User).where(User.id == user["sub"]))
    db_user = result.scalars().first()
    if not db_user:
        return {"authenticated": False}

    return {
        "authenticated": True,
        "user": {
            "id": db_user.id,
            "email": db_user.email,
            "name": db_user.name,
            "role": db_user.role,
            "plan": db_user.plan,
            "upload_count": db_user.upload_count,
            "is_active": db_user.is_active,
        },
    }


@app.post("/auth/invite")
async def create_invite(
    req: InviteRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> dict:
    """Create an invite token for a new user (admin/owner only)."""
    from business_brain.db.v3_models import InviteToken

    token_str = secrets.token_urlsafe(32)
    invite = InviteToken(
        email=req.email,
        role=req.role,
        plan=req.plan,
        token=token_str,
        expires_at=datetime.utcnow() + timedelta(days=7),
        created_by=user.get("sub"),
    )
    session.add(invite)
    await session.commit()

    return {"status": "created", "token": token_str, "email": req.email, "role": req.role}


@app.post("/auth/accept-invite")
async def accept_invite(
    req: AcceptInviteRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Accept an invite and create a new user account."""
    from sqlalchemy import select

    from business_brain.db.v3_models import InviteToken, User

    result = await session.execute(
        select(InviteToken).where(InviteToken.token == req.token, InviteToken.used == False)  # noqa: E712
    )
    invite = result.scalars().first()
    if not invite:
        raise HTTPException(status_code=400, detail="Invalid or expired invite token")

    if invite.expires_at and invite.expires_at < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invite token has expired")

    # Check if email already exists
    existing = await session.execute(select(User).where(User.email == invite.email))
    if existing.scalars().first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=invite.email,
        name=req.name,
        password_hash=_hash_password(req.password),
        role=invite.role,
        plan=invite.plan,
        company_id=invite.company_id,
        is_active=True,
    )
    session.add(user)

    invite.used = True
    await session.commit()
    await session.refresh(user)

    token = _create_jwt(user.id, user.email, user.role, user.plan)
    return {
        "status": "registered",
        "token": token,
        "user": {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "plan": user.plan,
        },
    }


@app.get("/users")
async def list_users(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> list[dict]:
    """List all users (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    result = await session.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return [
        {
            "id": u.id,
            "email": u.email,
            "name": u.name,
            "role": u.role,
            "plan": u.plan,
            "is_active": u.is_active,
            "upload_count": u.upload_count,
            "created_at": str(u.created_at) if u.created_at else None,
            "last_login_at": str(u.last_login_at) if u.last_login_at else None,
        }
        for u in users
    ]


@app.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    req: UpdateRoleRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> dict:
    """Change a user's role (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    if req.role not in ROLE_LEVELS:
        raise HTTPException(status_code=400, detail=f"Invalid role. Valid: {list(ROLE_LEVELS.keys())}")

    result = await session.execute(select(User).where(User.id == user_id))
    target = result.scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    # Can't change owner role unless you're the owner
    if target.role == "owner" and user.get("role") != "owner":
        raise HTTPException(status_code=403, detail="Only the owner can change another owner's role")

    target.role = req.role
    await session.commit()
    return {"status": "updated", "user_id": user_id, "new_role": req.role}


@app.delete("/users/{user_id}")
async def deactivate_user(
    user_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("owner")),
) -> dict:
    """Deactivate a user account (owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import User

    result = await session.execute(select(User).where(User.id == user_id))
    target = result.scalars().first()
    if not target:
        raise HTTPException(status_code=404, detail="User not found")

    if target.id == user.get("sub"):
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")

    target.is_active = False
    await session.commit()
    return {"status": "deactivated", "user_id": user_id}


@app.get("/auth/invites")
async def list_invites(
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> list[dict]:
    """List all pending (unused) invite tokens (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import InviteToken

    try:
        result = await session.execute(
            select(InviteToken)
            .where(InviteToken.used == False)  # noqa: E712
            .order_by(InviteToken.expires_at.desc())
        )
        invites = list(result.scalars().all())
        return [
            {
                "id": inv.id,
                "email": inv.email,
                "role": inv.role,
                "plan": inv.plan,
                "token": inv.token,
                "expires_at": str(inv.expires_at) if inv.expires_at else None,
                "created_by": inv.created_by,
                "expired": inv.expires_at is not None and inv.expires_at < datetime.utcnow(),
            }
            for inv in invites
        ]
    except Exception:
        logger.exception("Error listing invites")
        return []


@app.delete("/auth/invites/{invite_id}")
async def revoke_invite(
    invite_id: str,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(require_role("admin")),
) -> dict:
    """Revoke (delete) a pending invite token (admin/owner only)."""
    from sqlalchemy import select

    from business_brain.db.v3_models import InviteToken

    result = await session.execute(select(InviteToken).where(InviteToken.id == invite_id))
    invite = result.scalars().first()
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.used:
        raise HTTPException(status_code=400, detail="Invite already used, cannot revoke")

    await session.delete(invite)
    await session.commit()
    return {"status": "revoked", "invite_id": invite_id}
