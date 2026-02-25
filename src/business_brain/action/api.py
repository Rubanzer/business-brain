"""FastAPI application with external trigger endpoints."""

import asyncio
import hashlib
import hmac
import json
import logging
import os
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

app = FastAPI(title="Business Brain API", version="4.0.0")

# ---------------------------------------------------------------------------
# Include modular routers
# ---------------------------------------------------------------------------
from business_brain.action.routers.auth import router as auth_router
from business_brain.action.routers.table_analysis import router as table_analysis_router
from business_brain.action.routers.process import router as process_router

app.include_router(auth_router)
app.include_router(table_analysis_router)
app.include_router(process_router)

# Background sync task handle
_sync_task: Optional[asyncio.Task] = None


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
    # Include the actual error message so the frontend can display it
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"Server error: {exc}",
            "error": str(exc),
            "status": "failed",
        },
    )


@app.on_event("startup")
async def _startup_enrich_descriptions():
    """Auto-enrich column descriptions for all tables on startup.

    Uses fast pattern-matching (no LLM calls) to fill in or fix weak/wrong
    descriptions.  This ensures the SQL agent always has useful and ACCURATE
    column descriptions.

    Fixes:
      - Weak generic fallbacks (e.g. "Numeric (decimal) column: yield")
      - Misleading descriptions (e.g. "rate" described as "percentage")
    """
    try:
        from business_brain.discovery.data_dictionary import auto_describe_column

        async with async_session() as session:
            entries = await metadata_store.get_all(session)
            updated_count = 0
            for entry in entries:
                if not entry.columns_metadata:
                    continue
                changed = False
                for col in entry.columns_metadata:
                    old_desc = col.get("description", "")
                    col_name = col.get("name", "")
                    col_lower = col_name.lower()

                    # Regenerate if description is missing, or is a weak generic fallback
                    is_weak = (
                        not old_desc
                        or old_desc.startswith("Numeric (decimal) column:")
                        or old_desc.startswith("Integer column:")
                        or old_desc.startswith("Text column:")
                        or old_desc.startswith("Data column:")
                    )

                    # Also fix WRONG descriptions — "rate" described as percentage
                    is_wrong = False
                    if "rate" in col_lower and "Percentage or ratio" in old_desc:
                        is_wrong = True  # rate is price per unit, not percentage
                    if col_lower in ("yield", "yield_pct") and "quantity" in old_desc.lower():
                        is_wrong = True  # yield is percentage, not quantity

                    if is_weak or is_wrong:
                        new_desc = auto_describe_column(
                            col_name,
                            col.get("type", ""),
                            {},
                        )
                        # Only update if the new description is actually better
                        if new_desc and not new_desc.startswith(
                            ("Numeric (decimal) column:", "Integer column:", "Text column:", "Data column:")
                        ):
                            col["description"] = new_desc
                            changed = True

                if changed:
                    await metadata_store.upsert(
                        session,
                        table_name=entry.table_name,
                        description=entry.description or f"Table {entry.table_name}",
                        columns_metadata=entry.columns_metadata,
                    )
                    updated_count += 1
            if updated_count:
                logger.info("Startup: enriched column descriptions for %d table(s)", updated_count)
    except Exception:
        logger.debug("Startup column description enrichment failed — non-critical")


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

# CORS: lock down in production via CORS_ORIGINS env var (comma-separated).
# Falls back to "*" for local dev if not set.
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Rate limiting (lightweight, in-memory — resets on serverless cold start)
# ---------------------------------------------------------------------------
_rate_limit_store: dict = {}  # {ip_key: [timestamp, ...]}


def _check_rate_limit(client_ip: str, endpoint: str, max_requests: int, window_seconds: int) -> bool:
    """Return True if the request is within rate limits, False if exceeded.

    Uses a sliding window counter per IP + endpoint combination.
    """
    now = time.time()
    key = f"{client_ip}:{endpoint}"
    timestamps = _rate_limit_store.get(key, [])
    # Remove timestamps outside the window
    timestamps = [t for t in timestamps if now - t < window_seconds]
    if len(timestamps) >= max_requests:
        _rate_limit_store[key] = timestamps
        return False
    timestamps.append(now)
    _rate_limit_store[key] = timestamps
    return True


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
        "version": "4.0.0",
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
    try:
        result = await graph.ainvoke(invoke_state)
    except Exception as exc:
        logger.exception("Analysis pipeline failed for question: %s", req.question)
        return {
            "error": str(exc),
            "question": req.question,
            "session_id": session_id,
            "status": "failed",
        }

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
    # v4 fields
    result.setdefault("query_type", "custom")
    result.setdefault("query_confidence", 0.5)
    result.setdefault("key_metrics", [])
    result.setdefault("leakage_patterns", [])
    result.pop("router_reasoning", None)  # internal, don't expose
    result["session_id"] = session_id

    # Include pipeline diagnostics so frontend can show what happened
    diagnostics = result.pop("_diagnostics", [])
    result["diagnostics"] = diagnostics

    # Compute overall pipeline health from diagnostics
    errors = [d for d in diagnostics if d.get("status") == "error"]
    warnings = [d for d in diagnostics if d.get("status") == "warn"]
    total_ms = sum(d.get("duration_ms", 0) for d in diagnostics)

    if errors:
        result["pipeline_status"] = "partial"
        result["pipeline_message"] = f"{len(errors)} stage(s) had errors. Results may be incomplete."
    elif warnings:
        result["pipeline_status"] = "ok_with_warnings"
        result["pipeline_message"] = f"Analysis completed with {len(warnings)} warning(s)."
    else:
        result["pipeline_status"] = "ok"
        result["pipeline_message"] = "Analysis completed successfully."
    result["pipeline_duration_ms"] = total_ms

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
_JWT_SECRET = os.environ.get("JWT_SECRET", "")
if not _JWT_SECRET:
    _JWT_SECRET = secrets.token_hex(32)
    logger.warning(
        "JWT_SECRET not set in environment — generated ephemeral secret. "
        "Tokens will NOT survive serverless cold starts. Set JWT_SECRET env var in production."
    )
_JWT_ALGORITHM = "HS256"
_JWT_EXPIRE_DAYS = 7

# Role hierarchy (higher index = more permissions)
ROLE_LEVELS = {"viewer": 0, "operator": 1, "manager": 2, "admin": 3, "owner": 4}


async def _get_accessible_tables(
    session: AsyncSession, user: Optional[dict] = None
) -> Optional[list]:
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
    """Hash password using bcrypt (secure, GPU-resistant).

    Falls back to SHA-256 with salt if bcrypt is not installed.
    """
    try:
        import bcrypt

        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    except ImportError:
        logger.warning("bcrypt not installed — falling back to SHA-256 hashing")
        salt = secrets.token_hex(16)
        hashed = hashlib.sha256((salt + password).encode()).hexdigest()
        return f"{salt}:{hashed}"


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash.

    Supports both bcrypt (new) and legacy SHA-256 salt:hash format.
    """
    try:
        # Try bcrypt first (new format — starts with $2b$)
        if password_hash.startswith("$2b$") or password_hash.startswith("$2a$"):
            import bcrypt

            return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

        # Legacy SHA-256 salt:hash format
        salt, hashed = password_hash.split(":")
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


async def _migrate_password_to_bcrypt(user, password: str, session) -> None:
    """Re-hash a legacy SHA-256 password with bcrypt on successful login.

    Called after verifying a legacy password so the next login uses bcrypt.
    """
    try:
        import bcrypt

        new_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user.password_hash = new_hash
        await session.commit()
        logger.info("Migrated password to bcrypt for user %s", user.email)
    except ImportError:
        pass  # bcrypt not available — skip migration
    except Exception:
        logger.debug("bcrypt migration failed for user %s — non-critical", user.email, exc_info=True)


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


def _decode_jwt(token: str) -> Optional[dict]:
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


async def get_current_user(authorization: str = Header(default="")) -> Optional[dict]:
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
    user: Optional[dict] = Depends(get_current_user),
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

        # Auto-enrich column descriptions immediately so they're available
        # for the first query (before any discovery run)
        try:
            await _enrich_column_descriptions(session, table_name)
        except Exception:
            logger.debug("Column description enrichment failed — non-critical")

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
    user: Optional[dict] = Depends(get_current_user),
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
    user: Optional[dict] = Depends(get_current_user),
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
    user: Optional[dict] = Depends(get_current_user),
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

async def _get_focus_tables(session: AsyncSession) -> Optional[list]:
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
        await session.rollback()
        return None


async def _enrich_column_descriptions(session: AsyncSession, table_name: str) -> None:
    """Auto-generate column descriptions using pattern matching.

    Reads the metadata entry for the table, and for each column that lacks a
    description, generates one using data_dictionary.auto_describe_column().
    Updates metadata_store with the enriched columns_metadata.
    """
    from business_brain.discovery.data_dictionary import auto_describe_column

    entry = await metadata_store.get_by_table(session, table_name)
    if not entry or not entry.columns_metadata:
        return

    changed = False
    for col in entry.columns_metadata:
        if not col.get("description"):
            desc = auto_describe_column(
                col.get("name", ""),
                col.get("type", ""),
                {},  # no stats available at upload time
            )
            if desc:
                col["description"] = desc
                changed = True

    if changed:
        await metadata_store.upsert(
            session,
            table_name=table_name,
            description=entry.description or f"Table {table_name}",
            columns_metadata=entry.columns_metadata,
        )
        logger.info("Enriched column descriptions for table '%s'", table_name)


async def _run_discovery_background(trigger: str = "manual", table_filter: Optional[list] = None) -> None:
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

    try:
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
    except Exception:
        logger.exception("Feed fetch failed")
        await session.rollback()
        return []


@app.post("/feed/rescore")
async def rescore_feed(session: AsyncSession = Depends(get_session)) -> dict:
    """Re-score existing insights that have NULL quality_score.

    Runs the quality gate scoring on insights that were created before
    the quality gate existed, assigning them a proper quality_score.
    """
    from sqlalchemy import select as sa_select

    from business_brain.db.discovery_models import Insight
    from business_brain.discovery.insight_quality_gate import apply_quality_gate

    try:
        result = await session.execute(
            sa_select(Insight).where(Insight.quality_score == None)  # noqa: E711
        )
        null_insights = list(result.scalars().all())

        if not null_insights:
            return {"status": "ok", "rescored": 0, "message": "No insights with NULL quality_score"}

        # Run through quality gate (it sets quality_score and impact_score)
        scored = apply_quality_gate(null_insights, [])

        # Update in-place (objects are already attached to session)
        for insight in null_insights:
            if insight not in scored:
                # Quality gate filtered it out — mark as low quality
                insight.quality_score = 0
                insight.impact_score = insight.impact_score or 0

        await session.commit()
        return {
            "status": "ok",
            "rescored": len(null_insights),
            "kept": len(scored),
            "filtered": len(null_insights) - len(scored),
        }
    except Exception as exc:
        logger.exception("Feed rescore failed")
        await session.rollback()
        return {"status": "error", "error": str(exc)}


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
_last_sync_check: Optional[str] = None
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

    try:
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
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error updating focus scope")
        await session.rollback()
        return JSONResponse({"error": "Failed to update focus scope"}, status_code=500)


@app.delete("/focus")
async def clear_focus(session: AsyncSession = Depends(get_session)) -> dict:
    """Clear focus scope — disable focus mode (analyze all tables)."""
    from business_brain.db.v3_models import FocusScope
    from sqlalchemy import delete as sa_delete

    try:
        result = await session.execute(sa_delete(FocusScope))
        await session.commit()
        return {"status": "cleared", "rows_removed": result.rowcount}
    except Exception:
        logger.exception("Error clearing focus scope")
        await session.rollback()
        return {"status": "cleared", "rows_removed": 0}


# Auth & user management routes moved to routers/auth.py (included via app.include_router)
# Re-export for backward compatibility with tests and external imports
from business_brain.action.routers.auth import (  # noqa: F401, E402
    AcceptInviteRequest,
    InviteRequest,
    LoginRequest,
    RegisterRequest,
    UpdateRoleRequest,
    accept_invite,
    create_invite,
    deactivate_user,
    get_me,
    list_invites,
    list_users,
    login_user,
    register_user,
    revoke_invite,
    update_user_role,
)

# Process & setup routes moved to routers/process.py (included via app.include_router)
# Re-export for backward compatibility with tests and external imports
from business_brain.action.routers.process import (  # noqa: F401, E402
    ProcessStepRequest,
    auto_link_metrics,
    create_derived_metric,
    create_process_step,
    get_step_metrics,
    link_metrics_to_step,
    list_process_steps,
    suggest_metrics,
    unlink_metric_from_step,
    update_process_step,
)
