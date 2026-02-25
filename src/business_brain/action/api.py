"""FastAPI application — slim app factory with modular routers.

v4: Routes have been extracted into routers/ modules.
This file contains only:
- App creation + middleware
- Startup/shutdown lifecycle events
- Core /health and /analyze endpoints
- Background sync loop
- Backward-compatibility re-exports
"""

import asyncio
import logging
import os
import uuid
from typing import Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.cognitive.graph import build_graph
from business_brain.db.connection import async_session, engine, get_session
from business_brain.db.models import Base
from business_brain.memory import chat_store, metadata_store

from business_brain.action.dependencies import (
    PLAN_LIMITS,
    ROLE_LEVELS,
    decode_jwt,
    get_current_user,
    get_focus_tables,
    require_role,
    get_accessible_tables,
    hash_password as _hash_password,
    verify_password as _verify_password,
    migrate_password_to_bcrypt as _migrate_password_to_bcrypt,
    create_jwt as _create_jwt,
    check_rate_limit as _check_rate_limit,
    enrich_column_descriptions as _enrich_column_descriptions,
    run_discovery_background as _run_discovery_background,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Business Brain API", version="4.0.0")

# ---------------------------------------------------------------------------
# Include modular routers
# ---------------------------------------------------------------------------
from business_brain.action.routers.auth import router as auth_router
from business_brain.action.routers.table_analysis import router as table_analysis_router
from business_brain.action.routers.process import router as process_router
from business_brain.action.routers.sources import router as sources_router
from business_brain.action.routers.alerts import router as alerts_router
from business_brain.action.routers.feed import router as feed_router
from business_brain.action.routers.reports import router as reports_router
from business_brain.action.routers.discovery import router as discovery_router
from business_brain.action.routers.quality import router as quality_router
from business_brain.action.routers.patterns import router as patterns_router
from business_brain.action.routers.company import router as company_router
from business_brain.action.routers.integrations import router as integrations_router
from business_brain.action.routers.data import router as data_router
from business_brain.action.routers.context import router as context_router
from business_brain.action.routers.focus import router as focus_router

app.include_router(auth_router)
app.include_router(table_analysis_router)
app.include_router(process_router)
app.include_router(sources_router)
app.include_router(alerts_router)
app.include_router(feed_router)
app.include_router(reports_router)
app.include_router(discovery_router)
app.include_router(quality_router)
app.include_router(patterns_router)
app.include_router(company_router)
app.include_router(integrations_router)
app.include_router(data_router)
app.include_router(context_router)
app.include_router(focus_router)

# Background sync task handle
_sync_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def _ensure_tables():
    """Create missing tables and add columns missing from existing tables."""
    from sqlalchemy import text as sql_text

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        _migrations = [
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS version INTEGER DEFAULT 1",
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS superseded_at TIMESTAMPTZ",
            "ALTER TABLE business_contexts ADD COLUMN IF NOT EXISTS last_validated_at TIMESTAMPTZ",
            "ALTER TABLE metadata_entries ADD COLUMN IF NOT EXISTS uploaded_by VARCHAR(36)",
            "ALTER TABLE metadata_entries ADD COLUMN IF NOT EXISTS uploaded_by_role VARCHAR(20)",
            "ALTER TABLE process_steps ADD COLUMN IF NOT EXISTS key_metrics JSON",
            "ALTER TABLE process_steps ADD COLUMN IF NOT EXISTS target_ranges JSON",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS quality_score INTEGER DEFAULT 0",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS narrative TEXT",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS related_insights JSON",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS suggested_actions JSON",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS composite_template VARCHAR(100)",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'new'",
            "ALTER TABLE insights ADD COLUMN IF NOT EXISTS session_id VARCHAR(64)",
            "ALTER TABLE deployed_reports ADD COLUMN IF NOT EXISTS session_id VARCHAR(64)",
            "ALTER TABLE deployed_reports ADD COLUMN IF NOT EXISTS active BOOLEAN DEFAULT TRUE",
            "ALTER TABLE deployed_reports ADD COLUMN IF NOT EXISTS refresh_frequency VARCHAR(20) DEFAULT 'manual'",
            "ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS session_id VARCHAR(64)",
            "ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS uploaded_by VARCHAR(36)",
            "ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS uploaded_by_role VARCHAR(20)",
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
    """Catch-all: ensure every unhandled error returns JSON."""
    logger.exception("Unhandled error on %s %s: %s", request.method, request.url.path, exc)
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
    """Auto-enrich column descriptions for all tables on startup."""
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

                    is_weak = (
                        not old_desc
                        or old_desc.startswith("Numeric (decimal) column:")
                        or old_desc.startswith("Integer column:")
                        or old_desc.startswith("Text column:")
                        or old_desc.startswith("Data column:")
                    )

                    is_wrong = False
                    if "rate" in col_lower and "Percentage or ratio" in old_desc:
                        is_wrong = True
                    if col_lower in ("yield", "yield_pct") and "quantity" in old_desc.lower():
                        is_wrong = True

                    if is_weak or is_wrong:
                        new_desc = auto_describe_column(col_name, col.get("type", ""), {})
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
    """Start the background sync loop."""
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

    await asyncio.sleep(5)

    while True:
        try:
            async with async_session() as session:
                results = await sync_all_due(session)
                if results:
                    synced = [r for r in results if r.get("status") != "skipped" and "error" not in r]
                    if synced:
                        logger.info("Background sync completed: %d sources synced", len(synced))
                        for r in synced:
                            try:
                                await _run_discovery_background(f"auto_sync:{r.get('name', 'unknown')}")
                            except Exception:
                                logger.exception("Discovery after auto-sync failed for %s", r.get("name"))
        except Exception:
            logger.exception("Background sync loop error")

        await asyncio.sleep(60)


# CORS: lock down in production via CORS_ORIGINS env var (comma-separated).
_cors_origins = os.environ.get("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Core endpoints (kept in api.py — tightly coupled with pipeline)
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    parent_finding: Optional[dict] = None


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
    session_id = req.session_id or str(uuid.uuid4())

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

    focus_tables = await get_focus_tables(session)

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
        summary_parts = []
        analysis = result.get("analysis", {})
        if analysis.get("summary"):
            summary_parts.append(analysis["summary"])
        python_analysis = result.get("python_analysis", {})
        if python_analysis.get("narrative"):
            summary_parts.append(python_analysis["narrative"])
        assistant_content = " ".join(summary_parts) or "Analysis completed."
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
    result.pop("column_classification", None)
    result.pop("_rag_tables", None)
    result.pop("_rag_contexts", None)
    result.setdefault("cfo_key_metrics", [])
    result.setdefault("cfo_chart_suggestions", [])
    result.setdefault("query_type", "custom")
    result.setdefault("query_confidence", 0.5)
    result.setdefault("key_metrics", [])
    result.setdefault("leakage_patterns", [])
    result.pop("router_reasoning", None)
    result["session_id"] = session_id

    diagnostics = result.pop("_diagnostics", [])
    result["diagnostics"] = diagnostics

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


@app.get("/plan/limits")
async def get_plan_limits(
    authorization: str = Header(default=""),
) -> dict:
    """Get current user's plan limits and usage."""
    user_data = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        user_data = decode_jwt(token)

    plan = user_data.get("plan", "free") if user_data else "pro"
    limits = PLAN_LIMITS.get(plan, PLAN_LIMITS["free"])

    return {
        "plan": plan,
        "limits": limits,
        "upload_count": 0,
    }


# ---------------------------------------------------------------------------
# Backward-compatibility re-exports
# ---------------------------------------------------------------------------

# Auth routes moved to routers/auth.py
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

# Process routes moved to routers/process.py
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

# Feed routes moved to routers/feed.py
from business_brain.action.routers.feed import (  # noqa: F401, E402
    DeployRequest,
    StatusRequest,
    SaveToFeedRequest,
    get_feed,
    rescore_feed,
    dismiss_all_insights,
    update_insight_status,
    deploy_insight_as_report,
    save_analysis_to_feed,
    export_feed,
)

# Context routes moved to routers/context.py
from business_brain.action.routers.context import (  # noqa: F401, E402
    ContextRequest,
    submit_context,
    list_context_entries,
    update_context_entry,
    delete_context_entry,
)

# Data routes moved to routers/data.py
from business_brain.action.routers.data import (  # noqa: F401, E402
    upload_csv,
    upload_file,
    upload_context_file,
    list_metadata,
    get_table_metadata,
    delete_table_metadata,
    drop_table_cascade,
    cleanup_orphaned_tables,
    get_table_data,
    update_cell,
    generate_chart,
    export_table,
)

# Sources routes moved to routers/sources.py
from business_brain.action.routers.sources import (  # noqa: F401, E402
    GoogleSheetRequest,
    ApiSourceRequest,
    SourceUpdateRequest,
    list_sources,
    connect_google_sheet_endpoint,
    connect_api_source,
    sync_source_endpoint,
    update_source_endpoint,
    delete_source_endpoint,
    get_source_changes,
    pause_source_endpoint,
    resume_source_endpoint,
    get_sync_status,
)

# Quality routes moved to routers/quality.py
from business_brain.action.routers.quality import (  # noqa: F401, E402
    ResolveRequest,
    get_sanctity_issues,
    get_sanctity_summary,
    resolve_sanctity_issue,
    get_all_changes,
    get_data_quality,
)

# Alerts routes moved to routers/alerts.py
from business_brain.action.routers.alerts import (  # noqa: F401, E402
    AlertParseRequest,
    AlertDeployRequest,
    AlertUpdateRequest,
    parse_alert_endpoint,
    deploy_alert_endpoint,
    list_alerts,
    get_alert_detail,
    update_alert_endpoint,
    pause_alert_endpoint,
    resume_alert_endpoint,
    delete_alert_endpoint,
    get_alert_events_endpoint,
    evaluate_alerts_endpoint,
    preview_alert,
)

# Integrations routes moved to routers/integrations.py
from business_brain.action.routers.integrations import (  # noqa: F401, E402
    TelegramRegisterRequest,
    ConfirmMappingRequest,
    register_telegram,
    telegram_status,
    detect_duplicates,
    confirm_mapping_endpoint,
    list_mappings,
)

# Patterns routes moved to routers/patterns.py
from business_brain.action.routers.patterns import (  # noqa: F401, E402
    PatternCreateRequest,
    PatternFeedbackRequest,
    create_pattern_endpoint,
    list_patterns,
    get_pattern_detail,
    pattern_feedback_endpoint,
    delete_pattern_endpoint,
)

# Company routes moved to routers/company.py
from business_brain.action.routers.company import (  # noqa: F401, E402
    get_company,
    update_company,
    full_onboard,
    list_thresholds,
    create_threshold_endpoint,
    update_threshold_endpoint,
    delete_threshold_endpoint,
)

# Reports routes moved to routers/reports.py
from business_brain.action.routers.reports import (  # noqa: F401, E402
    list_reports,
    get_report,
    refresh_report,
    delete_report,
    export_report,
)

# Discovery routes moved to routers/discovery.py
from business_brain.action.routers.discovery import (  # noqa: F401, E402
    trigger_discovery,
    discovery_status,
    get_suggestions,
    discovery_history,
)

# Focus routes moved to routers/focus.py
from business_brain.action.routers.focus import (  # noqa: F401, E402
    get_focus,
    update_focus,
    clear_focus,
)

# Also re-export auth helpers and shared dependencies for any external code
# that imports directly from api.py (legacy backward compat)
_decode_jwt = decode_jwt  # noqa: F841
_get_focus_tables = get_focus_tables  # noqa: F841
_get_accessible_tables = get_accessible_tables  # noqa: F841
