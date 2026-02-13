"""FastAPI application with external trigger endpoints."""

import json
import logging
import uuid
from io import StringIO
from typing import Optional

from fastapi import BackgroundTasks, Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.cognitive.graph import build_graph
from business_brain.db.connection import async_session, engine, get_session
from business_brain.db.models import Base
from business_brain.ingestion.context_ingestor import ingest_context
from business_brain.memory import chat_store, metadata_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Business Brain API", version="2.0.0")


@app.on_event("startup")
async def _ensure_tables():
    """Create any missing tables (e.g. chat_messages) on first deploy."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables ensured")
    except Exception:
        logger.exception("Failed to auto-create tables — chat history may be unavailable")

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
        "version": "2.0.0",
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

    graph = build_graph()
    invoke_state = {
        "question": req.question,
        "db_session": session,
        "session_id": session_id,
        "chat_history": chat_history,
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

    # Strip non-serializable db_session from response
    result.pop("db_session", None)
    result.pop("chat_history", None)
    result.pop("column_classification", None)  # internal, don't expose
    result.setdefault("cfo_key_metrics", [])
    result.setdefault("cfo_chart_suggestions", [])
    result["session_id"] = session_id
    return result


@app.post("/context")
async def submit_context(req: ContextRequest, session: AsyncSession = Depends(get_session)) -> dict:
    """Submit natural-language business context for embedding."""
    ids = await ingest_context(req.text, session, source=req.source)
    return {"status": "created", "ids": ids, "chunks": len(ids), "source": req.source}


@app.post("/csv")
async def upload_csv(file: UploadFile = File(...), session: AsyncSession = Depends(get_session)) -> dict:
    """Upload a CSV file for ingestion into the database."""
    import pandas as pd
    from business_brain.ingestion.csv_loader import upsert_dataframe

    contents = await file.read()
    table_name = (file.filename or "upload").rsplit(".", 1)[0].replace("-", "_").replace(" ", "_")
    df = pd.read_csv(StringIO(contents.decode("utf-8")))
    rows = await upsert_dataframe(df, session, table_name)
    return {"status": "loaded", "table": table_name, "rows": rows}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Smart file upload — parse, clean, load, and auto-generate metadata."""
    from business_brain.cognitive.data_engineer_agent import DataEngineerAgent

    file_bytes = await file.read()
    file_name = file.filename or "upload.csv"

    try:
        agent = DataEngineerAgent()
        report = await agent.invoke({
            "file_bytes": file_bytes,
            "file_name": file_name,
            "db_session": session,
        })

        # Trigger discovery in background after successful upload
        table_name = report.get("table_name", "unknown")
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
    from business_brain.cognitive.data_engineer_agent import parse_pdf
    from business_brain.ingestion.context_ingestor import chunk_text

    file_bytes = await file.read()
    file_name = file.filename or "context.txt"
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
async def list_metadata(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all table metadata."""
    entries = await metadata_store.get_all(session)
    return [
        {
            "table_name": e.table_name,
            "description": e.description,
            "columns": e.columns_metadata,
        }
        for e in entries
    ]


@app.get("/metadata/{table}")
async def get_table_metadata(table: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get metadata for a single table."""
    entry = await metadata_store.get_by_table(session, table)
    if entry is None:
        return {"error": "Table not found"}
    return {
        "table_name": entry.table_name,
        "description": entry.description,
        "columns": entry.columns_metadata,
    }


@app.delete("/metadata/{table}")
async def delete_table_metadata(table: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Delete metadata for a table (e.g. after dropping the table)."""
    deleted = await metadata_store.delete(session, table)
    if not deleted:
        return {"error": "Table not found"}
    return {"status": "deleted", "table": table}


@app.get("/data/{table}")
async def get_table_data(
    table: str,
    page: int = 1,
    page_size: int = 50,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Paginated read-only access to any uploaded table."""
    from sqlalchemy import text as sql_text
    import re

    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", table)
    if not safe_table:
        return {"error": "Invalid table name"}

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

    # Get the primary key column (first column) from metadata
    entry = await metadata_store.get_by_table(session, safe_table)
    if entry and entry.columns_metadata:
        pk_col = entry.columns_metadata[0]["name"]
    else:
        pk_col = "id"  # fallback

    safe_pk = re.sub(r"[^a-zA-Z0-9_]", "", pk_col)

    try:
        query = f'UPDATE "{safe_table}" SET "{safe_col}" = :val WHERE "{safe_pk}" = :pk'
        await session.execute(sql_text(query), {"val": value, "pk": row_id})
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

async def _run_discovery_background(trigger: str = "manual") -> None:
    """Run discovery engine in a background task with its own session."""
    from business_brain.discovery.engine import run_discovery

    try:
        async with async_session() as session:
            await run_discovery(session, trigger=trigger)
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
    """Get ranked insight feed."""
    from business_brain.discovery.feed_store import get_feed as _get_feed

    insights = await _get_feed(session)
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
    """Manually trigger a discovery sweep."""
    background_tasks.add_task(_run_discovery_background, "manual")
    return {"status": "started", "message": "Discovery sweep triggered in background"}


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
