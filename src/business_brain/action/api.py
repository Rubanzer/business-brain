"""FastAPI application with external trigger endpoints."""

import json
import logging
import uuid
from io import StringIO
from typing import Optional

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.cognitive.graph import build_graph
from business_brain.db.connection import engine, get_session
from business_brain.db.models import Base
from business_brain.ingestion.context_ingestor import ingest_context
from business_brain.memory import chat_store, metadata_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Business Brain API", version="0.1.0")


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
    result = await graph.ainvoke({
        "question": req.question,
        "db_session": session,
        "session_id": session_id,
        "chat_history": chat_history,
    })

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
    file: UploadFile = File(...), session: AsyncSession = Depends(get_session)
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
