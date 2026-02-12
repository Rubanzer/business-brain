"""FastAPI application with external trigger endpoints."""

import logging
from io import StringIO

from fastapi import Depends, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.cognitive.graph import build_graph
from business_brain.db.connection import get_session
from business_brain.ingestion.context_ingestor import ingest_context
from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Business Brain API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    question: str


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
    graph = build_graph()
    result = await graph.ainvoke({"question": req.question, "db_session": session})
    # Strip non-serializable db_session from response
    result.pop("db_session", None)
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
