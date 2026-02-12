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
    return {"status": "ok"}


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
    row_id = await ingest_context(req.text, session, source=req.source)
    return {"status": "created", "id": row_id, "source": req.source}


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
