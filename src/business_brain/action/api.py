"""FastAPI application with external trigger endpoints."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Business Brain API", version="0.1.0")


class AnalyzeRequest(BaseModel):
    question: str


class ContextRequest(BaseModel):
    text: str
    source: str = "api"


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(req: AnalyzeRequest) -> dict:
    """Trigger an analysis run for the given business question.

    TODO: invoke cognitive.graph.build_graph() with the question.
    """
    return {"status": "accepted", "question": req.question}


@app.post("/context")
async def submit_context(req: ContextRequest) -> dict:
    """Submit natural-language business context for embedding.

    TODO: call ingestion.context_ingestor.ingest_context().
    """
    return {"status": "accepted", "source": req.source}
