# The Business Brain

An autonomous AI agent that acts as a proactive business analyst.

## Architecture

| Layer | Package | Purpose |
|-------|---------|---------|
| 1 — Sensory | `ingestion` | File watchers, CSV/API loaders, context embedding |
| 2 — Memory | `memory` | Schema metadata, vector store, RAG retrieval |
| 3 — Cognitive | `cognitive` | LangGraph agent swarm (Supervisor → SQL → Analyst → CFO) |
| 4 — Action | `action` | FastAPI endpoints, alerts, dashboard |

## Quick Start

```bash
uv sync
docker compose up -d
uv run python scripts/init_db.py
uv run python -m pytest
```
