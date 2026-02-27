"""Process & setup router — process steps, metrics, templates, setup.

Extracted from api.py: process step CRUD, I/O mapping, metric linking,
industry templates and auto-link metrics.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session
from business_brain.memory import metadata_store

router = APIRouter(tags=["process-setup"])


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


@router.get("/process-steps")
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


@router.post("/process-steps")
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


@router.put("/process-steps/{step_id}")
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


@router.delete("/process-steps/{step_id}")
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


@router.get("/process-io")
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


@router.post("/process-io")
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


@router.put("/process-io/{io_id}")
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


@router.delete("/process-io/{io_id}")
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


@router.post("/setup/auto-link-metrics")
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


@router.post("/setup/suggest-metrics")
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


@router.post("/metrics/derived")
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


@router.post("/process-steps/{step_id}/metrics")
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


@router.get("/process-steps/{step_id}/metrics")
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


@router.delete("/process-steps/{step_id}/metrics/{metric_id}")
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


@router.get("/setup/template/{industry}")
async def get_industry_setup_template(industry: str) -> dict:
    """Get pre-built setup template for a given industry."""
    from business_brain.cognitive.domain_knowledge import get_industry_template

    template = get_industry_template(industry)
    if not template:
        return {"error": f"No template available for industry: {industry}"}

    return {
        "industry": industry,
        "process_steps": template.get("process_steps", []),
        "metrics": template.get("metrics", []),
        "inputs": template.get("inputs", []),
        "outputs": template.get("outputs", []),
    }


@router.post("/setup/apply-template/{industry}")
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



