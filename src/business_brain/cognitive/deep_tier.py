"""Deep Tier — Claude API integration for thorough analysis.

Called when Fast Tier confidence is low (< threshold) or the user requests
deeper investigation. Provides:

1. Multi-angle analysis (patterns, root causes, anomaly explanation)
2. Cross-table correlation discovery
3. Actionable recommendations with ₹ quantification
4. Entity anonymization for privacy before sending to Claude

The Deep Tier is async — results are stored in AnalysisTask and polled.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings

logger = logging.getLogger(__name__)

# Lazy import to avoid import errors when anthropic is not installed
_anthropic_client = None


def _get_client():
    """Get or create Anthropic client (lazy singleton)."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            _anthropic_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        except ImportError:
            raise ImportError(
                "anthropic package not installed. Run: pip install anthropic"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    return _anthropic_client


def is_available() -> bool:
    """Check if Deep Tier is configured (API key present)."""
    return bool(settings.anthropic_api_key)


# ---------------------------------------------------------------------------
# Entity Anonymization
# ---------------------------------------------------------------------------

# Patterns to anonymize before sending to Claude
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
_AADHAAR_RE = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
_PAN_RE = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_PERSON_NAME_PREFIXES = {"mr", "mrs", "ms", "dr", "shri", "smt"}


def anonymize_text(text: str) -> tuple[str, dict]:
    """Anonymize PII in text. Returns (anonymized_text, mapping).

    The mapping can be used to de-anonymize results if needed.
    """
    mapping: dict[str, str] = {}
    counter = {"email": 0, "phone": 0, "id": 0}

    def _replace(pattern, prefix, key):
        nonlocal text
        for match in pattern.finditer(text):
            original = match.group()
            if original not in mapping.values():
                counter[key] += 1
                placeholder = f"[{prefix}_{counter[key]}]"
                mapping[placeholder] = original
                text = text.replace(original, placeholder)

    _replace(_EMAIL_RE, "EMAIL", "email")
    _replace(_PHONE_RE, "PHONE", "phone")
    _replace(_AADHAAR_RE, "AADHAAR_ID", "id")
    _replace(_PAN_RE, "TAX_ID", "id")

    return text, mapping


def anonymize_rows(rows: list[dict], max_rows: int = 50) -> list[dict]:
    """Anonymize a sample of data rows for Claude analysis.

    - Limits to max_rows for token efficiency
    - Anonymizes string values that look like PII
    - Preserves numeric values (essential for analysis)
    """
    sample = rows[:max_rows]
    anonymized = []

    for row in sample:
        clean_row = {}
        for col, val in row.items():
            if val is None:
                clean_row[col] = None
            elif isinstance(val, (int, float, bool)):
                clean_row[col] = val
            else:
                text_val = str(val)
                text_val, _ = anonymize_text(text_val)
                clean_row[col] = text_val
        anonymized.append(clean_row)

    return anonymized


# ---------------------------------------------------------------------------
# Deep Tier Prompt
# ---------------------------------------------------------------------------

DEEP_ANALYSIS_PROMPT = """\
You are a senior business intelligence analyst performing a deep investigation.

## Context
The Fast Tier (quick analysis) returned a low-confidence result for this question.
Your job is to provide a thorough, multi-angle analysis that the Fast Tier couldn't.

## Fast Tier Summary
Question: {question}
Fast Tier Confidence: {confidence}
Query Type: {query_type}
Tables Used: {tables}
SQL Query: {sql_query}

Fast Tier Findings:
{fast_findings}

## Data Sample ({row_count} rows from SQL query)
{data_sample}

## Your Analysis Should Include

1. **Root Cause Analysis**: Why might the Fast Tier have low confidence? Is the data
   ambiguous, incomplete, or does the question span multiple domains?

2. **Multi-Angle Findings**: Analyze from at least 3 perspectives:
   - Statistical patterns (trends, outliers, distributions)
   - Business implications (cost impact, efficiency, risk)
   - Operational insights (process bottlenecks, resource allocation)

3. **Cross-Correlation Opportunities**: What other data could be joined to enrich
   this analysis? What questions should be asked next?

4. **Confidence Assessment**: Rate your own confidence (0.0-1.0) in each finding.

5. **Actionable Recommendations**: Specific, prioritized actions with estimated
   impact in ₹ where possible.

## Steel/Manufacturing Domain Knowledge
- Energy: 35-50% of production cost, SEC benchmark: good <500, poor >600 kWh/ton
- Scrap: 55-70% of cost, yield target: 85-95%
- Conversion cost: ₹3,000-5,000/ton
- Key leakage patterns: weighbridge manipulation, scrap grade mismatch,
  alloy over-addition, power theft, furnace idle time, refractory over-consumption

Return a JSON object:
{{
  "deep_findings": [
    {{
      "angle": "statistical|business|operational|root_cause",
      "description": "Detailed finding with specific numbers",
      "confidence": 0.0-1.0,
      "business_impact": "₹X impact estimate",
      "verdict": "good|warning|critical"
    }}
  ],
  "summary": "3-5 sentence executive summary of the deep analysis",
  "root_cause_analysis": "Why was the Fast Tier uncertain",
  "cross_correlations": ["Suggested follow-up analyses"],
  "recommendations": [
    {{
      "action": "Specific recommendation",
      "impact": "₹X estimated impact",
      "priority": "high|medium|low",
      "data_needed": "What additional data would help"
    }}
  ],
  "confidence_overall": 0.0-1.0,
  "next_questions": ["Follow-up questions the user should explore"]
}}

Return ONLY valid JSON, no markdown fences or explanation.
"""


def _extract_json(raw: str) -> Optional[dict]:
    """Extract JSON from Claude's response."""
    # Try direct parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip markdown fences
    if "```" in raw:
        match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

    # Find first { ... last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Core Deep Tier Execution
# ---------------------------------------------------------------------------


async def run_deep_analysis(
    question: str,
    fast_tier_result: dict,
    sql_query: str = "",
    sql_rows: list[dict] = None,
    tables_used: list[str] = None,
    fast_confidence: float = 0.0,
    query_type: str = "custom",
) -> dict:
    """Run Deep Tier analysis using Claude API.

    Args:
        question: Original user question.
        fast_tier_result: Summary from Fast Tier (findings, summary, etc.).
        sql_query: SQL used by Fast Tier.
        sql_rows: Raw data rows from Fast Tier.
        tables_used: Tables referenced.
        fast_confidence: Router confidence score.
        query_type: Query classification from router.

    Returns:
        Dict with deep_findings, summary, recommendations, etc.
    """
    if not is_available():
        return {
            "error": "Deep Tier not configured — set ANTHROPIC_API_KEY",
            "status": "unavailable",
        }

    # Anonymize data
    rows = sql_rows or []
    anonymized_rows = anonymize_rows(rows)
    row_count = len(rows)

    # Format fast tier findings
    findings = fast_tier_result.get("findings", [])
    fast_findings = "\n".join(
        f"- [{f.get('type', 'insight')}] {f.get('description', '')}"
        f" (confidence: {f.get('confidence', '?')})"
        for f in findings
    ) or "No findings from Fast Tier."

    data_sample = json.dumps(anonymized_rows[:20], default=str, indent=1) if anonymized_rows else "No data available."

    prompt = DEEP_ANALYSIS_PROMPT.format(
        question=question,
        confidence=fast_confidence,
        query_type=query_type,
        tables=", ".join(tables_used or []),
        sql_query=sql_query,
        fast_findings=fast_findings,
        row_count=row_count,
        data_sample=data_sample,
    )

    try:
        client = _get_client()
        response = client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        result = _extract_json(raw)

        if result is None:
            return {
                "error": f"Failed to parse Claude response: {raw[:200]}",
                "status": "parse_error",
                "raw_response": raw[:1000],
            }

        result["status"] = "completed"
        result["model"] = settings.claude_model
        result["tier"] = "deep"

        return result

    except Exception as exc:
        logger.exception("Deep Tier analysis failed")
        return {
            "error": str(exc),
            "status": "failed",
        }


# ---------------------------------------------------------------------------
# Task Queue Operations
# ---------------------------------------------------------------------------


async def create_task(
    session: AsyncSession,
    question: str,
    *,
    fast_tier_result: dict = None,
    sql_query: str = "",
    sql_rows: list[dict] = None,
    tables_used: list[str] = None,
    fast_confidence: float = 0.0,
    session_id: str = "",
    source_tier: str = "fast",
    requested_by: str = "auto",
    priority: int = 0,
) -> dict:
    """Create a Deep Tier analysis task in the queue.

    Returns task dict with id and status.
    """
    from business_brain.db.discovery_models import AnalysisTask

    # Anonymize data before storing
    anonymized_rows = anonymize_rows(sql_rows or [])

    # Summarize fast tier result (don't store full rows)
    fast_summary = None
    if fast_tier_result:
        fast_summary = {
            "findings": fast_tier_result.get("findings", [])[:10],
            "summary": fast_tier_result.get("summary", ""),
            "key_metrics": fast_tier_result.get("key_metrics", [])[:5],
            "query_type": fast_tier_result.get("query_type", ""),
        }

    task = AnalysisTask(
        question=question,
        source_tier=source_tier,
        status="pending",
        priority=priority,
        fast_tier_result=fast_summary,
        sql_query=sql_query,
        sql_data=anonymized_rows[:50],
        tables_used=tables_used,
        fast_confidence=fast_confidence,
        session_id=session_id,
        requested_by=requested_by,
    )
    session.add(task)
    await session.commit()
    await session.refresh(task)

    logger.info(
        "Deep Tier task created: id=%s, source=%s, confidence=%.2f",
        task.id, source_tier, fast_confidence,
    )

    return {
        "task_id": task.id,
        "status": "pending",
        "question": question,
        "source_tier": source_tier,
        "fast_confidence": fast_confidence,
    }


async def execute_task(session: AsyncSession, task_id: str) -> dict:
    """Execute a pending Deep Tier task.

    Fetches the task, runs Claude analysis, stores the result.
    """
    from business_brain.db.discovery_models import AnalysisTask

    result = await session.execute(
        select(AnalysisTask).where(AnalysisTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if not task:
        return {"error": "Task not found"}

    if task.status not in ("pending", "failed"):
        return {"error": f"Task is {task.status}, not runnable"}

    # Mark as running
    task.status = "running"
    task.started_at = datetime.now(timezone.utc)
    await session.commit()

    try:
        analysis_result = await run_deep_analysis(
            question=task.question,
            fast_tier_result=task.fast_tier_result or {},
            sql_query=task.sql_query or "",
            sql_rows=task.sql_data or [],
            tables_used=task.tables_used or [],
            fast_confidence=task.fast_confidence or 0.0,
            query_type=(task.fast_tier_result or {}).get("query_type", "custom"),
        )

        if analysis_result.get("status") == "completed":
            task.status = "completed"
            task.result = analysis_result
        else:
            task.status = "failed"
            task.error = analysis_result.get("error", "Unknown error")
            task.result = analysis_result

        task.completed_at = datetime.now(timezone.utc)
        await session.commit()

        return {
            "task_id": task_id,
            "status": task.status,
            "result": task.result,
        }

    except Exception as exc:
        logger.exception("Deep Tier task execution failed: %s", task_id)
        task.status = "failed"
        task.error = str(exc)
        task.completed_at = datetime.now(timezone.utc)
        await session.commit()
        return {"task_id": task_id, "status": "failed", "error": str(exc)}


async def get_task_status(session: AsyncSession, task_id: str) -> Optional[dict]:
    """Get current status and result of a Deep Tier task."""
    from business_brain.db.discovery_models import AnalysisTask

    result = await session.execute(
        select(AnalysisTask).where(AnalysisTask.id == task_id)
    )
    task = result.scalar_one_or_none()
    if not task:
        return None

    return {
        "task_id": task.id,
        "question": task.question,
        "status": task.status,
        "source_tier": task.source_tier,
        "fast_confidence": task.fast_confidence,
        "priority": task.priority,
        "result": task.result,
        "error": task.error,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None,
        "requested_by": task.requested_by,
    }


async def list_tasks(
    session: AsyncSession,
    status: Optional[str] = None,
    limit: int = 20,
) -> list[dict]:
    """List Deep Tier tasks, optionally filtered by status."""
    from business_brain.db.discovery_models import AnalysisTask

    query = select(AnalysisTask).order_by(AnalysisTask.created_at.desc()).limit(limit)
    if status:
        query = query.where(AnalysisTask.status == status)

    result = await session.execute(query)
    tasks = list(result.scalars().all())

    return [
        {
            "task_id": t.id,
            "question": t.question,
            "status": t.status,
            "source_tier": t.source_tier,
            "fast_confidence": t.fast_confidence,
            "priority": t.priority,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "completed_at": t.completed_at.isoformat() if t.completed_at else None,
            "has_result": t.result is not None,
        }
        for t in tasks
    ]
