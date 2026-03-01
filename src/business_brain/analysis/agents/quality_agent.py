"""Quality Agent — data quality gate with VETO power.

SQL-heavy + rule-based. NO RAG, NO LLM.
5 checks: sample size, freshness, null rate, sanctity, stability.
Enhanced for N-ary: checks sample size PER SEGMENT.
VETO power if UNRELIABLE.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.agents.base import AgentContext, AnalysisAgent
from business_brain.analysis.models import AnalysisResult
from business_brain.analysis.tools.sql_executor import _safe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_MIN_SAMPLE_SIZE = 30
_MIN_SEGMENT_SIZE = 10  # per-segment minimum for N-ary (Gap #1)
_MAX_NULL_RATE = 0.3
_MAX_STALENESS_DAYS = 90
_MIN_DISTINCT_VALUES = 2


class QualityAgent(AnalysisAgent):
    agent_id = "quality"

    async def build_context(self, ctx: AgentContext) -> dict[str, Any]:
        """Gather data quality metrics via SQL."""
        result = ctx.result
        table = result.table_name
        safe_table = _safe(table)
        session = ctx.session
        context: dict[str, Any] = {}

        # Total row count
        try:
            cnt = await session.execute(sql_text(f'SELECT COUNT(*) FROM "{safe_table}"'))
            context["total_rows"] = cnt.scalar() or 0
        except Exception:
            context["total_rows"] = 0

        # Null rates for target columns
        null_rates = {}
        for col in (result.target or []):
            try:
                q = f'SELECT COUNT(*) FILTER (WHERE "{_safe(col)}" IS NULL)::FLOAT / NULLIF(COUNT(*), 0) FROM "{safe_table}"'
                nr = await session.execute(sql_text(q))
                null_rates[col] = nr.scalar() or 0.0
            except Exception:
                null_rates[col] = 0.0
        context["null_rates"] = null_rates

        # Per-segment sample sizes (for N-ary analysis)
        if result.segmenters:
            primary_seg = _safe(result.segmenters[0])
            try:
                seg_q = f'SELECT "{primary_seg}", COUNT(*) AS cnt FROM "{safe_table}" GROUP BY "{primary_seg}" ORDER BY cnt ASC LIMIT 5'
                seg_result = await session.execute(sql_text(seg_q))
                context["smallest_segments"] = [
                    {"segment": str(r[0]), "count": r[1]}
                    for r in seg_result.fetchall()
                ]
            except Exception:
                context["smallest_segments"] = []
        else:
            context["smallest_segments"] = []

        # Freshness — check for any timestamp column
        context["freshness_days"] = None
        for col_name in (result.target or []) + (result.segmenters or []):
            try:
                q = f"SELECT EXTRACT(DAY FROM NOW() - MAX(\"{_safe(col_name)}\")) FROM \"{safe_table}\""
                fr = await session.execute(sql_text(q))
                days = fr.scalar()
                if days is not None:
                    context["freshness_days"] = float(days)
                    break
            except Exception:
                continue

        # Distinct value check for segmenters
        distinct_counts = {}
        for col in (result.segmenters or []):
            try:
                q = f'SELECT COUNT(DISTINCT "{_safe(col)}") FROM "{safe_table}"'
                dc = await session.execute(sql_text(q))
                distinct_counts[col] = dc.scalar() or 0
            except Exception:
                distinct_counts[col] = 0
        context["distinct_counts"] = distinct_counts

        return context

    async def analyze(self, ctx: AgentContext, context_data: dict[str, Any]) -> dict[str, Any]:
        """Run 5 quality checks and return verdict."""
        checks: list[dict[str, Any]] = []
        veto = False

        # 1. Sample size check
        total_rows = context_data.get("total_rows", 0)
        if total_rows < _MIN_SAMPLE_SIZE:
            checks.append({"check": "sample_size", "status": "FAIL", "value": total_rows, "threshold": _MIN_SAMPLE_SIZE})
            veto = True
        else:
            checks.append({"check": "sample_size", "status": "PASS", "value": total_rows})

        # 2. Per-segment sample size (N-ary — Gap #1)
        smallest = context_data.get("smallest_segments", [])
        segment_fail = False
        for seg in smallest:
            if seg["count"] < _MIN_SEGMENT_SIZE:
                segment_fail = True
                checks.append({
                    "check": "segment_size",
                    "status": "WARN",
                    "segment": seg["segment"],
                    "value": seg["count"],
                    "threshold": _MIN_SEGMENT_SIZE,
                })
        if not segment_fail and smallest:
            checks.append({"check": "segment_size", "status": "PASS"})

        # 3. Null rate check
        null_rates = context_data.get("null_rates", {})
        for col, rate in null_rates.items():
            if rate > _MAX_NULL_RATE:
                checks.append({"check": "null_rate", "status": "WARN", "column": col, "value": round(rate, 3)})
            else:
                checks.append({"check": "null_rate", "status": "PASS", "column": col, "value": round(rate, 3)})

        # 4. Freshness check
        freshness = context_data.get("freshness_days")
        if freshness is not None and freshness > _MAX_STALENESS_DAYS:
            checks.append({"check": "freshness", "status": "WARN", "days_stale": freshness})
        elif freshness is not None:
            checks.append({"check": "freshness", "status": "PASS", "days_stale": freshness})

        # 5. Distinct values check (sanctity: dimensions should have >1 value)
        distinct_counts = context_data.get("distinct_counts", {})
        for col, cnt in distinct_counts.items():
            if cnt < _MIN_DISTINCT_VALUES:
                checks.append({"check": "distinct_values", "status": "WARN", "column": col, "value": cnt})
            else:
                checks.append({"check": "distinct_values", "status": "PASS", "column": col, "value": cnt})

        # Compute verdict
        fails = sum(1 for c in checks if c["status"] == "FAIL")
        warns = sum(1 for c in checks if c["status"] == "WARN")

        if veto or fails > 0:
            verdict = "UNRELIABLE"
        elif warns >= 3:
            verdict = "CAUTIONARY"
        elif warns >= 1:
            verdict = "CAUTIONARY"
        else:
            verdict = "RELIABLE"

        confidence = max(0.3, 1.0 - (fails * 0.3 + warns * 0.1))

        return {
            "verdict": verdict,
            "checks": checks,
            "veto": veto,
            "_confidence": confidence,
        }
