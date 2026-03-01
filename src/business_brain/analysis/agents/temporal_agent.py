"""Temporal Agent — time-pattern analysis with seasonality + trend detection.

Uses SQL + Compute + RAG + LLM.
Output:
- seasonality_adjusted: whether the finding accounts for seasonal patterns
- event_attribution: links to known events that explain the pattern
- trend_status: ACCELERATING / STABLE / DECELERATING / REVERSING
- novelty: NEW / RECURRING / CHRONIC

Enhanced for time scoping (Gap #5): uses the run's TimeScope baseline.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sql_text

from business_brain.analysis.agents.base import AgentContext, AnalysisAgent
from business_brain.analysis.tools import compute, llm_gateway, rag_service
from business_brain.analysis.tools.sql_executor import _safe

logger = logging.getLogger(__name__)


class TemporalAgent(AnalysisAgent):
    agent_id = "temporal"

    async def build_context(self, ctx: AgentContext) -> dict[str, Any]:
        """Gather time-series data and historical context."""
        result = ctx.result
        session = ctx.session
        context: dict[str, Any] = {"has_time_data": False}

        # Detect time column from result or time_scope
        time_col = None
        time_scope = ctx.time_scope or {}
        if time_scope.get("column"):
            time_col = time_scope["column"]

        if not time_col:
            return context

        safe_table = _safe(result.table_name)
        safe_time = _safe(time_col)
        target_col = result.target[0] if result.target else None
        if not target_col:
            return context

        safe_target = _safe(target_col)

        # Fetch time-series data
        try:
            q = (
                f'SELECT date_trunc(\'day\', "{safe_time}") AS period, '
                f'AVG(CAST("{safe_target}" AS DOUBLE PRECISION)) AS val, '
                f'COUNT(*) AS cnt '
                f'FROM "{safe_table}" '
                f'WHERE "{safe_target}" IS NOT NULL AND "{safe_time}" IS NOT NULL '
                f'GROUP BY period ORDER BY period'
            )
            ts_result = await session.execute(sql_text(q))
            rows = [dict(r._mapping) for r in ts_result.fetchall()]
            context["time_series"] = rows
            context["has_time_data"] = len(rows) >= 7
            context["time_column"] = time_col
        except Exception:
            logger.debug("Failed to fetch time series for %s.%s", result.table_name, target_col)

        # Search for historical events
        rag_query = f"events affecting {target_col} in {result.table_name}"
        try:
            rag_hits = await rag_service.search(session, "business_context", rag_query, top_k=3)
            context["event_context"] = [h["content"][:200] for h in rag_hits]
        except Exception:
            context["event_context"] = []

        return context

    async def analyze(self, ctx: AgentContext, context_data: dict[str, Any]) -> dict[str, Any]:
        """Analyze temporal patterns."""
        if not context_data.get("has_time_data"):
            return {
                "seasonality_adjusted": False,
                "event_attribution": None,
                "trend_status": "UNKNOWN",
                "novelty": "UNKNOWN",
                "reason": "no time-series data available",
                "_confidence": 0.2,
            }

        ts_rows = context_data.get("time_series", [])
        values = [r["val"] for r in ts_rows if r.get("val") is not None]

        if len(values) < 7:
            return {
                "seasonality_adjusted": False,
                "event_attribution": None,
                "trend_status": "UNKNOWN",
                "novelty": "UNKNOWN",
                "reason": "insufficient time points",
                "_confidence": 0.3,
            }

        # Decompose
        period = min(7, len(values) // 2)
        decomposition = compute.decompose_series(values, period=period)

        # Trend analysis
        trend = decomposition.get("trend", [])
        trend_direction = decomposition.get("trend_direction", "stable")
        seasonal_strength = decomposition.get("seasonal_strength", 0.0)

        # Determine trend status
        if len(trend) >= 10:
            recent = trend[-5:]
            older = trend[-10:-5]
            recent_slope = (recent[-1] - recent[0]) / max(len(recent), 1)
            older_slope = (older[-1] - older[0]) / max(len(older), 1)
            if abs(recent_slope) > abs(older_slope) * 1.5:
                trend_status = "ACCELERATING"
            elif abs(recent_slope) < abs(older_slope) * 0.5:
                trend_status = "DECELERATING"
            elif (recent_slope > 0) != (older_slope > 0):
                trend_status = "REVERSING"
            else:
                trend_status = "STABLE"
        else:
            trend_status = "STABLE" if trend_direction == "stable" else "ACCELERATING"

        # Event attribution via LLM
        event_context = context_data.get("event_context", [])
        event_attribution = None
        if event_context:
            prompt = (
                f"Given these known business events:\n"
                + "\n".join(f"- {e}" for e in event_context)
                + f"\n\nAnd a {trend_status.lower()} trend in {ctx.result.target[0]}, "
                f"which event (if any) best explains this pattern? "
                f"Respond with JSON: {{\"event\": \"...\", \"confidence\": 0.0-1.0}}"
            )
            try:
                response = await llm_gateway.extract(prompt)
                if response.get("event") and response.get("confidence", 0) > 0.3:
                    event_attribution = response
            except Exception:
                pass

        # Novelty: check if similar pattern exists in analysis history
        try:
            history_hits = await rag_service.search(
                ctx.session,
                "analysis_history",
                f"{ctx.result.operation_type} {ctx.result.table_name} trend",
                top_k=3,
            )
            if history_hits and history_hits[0]["similarity"] > 0.8:
                novelty = "RECURRING"
            elif history_hits and history_hits[0]["similarity"] > 0.5:
                novelty = "CHRONIC"
            else:
                novelty = "NEW"
        except Exception:
            novelty = "NEW"

        confidence = 0.6
        if len(values) >= 30:
            confidence = 0.8
        if seasonal_strength > 1.0:
            confidence = min(confidence + 0.1, 1.0)

        return {
            "seasonality_adjusted": seasonal_strength > 0.5,
            "seasonal_strength": round(seasonal_strength, 2),
            "event_attribution": event_attribution,
            "trend_status": trend_status,
            "trend_direction": trend_direction,
            "novelty": novelty,
            "data_points": len(values),
            "_confidence": confidence,
        }
