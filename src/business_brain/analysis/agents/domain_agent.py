"""Domain Agent — business context enrichment via RAG + LLM.

Output:
- relevance_score (0-1)
- classification: CONFIRMING / SURPRISING / NOVEL / IRRELEVANT
- business_translation: natural-language explanation
- benchmark_context: relevant benchmarks from RAG
- estimated_impact: business impact assessment

Enhanced for N-ary: segment-aware translations.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from business_brain.analysis.agents.base import AgentContext, AnalysisAgent
from business_brain.analysis.tools import llm_gateway, rag_service

logger = logging.getLogger(__name__)


class DomainAgent(AnalysisAgent):
    agent_id = "domain"

    async def build_context(self, ctx: AgentContext) -> dict[str, Any]:
        """Fetch business context via RAG."""
        result = ctx.result

        # Build a search query from the finding
        query_parts = [
            result.operation_type,
            result.table_name,
            " ".join(result.target or []),
        ]
        if result.segmenters:
            query_parts.append("segmented by " + ", ".join(result.segmenters))
        query = " ".join(query_parts)

        # Search both stores
        rag_hits = await rag_service.search_multi(
            ctx.session,
            ["business_context", "analysis_history"],
            query,
            top_k=5,
        )

        return {
            "rag_hits": rag_hits,
            "query": query,
        }

    async def analyze(self, ctx: AgentContext, context_data: dict[str, Any]) -> dict[str, Any]:
        """Use LLM to classify and translate the finding in business terms."""
        result = ctx.result
        rag_hits = context_data.get("rag_hits", [])

        # Build context block from RAG
        rag_context = ""
        if rag_hits:
            rag_context = "\n\nRelevant business context:\n" + "\n".join(
                f"- {h['content'][:200]}" for h in rag_hits[:3]
            )

        # Build finding summary
        result_data = result.result_data or {}
        finding_summary = self._summarize_finding(result, result_data)

        prompt = f"""You are a business domain analyst. Evaluate this analytical finding.

Finding:
{finding_summary}
{rag_context}

Respond with JSON:
{{
  "relevance_score": 0.0-1.0,
  "classification": "CONFIRMING|SURPRISING|NOVEL|IRRELEVANT",
  "business_translation": "Plain English explanation for a business user",
  "benchmark_context": "How this compares to known benchmarks or expectations",
  "estimated_impact": "LOW|MEDIUM|HIGH with brief explanation"
}}"""

        response = await llm_gateway.extract(prompt)

        # Ensure all required keys exist
        relevance = float(response.get("relevance_score", 0.5))
        classification = response.get("classification", "NOVEL")
        if classification not in ("CONFIRMING", "SURPRISING", "NOVEL", "IRRELEVANT"):
            classification = "NOVEL"

        return {
            "relevance_score": relevance,
            "classification": classification,
            "business_translation": response.get("business_translation", finding_summary),
            "benchmark_context": response.get("benchmark_context", "No benchmarks available"),
            "estimated_impact": response.get("estimated_impact", "MEDIUM"),
            "rag_hit_count": len(rag_hits),
            "_confidence": relevance,
        }

    def _summarize_finding(self, result: Any, data: dict) -> str:
        """Build a human-readable summary of the finding."""
        parts = [f"Operation: {result.operation_type}"]
        parts.append(f"Table: {result.table_name}")
        parts.append(f"Target columns: {result.target}")

        if result.segmenters:
            parts.append(f"Segmented by: {result.segmenters}")
        if result.controls:
            parts.append(f"Controlled for: {result.controls}")

        if result.operation_type == "CORRELATE":
            r = data.get("pearson_r", "N/A")
            p = data.get("pearson_p", "N/A")
            parts.append(f"Pearson r={r}, p={p}")

        elif result.operation_type == "RANK":
            comparison = data.get("comparison", {})
            if comparison:
                parts.append(f"Effect: {comparison.get('test', 'unknown')}, p={comparison.get('p_value', 'N/A')}")
            ranked = data.get("ranked", [])
            if ranked:
                parts.append(f"Top group: {ranked[0]}")

        elif result.operation_type == "DETECT_ANOMALY":
            count = data.get("count", 0)
            parts.append(f"Anomalies found: {count}")

        elif result.operation_type in ("DESCRIBE", "DESCRIBE_CATEGORICAL"):
            stats = data.get("stats", {})
            if "mean" in stats:
                parts.append(f"Mean={stats['mean']:.2f}, Stdev={stats.get('stdev', 0):.2f}")
            elif "unique" in stats:
                parts.append(f"Unique values: {stats['unique']}, Top: {stats.get('top_value', 'N/A')}")

        parts.append(f"Interestingness: {result.interestingness_score:.2f}")

        return "\n".join(parts)
