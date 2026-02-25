"""Insight Formatter — unified analysis + economic assessment.

Merges the Analyst Agent and CFO Agent into a single LLM call that produces
statistical analysis WITH economic context, verdicts, and ₹ quantification.
This reduces 2 LLM calls to 1 while preserving all domain knowledge.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a senior business analyst AND CFO for a secondary steel manufacturing
company (induction furnace). Given SQL query results, column classifications,
business context, and metric thresholds, perform RIGOROUS statistical analysis
AND economic assessment in a SINGLE pass.

COLUMN CLASSIFICATION (auto-detected):
{column_classification}

DETECTED BUSINESS DOMAIN: {domain_hint}

BUSINESS CONTEXT:
{business_context}

METRIC THRESHOLDS:
{threshold_context}

═══════════════════════════════════════════════════
PART A: STATISTICAL ANALYSIS (Analyst perspective)
═══════════════════════════════════════════════════

Perform ALL that apply based on column types present:

1. DESCRIPTIVE STATISTICS (numeric columns):
   Mean, median, std deviation, min, max, P25/P75.
   Report coefficient of variation (std/mean) to indicate consistency.
   Compare against industry benchmarks.

2. GROUP-BY COMPARISONS (categorical + numeric columns):
   Mean/median of each numeric column per group. Rank groups.
   Identify best and worst performers with specific values.
   State N per group. Flag groups with N < 5.

3. CORRELATIONS (2+ numeric columns):
   Report pairs with |r| > 0.5 that are NOT trivially obvious.

4. TIME TRENDS (temporal + numeric columns):
   Trend direction and magnitude. Period-over-period changes.
   Flag deteriorating trends.

5. THRESHOLD COMPARISON (when thresholds available):
   Compare actual values against defined thresholds.
   Count data points in normal/warning/critical ranges.

═══════════════════════════════════════════════════
PART B: ECONOMIC ASSESSMENT (CFO perspective)
═══════════════════════════════════════════════════

STEEL PLANT ECONOMICS you know:
- Energy cost: 35-50% of total production cost (₹/ton)
- Scrap cost: 55-70% of total production cost
- Conversion cost (total - scrap): ₹3,000-5,000/ton
- Power tariff: ₹7-10/kWh by state
- SEC (kWh/ton): good <500, average 500-600, poor >600
- Yield: 1% improvement ≈ ₹300-500/ton savings
- Power factor below 0.85 incurs DISCOM penalty (₹/kVARh)

DOMAIN EXPERTISE:
- SEC = Specific Energy Consumption (kWh/ton), NOT seconds
- yield_pct = production yield percentage, NOT quantity
- rate = price per unit, NOT a percentage
- quantity = physical amount (weight/count), NOT yield
- fe_t, fe_mz, fe_content = Iron content variants

FINANCIAL LEAKAGE PATTERNS to flag:
- Weighbridge manipulation (input inflated / output deflated)
- Scrap grade mismatch (paying Grade A, receiving Grade B)
- Alloy over-addition (FeSi/FeMn > chemistry requirements)
- Power theft via meter bypass (consumption > metered)
- Furnace idle time (heat loss = wasted kWh = wasted ₹)
- Refractory over-consumption (relining too often)
- Casting yield loss (higher than expected metal loss)

YOUR ASSESSMENT MUST INCLUDE:

1. KEY METRICS: 3-5 most important numbers with verdicts:
   - "good": within normal range
   - "warning": needs monitoring
   - "critical": requires immediate action

2. COST/VALUE IMPACT: Quantify in ₹ where possible.
   Be specific with numbers. Annualize monthly impacts.

3. RECOMMENDATIONS: 2-4 actionable recommendations with ₹ impact.

Return ONLY a JSON object:
{{
  "findings": [
    {{
      "type": "trend|anomaly|insight|comparison|correlation|threshold_breach|leakage",
      "description": "Specific finding with actual numbers",
      "confidence": 0.0-1.0,
      "business_impact": "Why this matters — quantify in ₹ if possible",
      "verdict": "good|warning|critical"
    }}
  ],
  "summary": "2-3 sentence executive summary — most important takeaway FIRST",
  "key_metrics": [
    {{"label": "name", "value": "formatted value", "unit": "unit",
      "verdict": "good|warning|critical", "rupee_impact": "₹X/month or ₹X/year if calculable"}}
  ],
  "recommendations": [
    {{"action": "Specific recommendation", "impact": "₹X estimated impact", "priority": "high|medium|low"}}
  ],
  "chart_suggestions": [
    {{"type": "bar|line|scatter|pie", "x": "column", "y": ["column"],
      "title": "Title", "x_label": "Label", "y_label": "Label",
      "number_format": "decimal|currency|percentage",
      "insight": "What this chart reveals"}}
  ],
  "leakage_patterns": ["Any detected leakage pattern descriptions"],
  "approved": true
}}

Approve if the portfolio is healthy or improvements are clearly achievable.
Reject only if data reveals systemic problems requiring urgent attention.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _extract_json(raw: str) -> dict | None:
    """Robustly extract a JSON object from an LLM response."""
    text = raw.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            block = part.strip()
            for tag in ("json", "JSON"):
                if block.startswith(tag):
                    block = block[len(tag):].strip()
            try:
                return json.loads(block)
            except (json.JSONDecodeError, ValueError):
                continue
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return None


def _format_classification(classification: dict) -> tuple[str, str]:
    """Return (classification_summary, domain_hint) for the prompt."""
    if not classification:
        return "No classification available.", "general"
    cols = classification.get("columns", {})
    lines = []
    for name, info in cols.items():
        sem = info.get("semantic_type", "unknown")
        stats = info.get("stats")
        if stats:
            lines.append(
                f"  {name}: {sem} (mean={stats.get('mean')}, "
                f"range={stats.get('min')}-{stats.get('max')})"
            )
        else:
            lines.append(f"  {name}: {sem}")
    domain = classification.get("domain_hint", "general")
    return "\n".join(lines), domain


def _format_business_context(contexts: list[dict] | None) -> str:
    """Format business context from RAG state."""
    if not contexts:
        return "No business context available."
    parts = []
    for ctx in contexts:
        source = ctx.get("source", "unknown")
        content = ctx.get("content", "")
        if content:
            parts.append(f"[{source}] {content}")
    return "\n".join(parts) if parts else "No business context available."


def _extract_threshold_context(contexts: list[dict] | None) -> str:
    """Extract metric threshold context from RAG contexts."""
    if not contexts:
        return "No thresholds defined. Use industry benchmarks for secondary steel."
    for ctx in contexts:
        if ctx.get("source") == "metric_thresholds":
            return ctx.get("content", "No thresholds defined.")
    return "No thresholds defined. Use industry benchmarks for secondary steel."


def _build_data_section(state: dict[str, Any]) -> tuple[str, int]:
    """Build the data section from single SQL result.

    v4: Single-query pipeline — reads from ``sql_result`` directly.
    """
    result = state.get("sql_result", {})
    if not result:
        return "", 0

    rows = result.get("rows", [])
    if not rows:
        return "", 0

    query = result.get("query", "")
    task = result.get("task", "Query")
    sample = rows[:20]
    data_text = (
        f"--- {task} ---\n"
        f"SQL: {query}\n"
        f"Result rows ({len(rows)} total, showing {len(sample)}):\n"
        f"{json.dumps(sample, default=str, indent=2)}"
    )
    return data_text, len(rows)


class InsightFormatter:
    """Unified analysis + economic assessment in a single LLM call.

    Merges AnalystAgent + CFOAgent into one step:
    - Statistical analysis (5 core categories)
    - Economic assessment (verdicts, ₹ quantification)
    - Leakage pattern detection (7 patterns)
    - Recommendations with impact estimates
    - Chart suggestions (unified list)
    """

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Produce findings, key metrics, recommendations, and approval."""
        question = state.get("question", "")
        classification = state.get("column_classification", {})
        rag_contexts = state.get("_rag_contexts", [])

        classification_text, domain_hint = _format_classification(classification)
        business_context = _format_business_context(rag_contexts)
        threshold_context = _extract_threshold_context(rag_contexts)

        # Build data section
        data_text, total_rows = _build_data_section(state)

        if not total_rows:
            state["analysis"] = {
                "findings": [],
                "summary": "No data was returned by the SQL query. The question may not "
                           "match any uploaded data, or the table might be empty.",
                "chart_suggestions": [],
            }
            state["key_metrics"] = []
            state["recommendations"] = []
            state["leakage_patterns"] = []
            state["approved"] = True
            state["cfo_notes"] = "No data to assess."
            state["cfo_key_metrics"] = []
            state["cfo_chart_suggestions"] = []
            return state

        system = SYSTEM_PROMPT.format(
            column_classification=classification_text,
            domain_hint=domain_hint,
            business_context=business_context,
            threshold_context=threshold_context,
        )

        prompt_parts = [
            system,
            "",
            f"Original question: {question}",
            f"\nDATA:\n{data_text}",
        ]

        prompt = "\n".join(prompt_parts)

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            raw = response.text.strip()
            result = _extract_json(raw)
            if result is None:
                raise ValueError(f"Could not parse Insight Formatter JSON: {raw[:200]}")

            # Populate analyst-compatible fields
            state["analysis"] = {
                "findings": result.get("findings", []),
                "summary": result.get("summary", ""),
                "chart_suggestions": result.get("chart_suggestions", []),
            }

            # Populate CFO-compatible fields
            state["approved"] = result.get("approved", True)
            state["cfo_notes"] = result.get("summary", "")
            state["recommendations"] = [
                r.get("action", str(r)) if isinstance(r, dict) else str(r)
                for r in result.get("recommendations", [])
            ]
            state["cfo_key_metrics"] = result.get("key_metrics", [])
            state["cfo_chart_suggestions"] = result.get("chart_suggestions", [])

            # New v4 fields
            state["key_metrics"] = result.get("key_metrics", [])
            state["leakage_patterns"] = result.get("leakage_patterns", [])

        except Exception:
            logger.exception("Insight Formatter failed")
            state["analysis"] = {
                "findings": [
                    {
                        "type": "insight",
                        "description": "Automated analysis failed. Raw SQL data is available.",
                        "confidence": 0,
                        "business_impact": "Unable to assess — manual review required.",
                    }
                ],
                "summary": "Automated analysis failed. See raw data in details section.",
                "chart_suggestions": [],
            }
            state["approved"] = False
            state["cfo_notes"] = "Insight Formatter failed — manual review required."
            state["recommendations"] = []
            state["cfo_key_metrics"] = []
            state["cfo_chart_suggestions"] = []
            state["key_metrics"] = []
            state["leakage_patterns"] = []

        return state
