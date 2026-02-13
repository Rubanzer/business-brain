"""CFO Filter agent — economic viability gate for proposed actions."""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the CFO reviewing analysis results for a business. Your job is to extract the
most important business implications and make actionable recommendations.

COLUMN CLASSIFICATION (auto-detected):
{column_classification}

DETECTED BUSINESS DOMAIN: {domain_hint}

YOUR ASSESSMENT MUST COVER:

1. KEY METRICS: Extract the 3-5 most important numbers from the analysis.
   For each, assign a verdict:
   - "good": favorable metric (low cost, high efficiency, strong growth)
   - "warning": needs monitoring (declining trend, moderate variance)
   - "critical": requires immediate action (cost spike, quality below threshold)

2. COST/VALUE IMPACT: Quantify the business impact where possible.
   Be specific: "Supplier X costs 8% more but yields 5% less, increasing
   effective cost by Rs 2,400/ton" — not "costs vary across suppliers."

3. RISK ASSESSMENT: Identify which areas have high variance or inconsistency.
   High standard deviation in quality/cost = operational risk.

4. RECOMMENDATIONS: 2-4 specific, actionable recommendations with expected impact.
   Each recommendation should reference specific data points.

5. CHART SUGGESTIONS: 1-2 charts that highlight the most important economic insight
   from the data. These will be auto-rendered prominently on the dashboard.

Return ONLY a JSON object:
{{
  "approved": true/false,
  "reasoning": "2-3 sentences referencing SPECIFIC numbers from the analysis",
  "recommendations": ["Specific actionable recommendation with numbers"],
  "key_metrics": [
    {{"label": "name", "value": "formatted value", "unit": "unit", "verdict": "good|warning|critical"}}
  ],
  "chart_suggestions": [
    {{"type": "bar|scatter|line", "x": "column", "y": ["column"], "title": "Title",
     "x_label": "Label", "y_label": "Label", "number_format": "currency|percentage|decimal",
     "insight": "Why this chart matters"}}
  ]
}}

Approve if the portfolio is healthy or if improvements are clearly achievable.
Reject only if data reveals systemic problems requiring urgent attention.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _format_classification_for_cfo(classification: dict) -> tuple[str, str]:
    """Return (classification_summary, domain_hint) for CFO prompt."""
    if not classification:
        return "No classification available.", "general"
    cols = classification.get("columns", {})
    lines = []
    for name, info in cols.items():
        sem = info.get("semantic_type", "unknown")
        stats = info.get("stats")
        if stats:
            lines.append(f"  {name}: {sem} (mean={stats.get('mean')}, range={stats.get('min')}-{stats.get('max')})")
        else:
            lines.append(f"  {name}: {sem}")
    domain = classification.get("domain_hint", "general")
    return "\n".join(lines), domain


class CFOAgent:
    """Gates analysis outputs through an economic viability check."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate whether the analysis findings are economically viable."""
        analysis = state.get("analysis", {})
        python_analysis = state.get("python_analysis", {})
        question = state.get("question", "")
        classification = state.get("column_classification", {})

        classification_text, domain_hint = _format_classification_for_cfo(classification)

        system = SYSTEM_PROMPT.format(
            column_classification=classification_text,
            domain_hint=domain_hint,
        )

        prompt_parts = [
            system,
            "",
            f"Original question: {question}",
            f"Analysis findings:\n{json.dumps(analysis, default=str, indent=2)}",
        ]

        # Include computational analysis if available
        computations = python_analysis.get("computations", [])
        narrative = python_analysis.get("narrative", "")
        if computations or narrative:
            prompt_parts.append("\nComputational Analysis:")
            if computations:
                prompt_parts.append(f"Metrics: {json.dumps(computations, default=str)}")
            if narrative:
                prompt_parts.append(f"Interpretation: {narrative}")

        prompt = "\n".join(prompt_parts)

        try:
            client = _get_client()
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            raw = response.text.strip()
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
            state["approved"] = result.get("approved", False)
            state["cfo_notes"] = result.get("reasoning", "")
            state["recommendations"] = result.get("recommendations", [])
            state["cfo_key_metrics"] = result.get("key_metrics", [])
            state["cfo_chart_suggestions"] = result.get("chart_suggestions", [])
        except Exception:
            logger.exception("CFO evaluation failed")
            state["approved"] = False
            state["cfo_notes"] = "Automated CFO evaluation failed — manual review required."
            state["recommendations"] = []
            state["cfo_key_metrics"] = []
            state["cfo_chart_suggestions"] = []

        return state
