"""CFO Filter agent — economic viability gate for proposed actions."""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a CFO reviewing analysis findings and computational metrics. For each
finding, assess:
1. Expected revenue impact (positive/negative/neutral)
2. Implementation cost (low/medium/high)
3. Risk level (low/medium/high)

Review both the analytical findings AND the computational analysis (metrics,
statistical computations, and narrative interpretation) to make your decision.

Return ONLY a JSON object:
{
  "approved": true/false,
  "reasoning": "Brief explanation of your decision",
  "recommendations": ["actionable recommendation 1", "..."]
}

Only approve if the overall expected ROI is positive and risk is acceptable.
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


class CFOAgent:
    """Gates analysis outputs through an economic viability check."""

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Evaluate whether the analysis findings are economically viable."""
        analysis = state.get("analysis", {})
        python_analysis = state.get("python_analysis", {})
        question = state.get("question", "")

        prompt_parts = [
            SYSTEM_PROMPT,
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
        except Exception:
            logger.exception("CFO evaluation failed")
            state["approved"] = False
            state["cfo_notes"] = "Automated CFO evaluation failed — manual review required."
            state["recommendations"] = []

        return state
