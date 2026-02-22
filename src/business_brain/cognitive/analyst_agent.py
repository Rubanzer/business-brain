"""Data Scientist agent — statistical analysis with Gemini.

Now receives business context and metric thresholds from the RAG pipeline
so it can interpret findings in the correct business domain and flag values
against known thresholds.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from google import genai

from config.settings import settings

logger = logging.getLogger(__name__)


def _extract_json(raw: str) -> dict | None:
    """Robustly extract a JSON object from an LLM response.

    Handles:
      - Raw JSON
      - JSON wrapped in ```json ... ``` fences
      - JSON embedded in prose text
    """
    text = raw.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown fences
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:  # odd-indexed parts are inside fences
            block = part.strip()
            for tag in ("json", "JSON"):
                if block.startswith(tag):
                    block = block[len(tag):].strip()
            try:
                return json.loads(block)
            except (json.JSONDecodeError, ValueError):
                continue

    # Try finding JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except (json.JSONDecodeError, ValueError):
            pass

    return None


SYSTEM_PROMPT = """\
You are a senior data analyst with DEEP EXPERTISE in manufacturing operations,
especially secondary steel manufacturing (induction furnace). Given SQL query results,
a column classification report, business context, and metric thresholds, perform
RIGOROUS statistical analysis with DOMAIN-AWARE interpretation.

COLUMN CLASSIFICATION (auto-detected):
{column_classification}

BUSINESS CONTEXT (company profile, domain knowledge, KPIs):
{business_context}

METRIC THRESHOLDS (flag values against these benchmarks):
{threshold_context}

DOMAIN EXPERTISE — use this knowledge when interpreting results:
- Know that SEC (Specific Energy Consumption) is measured in kWh/ton; good is <500, poor is >600
- Know that Yield = output/input × 100; good is >92%, poor is <88%
- Know that Power Factor should be >0.95; below 0.85 means capacitor bank failure
- Know that tap-to-tap time should be 45-90 min; >120 min is a red flag
- Know that furnace temperature should be 1550-1650°C for tapping
- Compare values against INDUSTRY BENCHMARKS, not just statistical norms
- Always explain "So what?" — what does the finding mean for the business?
- Flag RED FLAGS automatically (values that suggest equipment issues, data errors, or financial leakage)
- When column names include "fe_t", "fe_mz", etc., interpret these as Iron Content variants
- When you see "kva" vs "kwh", explain the difference (apparent vs real power)

REQUIRED ANALYSIS — perform ALL that apply based on column types present:

1. DESCRIPTIVE STATISTICS (when numeric columns exist):
   For each numeric column: mean, median, std deviation, min, max, P25/P75.
   Report the coefficient of variation (std/mean) to indicate consistency.
   Compare against industry benchmarks if available.

2. GROUP-BY COMPARISONS (when categorical + numeric columns exist):
   For each categorical column, compute mean/median of each numeric column per group.
   Rank groups. Identify the best and worst performers with specific values.
   State the sample size (N) per group. Flag groups with N < 5 as unreliable.
   Quantify the gap between best and worst performers.

3. CORRELATIONS (when 2+ numeric columns exist):
   State which numeric pairs are positively/negatively correlated.
   Only report correlations with |r| > 0.5 that are NOT trivially obvious
   (e.g., skip correlations between variants of the same measurement like fe_t ↔ fe_mz).

4. TIME TRENDS (when temporal + numeric columns exist):
   Describe the trend direction and magnitude.
   Note period-over-period changes.
   Flag deteriorating trends that need attention.

5. OUTLIER FLAGGING (when numeric columns exist):
   Identify values more than 3 standard deviations from mean.
   Note which categories they belong to.
   Assess if outliers are data errors vs legitimate extreme events.

6. DISTRIBUTION SHAPE (when numeric columns exist):
   Is the data normally distributed, skewed, or multimodal?
   Are there clusters?

7. THRESHOLD COMPARISON (when thresholds are available):
   Compare actual values against defined thresholds.
   Flag any values in warning or critical ranges with specific numbers.
   State how many data points fall in each range (normal/warning/critical).

8. BUSINESS IMPACT ASSESSMENT (always):
   For every finding, add a "business_impact" that explains WHY this matters.
   Quantify financial impact where possible (e.g., "this variance costs ~₹X per month").
   Reference domain benchmarks to contextualize findings.

Return ONLY a JSON object:
{{
  "findings": [
    {{
      "type": "trend|anomaly|insight|comparison|correlation|threshold_breach",
      "description": "Specific finding with actual numbers and names",
      "confidence": 0.0-1.0,
      "business_impact": "Why this matters for business decisions"
    }}
  ],
  "summary": "2-3 sentence executive summary — most important takeaway FIRST",
  "chart_suggestions": [
    {{
      "type": "bar|line|pie|scatter",
      "x": "exact_column_name",
      "y": ["exact_column_name"],
      "title": "Descriptive Title",
      "x_label": "Axis Label (unit)",
      "y_label": "Axis Label (unit)",
      "number_format": "decimal|currency|percentage",
      "insight": "What this chart reveals"
    }}
  ]
}}
"""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def _build_data_section(state: dict[str, Any]) -> tuple[str, int]:
    """Build the data section for the analyst prompt from single or multi-query results.

    Returns (data_text, total_row_count).
    """
    # Try multi-query results first, fall back to single sql_result
    sql_results = state.get("sql_results")
    if not sql_results:
        single = state.get("sql_result", {})
        if single:
            sql_results = [single]
        else:
            return "", 0

    parts = []
    total_rows = 0
    for i, res in enumerate(sql_results):
        rows = res.get("rows", [])
        total_rows += len(rows)
        query = res.get("query", "")
        task = res.get("task", f"Query {i + 1}")
        sample = rows[:20]
        header = f"--- Query {i + 1}: {task} ---"
        parts.append(
            f"{header}\n"
            f"SQL: {query}\n"
            f"Result rows ({len(rows)} total, showing {len(sample)}):\n"
            f"{json.dumps(sample, default=str, indent=2)}"
        )

    return "\n\n".join(parts), total_rows


def _get_all_rows(state: dict[str, Any]) -> list[dict]:
    """Collect all rows from single or multi-query results."""
    sql_results = state.get("sql_results")
    if sql_results:
        rows = []
        for res in sql_results:
            rows.extend(res.get("rows", []))
        return rows
    single = state.get("sql_result", {})
    return single.get("rows", [])


def _format_business_context_for_analyst(contexts: list[dict] | None) -> str:
    """Format business context from RAG state for the analyst prompt."""
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
        return "No thresholds defined."
    for ctx in contexts:
        if ctx.get("source") == "metric_thresholds":
            return ctx.get("content", "No thresholds defined.")
    return "No thresholds defined."


class AnalystAgent:
    """Performs analysis on query results using Gemini.

    Now receives business context and thresholds from the RAG pipeline
    to ground its analysis in domain knowledge.
    """

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Analyse the SQL result set(s)."""
        from business_brain.cognitive.column_classifier import (
            classify_columns,
            format_classification_for_prompt,
        )

        question = state.get("question", "")
        data_section, total_rows = _build_data_section(state)

        if not total_rows:
            logger.warning("No rows to analyse")
            state["analysis"] = {
                "findings": [],
                "summary": "No data was returned by the query.",
            }
            return state

        # Run column classifier on actual data
        all_rows = _get_all_rows(state)
        columns = list(all_rows[0].keys()) if all_rows else []
        classification = classify_columns(columns, all_rows[:100])
        state["column_classification"] = classification

        classification_text = format_classification_for_prompt(classification)
        domain_hint = classification.get("domain_hint", "general")

        # Get business context and thresholds from RAG state
        rag_contexts = state.get("_rag_contexts", [])
        business_context = _format_business_context_for_analyst(rag_contexts)
        threshold_context = _extract_threshold_context(rag_contexts)

        system = SYSTEM_PROMPT.format(
            column_classification=classification_text,
            business_context=business_context,
            threshold_context=threshold_context,
        )

        prompt = (
            f"{system}\n\n"
            f"Original question: {question}\n\n"
            f"{data_section}"
        )

        analysis = None
        for attempt in range(2):
            try:
                client = _get_client()
                use_prompt = prompt if attempt == 0 else (
                    f"{system}\n\n"
                    f"Original question: {question}\n\n"
                    f"Previous analysis attempt returned invalid or empty results. "
                    f"Please provide a simpler analysis focusing on basic statistics.\n\n"
                    f"{data_section}"
                )
                response = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=use_prompt,
                )
                raw = response.text.strip()
                # Robust JSON extraction from LLM response
                parsed = _extract_json(raw)
                if parsed is None:
                    logger.warning("Could not parse analyst JSON (attempt %d): %s",
                                   attempt, raw[:200])
                    if attempt == 0:
                        continue
                    # Last resort: wrap the raw text as a finding
                    analysis = {
                        "findings": [{"type": "insight", "description": raw[:500], "confidence": 0.3}],
                        "summary": raw[:200],
                        "chart_suggestions": [],
                    }
                    break

                # Validate: must have findings list and summary
                if parsed.get("findings") and parsed.get("summary"):
                    analysis = parsed
                    break
                elif attempt == 0:
                    logger.warning("Analysis returned empty findings, retrying with simplified prompt")
                    continue
                else:
                    analysis = parsed
            except Exception:
                logger.exception("Analysis LLM call failed (attempt %d)", attempt)
                if attempt == 0:
                    continue

        if analysis is None:
            analysis = {
                "findings": [{"type": "insight", "description": f"Retrieved {total_rows} rows but automated analysis could not complete.", "confidence": 0.3}],
                "summary": f"Query returned {total_rows} rows. Automated analysis failed — see SQL results in the details section for raw data.",
                "chart_suggestions": [],
            }

        # Ensure chart_suggestions always exists (frontend expects it)
        analysis.setdefault("chart_suggestions", [])

        state["analysis"] = analysis
        return state
