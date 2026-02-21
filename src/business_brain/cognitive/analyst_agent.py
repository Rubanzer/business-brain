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

SYSTEM_PROMPT = """\
You are a senior data analyst at a manufacturing company. Given SQL query results,
a column classification report, business context, and metric thresholds, perform
RIGOROUS statistical analysis.

COLUMN CLASSIFICATION (auto-detected):
{column_classification}

BUSINESS CONTEXT (company profile, domain knowledge, KPIs):
{business_context}

METRIC THRESHOLDS (flag values against these benchmarks):
{threshold_context}

REQUIRED ANALYSIS — perform ALL that apply based on column types present:

1. DESCRIPTIVE STATISTICS (when numeric columns exist):
   For each numeric column: mean, median, std deviation, min, max, P25/P75.
   Report the coefficient of variation (std/mean) to indicate consistency.

2. GROUP-BY COMPARISONS (when categorical + numeric columns exist):
   For each categorical column, compute mean/median of each numeric column per group.
   Rank groups. Identify the best and worst performers with specific values.
   State the sample size (N) per group. Flag groups with N < 5 as unreliable.

3. CORRELATIONS (when 2+ numeric columns exist):
   State which numeric pairs are positively/negatively correlated.
   Only report correlations with |r| > 0.3.

4. TIME TRENDS (when temporal + numeric columns exist):
   Describe the trend direction and magnitude.
   Note period-over-period changes.

5. OUTLIER FLAGGING (when numeric columns exist):
   Identify values more than 2 standard deviations from mean.
   Note which categories they belong to.

6. DISTRIBUTION SHAPE (when numeric columns exist):
   Is the data normally distributed, skewed, or multimodal?
   Are there clusters?

7. THRESHOLD COMPARISON (when thresholds are available):
   Compare actual values against defined thresholds.
   Flag any values in warning or critical ranges with specific numbers.
   State how many data points fall in each range (normal/warning/critical).

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
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                parsed = json.loads(raw)
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
                "findings": [{"type": "insight", "description": f"Retrieved {total_rows} rows.", "confidence": 0.5}],
                "summary": f"Query returned {total_rows} rows but automated analysis failed.",
            }

        state["analysis"] = analysis
        return state
