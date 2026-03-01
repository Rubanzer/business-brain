"""Delta Engine — Track 3 disagreement detection.

Compares algorithmic findings (Track 1) against contextual evaluation (Track 2)
to surface the most interesting discrepancies.

5 delta types:
- FILTERED_BY_CONTEXT: High algorithmic score, but domain agent says IRRELEVANT.
- EXPECTED_BUT_ABSENT: Domain context suggests something should exist, but Track 1 didn't find it.
- MAGNITUDE_DISAGREEMENT: Algorithmic magnitude differs from domain benchmarks.
- UNEXPLAINED_SIGNAL: Track 1 found something with no domain context at all.
- SEGMENT_REVERSAL: Finding is true globally but reverses in a specific segment (Simpson's paradox).
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.models import AnalysisDelta, AnalysisResult, AgentOutput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_HIGH_ALGORITHMIC_SCORE = 0.6
_LOW_DOMAIN_RELEVANCE = 0.3
_HIGH_DOMAIN_RELEVANCE = 0.7
_SEGMENT_REVERSAL_THRESHOLD = 0.5  # effect sign flip with this min magnitude


# ---------------------------------------------------------------------------
# Delta detectors
# ---------------------------------------------------------------------------


def _detect_filtered_by_context(
    result: AnalysisResult,
    domain_output: dict | None,
) -> AnalysisDelta | None:
    """High algorithmic score, but domain says irrelevant."""
    if result.interestingness_score < _HIGH_ALGORITHMIC_SCORE:
        return None
    if not domain_output:
        return None

    classification = domain_output.get("classification", "")
    relevance = domain_output.get("relevance_score", 0.5)

    if classification == "IRRELEVANT" or relevance < _LOW_DOMAIN_RELEVANCE:
        return AnalysisDelta(
            run_id=result.run_id,
            result_id=result.id,
            delta_type="FILTERED_BY_CONTEXT",
            description=(
                f"Algorithmically interesting ({result.interestingness_score:.2f}) "
                f"but domain context classifies as {classification} (relevance={relevance:.2f})"
            ),
            algorithmic_view={
                "interestingness": result.interestingness_score,
                "breakdown": result.interestingness_breakdown,
            },
            contextual_view={
                "classification": classification,
                "relevance": relevance,
                "translation": domain_output.get("business_translation", ""),
            },
        )
    return None


def _detect_unexplained_signal(
    result: AnalysisResult,
    domain_output: dict | None,
) -> AnalysisDelta | None:
    """Track 1 found something interesting with no domain context."""
    if result.interestingness_score < _HIGH_ALGORITHMIC_SCORE:
        return None

    # No domain output, or domain has no benchmarks and classified as NOVEL
    if domain_output is None:
        return AnalysisDelta(
            run_id=result.run_id,
            result_id=result.id,
            delta_type="UNEXPLAINED_SIGNAL",
            description=(
                f"Interesting finding ({result.interestingness_score:.2f}) "
                f"with no domain context available"
            ),
            algorithmic_view={
                "interestingness": result.interestingness_score,
                "operation": result.operation_type,
                "target": result.target,
            },
            contextual_view=None,
        )

    if domain_output.get("classification") == "NOVEL" and domain_output.get("relevance_score", 0) > _HIGH_DOMAIN_RELEVANCE:
        return AnalysisDelta(
            run_id=result.run_id,
            result_id=result.id,
            delta_type="UNEXPLAINED_SIGNAL",
            description=(
                f"Novel and relevant ({domain_output.get('relevance_score', 0):.2f}) "
                f"finding with no historical precedent"
            ),
            algorithmic_view={
                "interestingness": result.interestingness_score,
                "operation": result.operation_type,
            },
            contextual_view={
                "classification": "NOVEL",
                "translation": domain_output.get("business_translation", ""),
            },
        )
    return None


def _detect_magnitude_disagreement(
    result: AnalysisResult,
    domain_output: dict | None,
) -> AnalysisDelta | None:
    """Algorithmic magnitude differs significantly from domain expectations."""
    if not domain_output:
        return None

    benchmark = domain_output.get("benchmark_context", "")
    if not benchmark or benchmark == "No benchmarks available":
        return None

    # Check if domain agent flagged it as SURPRISING
    if domain_output.get("classification") == "SURPRISING" and result.interestingness_score > 0.5:
        return AnalysisDelta(
            run_id=result.run_id,
            result_id=result.id,
            delta_type="MAGNITUDE_DISAGREEMENT",
            description=(
                f"Finding surprises domain expectations: "
                f"{domain_output.get('business_translation', '')[:200]}"
            ),
            algorithmic_view={
                "interestingness": result.interestingness_score,
                "result_data_summary": _summarize_result(result),
            },
            contextual_view={
                "classification": "SURPRISING",
                "benchmark": benchmark[:300],
                "impact": domain_output.get("estimated_impact", "MEDIUM"),
            },
        )
    return None


def _detect_segment_reversal(
    result: AnalysisResult,
    all_results: list[AnalysisResult],
) -> AnalysisDelta | None:
    """Simpson's paradox: finding is true globally but reverses in a segment."""
    if not result.segmenters:
        return None

    # Look for the same operation on the same target without segmenters
    global_match = None
    for r in all_results:
        if (
            r.id != result.id
            and r.operation_type == result.operation_type
            and r.table_name == result.table_name
            and r.target == result.target
            and not r.segmenters
        ):
            global_match = r
            break

    if not global_match:
        return None

    # Compare directions for CORRELATE
    if result.operation_type == "CORRELATE":
        global_r = (global_match.result_data or {}).get("pearson_r", 0)
        segment_r = (result.result_data or {}).get("pearson_r", 0)
        if (
            abs(global_r) > _SEGMENT_REVERSAL_THRESHOLD
            and abs(segment_r) > _SEGMENT_REVERSAL_THRESHOLD
            and (global_r > 0) != (segment_r > 0)
        ):
            return AnalysisDelta(
                run_id=result.run_id,
                result_id=result.id,
                delta_type="SEGMENT_REVERSAL",
                description=(
                    f"Simpson's paradox: correlation is {global_r:+.2f} globally "
                    f"but {segment_r:+.2f} when segmented by {result.segmenters}"
                ),
                algorithmic_view={
                    "global_r": global_r,
                    "segmented_r": segment_r,
                    "segmenters": result.segmenters,
                },
                contextual_view=None,
            )

    # Compare directions for RANK
    if result.operation_type == "RANK":
        global_ranked = (global_match.result_data or {}).get("ranked", [])
        segment_ranked = (result.result_data or {}).get("ranked", [])
        if global_ranked and segment_ranked and len(global_ranked) >= 2 and len(segment_ranked) >= 2:
            # Check if top and bottom swap
            global_top = str(list(global_ranked[0].values())[0]) if global_ranked[0] else ""
            segment_bottom = str(list(segment_ranked[-1].values())[0]) if segment_ranked[-1] else ""
            if global_top and global_top == segment_bottom:
                return AnalysisDelta(
                    run_id=result.run_id,
                    result_id=result.id,
                    delta_type="SEGMENT_REVERSAL",
                    description=(
                        f"Ranking reversal: '{global_top}' is top globally "
                        f"but bottom when segmented by {result.segmenters}"
                    ),
                    algorithmic_view={"global_top": global_top, "segmented_ranking": segment_ranked[:3]},
                    contextual_view=None,
                )

    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _summarize_result(result: AnalysisResult) -> dict:
    """Extract a compact summary from result_data."""
    data = result.result_data or {}
    summary = {"operation": result.operation_type}

    if "pearson_r" in data:
        summary["correlation"] = data["pearson_r"]
    if "count" in data:
        summary["anomaly_count"] = data["count"]
    if "comparison" in data:
        comp = data["comparison"]
        if isinstance(comp, dict):
            summary["p_value"] = comp.get("p_value")
            summary["effect_size"] = comp.get("cohens_d") or comp.get("eta_squared")
    if "stats" in data:
        stats = data["stats"]
        summary["mean"] = stats.get("mean")
        summary["stdev"] = stats.get("stdev")

    return summary


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


async def compute_deltas(
    session: AsyncSession,
    results: list[AnalysisResult],
    domain_outputs: dict[str, AgentOutput],
    run_id: str,
) -> list[AnalysisDelta]:
    """Compute all deltas for a set of analysis results.

    Args:
        results: All AnalysisResults from this run.
        domain_outputs: result_id -> AgentOutput from domain agent.
        run_id: The analysis run ID.

    Returns:
        List of AnalysisDelta records (already added to session).
    """
    deltas: list[AnalysisDelta] = []

    for result in results:
        domain_out = domain_outputs.get(result.id)
        domain_data = domain_out.output if domain_out else None

        # Check each delta type
        for detector in (
            lambda r: _detect_filtered_by_context(r, domain_data),
            lambda r: _detect_unexplained_signal(r, domain_data),
            lambda r: _detect_magnitude_disagreement(r, domain_data),
            lambda r: _detect_segment_reversal(r, results),
        ):
            delta = detector(result)
            if delta:
                session.add(delta)
                deltas.append(delta)

                # Tag the result
                result.delta_type = delta.delta_type

    await session.flush()
    logger.info("Delta engine: %d deltas found across %d results", len(deltas), len(results))
    return deltas
