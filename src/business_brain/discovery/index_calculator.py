"""Composite index calculator — construct weighted index scores from multiple metrics.

Pure functions for building standardized index scores (0-100) from
multiple input metrics with configurable weights and normalization.
Common use: supplier score, risk score, performance index.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IndexComponent:
    """A single component of the index."""
    name: str
    raw_value: float
    normalized_value: float  # 0-100 scale
    weight: float
    weighted_contribution: float  # normalized * weight
    direction: str  # "higher_is_better" or "lower_is_better"


@dataclass
class IndexScore:
    """Computed index score for one entity."""
    entity: str
    score: float  # 0-100
    grade: str  # A/B/C/D/F
    components: list[IndexComponent]
    rank: int


@dataclass
class IndexResult:
    """Complete index computation result."""
    name: str
    scores: list[IndexScore]
    entity_count: int
    mean_score: float
    median_score: float
    std_score: float
    top_entity: str
    bottom_entity: str
    summary: str


def compute_index(
    rows: list[dict],
    entity_column: str,
    metrics: list[dict],
    index_name: str = "Index",
) -> IndexResult | None:
    """Compute a composite index score for each entity.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying each entity.
        metrics: List of metric configs:
            [{"column": "revenue", "weight": 0.4, "direction": "higher_is_better"},
             {"column": "defects", "weight": 0.3, "direction": "lower_is_better"},
             {"column": "on_time_pct", "weight": 0.3, "direction": "higher_is_better"}]
        index_name: Name for the index.

    Returns:
        IndexResult or None if insufficient data.
    """
    if not rows or not metrics:
        return None

    # Aggregate by entity (take mean for each metric)
    entity_data: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        entity = row.get(entity_column)
        if entity is None:
            continue
        key = str(entity)
        if key not in entity_data:
            entity_data[key] = {}
        for m in metrics:
            col = m["column"]
            val = row.get(col)
            if val is None:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if col not in entity_data[key]:
                entity_data[key][col] = []
            entity_data[key][col].append(fval)

    if len(entity_data) < 2:
        return None

    # Compute means per entity per metric
    entity_means: dict[str, dict[str, float]] = {}
    for entity, cols in entity_data.items():
        entity_means[entity] = {}
        for m in metrics:
            col = m["column"]
            vals = cols.get(col, [])
            if vals:
                entity_means[entity][col] = sum(vals) / len(vals)

    # Find min/max for each metric (for normalization)
    metric_ranges: dict[str, tuple[float, float]] = {}
    for m in metrics:
        col = m["column"]
        all_vals = [em.get(col, 0) for em in entity_means.values() if col in em]
        if all_vals:
            metric_ranges[col] = (min(all_vals), max(all_vals))
        else:
            metric_ranges[col] = (0, 1)

    # Normalize weights
    total_weight = sum(m.get("weight", 1.0) for m in metrics)
    if total_weight == 0:
        total_weight = len(metrics)

    # Compute scores
    scores: list[IndexScore] = []
    for entity, means in entity_means.items():
        components = []
        total_score = 0.0

        for m in metrics:
            col = m["column"]
            weight = m.get("weight", 1.0) / total_weight
            direction = m.get("direction", "higher_is_better")
            raw = means.get(col, 0.0)

            # Normalize to 0-100
            lo, hi = metric_ranges.get(col, (0, 1))
            if hi - lo > 0:
                normalized = (raw - lo) / (hi - lo) * 100
            else:
                normalized = 50.0  # all same value

            # Invert for "lower_is_better"
            if direction == "lower_is_better":
                normalized = 100 - normalized

            weighted = normalized * weight

            components.append(IndexComponent(
                name=col,
                raw_value=round(raw, 4),
                normalized_value=round(normalized, 2),
                weight=round(weight, 4),
                weighted_contribution=round(weighted, 2),
                direction=direction,
            ))
            total_score += weighted

        grade = _score_to_grade(total_score)
        scores.append(IndexScore(
            entity=entity,
            score=round(total_score, 2),
            grade=grade,
            components=components,
            rank=0,  # set after sorting
        ))

    # Sort and rank
    scores.sort(key=lambda s: -s.score)
    for i, s in enumerate(scores):
        s.rank = i + 1

    # Statistics
    all_scores = [s.score for s in scores]
    mean_score = sum(all_scores) / len(all_scores)
    sorted_scores = sorted(all_scores)
    n = len(sorted_scores)
    median_score = sorted_scores[n // 2] if n % 2 else (sorted_scores[n // 2 - 1] + sorted_scores[n // 2]) / 2
    variance = sum((s - mean_score) ** 2 for s in all_scores) / n
    std_score = variance ** 0.5

    summary = (
        f"{index_name}: {len(scores)} entities scored. "
        f"Mean: {mean_score:.1f}, Median: {median_score:.1f}. "
        f"Top: {scores[0].entity} ({scores[0].score:.1f}), "
        f"Bottom: {scores[-1].entity} ({scores[-1].score:.1f})."
    )

    return IndexResult(
        name=index_name,
        scores=scores,
        entity_count=len(scores),
        mean_score=round(mean_score, 2),
        median_score=round(median_score, 2),
        std_score=round(std_score, 2),
        top_entity=scores[0].entity,
        bottom_entity=scores[-1].entity,
        summary=summary,
    )


def compare_entities(result: IndexResult, entity_a: str, entity_b: str) -> dict:
    """Compare two entities' index scores component by component."""
    score_a = next((s for s in result.scores if s.entity == entity_a), None)
    score_b = next((s for s in result.scores if s.entity == entity_b), None)

    if not score_a or not score_b:
        return {"error": "Entity not found"}

    components = []
    for ca, cb in zip(score_a.components, score_b.components):
        components.append({
            "metric": ca.name,
            "a_normalized": ca.normalized_value,
            "b_normalized": cb.normalized_value,
            "a_raw": ca.raw_value,
            "b_raw": cb.raw_value,
            "winner": entity_a if ca.normalized_value >= cb.normalized_value else entity_b,
        })

    return {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "score_a": score_a.score,
        "score_b": score_b.score,
        "winner": entity_a if score_a.score >= score_b.score else entity_b,
        "components": components,
    }


def format_index_table(result: IndexResult) -> str:
    """Format index scores as a text table."""
    lines = [
        f"{result.name}",
        f"{'='*60}",
        f"{'Rank':<6}{'Entity':<25}{'Score':>8}{'Grade':>6}",
        f"{'-'*45}",
    ]
    for s in result.scores:
        lines.append(f"{s.rank:<6}{s.entity:<25}{s.score:>8.1f}{s.grade:>6}")
    lines.append(f"{'-'*45}")
    lines.append(f"Mean: {result.mean_score:.1f}  |  Std: {result.std_score:.1f}")
    return "\n".join(lines)


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"
