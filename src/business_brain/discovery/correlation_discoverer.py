"""Correlation discovery — finds notable correlations from table profiles.

Uses column classification stats to identify potentially correlated numeric columns
and generates insights. Includes both sample-based (fast) and SQL-backed (accurate)
versions.
"""

from __future__ import annotations

import logging
import re
import uuid

from business_brain.db.discovery_models import Insight

logger = logging.getLogger(__name__)


def _humanize_table(name: str) -> str:
    """Convert table name to human-readable form."""
    cleaned = re.sub(r"^\d+[_\s]*", "", name)
    return cleaned.replace("_", " ").title() if cleaned else name.replace("_", " ").title()


def _humanize_col(name: str) -> str:
    """Convert column name to human-readable form."""
    return name.replace("_", " ").title()


def discover_correlations_from_profiles(profiles: list) -> list[Insight]:
    """Scan profiles for numeric column pairs and flag potential correlations.

    Uses column classification to identify tables with 2+ numeric columns
    and generates correlation analysis insights.

    Pure function — uses profile data only.
    """
    insights: list[Insight] = []

    for profile in profiles:
        cls = getattr(profile, "column_classification", None)
        if not cls or "columns" not in cls:
            continue

        cols = cls["columns"]
        row_count = getattr(profile, "row_count", 0) or 0

        if row_count < 10:
            continue

        # Find numeric columns with stats
        numeric_cols = []
        for col_name, info in cols.items():
            sem_type = info.get("semantic_type", "")
            if sem_type in ("numeric_metric", "numeric_currency", "numeric_percentage"):
                stats = info.get("stats")
                if stats and stats.get("stdev", 0) > 0:
                    numeric_cols.append((col_name, info))

        if len(numeric_cols) < 2:
            continue

        # Use sample values to estimate correlations (if available)
        corr_pairs = _estimate_sample_correlations(numeric_cols)

        if not corr_pairs:
            continue  # No correlations found — don't generate meta-insights that clutter the feed

        # Generate insights for strong correlations only
        h_table = _humanize_table(profile.table_name)
        for col_a, col_b, est_r, direction in corr_pairs:
            h_a = _humanize_col(col_a)
            h_b = _humanize_col(col_b)
            strength = "strong" if abs(est_r) >= 0.7 else "moderate"
            move_verb = "move together" if direction == "positive" else "move in opposite directions"
            when_desc = (
                f"when {h_a} goes up, {h_b} goes up too"
                if direction == "positive"
                else f"when {h_a} goes up, {h_b} tends to go down"
            )
            insights.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="correlation",
                severity="warning" if abs(est_r) >= 0.7 else "info",
                impact_score=55 if abs(est_r) >= 0.85 else (45 if abs(est_r) >= 0.7 else (30 if abs(est_r) >= 0.6 else 25)),
                title=f"{h_a} and {h_b} {move_verb} in {h_table}",
                description=(
                    f"There's a {strength} link between {h_a} and {h_b} — "
                    f"{when_desc}. "
                    f"This {strength} relationship (correlation: {est_r:.2f}) could help "
                    f"predict one from the other."
                ),
                source_tables=[profile.table_name],
                source_columns=[col_a, col_b],
                evidence={
                    "estimated_correlation": round(est_r, 3),
                    "direction": direction,
                    "strength": strength,
                    "query": (
                        f'SELECT "{col_a}", "{col_b}" FROM "{profile.table_name}" '
                        f'ORDER BY ctid LIMIT 500'
                    ),
                    "chart_spec": {
                        "type": "scatter",
                        "x": col_a,
                        "y": col_b,
                        "title": f"{h_a} vs {h_b}",
                    },
                },
                suggested_actions=[
                    f"Investigate whether {h_a} actually drives {h_b} or if something else causes both",
                    f"Use this relationship to predict {h_b} when you know {h_a}",
                ],
            ))

    return insights


def _estimate_sample_correlations(
    numeric_cols: list[tuple[str, dict]],
) -> list[tuple[str, str, float, str]]:
    """Estimate correlations from sample values in column info.

    Returns list of (col_a, col_b, estimated_r, direction).
    Only includes pairs with sufficient data and notable correlation.
    """
    results = []

    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            col_a, info_a = numeric_cols[i]
            col_b, info_b = numeric_cols[j]

            samples_a = info_a.get("sample_values", [])
            samples_b = info_b.get("sample_values", [])

            if len(samples_a) < 5 or len(samples_b) < 5:
                continue

            # Parse to floats while maintaining row alignment:
            # zip the raw samples and only keep rows where BOTH parse to float.
            # This ensures vals_a[i] and vals_b[i] come from the same DB row.
            vals_a: list[float] = []
            vals_b: list[float] = []
            n = min(len(samples_a), len(samples_b))
            for k in range(n):
                sa, sb = samples_a[k], samples_b[k]
                if sa is None or sb is None:
                    continue
                try:
                    va = float(str(sa).replace(",", ""))
                    vb = float(str(sb).replace(",", ""))
                    vals_a.append(va)
                    vals_b.append(vb)
                except (ValueError, TypeError):
                    continue

            if len(vals_a) < 5:
                continue

            r = _quick_pearson(vals_a, vals_b)
            if r is None:
                continue

            if abs(r) >= 0.5:
                direction = "positive" if r > 0 else "negative"
                results.append((col_a, col_b, r, direction))

    return results


def _parse_samples(samples: list) -> list[float]:
    """Parse sample values to floats, skipping non-numeric."""
    result = []
    for s in samples:
        try:
            result.append(float(str(s).replace(",", "")))
        except (ValueError, TypeError):
            pass
    return result


def _quick_pearson(x: list[float], y: list[float]) -> float | None:
    """Quick Pearson correlation for sample data."""
    import math
    import statistics

    n = len(x)
    if n < 3:
        return None

    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)

    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    dx = math.sqrt(sum((xi - x_mean) ** 2 for xi in x))
    dy = math.sqrt(sum((yi - y_mean) ** 2 for yi in y))

    if dx == 0 or dy == 0:
        return None

    return num / (dx * dy)


# ---------------------------------------------------------------------------
# SQL-backed full-data correlation (async, uses CORR() or fetches raw data)
# ---------------------------------------------------------------------------

def _safe(name: str) -> str:
    """Strip non-alphanumeric/underscore chars for SQL safety."""
    return re.sub(r"[^a-zA-Z0-9_]", "", name)


async def discover_correlations_sql(
    session,
    profiles: list,
) -> list[Insight]:
    """SQL-backed correlation — computes Pearson on full data.

    Uses PostgreSQL's CORR() aggregate for each viable numeric column pair,
    giving accurate correlations instead of sample-based estimates.
    """
    from sqlalchemy import text as sql_text

    insights: list[Insight] = []

    for profile in profiles:
        try:
            cls = getattr(profile, "column_classification", None)
            if not cls or "columns" not in cls:
                continue

            cols = cls["columns"]
            row_count = getattr(profile, "row_count", 0) or 0
            if row_count < 10:
                continue

            numeric_cols = []
            for col_name, info in cols.items():
                sem_type = info.get("semantic_type", "")
                if sem_type in ("numeric_metric", "numeric_currency", "numeric_percentage"):
                    stats = info.get("stats")
                    if stats and stats.get("stdev", 0) > 0:
                        numeric_cols.append((col_name, info))

            if len(numeric_cols) < 2:
                continue

            s_table = _safe(profile.table_name)
            h_table = _humanize_table(profile.table_name)

            # Build a single query that computes CORR() for all pairs at once
            # PostgreSQL's CORR(y, x) returns Pearson's r
            corr_selects = []
            pair_map = []  # Track which pair each SELECT corresponds to
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    col_a, info_a = numeric_cols[i]
                    col_b, info_b = numeric_cols[j]
                    s_a = _safe(col_a)
                    s_b = _safe(col_b)
                    alias = f"r_{i}_{j}"
                    corr_selects.append(f'CORR("{s_a}"::float, "{s_b}"::float) AS "{alias}"')
                    pair_map.append((col_a, col_b, info_a, info_b, alias))

            if not corr_selects:
                continue

            # Limit to 20 pairs per table to avoid huge queries
            if len(corr_selects) > 20:
                corr_selects = corr_selects[:20]
                pair_map = pair_map[:20]

            query = f'SELECT {", ".join(corr_selects)} FROM "{s_table}"'
            try:
                result = await session.execute(sql_text(query))
                row = dict(result.fetchone()._mapping)
            except Exception:
                logger.debug("SQL correlation query failed for %s", s_table)
                continue

            for col_a, col_b, info_a, info_b, alias in pair_map:
                r = row.get(alias)
                if r is None:
                    continue
                r = float(r)
                if abs(r) < 0.5:
                    continue

                h_a = _humanize_col(col_a)
                h_b = _humanize_col(col_b)
                direction = "positive" if r > 0 else "negative"
                strength = "strong" if abs(r) >= 0.7 else "moderate"
                move_verb = "move together" if direction == "positive" else "move in opposite directions"
                when_desc = (
                    f"when {h_a} goes up, {h_b} goes up too"
                    if direction == "positive"
                    else f"when {h_a} goes up, {h_b} tends to go down"
                )

                insights.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="correlation",
                    severity="warning" if abs(r) >= 0.7 else "info",
                    impact_score=55 if abs(r) >= 0.85 else (45 if abs(r) >= 0.7 else (30 if abs(r) >= 0.6 else 25)),
                    title=f"{h_a} and {h_b} {move_verb} in {h_table}",
                    description=(
                        f"There's a {strength} link between {h_a} and {h_b} — "
                        f"{when_desc}. "
                        f"Correlation: {r:.2f} (computed on all {row_count:,} rows of data)."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_a, col_b],
                    evidence={
                        "estimated_correlation": round(r, 3),
                        "direction": direction,
                        "strength": strength,
                        "total_rows": row_count,
                        "query": (
                            f'SELECT "{col_a}", "{col_b}" FROM "{profile.table_name}" '
                            f'ORDER BY ctid LIMIT 500'
                        ),
                        "chart_spec": {
                            "type": "scatter",
                            "x": col_a,
                            "y": col_b,
                            "title": f"{h_a} vs {h_b}",
                        },
                    },
                    suggested_actions=[
                        f"Investigate whether {h_a} actually drives {h_b} or if something else causes both",
                        f"Use this relationship to predict {h_b} when you know {h_a}",
                    ],
                ))

        except Exception:
            logger.exception("SQL correlation discovery failed for %s", profile.table_name)

    return insights

