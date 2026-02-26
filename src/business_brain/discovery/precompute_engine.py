"""Pre-compute engine — runs top-N analyses in background after discovery.

After discovery profiles tables, this engine:
1. Scores ALL column combinations using existing recommender scorers
2. Picks the top-N most promising candidates
3. Runs actual SQL queries (GROUP BY, CORR, z-score, regression)
4. Stores results in PrecomputedAnalysis so recommendations can show real data

This is the first component in the system that queries the actual underlying
data for recommendation purposes. All existing discovery modules work from
profile data only (100-row sample stats). The precompute engine bridges the
gap between "we guess this might be interesting" and "we computed it."
"""

from __future__ import annotations

import logging
import re
import statistics
from datetime import datetime, timezone
from itertools import combinations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text as sql_text

from business_brain.db.discovery_models import PrecomputedAnalysis, TableProfile
from business_brain.discovery.insight_recommender import (
    _coefficient_of_variation,
    _column_completeness,
    _is_unnamed_column,
    _rank_columns,
    _score_categorical_for_benchmark,
    _score_numeric_for_anomaly,
    _score_numeric_for_benchmark,
    _score_numeric_for_correlation,
    _score_numeric_for_trend,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safe SQL helpers (same pattern as profiler.py + pattern_memory.py)
# ---------------------------------------------------------------------------

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe(name: str) -> str:
    """Sanitize a table/column name to prevent SQL injection."""
    return _SAFE_NAME_RE.sub("", name)


# ---------------------------------------------------------------------------
# Candidate Generation
# ---------------------------------------------------------------------------


def generate_candidates(
    profiles: list[TableProfile],
    max_per_table: int = 8,
) -> list[dict]:
    """Score ALL column combinations and return top candidates per table.

    Reuses existing scorer functions from insight_recommender.py to rank
    columns, then generates candidate analysis specs for the best combos.

    Returns list sorted by priority_score descending:
        [{"table_name", "analysis_type", "columns", "column_scores",
          "priority_score", "data_hash"}]
    """
    all_candidates: list[dict] = []

    for profile in profiles:
        cls = profile.column_classification
        if not cls or "columns" not in cls:
            continue

        columns: dict[str, dict] = cls["columns"]
        row_count = profile.row_count or 0
        domain = cls.get("domain_hint", "general")

        # Build a profile dict compatible with scorer functions
        prof_dict: dict = {
            "table_name": profile.table_name,
            "row_count": row_count,
            "column_classification": cls,
        }

        # Classify columns by semantic type
        cat_cols = {c: info for c, info in columns.items()
                    if info.get("semantic_type") == "categorical"}
        num_cols = {c: info for c, info in columns.items()
                    if (info.get("semantic_type") or "").startswith("numeric")}
        temp_cols = {c: info for c, info in columns.items()
                     if info.get("semantic_type") == "temporal"}

        table_candidates: list[dict] = []

        # --- Benchmark candidates: top-3 cat × top-3 num → top-3 combos ---
        if cat_cols and num_cols:
            ranked_cats = _rank_columns(cat_cols, row_count, prof_dict,
                                        _score_categorical_for_benchmark)[:3]
            ranked_nums = _rank_columns(num_cols, row_count, prof_dict,
                                        _score_numeric_for_benchmark)[:3]
            bench_combos = []
            for cat in ranked_cats:
                cat_score = _score_categorical_for_benchmark(
                    cat, cat_cols[cat], row_count, prof_dict)
                for num in ranked_nums:
                    num_score = _score_numeric_for_benchmark(
                        num, num_cols[num], row_count, prof_dict)
                    combined = (cat_score + num_score) / 2
                    bench_combos.append((cat, num, cat_score, num_score, combined))
            bench_combos.sort(key=lambda x: -x[4])
            for cat, num, cs, ns, combo_score in bench_combos[:3]:
                table_candidates.append({
                    "table_name": profile.table_name,
                    "analysis_type": "benchmark",
                    "columns": [cat, num],
                    "column_scores": {cat: round(cs, 3), num: round(ns, 3)},
                    "priority_score": round(combo_score, 3),
                    "data_hash": profile.data_hash,
                })

        # --- Correlation candidates: top-5 numeric → top-3 pairs ---
        if len(num_cols) >= 2:
            ranked_corr = _rank_columns(num_cols, row_count, prof_dict,
                                        _score_numeric_for_correlation)[:5]
            corr_pairs = []
            for a, b in combinations(ranked_corr, 2):
                sa = _score_numeric_for_correlation(
                    a, num_cols[a], row_count, prof_dict)
                sb = _score_numeric_for_correlation(
                    b, num_cols[b], row_count, prof_dict)
                combined = (sa + sb) / 2
                corr_pairs.append((a, b, sa, sb, combined))
            corr_pairs.sort(key=lambda x: -x[4])
            for a, b, sa, sb, combo_score in corr_pairs[:3]:
                table_candidates.append({
                    "table_name": profile.table_name,
                    "analysis_type": "correlation",
                    "columns": [a, b],
                    "column_scores": {a: round(sa, 3), b: round(sb, 3)},
                    "priority_score": round(combo_score, 3),
                    "data_hash": profile.data_hash,
                })

        # --- Anomaly candidates: top-2 numeric ---
        if num_cols:
            ranked_anom = _rank_columns(num_cols, row_count, prof_dict,
                                        _score_numeric_for_anomaly)[:2]
            for col in ranked_anom:
                sc = _score_numeric_for_anomaly(
                    col, num_cols[col], row_count, prof_dict)
                if sc > 0:
                    table_candidates.append({
                        "table_name": profile.table_name,
                        "analysis_type": "anomaly",
                        "columns": [col],
                        "column_scores": {col: round(sc, 3)},
                        "priority_score": round(sc, 3),
                        "data_hash": profile.data_hash,
                    })

        # --- Trend candidates: temporal + top-2 numeric ---
        if temp_cols and num_cols:
            time_col = list(temp_cols.keys())[0]  # Use first temporal column
            ranked_trend = _rank_columns(
                num_cols, row_count, prof_dict,
                _score_numeric_for_trend, domain=domain)[:2]
            for col in ranked_trend:
                sc = _score_numeric_for_trend(
                    col, num_cols[col], row_count, prof_dict, domain=domain)
                if sc > 0:
                    table_candidates.append({
                        "table_name": profile.table_name,
                        "analysis_type": "trend",
                        "columns": [time_col, col],
                        "column_scores": {col: round(sc, 3)},
                        "priority_score": round(sc, 3),
                        "data_hash": profile.data_hash,
                    })

        # Cap per table
        table_candidates.sort(key=lambda x: -x["priority_score"])
        all_candidates.extend(table_candidates[:max_per_table])

    # Sort globally by priority_score
    all_candidates.sort(key=lambda x: -x["priority_score"])
    return all_candidates


# ---------------------------------------------------------------------------
# SQL Pre-computation Functions
# ---------------------------------------------------------------------------


async def _precompute_benchmark(
    session: AsyncSession,
    table_name: str,
    cat_col: str,
    num_col: str,
) -> tuple[dict, dict, float]:
    """Run GROUP BY benchmark and return (summary, detail, quality_score).

    SQL: SELECT cat, COUNT(*), AVG(num), STDDEV(num)
         FROM table GROUP BY cat ORDER BY AVG(num) DESC
    """
    safe_table = _safe(table_name)
    safe_cat = _safe(cat_col)
    safe_num = _safe(num_col)

    query = (
        f'SELECT "{safe_cat}", COUNT(*) AS cnt, '
        f'AVG(CAST("{safe_num}" AS DOUBLE PRECISION)) AS avg_val '
        f'FROM "{safe_table}" '
        f'WHERE "{safe_num}" IS NOT NULL AND "{safe_cat}" IS NOT NULL '
        f'GROUP BY "{safe_cat}" ORDER BY avg_val DESC'
    )
    result = await session.execute(sql_text(query))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {}, {}, 0.0

    group_count = len(rows)
    top_group = str(rows[0][safe_cat])
    top_value = float(rows[0]["avg_val"]) if rows[0]["avg_val"] is not None else 0.0
    bottom_group = str(rows[-1][safe_cat])
    bottom_value = float(rows[-1]["avg_val"]) if rows[-1]["avg_val"] is not None else 0.0

    # Spread: percentage difference between top and bottom
    denominator = bottom_value if bottom_value != 0 else 1.0
    spread_pct = round(abs(top_value - bottom_value) / abs(denominator) * 100, 1)

    summary = {
        "group_count": group_count,
        "top_group": top_group,
        "top_value": round(top_value, 2),
        "bottom_group": bottom_group,
        "bottom_value": round(bottom_value, 2),
        "spread_pct": spread_pct,
    }

    detail = {
        "query": query,
        "groups": [
            {"group": str(r[safe_cat]), "count": int(r["cnt"]),
             "avg": round(float(r["avg_val"]), 2) if r["avg_val"] is not None else None}
            for r in rows[:20]  # Cap detail to top 20 groups
        ],
    }

    # Quality: high spread + reasonable group count = interesting
    quality = 0.0
    if spread_pct > 20:
        quality += 0.4
    elif spread_pct > 10:
        quality += 0.25
    elif spread_pct > 5:
        quality += 0.1

    if 3 <= group_count <= 50:
        quality += 0.3
    elif group_count == 2:
        quality += 0.15

    if group_count >= 3 and spread_pct > 5:
        quality += 0.3  # Bonus for actionable benchmarks

    return summary, detail, min(quality, 1.0)


async def _precompute_correlation(
    session: AsyncSession,
    table_name: str,
    col_a: str,
    col_b: str,
) -> tuple[dict, dict, float]:
    """Run CORR() and return (summary, detail, quality_score).

    SQL: SELECT CORR(a, b), COUNT(*)
         FROM table WHERE a IS NOT NULL AND b IS NOT NULL
    """
    safe_table = _safe(table_name)
    safe_a = _safe(col_a)
    safe_b = _safe(col_b)

    query = (
        f'SELECT CORR(CAST("{safe_a}" AS DOUBLE PRECISION), '
        f'CAST("{safe_b}" AS DOUBLE PRECISION)) AS r_val, '
        f'COUNT(*) AS sample_size '
        f'FROM "{safe_table}" '
        f'WHERE "{safe_a}" IS NOT NULL AND "{safe_b}" IS NOT NULL'
    )
    result = await session.execute(sql_text(query))
    row = result.fetchone()

    if not row or row.r_val is None:
        return {}, {}, 0.0

    r_value = round(float(row.r_val), 4)
    sample_size = int(row.sample_size)
    abs_r = abs(r_value)

    direction = "positive" if r_value > 0 else "negative" if r_value < 0 else "none"

    # Rough significance approximation
    if sample_size >= 30 and abs_r > 0.35:
        p_approx = "significant"
    elif sample_size >= 100 and abs_r > 0.2:
        p_approx = "significant"
    else:
        p_approx = "not_significant"

    summary = {
        "r_value": r_value,
        "direction": direction,
        "sample_size": sample_size,
        "p_approx": p_approx,
    }

    # Also fetch a scatter sample for visualization
    scatter_query = (
        f'SELECT CAST("{safe_a}" AS DOUBLE PRECISION) AS x, '
        f'CAST("{safe_b}" AS DOUBLE PRECISION) AS y '
        f'FROM "{safe_table}" '
        f'WHERE "{safe_a}" IS NOT NULL AND "{safe_b}" IS NOT NULL '
        f'ORDER BY RANDOM() LIMIT 100'
    )
    try:
        scatter_result = await session.execute(sql_text(scatter_query))
        scatter_rows = [{"x": round(float(r.x), 2), "y": round(float(r.y), 2)}
                        for r in scatter_result.fetchall()]
    except Exception:
        scatter_rows = []

    detail = {
        "query": query,
        "scatter_sample": scatter_rows,
    }

    # Quality: stronger correlation = more interesting
    quality = 0.0
    if abs_r >= 0.8:
        quality = 0.95
    elif abs_r >= 0.7:
        quality = 0.8
    elif abs_r >= 0.5:
        quality = 0.6
    elif abs_r >= 0.3:
        quality = 0.3
    else:
        quality = 0.1

    if p_approx == "not_significant":
        quality *= 0.5

    return summary, detail, min(quality, 1.0)


async def _precompute_anomaly(
    session: AsyncSession,
    table_name: str,
    num_col: str,
) -> tuple[dict, dict, float]:
    """Run z-score anomaly detection and return (summary, detail, quality_score).

    Fetches mean + stdev from DB, then identifies rows > 3 stdev from mean.
    """
    safe_table = _safe(table_name)
    safe_col = _safe(num_col)

    # Get stats
    stats_query = (
        f'SELECT AVG(CAST("{safe_col}" AS DOUBLE PRECISION)) AS mean_val, '
        f'STDDEV(CAST("{safe_col}" AS DOUBLE PRECISION)) AS stdev_val, '
        f'COUNT(*) AS cnt '
        f'FROM "{safe_table}" WHERE "{safe_col}" IS NOT NULL'
    )
    result = await session.execute(sql_text(stats_query))
    stats_row = result.fetchone()

    if not stats_row or stats_row.mean_val is None or stats_row.stdev_val is None:
        return {}, {}, 0.0

    mean_val = float(stats_row.mean_val)
    stdev_val = float(stats_row.stdev_val)
    total_count = int(stats_row.cnt)

    if stdev_val == 0 or total_count < 10:
        return {}, {}, 0.0

    # Find outliers (|z| > 3)
    z_threshold = 3.0
    lower_bound = mean_val - z_threshold * stdev_val
    upper_bound = mean_val + z_threshold * stdev_val

    outlier_query = (
        f'SELECT CAST("{safe_col}" AS DOUBLE PRECISION) AS val '
        f'FROM "{safe_table}" '
        f'WHERE "{safe_col}" IS NOT NULL '
        f'AND (CAST("{safe_col}" AS DOUBLE PRECISION) < {lower_bound} '
        f'     OR CAST("{safe_col}" AS DOUBLE PRECISION) > {upper_bound}) '
        f'ORDER BY ABS(CAST("{safe_col}" AS DOUBLE PRECISION) - {mean_val}) DESC '
        f'LIMIT 20'
    )
    outlier_result = await session.execute(sql_text(outlier_query))
    outlier_rows = [float(r.val) for r in outlier_result.fetchall()]

    outlier_count = len(outlier_rows)
    max_z_score = 0.0
    most_anomalous = None
    if outlier_rows:
        most_anomalous = outlier_rows[0]
        max_z_score = round(abs(most_anomalous - mean_val) / stdev_val, 2)

    summary = {
        "outlier_count": outlier_count,
        "max_z_score": max_z_score,
        "most_anomalous_value": round(most_anomalous, 2) if most_anomalous is not None else None,
        "mean": round(mean_val, 2),
        "stdev": round(stdev_val, 2),
        "z_threshold": z_threshold,
        "total_rows": total_count,
    }

    detail = {
        "query": stats_query,
        "outlier_values": [round(v, 2) for v in outlier_rows],
        "bounds": {"lower": round(lower_bound, 2), "upper": round(upper_bound, 2)},
    }

    # Quality: more outliers = more interesting
    quality = 0.0
    if outlier_count >= 5:
        quality = 0.9
    elif outlier_count >= 3:
        quality = 0.7
    elif outlier_count >= 1:
        quality = 0.5
    else:
        quality = 0.1

    if max_z_score >= 5:
        quality = min(1.0, quality + 0.1)

    return summary, detail, min(quality, 1.0)


async def _precompute_trend(
    session: AsyncSession,
    table_name: str,
    time_col: str,
    num_col: str,
) -> tuple[dict, dict, float]:
    """Run time aggregation + linear regression and return (summary, detail, quality_score).

    Groups by time column, computes AVG(num), then fits simple OLS in Python.
    """
    safe_table = _safe(table_name)
    safe_time = _safe(time_col)
    safe_num = _safe(num_col)

    query = (
        f'SELECT "{safe_time}" AS period, '
        f'AVG(CAST("{safe_num}" AS DOUBLE PRECISION)) AS avg_val, '
        f'COUNT(*) AS cnt '
        f'FROM "{safe_table}" '
        f'WHERE "{safe_num}" IS NOT NULL AND "{safe_time}" IS NOT NULL '
        f'GROUP BY "{safe_time}" ORDER BY "{safe_time}"'
    )
    result = await session.execute(sql_text(query))
    rows = result.fetchall()

    if len(rows) < 3:
        return {}, {}, 0.0

    # Extract values for regression
    y_values = [float(r.avg_val) for r in rows if r.avg_val is not None]
    periods = len(y_values)

    if periods < 3:
        return {}, {}, 0.0

    x_values = list(range(periods))

    # Simple OLS: y = slope * x + intercept
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        return {}, {}, 0.0

    slope = numerator / denominator

    # R-squared
    ss_res = sum((y - (slope * x + (y_mean - slope * x_mean))) ** 2
                 for x, y in zip(x_values, y_values))
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Slope as percentage of mean per period
    slope_pct = (slope / abs(y_mean) * 100) if y_mean != 0 else 0.0

    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"

    summary = {
        "direction": direction,
        "slope_pct_per_period": round(slope_pct, 2),
        "r_squared": round(r_squared, 4),
        "periods": periods,
    }

    detail = {
        "query": query,
        "time_series": [
            {"period": str(r.period), "avg_value": round(float(r.avg_val), 2),
             "count": int(r.cnt)}
            for r in rows[:50]  # Cap at 50 points
        ],
    }

    # Quality: strong trend = more interesting
    quality = 0.0
    if r_squared >= 0.7:
        quality += 0.5
    elif r_squared >= 0.4:
        quality += 0.3
    elif r_squared >= 0.2:
        quality += 0.15

    if abs(slope_pct) >= 5:
        quality += 0.3
    elif abs(slope_pct) >= 2:
        quality += 0.2
    elif abs(slope_pct) >= 0.5:
        quality += 0.1

    if periods >= 10:
        quality += 0.2
    elif periods >= 5:
        quality += 0.1

    return summary, detail, min(quality, 1.0)


# ---------------------------------------------------------------------------
# Dispatcher — maps analysis_type to SQL function
# ---------------------------------------------------------------------------

_PRECOMPUTE_DISPATCH = {
    "benchmark": lambda s, t, cols: _precompute_benchmark(s, t, cols[0], cols[1]),
    "correlation": lambda s, t, cols: _precompute_correlation(s, t, cols[0], cols[1]),
    "anomaly": lambda s, t, cols: _precompute_anomaly(s, t, cols[0]),
    "trend": lambda s, t, cols: _precompute_trend(s, t, cols[0], cols[1]),
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


async def run_precomputation(
    session: AsyncSession,
    profiles: list[TableProfile],
    max_total: int = 20,
    max_per_table: int = 8,
) -> list[PrecomputedAnalysis]:
    """Main entry point: generate candidates, run SQL, store results.

    1. Generate ranked candidates from all tables
    2. Truncate to max_total
    3. Run SQL for each candidate (tolerant of individual failures)
    4. Store PrecomputedAnalysis records
    5. Return completed analyses
    """
    candidates = generate_candidates(profiles, max_per_table=max_per_table)
    candidates = candidates[:max_total]

    if not candidates:
        return []

    completed: list[PrecomputedAnalysis] = []

    for candidate in candidates:
        table_name = candidate["table_name"]
        analysis_type = candidate["analysis_type"]
        columns = candidate["columns"]

        # Check if we already have a non-stale result for this exact analysis
        existing = (await session.execute(
            select(PrecomputedAnalysis).where(
                PrecomputedAnalysis.table_name == table_name,
                PrecomputedAnalysis.analysis_type == analysis_type,
                PrecomputedAnalysis.columns == columns,
                PrecomputedAnalysis.status.in_(["completed", "running"]),
                PrecomputedAnalysis.data_hash == candidate.get("data_hash"),
            )
        )).scalar_one_or_none()

        if existing:
            completed.append(existing)
            continue

        # Create record as pending
        record = PrecomputedAnalysis(
            table_name=table_name,
            analysis_type=analysis_type,
            columns=columns,
            column_scores=candidate.get("column_scores"),
            status="running",
            discovery_run_id=None,
            data_hash=candidate.get("data_hash"),
        )
        session.add(record)
        await session.flush()

        # Run the actual SQL
        dispatch_fn = _PRECOMPUTE_DISPATCH.get(analysis_type)
        if not dispatch_fn:
            record.status = "failed"
            record.error = f"Unknown analysis type: {analysis_type}"
            continue

        try:
            summary, detail, quality = await dispatch_fn(session, table_name, columns)
            record.status = "completed"
            record.result_summary = summary
            record.result_detail = detail
            record.quality_score = quality
            record.computed_at = datetime.now(timezone.utc)
            completed.append(record)
            logger.debug(
                "Pre-computed %s for %s.%s (quality=%.2f)",
                analysis_type, table_name, columns, quality,
            )
        except Exception as exc:
            record.status = "failed"
            record.error = str(exc)[:500]
            logger.warning(
                "Pre-computation failed: %s for %s.%s: %s",
                analysis_type, table_name, columns, exc,
            )

    await session.flush()
    logger.info("Pre-computation complete: %d/%d analyses succeeded",
                len([r for r in completed if r.status == "completed"]),
                len(candidates))
    return completed


# ---------------------------------------------------------------------------
# Staleness Management
# ---------------------------------------------------------------------------


async def invalidate_stale(
    session: AsyncSession,
    changed_tables: set[str],
) -> int:
    """Mark pre-computed results as 'stale' for tables whose data changed.

    Called at the START of discovery, before profiling overwrites data_hash.
    Stale results are excluded from recommendation enrichment.
    """
    if not changed_tables:
        return 0

    result = await session.execute(
        update(PrecomputedAnalysis)
        .where(
            PrecomputedAnalysis.table_name.in_(list(changed_tables)),
            PrecomputedAnalysis.status == "completed",
        )
        .values(status="stale")
    )
    count = result.rowcount
    if count:
        logger.info("Marked %d pre-computed analyses as stale", count)
    return count


async def cleanup_old(
    session: AsyncSession,
    keep_days: int = 7,
) -> int:
    """Delete pre-computed results older than keep_days."""
    from datetime import timedelta

    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    result = await session.execute(
        select(PrecomputedAnalysis).where(
            PrecomputedAnalysis.computed_at < cutoff,
        )
    )
    old_records = list(result.scalars().all())
    for record in old_records:
        await session.delete(record)
    if old_records:
        await session.flush()
        logger.info("Cleaned up %d old pre-computed analyses", len(old_records))
    return len(old_records)
