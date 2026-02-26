"""Table analysis router — domain-specific analytics endpoints.

Extracted from api.py: ~120 routes covering statistical analysis,
manufacturing metrics, supply chain, financial analysis, quality,
HR, logistics, and more.

All endpoints follow a common pattern:
1. Load TableProfile for the given table
2. Run domain-specific analysis
3. Return structured JSON results
"""

from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

router = APIRouter(tags=["table-analysis"])


@router.get("/tables/{table_name}/columns")
async def get_table_columns(
    table_name: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get column-level stats for a profiled table."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile

    result = await session.execute(
        select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"Table '{table_name}' not profiled"})

    cls = profile.column_classification or {}
    columns_info = cls.get("columns", {})

    columns = []
    for col_name, info in columns_info.items():
        columns.append({
            "name": col_name,
            "semantic_type": info.get("semantic_type"),
            "cardinality": info.get("cardinality"),
            "sample_values": info.get("sample_values", [])[:5],
            **{k: v for k, v in info.items() if k not in ("semantic_type", "cardinality", "sample_values")},
        })

    return {
        "table_name": table_name,
        "row_count": profile.row_count,
        "domain_hint": profile.domain_hint or cls.get("domain_hint"),
        "profiled_at": profile.profiled_at.isoformat() if profile.profiled_at else None,
        "column_count": len(columns),
        "columns": columns,
    }


@router.post("/tables/{table_name}/time-analysis")
async def time_analysis(
    table_name: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Run time intelligence analysis on a table's numeric series data.

    Returns trend direction, period-over-period changes, and changepoints
    for each numeric column that has a temporal companion.
    """
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.time_intelligence import (
        compute_period_change,
        detect_changepoints,
        detect_trend,
        find_min_max_periods,
    )

    result = await session.execute(
        select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"error": f"Table '{table_name}' not profiled"})

    cls = profile.column_classification or {}
    columns = cls.get("columns", {})

    temp_cols = [c for c, i in columns.items() if i.get("semantic_type") == "temporal"]
    num_cols = [
        c for c, i in columns.items()
        if i.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ]

    if not temp_cols or not num_cols:
        return {"table_name": table_name, "analysis": [], "message": "No temporal+numeric column pairs found"}

    # For each numeric column, compute trend from sample values
    analyses = []
    for col_name in num_cols[:5]:
        info = columns.get(col_name, {})
        samples = info.get("sample_values", [])

        # Convert to floats
        values = []
        for s in samples:
            try:
                values.append(float(str(s).replace(",", "")))
            except (ValueError, TypeError):
                pass

        if len(values) < 2:
            continue

        trend = detect_trend(values)
        min_max = find_min_max_periods(values)
        changepoints = detect_changepoints(values)

        analysis = {
            "column": col_name,
            "sample_count": len(values),
            "trend": {
                "direction": trend.direction,
                "magnitude_pct_per_period": trend.magnitude,
                "r_squared": trend.r_squared,
            },
            "changepoints": changepoints,
        }

        if min_max:
            analysis["min_max"] = {
                "max_value": min_max.max_value,
                "max_index": min_max.max_index,
                "min_value": min_max.min_value,
                "min_index": min_max.min_index,
            }

        if len(values) >= 2:
            pop = compute_period_change(values[-1], values[-2])
            analysis["latest_period_change"] = {
                "current": pop.current,
                "previous": pop.previous,
                "absolute_change": pop.absolute_change,
                "pct_change": pop.pct_change,
            }

        analyses.append(analysis)

    return {
        "table_name": table_name,
        "temporal_columns": temp_cols,
        "analysis": analyses,
    }


@router.post("/tables/{table_name}/forecast")
async def forecast_table(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Forecast numeric columns using linear and exponential smoothing."""
    from sqlalchemy import select as sa_select, text as sql_text
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.time_intelligence import forecast_exponential, forecast_linear

    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile or not profile.column_classification:
        return {"error": f"No profile for table '{table_name}'"}

    cls = profile.column_classification
    if "columns" not in cls:
        return {"error": "No column data"}

    cols = cls["columns"]
    num_cols = [
        c for c, info in cols.items()
        if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ][:5]

    if not num_cols:
        return {"error": "No numeric columns found"}

    forecasts = []
    for col_name in num_cols:
        try:
            query = f'SELECT "{col_name}" FROM "{table_name}" ORDER BY ctid LIMIT 200'
            rows = await session.execute(sql_text(query))
            values = []
            for row in rows.fetchall():
                try:
                    values.append(float(str(row[0]).replace(",", "")))
                except (ValueError, TypeError):
                    pass

            if len(values) < 3:
                continue

            linear = forecast_linear(values, 5)
            exponential = forecast_exponential(values, 5)

            forecasts.append({
                "column": col_name,
                "data_points": len(values),
                "linear": {
                    "predicted": linear.predicted_values,
                    "confidence": linear.confidence,
                },
                "exponential": {
                    "predicted": exponential.predicted_values,
                    "confidence": exponential.confidence,
                },
            })
        except Exception:
            logger.exception("Forecast failed for %s.%s", table_name, col_name)

    return {"table_name": table_name, "forecasts": forecasts}


@router.post("/tables/{table_name}/correlations")
async def compute_correlations(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Compute pairwise correlations between numeric columns in a table."""
    from sqlalchemy import select as sa_select, text as sql_text
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.correlation_engine import (
        compute_correlation_matrix,
        correlation_summary,
        find_strong_correlations,
        find_surprising_correlations,
    )

    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile or not profile.column_classification:
        return {"error": f"No profile for table '{table_name}'"}

    cls = profile.column_classification
    if "columns" not in cls:
        return {"error": "No column data"}

    cols = cls["columns"]
    num_cols = [
        c for c, info in cols.items()
        if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ][:10]

    if len(num_cols) < 2:
        return {"error": "Need at least 2 numeric columns for correlation analysis"}

    # Fetch data for all numeric columns
    col_list = ", ".join(f'"{c}"' for c in num_cols)
    try:
        rows = await session.execute(sql_text(f'SELECT {col_list} FROM "{table_name}" LIMIT 500'))
        all_rows = rows.fetchall()
    except Exception:
        return {"error": "Failed to query table data"}

    data: dict[str, list[float]] = {c: [] for c in num_cols}
    for row in all_rows:
        for i, col in enumerate(num_cols):
            try:
                data[col].append(float(str(row[i]).replace(",", "")))
            except (ValueError, TypeError):
                data[col].append(float("nan"))

    # Remove nan rows (paired deletion)
    clean_data: dict[str, list[float]] = {c: [] for c in num_cols}
    for idx in range(len(all_rows)):
        if all(not (data[c][idx] != data[c][idx]) for c in num_cols):  # nan check
            for c in num_cols:
                clean_data[c].append(data[c][idx])

    pairs = compute_correlation_matrix(clean_data)
    strong = find_strong_correlations(pairs)
    surprising = find_surprising_correlations(pairs)
    summary = correlation_summary(pairs)

    return {
        "table_name": table_name,
        "columns_analyzed": num_cols,
        "summary": summary,
        "strong_correlations": [
            {"columns": [p.column_a, p.column_b], "correlation": p.correlation, "strength": p.strength, "direction": p.direction}
            for p in strong
        ],
        "surprising_correlations": [
            {"columns": [p.column_a, p.column_b], "correlation": p.correlation}
            for p in surprising
        ],
        "all_pairs": [
            {"columns": [p.column_a, p.column_b], "correlation": p.correlation, "strength": p.strength, "direction": p.direction, "sample_size": p.sample_size}
            for p in pairs
        ],
    }


@router.post("/tables/{table_name}/distribution")
async def get_distribution(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get distribution profile for numeric columns in a table."""
    from sqlalchemy import select as sa_select, text as sql_text
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.distribution_profiler import profile_distribution

    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()
    if not profile or not profile.column_classification:
        return {"error": f"No profile for table '{table_name}'"}

    cls = profile.column_classification
    if "columns" not in cls:
        return {"error": "No column data"}

    cols = cls["columns"]
    num_cols = [
        c for c, info in cols.items()
        if info.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ][:8]

    if not num_cols:
        return {"error": "No numeric columns found"}

    distributions = []
    for col_name in num_cols:
        try:
            query = f'SELECT "{col_name}" FROM "{table_name}" LIMIT 500'
            rows = await session.execute(sql_text(query))
            values = []
            for row in rows.fetchall():
                try:
                    values.append(float(str(row[0]).replace(",", "")))
                except (ValueError, TypeError):
                    pass

            dp = profile_distribution(values)
            if dp:
                distributions.append({
                    "column": col_name,
                    "count": dp.count,
                    "mean": dp.mean,
                    "median": dp.median,
                    "stdev": dp.stdev,
                    "min": dp.min_val,
                    "max": dp.max_val,
                    "q1": dp.q1,
                    "q3": dp.q3,
                    "iqr": dp.iqr,
                    "skewness": dp.skewness,
                    "kurtosis": dp.kurtosis,
                    "shape": dp.shape,
                    "histogram": dp.histogram,
                })
        except Exception:
            logger.exception("Distribution profiling failed for %s.%s", table_name, col_name)

    return {"table_name": table_name, "distributions": distributions}


@router.get("/dashboard/summary")
async def get_dashboard_summary(session: AsyncSession = Depends(get_session)) -> dict:
    """Get high-level dashboard KPIs."""
    from sqlalchemy import select as sa_select
    from business_brain.db.discovery_models import (
        DeployedReport,
        DiscoveryRun,
        Insight,
        TableProfile,
    )
    from business_brain.discovery.dashboard_summary import compute_dashboard_summary

    profiles = list((await session.execute(sa_select(TableProfile))).scalars().all())
    insights = list((await session.execute(sa_select(Insight))).scalars().all())
    reports = list((await session.execute(sa_select(DeployedReport))).scalars().all())
    runs = list((await session.execute(sa_select(DiscoveryRun))).scalars().all())

    summary = compute_dashboard_summary(profiles, insights, reports, runs)

    return {
        "total_tables": summary.total_tables,
        "total_rows": summary.total_rows,
        "total_columns": summary.total_columns,
        "total_insights": summary.total_insights,
        "total_reports": summary.total_reports,
        "avg_quality_score": summary.avg_quality_score,
        "data_freshness_pct": summary.data_freshness_pct,
        "insight_breakdown": summary.insight_breakdown,
        "severity_breakdown": summary.severity_breakdown,
        "top_tables": summary.top_tables,
        "last_discovery_at": summary.last_discovery_at,
    }


@router.get("/data-freshness")
async def get_data_freshness(session: AsyncSession = Depends(get_session)) -> dict:
    """Get data freshness scores comparing current vs previous profiles."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import TableProfile
    from business_brain.discovery.data_freshness import compute_freshness_score, detect_stale_tables

    result = await session.execute(select(TableProfile))
    profiles = list(result.scalars().all())

    if not profiles:
        return {"score": 100, "stale_count": 0, "fresh_count": 0, "unknown_count": 0, "total_tables": 0, "stale_tables": []}

    # Use profiles as both current and previous (since we only have one snapshot)
    # In production, you'd compare against a saved snapshot from the previous run
    freshness = compute_freshness_score(profiles, profiles)
    stale_insights = detect_stale_tables(profiles, profiles)

    return {
        **freshness,
        "stale_tables": [i.source_tables[0] for i in stale_insights],
    }


@router.get("/relationships")
async def get_relationships(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """Get all discovered cross-table relationships."""
    from sqlalchemy import select
    from business_brain.db.discovery_models import DiscoveredRelationship

    result = await session.execute(
        select(DiscoveredRelationship).order_by(DiscoveredRelationship.confidence.desc())
    )
    rels = list(result.scalars().all())

    return [
        {
            "id": r.id,
            "table_a": r.table_a,
            "column_a": r.column_a,
            "table_b": r.table_b,
            "column_b": r.column_b,
            "relationship_type": r.relationship_type,
            "confidence": round(r.confidence, 2) if r.confidence else 0,
            "overlap_count": r.overlap_count,
            "discovered_at": r.discovered_at.isoformat() if r.discovered_at else None,
        }
        for r in rels
    ]


@router.get("/lineage/{table_name}")
async def get_lineage(table_name: str, session: AsyncSession = Depends(get_session)) -> dict:
    """Get data lineage for a specific table — what depends on it."""
    from business_brain.discovery.lineage_tracker import get_lineage_for_table

    return await get_lineage_for_table(session, table_name)


@router.get("/lineage")
async def get_full_lineage(session: AsyncSession = Depends(get_session)) -> dict:
    """Get full lineage graph with impact rankings and orphaned tables."""
    from sqlalchemy import select as sa_select
    from business_brain.db.discovery_models import (
        DeployedReport,
        DiscoveredRelationship,
        Insight,
        TableProfile,
    )
    from business_brain.discovery.lineage_tracker import (
        build_lineage_graph,
        find_orphaned_tables,
        get_impact_ranking,
    )

    profiles = list((await session.execute(sa_select(TableProfile))).scalars().all())
    relationships = list((await session.execute(sa_select(DiscoveredRelationship))).scalars().all())
    insights = list((await session.execute(sa_select(Insight))).scalars().all())
    reports = list((await session.execute(sa_select(DeployedReport))).scalars().all())

    graph = build_lineage_graph(profiles, relationships, insights, reports)

    return {
        "impact_ranking": get_impact_ranking(graph),
        "orphaned_tables": find_orphaned_tables(graph),
        "table_count": len(graph["tables"]),
        "insight_count": len(graph["insights"]),
        "report_count": len(graph["reports"]),
        "edge_count": len(graph["edges"]),
    }


# ---------------------------------------------------------------------------
# Data Comparison
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/compare")
async def compare_table_data(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare two snapshots of a table to find changes.

    Body: {"key_columns": ["id"], "old_rows": [...], "new_rows": [...]}
    If old_rows/new_rows not provided, compares current data vs previous profile.
    """
    from business_brain.discovery.data_comparator import (
        classify_change,
        compare_snapshots,
        compute_change_rate,
    )

    key_columns = body.get("key_columns", [])

    if "old_rows" in body and "new_rows" in body:
        old_rows = body["old_rows"]
        new_rows = body["new_rows"]
    else:
        # Compare current table data with itself (useful with offset/limit)
        meta_store = MetadataStore(session)
        tables = await meta_store.list_tables()
        if table_name not in tables:
            return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)
        result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 500'))
        rows = [dict(r._mapping) for r in result.fetchall()]
        return {"message": "Provide old_rows and new_rows in request body", "current_row_count": len(rows)}

    diff = compare_snapshots(old_rows, new_rows, key_columns, table_name)
    change_type = classify_change(diff)
    rate = compute_change_rate(diff)

    return {
        "table_name": diff.table_name,
        "added_rows": diff.added_rows,
        "removed_rows": diff.removed_rows,
        "changed_rows": diff.changed_rows,
        "unchanged_rows": diff.unchanged_rows,
        "total_old": diff.total_old,
        "total_new": diff.total_new,
        "column_changes": diff.column_changes,
        "change_rate": round(rate, 1),
        "change_type": change_type,
        "summary": diff.summary,
        "sample_additions": diff.sample_additions[:5],
        "sample_removals": diff.sample_removals[:5],
        "sample_changes": [
            {
                "key": sc.key_values,
                "changes": [{"column": c.column, "old": c.old_value, "new": c.new_value} for c in sc.changes],
            }
            for sc in diff.sample_changes[:5]
        ],
    }


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


@router.post("/goals/evaluate")
async def evaluate_goals_endpoint(body: dict):
    """Evaluate metric goals against current values.

    Body: {
        "goals": [{"metric_name": "revenue", "target_value": 1000, "direction": "above", "baseline": 0}],
        "current_values": {"revenue": 800}
    }
    """
    from business_brain.discovery.goal_tracker import (
        Goal,
        compute_overall_health,
        evaluate_goals,
    )

    raw_goals = body.get("goals", [])
    current_values = body.get("current_values", {})

    goals = [
        Goal(
            metric_name=g["metric_name"],
            target_value=g["target_value"],
            direction=g.get("direction", "above"),
            target_min=g.get("target_min"),
            baseline=g.get("baseline"),
            deadline=g.get("deadline"),
        )
        for g in raw_goals
    ]

    progress = evaluate_goals(goals, current_values)
    health = compute_overall_health(progress)

    return {
        "goals": [
            {
                "metric_name": p.metric_name,
                "current_value": p.current_value,
                "target_value": p.target_value,
                "direction": p.direction,
                "progress_pct": p.progress_pct,
                "status": p.status,
                "remaining": p.remaining,
                "summary": p.summary,
            }
            for p in progress
        ],
        "health": health,
    }


# ---------------------------------------------------------------------------
# Anomaly Classification
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/classify-anomalies")
async def classify_anomalies_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Classify anomalies in a numeric column.

    Body: {"column": "amount", "threshold_std": 2.0}
    """
    from business_brain.discovery.anomaly_classifier import (
        classify_series,
        compute_anomaly_score,
        summarize_anomalies,
    )

    column = body.get("column")
    threshold = body.get("threshold_std", 2.0)

    if not column:
        return JSONResponse({"error": "column is required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 1000'))
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < 5:
        return {"classifications": [], "summary": {"total": 0, "summary": "Not enough numeric data."}}

    classifications = classify_series(values, threshold_std=threshold)
    summary = summarize_anomalies(classifications)

    return {
        "table": table_name,
        "column": column,
        "total_values": len(values),
        "classifications": [
            {
                "pattern": c.pattern,
                "confidence": c.confidence,
                "description": c.description,
                "severity": c.severity,
                "affected_indices": c.affected_indices[:20],
                "score": compute_anomaly_score(c),
            }
            for c in classifications
        ],
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Profile Report
# ---------------------------------------------------------------------------


@router.get("/tables/{table_name}/profile-report")
async def get_profile_report(
    table_name: str,
    session: AsyncSession = Depends(get_session),
):
    """Generate a comprehensive profile report for a table."""
    from sqlalchemy import select as sa_select
    from business_brain.discovery.profile_report import (
        compute_report_priority,
        format_report_text,
        generate_profile_report,
    )

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    # Get profile from DB
    result = await session.execute(
        sa_select(TableProfile).where(TableProfile.table_name == table_name)
    )
    profile = result.scalar_one_or_none()

    columns = {}
    row_count = 0
    domain = "general"
    if profile:
        row_count = profile.row_count or 0
        domain = profile.domain_hint or "general"
        if profile.column_classification and "columns" in profile.column_classification:
            columns = profile.column_classification["columns"]
    else:
        # Fallback: count rows
        count_result = await session.execute(text(f'SELECT COUNT(*) FROM "{table_name}"'))
        row_count = count_result.scalar() or 0

    # Get relationships
    rels_result = await session.execute(
        sa_select(DiscoveredRelationship).where(
            (DiscoveredRelationship.table_a == table_name) | (DiscoveredRelationship.table_b == table_name)
        )
    )
    rels = [
        {"table_a": r.table_a, "column_a": r.column_a, "table_b": r.table_b, "column_b": r.column_b,
         "relationship_type": r.relationship_type, "confidence": r.confidence}
        for r in rels_result.scalars().all()
    ]

    report = generate_profile_report(table_name, row_count, columns, domain, rels)

    return {
        "table_name": report.table_name,
        "row_count": report.row_count,
        "column_count": report.column_count,
        "domain": report.domain,
        "quality_score": report.quality_score,
        "priority": compute_report_priority(report),
        "summary": report.summary,
        "sections": [
            {"title": s.title, "content": s.content, "severity": s.severity}
            for s in report.sections
        ],
        "recommendations": report.recommendations,
        "text_report": format_report_text(report),
    }


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/benchmark")
async def benchmark_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare a metric across groups in a table.

    Body: {"group_column": "supplier", "metric_column": "cost", "metric_name": "Unit Cost"}
    """
    from business_brain.discovery.benchmarking import benchmark_groups

    group_col = body.get("group_column")
    metric_col = body.get("metric_column")
    if not group_col or not metric_col:
        return JSONResponse({"error": "group_column and metric_column required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 5000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    benchmark = benchmark_groups(rows, group_col, metric_col, body.get("metric_name"))
    if not benchmark:
        return {"error": "Insufficient data for benchmarking (need 2+ groups with numeric values)"}

    return {
        "metric_name": benchmark.metric_name,
        "group_column": benchmark.group_column,
        "best_group": benchmark.best_group,
        "worst_group": benchmark.worst_group,
        "spread": benchmark.spread,
        "spread_pct": benchmark.spread_pct,
        "ranking": benchmark.ranking,
        "significant_gaps": benchmark.significant_gaps,
        "summary": benchmark.summary,
        "groups": [
            {"name": g.group_name, "count": g.count, "mean": g.mean, "median": g.median,
             "min": g.min_val, "max": g.max_val, "std": g.std, "total": g.total}
            for g in benchmark.groups
        ],
    }


# ---------------------------------------------------------------------------
# Cohort Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/cohort")
async def cohort_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run cohort analysis on a table.

    Body: {"cohort_column": "supplier", "time_column": "month", "metric_column": "revenue"}
    """
    from business_brain.discovery.cohort_analysis import (
        build_cohorts,
        compute_cohort_health,
        find_declining_cohorts,
        pivot_cohort_table,
    )

    cohort_col = body.get("cohort_column")
    time_col = body.get("time_column")
    metric_col = body.get("metric_column")
    if not cohort_col or not time_col or not metric_col:
        return JSONResponse({"error": "cohort_column, time_column, and metric_column required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cohort_result = build_cohorts(rows, cohort_col, time_col, metric_col)
    if not cohort_result:
        return {"error": "Insufficient data for cohort analysis (need 2+ cohort-period combinations)"}

    health = compute_cohort_health(cohort_result)
    declining = find_declining_cohorts(cohort_result)
    pivot = pivot_cohort_table(cohort_result)

    return {
        "cohort_column": cohort_result.cohort_column,
        "time_column": cohort_result.time_column,
        "metric_column": cohort_result.metric_column,
        "cohorts": cohort_result.cohorts,
        "periods": cohort_result.periods,
        "summary": cohort_result.summary,
        "health": health,
        "declining_cohorts": declining,
        "pivot_table": pivot,
        "retention_matrix": cohort_result.retention_matrix,
        "growth_matrix": cohort_result.growth_matrix,
    }


# ---------------------------------------------------------------------------
# Validation Rules
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/validate")
async def validate_table(
    table_name: str,
    body: Optional[dict] = None,
    session: AsyncSession = Depends(get_session),
):
    """Validate table data against auto-generated or custom rules.

    Body (optional): {"rules": [{"name": "...", "column": "...", "rule_type": "not_null"}]}
    If no rules provided, auto-generates from column profile.
    """
    from sqlalchemy import select as sa_select
    from business_brain.discovery.validation_rules import (
        auto_generate_rules,
        create_rule,
        evaluate_rules,
    )

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    # Get rules
    rules = []
    if body and "rules" in body:
        for r in body["rules"]:
            rules.append(create_rule(
                r.get("name", r.get("column", "rule")),
                r["column"],
                r["rule_type"],
                severity=r.get("severity", "warning"),
                **{k: v for k, v in r.items() if k not in ("name", "column", "rule_type", "severity")},
            ))
    else:
        # Auto-generate from profile
        profile_result = await session.execute(
            sa_select(TableProfile).where(TableProfile.table_name == table_name)
        )
        profile = profile_result.scalar_one_or_none()
        if profile and profile.column_classification and "columns" in profile.column_classification:
            rules = auto_generate_rules(profile.column_classification["columns"])
        else:
            return {"message": "No profile found. Run discovery first to auto-generate validation rules."}

    # Get data
    result = await session.execute(text(f'SELECT * FROM "{table_name}" LIMIT 5000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    report = evaluate_rules(rows, rules)

    return {
        "table": table_name,
        "total_rules": report.total_rules,
        "total_rows": report.total_rows,
        "rules_passed": report.rules_passed,
        "rules_failed": report.rules_failed,
        "pass_rate": report.pass_rate,
        "total_violations": report.total_violations,
        "summary": report.summary,
        "rule_results": report.rule_results,
        "violations": [
            {"rule": v.rule_name, "column": v.column, "row": v.row_index,
             "value": str(v.value) if v.value is not None else None,
             "message": v.message, "severity": v.severity}
            for v in report.violations[:50]
        ],
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


@router.get("/recommendations")
async def get_recommendations(session: AsyncSession = Depends(get_session)):
    """Get analysis recommendations based on current data state.

    Uses Tier 1 (hardcoded templates) + Tier 2 (LLM-generated) cross-table
    intelligence. Tier 2 always runs — even when templates match — to suggest
    complementary, industry-specific analyses.
    """
    from sqlalchemy import select as sa_select

    from business_brain.action.onboarding import get_company_profile
    from business_brain.db.discovery_models import (
        DiscoveredRelationship,
        Insight,
        TableProfile,
    )
    from business_brain.discovery.insight_recommender import (
        compute_coverage,
        recommend_analyses_async,
    )

    profiles = list((await session.execute(sa_select(TableProfile))).scalars().all())
    insights = list((await session.execute(sa_select(Insight))).scalars().all())
    relationships = list((await session.execute(sa_select(DiscoveredRelationship))).scalars().all())

    # Convert ORM objects to dicts — include ALL fields for relationship-based
    # entity inference (confidence, column_a, column_b are critical)
    profile_dicts = [
        {"table_name": p.table_name, "row_count": p.row_count,
         "column_classification": p.column_classification}
        for p in profiles
    ]
    insight_dicts = [
        {"insight_type": i.insight_type, "source_tables": i.source_tables}
        for i in insights
    ]
    rel_dicts = [
        {"table_a": r.table_a, "table_b": r.table_b,
         "column_a": r.column_a, "column_b": r.column_b,
         "confidence": r.confidence,
         "relationship_type": getattr(r, "relationship_type", None)}
        for r in relationships
    ]

    # Fetch company context for Tier 2 LLM suggestions
    company_context = None
    try:
        cp = await get_company_profile(session)
        if cp:
            company_context = {
                "industry": cp.industry or "",
                "products": cp.products or [],
                "process_flow": cp.process_flow or "",
                "departments": cp.departments or [],
            }
    except Exception:
        import logging
        logging.getLogger(__name__).debug("Could not fetch company profile for recommendations")

    recs = await recommend_analyses_async(
        profile_dicts, insight_dicts, rel_dicts, company_context,
    )
    coverage = compute_coverage(profile_dicts, insight_dicts)

    return {
        "recommendations": [
            {"title": r.title, "description": r.description, "analysis_type": r.analysis_type,
             "target_table": r.target_table, "columns": r.columns, "priority": r.priority, "reason": r.reason}
            for r in recs
        ],
        "coverage": coverage,
    }


# ---------------------------------------------------------------------------
# KPI Calculator
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/kpis")
async def compute_table_kpis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute KPIs for a numeric column.

    Body: {"column": "revenue", "target": 1000, "capacity": 5000}
    """
    from business_brain.discovery.kpi_calculator import compute_all_kpis, moving_average, rate_of_change

    column = body.get("column")
    if not column:
        return JSONResponse({"error": "column required"}, 400)

    meta_store = MetadataStore(session)
    tables = await meta_store.list_tables()
    if table_name not in tables:
        return JSONResponse({"error": f"Table '{table_name}' not found"}, 404)

    result = await session.execute(text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 2000'))
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if not values:
        return {"error": "No numeric values found"}

    target = body.get("target")
    capacity = body.get("capacity")
    kpis = compute_all_kpis(values, target=target, capacity=capacity)
    ma = moving_average(values, window=min(5, len(values)))
    roc = rate_of_change(values)

    return {
        "table": table_name,
        "column": column,
        "total_values": len(values),
        "kpis": [
            {"name": k.name, "value": k.value, "unit": k.unit,
             "interpretation": k.interpretation, "trend": k.trend, "status": k.status}
            for k in kpis
        ],
        "moving_average": ma[-10:] if len(ma) > 10 else ma,
        "rate_of_change": roc[-10:] if len(roc) > 10 else roc,
    }


# ---------------------------------------------------------------------------
# Pareto Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/pareto")
async def pareto_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run Pareto (80/20) analysis on a table.

    Body: {"group_column": "supplier", "metric_column": "cost", "threshold": 80.0}
    """
    from business_brain.discovery.pareto_analysis import (
        compare_pareto,
        find_concentration_risk,
        pareto_analysis,
    )

    group_col = body.get("group_column")
    metric_col = body.get("metric_column")
    if not group_col or not metric_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "group_column and metric_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found in table"}

    threshold = body.get("threshold", 80.0)
    pareto = pareto_analysis(rows, group_col, metric_col, threshold)
    if not pareto:
        return {"error": "Insufficient data for Pareto analysis (need 2+ groups)"}

    risks = find_concentration_risk(pareto)

    return {
        "total": pareto.total,
        "is_pareto": pareto.is_pareto,
        "pareto_ratio": pareto.pareto_ratio,
        "vital_few_count": pareto.vital_few_count,
        "vital_few_pct": pareto.vital_few_pct,
        "vital_few_contribution": pareto.vital_few_contribution,
        "trivial_many_count": pareto.trivial_many_count,
        "summary": pareto.summary,
        "items": [
            {
                "name": item.name,
                "value": item.value,
                "pct": item.pct_of_total,
                "cumulative_pct": item.cumulative_pct,
                "rank": item.rank,
                "is_vital": item.is_vital,
            }
            for item in pareto.items
        ],
        "concentration_risks": risks,
    }


# ---------------------------------------------------------------------------
# What-If Scenario Engine
# ---------------------------------------------------------------------------


class ScenarioRequest(BaseModel):
    name: str
    parameters: dict[str, float]
    description: str = ""


class WhatIfRequest(BaseModel):
    scenarios: list[ScenarioRequest]
    formula: str
    base_values: Optional[dict] = None


@router.post("/scenarios/evaluate")
async def evaluate_scenarios(body: WhatIfRequest):
    """Evaluate what-if scenarios against a formula.

    Body: {
        "scenarios": [{"name": "base", "parameters": {"price": 100, "qty": 50}}, ...],
        "formula": "price * qty",
        "base_values": {"price": 100, "qty": 50}
    }
    """
    from business_brain.discovery.whatif_engine import (
        Scenario,
        compare_scenarios,
        evaluate_scenario,
    )

    scenarios = [Scenario(s.name, s.parameters, s.description) for s in body.scenarios]
    base = body.base_values

    if len(scenarios) == 1:
        outcome = evaluate_scenario(scenarios[0], body.formula, base)
        return {
            "scenario_name": outcome.scenario_name,
            "result": outcome.result,
            "parameters": outcome.parameters,
            "interpretation": outcome.interpretation,
        }

    comp = compare_scenarios(scenarios, body.formula, base)
    return {
        "best_scenario": comp.best_scenario,
        "worst_scenario": comp.worst_scenario,
        "range_min": comp.range_min,
        "range_max": comp.range_max,
        "sensitivity": comp.sensitivity,
        "summary": comp.summary,
        "outcomes": [
            {
                "scenario_name": o.scenario_name,
                "result": o.result,
                "parameters": o.parameters,
                "interpretation": o.interpretation,
            }
            for o in comp.outcomes
        ],
    }


@router.post("/scenarios/breakeven")
async def breakeven(body: dict):
    """Find breakeven value for a variable.

    Body: {"formula": "revenue - cost", "variable": "revenue",
           "base_values": {"revenue": 0, "cost": 500}, "target": 0,
           "search_range": [0, 1000]}
    """
    from business_brain.discovery.whatif_engine import breakeven_analysis

    formula = body.get("formula", "")
    variable = body.get("variable", "")
    base_values = body.get("base_values", {})
    target = body.get("target", 0.0)
    sr = body.get("search_range", [-1000, 1000])

    result = breakeven_analysis(formula, variable, base_values, target, tuple(sr))
    return result


@router.post("/scenarios/sensitivity")
async def sensitivity(body: dict):
    """Generate sensitivity table for a variable.

    Body: {"formula": "price * qty", "variable": "price",
           "base_values": {"price": 100, "qty": 50},
           "variations": [0.8, 0.9, 1.0, 1.1, 1.2]}
    """
    from business_brain.discovery.whatif_engine import sensitivity_table

    formula = body.get("formula", "")
    variable = body.get("variable", "")
    base_values = body.get("base_values", {})
    variations = body.get("variations")

    table = sensitivity_table(formula, variable, base_values, variations)
    return {"variable": variable, "base_value": base_values.get(variable, 0), "rows": table}


# ---------------------------------------------------------------------------
# Statistical Summary
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/stats")
async def stat_summary(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute comprehensive statistical summary for a numeric column.

    Body: {"column": "revenue"}
    """
    from business_brain.discovery.stat_summary import (
        compute_stat_summary,
        format_stat_table,
    )

    column = body.get("column")
    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(
        sql_text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 5000')
    )
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < 3:
        return {"error": f"Need at least 3 numeric values, found {len(values)}"}

    summary = compute_stat_summary(values, column)
    if summary is None:
        return {"error": "Could not compute summary"}

    text_table = format_stat_table(summary)

    return {
        "column": summary.column,
        "count": summary.count,
        "mean": summary.mean,
        "median": summary.median,
        "mode": summary.mode,
        "std": summary.std,
        "variance": summary.variance,
        "min": summary.min_val,
        "max": summary.max_val,
        "range": summary.range_val,
        "q1": summary.q1,
        "q3": summary.q3,
        "iqr": summary.iqr,
        "skewness": summary.skewness,
        "kurtosis": summary.kurtosis,
        "cv": summary.cv,
        "percentiles": summary.percentiles,
        "ci_95_lower": summary.ci_95_lower,
        "ci_95_upper": summary.ci_95_upper,
        "normality": summary.normality,
        "outlier_count": summary.outlier_count,
        "interpretation": summary.interpretation,
        "formatted_table": text_table,
    }


@router.post("/tables/{table_name}/compare-distributions")
async def compare_distributions_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare distributions of two numeric columns.

    Body: {"column_a": "cost_q1", "column_b": "cost_q2"}
    """
    from business_brain.discovery.stat_summary import (
        compare_distributions,
        compute_stat_summary,
    )

    col_a = body.get("column_a")
    col_b = body.get("column_b")
    if not col_a or not col_b:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column_a and column_b required"}, 400)

    from sqlalchemy import text as sql_text

    async def _get_values(col):
        res = await session.execute(
            sql_text(f'SELECT "{col}" FROM "{table_name}" WHERE "{col}" IS NOT NULL LIMIT 5000')
        )
        vals = []
        for r in res.fetchall():
            try:
                vals.append(float(r[0]))
            except (TypeError, ValueError):
                continue
        return vals

    vals_a = await _get_values(col_a)
    vals_b = await _get_values(col_b)

    if len(vals_a) < 3 or len(vals_b) < 3:
        return {"error": "Need at least 3 values in each column"}

    summary_a = compute_stat_summary(vals_a, col_a)
    summary_b = compute_stat_summary(vals_b, col_b)
    if not summary_a or not summary_b:
        return {"error": "Could not compute summaries"}

    comp = compare_distributions(summary_a, summary_b)
    return comp


# ---------------------------------------------------------------------------
# Segmentation Engine
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/segment")
async def segment_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Cluster table rows into segments based on numeric features.

    Body: {"features": ["revenue", "cost"], "n_segments": 3}
    """
    from business_brain.discovery.segmentation_engine import (
        find_segment_drivers,
        label_segments,
        segment_data,
    )

    features = body.get("features", [])
    n_segments = body.get("n_segments", 3)
    if not features:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "features list required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found"}

    seg_result = segment_data(rows, features, n_segments)
    if seg_result is None:
        return {"error": "Insufficient data for segmentation"}

    seg_result.segments = label_segments(seg_result.segments, features)
    drivers = find_segment_drivers(seg_result.segments, features)

    return {
        "n_segments": seg_result.n_segments,
        "total_rows": seg_result.total_rows,
        "features": seg_result.features,
        "quality_score": seg_result.quality_score,
        "summary": seg_result.summary,
        "segments": [
            {
                "segment_id": s.segment_id,
                "label": s.label,
                "size": s.size,
                "center": s.center,
                "spread": s.spread,
            }
            for s in seg_result.segments
        ],
        "drivers": drivers,
    }


# ---------------------------------------------------------------------------
# Trend Decomposition
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/decompose")
async def decompose_trend(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Decompose a numeric column into trend + seasonal + residual.

    Body: {"column": "revenue", "period": null}
    """
    from business_brain.discovery.trend_decomposer import (
        decompose,
        find_anomalous_residuals,
    )

    column = body.get("column")
    period = body.get("period")
    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(
        sql_text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 5000')
    )
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < 6:
        return {"error": f"Need at least 6 values, found {len(values)}"}

    dec = decompose(values, period=period)
    if dec is None:
        return {"error": "Could not decompose series"}

    anomalies = find_anomalous_residuals(dec.residual)

    return {
        "column": column,
        "period": dec.period,
        "trend_direction": dec.trend_direction,
        "trend_strength": dec.trend_strength,
        "seasonal_strength": dec.seasonal_strength,
        "summary": dec.summary,
        "trend": dec.trend[-50:] if len(dec.trend) > 50 else dec.trend,
        "seasonal": dec.seasonal[-50:] if len(dec.seasonal) > 50 else dec.seasonal,
        "residual": dec.residual[-50:] if len(dec.residual) > 50 else dec.residual,
        "anomalous_residuals": anomalies[:10],
    }


# ---------------------------------------------------------------------------
# Pivot Tables
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/pivot")
async def pivot_table(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Create a pivot table from table data.

    Body: {"row_field": "supplier", "col_field": "quarter", "value_field": "amount", "agg_func": "sum"}
    """
    from business_brain.discovery.pivot_engine import (
        create_pivot,
        find_pivot_outliers,
        format_pivot_text,
    )

    row_field = body.get("row_field")
    col_field = body.get("col_field")
    value_field = body.get("value_field")
    agg_func = body.get("agg_func", "sum")

    if not row_field or not col_field or not value_field:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "row_field, col_field, and value_field required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    pivot = create_pivot(rows, row_field, col_field, value_field, agg_func)
    if pivot is None:
        return {"error": "Insufficient data for pivot table"}

    outliers = find_pivot_outliers(pivot)
    text = format_pivot_text(pivot)

    # Build cells grid
    cells = {}
    for (rk, ck), cell in pivot.cells.items():
        if rk not in cells:
            cells[rk] = {}
        cells[rk][ck] = cell.value

    return {
        "row_keys": pivot.row_keys,
        "col_keys": pivot.col_keys,
        "cells": cells,
        "row_totals": pivot.row_totals,
        "col_totals": pivot.col_totals,
        "grand_total": pivot.grand_total,
        "agg_func": pivot.agg_func,
        "summary": pivot.summary,
        "formatted_text": text,
        "outliers": outliers[:10],
    }


# ---------------------------------------------------------------------------
# Variance Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/variance")
async def variance_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute budget vs actual variance analysis.

    Body: {"category_column": "department", "planned_column": "budget",
           "actual_column": "actual", "favorable_direction": "higher"}
    """
    from business_brain.discovery.variance_analysis import (
        compute_variance,
        find_root_causes,
        waterfall_breakdown,
    )

    cat_col = body.get("category_column")
    planned_col = body.get("planned_column")
    actual_col = body.get("actual_column")
    fav_dir = body.get("favorable_direction", "higher")

    if not cat_col or not planned_col or not actual_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column, planned_column, actual_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    report = compute_variance(rows, cat_col, planned_col, actual_col, fav_dir)
    if report is None:
        return {"error": "Insufficient data for variance analysis"}

    waterfall = waterfall_breakdown(report)
    causes = find_root_causes(report)

    return {
        "total_planned": report.total_planned,
        "total_actual": report.total_actual,
        "total_variance": report.total_variance,
        "total_variance_pct": report.total_variance_pct,
        "favorable_count": report.favorable_count,
        "unfavorable_count": report.unfavorable_count,
        "summary": report.summary,
        "items": [
            {
                "category": it.category,
                "planned": it.planned,
                "actual": it.actual,
                "variance": it.variance,
                "variance_pct": it.variance_pct,
                "is_favorable": it.is_favorable,
                "severity": it.severity,
            }
            for it in report.items
        ],
        "waterfall": waterfall,
        "root_causes": causes,
    }


# ---------------------------------------------------------------------------
# Schema Re-describe — generate column descriptions for existing tables
# ---------------------------------------------------------------------------


@router.post("/schema/redescribe")
async def redescribe_schema(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Re-generate column descriptions for all tables using data dictionary heuristics.

    This enriches metadata with human-readable column descriptions so the SQL
    agent can distinguish between similarly-typed columns (e.g. quantity vs yield).
    No LLM calls — uses fast pattern-matching from the data dictionary module.
    """
    from sqlalchemy import text as sql_text

    from business_brain.discovery.data_dictionary import (
        auto_describe_column,
        infer_column_type,
    )

    try:
        entries = await metadata_store.get_all(session)
        updated = 0
        for entry in entries:
            try:
                # Sample rows to infer column types
                result = await session.execute(
                    sql_text(f'SELECT * FROM "{entry.table_name}" LIMIT 100')
                )
                sample_rows = [dict(r._mapping) for r in result.fetchall()]
                if not sample_rows:
                    continue

                columns_meta = []
                for col_name in sample_rows[0].keys():
                    col_values = [row.get(col_name) for row in sample_rows]
                    col_type = infer_column_type(col_values)
                    non_null = [v for v in col_values if v is not None]
                    desc = auto_describe_column(col_name, col_type, {
                        "unique_pct": len(set(non_null)) / max(len(non_null), 1) * 100,
                        "null_pct": (len(col_values) - len(non_null)) / max(len(col_values), 1) * 100,
                    })
                    # Preserve original type from existing metadata
                    orig_type = col_type
                    if entry.columns_metadata:
                        for cm in entry.columns_metadata:
                            if cm.get("name") == col_name:
                                orig_type = cm.get("type", col_type)
                                break
                    columns_meta.append({
                        "name": col_name,
                        "type": orig_type,
                        "description": desc,
                    })

                entry.columns_metadata = columns_meta
                updated += 1
            except Exception:
                logger.debug("Failed to redescribe table %s", entry.table_name)
                await session.rollback()

        await session.commit()
        return {"status": "completed", "tables_updated": updated, "total_tables": len(entries)}
    except Exception:
        logger.exception("Schema redescribe failed")
        await session.rollback()
        return JSONResponse({"error": "Schema redescribe failed"}, 500)


# ---------------------------------------------------------------------------
# Data Dictionary
# ---------------------------------------------------------------------------


@router.get("/tables/{table_name}/dictionary")
async def get_data_dictionary(
    table_name: str,
    session: AsyncSession = Depends(get_session),
):
    """Auto-generate a data dictionary for a table."""
    from business_brain.discovery.data_dictionary import (
        format_dictionary_markdown,
        generate_data_dictionary,
    )

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 500'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found"}

    dd = generate_data_dictionary(rows, table_name)
    if dd is None:
        return {"error": "Could not generate dictionary"}

    markdown = format_dictionary_markdown(dd)

    return {
        "table_name": dd.table_name,
        "row_count": dd.row_count,
        "column_count": dd.column_count,
        "summary": dd.summary,
        "markdown": markdown,
        "columns": [
            {
                "name": c.name,
                "type": c.inferred_type,
                "description": c.description,
                "null_pct": c.null_pct,
                "unique_pct": c.unique_pct,
                "min": c.min_value,
                "max": c.max_value,
                "mean": c.mean_value,
                "sample_values": c.sample_values,
                "tags": c.tags,
            }
            for c in dd.columns
        ],
        "relationships": dd.relationships_hint,
    }


# ---------------------------------------------------------------------------
# Quality Score
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/quality-score")
async def compute_quality_score(
    table_name: str,
    session: AsyncSession = Depends(get_session),
):
    """Compute a unified data quality score for a table."""
    from business_brain.discovery.quality_scorer import compute_quality_report

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 5000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return {"error": "No data found"}

    report = compute_quality_report(rows)

    return {
        "overall_score": report.overall_score,
        "grade": report.grade,
        "summary": report.summary,
        "dimensions": [
            {
                "dimension": d.dimension,
                "score": d.score,
                "weight": d.weight,
                "issues": d.issues,
            }
            for d in report.dimensions
        ],
        "critical_issues": report.critical_issues,
        "recommendations": report.recommendations,
    }


# ---------------------------------------------------------------------------
# ABC Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/abc")
async def abc_analysis_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Run ABC classification on table data.

    Body: {"name_column": "supplier", "value_column": "cost", "a_threshold": 80, "b_threshold": 95}
    """
    from business_brain.discovery.abc_analysis import abc_analysis, format_abc_table

    name_col = body.get("name_column")
    value_col = body.get("value_column")
    if not name_col or not value_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "name_column and value_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    a_thresh = body.get("a_threshold", 80.0)
    b_thresh = body.get("b_threshold", 95.0)
    abc = abc_analysis(rows, name_col, value_col, a_thresh, b_thresh)
    if abc is None:
        return {"error": "Insufficient data for ABC analysis"}

    return {
        "total_value": abc.total_value,
        "a_count": abc.a_count,
        "b_count": abc.b_count,
        "c_count": abc.c_count,
        "a_value_pct": abc.a_value_pct,
        "b_value_pct": abc.b_value_pct,
        "c_value_pct": abc.c_value_pct,
        "summary": abc.summary,
        "formatted_table": format_abc_table(abc),
        "items": [
            {
                "name": it.name,
                "value": it.value,
                "pct": it.pct_of_total,
                "cumulative_pct": it.cumulative_pct,
                "rank": it.rank,
                "category": it.category,
            }
            for it in abc.items
        ],
    }


# ---------------------------------------------------------------------------
# Funnel Analysis
# ---------------------------------------------------------------------------


@router.post("/funnel/analyze")
async def funnel_analyze(body: dict):
    """Analyze a conversion funnel.

    Body: {"stages": [["Visits", 1000], ["Signups", 200], ["Purchases", 50]]}
    """
    from business_brain.discovery.funnel_analysis import analyze_funnel, format_funnel_text

    stages_raw = body.get("stages", [])
    if len(stages_raw) < 2:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Need at least 2 stages"}, 400)

    stages = [(s[0], int(s[1])) for s in stages_raw]
    result = analyze_funnel(stages)
    if result is None:
        return {"error": "Could not analyze funnel"}

    return {
        "initial_count": result.initial_count,
        "final_count": result.final_count,
        "overall_conversion": result.overall_conversion,
        "biggest_drop_stage": result.biggest_drop_stage,
        "biggest_drop_pct": result.biggest_drop_pct,
        "summary": result.summary,
        "formatted_text": format_funnel_text(result),
        "stages": [
            {
                "name": s.name,
                "count": s.count,
                "pct_of_total": s.pct_of_total,
                "conversion_rate": s.conversion_rate,
                "drop_off": s.drop_off,
                "drop_off_pct": s.drop_off_pct,
            }
            for s in result.stages
        ],
    }


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/rolling")
async def rolling_stats_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute rolling statistics for a numeric column.

    Body: {"column": "revenue", "window": 5}
    """
    from business_brain.discovery.rolling_stats import (
        detect_regime_changes,
        rolling_statistics,
    )

    column = body.get("column")
    window = body.get("window", 5)
    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(
        sql_text(f'SELECT "{column}" FROM "{table_name}" WHERE "{column}" IS NOT NULL LIMIT 5000')
    )
    rows = result.fetchall()
    values = []
    for r in rows:
        try:
            values.append(float(r[0]))
        except (TypeError, ValueError):
            continue

    if len(values) < window:
        return {"error": f"Need at least {window} values, found {len(values)}"}

    stats = rolling_statistics(values, window)
    if stats is None:
        return {"error": "Could not compute rolling statistics"}

    regime_changes = detect_regime_changes(values, window)

    # Only return last 100 values to keep response size manageable
    n = min(100, len(values))

    return {
        "column": column,
        "window": window,
        "total_values": len(values),
        "summary": stats.summary,
        "rolling_mean": stats.rolling_mean[-n:],
        "rolling_std": stats.rolling_std[-n:],
        "z_scores": stats.z_scores[-n:],
        "regime_changes": regime_changes[:20],
    }


# ---------------------------------------------------------------------------
# Contribution Analysis
# ---------------------------------------------------------------------------


@router.post("/contribution/analyze")
async def contribution_analyze(body: dict):
    """Analyze what's driving change between two periods.

    Body: {"before": {"Sales": 100, "Services": 50}, "after": {"Sales": 120, "Services": 60}}
    """
    from business_brain.discovery.contribution_analysis import (
        analyze_contributions,
        waterfall_data,
    )

    before = body.get("before", {})
    after = body.get("after", {})
    result = analyze_contributions(before, after)
    if result is None:
        return {"error": "No data provided"}

    wf = waterfall_data(result)

    return {
        "total_before": result.total_before,
        "total_after": result.total_after,
        "total_change": result.total_change,
        "total_change_pct": result.total_change_pct,
        "top_positive_driver": result.top_positive_driver,
        "top_negative_driver": result.top_negative_driver,
        "concentration": result.concentration,
        "summary": result.summary,
        "items": [
            {
                "name": it.name,
                "value_before": it.value_before,
                "value_after": it.value_after,
                "absolute_change": it.absolute_change,
                "pct_change": it.pct_change,
                "contribution_pct": it.contribution_pct,
                "direction": it.direction,
            }
            for it in result.items
        ],
        "waterfall": wf,
    }


# ---------------------------------------------------------------------------
# Composite Index Calculator
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/index")
async def compute_index_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute a composite index score for entities in a table.

    Body: {
        "entity_column": "supplier",
        "metrics": [
            {"column": "quality", "weight": 0.5, "direction": "higher_is_better"},
            {"column": "cost", "weight": 0.5, "direction": "lower_is_better"}
        ],
        "index_name": "Supplier Score"
    }
    """
    from business_brain.discovery.index_calculator import compute_index, format_index_table

    entity_col = body.get("entity_column")
    metrics = body.get("metrics", [])
    index_name = body.get("index_name", "Index")

    if not entity_col or not metrics:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column and metrics required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    idx = compute_index(rows, entity_col, metrics, index_name)
    if idx is None:
        return {"error": "Insufficient data for index computation"}

    return {
        "name": idx.name,
        "entity_count": idx.entity_count,
        "mean_score": idx.mean_score,
        "median_score": idx.median_score,
        "std_score": idx.std_score,
        "top_entity": idx.top_entity,
        "bottom_entity": idx.bottom_entity,
        "summary": idx.summary,
        "formatted_table": format_index_table(idx),
        "scores": [
            {
                "entity": s.entity,
                "score": s.score,
                "grade": s.grade,
                "rank": s.rank,
                "components": [
                    {"name": c.name, "raw": c.raw_value, "normalized": c.normalized_value,
                     "weight": c.weight, "contribution": c.weighted_contribution}
                    for c in s.components
                ],
            }
            for s in idx.scores
        ],
    }


# ---------------------------------------------------------------------------
# Capacity Planning
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/capacity")
async def capacity_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute capacity utilization for entities in a table.

    Body: {
        "entity_column": "machine",
        "actual_column": "output",
        "capacity_column": "max_output",
        "time_column": null  (optional)
    }
    """
    from business_brain.discovery.capacity_planning import (
        compute_utilization,
        detect_bottlenecks,
        forecast_capacity_exhaustion,
        capacity_summary,
    )

    entity_col = body.get("entity_column")
    actual_col = body.get("actual_column")
    capacity_col = body.get("capacity_column")
    time_col = body.get("time_column")

    if not entity_col or not actual_col or not capacity_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, actual_column, capacity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    util = compute_utilization(rows, entity_col, actual_col, capacity_col, time_col)
    if util is None:
        return {"error": "Insufficient data for capacity analysis"}

    # Bottleneck detection if a stage-like entity is present
    bottlenecks = detect_bottlenecks(rows, entity_col, actual_col, time_col)

    # Exhaustion forecast
    forecasts = []
    if time_col:
        forecasts = forecast_capacity_exhaustion(rows, entity_col, actual_col, capacity_col, time_col)

    return {
        "entity_count": util.entity_count,
        "mean_utilization": util.mean_utilization,
        "summary": util.summary,
        "over_utilized": util.over_utilized,
        "under_utilized": util.under_utilized,
        "bottlenecks": util.bottlenecks,
        "entities": [
            {
                "entity": e.entity,
                "actual": e.actual,
                "capacity": e.capacity,
                "utilization_pct": e.utilization_pct,
                "status": e.status,
            }
            for e in util.entities
        ],
        "stage_bottlenecks": [
            {
                "stage": b.stage,
                "throughput": b.throughput,
                "throughput_pct_of_max": b.throughput_pct_of_max,
                "is_bottleneck": b.is_bottleneck,
                "constraint_ratio": b.constraint_ratio,
            }
            for b in bottlenecks
        ],
        "exhaustion_forecasts": [
            {
                "entity": f.entity,
                "current_utilization": f.current_utilization,
                "trend_per_period": f.trend_per_period,
                "periods_to_exhaustion": f.periods_to_exhaustion,
                "urgency": f.urgency,
            }
            for f in forecasts
        ],
    }


# ---------------------------------------------------------------------------
# Efficiency Metrics (OEE, Yield, Energy, Waste)
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/oee")
async def oee_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute OEE (Overall Equipment Effectiveness) for entities.

    Body: {
        "entity_column": "machine",
        "availability_column": "availability",
        "performance_column": "performance",
        "quality_column": "quality"
    }
    """
    from business_brain.discovery.efficiency_metrics import compute_oee

    entity_col = body.get("entity_column")
    avail_col = body.get("availability_column")
    perf_col = body.get("performance_column")
    qual_col = body.get("quality_column")

    if not all([entity_col, avail_col, perf_col, qual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, availability_column, performance_column, quality_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    oee = compute_oee(rows, entity_col, avail_col, perf_col, qual_col)
    if oee is None:
        return {"error": "Insufficient data for OEE computation"}

    return {
        "mean_oee": oee.mean_oee,
        "best_entity": oee.best_entity,
        "worst_entity": oee.worst_entity,
        "world_class_count": oee.world_class_count,
        "summary": oee.summary,
        "entities": [
            {
                "entity": e.entity,
                "availability": e.availability,
                "performance": e.performance,
                "quality": e.quality,
                "oee": e.oee,
                "oee_grade": e.oee_grade,
                "limiting_factor": e.limiting_factor,
            }
            for e in oee.entities
        ],
    }


@router.post("/tables/{table_name}/yield")
async def yield_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute yield analysis for entities.

    Body: {
        "entity_column": "line",
        "input_column": "raw_material",
        "output_column": "finished_goods",
        "defect_column": "defects"  (optional)
    }
    """
    from business_brain.discovery.efficiency_metrics import compute_yield_analysis

    entity_col = body.get("entity_column")
    input_col = body.get("input_column")
    output_col = body.get("output_column")
    defect_col = body.get("defect_column")

    if not all([entity_col, input_col, output_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, input_column, output_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    yld = compute_yield_analysis(rows, entity_col, input_col, output_col, defect_col)
    if yld is None:
        return {"error": "Insufficient data for yield analysis"}

    return {
        "mean_yield": yld.mean_yield,
        "best_entity": yld.best_entity,
        "worst_entity": yld.worst_entity,
        "summary": yld.summary,
        "entities": [
            {
                "entity": e.entity,
                "input_total": e.input_total,
                "output_total": e.output_total,
                "yield_pct": e.yield_pct,
                "defect_rate": e.defect_rate,
                "waste_pct": e.waste_pct,
            }
            for e in yld.entities
        ],
    }


@router.post("/tables/{table_name}/energy-efficiency")
async def energy_efficiency(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute energy efficiency (specific energy consumption).

    Body: {
        "entity_column": "machine",
        "output_column": "production_tons",
        "energy_column": "kwh_consumed"
    }
    """
    from business_brain.discovery.efficiency_metrics import compute_energy_efficiency

    entity_col = body.get("entity_column")
    output_col = body.get("output_column")
    energy_col = body.get("energy_column")

    if not all([entity_col, output_col, energy_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, output_column, energy_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    eng = compute_energy_efficiency(rows, entity_col, output_col, energy_col)
    if eng is None:
        return {"error": "Insufficient data for energy efficiency analysis"}

    return {
        "mean_sec": eng.mean_sec,
        "best_entity": eng.best_entity,
        "worst_entity": eng.worst_entity,
        "potential_savings_pct": eng.potential_savings_pct,
        "summary": eng.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_output": e.total_output,
                "total_energy": e.total_energy,
                "specific_energy": e.specific_energy,
                "efficiency_grade": e.efficiency_grade,
            }
            for e in eng.entities
        ],
    }


# ---------------------------------------------------------------------------
# Downtime Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/downtime")
async def downtime_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze equipment downtime patterns.

    Body: {
        "machine_column": "machine_id",
        "duration_column": "downtime_hours",
        "reason_column": "failure_reason",  (optional)
        "time_column": "date"  (optional)
    }
    """
    from business_brain.discovery.downtime_analyzer import (
        analyze_downtime,
        downtime_pareto,
        format_downtime_report,
    )

    machine_col = body.get("machine_column")
    duration_col = body.get("duration_column")
    reason_col = body.get("reason_column")
    time_col = body.get("time_column")

    if not machine_col or not duration_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "machine_column and duration_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    dt = analyze_downtime(rows, machine_col, duration_col, reason_col, time_col)
    if dt is None:
        return {"error": "Insufficient data for downtime analysis"}

    pareto = []
    if reason_col:
        pareto = downtime_pareto(rows, reason_col, duration_col)

    return {
        "total_downtime": dt.total_downtime,
        "total_events": dt.total_events,
        "worst_machine": dt.worst_machine,
        "best_machine": dt.best_machine,
        "summary": dt.summary,
        "report": format_downtime_report(dt, pareto or None),
        "machines": [
            {
                "machine": m.machine,
                "total_downtime": m.total_downtime,
                "event_count": m.event_count,
                "mttr": m.mttr,
                "availability_pct": m.availability_pct,
                "top_reason": m.top_reason,
            }
            for m in dt.machines
        ],
        "top_reasons": [
            {
                "reason": r.reason,
                "total_duration": r.total_duration,
                "event_count": r.event_count,
                "pct_of_total": r.pct_of_total,
            }
            for r in dt.top_reasons
        ],
        "pareto": [
            {
                "reason": p.reason,
                "total_duration": p.total_duration,
                "pct_of_total": p.pct_of_total,
                "cumulative_pct": p.cumulative_pct,
                "category": p.category,
            }
            for p in pareto
        ],
    }


# ---------------------------------------------------------------------------
# Supplier Scorecard
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/supplier-scorecard")
async def supplier_scorecard(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Build supplier scorecards.

    Body: {
        "supplier_column": "supplier_name",
        "metrics": [
            {"column": "quality", "weight": 0.4, "direction": "higher_is_better"},
            {"column": "delivery_time", "weight": 0.3, "direction": "lower_is_better"},
            {"column": "cost", "weight": 0.3, "direction": "lower_is_better"}
        ]
    }
    """
    from business_brain.discovery.supplier_scorecard import (
        build_scorecard,
        format_scorecard,
    )

    supplier_col = body.get("supplier_column")
    metrics = body.get("metrics", [])

    if not supplier_col or not metrics:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and metrics required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    sc = build_scorecard(rows, supplier_col, metrics)
    if sc is None:
        return {"error": "Insufficient data for scorecard"}

    return {
        "supplier_count": sc.supplier_count,
        "mean_score": sc.mean_score,
        "best_supplier": sc.best_supplier,
        "worst_supplier": sc.worst_supplier,
        "grade_distribution": sc.grade_distribution,
        "summary": sc.summary,
        "formatted_table": format_scorecard(sc),
        "suppliers": [
            {
                "supplier": s.supplier,
                "score": s.score,
                "grade": s.grade,
                "rank": s.rank,
                "strengths": s.strengths,
                "weaknesses": s.weaknesses,
                "metrics": [
                    {"name": m.metric_name, "raw": m.raw_value, "normalized": m.normalized, "weight": m.weight}
                    for m in s.metric_scores
                ],
            }
            for s in sc.suppliers
        ],
    }


@router.post("/tables/{table_name}/supplier-concentration")
async def supplier_concentration_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze supplier concentration (HHI).

    Body: {"supplier_column": "supplier", "value_column": "spend"}
    """
    from business_brain.discovery.supplier_scorecard import supplier_concentration

    supplier_col = body.get("supplier_column")
    value_col = body.get("value_column")

    if not supplier_col or not value_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and value_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    conc = supplier_concentration(rows, supplier_col, value_col)
    if conc is None:
        return {"error": "Insufficient data"}

    return {
        "hhi": conc.hhi,
        "concentration_level": conc.concentration_level,
        "top_supplier_share": conc.top_supplier_share,
        "summary": conc.summary,
        "suppliers": conc.suppliers,
    }


# ---------------------------------------------------------------------------
# Production Scheduling
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/shift-performance")
async def shift_performance(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze shift-wise production performance.

    Body: {
        "shift_column": "shift",
        "output_column": "production_tons",
        "target_column": "target_tons"  (optional)
    }
    """
    from business_brain.discovery.production_scheduler import analyze_shift_performance

    shift_col = body.get("shift_column")
    output_col = body.get("output_column")
    target_col = body.get("target_column")

    if not shift_col or not output_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "shift_column and output_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    sp = analyze_shift_performance(rows, shift_col, output_col, target_col)
    if sp is None:
        return {"error": "Insufficient data for shift analysis"}

    return {
        "best_shift": sp.best_shift,
        "worst_shift": sp.worst_shift,
        "variance_pct": sp.variance_pct,
        "total_output": sp.total_output,
        "summary": sp.summary,
        "shifts": [
            {
                "shift": s.shift,
                "total_output": s.total_output,
                "avg_output": s.avg_output,
                "event_count": s.event_count,
                "achievement_pct": s.achievement_pct,
                "std_dev": s.std_dev,
                "consistency_grade": s.consistency_grade,
            }
            for s in sp.shifts
        ],
    }


@router.post("/tables/{table_name}/plan-vs-actual")
async def plan_vs_actual_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare planned vs actual production.

    Body: {"entity_column": "product", "plan_column": "planned_qty", "actual_column": "actual_qty"}
    """
    from business_brain.discovery.production_scheduler import plan_vs_actual

    entity_col = body.get("entity_column")
    plan_col = body.get("plan_column")
    actual_col = body.get("actual_column")

    if not all([entity_col, plan_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, plan_column, actual_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    pva = plan_vs_actual(rows, entity_col, plan_col, actual_col)
    if pva is None:
        return {"error": "Insufficient data"}

    return {
        "overall_achievement_pct": pva.overall_achievement_pct,
        "over_achievers": pva.over_achievers,
        "under_achievers": pva.under_achievers,
        "summary": pva.summary,
        "entities": [
            {
                "entity": e.entity,
                "planned": e.planned,
                "actual": e.actual,
                "achievement_pct": e.achievement_pct,
                "variance": e.variance,
                "status": e.status,
            }
            for e in pva.entities
        ],
    }


# ---------------------------------------------------------------------------
# Inventory Optimization
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/inventory-turnover")
async def inventory_turnover(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute inventory turnover ratios.

    Body: {"item_column": "product", "cost_of_goods_column": "cogs", "avg_inventory_column": "avg_inventory"}
    """
    from business_brain.discovery.inventory_optimizer import compute_inventory_turnover

    item_col = body.get("item_column")
    cogs_col = body.get("cost_of_goods_column")
    inv_col = body.get("avg_inventory_column")

    if not all([item_col, cogs_col, inv_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "item_column, cost_of_goods_column, avg_inventory_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    tr = compute_inventory_turnover(rows, item_col, cogs_col, inv_col)
    if tr is None:
        return {"error": "Insufficient data"}

    return {
        "mean_turnover": tr.mean_turnover,
        "best_item": tr.best_item,
        "worst_item": tr.worst_item,
        "slow_movers": tr.slow_movers,
        "fast_movers": tr.fast_movers,
        "summary": tr.summary,
        "items": [
            {
                "item": i.item,
                "cogs": i.cogs,
                "avg_inventory": i.avg_inventory,
                "turnover_ratio": i.turnover_ratio,
                "days_of_inventory": i.days_of_inventory,
                "category": i.category,
            }
            for i in tr.items
        ],
    }


@router.post("/tables/{table_name}/inventory-health")
async def inventory_health(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze inventory health (overstocked, understocked, at reorder point).

    Body: {
        "item_column": "product",
        "quantity_column": "current_qty",
        "min_column": "min_qty",  (optional)
        "max_column": "max_qty",  (optional)
        "reorder_column": "reorder_point"  (optional)
    }
    """
    from business_brain.discovery.inventory_optimizer import analyze_inventory_health

    item_col = body.get("item_column")
    qty_col = body.get("quantity_column")
    min_col = body.get("min_column")
    max_col = body.get("max_column")
    reorder_col = body.get("reorder_column")

    if not item_col or not qty_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "item_column and quantity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    health = analyze_inventory_health(rows, item_col, qty_col, min_col, max_col, reorder_col)
    if health is None:
        return {"error": "Insufficient data"}

    return {
        "overstocked_count": health.overstocked_count,
        "understocked_count": health.understocked_count,
        "at_reorder_count": health.at_reorder_count,
        "healthy_count": health.healthy_count,
        "summary": health.summary,
        "items": [
            {
                "item": i.item,
                "current_qty": i.current_qty,
                "min_qty": i.min_qty,
                "max_qty": i.max_qty,
                "reorder_point": i.reorder_point,
                "status": i.status,
            }
            for i in health.items
        ],
    }


# ---------------------------------------------------------------------------
# Cost Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/cost-breakdown")
async def cost_breakdown_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Break down costs by category.

    Body: {"category_column": "dept", "amount_column": "cost"}
    """
    from business_brain.discovery.cost_analyzer import cost_breakdown

    cat_col = body.get("category_column")
    amt_col = body.get("amount_column")

    if not cat_col or not amt_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column and amount_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cb = cost_breakdown(rows, cat_col, amt_col)
    if cb is None:
        return {"error": "Insufficient data"}

    return {
        "total_cost": cb.total_cost,
        "top_category": cb.top_category,
        "top_3_share_pct": cb.top_3_share_pct,
        "summary": cb.summary,
        "categories": [
            {
                "name": c.name,
                "amount": c.amount,
                "share_pct": c.share_pct,
                "cumulative_pct": c.cumulative_pct,
                "rank": c.rank,
            }
            for c in cb.categories
        ],
    }


@router.post("/tables/{table_name}/cost-per-unit")
async def cost_per_unit_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute cost per unit for entities.

    Body: {"entity_column": "product", "cost_column": "total_cost", "quantity_column": "units_produced"}
    """
    from business_brain.discovery.cost_analyzer import cost_per_unit

    entity_col = body.get("entity_column")
    cost_col = body.get("cost_column")
    qty_col = body.get("quantity_column")

    if not all([entity_col, cost_col, qty_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, cost_column, quantity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cpu = cost_per_unit(rows, entity_col, cost_col, qty_col)
    if cpu is None:
        return {"error": "Insufficient data"}

    return {
        "mean_cpu": cpu.mean_cpu,
        "median_cpu": cpu.median_cpu,
        "best_entity": cpu.best_entity,
        "worst_entity": cpu.worst_entity,
        "spread_pct": cpu.spread_pct,
        "summary": cpu.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_cost": e.total_cost,
                "total_quantity": e.total_quantity,
                "cost_per_unit": e.cost_per_unit,
                "deviation_from_mean_pct": e.deviation_from_mean_pct,
            }
            for e in cpu.entities
        ],
    }


@router.post("/tables/{table_name}/cost-trend")
async def cost_trend_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Track cost trends over time.

    Body: {"time_column": "month", "cost_column": "total_cost", "entity_column": null (optional)}
    """
    from business_brain.discovery.cost_analyzer import cost_trend

    time_col = body.get("time_column")
    cost_col = body.get("cost_column")
    entity_col = body.get("entity_column")

    if not time_col or not cost_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "time_column and cost_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    ct = cost_trend(rows, time_col, cost_col, entity_col)
    if ct is None:
        return {"error": "Insufficient data"}

    return {
        "trend_direction": ct.trend_direction,
        "trend_pct_per_period": ct.trend_pct_per_period,
        "total_change_pct": ct.total_change_pct,
        "volatility": ct.volatility,
        "summary": ct.summary,
        "periods": [
            {
                "period": p.period,
                "total_cost": p.total_cost,
                "change_from_prev": p.change_from_prev,
                "change_pct": p.change_pct,
            }
            for p in ct.periods
        ],
    }


@router.post("/tables/{table_name}/cost-variance")
async def cost_variance_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze cost variance (actual vs budget).

    Body: {"entity_column": "dept", "actual_column": "actual_cost", "budget_column": "budget_cost"}
    """
    from business_brain.discovery.cost_analyzer import cost_variance

    entity_col = body.get("entity_column")
    actual_col = body.get("actual_column")
    budget_col = body.get("budget_column")

    if not all([entity_col, actual_col, budget_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, actual_column, budget_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cv = cost_variance(rows, entity_col, actual_col, budget_col)
    if cv is None:
        return {"error": "Insufficient data"}

    return {
        "total_actual": cv.total_actual,
        "total_budget": cv.total_budget,
        "total_variance": cv.total_variance,
        "total_variance_pct": cv.total_variance_pct,
        "favorable_count": cv.favorable_count,
        "unfavorable_count": cv.unfavorable_count,
        "summary": cv.summary,
        "entities": [
            {
                "entity": e.entity,
                "actual": e.actual,
                "budget": e.budget,
                "variance": e.variance,
                "variance_pct": e.variance_pct,
                "status": e.status,
            }
            for e in cv.entities
        ],
    }


# ---------------------------------------------------------------------------
# Material Balance
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/material-balance")
async def material_balance_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute material balance (input vs output, recovery rate).

    Body: {"entity_column": "furnace", "input_column": "raw_material_tons", "output_column": "product_tons"}
    """
    from business_brain.discovery.material_balance import (
        compute_material_balance,
        detect_material_leakage,
        format_balance_report,
    )

    entity_col = body.get("entity_column")
    input_col = body.get("input_column")
    output_col = body.get("output_column")
    loss_col = body.get("loss_column")

    if not all([entity_col, input_col, output_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, input_column, output_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    mb = compute_material_balance(rows, entity_col, input_col, output_col, loss_col)
    if mb is None:
        return {"error": "Insufficient data"}

    leakage = detect_material_leakage(rows, entity_col, output_col)

    return {
        "total_input": mb.total_input,
        "total_output": mb.total_output,
        "total_loss": mb.total_loss,
        "overall_recovery_pct": mb.overall_recovery_pct,
        "best_recovery_entity": mb.best_recovery_entity,
        "worst_recovery_entity": mb.worst_recovery_entity,
        "summary": mb.summary,
        "report": format_balance_report(mb, leakage or None),
        "entities": [
            {
                "entity": e.entity,
                "total_input": e.total_input,
                "total_output": e.total_output,
                "loss": e.loss,
                "recovery_pct": e.recovery_pct,
                "loss_pct": e.loss_pct,
            }
            for e in mb.entities
        ],
        "leakage_points": [
            {
                "from_stage": lp.from_stage,
                "to_stage": lp.to_stage,
                "input_qty": lp.input_qty,
                "output_qty": lp.output_qty,
                "loss": lp.loss,
                "loss_pct": lp.loss_pct,
                "severity": lp.severity,
            }
            for lp in (leakage or [])
        ],
    }


# ---------------------------------------------------------------------------
# Workforce Analytics
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/attendance")
async def attendance_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze employee attendance.

    Body: {"employee_column": "employee_name", "status_column": "attendance_status", "date_column": "date"}
    """
    from business_brain.discovery.workforce_analytics import analyze_attendance

    emp_col = body.get("employee_column")
    status_col = body.get("status_column")
    date_col = body.get("date_column")

    if not emp_col or not status_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column and status_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    att = analyze_attendance(rows, emp_col, status_col, date_col)
    if att is None:
        return {"error": "Insufficient data"}

    return {
        "total_employees": att.total_employees,
        "avg_attendance_rate": att.avg_attendance_rate,
        "chronic_absentees": att.chronic_absentees,
        "perfect_attendance": att.perfect_attendance,
        "summary": att.summary,
        "employees": [
            {
                "employee": e.employee,
                "total_days": e.total_days,
                "present_days": e.present_days,
                "absent_days": e.absent_days,
                "leave_days": e.leave_days,
                "attendance_rate": e.attendance_rate,
            }
            for e in att.employees
        ],
    }


@router.post("/tables/{table_name}/labor-productivity")
async def labor_productivity(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute labor productivity (output per hour).

    Body: {"entity_column": "department", "output_column": "units_produced", "hours_column": "labor_hours"}
    """
    from business_brain.discovery.workforce_analytics import compute_labor_productivity

    entity_col = body.get("entity_column")
    output_col = body.get("output_column")
    hours_col = body.get("hours_column")

    if not all([entity_col, output_col, hours_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, output_column, hours_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    prod = compute_labor_productivity(rows, entity_col, output_col, hours_col)
    if prod is None:
        return {"error": "Insufficient data"}

    return {
        "mean_productivity": prod.mean_productivity,
        "best_entity": prod.best_entity,
        "worst_entity": prod.worst_entity,
        "spread_ratio": prod.spread_ratio,
        "summary": prod.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_output": e.total_output,
                "total_hours": e.total_hours,
                "productivity": e.productivity,
                "productivity_index": e.productivity_index,
            }
            for e in prod.entities
        ],
    }


# ---------------------------------------------------------------------------
# Logistics Tracking
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/delivery-performance")
async def delivery_performance(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze delivery performance (on-time rate).

    Body: {"entity_column": "supplier", "promised_column": "promised_date", "actual_column": "delivery_date"}
    """
    from business_brain.discovery.logistics_tracker import analyze_delivery_performance

    entity_col = body.get("entity_column")
    promised_col = body.get("promised_column")
    actual_col = body.get("actual_column")

    if not all([entity_col, promised_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, promised_column, actual_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    dp = analyze_delivery_performance(rows, entity_col, promised_col, actual_col)
    if dp is None:
        return {"error": "Insufficient data"}

    return {
        "total_deliveries": dp.total_deliveries,
        "on_time_count": dp.on_time_count,
        "on_time_rate": dp.on_time_rate,
        "avg_delay": dp.avg_delay,
        "worst_entity": dp.worst_entity,
        "best_entity": dp.best_entity,
        "summary": dp.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_deliveries": e.total_deliveries,
                "on_time_count": e.on_time_count,
                "late_count": e.late_count,
                "early_count": e.early_count,
                "on_time_rate": e.on_time_rate,
                "avg_delay": e.avg_delay,
            }
            for e in dp.entities
        ],
    }


@router.post("/tables/{table_name}/vehicle-utilization")
async def vehicle_utilization(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute vehicle load utilization.

    Body: {"vehicle_column": "truck_id", "capacity_column": "capacity_tons", "load_column": "actual_load_tons"}
    """
    from business_brain.discovery.logistics_tracker import compute_vehicle_utilization

    vehicle_col = body.get("vehicle_column")
    capacity_col = body.get("capacity_column")
    load_col = body.get("load_column")

    if not all([vehicle_col, capacity_col, load_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vehicle_column, capacity_column, load_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    vu = compute_vehicle_utilization(rows, vehicle_col, capacity_col, load_col)
    if vu is None:
        return {"error": "Insufficient data"}

    return {
        "mean_utilization": vu.mean_utilization,
        "underloaded_count": vu.underloaded_count,
        "overloaded_count": vu.overloaded_count,
        "summary": vu.summary,
        "vehicles": [
            {
                "vehicle": v.vehicle,
                "total_trips": v.total_trips,
                "avg_load": v.avg_load,
                "avg_capacity": v.avg_capacity,
                "utilization_pct": v.utilization_pct,
                "status": v.status,
            }
            for v in vu.vehicles
        ],
    }


# ---------------------------------------------------------------------------
# Power & Energy Monitoring
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/load-profile")
async def load_profile(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze electrical load profile.

    Body: {"time_column": "timestamp", "power_column": "kw_demand"}
    """
    from business_brain.discovery.power_monitor import analyze_load_profile

    time_col = body.get("time_column")
    power_col = body.get("power_column")

    if not time_col or not power_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "time_column and power_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    lp = analyze_load_profile(rows, time_col, power_col)
    if lp is None:
        return {"error": "Insufficient data"}

    return {
        "peak_demand": lp.peak_demand,
        "avg_demand": lp.avg_demand,
        "min_demand": lp.min_demand,
        "load_factor": lp.load_factor,
        "peak_period": lp.peak_period,
        "off_peak_period": lp.off_peak_period,
        "summary": lp.summary,
        "periods": [
            {
                "period": p.period,
                "demand": p.demand,
                "pct_of_peak": p.pct_of_peak,
                "classification": p.classification,
            }
            for p in lp.periods
        ],
    }


@router.post("/tables/{table_name}/power-factor")
async def power_factor_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze power factor across entities.

    Body: {"entity_column": "transformer", "kw_column": "kw", "kva_column": "kva"}
    """
    from business_brain.discovery.power_monitor import analyze_power_factor

    entity_col = body.get("entity_column")
    kw_col = body.get("kw_column")
    kva_col = body.get("kva_column")

    if not all([entity_col, kw_col, kva_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, kw_column, kva_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    pf = analyze_power_factor(rows, entity_col, kw_col, kva_col)
    if pf is None:
        return {"error": "Insufficient data"}

    return {
        "mean_pf": pf.mean_pf,
        "penalty_risk_count": pf.penalty_risk_count,
        "excellent_count": pf.excellent_count,
        "summary": pf.summary,
        "entities": [
            {
                "entity": e.entity,
                "avg_kw": e.avg_kw,
                "avg_kva": e.avg_kva,
                "power_factor": e.power_factor,
                "status": e.status,
                "estimated_loss_pct": e.estimated_loss_pct,
            }
            for e in pf.entities
        ],
    }


# ---------------------------------------------------------------------------
# Rate Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/rate-comparison")
async def rate_comparison(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compare rates across suppliers.

    Body: {"supplier_column": "supplier", "rate_column": "rate", "item_column": "material"}
    """
    from business_brain.discovery.rate_analysis import compare_rates

    supplier_col = body.get("supplier_column")
    rate_col = body.get("rate_column")
    item_col = body.get("item_column")

    if not supplier_col or not rate_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and rate_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rc = compare_rates(rows, supplier_col, rate_col, item_col)
    if rc is None:
        return {"error": "Insufficient data"}

    return {
        "overall_savings_potential": rc.overall_savings_potential,
        "best_rate_supplier": rc.best_rate_supplier,
        "worst_rate_supplier": rc.worst_rate_supplier,
        "rate_spread_pct": rc.rate_spread_pct,
        "summary": rc.summary,
        "comparisons": [
            {
                "item": c.item,
                "best_supplier": c.best_supplier,
                "worst_supplier": c.worst_supplier,
                "spread": c.spread,
                "spread_pct": c.spread_pct,
                "suppliers": [
                    {"supplier": s.supplier, "avg_rate": s.avg_rate, "volume": s.volume, "total_value": s.total_value}
                    for s in c.suppliers
                ],
            }
            for c in rc.comparisons
        ],
    }


@router.post("/tables/{table_name}/rate-anomalies")
async def rate_anomalies(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Detect rate anomalies.

    Body: {"supplier_column": "supplier", "rate_column": "rate", "item_column": null, "threshold_pct": 20}
    """
    from business_brain.discovery.rate_analysis import detect_rate_anomalies

    supplier_col = body.get("supplier_column")
    rate_col = body.get("rate_column")
    item_col = body.get("item_column")
    threshold = body.get("threshold_pct", 20)

    if not supplier_col or not rate_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "supplier_column and rate_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    anomalies = detect_rate_anomalies(rows, supplier_col, rate_col, item_col, threshold)

    return {
        "anomaly_count": len(anomalies),
        "anomalies": [
            {
                "supplier": a.supplier,
                "item": a.item,
                "rate": a.rate,
                "avg_rate": a.avg_rate,
                "deviation_pct": a.deviation_pct,
                "anomaly_type": a.anomaly_type,
                "severity": a.severity,
            }
            for a in anomalies
        ],
    }


# ---------------------------------------------------------------------------
# Safety & Compliance
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/incidents")
async def incident_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze safety incidents.

    Body: {"type_column": "incident_type", "severity_column": "severity", "date_column": "date", "location_column": "area"}
    """
    from business_brain.discovery.safety_compliance import analyze_incidents

    type_col = body.get("type_column")
    severity_col = body.get("severity_column")
    date_col = body.get("date_column")
    location_col = body.get("location_column")

    if not type_col or not severity_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "type_column and severity_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    inc = analyze_incidents(rows, type_col, severity_col, date_col, location_col)
    if inc is None:
        return {"error": "Insufficient data"}

    return {
        "total_incidents": inc.total_incidents,
        "by_type": inc.by_type,
        "by_severity": inc.by_severity,
        "by_location": inc.by_location,
        "trend": inc.trend,
        "most_common_type": inc.most_common_type,
        "most_common_location": inc.most_common_location,
        "summary": inc.summary,
    }


@router.post("/tables/{table_name}/compliance")
async def compliance_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute compliance rates.

    Body: {"entity_column": "department", "total_checks_column": "total_audits", "passed_checks_column": "passed_audits"}
    """
    from business_brain.discovery.safety_compliance import compliance_rate

    entity_col = body.get("entity_column")
    total_col = body.get("total_checks_column")
    passed_col = body.get("passed_checks_column")

    if not all([entity_col, total_col, passed_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, total_checks_column, passed_checks_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    cr = compliance_rate(rows, entity_col, total_col, passed_col)
    if cr is None:
        return {"error": "Insufficient data"}

    return {
        "mean_compliance": cr.mean_compliance,
        "fully_compliant_count": cr.fully_compliant_count,
        "non_compliant_count": cr.non_compliant_count,
        "summary": cr.summary,
        "entities": [
            {
                "entity": e.entity,
                "total_checks": e.total_checks,
                "passed_checks": e.passed_checks,
                "compliance_pct": e.compliance_pct,
                "status": e.status,
            }
            for e in cr.entities
        ],
    }


@router.post("/tables/{table_name}/risk-matrix")
async def risk_matrix_endpoint(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute risk matrix.

    Body: {"likelihood_column": "probability", "impact_column": "impact", "entity_column": "risk_item"}
    """
    from business_brain.discovery.safety_compliance import risk_matrix

    likelihood_col = body.get("likelihood_column")
    impact_col = body.get("impact_column")
    entity_col = body.get("entity_column")

    if not likelihood_col or not impact_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "likelihood_column and impact_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rm = risk_matrix(rows, likelihood_col, impact_col, entity_col)
    if rm is None:
        return {"error": "Insufficient data"}

    return {
        "critical_count": rm.critical_count,
        "high_count": rm.high_count,
        "medium_count": rm.medium_count,
        "low_count": rm.low_count,
        "summary": rm.summary,
        "items": [
            {
                "entity": i.entity,
                "likelihood": i.likelihood,
                "impact": i.impact,
                "risk_score": i.risk_score,
                "risk_level": i.risk_level,
            }
            for i in rm.items
        ],
    }


# ---------------------------------------------------------------------------
# Quality Control (SPC)
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/process-capability")
async def process_capability(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute process capability indices (Cp/Cpk).

    Body: {"column": "measurement", "lsl": 9.5, "usl": 10.5}
    """
    from business_brain.discovery.quality_control import compute_process_capability

    column = body.get("column")
    lsl = body.get("lsl")
    usl = body.get("usl")

    if not column or lsl is None or usl is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column, lsl, usl required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT "{column}" FROM "{table_name}" LIMIT 10000'))
    values = []
    for row in result.fetchall():
        try:
            values.append(float(str(row[0]).replace(",", "")))
        except (ValueError, TypeError):
            pass

    cap = compute_process_capability(values, float(lsl), float(usl))
    if cap is None:
        return {"error": "Insufficient data (need >=2 values)"}

    return {
        "cp": cap.cp,
        "cpk": cap.cpk,
        "mean": cap.mean,
        "std": cap.std,
        "lsl": cap.lsl,
        "usl": cap.usl,
        "ppm_out_of_spec": cap.ppm_out_of_spec,
        "process_grade": cap.process_grade,
        "centered": cap.centered,
        "summary": cap.summary,
    }


@router.post("/tables/{table_name}/control-chart")
async def control_chart(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Generate control chart data.

    Body: {"column": "measurement", "subgroup_size": 1}
    """
    from business_brain.discovery.quality_control import control_chart_data

    column = body.get("column")
    subgroup_size = body.get("subgroup_size", 1)

    if not column:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT "{column}" FROM "{table_name}" LIMIT 10000'))
    values = []
    for row in result.fetchall():
        try:
            values.append(float(str(row[0]).replace(",", "")))
        except (ValueError, TypeError):
            pass

    cc = control_chart_data(values, subgroup_size)
    if cc is None:
        return {"error": "Insufficient data"}

    return {
        "mean": cc.mean,
        "ucl": cc.ucl,
        "lcl": cc.lcl,
        "out_of_control_count": cc.out_of_control_count,
        "out_of_control_indices": cc.out_of_control_indices,
        "in_control_pct": cc.in_control_pct,
        "summary": cc.summary,
        "values": cc.values[-100:],  # limit to last 100
    }


@router.post("/tables/{table_name}/defects")
async def defect_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze defect rates per entity.

    Body: {"entity_column": "line", "defect_column": "defect_count", "quantity_column": "total_produced"}
    """
    from business_brain.discovery.quality_control import analyze_defects

    entity_col = body.get("entity_column")
    defect_col = body.get("defect_column")
    quantity_col = body.get("quantity_column")

    if not entity_col or not defect_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column and defect_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    df = analyze_defects(rows, entity_col, defect_col, quantity_col)
    if df is None:
        return {"error": "Insufficient data"}

    return {
        "total_defects": df.total_defects,
        "total_quantity": df.total_quantity,
        "overall_defect_rate": df.overall_defect_rate,
        "worst_entity": df.worst_entity,
        "best_entity": df.best_entity,
        "summary": df.summary,
        "entities": [
            {
                "entity": e.entity,
                "defect_count": e.defect_count,
                "quantity": e.quantity,
                "defect_rate": e.defect_rate,
                "dpmo": e.dpmo,
            }
            for e in df.entities
        ],
    }


@router.post("/tables/{table_name}/rejections")
async def rejection_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Analyze rejection rates.

    Body: {"entity_column": "supplier", "accepted_column": "accepted_qty", "rejected_column": "rejected_qty"}
    """
    from business_brain.discovery.quality_control import analyze_rejections

    entity_col = body.get("entity_column")
    accepted_col = body.get("accepted_column")
    rejected_col = body.get("rejected_column")

    if not all([entity_col, accepted_col, rejected_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "entity_column, accepted_column, rejected_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rj = analyze_rejections(rows, entity_col, accepted_col, rejected_col)
    if rj is None:
        return {"error": "Insufficient data"}

    return {
        "total_accepted": rj.total_accepted,
        "total_rejected": rj.total_rejected,
        "overall_rejection_rate": rj.overall_rejection_rate,
        "worst_entity": rj.worst_entity,
        "best_entity": rj.best_entity,
        "summary": rj.summary,
        "entities": [
            {
                "entity": e.entity,
                "accepted": e.accepted,
                "rejected": e.rejected,
                "total": e.total,
                "rejection_rate": e.rejection_rate,
            }
            for e in rj.entities
        ],
    }


# ---------------------------------------------------------------------------
# RFM Customer Analysis
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/rfm")
async def rfm_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute RFM segmentation.

    Body: {"customer_column": "customer", "date_column": "order_date", "amount_column": "amount"}
    """
    from business_brain.discovery.rfm_analysis import compute_rfm, segment_customers

    customer_col = body.get("customer_column")
    date_col = body.get("date_column")
    amount_col = body.get("amount_column")

    if not all([customer_col, date_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, date_column, amount_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    rfm = compute_rfm(rows, customer_col, date_col, amount_col)
    if rfm is None:
        return {"error": "Insufficient data"}

    segments = segment_customers(rfm)

    return {
        "total_customers": rfm.total_customers,
        "avg_recency": rfm.avg_recency,
        "avg_frequency": rfm.avg_frequency,
        "avg_monetary": rfm.avg_monetary,
        "summary": rfm.summary,
        "segment_distribution": rfm.segment_distribution,
        "segments": {k: {"count": len(v), "customers": v[:10]} for k, v in segments.items()},
        "customers": [
            {
                "customer": c.customer,
                "recency_days": c.recency_days,
                "frequency": c.frequency,
                "monetary": c.monetary,
                "r_score": c.r_score,
                "f_score": c.f_score,
                "m_score": c.m_score,
                "rfm_score": c.rfm_score,
                "segment": c.segment,
            }
            for c in rfm.customers[:50]  # limit to 50
        ],
    }


@router.post("/tables/{table_name}/clv")
async def clv_analysis(
    table_name: str,
    body: dict,
    session: AsyncSession = Depends(get_session),
):
    """Compute Customer Lifetime Value.

    Body: {"customer_column": "customer", "amount_column": "amount", "date_column": "order_date"}
    """
    from business_brain.discovery.rfm_analysis import customer_lifetime_value

    customer_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")

    if not all([customer_col, amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, date_column required"}, 400)

    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]

    clv = customer_lifetime_value(rows, customer_col, amount_col, date_col)
    if clv is None:
        return {"error": "Insufficient data"}

    return {
        "avg_clv": clv.avg_clv,
        "top_customers": clv.top_customers,
        "summary": clv.summary,
        "customers": [
            {
                "customer": c.customer,
                "total_spent": c.total_spent,
                "purchase_count": c.purchase_count,
                "avg_purchase": c.avg_purchase,
                "lifespan_days": c.lifespan_days,
                "estimated_clv": c.estimated_clv,
            }
            for c in clv.customers[:50]
        ],
    }


@router.post("/tables/{table_name}/heat-analysis")
async def heat_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze heats: count, weight stats, grade distribution.
    Body: {"heat_column": "...", "weight_column": "...", "grade_column": null, "time_column": null}
    """
    from business_brain.discovery.heat_analysis import analyze_heats
    heat_col = body.get("heat_column")
    weight_col = body.get("weight_column")
    if not all([heat_col, weight_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "heat_column and weight_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_heats(rows, heat_col, weight_col, body.get("grade_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_heats": res.total_heats,
        "total_weight": res.total_weight,
        "avg_weight": res.avg_weight,
        "min_weight": res.min_weight,
        "max_weight": res.max_weight,
        "std_weight": res.std_weight,
        "grade_distribution": res.grade_distribution,
        "period_breakdown": res.period_breakdown,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/chemistry-analysis")
async def chemistry_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze chemical composition per heat.
    Body: {"heat_column": "...", "element_columns": ["C","Mn","Si"], "specs": null}
    """
    from business_brain.discovery.heat_analysis import analyze_chemistry
    heat_col = body.get("heat_column")
    elem_cols = body.get("element_columns")
    if not heat_col or not elem_cols:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "heat_column and element_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    specs_raw = body.get("specs")
    specs = None
    if specs_raw and isinstance(specs_raw, dict):
        specs = {k: tuple(v) for k, v in specs_raw.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    res = analyze_chemistry(rows, heat_col, elem_cols, specs)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_heats": res.total_heats,
        "elements": [
            {
                "element": e.element,
                "mean": e.mean,
                "std": e.std,
                "min": e.min_val,
                "max": e.max_val,
                "in_spec_pct": e.in_spec_pct,
                "out_of_spec_count": e.out_of_spec_count,
            }
            for e in res.elements
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/grade-analysis")
async def grade_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze production by steel grade.
    Body: {"grade_column": "...", "weight_column": "...", "value_column": null}
    """
    from business_brain.discovery.heat_analysis import grade_wise_analysis
    grade_col = body.get("grade_column")
    weight_col = body.get("weight_column")
    if not all([grade_col, weight_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "grade_column and weight_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = grade_wise_analysis(rows, grade_col, weight_col, body.get("value_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_weight": res.total_weight,
        "grades": [
            {
                "grade": g.grade,
                "heat_count": g.heat_count,
                "total_weight": g.total_weight,
                "pct_of_total": g.pct_of_total,
                "avg_weight": g.avg_weight,
                "total_value": g.total_value,
            }
            for g in res.grades
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/grade-anomalies")
async def grade_anomalies(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect heats where chemistry doesn't match grade spec.
    Body: {"heat_column": "...", "grade_column": "...", "element_columns": ["C","Mn"], "specs": null}
    """
    from business_brain.discovery.heat_analysis import detect_grade_anomalies
    heat_col = body.get("heat_column")
    grade_col = body.get("grade_column")
    elem_cols = body.get("element_columns")
    if not all([heat_col, grade_col, elem_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "heat_column, grade_column, element_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    specs_raw = body.get("specs")
    specs = None
    if specs_raw and isinstance(specs_raw, dict):
        specs = {}
        for grade, elems in specs_raw.items():
            if isinstance(elems, dict):
                specs[grade] = {e: tuple(v) for e, v in elems.items() if isinstance(v, (list, tuple)) and len(v) == 2}
    anomalies = detect_grade_anomalies(rows, heat_col, grade_col, elem_cols, specs)
    return {
        "anomaly_count": len(anomalies),
        "anomalies": [
            {
                "heat": a.heat,
                "grade": a.grade,
                "element": a.element,
                "value": a.value,
                "spec_range": list(a.spec_range),
                "deviation_pct": a.deviation_pct,
            }
            for a in anomalies[:100]
        ],
    }


@router.post("/tables/{table_name}/gate-traffic")
async def gate_traffic(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze gate traffic patterns.
    Body: {"time_column": "...", "vehicle_column": null, "direction_column": null}
    """
    from business_brain.discovery.dispatch_gate import analyze_gate_traffic
    time_col = body.get("time_column")
    if not time_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "time_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_gate_traffic(rows, time_col, body.get("vehicle_column"), body.get("direction_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_vehicles": res.total_vehicles,
        "periods": [{"period": p.period, "vehicle_count": p.vehicle_count, "pct_of_total": p.pct_of_total} for p in res.periods],
        "peak_period": res.peak_period,
        "off_peak_period": res.off_peak_period,
        "avg_per_period": res.avg_per_period,
        "direction_split": res.direction_split,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/weighbridge")
async def weighbridge(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze weighbridge data.
    Body: {"vehicle_column": "...", "gross_column": "...", "tare_column": "...", "material_column": null}
    """
    from business_brain.discovery.dispatch_gate import weighbridge_analysis
    vehicle_col = body.get("vehicle_column")
    gross_col = body.get("gross_column")
    tare_col = body.get("tare_column")
    if not all([vehicle_col, gross_col, tare_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vehicle_column, gross_column, tare_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = weighbridge_analysis(rows, vehicle_col, gross_col, tare_col, body.get("material_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_net_weight": res.total_net_weight,
        "avg_net_weight": res.avg_net_weight,
        "total_vehicles": res.total_vehicles,
        "by_material": res.by_material,
        "entries": [
            {"vehicle": e.vehicle, "gross_weight": e.gross_weight, "tare_weight": e.tare_weight, "net_weight": e.net_weight, "material": e.material}
            for e in res.entries[:50]
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/material-movement")
async def material_movement(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Track inward/outward material movement.
    Body: {"material_column": "...", "quantity_column": "...", "direction_column": "..."}
    """
    from business_brain.discovery.dispatch_gate import track_material_movement
    mat_col = body.get("material_column")
    qty_col = body.get("quantity_column")
    dir_col = body.get("direction_column")
    if not all([mat_col, qty_col, dir_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "material_column, quantity_column, direction_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = track_material_movement(rows, mat_col, qty_col, dir_col, body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_inward": res.total_inward,
        "total_outward": res.total_outward,
        "net_movement": res.net_movement,
        "materials": [
            {"material": m.material, "inward_qty": m.inward_qty, "outward_qty": m.outward_qty, "net_qty": m.net_qty, "movement_count": m.movement_count}
            for m in res.materials
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/dispatch-anomalies")
async def dispatch_anomalies(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Flag vehicles with unusual dispatch weights.
    Body: {"vehicle_column": "...", "weight_column": "...", "expected_min": null, "expected_max": null}
    """
    from business_brain.discovery.dispatch_gate import detect_dispatch_anomalies
    vehicle_col = body.get("vehicle_column")
    weight_col = body.get("weight_column")
    if not all([vehicle_col, weight_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vehicle_column and weight_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    anomalies = detect_dispatch_anomalies(rows, vehicle_col, weight_col, body.get("expected_min"), body.get("expected_max"))
    return {
        "anomaly_count": len(anomalies),
        "anomalies": [
            {"vehicle": a.vehicle, "weight": a.weight, "expected_range": list(a.expected_range), "deviation_pct": a.deviation_pct, "anomaly_type": a.anomaly_type}
            for a in anomalies[:100]
        ],
    }


@router.post("/tables/{table_name}/emissions")
async def emissions_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze emissions data.
    Body: {"source_column": "...", "pollutant_column": "...", "value_column": "...", "limit_column": null, "time_column": null}
    """
    from business_brain.discovery.environmental_monitor import analyze_emissions
    src_col = body.get("source_column")
    poll_col = body.get("pollutant_column")
    val_col = body.get("value_column")
    if not all([src_col, poll_col, val_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "source_column, pollutant_column, value_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_emissions(rows, src_col, poll_col, val_col, body.get("limit_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_readings": res.total_readings,
        "sources": [
            {"source": s.source, "pollutant": s.pollutant, "total": s.total, "average": s.average, "max_value": s.max_value, "count": s.count, "compliance_pct": s.compliance_pct, "trend": s.trend}
            for s in res.sources
        ],
        "exceedances": [
            {"source": e.source, "pollutant": e.pollutant, "value": e.value, "limit": e.limit, "excess_pct": e.excess_pct}
            for e in res.exceedances[:50]
        ],
        "overall_compliance_pct": res.overall_compliance_pct,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/waste-analysis")
async def waste_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze waste generation.
    Body: {"waste_type_column": "...", "quantity_column": "...", "disposal_column": null}
    """
    from business_brain.discovery.environmental_monitor import analyze_waste_generation
    wtype_col = body.get("waste_type_column")
    qty_col = body.get("quantity_column")
    if not all([wtype_col, qty_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "waste_type_column and quantity_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_waste_generation(rows, wtype_col, qty_col, body.get("disposal_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_waste": res.total_waste,
        "types": [
            {"waste_type": t.waste_type, "quantity": t.quantity, "pct_of_total": t.pct_of_total, "disposal_breakdown": t.disposal_breakdown}
            for t in res.types
        ],
        "recycling_rate": res.recycling_rate,
        "diversion_rate": res.diversion_rate,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/water-usage")
async def water_usage(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze water usage.
    Body: {"source_column": "...", "consumption_column": "...", "discharge_column": null}
    """
    from business_brain.discovery.environmental_monitor import analyze_water_usage
    src_col = body.get("source_column")
    cons_col = body.get("consumption_column")
    if not all([src_col, cons_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "source_column and consumption_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_water_usage(rows, src_col, cons_col, body.get("discharge_column"), body.get("time_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_consumption": res.total_consumption,
        "total_discharge": res.total_discharge,
        "net_consumption": res.net_consumption,
        "recycling_ratio": res.recycling_ratio,
        "sources": [
            {"source": s.source, "consumption": s.consumption, "discharge": s.discharge, "pct_of_total": s.pct_of_total}
            for s in res.sources
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/maintenance-history")
async def maintenance_history(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze equipment maintenance history.
    Body: {"equipment_column": "...", "date_column": "...", "type_column": "...", "duration_column": null, "cost_column": null}
    """
    from business_brain.discovery.maintenance_scheduler import analyze_maintenance_history
    equip_col = body.get("equipment_column")
    date_col = body.get("date_column")
    type_col = body.get("type_column")
    if not all([equip_col, date_col, type_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "equipment_column, date_column, type_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_maintenance_history(rows, equip_col, date_col, type_col, body.get("duration_column"), body.get("cost_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_events": res.total_events,
        "equipment": [
            {"equipment": e.equipment, "event_count": e.event_count, "type_breakdown": e.type_breakdown, "corrective_ratio": e.corrective_ratio, "total_downtime": e.total_downtime, "total_cost": e.total_cost, "avg_cost": e.avg_cost}
            for e in res.equipment
        ],
        "overall_corrective_ratio": res.overall_corrective_ratio,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/mtbf-mttr")
async def mtbf_mttr(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute MTBF and MTTR for equipment.
    Body: {"equipment_column": "...", "date_column": "...", "type_column": "...", "duration_column": "..."}
    """
    from business_brain.discovery.maintenance_scheduler import compute_mtbf_mttr
    equip_col = body.get("equipment_column")
    date_col = body.get("date_column")
    type_col = body.get("type_column")
    dur_col = body.get("duration_column")
    if not all([equip_col, date_col, type_col, dur_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "equipment_column, date_column, type_column, duration_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    metrics = compute_mtbf_mttr(rows, equip_col, date_col, type_col, dur_col)
    return {
        "metrics": [
            {"equipment": m.equipment, "mtbf_days": m.mtbf_days, "mttr_hours": m.mttr_hours, "failure_count": m.failure_count, "availability_pct": m.availability_pct}
            for m in metrics
        ],
    }


@router.post("/tables/{table_name}/spare-parts")
async def spare_parts(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze spare parts consumption.
    Body: {"part_column": "...", "quantity_column": "...", "cost_column": null, "equipment_column": null}
    """
    from business_brain.discovery.maintenance_scheduler import analyze_spare_parts
    part_col = body.get("part_column")
    qty_col = body.get("quantity_column")
    if not all([part_col, qty_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "part_column and quantity_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_spare_parts(rows, part_col, qty_col, body.get("cost_column"), body.get("equipment_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_parts": res.total_parts,
        "total_quantity": res.total_quantity,
        "total_cost": res.total_cost,
        "parts": [
            {"part": p.part, "total_quantity": p.total_quantity, "total_cost": p.total_cost, "abc_class": p.abc_class, "equipment_list": p.equipment_list}
            for p in res.parts[:50]
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/maintenance-schedule")
async def maintenance_schedule(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Generate predicted maintenance schedule.
    Body: {"equipment_column": "...", "date_column": "...", "type_column": "...", "interval_days": null}
    """
    from business_brain.discovery.maintenance_scheduler import generate_maintenance_schedule
    equip_col = body.get("equipment_column")
    date_col = body.get("date_column")
    type_col = body.get("type_column")
    if not all([equip_col, date_col, type_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "equipment_column, date_column, type_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    schedule = generate_maintenance_schedule(rows, equip_col, date_col, type_col, body.get("interval_days"))
    return {
        "schedule": [
            {"equipment": s.equipment, "last_maintenance": s.last_maintenance, "avg_interval_days": s.avg_interval_days, "next_maintenance_date": s.next_maintenance_date, "days_until_next": s.days_until_next, "overdue": s.overdue, "priority": s.priority}
            for s in schedule
        ],
    }


@router.post("/tables/{table_name}/contracts")
async def contracts_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze contract portfolio.
    Body: {"contract_column": "...", "vendor_column": "...", "value_column": "...", "start_column": null, "end_column": null, "status_column": null}
    """
    from business_brain.discovery.contract_analyzer import analyze_contracts
    contract_col = body.get("contract_column")
    vendor_col = body.get("vendor_column")
    value_col = body.get("value_column")
    if not all([contract_col, vendor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "contract_column, vendor_column, value_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_contracts(rows, contract_col, vendor_col, value_col, body.get("start_column"), body.get("end_column"), body.get("status_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_contracts": res.total_contracts,
        "total_value": res.total_value,
        "avg_value": res.avg_value,
        "vendor_count": res.vendor_count,
        "vendors": [
            {"vendor": v.vendor, "contract_count": v.contract_count, "total_value": v.total_value, "avg_value": v.avg_value}
            for v in res.vendors[:50]
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/expiring-contracts")
async def expiring_contracts(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Find contracts expiring soon.
    Body: {"contract_column": "...", "end_column": "...", "vendor_column": null, "value_column": null, "horizon_days": 90}
    """
    from business_brain.discovery.contract_analyzer import detect_expiring_contracts
    contract_col = body.get("contract_column")
    end_col = body.get("end_column")
    if not all([contract_col, end_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "contract_column and end_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    horizon = body.get("horizon_days", 90)
    expiring = detect_expiring_contracts(rows, contract_col, end_col, body.get("vendor_column"), body.get("value_column"), horizon_days=horizon)
    return {
        "expiring_count": len(expiring),
        "contracts": [
            {"contract": c.contract, "vendor": c.vendor, "end_date": c.end_date, "days_remaining": c.days_remaining, "value": c.value, "urgency": c.urgency}
            for c in expiring
        ],
    }


@router.post("/tables/{table_name}/vendor-concentration")
async def vendor_concentration(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Measure vendor concentration (HHI).
    Body: {"vendor_column": "...", "value_column": "..."}
    """
    from business_brain.discovery.contract_analyzer import vendor_contract_concentration
    vendor_col = body.get("vendor_column")
    value_col = body.get("value_column")
    if not all([vendor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vendor_column and value_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = vendor_contract_concentration(rows, vendor_col, value_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "hhi": res.hhi,
        "risk_rating": res.risk_rating,
        "vendor_count": res.vendor_count,
        "shares": [
            {"vendor": s.vendor, "share_pct": s.share_pct, "total_value": s.total_value}
            for s in res.shares
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/renewal-patterns")
async def renewal_patterns(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect contract renewal patterns.
    Body: {"contract_column": "...", "vendor_column": "...", "start_column": "...", "end_column": "...", "value_column": null}
    """
    from business_brain.discovery.contract_analyzer import analyze_renewal_patterns
    contract_col = body.get("contract_column")
    vendor_col = body.get("vendor_column")
    start_col = body.get("start_column")
    end_col = body.get("end_column")
    if not all([contract_col, vendor_col, start_col, end_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "contract_column, vendor_column, start_column, end_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_renewal_patterns(rows, contract_col, vendor_col, start_col, end_col, body.get("value_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_vendors": res.total_vendors,
        "vendors_with_renewals": res.vendors_with_renewals,
        "renewal_rate": res.renewal_rate,
        "renewals": [
            {"vendor": r.vendor, "renewal_count": r.renewal_count, "avg_contract_duration_days": r.avg_contract_duration_days, "value_trend": r.value_trend}
            for r in res.renewals
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/budget-vs-actual")
async def budget_vs_actual_endpoint(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Budget vs actual analysis.
    Body: {"category_column": "...", "budget_column": "...", "actual_column": "...", "period_column": null}
    """
    from business_brain.discovery.budget_tracker import budget_vs_actual
    cat_col = body.get("category_column")
    bud_col = body.get("budget_column")
    act_col = body.get("actual_column")
    if not all([cat_col, bud_col, act_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column, budget_column, actual_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = budget_vs_actual(rows, cat_col, bud_col, act_col, body.get("period_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_budget": res.total_budget,
        "total_actual": res.total_actual,
        "overall_variance_pct": res.overall_variance_pct,
        "categories": [
            {"category": c.category, "budget": c.budget, "actual": c.actual, "variance": c.variance, "variance_pct": c.variance_pct, "over_budget": c.over_budget}
            for c in res.categories
        ],
        "over_budget_count": res.over_budget_count,
        "periods": [
            {"period": p.period, "budget": p.budget, "actual": p.actual, "variance_pct": p.variance_pct}
            for p in (res.periods or [])
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/burn-rate")
async def burn_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute budget burn rate.
    Body: {"amount_column": "...", "date_column": "...", "total_budget": null}
    """
    from business_brain.discovery.budget_tracker import compute_burn_rate
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_burn_rate(rows, amount_col, date_col, body.get("total_budget"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_spent": res.total_spent,
        "days_elapsed": res.days_elapsed,
        "daily_burn": res.daily_burn,
        "monthly_burn": res.monthly_burn,
        "remaining_budget": res.remaining_budget,
        "days_until_exhaustion": res.days_until_exhaustion,
        "projected_end_date": res.projected_end_date,
        "trend": res.trend,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/spending-patterns")
async def spending_patterns(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze spending patterns.
    Body: {"category_column": "...", "amount_column": "...", "date_column": null, "vendor_column": null}
    """
    from business_brain.discovery.budget_tracker import analyze_spending_patterns
    cat_col = body.get("category_column")
    amt_col = body.get("amount_column")
    if not all([cat_col, amt_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column and amount_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_spending_patterns(rows, cat_col, amt_col, body.get("date_column"), body.get("vendor_column"))
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_spend": res.total_spend,
        "categories": [
            {"category": c.category, "amount": c.amount, "pct_of_total": c.pct_of_total}
            for c in res.categories
        ],
        "top_categories": res.top_categories,
        "vendors": [
            {"vendor": v.vendor, "amount": v.amount, "pct_of_total": v.pct_of_total}
            for v in (res.vendors or [])
        ] if res.vendors else None,
        "month_changes": [
            {"category": m.category, "month": m.month, "amount": m.amount, "prev_amount": m.prev_amount, "change_pct": m.change_pct}
            for m in (res.month_changes or [])
        ] if res.month_changes else None,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/budget-forecast")
async def budget_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast future budget periods.
    Body: {"amount_column": "...", "date_column": "...", "periods_ahead": 3}
    """
    from business_brain.discovery.budget_tracker import forecast_budget
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    periods = body.get("periods_ahead", 3)
    forecasts = forecast_budget(rows, amount_col, date_col, periods)
    return {
        "forecasts": [
            {"period": f.period, "projected_amount": f.projected_amount, "cumulative": f.cumulative, "confidence": f.confidence}
            for f in forecasts
        ],
    }


@router.post("/tables/{table_name}/demand-pattern")
async def demand_pattern(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze demand patterns.
    Body: {"product_column": "...", "quantity_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.demand_forecaster import analyze_demand_pattern
    prod_col = body.get("product_column")
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([prod_col, qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "product_column, quantity_column, date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_demand_pattern(rows, prod_col, qty_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "products": [
            {"product": p.product, "total_demand": p.total_demand, "avg_demand": p.avg_demand, "max_demand": p.max_demand, "min_demand": p.min_demand, "cv": p.cv, "pattern": p.pattern, "trend": p.trend, "adi": p.adi, "periods": p.periods}
            for p in res.products
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/demand-moving-avg")
async def demand_moving_avg(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute moving average on demand data.
    Body: {"quantity_column": "...", "date_column": "...", "window": 3}
    """
    from business_brain.discovery.demand_forecaster import compute_moving_average
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "quantity_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    window = body.get("window", 3)
    res = compute_moving_average(rows, qty_col, date_col, window)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "window": res.window,
        "points": [
            {"period": p.period, "actual": p.actual, "moving_avg": p.moving_avg}
            for p in res.points
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/demand-smoothing")
async def demand_smoothing(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Exponential smoothing on demand data.
    Body: {"quantity_column": "...", "date_column": "...", "alpha": 0.3}
    """
    from business_brain.discovery.demand_forecaster import exponential_smoothing
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "quantity_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    alpha = body.get("alpha", 0.3)
    res = exponential_smoothing(rows, qty_col, date_col, alpha)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "alpha_used": res.alpha_used,
        "optimal_alpha": res.optimal_alpha,
        "mad": res.mad,
        "points": [
            {"period": p.period, "actual": p.actual, "forecast": p.forecast}
            for p in res.points
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/demand-seasonality")
async def demand_seasonality(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect seasonal patterns in demand.
    Body: {"quantity_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.demand_forecaster import detect_demand_seasonality
    qty_col = body.get("quantity_column")
    date_col = body.get("date_column")
    if not all([qty_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "quantity_column and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = detect_demand_seasonality(rows, qty_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "is_seasonal": res.is_seasonal,
        "peak_seasons": res.peak_seasons,
        "low_seasons": res.low_seasons,
        "periods": [
            {"period": p.period, "avg_demand": p.avg_demand, "seasonal_index": p.seasonal_index}
            for p in res.periods
        ],
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Procurement Analytics endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/purchase-orders")
async def purchase_orders(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze purchase orders by vendor with monthly trends.
    Body: {"po_column": "...", "vendor_column": "...", "amount_column": "...", "date_column": "...", "status_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import analyze_purchase_orders
    po_col = body.get("po_column")
    vendor_col = body.get("vendor_column")
    amount_col = body.get("amount_column")
    if not all([po_col, vendor_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "po_column, vendor_column, and amount_column required"}, 400)
    date_col = body.get("date_column")
    status_col = body.get("status_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_purchase_orders(rows, po_col, vendor_col, amount_col, date_col, status_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_orders": res.total_orders,
        "total_value": res.total_value,
        "vendors": [
            {"vendor": v.vendor, "order_count": v.order_count, "total_value": v.total_value, "avg_value": v.avg_value}
            for v in res.vendors
        ],
        "monthly_trends": [
            {"month": m.month, "order_count": m.order_count, "total_value": m.total_value}
            for m in res.monthly_trends
        ],
        "status_breakdown": res.status_breakdown,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/purchase-price-variance")
async def purchase_price_variance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute purchase price variance per item.
    Body: {"item_column": "...", "quantity_column": "...", "actual_price_column": "...", "standard_price_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import compute_purchase_price_variance
    item_col = body.get("item_column")
    qty_col = body.get("quantity_column")
    actual_col = body.get("actual_price_column")
    if not all([item_col, qty_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "item_column, quantity_column, and actual_price_column required"}, 400)
    std_col = body.get("standard_price_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_purchase_price_variance(rows, item_col, qty_col, actual_col, std_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "items": [
            {"item": it.item, "quantity": it.quantity, "avg_actual_price": it.avg_actual_price, "standard_price": it.standard_price, "total_variance": it.total_variance, "variance_type": it.variance_type}
            for it in res.items
        ],
        "total_variance": res.total_variance,
        "favorable_count": res.favorable_count,
        "unfavorable_count": res.unfavorable_count,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/vendor-performance")
async def vendor_performance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze vendor delivery performance.
    Body: {"vendor_column": "...", "delivery_date_column": "...", "promised_date_column": "...", "quality_column": "...", "quantity_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import analyze_vendor_performance
    vendor_col = body.get("vendor_column")
    delivery_col = body.get("delivery_date_column")
    promised_col = body.get("promised_date_column")
    if not all([vendor_col, delivery_col, promised_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vendor_column, delivery_date_column, and promised_date_column required"}, 400)
    quality_col = body.get("quality_column")
    qty_col = body.get("quantity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_vendor_performance(rows, vendor_col, delivery_col, promised_col, quality_col, qty_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "vendors": [
            {"vendor": v.vendor, "total_deliveries": v.total_deliveries, "on_time_count": v.on_time_count, "late_count": v.late_count, "on_time_pct": v.on_time_pct, "avg_days_late": v.avg_days_late, "avg_quality": v.avg_quality, "total_quantity": v.total_quantity}
            for v in res.vendors
        ],
        "overall_on_time_pct": res.overall_on_time_pct,
        "avg_quality": res.avg_quality,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/spend-by-category")
async def spend_by_category(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze procurement spend by category.
    Body: {"category_column": "...", "amount_column": "...", "vendor_column": "..."}
    """
    from business_brain.discovery.procurement_analytics import analyze_spend_by_category
    cat_col = body.get("category_column")
    amount_col = body.get("amount_column")
    if not all([cat_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "category_column and amount_column required"}, 400)
    vendor_col = body.get("vendor_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_spend_by_category(rows, cat_col, amount_col, vendor_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "categories": [
            {"category": c.category, "total_spend": c.total_spend, "pct_of_total": c.pct_of_total, "transaction_count": c.transaction_count, "vendor_count": c.vendor_count}
            for c in res.categories
        ],
        "total_spend": res.total_spend,
        "hhi": res.hhi,
        "concentration_risk": res.concentration_risk,
        "top_categories": res.top_categories,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Financial Ratios endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/profitability-ratios")
async def profitability_ratios(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute profitability ratios from revenue and cost data.
    Body: {"revenue_column": "...", "cost_column": "...", "entity_column": "..."}
    """
    from business_brain.discovery.financial_ratios import compute_profitability_ratios
    rev_col = body.get("revenue_column")
    cost_col = body.get("cost_column")
    if not all([rev_col, cost_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and cost_column required"}, 400)
    entity_col = body.get("entity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_profitability_ratios(rows, rev_col, cost_col, entity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "entities": [
            {"entity": e.entity, "revenue": e.revenue, "cost": e.cost, "gross_profit": e.gross_profit, "gross_margin_pct": e.gross_margin_pct, "cost_ratio": e.cost_ratio}
            for e in res.entities
        ],
        "total_revenue": res.total_revenue,
        "total_cost": res.total_cost,
        "overall_margin": res.overall_margin,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/liquidity-ratios")
async def liquidity_ratios(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute liquidity ratios from balance sheet data.
    Body: {"current_assets_column": "...", "current_liabilities_column": "...", "cash_column": "...", "inventory_column": "...", "entity_column": "..."}
    """
    from business_brain.discovery.financial_ratios import compute_liquidity_ratios
    ca_col = body.get("current_assets_column")
    cl_col = body.get("current_liabilities_column")
    if not all([ca_col, cl_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "current_assets_column and current_liabilities_column required"}, 400)
    cash_col = body.get("cash_column")
    inv_col = body.get("inventory_column")
    entity_col = body.get("entity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_liquidity_ratios(rows, ca_col, cl_col, cash_col, inv_col, entity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "entities": [
            {"entity": e.entity, "current_assets": e.current_assets, "current_liabilities": e.current_liabilities, "current_ratio": e.current_ratio, "quick_ratio": e.quick_ratio, "cash_ratio": e.cash_ratio, "rating": e.rating}
            for e in res.entities
        ],
        "avg_current_ratio": res.avg_current_ratio,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/efficiency-ratios")
async def efficiency_ratios(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute efficiency ratios from financial data.
    Body: {"revenue_column": "...", "assets_column": "...", "receivables_column": "...", "payables_column": "...", "cogs_column": "..."}
    """
    from business_brain.discovery.financial_ratios import compute_efficiency_ratios
    rev_col = body.get("revenue_column")
    assets_col = body.get("assets_column")
    if not all([rev_col, assets_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and assets_column required"}, 400)
    recv_col = body.get("receivables_column")
    pay_col = body.get("payables_column")
    cogs_col = body.get("cogs_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_efficiency_ratios(rows, rev_col, assets_col, recv_col, pay_col, cogs_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "ratios": {
            "asset_turnover": res.ratios.asset_turnover,
            "receivables_turnover": res.ratios.receivables_turnover,
            "dso": res.ratios.dso,
            "payables_turnover": res.ratios.payables_turnover,
            "dpo": res.ratios.dpo,
            "cash_conversion_cycle": res.ratios.cash_conversion_cycle,
        },
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/financial-trends")
async def financial_trends(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze financial trends over periods.
    Body: {"metric_column": "...", "period_column": "...", "entity_column": "..."}
    """
    from business_brain.discovery.financial_ratios import analyze_financial_trends
    metric_col = body.get("metric_column")
    period_col = body.get("period_column")
    if not all([metric_col, period_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "metric_column and period_column required"}, 400)
    entity_col = body.get("entity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_financial_trends(rows, metric_col, period_col, entity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "entities": [
            {
                "entity": e.entity,
                "periods": [
                    {"period": p.period, "value": p.value, "change": p.change, "change_pct": p.change_pct}
                    for p in e.periods
                ],
                "trend": e.trend,
                "cagr": e.cagr,
                "best_period": e.best_period,
                "worst_period": e.worst_period,
            }
            for e in res.entities
        ],
        "overall_trend": res.overall_trend,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Project Tracker endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/project-status")
async def project_status(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze project status, dates, and budget.
    Body: {"project_column": "...", "status_column": "...", "start_column": "...", "end_column": "...", "budget_column": "..."}
    """
    from business_brain.discovery.project_tracker import analyze_project_status
    proj_col = body.get("project_column")
    status_col = body.get("status_column")
    if not all([proj_col, status_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "project_column and status_column required"}, 400)
    start_col = body.get("start_column")
    end_col = body.get("end_column")
    budget_col = body.get("budget_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_project_status(rows, proj_col, status_col, start_col, end_col, budget_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "projects": [
            {"project": p.project, "status": p.status, "start_date": p.start_date, "end_date": p.end_date, "budget": p.budget, "duration_days": p.duration_days}
            for p in res.projects
        ],
        "status_distribution": res.status_distribution,
        "completion_rate": res.completion_rate,
        "avg_duration_days": res.avg_duration_days,
        "total_budget": res.total_budget,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/milestones")
async def milestones(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze milestone progress and health.
    Body: {"project_column": "...", "milestone_column": "...", "due_date_column": "...", "completion_date_column": "...", "status_column": "..."}
    """
    from business_brain.discovery.project_tracker import analyze_milestones
    proj_col = body.get("project_column")
    ms_col = body.get("milestone_column")
    due_col = body.get("due_date_column")
    if not all([proj_col, ms_col, due_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "project_column, milestone_column, and due_date_column required"}, 400)
    comp_col = body.get("completion_date_column")
    status_col = body.get("status_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_milestones(rows, proj_col, ms_col, due_col, comp_col, status_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "projects": [
            {"project": p.project, "total": p.total, "completed": p.completed, "overdue": p.overdue, "on_time_pct": p.on_time_pct, "avg_delay_days": p.avg_delay_days}
            for p in res.projects
        ],
        "total_milestones": res.total_milestones,
        "overall_on_time_pct": res.overall_on_time_pct,
        "health": res.health,
        "upcoming_count": res.upcoming_count,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/resource-allocation")
async def resource_allocation(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze resource allocation across projects.
    Body: {"resource_column": "...", "project_column": "...", "hours_column": "...", "role_column": "..."}
    """
    from business_brain.discovery.project_tracker import analyze_resource_allocation
    resource_col = body.get("resource_column")
    proj_col = body.get("project_column")
    hours_col = body.get("hours_column")
    if not all([resource_col, proj_col, hours_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "resource_column, project_column, and hours_column required"}, 400)
    role_col = body.get("role_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_resource_allocation(rows, resource_col, proj_col, hours_col, role_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "resources": [
            {"resource": r.resource, "total_hours": r.total_hours, "project_count": r.project_count, "avg_hours_per_project": r.avg_hours_per_project, "utilization_status": r.utilization_status}
            for r in res.resources
        ],
        "by_role": [
            {"role": rh.role, "total_hours": rh.total_hours, "resource_count": rh.resource_count}
            for rh in res.by_role
        ] if res.by_role is not None else None,
        "over_allocated": res.over_allocated,
        "under_utilized": res.under_utilized,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/project-health")
async def project_health(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute project health by comparing planned vs actual.
    Body: {"project_column": "...", "planned_column": "...", "actual_column": "...", "metric_type": "cost"}
    """
    from business_brain.discovery.project_tracker import compute_project_health
    proj_col = body.get("project_column")
    planned_col = body.get("planned_column")
    actual_col = body.get("actual_column")
    if not all([proj_col, planned_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "project_column, planned_column, and actual_column required"}, 400)
    metric_type = body.get("metric_type", "cost")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_project_health(rows, proj_col, planned_col, actual_col, metric_type)
    if not res:
        return {"error": "Insufficient data"}
    return {
        "projects": [
            {"project": p.project, "planned": p.planned, "actual": p.actual, "variance": p.variance, "variance_pct": p.variance_pct, "performance_index": p.performance_index, "health": p.health}
            for p in res
        ],
    }


# ---------------------------------------------------------------------------
# SCADA Analyzer endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/sensor-readings")
async def sensor_readings(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze sensor readings with per-sensor statistics.
    Body: {"sensor_column": "...", "value_column": "...", "timestamp_column": "...", "unit_column": "..."}
    """
    from business_brain.discovery.scada_analyzer import analyze_sensor_readings
    sensor_col = body.get("sensor_column")
    value_col = body.get("value_column")
    if not all([sensor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "sensor_column and value_column required"}, 400)
    ts_col = body.get("timestamp_column")
    unit_col = body.get("unit_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sensor_readings(rows, sensor_col, value_col, ts_col, unit_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "sensors": [
            {"sensor": s.sensor, "min_val": s.min_val, "max_val": s.max_val, "mean": s.mean, "std": s.std, "reading_count": s.reading_count, "stability_index": s.stability_index, "unit": s.unit}
            for s in res.sensors
        ],
        "stable_count": res.stable_count,
        "unstable_count": res.unstable_count,
        "total_readings": res.total_readings,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/sensor-anomalies")
async def sensor_anomalies(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Detect anomalies in sensor readings.
    Body: {"sensor_column": "...", "value_column": "...", "low_limit": 0.0, "high_limit": 100.0}
    """
    from business_brain.discovery.scada_analyzer import detect_sensor_anomalies
    sensor_col = body.get("sensor_column")
    value_col = body.get("value_column")
    if not all([sensor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "sensor_column and value_column required"}, 400)
    low_limit = body.get("low_limit")
    high_limit = body.get("high_limit")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = detect_sensor_anomalies(rows, sensor_col, value_col, low_limit, high_limit)
    return {
        "anomalies": [
            {"sensor": a.sensor, "value": a.value, "expected_range": list(a.expected_range), "anomaly_type": a.anomaly_type, "index": a.index}
            for a in res
        ],
        "total_anomalies": len(res),
    }


@router.post("/tables/{table_name}/process-stability")
async def process_stability(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute process capability (Cp, Cpk) per sensor.
    Body: {"sensor_column": "...", "value_column": "...", "target_column": "..."}
    """
    from business_brain.discovery.scada_analyzer import compute_process_stability
    sensor_col = body.get("sensor_column")
    value_col = body.get("value_column")
    if not all([sensor_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "sensor_column and value_column required"}, 400)
    target_col = body.get("target_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_process_stability(rows, sensor_col, value_col, target_col)
    if not res:
        return {"error": "Insufficient data"}
    return {
        "sensors": [
            {"sensor": s.sensor, "mean": s.mean, "std": s.std, "usl": s.usl, "lsl": s.lsl, "cp": s.cp, "cpk": s.cpk, "rating": s.rating}
            for s in res
        ],
    }


@router.post("/tables/{table_name}/alarm-frequency")
async def alarm_frequency(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze alarm frequency and chattering detection.
    Body: {"alarm_column": "...", "severity_column": "...", "timestamp_column": "...", "equipment_column": "..."}
    """
    from business_brain.discovery.scada_analyzer import analyze_alarm_frequency
    alarm_col = body.get("alarm_column")
    if not alarm_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "alarm_column required"}, 400)
    severity_col = body.get("severity_column")
    ts_col = body.get("timestamp_column")
    equip_col = body.get("equipment_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_alarm_frequency(rows, alarm_col, severity_col, ts_col, equip_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_alarms": res.total_alarms,
        "by_severity": res.by_severity,
        "by_equipment": [
            {"equipment": e.equipment, "alarm_count": e.alarm_count, "critical_count": e.critical_count}
            for e in res.by_equipment
        ],
        "top_alarms": [
            {"alarm": a.alarm, "count": a.count, "pct_of_total": a.pct_of_total}
            for a in res.top_alarms
        ],
        "chattering_alarms": res.chattering_alarms,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Customer Analytics endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/customer-segments")
async def customer_segments(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Segment customers into revenue tiers.
    Body: {"customer_column": "...", "revenue_column": "...", "frequency_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.customer_analytics import analyze_customer_segments
    cust_col = body.get("customer_column")
    rev_col = body.get("revenue_column")
    if not all([cust_col, rev_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column and revenue_column required"}, 400)
    freq_col = body.get("frequency_column")
    cat_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_customer_segments(rows, cust_col, rev_col, freq_col, cat_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "tiers": [
            {"tier": t.tier, "customer_count": t.customer_count, "total_revenue": t.total_revenue, "avg_revenue": t.avg_revenue, "pct_of_total": t.pct_of_total, "avg_frequency": t.avg_frequency}
            for t in res.tiers
        ],
        "total_customers": res.total_customers,
        "total_revenue": res.total_revenue,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/churn-risk")
async def churn_risk(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze customer churn risk based on recency.
    Body: {"customer_column": "...", "date_column": "...", "amount_column": "..."}
    """
    from business_brain.discovery.customer_analytics import analyze_churn_risk
    cust_col = body.get("customer_column")
    date_col = body.get("date_column")
    if not all([cust_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column and date_column required"}, 400)
    amount_col = body.get("amount_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_churn_risk(rows, cust_col, date_col, amount_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "statuses": [
            {"status": s.status, "customer_count": s.customer_count, "pct": s.pct, "avg_spend": s.avg_spend}
            for s in res.statuses
        ],
        "total_customers": res.total_customers,
        "churn_rate": res.churn_rate,
        "at_risk_rate": res.at_risk_rate,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/customer-concentration")
async def customer_concentration(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute customer revenue concentration and Pareto analysis.
    Body: {"customer_column": "...", "amount_column": "..."}
    """
    from business_brain.discovery.customer_analytics import compute_customer_concentration
    cust_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    if not all([cust_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column and amount_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_customer_concentration(rows, cust_col, amount_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "top_customers": [
            {"customer": c.customer, "total_spend": c.total_spend, "share_pct": c.share_pct, "cumulative_pct": c.cumulative_pct}
            for c in res.top_customers
        ],
        "hhi": res.hhi,
        "concentration_risk": res.concentration_risk,
        "customers_for_80pct": res.customers_for_80pct,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/purchase-behavior")
async def purchase_behavior(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze customer purchase behavior patterns.
    Body: {"customer_column": "...", "amount_column": "...", "date_column": "...", "product_column": "..."}
    """
    from business_brain.discovery.customer_analytics import analyze_purchase_behavior
    cust_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([cust_col, amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, and date_column required"}, 400)
    product_col = body.get("product_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_purchase_behavior(rows, cust_col, amount_col, date_col, product_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "customers": [
            {"customer": c.customer, "total_orders": c.total_orders, "total_spend": c.total_spend, "avg_order_value": c.avg_order_value, "first_purchase": c.first_purchase, "last_purchase": c.last_purchase, "lifespan_days": c.lifespan_days}
            for c in res.customers
        ],
        "avg_orders": res.avg_orders,
        "avg_aov": res.avg_aov,
        "repeat_purchase_rate": res.repeat_purchase_rate,
        "avg_lifespan_days": res.avg_lifespan_days,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Sales Analytics endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/sales-performance")
async def sales_performance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze overall sales performance with optional rep and region breakdown.
    Body: {"amount_column": "...", "date_column": "...", "rep_column": "...", "region_column": "..."}
    """
    from business_brain.discovery.sales_analytics import analyze_sales_performance
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    rep_col = body.get("rep_column")
    region_col = body.get("region_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sales_performance(rows, amount_col, date_col, rep_col, region_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_sales": res.total_sales,
        "period_count": res.period_count,
        "avg_per_period": res.avg_per_period,
        "growth_rate": res.growth_rate,
        "best_period": {"period": res.best_period.period, "amount": res.best_period.amount} if res.best_period else None,
        "worst_period": {"period": res.worst_period.period, "amount": res.worst_period.amount} if res.worst_period else None,
        "by_rep": [
            {"rep": r.rep, "total": r.total, "rank": r.rank}
            for r in res.by_rep
        ],
        "by_region": [
            {"region": r.region, "total": r.total, "pct": r.pct}
            for r in res.by_region
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/product-mix")
async def product_mix(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze revenue distribution across products.
    Body: {"product_column": "...", "amount_column": "...", "quantity_column": "..."}
    """
    from business_brain.discovery.sales_analytics import analyze_product_mix
    product_col = body.get("product_column")
    amount_col = body.get("amount_column")
    if not all([product_col, amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "product_column and amount_column required"}, 400)
    qty_col = body.get("quantity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_product_mix(rows, product_col, amount_col, qty_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "products": [
            {"product": p.product, "revenue": p.revenue, "pct_of_total": p.pct_of_total, "transaction_count": p.transaction_count, "avg_price": p.avg_price}
            for p in res.products
        ],
        "total_revenue": res.total_revenue,
        "hhi": res.hhi,
        "concentration_risk": res.concentration_risk,
        "top_products": [
            {"product": p.product, "revenue": p.revenue, "pct_of_total": p.pct_of_total, "transaction_count": p.transaction_count, "avg_price": p.avg_price}
            for p in res.top_products
        ],
        "bottom_products": [
            {"product": p.product, "revenue": p.revenue, "pct_of_total": p.pct_of_total, "transaction_count": p.transaction_count, "avg_price": p.avg_price}
            for p in res.bottom_products
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/sales-velocity")
async def sales_velocity(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute sales pipeline velocity metrics.
    Body: {"deal_column": "...", "amount_column": "...", "date_column": "...", "stage_column": "..."}
    """
    from business_brain.discovery.sales_analytics import compute_sales_velocity
    deal_col = body.get("deal_column")
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([deal_col, amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, amount_column, and date_column required"}, 400)
    stage_col = body.get("stage_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_sales_velocity(rows, deal_col, amount_col, date_col, stage_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_deals": res.total_deals,
        "total_value": res.total_value,
        "avg_deal_size": res.avg_deal_size,
        "win_rate": res.win_rate,
        "avg_days_to_close": res.avg_days_to_close,
        "pipeline_velocity": res.pipeline_velocity,
        "funnel": [
            {"stage": stage, "count": count}
            for stage, count in res.funnel
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/discount-impact")
async def discount_impact(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze the impact of discounts on revenue.
    Body: {"amount_column": "...", "discount_column": "...", "quantity_column": "...", "product_column": "..."}
    """
    from business_brain.discovery.sales_analytics import analyze_discount_impact
    amount_col = body.get("amount_column")
    discount_col = body.get("discount_column")
    if not all([amount_col, discount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and discount_column required"}, 400)
    qty_col = body.get("quantity_column")
    product_col = body.get("product_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_discount_impact(rows, amount_col, discount_col, qty_col, product_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_discount": res.avg_discount,
        "max_discount": res.max_discount,
        "revenue_impact": res.revenue_impact,
        "distribution": [
            {"range_label": b.range_label, "count": b.count, "pct": b.pct}
            for b in res.distribution
        ],
        "by_product": [
            {"product": p.product, "avg_discount": p.avg_discount, "deal_count": p.deal_count}
            for p in res.by_product
        ],
        "discount_volume_correlation": res.discount_volume_correlation,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Cash Flow Analyzer endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/cash-flow")
async def cash_flow(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze cash inflows and outflows.
    Body: {"inflow_column": "...", "outflow_column": "...", "date_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.cash_flow_analyzer import analyze_cash_flow
    inflow_col = body.get("inflow_column")
    outflow_col = body.get("outflow_column")
    if not all([inflow_col, outflow_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "inflow_column and outflow_column required"}, 400)
    date_col = body.get("date_column")
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_cash_flow(rows, inflow_col, outflow_col, date_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_inflow": res.total_inflow,
        "total_outflow": res.total_outflow,
        "net_flow": res.net_flow,
        "cash_flow_ratio": res.cash_flow_ratio,
        "negative_periods_count": res.negative_periods_count,
        "period_flows": [
            {"period": p.period, "inflow": p.inflow, "outflow": p.outflow, "net_flow": p.net_flow}
            for p in res.period_flows
        ] if res.period_flows else None,
        "category_flows": [
            {"category": c.category, "total_inflow": c.total_inflow, "total_outflow": c.total_outflow, "net_flow": c.net_flow}
            for c in res.category_flows
        ] if res.category_flows else None,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/working-capital")
async def working_capital(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute working capital metrics.
    Body: {"receivables_column": "...", "payables_column": "...", "inventory_column": "...", "period_column": "..."}
    """
    from business_brain.discovery.cash_flow_analyzer import compute_working_capital
    receivables_col = body.get("receivables_column")
    payables_col = body.get("payables_column")
    if not all([receivables_col, payables_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "receivables_column and payables_column required"}, 400)
    inventory_col = body.get("inventory_column")
    period_col = body.get("period_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_working_capital(rows, receivables_col, payables_col, inventory_col, period_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_working_capital": res.avg_working_capital,
        "min_wc": res.min_wc,
        "max_wc": res.max_wc,
        "health": res.health,
        "periods": [
            {"period": p.period, "receivables": p.receivables, "payables": p.payables, "inventory": p.inventory, "working_capital": p.working_capital}
            for p in res.periods
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/burn-rate")
async def burn_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze monthly burn rate from expense data.
    Body: {"expense_column": "...", "date_column": "...", "revenue_column": "..."}
    """
    from business_brain.discovery.cash_flow_analyzer import analyze_burn_rate
    expense_col = body.get("expense_column")
    date_col = body.get("date_column")
    if not all([expense_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "expense_column and date_column required"}, 400)
    revenue_col = body.get("revenue_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_burn_rate(rows, expense_col, date_col, revenue_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "gross_burn_rate": res.gross_burn_rate,
        "net_burn_rate": res.net_burn_rate,
        "months_analyzed": res.months_analyzed,
        "total_expenses": res.total_expenses,
        "monthly_expenses": [
            {"month": m.month, "amount": m.amount, "is_highest": m.is_highest}
            for m in res.monthly_expenses
        ],
        "trend": res.trend,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/cash-forecast")
async def cash_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast future cash position using linear projection.
    Body: {"amount_column": "...", "date_column": "...", "type_column": "...", "periods_ahead": 3}
    """
    from business_brain.discovery.cash_flow_analyzer import forecast_cash_position
    amount_col = body.get("amount_column")
    date_col = body.get("date_column")
    if not all([amount_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "amount_column and date_column required"}, 400)
    type_col = body.get("type_column")
    periods_ahead = body.get("periods_ahead", 3)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = forecast_cash_position(rows, amount_col, date_col, type_col, periods_ahead)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "current_net_monthly": res.current_net_monthly,
        "trend_direction": res.trend_direction,
        "trend_magnitude": res.trend_magnitude,
        "projections": [
            {"period_label": p.period_label, "projected_inflow": p.projected_inflow, "projected_outflow": p.projected_outflow, "projected_net": p.projected_net}
            for p in res.projections
        ],
        "confidence": res.confidence,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Risk Matrix endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/risk-assessment")
async def risk_assessment(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Assess risks by computing likelihood x impact scores.
    Body: {"risk_column": "...", "likelihood_column": "...", "impact_column": "...", "category_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.risk_matrix import assess_risks
    risk_col = body.get("risk_column")
    likelihood_col = body.get("likelihood_column")
    impact_col = body.get("impact_column")
    if not all([risk_col, likelihood_col, impact_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "risk_column, likelihood_column, and impact_column required"}, 400)
    category_col = body.get("category_column")
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = assess_risks(rows, risk_col, likelihood_col, impact_col, category_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_count": res.total_count,
        "critical_count": res.critical_count,
        "high_count": res.high_count,
        "medium_count": res.medium_count,
        "low_count": res.low_count,
        "risks": [
            {"name": r.name, "likelihood": r.likelihood, "impact": r.impact, "risk_score": r.risk_score, "risk_level": r.risk_level, "category": r.category, "owner": r.owner}
            for r in res.risks
        ],
        "by_category": [
            {"category": c.category, "count": c.count, "avg_score": c.avg_score}
            for c in res.by_category
        ],
        "by_owner": [
            {"owner": o.owner, "count": o.count, "critical_count": o.critical_count}
            for o in res.by_owner
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/risk-heatmap")
async def risk_heatmap(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Create a 5x5 likelihood-by-impact risk heatmap.
    Body: {"likelihood_column": "...", "impact_column": "..."}
    """
    from business_brain.discovery.risk_matrix import compute_risk_heatmap
    likelihood_col = body.get("likelihood_column")
    impact_col = body.get("impact_column")
    if not all([likelihood_col, impact_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "likelihood_column and impact_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_risk_heatmap(rows, likelihood_col, impact_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "matrix": res.matrix,
        "hotspots": [
            {"likelihood": h.likelihood, "impact": h.impact, "count": h.count, "risk_score": h.risk_score}
            for h in res.hotspots
        ],
        "total_risks": res.total_risks,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/risk-trends")
async def risk_trends(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze risk score trends over time.
    Body: {"risk_column": "...", "score_column": "...", "date_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.risk_matrix import analyze_risk_trends
    risk_col = body.get("risk_column")
    score_col = body.get("score_column")
    date_col = body.get("date_column")
    if not all([risk_col, score_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "risk_column, score_column, and date_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_risk_trends(rows, risk_col, score_col, date_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "avg_score": p.avg_score, "max_score": p.max_score, "risk_count": p.risk_count}
            for p in res.periods
        ],
        "trend_direction": res.trend_direction,
        "avg_score_change": res.avg_score_change,
        "by_category_trend": res.by_category_trend,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/risk-exposure")
async def risk_exposure(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute financial risk exposure (expected loss) for each risk.
    Body: {"risk_column": "...", "impact_value_column": "...", "probability_column": "..."}
    """
    from business_brain.discovery.risk_matrix import compute_risk_exposure
    risk_col = body.get("risk_column")
    impact_value_col = body.get("impact_value_column")
    probability_col = body.get("probability_column")
    if not all([risk_col, impact_value_col, probability_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "risk_column, impact_value_column, and probability_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_risk_exposure(rows, risk_col, impact_value_col, probability_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_expected_loss": res.total_expected_loss,
        "top_risks": [
            {"name": r.name, "impact_value": r.impact_value, "probability": r.probability, "expected_loss": r.expected_loss}
            for r in res.top_risks
        ],
        "risk_concentration_pct": res.risk_concentration_pct,
        "avg_expected_loss": res.avg_expected_loss,
        "max_expected_loss": res.max_expected_loss,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Compliance Tracker endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/compliance-status")
async def compliance_status(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Audit compliance status across requirements.
    Body: {"requirement_column": "...", "status_column": "...", "category_column": "...", "due_date_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import audit_compliance_status
    requirement_col = body.get("requirement_column")
    status_col = body.get("status_column")
    if not all([requirement_col, status_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "requirement_column and status_column required"}, 400)
    category_col = body.get("category_column")
    due_date_col = body.get("due_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = audit_compliance_status(rows, requirement_col, status_col, category_col, due_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_requirements": res.total_requirements,
        "compliant_count": res.compliant_count,
        "non_compliant_count": res.non_compliant_count,
        "compliance_rate": res.compliance_rate,
        "overdue_count": res.overdue_count,
        "by_category": [
            {"category": c.category, "total": c.total, "compliant": c.compliant, "compliance_rate": c.compliance_rate}
            for c in res.by_category
        ] if res.by_category else None,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/audit-findings")
async def audit_findings(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze audit findings by severity, area, and status.
    Body: {"finding_column": "...", "severity_column": "...", "date_column": "...", "area_column": "...", "status_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import analyze_audit_findings
    finding_col = body.get("finding_column")
    severity_col = body.get("severity_column")
    if not all([finding_col, severity_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "finding_column and severity_column required"}, 400)
    date_col = body.get("date_column")
    area_col = body.get("area_column")
    status_col = body.get("status_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_audit_findings(rows, finding_col, severity_col, date_col, area_col, status_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_findings": res.total_findings,
        "by_severity": [
            {"severity": s.severity, "count": s.count, "pct": s.pct}
            for s in res.by_severity
        ],
        "by_area": [
            {"area": a.area, "count": a.count, "critical_count": a.critical_count}
            for a in res.by_area
        ] if res.by_area else None,
        "open_count": res.open_count,
        "closed_count": res.closed_count,
        "closure_rate": res.closure_rate,
        "monthly_trend": [
            {"month": m.month, "count": m.count}
            for m in res.monthly_trend
        ] if res.monthly_trend else None,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/compliance-score")
async def compliance_score(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute a weighted compliance score.
    Body: {"requirement_column": "...", "weight_column": "...", "score_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import compute_compliance_score
    requirement_col = body.get("requirement_column")
    weight_col = body.get("weight_column")
    score_col = body.get("score_column")
    if not all([requirement_col, weight_col, score_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "requirement_column, weight_column, and score_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_compliance_score(rows, requirement_col, weight_col, score_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_score": res.overall_score,
        "rating": res.rating,
        "weighted_scores": [
            {"category": w.category, "score": w.score, "weight": w.weight}
            for w in res.weighted_scores
        ],
        "weakest_areas": [
            {"category": w.category, "score": w.score, "weight": w.weight}
            for w in res.weakest_areas
        ],
        "by_category": res.by_category,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/regulatory-deadlines")
async def regulatory_deadlines(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Track regulatory deadlines and classify urgency.
    Body: {"regulation_column": "...", "deadline_column": "...", "status_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.compliance_tracker import track_regulatory_deadlines
    regulation_col = body.get("regulation_column")
    deadline_col = body.get("deadline_column")
    if not all([regulation_col, deadline_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "regulation_column and deadline_column required"}, 400)
    status_col = body.get("status_column")
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = track_regulatory_deadlines(rows, regulation_col, deadline_col, status_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_items": res.total_items,
        "overdue_count": res.overdue_count,
        "due_this_week": res.due_this_week,
        "due_this_month": res.due_this_month,
        "upcoming_count": res.upcoming_count,
        "items": [
            {"regulation": i.regulation, "deadline": i.deadline, "days_until": i.days_until, "urgency": i.urgency, "status": i.status, "owner": i.owner}
            for i in res.items
        ],
        "by_owner": [
            {"owner": o.owner, "total": o.total, "overdue": o.overdue}
            for o in res.by_owner
        ] if res.by_owner else None,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Asset Depreciation endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/depreciation-schedule")
async def depreciation_schedule(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute annual depreciation schedule for each asset.
    Body: {"asset_column": "...", "cost_column": "...", "useful_life_column": "...", "method": "straight_line", "salvage_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import compute_depreciation_schedule
    asset_col = body.get("asset_column")
    cost_col = body.get("cost_column")
    useful_life_col = body.get("useful_life_column")
    if not all([asset_col, cost_col, useful_life_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column, cost_column, and useful_life_column required"}, 400)
    method = body.get("method", "straight_line")
    salvage_col = body.get("salvage_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_depreciation_schedule(rows, asset_col, cost_col, useful_life_col, method, salvage_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "assets": [
            {
                "asset": a.asset, "cost": a.cost, "salvage": a.salvage, "useful_life": a.useful_life,
                "annual_depreciation": a.annual_depreciation,
                "schedule": [
                    {"year": y.year, "depreciation_amount": y.depreciation_amount, "book_value": y.book_value}
                    for y in a.schedule
                ],
            }
            for a in res.assets
        ],
        "total_cost": res.total_cost,
        "total_annual_depreciation": res.total_annual_depreciation,
        "method": res.method,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/asset-age")
async def asset_age(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze asset age distribution.
    Body: {"asset_column": "...", "purchase_date_column": "...", "category_column": "...", "cost_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import analyze_asset_age
    asset_col = body.get("asset_column")
    purchase_date_col = body.get("purchase_date_column")
    if not all([asset_col, purchase_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column and purchase_date_column required"}, 400)
    category_col = body.get("category_column")
    cost_col = body.get("cost_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_asset_age(rows, asset_col, purchase_date_col, category_col, cost_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_assets": res.total_assets,
        "avg_age": res.avg_age,
        "by_lifecycle_stage": [
            {"stage": s.stage, "count": s.count, "pct": s.pct}
            for s in res.by_lifecycle_stage
        ],
        "by_category": [
            {"category": c.category, "count": c.count, "avg_age": c.avg_age}
            for c in res.by_category
        ] if res.by_category else None,
        "weighted_avg_age": res.weighted_avg_age,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/book-values")
async def book_values(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute current book value for each asset.
    Body: {"asset_column": "...", "cost_column": "...", "purchase_date_column": "...", "useful_life_column": "...", "salvage_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import compute_book_values
    asset_col = body.get("asset_column")
    cost_col = body.get("cost_column")
    purchase_date_col = body.get("purchase_date_column")
    useful_life_col = body.get("useful_life_column")
    if not all([asset_col, cost_col, purchase_date_col, useful_life_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column, cost_column, purchase_date_column, and useful_life_column required"}, 400)
    salvage_col = body.get("salvage_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_book_values(rows, asset_col, cost_col, purchase_date_col, useful_life_col, salvage_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "assets": [
            {"asset": a.asset, "original_cost": a.original_cost, "salvage_value": a.salvage_value, "age_years": a.age_years, "book_value": a.book_value, "depreciation_pct": a.depreciation_pct}
            for a in res.assets
        ],
        "total_original_cost": res.total_original_cost,
        "total_book_value": res.total_book_value,
        "depreciation_pct": res.depreciation_pct,
        "fully_depreciated_count": res.fully_depreciated_count,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/maintenance-cost-ratio")
async def maintenance_cost_ratio(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute maintenance cost as percentage of asset value.
    Body: {"asset_column": "...", "maintenance_cost_column": "...", "asset_value_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.asset_depreciation import analyze_maintenance_cost_ratio
    asset_col = body.get("asset_column")
    maintenance_cost_col = body.get("maintenance_cost_column")
    asset_value_col = body.get("asset_value_column")
    if not all([asset_col, maintenance_cost_col, asset_value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "asset_column, maintenance_cost_column, and asset_value_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_maintenance_cost_ratio(rows, asset_col, maintenance_cost_col, asset_value_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "assets": [
            {"asset": a.asset, "maintenance_cost": a.maintenance_cost, "asset_value": a.asset_value, "ratio_pct": a.ratio_pct, "is_replacement_candidate": a.is_replacement_candidate}
            for a in res.assets
        ],
        "avg_ratio": res.avg_ratio,
        "replacement_candidates": res.replacement_candidates,
        "total_maintenance": res.total_maintenance,
        "total_asset_value": res.total_asset_value,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Pricing Optimizer endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/price-distribution")
async def price_distribution(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze the distribution of prices.
    Body: {"price_column": "...", "product_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import analyze_price_distribution
    price_col = body.get("price_column")
    if not price_col:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "price_column required"}, 400)
    product_col = body.get("product_column")
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_price_distribution(rows, price_col, product_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "mean_price": res.mean_price,
        "median_price": res.median_price,
        "std_price": res.std_price,
        "min_price": res.min_price,
        "max_price": res.max_price,
        "outlier_count": res.outlier_count,
        "outlier_pct": res.outlier_pct,
        "by_product": [
            {"product": p.product, "min_price": p.min_price, "max_price": p.max_price, "avg_price": p.avg_price, "count": p.count}
            for p in res.by_product
        ],
        "by_category": [
            {"category": c.category, "avg_price": c.avg_price, "count": c.count}
            for c in res.by_category
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/price-elasticity")
async def price_elasticity(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute price elasticity of demand.
    Body: {"price_column": "...", "quantity_column": "...", "product_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import compute_price_elasticity
    price_col = body.get("price_column")
    quantity_col = body.get("quantity_column")
    if not all([price_col, quantity_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "price_column and quantity_column required"}, 400)
    product_col = body.get("product_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_price_elasticity(rows, price_col, quantity_col, product_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_elasticity": res.overall_elasticity,
        "elasticity_type": res.elasticity_type,
        "by_product": [
            {"product": p.product, "elasticity": p.elasticity, "elasticity_type": p.elasticity_type}
            for p in res.by_product
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/competitive-pricing")
async def competitive_pricing(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze competitive pricing gaps.
    Body: {"product_column": "...", "our_price_column": "...", "competitor_price_column": "...", "competitor_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import analyze_competitive_pricing
    product_col = body.get("product_column")
    our_price_col = body.get("our_price_column")
    competitor_price_col = body.get("competitor_price_column")
    if not all([product_col, our_price_col, competitor_price_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "product_column, our_price_column, and competitor_price_column required"}, 400)
    competitor_col = body.get("competitor_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_competitive_pricing(rows, product_col, our_price_col, competitor_price_col, competitor_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_price_gap": res.avg_price_gap,
        "premium_count": res.premium_count,
        "competitive_count": res.competitive_count,
        "discount_count": res.discount_count,
        "by_product": [
            {"product": p.product, "our_price": p.our_price, "competitor_avg_price": p.competitor_avg_price, "price_gap_pct": p.price_gap_pct, "position": p.position}
            for p in res.by_product
        ],
        "by_competitor": [
            {"competitor": c.competitor, "avg_gap_pct": c.avg_gap_pct, "product_count": c.product_count}
            for c in res.by_competitor
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/price-margins")
async def price_margins(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute gross margin analysis.
    Body: {"price_column": "...", "cost_column": "...", "product_column": "...", "quantity_column": "..."}
    """
    from business_brain.discovery.pricing_optimizer import compute_price_margin_analysis
    price_col = body.get("price_column")
    cost_col = body.get("cost_column")
    if not all([price_col, cost_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "price_column and cost_column required"}, 400)
    product_col = body.get("product_column")
    quantity_col = body.get("quantity_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_price_margin_analysis(rows, price_col, cost_col, product_col, quantity_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_margin": res.avg_margin,
        "weighted_margin": res.weighted_margin,
        "min_margin": res.min_margin,
        "max_margin": res.max_margin,
        "negative_margin_count": res.negative_margin_count,
        "by_product": [
            {"product": p.product, "avg_price": p.avg_price, "avg_cost": p.avg_cost, "margin_pct": p.margin_pct, "volume": p.volume}
            for p in res.by_product
        ],
        "margin_distribution": [
            {"range_label": b.range_label, "count": b.count, "pct": b.pct}
            for b in res.margin_distribution
        ],
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Employee Attrition endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/attrition-rate")
async def attrition_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze employee attrition rate.
    Body: {"employee_column": "...", "status_column": "...", "date_column": "...", "department_column": "..."}
    """
    from business_brain.discovery.employee_attrition import analyze_attrition_rate
    employee_col = body.get("employee_column")
    status_col = body.get("status_column")
    if not all([employee_col, status_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column and status_column required"}, 400)
    date_col = body.get("date_column")
    department_col = body.get("department_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_attrition_rate(rows, employee_col, status_col, date_col, department_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_employees": res.total_employees,
        "active_count": res.active_count,
        "left_count": res.left_count,
        "attrition_rate": res.attrition_rate,
        "monthly_trends": [
            {"month": m.month, "active": m.active, "left": m.left, "rate": m.rate}
            for m in res.monthly_trends
        ],
        "by_department": [
            {"department": d.department, "total": d.total, "left": d.left, "rate": d.rate}
            for d in res.by_department
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/tenure-distribution")
async def tenure_distribution(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze employee tenure distribution.
    Body: {"employee_column": "...", "hire_date_column": "...", "termination_date_column": "...", "department_column": "..."}
    """
    from business_brain.discovery.employee_attrition import analyze_tenure_distribution
    employee_col = body.get("employee_column")
    hire_date_col = body.get("hire_date_column")
    if not all([employee_col, hire_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column and hire_date_column required"}, 400)
    termination_date_col = body.get("termination_date_column")
    department_col = body.get("department_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_tenure_distribution(rows, employee_col, hire_date_col, termination_date_col, department_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_tenure": res.avg_tenure,
        "median_tenure": res.median_tenure,
        "buckets": [
            {"range_label": b.range_label, "count": b.count, "pct": b.pct}
            for b in res.buckets
        ],
        "leaver_avg_tenure": res.leaver_avg_tenure,
        "stayer_avg_tenure": res.stayer_avg_tenure,
        "by_department": [
            {"department": d.department, "avg_tenure": d.avg_tenure, "count": d.count}
            for d in res.by_department
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/retention-cohorts")
async def retention_cohorts(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Group employees by hire quarter and compute retention rates.
    Body: {"employee_column": "...", "hire_date_column": "...", "termination_date_column": "..."}
    """
    from business_brain.discovery.employee_attrition import compute_retention_cohorts
    employee_col = body.get("employee_column")
    hire_date_col = body.get("hire_date_column")
    termination_date_col = body.get("termination_date_column")
    if not all([employee_col, hire_date_col, termination_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column, hire_date_column, and termination_date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_retention_cohorts(rows, employee_col, hire_date_col, termination_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "cohorts": [
            {
                "cohort_label": c.cohort_label, "starting_count": c.starting_count,
                "retained_count": c.retained_count, "retention_rate": c.retention_rate,
                "retention_milestones": [
                    {"period": m.period, "retained_pct": m.retained_pct}
                    for m in c.retention_milestones
                ],
            }
            for c in res.cohorts
        ],
        "overall_1yr_retention": res.overall_1yr_retention,
        "best_cohort": res.best_cohort,
        "worst_cohort": res.worst_cohort,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/attrition-drivers")
async def attrition_drivers(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze factors that drive employee attrition.
    Body: {"employee_column": "...", "status_column": "...", "factor_columns": ["col1", "col2", ...]}
    """
    from business_brain.discovery.employee_attrition import analyze_attrition_drivers
    employee_col = body.get("employee_column")
    status_col = body.get("status_column")
    factor_cols = body.get("factor_columns")
    if not all([employee_col, status_col, factor_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "employee_column, status_column, and factor_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_attrition_drivers(rows, employee_col, status_col, factor_cols)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "factors": [
            {"factor_name": f.factor_name, "factor_type": f.factor_type, "leaver_value": f.leaver_value, "stayer_value": f.stayer_value, "impact": f.impact, "direction": f.direction}
            for f in res.factors
        ],
        "top_driver": res.top_driver,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Revenue Forecaster endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/revenue-forecast")
async def revenue_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast future revenue using linear extrapolation.
    Body: {"revenue_column": "...", "date_column": "...", "periods_ahead": 3}
    """
    from business_brain.discovery.revenue_forecaster import forecast_revenue
    revenue_col = body.get("revenue_column")
    date_col = body.get("date_column")
    if not all([revenue_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and date_column required"}, 400)
    periods_ahead = body.get("periods_ahead", 3)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = forecast_revenue(rows, revenue_col, date_col, periods_ahead)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "revenue": p.revenue, "growth_rate": p.growth_rate}
            for p in res.periods
        ],
        "forecasts": [
            {"period": f.period, "revenue": f.revenue, "growth_rate": f.growth_rate}
            for f in res.forecasts
        ],
        "avg_growth_rate": res.avg_growth_rate,
        "trend": res.trend,
        "total_historical": res.total_historical,
        "total_forecast": res.total_forecast,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/revenue-segments")
async def revenue_segments(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze revenue breakdown by segment.
    Body: {"revenue_column": "...", "segment_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.revenue_forecaster import analyze_revenue_segments
    revenue_col = body.get("revenue_column")
    segment_col = body.get("segment_column")
    if not all([revenue_col, segment_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and segment_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_revenue_segments(rows, revenue_col, segment_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "segments": [
            {"segment": s.segment, "revenue": s.revenue, "share_pct": s.share_pct, "rank": s.rank, "transaction_count": s.transaction_count}
            for s in res.segments
        ],
        "total_revenue": res.total_revenue,
        "top_segment": res.top_segment,
        "concentration_index": res.concentration_index,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/revenue-growth")
async def revenue_growth(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute period-over-period revenue growth rates.
    Body: {"revenue_column": "...", "date_column": "...", "comparison": "period_over_period"}
    """
    from business_brain.discovery.revenue_forecaster import compute_revenue_growth
    revenue_col = body.get("revenue_column")
    date_col = body.get("date_column")
    if not all([revenue_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and date_column required"}, 400)
    comparison = body.get("comparison", "period_over_period")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_revenue_growth(rows, revenue_col, date_col, comparison)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "revenue": p.revenue, "growth_rate": p.growth_rate, "growth_absolute": p.growth_absolute}
            for p in res.periods
        ],
        "cagr": res.cagr,
        "avg_growth": res.avg_growth,
        "best_period": res.best_period,
        "worst_period": res.worst_period,
        "volatility": res.volatility,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/revenue-drivers")
async def revenue_drivers(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze which driver columns correlate with revenue.
    Body: {"revenue_column": "...", "driver_columns": ["col1", "col2", ...]}
    """
    from business_brain.discovery.revenue_forecaster import analyze_revenue_drivers
    revenue_col = body.get("revenue_column")
    driver_cols = body.get("driver_columns")
    if not all([revenue_col, driver_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and driver_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_revenue_drivers(rows, revenue_col, driver_cols)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "drivers": [
            {"driver": d.driver, "correlation": d.correlation, "direction": d.direction, "avg_when_high_revenue": d.avg_when_high_revenue, "avg_when_low_revenue": d.avg_when_low_revenue}
            for d in res.drivers
        ],
        "top_driver": res.top_driver,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Accounts Aging endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/receivables-aging")
async def receivables_aging(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze receivables aging by customer.
    Body: {"customer_column": "...", "amount_column": "...", "invoice_date_column": "...", "due_date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import analyze_receivables_aging
    customer_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    invoice_date_col = body.get("invoice_date_column")
    if not all([customer_col, amount_col, invoice_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, and invoice_date_column required"}, 400)
    due_date_col = body.get("due_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_receivables_aging(rows, customer_col, amount_col, invoice_date_col, due_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "buckets": [
            {"label": b.label, "min_days": b.min_days, "max_days": b.max_days, "count": b.count, "amount": b.amount, "pct_of_total": b.pct_of_total}
            for b in res.buckets
        ],
        "by_customer": [
            {"customer": c.customer, "total": c.total, "current": c.current, "days_31_60": c.days_31_60, "days_61_90": c.days_61_90, "days_91_120": c.days_91_120, "over_120": c.over_120}
            for c in res.by_customer
        ],
        "total_outstanding": res.total_outstanding,
        "total_overdue": res.total_overdue,
        "avg_days_outstanding": res.avg_days_outstanding,
        "worst_customers": res.worst_customers,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/payables-aging")
async def payables_aging(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze payables aging by vendor.
    Body: {"vendor_column": "...", "amount_column": "...", "invoice_date_column": "...", "due_date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import analyze_payables_aging
    vendor_col = body.get("vendor_column")
    amount_col = body.get("amount_column")
    invoice_date_col = body.get("invoice_date_column")
    if not all([vendor_col, amount_col, invoice_date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "vendor_column, amount_column, and invoice_date_column required"}, 400)
    due_date_col = body.get("due_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_payables_aging(rows, vendor_col, amount_col, invoice_date_col, due_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "buckets": [
            {"label": b.label, "min_days": b.min_days, "max_days": b.max_days, "count": b.count, "amount": b.amount, "pct_of_total": b.pct_of_total}
            for b in res.buckets
        ],
        "by_vendor": [
            {"vendor": v.vendor, "total": v.total, "current": v.current, "days_31_60": v.days_31_60, "days_61_90": v.days_61_90, "days_91_120": v.days_91_120, "over_120": v.over_120}
            for v in res.by_vendor
        ],
        "total_outstanding": res.total_outstanding,
        "total_overdue": res.total_overdue,
        "avg_days_outstanding": res.avg_days_outstanding,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/dso")
async def dso_analysis(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute Days Sales Outstanding (DSO).
    Body: {"revenue_column": "...", "receivables_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import compute_dso
    revenue_col = body.get("revenue_column")
    receivables_col = body.get("receivables_column")
    if not all([revenue_col, receivables_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "revenue_column and receivables_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_dso(rows, revenue_col, receivables_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_dso": res.overall_dso,
        "periods": [
            {"period": p.period, "dso": p.dso, "revenue": p.revenue, "receivables": p.receivables}
            for p in res.periods
        ],
        "trend": res.trend,
        "benchmark_status": res.benchmark_status,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/collection-effectiveness")
async def collection_effectiveness(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze collection effectiveness by customer.
    Body: {"customer_column": "...", "amount_column": "...", "paid_amount_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.accounts_aging import analyze_collection_effectiveness
    customer_col = body.get("customer_column")
    amount_col = body.get("amount_column")
    paid_amount_col = body.get("paid_amount_column")
    if not all([customer_col, amount_col, paid_amount_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "customer_column, amount_column, and paid_amount_column required"}, 400)
    date_col = body.get("date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_collection_effectiveness(rows, customer_col, amount_col, paid_amount_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_rate": res.overall_rate,
        "total_invoiced": res.total_invoiced,
        "total_collected": res.total_collected,
        "total_outstanding": res.total_outstanding,
        "by_customer": [
            {"customer": c.customer, "invoiced": c.invoiced, "collected": c.collected, "collection_rate": c.collection_rate, "outstanding": c.outstanding}
            for c in res.by_customer
        ],
        "best_collectors": res.best_collectors,
        "worst_collectors": res.worst_collectors,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Pipeline Analyzer endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/pipeline-stages")
async def pipeline_stages(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze pipeline by stage, computing counts, values, and conversions.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import analyze_pipeline_stages
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    value_col = body.get("value_column")
    if not all([deal_col, stage_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, stage_column, and value_column required"}, 400)
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_pipeline_stages(rows, deal_col, stage_col, value_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "stages": [
            {"stage": s.stage, "deal_count": s.deal_count, "total_value": s.total_value, "avg_value": s.avg_value, "pct_of_deals": s.pct_of_deals, "pct_of_value": s.pct_of_value}
            for s in res.stages
        ],
        "conversions": [
            {"from_stage": c.from_stage, "to_stage": c.to_stage, "conversion_rate": c.conversion_rate}
            for c in res.conversions
        ],
        "total_deals": res.total_deals,
        "total_value": res.total_value,
        "weighted_value": res.weighted_value,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/pipeline-velocity")
async def pipeline_velocity(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Calculate pipeline velocity and identify bottleneck stages.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "date_column": "...", "close_date_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import analyze_pipeline_velocity
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    value_col = body.get("value_column")
    date_col = body.get("date_column")
    if not all([deal_col, stage_col, value_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, stage_column, value_column, and date_column required"}, 400)
    close_date_col = body.get("close_date_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_pipeline_velocity(rows, deal_col, stage_col, value_col, date_col, close_date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "stage_velocities": [
            {"stage": sv.stage, "avg_days": sv.avg_days, "deal_count": sv.deal_count}
            for sv in res.stage_velocities
        ],
        "avg_cycle_days": res.avg_cycle_days,
        "fastest_deal_days": res.fastest_deal_days,
        "slowest_deal_days": res.slowest_deal_days,
        "bottleneck_stage": res.bottleneck_stage,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/win-rate")
async def win_rate(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute win/loss rates overall and by owner.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "owner_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import compute_win_rate
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    if not all([deal_col, stage_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column and stage_column required"}, 400)
    value_col = body.get("value_column")
    owner_col = body.get("owner_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_win_rate(rows, deal_col, stage_col, value_col, owner_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "overall_win_rate": res.overall_win_rate,
        "total_won": res.total_won,
        "total_lost": res.total_lost,
        "won_value": res.won_value,
        "lost_value": res.lost_value,
        "by_owner": [
            {"owner": o.owner, "won": o.won, "lost": o.lost, "win_rate": o.win_rate, "won_value": o.won_value}
            for o in res.by_owner
        ],
        "best_performer": res.best_performer,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/pipeline-forecast")
async def pipeline_forecast(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Forecast weighted pipeline value by stage.
    Body: {"deal_column": "...", "stage_column": "...", "value_column": "...", "probability_column": "..."}
    """
    from business_brain.discovery.pipeline_analyzer import forecast_pipeline
    deal_col = body.get("deal_column")
    stage_col = body.get("stage_column")
    value_col = body.get("value_column")
    if not all([deal_col, stage_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "deal_column, stage_column, and value_column required"}, 400)
    probability_col = body.get("probability_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = forecast_pipeline(rows, deal_col, stage_col, value_col, probability_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "stages": [
            {"stage": s.stage, "deal_count": s.deal_count, "raw_value": s.raw_value, "probability": s.probability, "weighted_value": s.weighted_value}
            for s in res.stages
        ],
        "total_raw": res.total_raw,
        "total_weighted": res.total_weighted,
        "expected_close_value": res.expected_close_value,
        "coverage_ratio": res.coverage_ratio,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Market Basket endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/product-associations")
async def product_associations(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Find pairs of products that frequently appear together in transactions.
    Body: {"transaction_column": "...", "product_column": "...", "min_support": 0.01}
    """
    from business_brain.discovery.market_basket import find_product_associations
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    if not all([transaction_col, product_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column and product_column required"}, 400)
    min_support = body.get("min_support", 0.01)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = find_product_associations(rows, transaction_col, product_col, min_support)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "pairs": [
            {"product_a": p.product_a, "product_b": p.product_b, "support": p.support, "confidence_a_to_b": p.confidence_a_to_b, "confidence_b_to_a": p.confidence_b_to_a, "lift": p.lift, "co_occurrence_count": p.co_occurrence_count}
            for p in res.pairs
        ],
        "total_transactions": res.total_transactions,
        "unique_products": res.unique_products,
        "avg_basket_size": res.avg_basket_size,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/basket-size")
async def basket_size(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze the distribution of basket sizes (items per transaction).
    Body: {"transaction_column": "...", "product_column": "...", "value_column": "..."}
    """
    from business_brain.discovery.market_basket import analyze_basket_size
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    if not all([transaction_col, product_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column and product_column required"}, 400)
    value_col = body.get("value_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_basket_size(rows, transaction_col, product_col, value_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_size": res.avg_size,
        "median_size": res.median_size,
        "max_size": res.max_size,
        "min_size": res.min_size,
        "distribution": [
            {"size": b.size, "count": b.count, "pct": b.pct}
            for b in res.distribution
        ],
        "avg_value": res.avg_value,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/cross-sell")
async def cross_sell(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Find products most frequently purchased alongside a target product.
    Body: {"transaction_column": "...", "product_column": "...", "target_product": "..."}
    """
    from business_brain.discovery.market_basket import find_cross_sell_opportunities
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    target_product = body.get("target_product")
    if not all([transaction_col, product_col, target_product]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column, product_column, and target_product required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = find_cross_sell_opportunities(rows, transaction_col, product_col, target_product)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "target_product": res.target_product,
        "target_transactions": res.target_transactions,
        "recommendations": [
            {"product": r.product, "co_purchase_rate": r.co_purchase_rate, "lift": r.lift, "co_occurrence_count": r.co_occurrence_count}
            for r in res.recommendations
        ],
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/product-frequency")
async def product_frequency(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze how often each product appears across transactions.
    Body: {"transaction_column": "...", "product_column": "...", "customer_column": "..."}
    """
    from business_brain.discovery.market_basket import analyze_product_frequency
    transaction_col = body.get("transaction_column")
    product_col = body.get("product_column")
    if not all([transaction_col, product_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "transaction_column and product_column required"}, 400)
    customer_col = body.get("customer_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_product_frequency(rows, transaction_col, product_col, customer_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "products": [
            {"product": p.product, "frequency": p.frequency, "pct_of_transactions": p.pct_of_transactions, "unique_customers": p.unique_customers, "rank": p.rank}
            for p in res.products
        ],
        "total_transactions": res.total_transactions,
        "total_products": res.total_products,
        "most_popular": res.most_popular,
        "least_popular": res.least_popular,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# SLA Monitor endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/sla-compliance")
async def sla_compliance(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze SLA compliance by comparing actual vs target for each ticket.
    Body: {"ticket_column": "...", "sla_target_column": "...", "actual_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.sla_monitor import analyze_sla_compliance
    ticket_col = body.get("ticket_column")
    sla_target_col = body.get("sla_target_column")
    actual_col = body.get("actual_column")
    if not all([ticket_col, sla_target_col, actual_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column, sla_target_column, and actual_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sla_compliance(rows, ticket_col, sla_target_col, actual_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_tickets": res.total_tickets,
        "met_count": res.met_count,
        "breached_count": res.breached_count,
        "compliance_rate": res.compliance_rate,
        "by_category": [
            {"category": c.category, "total": c.total, "met": c.met, "breached": c.breached, "compliance_rate": c.compliance_rate, "avg_actual": c.avg_actual, "avg_target": c.avg_target}
            for c in res.by_category
        ],
        "worst_category": res.worst_category,
        "avg_performance_ratio": res.avg_performance_ratio,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/response-times")
async def response_times(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Analyze response time distribution across tickets.
    Body: {"ticket_column": "...", "response_time_column": "...", "priority_column": "...", "agent_column": "..."}
    """
    from business_brain.discovery.sla_monitor import analyze_response_times
    ticket_col = body.get("ticket_column")
    response_time_col = body.get("response_time_column")
    if not all([ticket_col, response_time_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column and response_time_column required"}, 400)
    priority_col = body.get("priority_column")
    agent_col = body.get("agent_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_response_times(rows, ticket_col, response_time_col, priority_col, agent_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "avg_response_time": res.avg_response_time,
        "median_response_time": res.median_response_time,
        "p95_response_time": res.p95_response_time,
        "min_time": res.min_time,
        "max_time": res.max_time,
        "by_priority": [
            {"priority": p.priority, "count": p.count, "avg_time": p.avg_time, "median_time": p.median_time, "p95_time": p.p95_time}
            for p in res.by_priority
        ],
        "by_agent": [
            {"agent": a.agent, "count": a.count, "avg_time": a.avg_time, "compliance_rate": a.compliance_rate}
            for a in res.by_agent
        ],
        "outlier_count": res.outlier_count,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/resolution-metrics")
async def resolution_metrics(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compute ticket resolution metrics.
    Body: {"ticket_column": "...", "created_column": "...", "resolved_column": "...", "status_column": "...", "priority_column": "..."}
    """
    from business_brain.discovery.sla_monitor import compute_resolution_metrics
    ticket_col = body.get("ticket_column")
    created_col = body.get("created_column")
    resolved_col = body.get("resolved_column")
    if not all([ticket_col, created_col, resolved_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column, created_column, and resolved_column required"}, 400)
    status_col = body.get("status_column")
    priority_col = body.get("priority_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_resolution_metrics(rows, ticket_col, created_col, resolved_col, status_col, priority_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "total_tickets": res.total_tickets,
        "resolved_count": res.resolved_count,
        "open_count": res.open_count,
        "resolution_rate": res.resolution_rate,
        "avg_resolution_hours": res.avg_resolution_hours,
        "median_resolution_hours": res.median_resolution_hours,
        "by_priority": [
            {"priority": p.priority, "total": p.total, "resolved": p.resolved, "avg_resolution_hours": p.avg_resolution_hours, "resolution_rate": p.resolution_rate}
            for p in res.by_priority
        ],
        "backlog_age_avg_hours": res.backlog_age_avg_hours,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/sla-trends")
async def sla_trends(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Track SLA compliance rate over time periods.
    Body: {"ticket_column": "...", "sla_met_column": "...", "date_column": "...", "category_column": "..."}
    """
    from business_brain.discovery.sla_monitor import analyze_sla_trends
    ticket_col = body.get("ticket_column")
    sla_met_col = body.get("sla_met_column")
    date_col = body.get("date_column")
    if not all([ticket_col, sla_met_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "ticket_column, sla_met_column, and date_column required"}, 400)
    category_col = body.get("category_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_sla_trends(rows, ticket_col, sla_met_col, date_col, category_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "periods": [
            {"period": p.period, "total": p.total, "met": p.met, "compliance_rate": p.compliance_rate}
            for p in res.periods
        ],
        "trend_direction": res.trend_direction,
        "overall_compliance": res.overall_compliance,
        "best_period": res.best_period,
        "worst_period": res.worst_period,
        "summary": res.summary,
    }


# ---------------------------------------------------------------------------
# Geo Analytics endpoints
# ---------------------------------------------------------------------------


@router.post("/tables/{table_name}/regional-distribution")
async def regional_distribution(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Breakdown of value by region with concentration analysis.
    Body: {"region_column": "...", "value_column": "...", "count_column": "..."}
    """
    from business_brain.discovery.geo_analytics import analyze_regional_distribution
    region_col = body.get("region_column")
    value_col = body.get("value_column")
    if not all([region_col, value_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column and value_column required"}, 400)
    count_col = body.get("count_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_regional_distribution(rows, region_col, value_col, count_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "regions": [
            {"region": r.region, "total_value": r.total_value, "share_pct": r.share_pct, "count": r.count, "avg_value": r.avg_value, "rank": r.rank}
            for r in res.regions
        ],
        "total_value": res.total_value,
        "total_count": res.total_count,
        "top_region": res.top_region,
        "concentration_ratio": res.concentration_ratio,
        "hhi_index": res.hhi_index,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/region-comparison")
async def region_comparison(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Compare multiple metrics across regions.
    Body: {"region_column": "...", "metric_columns": ["col1", "col2", ...]}
    """
    from business_brain.discovery.geo_analytics import compare_regions
    region_col = body.get("region_column")
    metric_cols = body.get("metric_columns")
    if not all([region_col, metric_cols]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column and metric_columns required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compare_regions(rows, region_col, metric_cols)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "comparisons": [
            {"metric": c.metric, "best_region": c.best_region, "best_value": c.best_value, "worst_region": c.worst_region, "worst_value": c.worst_value, "avg_value": c.avg_value, "std_dev": c.std_dev}
            for c in res.comparisons
        ],
        "region_scores": [
            {"region": rs.region, "metrics": rs.metrics, "overall_score": rs.overall_score}
            for rs in res.region_scores
        ],
        "best_overall": res.best_overall,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/geographic-growth")
async def geographic_growth(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Growth rates per region over time.
    Body: {"region_column": "...", "value_column": "...", "date_column": "..."}
    """
    from business_brain.discovery.geo_analytics import analyze_geographic_growth
    region_col = body.get("region_column")
    value_col = body.get("value_column")
    date_col = body.get("date_column")
    if not all([region_col, value_col, date_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column, value_column, and date_column required"}, 400)
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = analyze_geographic_growth(rows, region_col, value_col, date_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "regions": [
            {"region": r.region, "first_period_value": r.first_period_value, "last_period_value": r.last_period_value, "growth_rate": r.growth_rate, "periods": r.periods}
            for r in res.regions
        ],
        "fastest_growing": res.fastest_growing,
        "slowest_growing": res.slowest_growing,
        "avg_growth": res.avg_growth,
        "summary": res.summary,
    }


@router.post("/tables/{table_name}/market-penetration")
async def market_penetration(table_name: str, body: dict, session: AsyncSession = Depends(get_session)):
    """Unique customers per region and market penetration.
    Body: {"region_column": "...", "customer_column": "...", "potential_column": "..."}
    """
    from business_brain.discovery.geo_analytics import compute_market_penetration
    region_col = body.get("region_column")
    customer_col = body.get("customer_column")
    if not all([region_col, customer_col]):
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "region_column and customer_column required"}, 400)
    potential_col = body.get("potential_column")
    from sqlalchemy import text as sql_text
    result = await session.execute(sql_text(f'SELECT * FROM "{table_name}" LIMIT 10000'))
    rows = [dict(r._mapping) for r in result.fetchall()]
    res = compute_market_penetration(rows, region_col, customer_col, potential_col)
    if res is None:
        return {"error": "Insufficient data"}
    return {
        "regions": [
            {"region": r.region, "customer_count": r.customer_count, "potential": r.potential, "penetration_pct": r.penetration_pct, "rank": r.rank}
            for r in res.regions
        ],
        "total_customers": res.total_customers,
        "total_regions": res.total_regions,
        "best_penetration": res.best_penetration,
        "untapped_regions": res.untapped_regions,
        "summary": res.summary,
    }

