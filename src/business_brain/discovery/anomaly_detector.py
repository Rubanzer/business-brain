"""Per-table anomaly scanning — outliers, null spikes, impossible values, rare categories."""

from __future__ import annotations

import logging
import uuid

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


def detect_anomalies(profiles: list[TableProfile]) -> list[Insight]:
    """Scan all profiled tables for anomalies. No DB queries needed — uses profile data."""
    insights: list[Insight] = []

    for profile in profiles:
        try:
            insights.extend(_scan_table(profile))
        except Exception:
            logger.exception("Anomaly scan failed for %s", profile.table_name)

    return insights


def _scan_table(profile: TableProfile) -> list[Insight]:
    """Scan a single table's profile for anomalies."""
    results: list[Insight] = []
    cls = profile.column_classification
    if not cls or "columns" not in cls:
        return results

    cols = cls["columns"]
    row_count = profile.row_count or 0

    for col_name, info in cols.items():
        sem_type = info.get("semantic_type", "")
        null_count = info.get("null_count", 0)
        stats = info.get("stats")
        cardinality = info.get("cardinality", 0)
        samples = info.get("sample_values", [])

        # 1. Null spike: > 10% nulls
        if row_count > 0 and null_count / row_count > 0.10:
            pct = round(null_count / row_count * 100, 1)
            results.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="anomaly",
                severity="warning" if pct > 30 else "info",
                impact_score=min(int(pct), 80),
                title=f"High null rate in {profile.table_name}.{col_name}",
                description=f"{col_name} has {null_count} null values ({pct}% of {row_count} rows).",
                source_tables=[profile.table_name],
                source_columns=[col_name],
                evidence={"null_count": null_count, "row_count": row_count, "pct": pct},
                suggested_actions=[
                    f"Investigate why {col_name} has missing values",
                    "Consider imputation or data quality enforcement",
                ],
            ))

        # 2. Numeric outliers: values > 2 stdev from mean
        if stats and "stdev" in stats and stats["stdev"] > 0:
            mean = stats["mean"]
            stdev = stats["stdev"]
            low_bound = mean - 2 * stdev
            high_bound = mean + 2 * stdev

            outlier_samples = []
            for s in samples:
                try:
                    v = float(str(s).replace(",", ""))
                    if v < low_bound or v > high_bound:
                        outlier_samples.append(s)
                except (ValueError, TypeError):
                    pass

            if outlier_samples:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="warning",
                    impact_score=45,
                    title=f"Outlier values in {profile.table_name}.{col_name}",
                    description=(
                        f"{col_name} has values beyond 2 standard deviations from mean "
                        f"(mean={stats['mean']}, stdev={stats['stdev']}). "
                        f"Examples: {outlier_samples[:3]}"
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={
                        "mean": stats["mean"],
                        "stdev": stats["stdev"],
                        "low_bound": round(low_bound, 2),
                        "high_bound": round(high_bound, 2),
                        "outlier_samples": outlier_samples[:5],
                    },
                    suggested_actions=[
                        f"Review outlier values in {col_name} for data entry errors",
                        "Check if outliers represent legitimate edge cases",
                    ],
                ))

        # 3. Impossible values in currency columns (negative amounts)
        if sem_type == "numeric_currency" and stats:
            if stats.get("min", 0) < 0:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="critical",
                    impact_score=70,
                    title=f"Negative values in currency column {profile.table_name}.{col_name}",
                    description=(
                        f"{col_name} contains negative values (min={stats['min']}). "
                        f"Currency columns typically should not have negative values."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={"min": stats["min"], "column_type": "numeric_currency"},
                    suggested_actions=[
                        "Check if negative values represent refunds/credits (valid) or errors",
                        "Separate refunds into a dedicated column if applicable",
                    ],
                ))

        # 4. Impossible values in percentage columns (> 100 or < 0)
        if sem_type == "numeric_percentage" and stats:
            if stats.get("max", 0) > 100 or stats.get("min", 0) < 0:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="critical",
                    impact_score=65,
                    title=f"Out-of-range percentage in {profile.table_name}.{col_name}",
                    description=(
                        f"{col_name} has values outside 0-100 range "
                        f"(min={stats.get('min')}, max={stats.get('max')})."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={"min": stats.get("min"), "max": stats.get("max")},
                    suggested_actions=["Fix values outside the valid 0-100 percentage range"],
                ))

        # 5. Rare categories: values appearing < 2% of the time
        if sem_type == "categorical" and cardinality > 5 and samples:
            # We can only estimate from sample_values — flag if cardinality is very high
            if row_count > 0 and cardinality > row_count * 0.5:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="info",
                    impact_score=20,
                    title=f"High cardinality categorical {profile.table_name}.{col_name}",
                    description=(
                        f"{col_name} has {cardinality} unique values across {row_count} rows. "
                        f"This may actually be an identifier column, not categorical."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={"cardinality": cardinality, "row_count": row_count},
                    suggested_actions=["Review if this column should be treated as an identifier"],
                ))

        # 6. Constant columns: cardinality of 1 (useless for analysis)
        if cardinality == 1 and row_count > 1:
            results.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="anomaly",
                severity="info",
                impact_score=10,
                title=f"Constant column {profile.table_name}.{col_name}",
                description=(
                    f"{col_name} has only 1 unique value across {row_count} rows. "
                    f"This column provides no analytical value."
                ),
                source_tables=[profile.table_name],
                source_columns=[col_name],
                evidence={"cardinality": 1, "sample": samples[:1] if samples else []},
                suggested_actions=["Consider removing this column from analysis"],
            ))

    # 7. Time-based insights: detect temporal + numeric combinations for trend detection
    temp_cols = [c for c, i in cols.items() if i.get("semantic_type") == "temporal"]
    num_cols = [
        c for c, i in cols.items()
        if i.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ]

    if temp_cols and num_cols:
        results.append(Insight(
            id=str(uuid.uuid4()),
            insight_type="trend",
            severity="info",
            impact_score=35,
            title=f"Time series data available in {profile.table_name}",
            description=(
                f"Table has temporal column(s) {temp_cols} and numeric column(s) {num_cols[:3]}. "
                f"Period-over-period trend analysis is possible."
            ),
            source_tables=[profile.table_name],
            source_columns=temp_cols + num_cols[:3],
            evidence={
                "temporal_columns": temp_cols,
                "numeric_columns": num_cols[:3],
                "query": (
                    f'SELECT "{temp_cols[0]}", '
                    + ", ".join(f'AVG("{n}")' for n in num_cols[:3])
                    + f' FROM "{profile.table_name}" GROUP BY "{temp_cols[0]}" '
                    f'ORDER BY "{temp_cols[0]}"'
                ),
                "chart_spec": {
                    "type": "line",
                    "x": temp_cols[0],
                    "y": num_cols[:2],
                    "title": f"{num_cols[0]} trend over {temp_cols[0]}",
                },
            },
            suggested_actions=[
                f"Analyze {num_cols[0]} trend over {temp_cols[0]}",
                "Check for period-over-period growth or decline",
            ],
        ))

    return results
