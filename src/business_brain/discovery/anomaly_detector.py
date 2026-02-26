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

    # 7. Time-based: detect actual trends from sample data (not just "analysis possible")
    temp_cols = [c for c, i in cols.items() if i.get("semantic_type") == "temporal"]
    num_cols = [
        c for c, i in cols.items()
        if i.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ]

    if temp_cols and num_cols:
        # Only create trend insight if we can detect an actual trend from sample data
        trend_insight = _detect_actual_trend(profile, cols, temp_cols[0], num_cols[:3])
        if trend_insight:
            results.append(trend_insight)

    # 8. Domain-specific anomalies for manufacturing
    domain = (profile.domain_hint or "general").lower()
    if domain in ("manufacturing", "energy"):
        results.extend(_manufacturing_anomalies(profile, cols))

    return results


# ---------------------------------------------------------------------------
# Manufacturing-specific anomaly checks
# ---------------------------------------------------------------------------

# Known manufacturing column patterns and their expected ranges
_MANUFACTURING_RANGES: list[dict] = [
    {
        "keywords": ["temperature", "temp"],
        "name": "Furnace Temperature",
        "min": 1400,
        "max": 1700,
        "unit": "°C",
        "context": "steelmaking furnace operating range",
    },
    {
        "keywords": ["power_factor", "pf"],
        "name": "Power Factor",
        "min": 0.80,
        "max": 1.0,
        "unit": "",
        "context": "electrical power factor should be ≥ 0.85",
    },
    {
        "keywords": ["kva", "power_kva"],
        "name": "KVA Rating",
        "min": 0,
        "max": 2000,
        "unit": "kVA",
        "context": "furnace power consumption",
    },
]


def _detect_actual_trend(
    profile: TableProfile,
    cols: dict,
    temp_col: str,
    num_cols: list[str],
) -> Insight | None:
    """Detect an actual trend from sample data — not just flag that analysis is possible.

    Returns an insight only if there's a real, quantified finding
    (e.g., values increasing/decreasing by X% over the time range).
    """
    temp_info = cols.get(temp_col, {})
    samples = temp_info.get("sample_values", [])

    if len(samples) < 5:
        return None

    # For each numeric column, check if there's a clear directional trend
    # by comparing the first-third average to the last-third average
    for num_col in num_cols:
        num_info = cols.get(num_col, {})
        num_samples = num_info.get("sample_values", [])
        stats = num_info.get("stats")

        if not num_samples or not stats or len(num_samples) < 6:
            continue

        # Convert to floats, skip non-numeric
        values = []
        for s in num_samples:
            try:
                values.append(float(str(s).replace(",", "")))
            except (ValueError, TypeError):
                pass

        if len(values) < 6:
            continue

        # Compare first third to last third
        third = len(values) // 3
        first_avg = sum(values[:third]) / third
        last_avg = sum(values[-third:]) / third

        if first_avg == 0:
            continue

        pct_change = ((last_avg - first_avg) / abs(first_avg)) * 100

        # Only report if change is significant (> 15%)
        if abs(pct_change) < 15:
            continue

        direction = "increased" if pct_change > 0 else "decreased"
        severity = "warning" if abs(pct_change) > 30 else "info"

        return Insight(
            id=str(uuid.uuid4()),
            insight_type="trend",
            severity=severity,
            impact_score=min(int(abs(pct_change) / 2) + 30, 80),
            title=f"{num_col} {direction} by {abs(pct_change):.0f}% in {profile.table_name}",
            description=(
                f"{num_col} {direction} from avg {first_avg:,.2f} (early period) to "
                f"{last_avg:,.2f} (recent period), a {abs(pct_change):.1f}% change. "
                f"Mean: {stats['mean']:,.2f}, Std dev: {stats.get('stdev', 0):,.2f}."
            ),
            source_tables=[profile.table_name],
            source_columns=[temp_col, num_col],
            evidence={
                "temporal_column": temp_col,
                "metric_column": num_col,
                "first_period_avg": round(first_avg, 2),
                "last_period_avg": round(last_avg, 2),
                "pct_change": round(pct_change, 1),
                "direction": direction,
                "chart_spec": {
                    "type": "line",
                    "x": temp_col,
                    "y": [num_col],
                    "title": f"{num_col} trend over {temp_col}",
                },
            },
            suggested_actions=[
                f"Investigate what caused {num_col} to {direction[:8]}e by {abs(pct_change):.0f}%",
                f"Check if the {direction[:8]}e in {num_col} correlates with operational changes",
            ],
        )

    return None


def _manufacturing_anomalies(
    profile: TableProfile,
    cols: dict,
) -> list[Insight]:
    """Check for manufacturing-specific anomalies based on column names and values."""
    results: list[Insight] = []

    for col_name, info in cols.items():
        stats = info.get("stats")
        if not stats:
            continue

        col_lower = col_name.lower()
        for rule in _MANUFACTURING_RANGES:
            matched = any(kw in col_lower for kw in rule["keywords"])
            if not matched:
                continue

            val_min = stats.get("min")
            val_max = stats.get("max")

            if val_min is not None and val_min < rule["min"]:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="warning",
                    impact_score=55,
                    title=f"{rule['name']} below expected range in {profile.table_name}.{col_name}",
                    description=(
                        f"{col_name} has minimum value {val_min}{rule['unit']} "
                        f"which is below the expected range "
                        f"({rule['min']}-{rule['max']}{rule['unit']}). "
                        f"Context: {rule['context']}."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={
                        "value": val_min,
                        "expected_min": rule["min"],
                        "expected_max": rule["max"],
                        "rule": rule["name"],
                    },
                    suggested_actions=[
                        f"Check if {col_name} readings below {rule['min']}{rule['unit']} are measurement errors",
                        f"Investigate operating conditions during low {rule['name']} readings",
                    ],
                ))

            if val_max is not None and val_max > rule["max"]:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="warning",
                    impact_score=55,
                    title=f"{rule['name']} above expected range in {profile.table_name}.{col_name}",
                    description=(
                        f"{col_name} has maximum value {val_max}{rule['unit']} "
                        f"which is above the expected range "
                        f"({rule['min']}-{rule['max']}{rule['unit']}). "
                        f"Context: {rule['context']}."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={
                        "value": val_max,
                        "expected_min": rule["min"],
                        "expected_max": rule["max"],
                        "rule": rule["name"],
                    },
                    suggested_actions=[
                        f"Check if {col_name} readings above {rule['max']}{rule['unit']} indicate equipment issues",
                        f"Review safety limits for {rule['name']}",
                    ],
                ))
            break  # Only match first rule per column

    return results
