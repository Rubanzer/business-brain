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

        # 2. Numeric outliers: values > 3 stdev from mean (raised from 2σ to reduce noise)
        if stats and "stdev" in stats and stats["stdev"] > 0:
            mean = stats["mean"]
            stdev = stats["stdev"]
            low_bound = mean - 3 * stdev
            high_bound = mean + 3 * stdev

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
                        f"{col_name} has values beyond 3 standard deviations from mean "
                        f"(mean={stats['mean']:.2f}, stdev={stats['stdev']:.2f}). "
                        f"Examples: {outlier_samples[:3]}. "
                        f"These extreme values may indicate measurement errors or exceptional events."
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
        # NOTE: "High cardinality categorical" is a data quality note, not a business insight.
        # Suppressed from Feed — the quality gate will filter it anyway.
        if sem_type == "categorical" and cardinality > 5 and samples:
            if row_count > 0 and cardinality > row_count * 0.5:
                # Log for quality tab but don't create a feed insight
                logger.debug(
                    "High cardinality categorical %s.%s: %d unique / %d rows",
                    profile.table_name, col_name, cardinality, row_count,
                )

        # 6. Constant columns: cardinality of 1 (useless for analysis)
        # NOTE: "Constant column" is a data quality note, not a business insight.
        # Suppressed from Feed — logged for quality tracking only.
        if cardinality == 1 and row_count > 1:
            logger.debug(
                "Constant column %s.%s: 1 unique value across %d rows",
                profile.table_name, col_name, row_count,
            )

    # 7. Time-based insights: detect temporal + numeric combinations for trend detection
    # NOTE: "Time series data available" is a meta-observation, not a business insight.
    # Suppressed from Feed. The system already uses this info for trend analysis internally.
    temp_cols = [c for c, i in cols.items() if i.get("semantic_type") == "temporal"]
    num_cols = [
        c for c, i in cols.items()
        if i.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ]

    if temp_cols and num_cols:
        logger.debug(
            "Time series available: %s has temporal=%s, numeric=%s",
            profile.table_name, temp_cols, num_cols[:3],
        )

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
