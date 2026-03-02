"""Per-table anomaly scanning — outliers, null spikes, impossible values, rare categories."""

from __future__ import annotations

import logging
import re
import uuid

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


def _humanize_table(name: str) -> str:
    """Convert table name to human-readable form: 08_nary_segments → Nary Segments."""
    cleaned = re.sub(r"^\d+[_\s]*", "", name)
    return cleaned.replace("_", " ").title() if cleaned else name.replace("_", " ").title()


def _humanize_col(name: str) -> str:
    """Convert column name to human-readable form: downtime_hrs → Downtime Hrs."""
    return name.replace("_", " ").title()


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
            h_col = _humanize_col(col_name)
            h_table = _humanize_table(profile.table_name)
            results.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="anomaly",
                severity="warning" if pct > 30 else "info",
                impact_score=min(int(pct), 80),
                title=f"{pct}% of {h_col} values are missing in {h_table}",
                description=(
                    f"{h_col} has {null_count:,} missing values out of {row_count:,} rows ({pct}%). "
                    f"This could affect the reliability of any analysis involving this field."
                ),
                source_tables=[profile.table_name],
                source_columns=[col_name],
                evidence={"null_count": null_count, "row_count": row_count, "pct": pct},
                suggested_actions=[
                    f"Check why {h_col} has so many missing values — is it optional or a data entry gap?",
                    f"Consider filling in missing {h_col} values or flagging affected rows",
                ],
            ))

        # 2. Numeric outliers: values > 2.5 stdev from mean
        # (2.5σ catches ~1.2% of values in a normal distribution — reasonable for 100-row samples)
        if stats and "stdev" in stats and stats["stdev"] > 0:
            mean = stats["mean"]
            stdev = stats["stdev"]
            low_bound = mean - 2.5 * stdev
            high_bound = mean + 2.5 * stdev

            outlier_samples = []
            for s in samples:
                try:
                    v = float(str(s).replace(",", ""))
                    if v < low_bound or v > high_bound:
                        outlier_samples.append(s)
                except (ValueError, TypeError):
                    pass

            if outlier_samples:
                # Parse all outlier values for range info
                outlier_vals = []
                for s in outlier_samples:
                    try:
                        outlier_vals.append(float(str(s).replace(",", "")))
                    except (ValueError, TypeError):
                        pass
                outlier_min = min(outlier_vals) if outlier_vals else None
                outlier_max = max(outlier_vals) if outlier_vals else None

                count_desc = f"{len(outlier_samples)} outlier value{'s' if len(outlier_samples) != 1 else ''}"
                range_desc = ""
                if outlier_min is not None and outlier_max is not None and outlier_min != outlier_max:
                    range_desc = f" ranging from {outlier_min:,.2f} to {outlier_max:,.2f}"
                elif outlier_min is not None:
                    range_desc = f" ({outlier_min:,.2f})"

                h_col = _humanize_col(col_name)
                h_table = _humanize_table(profile.table_name)
                outlier_pct = round(len(outlier_samples) / len(samples) * 100, 1) if samples else 0
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="warning" if len(outlier_samples) > 3 else "info",
                    impact_score=min(45 + len(outlier_samples) * 2, 80),
                    title=f"{len(outlier_samples)} unusual {h_col} values found in {h_table}",
                    description=(
                        f"{h_col} has {len(outlier_samples)} values that are far outside the normal range "
                        f"(typical range: {low_bound:,.0f} to {high_bound:,.0f}){range_desc}. "
                        f"That's {outlier_pct}% of sampled values — worth checking if these are "
                        f"data entry errors or genuine outliers."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={
                        "mean": stats["mean"],
                        "stdev": stats["stdev"],
                        "low_bound": round(low_bound, 2),
                        "high_bound": round(high_bound, 2),
                        "outlier_count": len(outlier_samples),
                        "sample_size": len(samples),
                        "outlier_pct": outlier_pct,
                        "outlier_range": {"min": outlier_min, "max": outlier_max},
                        "outlier_samples": outlier_samples[:10],
                    },
                    suggested_actions=[
                        f"Review the {len(outlier_samples)} unusual {h_col} values — are they typos or real?",
                        f"If these are genuine, investigate what caused these extreme {h_col} readings",
                    ],
                ))

        # 3. Impossible values in currency columns (negative amounts)
        if sem_type == "numeric_currency" and stats:
            if stats.get("min", 0) < 0:
                h_col = _humanize_col(col_name)
                h_table = _humanize_table(profile.table_name)
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="critical",
                    impact_score=70,
                    title=f"{h_col} in {h_table} has negative amounts",
                    description=(
                        f"{h_col} contains negative values (lowest: {stats['min']:,.2f}). "
                        f"Money fields usually shouldn't go negative — this could mean "
                        f"refunds, credits, or a data entry error."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={"min": stats["min"], "column_type": "numeric_currency"},
                    suggested_actions=[
                        f"Check if the negative {h_col} values are refunds/credits (valid) or mistakes",
                        "If they're refunds, consider tracking them in a separate field for clarity",
                    ],
                ))

        # 4. Impossible values in percentage columns (> 100 or < 0)
        if sem_type == "numeric_percentage" and stats:
            if stats.get("max", 0) > 100 or stats.get("min", 0) < 0:
                h_col = _humanize_col(col_name)
                h_table = _humanize_table(profile.table_name)
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="critical",
                    impact_score=65,
                    title=f"{h_col} in {h_table} has impossible percentage values",
                    description=(
                        f"{h_col} has values outside the 0–100% range "
                        f"(lowest: {stats.get('min')}, highest: {stats.get('max')}). "
                        f"Percentages should stay between 0 and 100 — this likely indicates a data issue."
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={"min": stats.get("min"), "max": stats.get("max")},
                    suggested_actions=[
                        f"Fix {h_col} values that fall outside 0–100%",
                        "Check if the values are stored as decimals (e.g. 0.85 instead of 85%)",
                    ],
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

    # 7. Time-based: detect actual trends from sample data (not just "analysis possible")
    # NOTE: meta-observations like "time series data available" are suppressed from Feed.
    temp_cols = [c for c, i in cols.items() if i.get("semantic_type") == "temporal"]
    num_cols = [
        c for c, i in cols.items()
        if i.get("semantic_type") in ("numeric_metric", "numeric_currency", "numeric_percentage")
    ]

    if temp_cols and num_cols:
        # Check every numeric column for trends (not just the first 3)
        for nc in num_cols:
            trend_insight = _detect_actual_trend(profile, cols, temp_cols[0], [nc])
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
    {
        "keywords": ["sec", "specific_energy", "kwh_per_ton"],
        "name": "Specific Energy Consumption",
        "min": 350,
        "max": 700,
        "unit": " kWh/ton",
        "context": "SEC good <500, average 500-600, poor >600. Above 700 suggests furnace issues",
    },
    {
        "keywords": ["yield_pct", "yield_percent"],
        "name": "Yield",
        "min": 80,
        "max": 98,
        "unit": "%",
        "context": "output/input weight ratio. Below 85% likely indicates measurement error",
    },
    {
        "keywords": ["tap_to_tap"],
        "name": "Tap-to-Tap Time",
        "min": 30,
        "max": 120,
        "unit": " min",
        "context": "total cycle time. Above 120 min suggests operational problems",
    },
    {
        "keywords": ["electrode_consumption"],
        "name": "Electrode Consumption",
        "min": 0,
        "max": 6,
        "unit": " kg/ton",
        "context": "electrode consumption per ton of steel produced. Above 5 kg/ton is poor",
    },
    {
        "keywords": ["rejection_rate", "rejection_pct"],
        "name": "Rejection Rate",
        "min": 0,
        "max": 5,
        "unit": "%",
        "context": "percentage of output rejected. Above 3% is poor, above 5% is systemic",
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

        # Only report if change is significant (> 10%)
        if abs(pct_change) < 10:
            continue

        direction = "increased" if pct_change > 0 else "decreased"
        direction_verb = "going up" if pct_change > 0 else "going down"
        severity = "warning" if abs(pct_change) > 25 else "info"
        h_col = _humanize_col(num_col)
        h_table = _humanize_table(profile.table_name)
        h_time = _humanize_col(temp_col)

        return Insight(
            id=str(uuid.uuid4()),
            insight_type="trend",
            severity=severity,
            impact_score=min(int(abs(pct_change) / 2) + 30, 80),
            title=f"{h_col} is {direction_verb} — {abs(pct_change):.0f}% change in {h_table}",
            description=(
                f"{h_col} has {direction} by {abs(pct_change):.0f}% over time, "
                f"moving from an average of {first_avg:,.1f} (earlier) to {last_avg:,.1f} (recent). "
                f"This is a notable shift worth investigating."
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
                    "title": f"{h_col} over time",
                },
            },
            suggested_actions=[
                f"Investigate what's driving the {abs(pct_change):.0f}% change in {h_col}",
                f"Check if this {h_col} trend lines up with any operational or process changes",
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

            h_col = _humanize_col(col_name)
            h_table = _humanize_table(profile.table_name)

            if val_min is not None and val_min < rule["min"]:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="warning",
                    impact_score=55,
                    title=f"{rule['name']} is unusually low in {h_table} ({val_min}{rule['unit']})",
                    description=(
                        f"{h_col} dropped to {val_min}{rule['unit']}, which is below "
                        f"the expected range of {rule['min']}–{rule['max']}{rule['unit']}. "
                        f"{rule['context']}."
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
                        f"Check if the low {rule['name']} reading ({val_min}{rule['unit']}) is a measurement error",
                        f"Investigate what operating conditions led to low {rule['name']}",
                    ],
                ))

            if val_max is not None and val_max > rule["max"]:
                results.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity="warning",
                    impact_score=55,
                    title=f"{rule['name']} is unusually high in {h_table} ({val_max}{rule['unit']})",
                    description=(
                        f"{h_col} reached {val_max}{rule['unit']}, which exceeds "
                        f"the expected range of {rule['min']}–{rule['max']}{rule['unit']}. "
                        f"{rule['context']}."
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
                        f"Check if the high {rule['name']} reading ({val_max}{rule['unit']}) signals equipment issues",
                        f"Review safety limits for {rule['name']}",
                    ],
                ))
            break  # Only match first rule per column

    return results
