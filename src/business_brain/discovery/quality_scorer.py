"""Unified data quality scoring across multiple dimensions.

Pure-function module — no external dependencies, only Python stdlib.
Computes quality scores for completeness, uniqueness, consistency,
accuracy, and freshness, then combines them into a weighted report.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DimensionScore:
    """Score for a single quality dimension."""

    dimension: str  # "completeness", "uniqueness", "consistency", "accuracy", "freshness"
    score: float  # 0-100
    weight: float  # 0-1
    issues: list[str]  # specific problems found
    details: dict  # dimension-specific metrics


@dataclass
class QualityReport:
    """Comprehensive quality report across all dimensions."""

    overall_score: float  # 0-100 weighted average
    grade: str  # "A" (90+), "B" (80+), "C" (70+), "D" (60+), "F" (<60)
    dimensions: list[DimensionScore]
    critical_issues: list[str]  # issues that are urgent
    recommendations: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DATE_PATTERNS: list[tuple[str, str]] = [
    # (regex, strptime format)
    (r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", "%Y-%m-%dT%H:%M:%S"),
    (r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", "%Y-%m-%d %H:%M:%S"),
    (r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    (r"^\d{2}/\d{2}/\d{4}$", "%d/%m/%Y"),
    (r"^\d{2}-\d{2}-\d{4}$", "%d-%m-%Y"),
    (r"^\d{4}/\d{2}/\d{2}$", "%Y/%m/%d"),
]


def _try_parse_date(value: str) -> datetime | None:
    """Attempt to parse *value* as a date using common patterns."""
    if not isinstance(value, str):
        return None
    value_stripped = value.strip()
    for pattern, fmt in _DATE_PATTERNS:
        if re.match(pattern, value_stripped):
            try:
                return datetime.strptime(value_stripped[:len(fmt.replace("%", "x"))], fmt)
            except (ValueError, IndexError):
                # strptime format length trick may not be precise; just try directly
                pass
            try:
                return datetime.strptime(value_stripped, fmt)
            except ValueError:
                continue
    return None


def _is_null(value: object) -> bool:
    """Return True if *value* should be considered null/missing."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ("", "null", "NULL", "None", "N/A", "n/a", "NA", "na"):
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _classify_type(value: object) -> str:
    """Classify a non-null value into a coarse type bucket."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        stripped = value.strip()
        # Check numeric-looking strings
        if re.match(r"^-?\d+$", stripped):
            return "int-like"
        if re.match(r"^-?\d+\.\d+$", stripped):
            return "float-like"
        if _try_parse_date(stripped) is not None:
            return "date-like"
        return "text"
    return "other"


def _assign_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _numeric_values(rows: list[dict], col: str) -> list[float]:
    """Extract numeric values from a column, coercing strings where possible."""
    nums: list[float] = []
    for row in rows:
        v = row.get(col)
        if _is_null(v):
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums.append(float(v))
        elif isinstance(v, str):
            try:
                nums.append(float(v.strip()))
            except ValueError:
                pass
    return nums


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

def score_completeness(rows: list[dict], columns: list[str]) -> DimensionScore:
    """Score based on null/missing value rates per column.

    Score = 100 * (1 - avg_null_rate).
    Issues: columns with > 10 % nulls.
    """
    if not rows or not columns:
        return DimensionScore(
            dimension="completeness",
            score=0.0 if not rows else 100.0,
            weight=0.25,
            issues=["No data to evaluate"] if not rows else [],
            details={"column_null_rates": {}, "total_values": 0, "total_nulls": 0},
        )

    column_null_rates: dict[str, float] = {}
    total_values = 0
    total_nulls = 0
    issues: list[str] = []

    for col in columns:
        values = [row.get(col) for row in rows]
        n = len(values)
        if n == 0:
            column_null_rates[col] = 1.0
            continue
        null_count = sum(1 for v in values if _is_null(v))
        rate = null_count / n
        column_null_rates[col] = rate
        total_values += n
        total_nulls += null_count
        if rate > 0.10:
            issues.append(f"Column '{col}' has {rate * 100:.1f}% null values")

    avg_null_rate = (total_nulls / total_values) if total_values else 0
    score = _clamp(100 * (1 - avg_null_rate))

    return DimensionScore(
        dimension="completeness",
        score=round(score, 2),
        weight=0.25,
        issues=issues,
        details={
            "column_null_rates": {k: round(v, 4) for k, v in column_null_rates.items()},
            "total_values": total_values,
            "total_nulls": total_nulls,
        },
    )


def score_uniqueness(rows: list[dict], key_columns: list[str]) -> DimensionScore:
    """Score based on duplicate detection in key columns.

    Score = 100 * (unique_rows / total_rows).
    Issues: exact duplicates, near-duplicates (case-insensitive matches).
    """
    if not rows:
        return DimensionScore(
            dimension="uniqueness",
            score=0.0,
            weight=0.2,
            issues=["No data to evaluate"],
            details={"total_rows": 0, "unique_rows": 0, "duplicate_rows": 0},
        )

    if not key_columns:
        # Fall back to using all columns
        key_columns = list(rows[0].keys()) if rows else []

    total = len(rows)
    issues: list[str] = []

    # Exact duplicate detection
    seen_exact: dict[tuple, int] = {}
    for row in rows:
        key = tuple(row.get(c) for c in key_columns)
        seen_exact[key] = seen_exact.get(key, 0) + 1

    exact_dupes = sum(count - 1 for count in seen_exact.values() if count > 1)
    unique_rows = sum(1 for count in seen_exact.values())

    if exact_dupes > 0:
        issues.append(f"{exact_dupes} exact duplicate row(s) found across key columns")

    # Near-duplicate detection (case-insensitive)
    seen_near: dict[tuple, int] = {}
    for row in rows:
        key = tuple(
            str(row.get(c, "")).strip().lower() if isinstance(row.get(c), str) else row.get(c)
            for c in key_columns
        )
        seen_near[key] = seen_near.get(key, 0) + 1

    near_dupes = sum(count - 1 for count in seen_near.values() if count > 1)
    case_only_dupes = near_dupes - exact_dupes
    if case_only_dupes > 0:
        issues.append(f"{case_only_dupes} near-duplicate row(s) found (differ only by case)")

    score = _clamp(100 * (unique_rows / total)) if total else 0.0

    return DimensionScore(
        dimension="uniqueness",
        score=round(score, 2),
        weight=0.2,
        issues=issues,
        details={
            "total_rows": total,
            "unique_rows": unique_rows,
            "duplicate_rows": exact_dupes,
            "near_duplicate_rows": case_only_dupes,
        },
    )


def score_consistency(rows: list[dict], columns: list[str]) -> DimensionScore:
    """Score based on format/type consistency within columns.

    Checks: mixed types in same column, inconsistent casing, inconsistent formats.
    Score = 100 * (consistent_values / total_values).
    """
    if not rows or not columns:
        return DimensionScore(
            dimension="consistency",
            score=0.0 if not rows else 100.0,
            weight=0.25,
            issues=["No data to evaluate"] if not rows else [],
            details={"column_details": {}, "total_checked": 0, "total_inconsistent": 0},
        )

    total_checked = 0
    total_inconsistent = 0
    issues: list[str] = []
    column_details: dict[str, dict] = {}

    for col in columns:
        non_null_values = [row.get(col) for row in rows if not _is_null(row.get(col))]
        if not non_null_values:
            column_details[col] = {"type_mix": {}, "casing_inconsistent": False}
            continue

        col_checked = len(non_null_values)
        col_inconsistent = 0

        # --- Type consistency ---
        type_counts: dict[str, int] = {}
        for v in non_null_values:
            t = _classify_type(v)
            type_counts[t] = type_counts.get(t, 0) + 1

        # Merge int/int-like and float/float-like for comparison
        merged: dict[str, int] = {}
        merge_map = {"int-like": "int", "float-like": "float"}
        for t, cnt in type_counts.items():
            key = merge_map.get(t, t)
            merged[key] = merged.get(key, 0) + cnt

        has_mixed_types = len(merged) > 1
        if has_mixed_types:
            dominant = max(merged, key=merged.get)  # type: ignore[arg-type]
            minority_count = sum(c for t, c in merged.items() if t != dominant)
            col_inconsistent += minority_count
            issues.append(
                f"Column '{col}' has mixed types: {dict(merged)}"
            )

        # --- Casing consistency (only for text values) ---
        string_vals = [str(v) for v in non_null_values if isinstance(v, str)]
        casing_inconsistent = False
        if len(string_vals) > 1:
            lowered: dict[str, list[str]] = {}
            for s in string_vals:
                low = s.strip().lower()
                lowered.setdefault(low, []).append(s.strip())
            for low, variants in lowered.items():
                unique_variants = set(variants)
                if len(unique_variants) > 1:
                    casing_inconsistent = True
                    # Count the non-dominant variants as inconsistent
                    variant_counts = {}
                    for v in variants:
                        variant_counts[v] = variant_counts.get(v, 0) + 1
                    dominant_variant = max(variant_counts, key=variant_counts.get)  # type: ignore[arg-type]
                    minority = sum(c for v, c in variant_counts.items() if v != dominant_variant)
                    col_inconsistent += minority
                    issues.append(
                        f"Column '{col}' has inconsistent casing: {sorted(unique_variants)}"
                    )

        total_checked += col_checked
        total_inconsistent += col_inconsistent

        column_details[col] = {
            "type_mix": type_counts,
            "casing_inconsistent": casing_inconsistent,
        }

    score = _clamp(100 * ((total_checked - total_inconsistent) / total_checked)) if total_checked else 100.0

    return DimensionScore(
        dimension="consistency",
        score=round(score, 2),
        weight=0.25,
        issues=issues,
        details={
            "column_details": column_details,
            "total_checked": total_checked,
            "total_inconsistent": total_inconsistent,
        },
    )


def score_accuracy(
    rows: list[dict],
    columns: list[str],
    rules: list[dict] | None = None,
) -> DimensionScore:
    """Score based on value validity.

    Built-in checks:
    - Negative values in typically-positive columns (containing 'price', 'amount',
      'count', 'quantity', 'age', 'total', 'revenue', 'cost', 'salary').
    - Outliers using IQR method (values beyond 1.5 * IQR from Q1/Q3).

    Custom *rules*: list of dicts with keys ``column``, and optionally ``min``,
    ``max``, ``pattern`` (regex).

    Score = 100 * (valid_values / total_checked).
    """
    if not rows or not columns:
        return DimensionScore(
            dimension="accuracy",
            score=0.0 if not rows else 100.0,
            weight=0.2,
            issues=["No data to evaluate"] if not rows else [],
            details={"total_checked": 0, "total_invalid": 0, "outlier_counts": {}},
        )

    issues: list[str] = []
    total_checked = 0
    total_invalid = 0
    outlier_counts: dict[str, int] = {}
    rule_violations: dict[str, int] = {}

    positive_keywords = {"price", "amount", "count", "quantity", "age", "total", "revenue", "cost", "salary"}

    for col in columns:
        nums = _numeric_values(rows, col)
        if not nums:
            continue

        col_invalid = 0
        total_checked += len(nums)

        # Negative check for typically-positive columns
        col_lower = col.lower()
        if any(kw in col_lower for kw in positive_keywords):
            neg_count = sum(1 for n in nums if n < 0)
            if neg_count:
                col_invalid += neg_count
                issues.append(f"Column '{col}' has {neg_count} negative value(s)")

        # Outlier detection using IQR
        if len(nums) >= 4:
            sorted_nums = sorted(nums)
            n = len(sorted_nums)
            q1 = sorted_nums[n // 4]
            q3 = sorted_nums[(3 * n) // 4]
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = sum(1 for v in nums if v < lower_bound or v > upper_bound)
            if outliers:
                outlier_counts[col] = outliers
                col_invalid += outliers
                issues.append(f"Column '{col}' has {outliers} outlier(s) by IQR method")

        total_invalid += col_invalid

    # Custom rules
    if rules:
        for rule in rules:
            col = rule.get("column", "")
            if col not in columns:
                continue
            violations = 0
            for row in rows:
                v = row.get(col)
                if _is_null(v):
                    continue
                total_checked += 1

                violated = False
                if "min" in rule:
                    try:
                        if float(v) < rule["min"]:
                            violated = True
                    except (TypeError, ValueError):
                        pass
                if "max" in rule:
                    try:
                        if float(v) > rule["max"]:
                            violated = True
                    except (TypeError, ValueError):
                        pass
                if "pattern" in rule:
                    if isinstance(v, str) and not re.match(rule["pattern"], v):
                        violated = True

                if violated:
                    violations += 1
                    total_invalid += 1

            if violations:
                rule_violations[col] = violations
                issues.append(f"Column '{col}' has {violations} rule violation(s)")

    score = _clamp(100 * ((total_checked - total_invalid) / total_checked)) if total_checked else 100.0

    return DimensionScore(
        dimension="accuracy",
        score=round(score, 2),
        weight=0.2,
        issues=issues,
        details={
            "total_checked": total_checked,
            "total_invalid": total_invalid,
            "outlier_counts": outlier_counts,
            "rule_violations": rule_violations,
        },
    )


def score_freshness(
    rows: list[dict],
    date_columns: list[str],
    reference_date: str | None = None,
) -> DimensionScore:
    """Score based on how recent the data is.

    Looks at the maximum date found in *date_columns*.
    Score: 100 if within 1 day, -10 per additional day old, minimum 0.
    If no date columns are provided or no dates can be parsed, returns 80
    (neutral score).
    """
    if not rows:
        return DimensionScore(
            dimension="freshness",
            score=0.0,
            weight=0.1,
            issues=["No data to evaluate"],
            details={"max_date": None, "days_old": None},
        )

    if not date_columns:
        return DimensionScore(
            dimension="freshness",
            score=80.0,
            weight=0.1,
            issues=[],
            details={"max_date": None, "days_old": None, "note": "No date columns provided"},
        )

    # Determine reference datetime
    if reference_date:
        ref_dt = _try_parse_date(reference_date)
        if ref_dt is None:
            ref_dt = datetime.now()
    else:
        ref_dt = datetime.now()

    max_date: datetime | None = None
    dates_found = 0

    for col in date_columns:
        for row in rows:
            v = row.get(col)
            if _is_null(v):
                continue
            if isinstance(v, datetime):
                dt = v
            elif isinstance(v, str):
                dt = _try_parse_date(v)
            else:
                continue
            if dt is not None:
                dates_found += 1
                if max_date is None or dt > max_date:
                    max_date = dt

    if max_date is None:
        return DimensionScore(
            dimension="freshness",
            score=80.0,
            weight=0.1,
            issues=["Could not parse any dates from date columns"],
            details={"max_date": None, "days_old": None, "dates_found": 0},
        )

    days_old = (ref_dt - max_date).days
    if days_old < 0:
        days_old = 0  # future dates treated as fresh

    issues: list[str] = []
    if days_old <= 1:
        score = 100.0
    else:
        score = _clamp(100.0 - 10 * (days_old - 1))
        if days_old > 7:
            issues.append(f"Data is {days_old} days old")

    return DimensionScore(
        dimension="freshness",
        score=round(score, 2),
        weight=0.1,
        issues=issues,
        details={
            "max_date": max_date.isoformat(),
            "days_old": days_old,
            "dates_found": dates_found,
        },
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS: dict[str, float] = {
    "completeness": 0.25,
    "uniqueness": 0.20,
    "consistency": 0.25,
    "accuracy": 0.20,
    "freshness": 0.10,
}


def _auto_detect_key_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if "id" in c.lower()]


def _auto_detect_date_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if "date" in c.lower() or "time" in c.lower()]


def _generate_recommendations(dimensions: list[DimensionScore]) -> list[str]:
    recs: list[str] = []
    for dim in dimensions:
        if dim.dimension == "completeness" and dim.score < 90:
            recs.append("Fill or impute missing values in columns with high null rates.")
        if dim.dimension == "uniqueness" and dim.score < 90:
            recs.append("Investigate and resolve duplicate records in key columns.")
        if dim.dimension == "consistency" and dim.score < 90:
            recs.append("Standardize data types and casing across columns.")
        if dim.dimension == "accuracy" and dim.score < 90:
            recs.append("Review and correct outlier values or negative values in positive-only columns.")
        if dim.dimension == "freshness" and dim.score < 90:
            recs.append("Ensure data pipelines are refreshing data regularly.")
    return recs


def compute_quality_report(
    rows: list[dict],
    columns: list[str] | None = None,
    key_columns: list[str] | None = None,
    date_columns: list[str] | None = None,
    rules: list[dict] | None = None,
    weights: dict[str, float] | None = None,
) -> QualityReport:
    """Compute a comprehensive quality report across all dimensions.

    Parameters
    ----------
    rows:
        The data as a list of dicts (each dict is one row).
    columns:
        Columns to evaluate.  Auto-detected from the first row if *None*.
    key_columns:
        Columns used for uniqueness checks.  Auto-detected (columns
        containing 'id') if *None*.
    date_columns:
        Columns used for freshness checks.  Auto-detected (columns
        containing 'date' or 'time') if *None*.
    rules:
        Optional list of custom accuracy rules.
    weights:
        Override dimension weights.  Keys are dimension names; values are
        floats 0-1.  Missing dimensions use default weights.
    """
    # Auto-detect columns from first row
    if columns is None:
        columns = list(rows[0].keys()) if rows else []

    if key_columns is None:
        key_columns = _auto_detect_key_columns(columns)

    if date_columns is None:
        date_columns = _auto_detect_date_columns(columns)

    # Merge user weights with defaults
    effective_weights = dict(_DEFAULT_WEIGHTS)
    if weights:
        effective_weights.update(weights)

    # Normalise weights so they sum to 1
    total_w = sum(effective_weights.values())
    if total_w > 0:
        effective_weights = {k: v / total_w for k, v in effective_weights.items()}

    # Score each dimension
    completeness = score_completeness(rows, columns)
    completeness.weight = effective_weights.get("completeness", 0.25)

    uniqueness = score_uniqueness(rows, key_columns)
    uniqueness.weight = effective_weights.get("uniqueness", 0.20)

    consistency = score_consistency(rows, columns)
    consistency.weight = effective_weights.get("consistency", 0.25)

    accuracy = score_accuracy(rows, columns, rules)
    accuracy.weight = effective_weights.get("accuracy", 0.20)

    freshness = score_freshness(rows, date_columns)
    freshness.weight = effective_weights.get("freshness", 0.10)

    dimensions = [completeness, uniqueness, consistency, accuracy, freshness]

    # Weighted overall score
    overall = sum(d.score * d.weight for d in dimensions)
    overall = round(_clamp(overall), 2)

    grade = _assign_grade(overall)

    # Critical issues: any dimension below 50
    critical_issues: list[str] = []
    for d in dimensions:
        if d.score < 50:
            critical_issues.append(f"{d.dimension.capitalize()} score critically low ({d.score})")
            critical_issues.extend(d.issues)

    recommendations = _generate_recommendations(dimensions)

    summary_parts = [f"Overall quality: {grade} ({overall}/100)."]
    for d in dimensions:
        summary_parts.append(f"  {d.dimension.capitalize()}: {d.score}")
    if critical_issues:
        summary_parts.append(f"  Critical issues: {len(critical_issues)}")

    return QualityReport(
        overall_score=overall,
        grade=grade,
        dimensions=dimensions,
        critical_issues=critical_issues,
        recommendations=recommendations,
        summary="\n".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_quality(report_a: QualityReport, report_b: QualityReport) -> dict:
    """Compare two quality reports.

    Returns a dict with:
    - ``overall_change``: difference in overall score (b - a).
    - ``grade_change``: tuple (grade_a, grade_b).
    - ``dimensions``: per-dimension change dicts.
    """
    dim_a = {d.dimension: d for d in report_a.dimensions}
    dim_b = {d.dimension: d for d in report_b.dimensions}

    all_dims = sorted(set(dim_a) | set(dim_b))
    dimension_changes: dict[str, dict] = {}
    for dim in all_dims:
        score_a = dim_a[dim].score if dim in dim_a else None
        score_b = dim_b[dim].score if dim in dim_b else None
        change = None
        if score_a is not None and score_b is not None:
            change = round(score_b - score_a, 2)
        status = "unchanged"
        if change is not None:
            if change > 0:
                status = "improved"
            elif change < 0:
                status = "degraded"
        dimension_changes[dim] = {
            "score_a": score_a,
            "score_b": score_b,
            "change": change,
            "status": status,
        }

    return {
        "overall_change": round(report_b.overall_score - report_a.overall_score, 2),
        "grade_change": (report_a.grade, report_b.grade),
        "dimensions": dimension_changes,
    }
