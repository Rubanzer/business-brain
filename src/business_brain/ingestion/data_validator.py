"""Data validation for uploaded files — hard checks and statistical outlier detection.

Validates rows *after* cleaning but *before* DB insertion. Rows that fail hard
checks are quarantined; soft-check outliers are flagged as sanctity issues but
still inserted.

Hard checks (quarantine):
    - Negative values in quantity/weight/currency columns
    - Future dates in date columns
    - Duplicate primary-key values within the upload batch
    - Missing values in identifier / primary-key columns

Soft checks (flag as SanctityIssue, still insert):
    - Z-score > 3 on numeric columns (statistical outliers)
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


class ValidationResult:
    """Result of running validation on a set of rows."""

    __slots__ = ("clean_rows", "quarantined", "outliers", "stats")

    def __init__(self) -> None:
        self.clean_rows: list[dict] = []
        self.quarantined: list[dict] = []  # [{row_index, row_data, issues}]
        self.outliers: list[dict] = []  # [{row_index, column, value, z_score, mean, stdev}]
        self.stats: dict[str, Any] = {}

    def summary(self) -> dict:
        return {
            "rows_clean": len(self.clean_rows),
            "rows_quarantined": len(self.quarantined),
            "outliers_flagged": len(self.outliers),
        }


# ---------------------------------------------------------------------------
# Column classification heuristics
# ---------------------------------------------------------------------------

_NEGATIVE_FORBIDDEN_KEYWORDS = frozenset({
    "quantity", "qty", "weight", "wt", "amount", "amt", "price", "cost",
    "total", "count", "units", "pieces", "pcs", "volume", "vol",
    "output", "production", "consumption", "revenue", "salary", "wage",
    "length", "width", "height", "area", "tonnage", "mt", "kg",
})

_DATE_KEYWORDS = frozenset({
    "date", "time", "timestamp", "datetime", "created", "updated",
    "at", "on", "when", "day", "month", "year", "period",
})


def _is_quantity_column(col_name: str, pg_type: str) -> bool:
    """Heuristic: column likely holds non-negative quantities."""
    if pg_type not in ("BIGINT", "DOUBLE PRECISION"):
        return False
    name_lower = col_name.lower().replace("_", " ")
    return any(kw in name_lower for kw in _NEGATIVE_FORBIDDEN_KEYWORDS)


def _is_date_column(col_name: str, pg_type: str) -> bool:
    """Heuristic: column likely holds dates."""
    if pg_type == "TIMESTAMP":
        return True
    name_lower = col_name.lower().replace("_", " ")
    return any(kw in name_lower for kw in _DATE_KEYWORDS)


def _is_identifier_column(col_name: str, col_index: int) -> bool:
    """Heuristic: column is likely a primary key / identifier."""
    if col_index == 0:
        return True
    name_lower = col_name.lower()
    return any(kw in name_lower for kw in ("id", "key", "code", "number", "no"))


# ---------------------------------------------------------------------------
# Hard checks
# ---------------------------------------------------------------------------


def _check_negative(row: dict, col: str, row_index: int) -> dict | None:
    """Flag negative value in a quantity/currency column."""
    val = row.get(col)
    if val is None:
        return None
    try:
        fval = float(str(val).replace(",", ""))
        if fval < 0:
            return {
                "check": "negative_value",
                "column": col,
                "value": str(val),
                "message": f"Negative value ({val}) in quantity/currency column '{col}'",
                "severity": "critical",
            }
    except (ValueError, TypeError):
        pass
    return None


def _check_future_date(row: dict, col: str, row_index: int, now: datetime) -> dict | None:
    """Flag dates in the future."""
    val = row.get(col)
    if val is None or str(val).strip() == "":
        return None
    val_str = str(val).strip()
    # Try common date formats
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(val_str[:len(fmt) + 2], fmt)
            # Ensure both are tz-aware (or both naive) for comparison
            if dt.tzinfo is None and now.tzinfo is not None:
                dt = dt.replace(tzinfo=timezone.utc)
            elif dt.tzinfo is not None and now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            if dt > now:
                return {
                    "check": "future_date",
                    "column": col,
                    "value": val_str,
                    "message": f"Future date ({val_str}) in column '{col}'",
                    "severity": "warning",
                }
            return None  # valid, non-future date
        except ValueError:
            continue
    return None


def _check_missing_identifier(row: dict, col: str, row_index: int) -> dict | None:
    """Flag missing values in identifier columns."""
    val = row.get(col)
    if val is None or str(val).strip() == "":
        return {
            "check": "missing_identifier",
            "column": col,
            "value": None,
            "message": f"Missing value in identifier column '{col}'",
            "severity": "critical",
        }
    return None


def _check_duplicate_pks(
    rows: list[dict],
    pk_col: str,
) -> dict[int, dict]:
    """Find duplicate primary key values. Returns {row_index: issue_dict}."""
    seen: dict[str, int] = {}  # value -> first row index
    duplicates: dict[int, dict] = {}

    for i, row in enumerate(rows):
        val = str(row.get(pk_col, "")).strip()
        if not val:
            continue
        if val in seen:
            duplicates[i] = {
                "check": "duplicate_pk",
                "column": pk_col,
                "value": val,
                "message": (
                    f"Duplicate primary key '{val}' in column '{pk_col}' "
                    f"(first seen at row {seen[val]})"
                ),
                "severity": "warning",
            }
        else:
            seen[val] = i

    return duplicates


# ---------------------------------------------------------------------------
# Soft checks (z-score outlier detection)
# ---------------------------------------------------------------------------


def _compute_column_stats(rows: list[dict], col: str) -> tuple[float, float, int]:
    """Compute mean and stdev for a numeric column. Returns (mean, stdev, count)."""
    values: list[float] = []
    for row in rows:
        val = row.get(col)
        if val is None:
            continue
        try:
            fval = float(str(val).replace(",", ""))
            values.append(fval)
        except (ValueError, TypeError):
            continue

    if len(values) < 5:  # need minimum sample
        return 0.0, 0.0, 0

    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    stdev = math.sqrt(variance) if variance > 0 else 0.0

    return mean, stdev, n


def detect_outliers(
    rows: list[dict],
    col_types: dict[str, str],
    z_threshold: float = 3.0,
    max_outliers_per_col: int = 10,
) -> list[dict]:
    """Find statistical outliers (z-score > threshold) in numeric columns.

    Returns list of {row_index, column, value, z_score, mean, stdev}.
    """
    numeric_cols = [
        col for col, pg_type in col_types.items()
        if pg_type in ("BIGINT", "DOUBLE PRECISION")
    ]
    outliers: list[dict] = []

    for col in numeric_cols:
        mean, stdev, count = _compute_column_stats(rows, col)
        if stdev == 0 or count < 5:
            continue

        col_outliers = 0
        for i, row in enumerate(rows):
            if col_outliers >= max_outliers_per_col:
                break
            val = row.get(col)
            if val is None:
                continue
            try:
                fval = float(str(val).replace(",", ""))
                z = abs(fval - mean) / stdev
                if z > z_threshold:
                    outliers.append({
                        "row_index": i,
                        "column": col,
                        "value": fval,
                        "z_score": round(z, 2),
                        "mean": round(mean, 4),
                        "stdev": round(stdev, 4),
                    })
                    col_outliers += 1
            except (ValueError, TypeError):
                continue

    return outliers


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------


def validate_rows(
    rows: list[dict],
    col_types: dict[str, str],
    *,
    use_serial_pk: bool = False,
    z_threshold: float = 3.0,
) -> ValidationResult:
    """Run all validation checks on a batch of rows.

    Hard checks quarantine rows; soft checks flag outliers but keep them.

    Args:
        rows: Cleaned rows ready for insertion.
        col_types: Column name → PG type mapping.
        use_serial_pk: If True, skip duplicate-PK check (auto-generated PKs).
        z_threshold: Z-score threshold for outlier detection.

    Returns:
        ValidationResult with clean_rows, quarantined rows, and outlier flags.
    """
    result = ValidationResult()
    if not rows:
        return result

    columns = list(col_types.keys())
    now = datetime.now(timezone.utc)
    batch_id = str(uuid.uuid4())

    # Classify columns
    quantity_cols = [c for c in columns if _is_quantity_column(c, col_types[c])]
    date_cols = [c for c in columns if _is_date_column(c, col_types[c])]
    id_cols = [
        c for i, c in enumerate(columns)
        if _is_identifier_column(c, i) and col_types[c] in ("TEXT", "BIGINT")
    ]

    # Duplicate PK check (only for natural PKs)
    pk_col = columns[0] if columns else None
    dup_issues: dict[int, dict] = {}
    if pk_col and not use_serial_pk:
        dup_issues = _check_duplicate_pks(rows, pk_col)

    # Per-row hard checks
    for i, row in enumerate(rows):
        row_issues: list[dict] = []

        # Negative quantity/currency check
        for col in quantity_cols:
            issue = _check_negative(row, col, i)
            if issue:
                row_issues.append(issue)

        # Future date check
        for col in date_cols:
            issue = _check_future_date(row, col, i, now)
            if issue:
                row_issues.append(issue)

        # Missing identifier check (first column only for natural PK)
        if not use_serial_pk and pk_col:
            issue = _check_missing_identifier(row, pk_col, i)
            if issue:
                row_issues.append(issue)

        # Duplicate PK
        if i in dup_issues:
            row_issues.append(dup_issues[i])

        if row_issues:
            # Has at least one critical issue → quarantine
            has_critical = any(iss["severity"] == "critical" for iss in row_issues)
            if has_critical:
                result.quarantined.append({
                    "batch_id": batch_id,
                    "row_index": i,
                    "row_data": row,
                    "issues": row_issues,
                })
            else:
                # Warning-only issues: insert but record issues
                result.clean_rows.append(row)
                # Still track the warning issues as outlier-like info
                for iss in row_issues:
                    result.outliers.append({
                        "row_index": i,
                        "column": iss["column"],
                        "value": iss.get("value"),
                        "issue": iss["message"],
                        "severity": iss["severity"],
                    })
        else:
            result.clean_rows.append(row)

    # Soft checks: z-score outlier detection on clean rows
    z_outliers = detect_outliers(result.clean_rows, col_types, z_threshold)
    result.outliers.extend(z_outliers)

    result.stats = {
        "batch_id": batch_id,
        "total_rows": len(rows),
        "quantity_columns_checked": quantity_cols,
        "date_columns_checked": date_cols,
        "identifier_columns_checked": id_cols,
        "z_threshold": z_threshold,
    }

    logger.info(
        "Validation complete: %d clean, %d quarantined, %d outliers flagged",
        len(result.clean_rows),
        len(result.quarantined),
        len(result.outliers),
    )

    return result
