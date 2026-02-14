"""Pivot table engine — create and format pivot tables from raw row data.

Pure-function module with no external dependencies (no pandas).
Supports sum, mean, count, min, max aggregation, text/CSV formatting,
percentage breakdowns, and outlier detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class PivotCell:
    """A single cell in the pivot table."""

    row_key: str
    col_key: str
    value: float
    count: int


@dataclass
class PivotTable:
    """Complete pivot table result."""

    row_keys: list[str]  # sorted unique row header values
    col_keys: list[str]  # sorted unique column header values
    cells: dict[tuple[str, str], PivotCell]  # (row_key, col_key) -> cell
    row_totals: dict[str, float]
    col_totals: dict[str, float]
    grand_total: float
    row_field: str
    col_field: str
    value_field: str
    agg_func: str  # "sum", "mean", "count", "min", "max"
    summary: str


_VALID_AGG_FUNCS = {"sum", "mean", "count", "min", "max"}


def _safe_float(val: object) -> float | None:
    """Try to convert a value to float, return None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def create_pivot(
    rows: list[dict],
    row_field: str,
    col_field: str,
    value_field: str,
    agg_func: str = "sum",
) -> PivotTable | None:
    """Create a pivot table from row data.

    Args:
        rows: List of dicts with data.
        row_field: Column to use as row headers.
        col_field: Column to use as column headers.
        value_field: Column to aggregate.
        agg_func: Aggregation function - "sum", "mean", "count", "min", "max".

    Returns None if insufficient data (< 2 rows or missing fields).
    """
    if agg_func not in _VALID_AGG_FUNCS:
        return None

    if not rows or len(rows) < 2:
        return None

    # Check that at least some rows contain the required fields
    valid_rows: list[dict] = []
    for r in rows:
        if row_field not in r or col_field not in r:
            continue
        if r[row_field] is None or r[col_field] is None:
            continue
        # For count, we don't strictly need the value_field to be numeric
        if agg_func != "count":
            v = _safe_float(r.get(value_field))
            if v is None:
                continue
        valid_rows.append(r)

    if len(valid_rows) < 2:
        return None

    # Collect unique keys
    row_key_set: set[str] = set()
    col_key_set: set[str] = set()
    for r in valid_rows:
        row_key_set.add(str(r[row_field]))
        col_key_set.add(str(r[col_field]))

    row_keys = sorted(row_key_set)
    col_keys = sorted(col_key_set)

    # Accumulate per-cell data
    # For each cell, track: sum, count, min, max
    cell_acc: dict[tuple[str, str], dict] = {}
    for rk in row_keys:
        for ck in col_keys:
            cell_acc[(rk, ck)] = {"sum": 0.0, "count": 0, "min": None, "max": None}

    for r in valid_rows:
        rk = str(r[row_field])
        ck = str(r[col_field])
        if agg_func == "count":
            cell_acc[(rk, ck)]["count"] += 1
            cell_acc[(rk, ck)]["sum"] += 1.0
        else:
            v = _safe_float(r.get(value_field))
            if v is None:
                continue
            acc = cell_acc[(rk, ck)]
            acc["sum"] += v
            acc["count"] += 1
            if acc["min"] is None or v < acc["min"]:
                acc["min"] = v
            if acc["max"] is None or v > acc["max"]:
                acc["max"] = v

    # Build PivotCell objects
    cells: dict[tuple[str, str], PivotCell] = {}
    for (rk, ck), acc in cell_acc.items():
        if agg_func == "sum":
            value = acc["sum"]
        elif agg_func == "mean":
            value = acc["sum"] / acc["count"] if acc["count"] > 0 else 0.0
        elif agg_func == "count":
            value = float(acc["count"])
        elif agg_func == "min":
            value = acc["min"] if acc["min"] is not None else 0.0
        elif agg_func == "max":
            value = acc["max"] if acc["max"] is not None else 0.0
        else:
            value = 0.0

        cells[(rk, ck)] = PivotCell(
            row_key=rk,
            col_key=ck,
            value=value,
            count=acc["count"],
        )

    # Compute row totals (sum of cell values across columns for each row)
    row_totals: dict[str, float] = {}
    for rk in row_keys:
        row_totals[rk] = sum(cells[(rk, ck)].value for ck in col_keys)

    # Compute column totals (sum of cell values across rows for each column)
    col_totals: dict[str, float] = {}
    for ck in col_keys:
        col_totals[ck] = sum(cells[(rk, ck)].value for rk in row_keys)

    grand_total = sum(row_totals.values())

    # Build summary
    summary = (
        f"Pivot of '{value_field}' by '{row_field}' (rows) x '{col_field}' (cols), "
        f"agg={agg_func}. {len(row_keys)} rows x {len(col_keys)} cols. "
        f"Grand total: {grand_total:,.2f}"
    )

    return PivotTable(
        row_keys=row_keys,
        col_keys=col_keys,
        cells=cells,
        row_totals=row_totals,
        col_totals=col_totals,
        grand_total=grand_total,
        row_field=row_field,
        col_field=col_field,
        value_field=value_field,
        agg_func=agg_func,
        summary=summary,
    )


def format_pivot_text(pivot: PivotTable, max_cols: int = 10) -> str:
    """Format pivot table as aligned text table with row/col totals.

    Truncate to *max_cols* columns if too wide.
    """
    display_cols = pivot.col_keys[:max_cols]
    truncated = len(pivot.col_keys) > max_cols

    # Determine column widths
    # First column is the row field header
    row_header_width = max(
        len(pivot.row_field),
        *(len(rk) for rk in pivot.row_keys),
    )

    def _fmt_val(v: float) -> str:
        if v == int(v) and abs(v) < 1e12:
            return f"{int(v):,}"
        return f"{v:,.2f}"

    col_widths: dict[str, int] = {}
    for ck in display_cols:
        w = len(ck)
        for rk in pivot.row_keys:
            w = max(w, len(_fmt_val(pivot.cells[(rk, ck)].value)))
        w = max(w, len(_fmt_val(pivot.col_totals[ck])))
        col_widths[ck] = w

    total_label = "Total"
    total_width = len(total_label)
    for rk in pivot.row_keys:
        total_width = max(total_width, len(_fmt_val(pivot.row_totals[rk])))
    total_width = max(total_width, len(_fmt_val(pivot.grand_total)))

    lines: list[str] = []

    # Header line
    header_parts = [pivot.row_field.ljust(row_header_width)]
    for ck in display_cols:
        header_parts.append(ck.rjust(col_widths[ck]))
    if truncated:
        header_parts.append("...")
    header_parts.append(total_label.rjust(total_width))
    lines.append("  ".join(header_parts))

    # Separator
    sep_parts = ["-" * row_header_width]
    for ck in display_cols:
        sep_parts.append("-" * col_widths[ck])
    if truncated:
        sep_parts.append("---")
    sep_parts.append("-" * total_width)
    lines.append("  ".join(sep_parts))

    # Data rows
    for rk in pivot.row_keys:
        parts = [rk.ljust(row_header_width)]
        for ck in display_cols:
            parts.append(_fmt_val(pivot.cells[(rk, ck)].value).rjust(col_widths[ck]))
        if truncated:
            parts.append("...")
        parts.append(_fmt_val(pivot.row_totals[rk]).rjust(total_width))
        lines.append("  ".join(parts))

    # Totals row
    sep_parts2 = ["-" * row_header_width]
    for ck in display_cols:
        sep_parts2.append("-" * col_widths[ck])
    if truncated:
        sep_parts2.append("---")
    sep_parts2.append("-" * total_width)
    lines.append("  ".join(sep_parts2))

    total_parts = [total_label.ljust(row_header_width)]
    for ck in display_cols:
        total_parts.append(_fmt_val(pivot.col_totals[ck]).rjust(col_widths[ck]))
    if truncated:
        total_parts.append("...")
    total_parts.append(_fmt_val(pivot.grand_total).rjust(total_width))
    lines.append("  ".join(total_parts))

    return "\n".join(lines)


def format_pivot_csv(pivot: PivotTable) -> str:
    """Format pivot table as CSV string."""
    lines: list[str] = []

    def _csv_escape(s: str) -> str:
        if "," in s or '"' in s or "\n" in s:
            return '"' + s.replace('"', '""') + '"'
        return s

    def _fmt_val(v: float) -> str:
        if v == int(v) and abs(v) < 1e12:
            return str(int(v))
        return f"{v:.2f}"

    # Header
    header = [_csv_escape(pivot.row_field)]
    for ck in pivot.col_keys:
        header.append(_csv_escape(ck))
    header.append("Total")
    lines.append(",".join(header))

    # Data rows
    for rk in pivot.row_keys:
        parts = [_csv_escape(rk)]
        for ck in pivot.col_keys:
            parts.append(_fmt_val(pivot.cells[(rk, ck)].value))

        parts.append(_fmt_val(pivot.row_totals[rk]))
        lines.append(",".join(parts))

    # Totals row
    total_parts = ["Total"]
    for ck in pivot.col_keys:
        total_parts.append(_fmt_val(pivot.col_totals[ck]))
    total_parts.append(_fmt_val(pivot.grand_total))
    lines.append(",".join(total_parts))

    return "\n".join(lines)


def compute_pivot_percentages(
    pivot: PivotTable, mode: str = "row"
) -> dict[tuple[str, str], float]:
    """Compute percentage breakdown.

    mode="row": each cell as % of row total
    mode="col": each cell as % of column total
    mode="total": each cell as % of grand total

    Returns dict mapping (row_key, col_key) -> percentage (0-100 scale).
    """
    result: dict[tuple[str, str], float] = {}
    for rk in pivot.row_keys:
        for ck in pivot.col_keys:
            cell_val = pivot.cells[(rk, ck)].value
            if mode == "row":
                denom = pivot.row_totals[rk]
            elif mode == "col":
                denom = pivot.col_totals[ck]
            elif mode == "total":
                denom = pivot.grand_total
            else:
                denom = pivot.grand_total

            if denom == 0:
                result[(rk, ck)] = 0.0
            else:
                result[(rk, ck)] = (cell_val / denom) * 100.0
    return result


def find_pivot_outliers(
    pivot: PivotTable, threshold: float = 2.0
) -> list[dict]:
    """Find cells that are unusually high or low compared to row/col average.

    Uses z-score within each row. A cell is flagged if its absolute z-score
    exceeds *threshold*.

    Returns list of dicts with keys: row, col, value, expected, deviation.
    """
    outliers: list[dict] = []

    for rk in pivot.row_keys:
        values = [pivot.cells[(rk, ck)].value for ck in pivot.col_keys]
        n = len(values)
        if n < 2:
            continue

        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        std = math.sqrt(variance) if variance > 0 else 0.0

        if std == 0.0:
            continue

        for i, ck in enumerate(pivot.col_keys):
            z = (values[i] - mean) / std
            if abs(z) > threshold:
                outliers.append(
                    {
                        "row": rk,
                        "col": ck,
                        "value": values[i],
                        "expected": round(mean, 4),
                        "deviation": round(z, 4),
                    }
                )

    return outliers


def multi_pivot(
    rows: list[dict],
    row_field: str,
    col_field: str,
    value_fields: list[str],
    agg_func: str = "sum",
) -> list[PivotTable]:
    """Create multiple pivot tables, one per value_field.

    Returns a list of PivotTable objects (skips value fields that produce None).
    """
    results: list[PivotTable] = []
    for vf in value_fields:
        pt = create_pivot(rows, row_field, col_field, vf, agg_func)
        if pt is not None:
            results.append(pt)
    return results
