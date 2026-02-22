"""Auto-generate a data dictionary from table data.

Pure-function module with no external dependencies.  Given a list of row
dicts it infers column types, computes statistics, detects semantic tags,
and produces human-readable descriptions -- all without reaching out to a
database or importing any third-party library.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ColumnEntry:
    """Complete metadata entry for a single column."""

    name: str
    inferred_type: str  # "integer", "float", "text", "date", "boolean", "categorical", "identifier"
    sample_values: list[str]  # up to 5 sample values
    null_count: int
    null_pct: float
    unique_count: int
    unique_pct: float
    min_value: str | None
    max_value: str | None
    mean_value: float | None  # only for numeric
    description: str  # auto-generated description
    tags: list[str] = field(default_factory=list)


@dataclass
class DataDictionary:
    """Full data dictionary for a table."""

    table_name: str
    row_count: int
    column_count: int
    columns: list[ColumnEntry]
    relationships_hint: list[str]
    summary: str
    generated_at: str


# ---------------------------------------------------------------------------
# Date-detection helpers
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d-%m-%Y",
    "%Y/%m/%d",
    "%b %d, %Y",
    "%B %d, %Y",
]


def _is_date(value: str) -> bool:
    """Return True if *value* parses as a date in any common format."""
    if not isinstance(value, str) or not value.strip():
        return False
    val = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            datetime.strptime(val, fmt)
            return True
        except ValueError:
            continue
    return False


# ---------------------------------------------------------------------------
# Boolean detection
# ---------------------------------------------------------------------------

_BOOLEAN_VALUES = {
    "true", "false", "yes", "no", "y", "n", "0", "1",
    "True", "False", "Yes", "No", "Y", "N",
}


def _is_boolean(value) -> bool:
    """Return True if *value* looks boolean."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and value in (0, 1, 0.0, 1.0):
        return True
    if isinstance(value, str) and value.strip() in _BOOLEAN_VALUES:
        return True
    return False


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def _try_numeric(value) -> float | None:
    """Try to convert *value* to a float.  Return None on failure."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        # strip leading currency symbols
        cleaned = re.sub(r"^[$€£¥₹]", "", cleaned)
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    return None


def _is_integer_value(value) -> bool:
    """Return True if *value* is representable as an integer."""
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value == int(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "")
        try:
            f = float(cleaned)
            return f == int(f)
        except (ValueError, TypeError):
            return False
    return False


# ---------------------------------------------------------------------------
# Phone / email heuristics
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^[\d\s\-\(\)\+\.]{7,}$")


# ---------------------------------------------------------------------------
# Public API -- type inference
# ---------------------------------------------------------------------------

def infer_column_type(values: list) -> str:
    """Infer the most likely data type for a column based on its values.

    Checks in order: boolean, integer, float, date, categorical, text.
    Categorical = fewer unique values than 50 % of total **and** < 20 unique values.
    """
    non_null = [v for v in values if v is not None and v != "" and str(v).strip() != ""]
    if not non_null:
        return "text"

    # --- boolean ---
    if all(_is_boolean(v) for v in non_null):
        return "boolean"

    # --- integer ---
    if all(_is_integer_value(v) for v in non_null):
        # If every value is a small set of int values, still mark integer
        return "integer"

    # --- float ---
    if all(_try_numeric(v) is not None for v in non_null):
        return "float"

    # --- date ---
    str_vals = [str(v) for v in non_null]
    if all(_is_date(s) for s in str_vals):
        return "date"

    # --- categorical ---
    unique = set(str(v) for v in non_null)
    total = len(non_null)
    if len(unique) < 20 and len(unique) < total * 0.5:
        return "categorical"

    # --- identifier heuristic (very high uniqueness for strings) ---
    # We don't infer "identifier" purely from values; that's done via tags.

    return "text"


# ---------------------------------------------------------------------------
# Public API -- auto-describe
# ---------------------------------------------------------------------------

def auto_describe_column(name: str, col_type: str, stats: dict) -> str:
    """Generate a human-readable description based on column name and type.

    ``stats`` may carry ``unique_pct``, ``null_pct``, ``min_value``,
    ``max_value``, ``mean_value`` -- they are used to enrich the text
    when available.

    IMPORTANT: Order of pattern checks matters!  More specific patterns must
    come before generic ones.  In business/manufacturing contexts:
      - "rate" = price per unit (NOT percentage/ratio)
      - "yield" = production yield % (NOT quantity)
      - "quantity" = weight/count (NOT yield)
    """
    lower = name.lower()

    # --- identifier / key patterns ---
    if lower.endswith("_id") or lower == "id":
        prefix = lower.replace("_id", "").replace("id", "record").strip() or "record"
        prefix = prefix.replace("_", " ")
        return f"Identifier/key field for {prefix}."

    # --- date / timestamp patterns ---
    if lower.endswith("_date") or lower.endswith("_at") or lower.endswith("_time") or lower.endswith("_ts"):
        event = lower.replace("_date", "").replace("_at", "").replace("_time", "").replace("_ts", "").replace("_", " ").strip()
        event = event or "the event"
        return f"Timestamp for when {event} occurred."

    # --- yield / production yield (BEFORE monetary/count to avoid misclassification) ---
    if lower in ("yield", "yield_pct", "yield_percent", "yield_percentage"):
        return "Production yield percentage (output/input ratio). NOT a quantity or count — this is a ratio/percentage value."

    # --- rate disambiguation (price-per-unit vs percentage/ratio) ---
    # Monetary rate: "rate" alone, or preceded by business/pricing qualifier
    _monetary_rate_prefixes = {"basic", "unit", "purchase", "selling", "billing",
                               "market", "contract", "avg", "net", "landed", "effective"}
    # Ratio rate: preceded by quality/performance qualifier → treat as percentage
    _ratio_rate_prefixes = {"success", "failure", "rejection", "defect", "interest",
                            "conversion", "growth", "attrition", "error", "churn",
                            "click", "bounce", "open", "close", "win", "hit",
                            "utilization", "occupancy", "fill", "pass", "fail"}
    if lower.endswith("_rate") or lower == "rate":
        prefix = lower.rsplit("_rate", 1)[0] if lower != "rate" else ""
        if prefix in _ratio_rate_prefixes:
            label = lower.replace("_", " ")
            return f"Percentage or ratio value for {label}."
        elif prefix in _monetary_rate_prefixes or lower == "rate" or prefix == "":
            label = lower.replace("_", " ")
            return f"Price per unit / rate value for {label}. This is a monetary rate (₹/unit), NOT a percentage."
        else:
            # Unknown prefix: default to price in manufacturing context
            label = lower.replace("_", " ")
            return f"Rate value for {label} (likely price per unit in manufacturing context)."

    # --- monetary patterns ---
    money_keywords = {"amount", "cost", "price", "revenue", "salary", "fee", "total",
                      "balance", "payment", "invoice", "charge", "freight", "discount",
                      "debit", "credit", "expense", "income", "margin", "profit", "loss"}
    if any(kw in lower for kw in money_keywords):
        label = lower.replace("_", " ")
        return f"Monetary value representing {label}."

    # --- count / quantity / weight (physical amounts) ---
    count_keywords = {"count", "qty", "quantity", "num", "number_of", "pieces", "bags",
                      "bundles", "units", "tons", "tonnes", "mt"}
    if any(kw in lower for kw in count_keywords):
        label = lower.replace("_", " ")
        return f"Count/quantity of {label} (physical amount or weight)."

    # --- percentage / ratio (explicit patterns only — "rate" is excluded) ---
    if "pct" in lower or "percent" in lower or "ratio" in lower:
        label = lower.replace("_", " ")
        return f"Percentage or ratio value for {label}."

    # --- energy / power metrics (manufacturing domain) ---
    _energy_patterns = {"kwh", "kva", "kvar", "sec", "power", "energy", "consumption",
                        "pf", "power_factor", "mwh", "unit_consumed"}
    if any(kw == lower or kw in lower.split("_") for kw in _energy_patterns):
        label = lower.replace("_", " ")
        return f"Energy/power metric: {label}."

    # --- production / manufacturing metrics ---
    _prod_patterns = {"heat_no", "heat", "tap_to_tap", "cycle_time", "furnace",
                      "melt", "melting", "tapping", "ladle", "ingot", "billet",
                      "sponge", "scrap", "alloy", "fesi", "femn", "rejection",
                      "slag", "refractory", "lining"}
    if any(kw == lower or kw in lower.split("_") for kw in _prod_patterns):
        label = lower.replace("_", " ")
        return f"Manufacturing/production field: {label}."

    # --- name fields ---
    if "name" in lower:
        label = lower.replace("_", " ")
        return f"Name field: {label}."

    # --- email ---
    if "email" in lower or "e_mail" in lower:
        return "Email address field."

    # --- phone ---
    if "phone" in lower or "tel" in lower or "mobile" in lower:
        return "Phone/telephone number."

    # --- address ---
    if "address" in lower or "street" in lower or "city" in lower or "zip" in lower or "postal" in lower:
        label = lower.replace("_", " ")
        return f"Address component: {label}."

    # --- status / type / category ---
    if "status" in lower or "state" in lower:
        return f"Status or state indicator for {lower.replace('_', ' ')}."

    if "type" in lower or "category" in lower or "group" in lower or "grade" in lower:
        return f"Categorical grouping field: {lower.replace('_', ' ')}."

    # --- party / vendor / supplier / customer names ---
    party_keywords = {"party", "vendor", "supplier", "customer", "buyer", "seller", "client"}
    if any(kw in lower for kw in party_keywords):
        label = lower.replace("_", " ")
        return f"Business entity name: {label} (buyer, seller, vendor, or customer)."

    # --- weight / volume / measure ---
    measure_keywords = {"weight", "volume", "length", "width", "height", "size", "area",
                        "diameter", "thickness", "depth"}
    if any(kw in lower for kw in measure_keywords):
        label = lower.replace("_", " ")
        return f"Physical measurement value: {label}."

    # --- description / notes / comment ---
    if "desc" in lower or "note" in lower or "comment" in lower or "remark" in lower:
        return "Free-text description or notes field."

    # --- boolean-ish names ---
    if lower.startswith("is_") or lower.startswith("has_") or lower.startswith("can_"):
        return f"Boolean flag indicating whether {lower.replace('_', ' ')}."

    # --- generic fallback by type ---
    type_labels = {
        "integer": "Integer",
        "float": "Numeric (decimal)",
        "text": "Text",
        "date": "Date/time",
        "boolean": "Boolean",
        "categorical": "Categorical",
        "identifier": "Identifier",
    }
    type_label = type_labels.get(col_type, "Data")
    return f"{type_label} column: {lower.replace('_', ' ')}."


# ---------------------------------------------------------------------------
# Public API -- tag detection
# ---------------------------------------------------------------------------

def detect_column_tags(
    name: str,
    col_type: str,
    values: list,
    unique_pct: float,
) -> list[str]:
    """Auto-detect semantic tags for a column.

    Possible tags: ``primary_key``, ``foreign_key``, ``currency``,
    ``percentage``, ``email``, ``phone``, ``required``, ``optional``.
    """
    tags: list[str] = []
    lower = name.lower()

    non_null = [v for v in values if v is not None and v != ""]
    null_count = len(values) - len(non_null)
    null_pct = (null_count / len(values) * 100) if values else 0.0

    # --- primary_key ---
    if (lower.endswith("_id") or lower == "id") and unique_pct >= 95.0:
        tags.append("primary_key")
    # --- foreign_key ---
    elif lower.endswith("_id") and unique_pct < 95.0:
        tags.append("foreign_key")

    # --- currency ---
    currency_words = {"cost", "price", "amount", "revenue", "salary", "fee", "total", "balance", "payment"}
    if any(cw in lower for cw in currency_words):
        tags.append("currency")

    # --- percentage ---
    pct_name = "pct" in lower or "percent" in lower or "%" in lower or "ratio" in lower or "rate" in lower
    if pct_name:
        tags.append("percentage")
    else:
        # Check values: all numeric and in 0-100 range
        numerics = [_try_numeric(v) for v in non_null]
        valid_nums = [n for n in numerics if n is not None]
        if valid_nums and all(0.0 <= n <= 100.0 for n in valid_nums):
            # Only tag percentage if name also hints at it (avoid false positives)
            pass  # already handled above

    # --- email ---
    str_non_null = [str(v) for v in non_null]
    if str_non_null and sum(1 for s in str_non_null if _EMAIL_RE.match(s)) > len(str_non_null) * 0.5:
        tags.append("email")

    # --- phone ---
    if str_non_null and sum(1 for s in str_non_null if _PHONE_RE.match(s)) > len(str_non_null) * 0.5:
        if "phone" in lower or "tel" in lower or "mobile" in lower or "fax" in lower:
            tags.append("phone")

    # --- required / optional ---
    if null_pct == 0.0:
        tags.append("required")
    elif null_pct > 10.0:
        tags.append("optional")

    return tags


# ---------------------------------------------------------------------------
# Internal statistics helpers
# ---------------------------------------------------------------------------

def _column_values(rows: list[dict], col: str) -> list:
    """Extract column values from rows, preserving order."""
    return [row.get(col) for row in rows]


def _non_null(values: list) -> list:
    """Filter out None and empty-string values."""
    return [v for v in values if v is not None and v != ""]


def _sample_values(values: list, n: int = 5) -> list[str]:
    """Pick up to *n* non-null unique sample values, preserving first-seen order."""
    seen: set[str] = set()
    samples: list[str] = []
    for v in values:
        if v is None or v == "":
            continue
        sv = str(v)
        if sv not in seen:
            seen.add(sv)
            samples.append(sv)
            if len(samples) >= n:
                break
    return samples


def _compute_stats(values: list, col_type: str) -> dict:
    """Compute min, max, mean for a column.  *mean* only for numeric types."""
    non_null = _non_null(values)
    total = len(values)

    null_count = total - len(non_null)
    null_pct = (null_count / total * 100) if total else 0.0

    unique = set(str(v) for v in non_null)
    unique_count = len(unique)
    unique_pct = (unique_count / len(non_null) * 100) if non_null else 0.0

    min_value: str | None = None
    max_value: str | None = None
    mean_value: float | None = None

    if col_type in ("integer", "float"):
        nums = [_try_numeric(v) for v in non_null]
        valid = [n for n in nums if n is not None]
        if valid:
            min_value = str(min(valid))
            max_value = str(max(valid))
            mean_value = round(statistics.mean(valid), 4)
    elif col_type == "date":
        # sort string representations; best-effort
        sorted_dates = sorted(str(v) for v in non_null)
        if sorted_dates:
            min_value = sorted_dates[0]
            max_value = sorted_dates[-1]
    elif non_null:
        sorted_str = sorted(str(v) for v in non_null)
        min_value = sorted_str[0]
        max_value = sorted_str[-1]

    return {
        "null_count": null_count,
        "null_pct": round(null_pct, 2),
        "unique_count": unique_count,
        "unique_pct": round(unique_pct, 2),
        "min_value": min_value,
        "max_value": max_value,
        "mean_value": mean_value,
    }


# ---------------------------------------------------------------------------
# Public API -- relationship hints
# ---------------------------------------------------------------------------

def _relationship_hints(columns: list[ColumnEntry]) -> list[str]:
    """Produce relationship hints based on column names."""
    hints: list[str] = []
    for col in columns:
        lower = col.name.lower()
        if lower.endswith("_id") and lower != "id":
            ref_table = lower[: -len("_id")]
            # pluralise naively
            if not ref_table.endswith("s"):
                ref_table_plural = ref_table + "s"
            else:
                ref_table_plural = ref_table
            hints.append(f"{col.name} likely references {ref_table_plural} table")
    return hints


# ---------------------------------------------------------------------------
# Public API -- main entry point
# ---------------------------------------------------------------------------

def generate_data_dictionary(
    rows: list[dict],
    table_name: str = "table",
) -> DataDictionary | None:
    """Auto-generate a data dictionary from sample data.

    Returns ``None`` if *rows* is empty.
    """
    if not rows:
        return None

    # Discover all column names (preserving first-seen order)
    col_names: list[str] = []
    seen_cols: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen_cols:
                seen_cols.add(key)
                col_names.append(key)

    columns: list[ColumnEntry] = []
    for col in col_names:
        values = _column_values(rows, col)
        col_type = infer_column_type(values)
        stats = _compute_stats(values, col_type)
        desc = auto_describe_column(col, col_type, stats)
        tags = detect_column_tags(col, col_type, values, stats["unique_pct"])
        samples = _sample_values(values)

        entry = ColumnEntry(
            name=col,
            inferred_type=col_type,
            sample_values=samples,
            null_count=stats["null_count"],
            null_pct=stats["null_pct"],
            unique_count=stats["unique_count"],
            unique_pct=stats["unique_pct"],
            min_value=stats["min_value"],
            max_value=stats["max_value"],
            mean_value=stats["mean_value"],
            description=desc,
            tags=tags,
        )
        columns.append(entry)

    hints = _relationship_hints(columns)

    summary_parts = [
        f"Table '{table_name}' has {len(rows)} rows and {len(col_names)} columns.",
    ]
    type_counts: dict[str, int] = {}
    for c in columns:
        type_counts[c.inferred_type] = type_counts.get(c.inferred_type, 0) + 1
    type_desc = ", ".join(f"{cnt} {tp}" for tp, cnt in sorted(type_counts.items()))
    summary_parts.append(f"Column types: {type_desc}.")

    nullable_cols = [c.name for c in columns if c.null_pct > 0]
    if nullable_cols:
        summary_parts.append(f"Columns with nulls: {', '.join(nullable_cols)}.")
    else:
        summary_parts.append("No null values detected.")

    if hints:
        summary_parts.append(f"Detected {len(hints)} potential relationship(s).")

    summary = " ".join(summary_parts)

    return DataDictionary(
        table_name=table_name,
        row_count=len(rows),
        column_count=len(col_names),
        columns=columns,
        relationships_hint=hints,
        summary=summary,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Public API -- formatting
# ---------------------------------------------------------------------------

def format_dictionary_markdown(dd: DataDictionary) -> str:
    """Format data dictionary as a Markdown document with tables and descriptions."""
    lines: list[str] = []
    lines.append(f"# Data Dictionary: {dd.table_name}")
    lines.append("")
    lines.append(f"**Rows:** {dd.row_count}  ")
    lines.append(f"**Columns:** {dd.column_count}  ")
    lines.append(f"**Generated at:** {dd.generated_at}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(dd.summary)
    lines.append("")

    # Column table
    lines.append("## Columns")
    lines.append("")
    lines.append("| Column | Type | Nulls (%) | Unique (%) | Min | Max | Mean | Tags |")
    lines.append("|--------|------|-----------|------------|-----|-----|------|------|")

    for col in dd.columns:
        tags_str = ", ".join(col.tags) if col.tags else "-"
        mean_str = str(col.mean_value) if col.mean_value is not None else "-"
        min_str = col.min_value if col.min_value is not None else "-"
        max_str = col.max_value if col.max_value is not None else "-"
        lines.append(
            f"| {col.name} | {col.inferred_type} | {col.null_pct}% | {col.unique_pct}% "
            f"| {min_str} | {max_str} | {mean_str} | {tags_str} |"
        )

    lines.append("")

    # Detailed descriptions
    lines.append("## Column Details")
    lines.append("")
    for col in dd.columns:
        lines.append(f"### {col.name}")
        lines.append("")
        lines.append(f"- **Type:** {col.inferred_type}")
        lines.append(f"- **Description:** {col.description}")
        lines.append(f"- **Sample values:** {', '.join(col.sample_values) if col.sample_values else 'N/A'}")
        lines.append(f"- **Nulls:** {col.null_count} ({col.null_pct}%)")
        lines.append(f"- **Unique values:** {col.unique_count} ({col.unique_pct}%)")
        if col.tags:
            lines.append(f"- **Tags:** {', '.join(col.tags)}")
        lines.append("")

    # Relationships
    if dd.relationships_hint:
        lines.append("## Relationship Hints")
        lines.append("")
        for hint in dd.relationships_hint:
            lines.append(f"- {hint}")
        lines.append("")

    return "\n".join(lines)


def format_dictionary_text(dd: DataDictionary) -> str:
    """Format data dictionary as plain text."""
    lines: list[str] = []
    lines.append(f"DATA DICTIONARY: {dd.table_name}")
    lines.append("=" * 60)
    lines.append(f"Rows: {dd.row_count}  |  Columns: {dd.column_count}")
    lines.append(f"Generated: {dd.generated_at}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append(dd.summary)
    lines.append("")
    lines.append("COLUMNS")
    lines.append("-" * 60)

    for col in dd.columns:
        lines.append(f"  {col.name} ({col.inferred_type})")
        lines.append(f"    Description : {col.description}")
        lines.append(f"    Nulls       : {col.null_count} ({col.null_pct}%)")
        lines.append(f"    Unique      : {col.unique_count} ({col.unique_pct}%)")
        if col.min_value is not None:
            lines.append(f"    Min         : {col.min_value}")
        if col.max_value is not None:
            lines.append(f"    Max         : {col.max_value}")
        if col.mean_value is not None:
            lines.append(f"    Mean        : {col.mean_value}")
        if col.sample_values:
            lines.append(f"    Samples     : {', '.join(col.sample_values)}")
        if col.tags:
            lines.append(f"    Tags        : {', '.join(col.tags)}")
        lines.append("")

    if dd.relationships_hint:
        lines.append("RELATIONSHIPS")
        lines.append("-" * 60)
        for hint in dd.relationships_hint:
            lines.append(f"  - {hint}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API -- compare two dictionaries
# ---------------------------------------------------------------------------

def compare_dictionaries(dd_a: DataDictionary, dd_b: DataDictionary) -> dict:
    """Compare two data dictionaries and report schema changes.

    Returns a dict with keys:
      - ``added_columns``   -- column names present in *dd_b* but not *dd_a*
      - ``removed_columns`` -- column names present in *dd_a* but not *dd_b*
      - ``type_changes``    -- list of ``{column, old_type, new_type}``
      - ``stat_changes``    -- list of ``{column, field, old_value, new_value}``
    """
    cols_a = {c.name: c for c in dd_a.columns}
    cols_b = {c.name: c for c in dd_b.columns}

    names_a = set(cols_a.keys())
    names_b = set(cols_b.keys())

    added = sorted(names_b - names_a)
    removed = sorted(names_a - names_b)

    type_changes: list[dict] = []
    stat_changes: list[dict] = []

    common = names_a & names_b
    for name in sorted(common):
        a = cols_a[name]
        b = cols_b[name]

        if a.inferred_type != b.inferred_type:
            type_changes.append({
                "column": name,
                "old_type": a.inferred_type,
                "new_type": b.inferred_type,
            })

        # Compare select statistics
        stat_fields = [
            ("null_pct", a.null_pct, b.null_pct),
            ("unique_pct", a.unique_pct, b.unique_pct),
            ("min_value", a.min_value, b.min_value),
            ("max_value", a.max_value, b.max_value),
            ("mean_value", a.mean_value, b.mean_value),
        ]
        for field_name, old_val, new_val in stat_fields:
            if old_val != new_val:
                stat_changes.append({
                    "column": name,
                    "field": field_name,
                    "old_value": old_val,
                    "new_value": new_val,
                })

    return {
        "added_columns": added,
        "removed_columns": removed,
        "type_changes": type_changes,
        "stat_changes": stat_changes,
    }
