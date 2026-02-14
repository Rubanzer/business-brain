"""Column semantic classifier — detects column types and business domain.

Pure Python, no LLM calls.  Runs BEFORE analysis to drive what
statistics to compute, what charts to suggest, and how to format numbers.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

# ---------------------------------------------------------------------------
# Column-name regex patterns (case-insensitive)
# ---------------------------------------------------------------------------

_ID_PATTERNS = re.compile(
    r"(^id$|_id$|^key$|_key$|^code$|_code$|_no$|_number$|^number$"
    r"|^invoice|^order_id|^employee_id|^customer_id|_row_id$"
    r"|^heat_no|^machine_id|^batch_id|^lot_no|^wo_id|^asset_code"
    r"|^challan|^vehicle_no|^truck_no|^serial_no|^equipment_id)",
    re.IGNORECASE,
)

_TEMPORAL_PATTERNS = re.compile(
    r"(date|_at$|_on$|timestamp|^time$|^month$|^year$|^quarter$|^week$"
    r"|^period$|^day$|^created|^updated|^modified|^hire_date|^order_date)",
    re.IGNORECASE,
)

_CURRENCY_PATTERNS = re.compile(
    r"(price|cost|amount|revenue|salary|budget|expense|payment|invoice"
    r"|fee|wage|income|profit|margin|billing|total_amount|unit_price"
    r"|^rate$|_rate$)",
    re.IGNORECASE,
)

_PERCENTAGE_PATTERNS = re.compile(
    r"(%|pct|percent|ratio|yield|completion|efficiency|utilization"
    r"|accuracy|coverage|conversion|bounce|churn|retention)",
    re.IGNORECASE,
)

_BOOLEAN_VALUES = frozenset({
    "true", "false", "yes", "no", "0", "1", "y", "n", "t", "f",
    "active", "inactive", "enabled", "disabled",
})

# ---------------------------------------------------------------------------
# Domain keyword maps
# ---------------------------------------------------------------------------

_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "manufacturing": [
        "furnace", "heat", "billet", "rolling", "tmt", "scada",
        "kva", "power_factor", "temperature", "pressure", "rpm",
        "machine", "breakdown", "downtime", "shift", "production",
        "tonnage", "output", "casting", "melt", "scrap",
    ],
    "quality": [
        "rejection", "defect", "inspection", "grade", "yield",
        "rework", "scrap_rate", "specification", "tolerance",
        "sample", "lab", "fe", "carbon", "test_result",
    ],
    "sales": [
        "customer", "order", "revenue", "sales", "product", "region",
        "discount", "quantity", "channel", "deal",
    ],
    "finance": [
        "expense", "budget", "account", "cost", "invoice", "payment",
        "ledger", "debit", "credit", "tax", "fiscal",
    ],
    "hr": [
        "employee", "salary", "department", "hire_date", "performance",
        "designation", "grade", "attendance", "leave", "payroll",
    ],
    "procurement": [
        "supplier", "material", "rate", "quality", "yield", "grade",
        "vendor", "purchase", "tender", "party", "fe",
    ],
    "logistics": [
        "truck", "gate", "vehicle", "dispatch", "delivery", "freight",
        "challan", "weighbridge", "transporter", "loading",
    ],
    "energy": [
        "power", "kva", "kwh", "voltage", "current", "consumption",
        "unit_consumption", "electricity", "energy", "meter",
    ],
    "marketing": [
        "campaign", "channel", "impression", "click", "conversion",
        "spend", "ctr", "cpc", "roi", "engagement",
    ],
    "inventory": [
        "product", "quantity", "warehouse", "stock", "location",
        "sku", "inventory", "reorder", "batch",
    ],
}

# ---------------------------------------------------------------------------
# SQL type helpers
# ---------------------------------------------------------------------------

_SQL_NUMERIC = frozenset({
    "bigint", "integer", "int", "smallint", "real", "float",
    "double precision", "numeric", "decimal", "money",
})

_SQL_TEMPORAL = frozenset({
    "timestamp", "timestamptz", "date", "time", "timetz",
    "timestamp without time zone", "timestamp with time zone",
})

_SQL_TEXT = frozenset({
    "text", "varchar", "character varying", "char", "character", "citext",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_columns(
    columns: list[str],
    sample_rows: list[dict[str, Any]],
    col_types: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Classify each column by semantic type and detect business domain.

    Parameters
    ----------
    columns : list[str]
        Ordered column names exactly as they appear in the data.
    sample_rows : list[dict]
        A representative sample of rows (typically first 50-100).
    col_types : dict[str, str] | None
        Optional mapping of column name -> SQL type string
        (e.g. ``{"RATE": "NUMERIC", "PARTY": "TEXT"}``).

    Returns
    -------
    dict with keys:
        columns : dict[str, dict]  — per-column classification
        domain_hint : str          — best-guess business domain
        analysis_plan : list[str]  — ordered list of analyses to run
        chart_plan : list[dict]    — suggested chart configs
    """
    col_types = col_types or {}
    total_rows = len(sample_rows)

    classified: dict[str, dict[str, Any]] = {}
    for col in columns:
        classified[col] = _classify_one(col, sample_rows, total_rows, col_types.get(col))

    # Derive domain hint
    domain_hint = _detect_domain(columns)

    # Build analysis + chart plans
    analysis_plan = _build_analysis_plan(classified)
    chart_plan = _build_chart_plan(classified, columns)

    return {
        "columns": classified,
        "domain_hint": domain_hint,
        "analysis_plan": analysis_plan,
        "chart_plan": chart_plan,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_one(
    col: str,
    sample_rows: list[dict],
    total_rows: int,
    sql_type: str | None,
) -> dict[str, Any]:
    """Classify a single column."""
    raw_values = [row.get(col) for row in sample_rows]
    non_null = [v for v in raw_values if v is not None]
    unique = set(str(v) for v in non_null)
    cardinality = len(unique)
    cardinality_ratio = cardinality / total_rows if total_rows else 0

    result: dict[str, Any] = {
        "cardinality": cardinality,
        "null_count": len(raw_values) - len(non_null),
        "sample_values": [str(v) for v in non_null[:5]],
    }

    sql_type_lower = (sql_type or "").lower().strip()

    # 1. SQL type check — temporal
    if sql_type_lower and any(t in sql_type_lower for t in _SQL_TEMPORAL):
        result["semantic_type"] = "temporal"
        return result

    # 2. SQL type check — boolean
    if sql_type_lower == "boolean":
        result["semantic_type"] = "boolean"
        return result

    # 3. Temporal — name patterns or date-like values
    if _TEMPORAL_PATTERNS.search(col):
        result["semantic_type"] = "temporal"
        return result

    # 4. Boolean — only 2 unique values
    if cardinality <= 2 and non_null:
        lower_vals = {str(v).lower().strip() for v in non_null}
        if lower_vals.issubset(_BOOLEAN_VALUES) or cardinality <= 2 and sql_type_lower == "boolean":
            result["semantic_type"] = "boolean"
            return result

    # 5. Numeric detection (BEFORE identifier — numeric IDs should still be classified as numeric)
    is_numeric = _is_numeric_column(non_null, sql_type_lower)
    if is_numeric:
        # Check if this is an identifier first (numeric ID columns)
        if _ID_PATTERNS.search(col) and cardinality_ratio > 0.8:
            result["semantic_type"] = "identifier"
            return result

        nums = _parse_numerics(non_null)
        if nums:
            result["stats"] = _basic_stats(nums)

        # Sub-classify numeric
        if _PERCENTAGE_PATTERNS.search(col) and nums:
            mn, mx = min(nums), max(nums)
            if mn >= 0 and mx <= 100:
                result["semantic_type"] = "numeric_percentage"
                return result

        if _CURRENCY_PATTERNS.search(col):
            result["semantic_type"] = "numeric_currency"
            return result

        result["semantic_type"] = "numeric_metric"
        return result

    # 6. Non-numeric identifier — high cardinality string + name match
    if _ID_PATTERNS.search(col) and cardinality_ratio > 0.8:
        result["semantic_type"] = "identifier"
        return result

    # 7. Text vs categorical
    if sql_type_lower and any(t in sql_type_lower for t in _SQL_TEXT):
        is_string = True
    else:
        is_string = all(isinstance(v, str) for v in non_null) if non_null else True

    if is_string:
        avg_len = (
            sum(len(str(v)) for v in non_null) / len(non_null)
            if non_null else 0
        )
        if cardinality_ratio > 0.9 and avg_len > 50:
            result["semantic_type"] = "text"
        elif cardinality <= 50:
            result["semantic_type"] = "categorical"
        elif cardinality_ratio > 0.9:
            result["semantic_type"] = "identifier"
        else:
            result["semantic_type"] = "categorical"
        return result

    # Fallback
    result["semantic_type"] = "text"
    return result


def columns_first(sample_rows: list[dict]) -> str | None:
    """Return the first column name from sample rows."""
    if sample_rows and sample_rows[0]:
        return next(iter(sample_rows[0]))
    return None


def _is_numeric_column(values: list, sql_type_lower: str) -> bool:
    """Check if column values are numeric."""
    if sql_type_lower and any(t in sql_type_lower for t in _SQL_NUMERIC):
        return True
    if not values:
        return False
    numeric_count = 0
    for v in values:
        if isinstance(v, (int, float)):
            numeric_count += 1
        elif isinstance(v, str):
            try:
                float(v.replace(",", ""))
                numeric_count += 1
            except (ValueError, AttributeError):
                pass
    return numeric_count / len(values) > 0.8


def _parse_numerics(values: list) -> list[float]:
    """Parse numeric values, skipping non-parseable ones."""
    nums = []
    for v in values:
        if isinstance(v, (int, float)):
            nums.append(float(v))
        elif isinstance(v, str):
            try:
                nums.append(float(v.replace(",", "")))
            except (ValueError, AttributeError):
                pass
    return nums


def _basic_stats(nums: list[float]) -> dict[str, float]:
    """Compute basic descriptive stats."""
    if not nums:
        return {}
    result = {
        "mean": round(statistics.mean(nums), 2),
        "min": round(min(nums), 2),
        "max": round(max(nums), 2),
    }
    if len(nums) >= 2:
        result["median"] = round(statistics.median(nums), 2)
        result["stdev"] = round(statistics.stdev(nums), 2)
    return result


def _detect_domain(columns: list[str]) -> str:
    """Score each domain by keyword overlap with column names."""
    col_lower = [c.lower() for c in columns]
    scores: dict[str, int] = {}
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            for c in col_lower:
                if kw in c:
                    score += 1
                    break
        scores[domain] = score

    if not scores:
        return "general"
    best = max(scores, key=scores.get)  # type: ignore[arg-type]
    return best if scores[best] >= 2 else "general"


def _build_analysis_plan(classified: dict[str, dict]) -> list[str]:
    """Determine which analyses to run based on column types present."""
    types_present = {info["semantic_type"] for info in classified.values()}
    plan = []

    numerics = {
        "numeric_metric", "numeric_currency", "numeric_percentage",
    }
    has_numeric = bool(types_present & numerics)
    has_categorical = "categorical" in types_present
    has_temporal = "temporal" in types_present

    if has_numeric:
        plan.append("descriptive_statistics")

    if has_categorical and has_numeric:
        plan.append("group_by_aggregation")

    if has_temporal and has_numeric:
        plan.append("time_trend")

    num_cols = [
        c for c, info in classified.items()
        if info["semantic_type"] in numerics
    ]
    if len(num_cols) >= 2:
        plan.append("correlation")

    if has_categorical and not has_numeric:
        plan.append("frequency_distribution")

    if has_numeric:
        plan.append("outlier_detection")

    if "numeric_currency" in types_present:
        plan.append("cost_analysis")

    if "numeric_percentage" in types_present:
        plan.append("percentage_distribution")

    return plan


def _build_chart_plan(
    classified: dict[str, dict],
    columns: list[str],
) -> list[dict[str, Any]]:
    """Suggest charts based on column type combinations."""
    charts: list[dict[str, Any]] = []

    numerics = {
        "numeric_metric", "numeric_currency", "numeric_percentage",
    }

    cat_cols = [c for c in columns if classified[c]["semantic_type"] == "categorical"]
    num_cols = [c for c in columns if classified[c]["semantic_type"] in numerics]
    cur_cols = [c for c in columns if classified[c]["semantic_type"] == "numeric_currency"]
    pct_cols = [c for c in columns if classified[c]["semantic_type"] == "numeric_percentage"]
    temp_cols = [c for c in columns if classified[c]["semantic_type"] == "temporal"]

    # Categorical + numeric → bar chart
    if cat_cols and num_cols:
        x = cat_cols[0]
        y_candidates = cur_cols or pct_cols or num_cols
        y = y_candidates[:2]
        fmt = (
            "currency" if cur_cols
            else "percentage" if pct_cols
            else "decimal"
        )
        charts.append({
            "type": "bar",
            "x": x,
            "y": [c for c in y],
            "reason": "categorical+numeric comparison",
            "number_format": fmt,
        })

    # Temporal + numeric → line chart
    if temp_cols and num_cols:
        x = temp_cols[0]
        y = (cur_cols or num_cols)[:2]
        fmt = "currency" if cur_cols else "decimal"
        charts.append({
            "type": "line",
            "x": x,
            "y": [c for c in y],
            "reason": "time trend",
            "number_format": fmt,
        })

    # 2+ numeric → scatter (strongest pair)
    if len(num_cols) >= 2:
        charts.append({
            "type": "scatter",
            "x": num_cols[0],
            "y": [num_cols[1]],
            "reason": "numeric correlation",
            "number_format": "decimal",
        })

    # Categorical with few values → pie
    for c in cat_cols:
        if classified[c]["cardinality"] <= 8:
            charts.append({
                "type": "pie",
                "x": c,
                "y": [],
                "reason": "category proportions",
                "number_format": "decimal",
            })
            break  # only one pie chart

    return charts


def format_classification_for_prompt(classification: dict[str, Any]) -> str:
    """Format classification dict into a readable string for LLM prompts."""
    lines = []
    cols = classification.get("columns", {})
    for name, info in cols.items():
        sem = info.get("semantic_type", "unknown")
        card = info.get("cardinality", "?")
        stats = info.get("stats")
        samples = info.get("sample_values", [])
        parts = [f"  {name}: {sem} (cardinality={card})"]
        if stats:
            parts.append(f"    stats: mean={stats.get('mean')}, min={stats.get('min')}, max={stats.get('max')}")
        if samples:
            parts.append(f"    samples: {samples[:3]}")
        lines.extend(parts)

    domain = classification.get("domain_hint", "general")
    plan = classification.get("analysis_plan", [])
    lines.insert(0, f"Domain hint: {domain}")
    lines.insert(1, f"Suggested analyses: {', '.join(plan)}")
    lines.insert(2, "Column details:")
    return "\n".join(lines)
