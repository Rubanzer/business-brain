"""Insight Quality Gate — filters garbage, scores business value, rewrites for readability.

This module sits between insight generation and storage in the discovery pipeline.
It kills trivial correlations, suppresses meta-insights, applies business-value scoring,
and rewrites titles/descriptions for human readability.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from business_brain.db.discovery_models import Insight

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column family definitions — columns in the same family are trivially related
# ---------------------------------------------------------------------------

_COLUMN_FAMILIES: dict[str, list[str]] = {
    "iron_content": [
        "fe", "iron", "fe_content", "fe_t", "fe_mz", "fe_avg", "fe_total",
        "iron_content", "iron_pct", "fe_pct", "fe_percentage",
    ],
    "power": [
        "kva", "kwh", "power", "energy", "watt", "electricity", "consumption",
        "power_kwh", "power_kva", "energy_kwh", "mwh", "kw", "mw",
        "power_consumption", "energy_consumption", "sec", "specific_energy",
    ],
    "weight": [
        "input", "output", "tonnage", "weight", "gross", "net", "tare",
        "input_weight", "output_weight", "gross_weight", "net_weight",
        "charge_weight", "tap_weight", "billet_weight",
    ],
    "temperature": [
        "temp", "temperature", "celsius", "furnace_temp", "bath_temp",
        "tapping_temp", "pouring_temp", "melt_temp",
    ],
    "time_duration": [
        "duration", "cycle_time", "tap_to_tap", "runtime", "hours",
        "minutes", "time_min", "heat_time", "melt_time", "idle_time",
    ],
    "cost_financial": [
        "cost", "price", "rate", "amount", "expense", "revenue",
        "total_cost", "unit_cost", "cost_per_ton",
    ],
    "quality_yield": [
        "rejection", "defect", "rework", "scrap", "yield", "yield_pct",
        "rejection_rate", "defect_rate", "scrap_rate", "recovery",
    ],
    "alloy": [
        "fesi", "femn", "simg", "alloy", "ferro", "silicon", "manganese",
        "carbon", "sulphur", "phosphorus", "fecr",
    ],
    "identifier": [
        "heat_no", "heat_number", "batch", "lot", "id", "serial",
        "invoice", "challan", "voucher", "entry_no",
    ],
    "electrode": [
        "electrode", "consumption_electrode", "electrode_kg",
        "electrode_consumption",
    ],
}


# ---------------------------------------------------------------------------
# Meta-insight titles that should be suppressed from the Feed
# ---------------------------------------------------------------------------

_META_INSIGHT_PATTERNS: list[str] = [
    r"Time series data available",
    r"Correlation analysis available",
    r"High cardinality categorical",
    r"Constant column",
    r"Day-of-week pattern opportunity",
    r"Monthly trend analysis available",
]

_META_RE = re.compile("|".join(_META_INSIGHT_PATTERNS), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Column readability mappings — technical column names → human labels
# ---------------------------------------------------------------------------

_COLUMN_LABELS: dict[str, str] = {
    "fe_t": "Iron Content (Total)",
    "fe_mz": "Iron Content (Measured Zone)",
    "fe_content": "Iron Content",
    "fe_avg": "Iron Content (Average)",
    "kva": "Power (KVA)",
    "kwh": "Energy (kWh)",
    "power_kwh": "Power Consumption (kWh)",
    "sec": "Specific Energy Consumption",
    "pf": "Power Factor",
    "power_factor": "Power Factor",
    "input_weight": "Input Weight",
    "output_weight": "Output Weight",
    "tap_to_tap": "Tap-to-Tap Time",
    "cycle_time": "Cycle Time",
    "yield_pct": "Yield %",
    "rejection_rate": "Rejection Rate",
    "scrap_rate": "Scrap Rate",
    "furnace_temp": "Furnace Temperature",
    "tapping_temp": "Tapping Temperature",
    "heat_no": "Heat Number",
    "charge_weight": "Charge Weight",
    "billet_weight": "Billet Weight",
    "fesi": "Ferro-Silicon",
    "femn": "Ferro-Manganese",
    "tonnage": "Tonnage",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_quality_gate(
    insights: list[Insight],
    profiles: list | None = None,
    reinforcement_weights=None,
) -> list[Insight]:
    """Run all quality filters and scoring on a list of insights.

    Steps:
        1. Filter trivial correlations (same-family columns)
        2. Suppress meta-insights (schema observations, not business findings)
        3. Apply business-value scoring (replaces generic scoring)
        4. Rewrite titles and descriptions for readability

    Args:
        reinforcement_weights: Optional ReinforcementWeights record from the
            reinforcement loop. When provided, novelty scores are modulated
            by learned multipliers.

    Returns filtered, scored, and rewritten insights.
    """
    original_count = len(insights)

    # Step 1: Kill trivial correlations
    insights = _filter_trivial_correlations(insights)
    trivial_killed = original_count - len(insights)

    # Step 2: Suppress meta-insights
    pre_meta = len(insights)
    insights = _suppress_meta_insights(insights)
    meta_suppressed = pre_meta - len(insights)

    # Step 3: Apply business-value scoring
    for insight in insights:
        _apply_business_scoring(insight, reinforcement_weights)

    # Step 4: Rewrite for readability
    for insight in insights:
        _rewrite_for_readability(insight)

    logger.info(
        "Quality gate: %d → %d insights (killed %d trivial correlations, "
        "suppressed %d meta-insights)",
        original_count,
        len(insights),
        trivial_killed,
        meta_suppressed,
    )

    return insights


# ---------------------------------------------------------------------------
# Step 1: Trivial correlation filter
# ---------------------------------------------------------------------------


def _get_column_family(col_name: str) -> str | None:
    """Return the family name for a column, or None if no family match."""
    col_lower = col_name.lower().strip()
    for family, members in _COLUMN_FAMILIES.items():
        for member in members:
            # Exact match or prefix/suffix match
            if col_lower == member or col_lower.startswith(member + "_") or col_lower.endswith("_" + member):
                return family
    return None


def _columns_share_family(col_a: str, col_b: str) -> bool:
    """Check if two columns belong to the same semantic family."""
    family_a = _get_column_family(col_a)
    family_b = _get_column_family(col_b)
    if family_a and family_b and family_a == family_b:
        return True
    return False


def _columns_similar_names(col_a: str, col_b: str, threshold: float = 0.80) -> bool:
    """Check if two column names are similar using sequence matching.

    Catches cases like: fe_t ↔ fe_mz, input_wt ↔ input_weight
    """
    # Normalize: lowercase, strip underscores at edges
    a = col_a.lower().replace("_", " ").strip()
    b = col_b.lower().replace("_", " ").strip()

    ratio = SequenceMatcher(None, a, b).ratio()
    if ratio >= threshold:
        return True

    # Also check token overlap
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if tokens_a and tokens_b:
        overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
        if overlap >= 0.5:
            return True

    # Check if one is prefix of the other (after removing common suffixes)
    suffixes = {"_t", "_mz", "_avg", "_total", "_pct", "_rate", "_min", "_max"}
    clean_a = col_a.lower()
    clean_b = col_b.lower()
    for s in suffixes:
        clean_a = clean_a.removesuffix(s)
        clean_b = clean_b.removesuffix(s)
    if clean_a and clean_b and (clean_a == clean_b):
        return True

    return False


def _is_trivial_correlation(insight: Insight) -> bool:
    """Determine if a correlation insight is trivially obvious."""
    if insight.insight_type != "correlation":
        return False

    cols = insight.source_columns or []
    if len(cols) < 2:
        return False

    col_a, col_b = cols[0], cols[1]

    # Check 1: Same column family
    if _columns_share_family(col_a, col_b):
        return True

    # Check 2: Similar column names
    if _columns_similar_names(col_a, col_b):
        return True

    return False


def _filter_trivial_correlations(insights: list[Insight]) -> list[Insight]:
    """Remove trivially obvious correlations."""
    return [i for i in insights if not _is_trivial_correlation(i)]


# ---------------------------------------------------------------------------
# Step 2: Meta-insight suppression
# ---------------------------------------------------------------------------


def _is_meta_insight(insight: Insight) -> bool:
    """Check if an insight is a meta-observation rather than a business finding."""
    title = insight.title or ""
    if _META_RE.search(title):
        return True

    # Suppress low-impact "pattern opportunity" insights
    desc = (insight.description or "").lower()
    if "pattern opportunity" in desc and (insight.impact_score or 0) < 20:
        return True

    return False


def _suppress_meta_insights(insights: list[Insight]) -> list[Insight]:
    """Remove meta-insights that are schema observations, not business findings."""
    return [i for i in insights if not _is_meta_insight(i)]


# ---------------------------------------------------------------------------
# Step 3: Business-value scoring (replaces generic _apply_scoring)
# ---------------------------------------------------------------------------


def _apply_business_scoring(insight: Insight, reinforcement_weights=None) -> None:
    """Score insight on business value dimensions.

    Scoring formula:
        Actionability   (0-30): Does the insight suggest a specific action?
        Novelty         (0-25): Is this something the user likely doesn't know?
        Domain Relevance(0-20): Does it match manufacturing domain concerns?
        Magnitude       (0-15): How big is the effect/anomaly?
        Cross-table     (0-10): Does it connect previously unrelated data?

    Total: 0-100, stored as both impact_score and quality_score.
    """
    score = 0

    # --- Actionability (0-30) ---
    actions = insight.suggested_actions or []
    if len(actions) >= 2:
        score += 25
    elif len(actions) >= 1:
        score += 15

    # Bonus for concrete actions (containing specific column/table names)
    for action in actions[:2]:
        if any(c in action for c in ("Check", "Investigate", "Review", "Fix", "Compare")):
            score += 5
            break

    score = min(score, 30)  # Cap at 30

    # --- Novelty (0-25) ---
    novelty = 0
    insight_type = insight.insight_type or ""

    # Cross-event and narrative insights are inherently more novel
    if insight_type == "cross_event":
        novelty += 20
    elif insight_type == "story":
        novelty += 22
    elif insight_type == "composite":
        novelty += 18
    elif insight_type == "anomaly":
        severity = insight.severity or "info"
        if severity == "critical":
            novelty += 20
        elif severity == "warning":
            novelty += 15
        else:
            novelty += 8
    elif insight_type == "correlation":
        novelty += 15  # Correlations are novel for non-technical plant owners
    elif insight_type == "seasonality":
        novelty += 12
    elif insight_type == "trend":
        novelty += 10
    elif insight_type == "schema_change":
        novelty += 15
    elif insight_type == "data_freshness":
        novelty += 12
    else:
        novelty += 5

    # Reinforcement adjustment: modulate novelty by learned multiplier
    if reinforcement_weights is not None:
        from business_brain.discovery.reinforcement_loop import get_multiplier
        novelty = int(novelty * get_multiplier(
            reinforcement_weights, "insight_type_multipliers", insight_type))

    score += min(novelty, 25)

    # --- Domain relevance (0-20) ---
    domain_score = 0
    title_lower = (insight.title or "").lower()
    desc_lower = (insight.description or "").lower()
    text_combined = title_lower + " " + desc_lower

    # Manufacturing-relevant keywords boost score
    mfg_keywords = [
        "yield", "sec", "power", "energy", "furnace", "temperature", "heat",
        "scrap", "tonnage", "production", "shift", "rejection", "defect",
        "alloy", "electrode", "kva", "kwh", "tap", "billet", "slag",
        "efficiency", "consumption", "cost per ton", "leakage", "loss",
    ]
    matches = sum(1 for kw in mfg_keywords if kw in text_combined)
    domain_score += min(matches * 5, 20)

    # Leakage/critical keyword boost
    leakage_keywords = [
        "weighbridge", "manipulation", "theft", "mismatch",
        "idle time", "refractory", "penalty",
    ]
    leakage_hits = sum(1 for kw in leakage_keywords if kw in text_combined)
    domain_score += min(leakage_hits * 8, 16)

    score += min(domain_score, 20)

    # --- Magnitude (0-15) ---
    magnitude = 0
    evidence = insight.evidence or {}

    # For anomalies: use outlier deviation
    if insight_type == "anomaly":
        severity = insight.severity or "info"
        if severity == "critical":
            magnitude += 15
        elif severity == "warning":
            magnitude += 10
        else:
            magnitude += 5

    # For correlations: stronger correlation = more magnitude
    est_r = evidence.get("estimated_correlation")
    if est_r is not None:
        if abs(est_r) >= 0.9:
            magnitude += 15
        elif abs(est_r) >= 0.8:
            magnitude += 10
        elif abs(est_r) >= 0.7:
            magnitude += 7

    # For cross-events: use the percentage difference
    pct_diff = evidence.get("pct_difference")
    if pct_diff is not None:
        if abs(pct_diff) >= 30:
            magnitude += 15
        elif abs(pct_diff) >= 20:
            magnitude += 10
        elif abs(pct_diff) >= 10:
            magnitude += 5

    score += min(magnitude, 15)

    # --- Cross-table bonus (0-10) ---
    source_tables = insight.source_tables or []
    if len(source_tables) > 1:
        score += 10
    elif len(source_tables) == 1:
        score += 0

    # Final score
    insight.impact_score = min(score, 100)
    insight.quality_score = insight.impact_score


# ---------------------------------------------------------------------------
# Step 4: Readability rewriter
# ---------------------------------------------------------------------------


def _humanize_column(col_name: str) -> str:
    """Convert a technical column name to a human-readable label."""
    # Check explicit label mapping first
    if col_name.lower() in _COLUMN_LABELS:
        return _COLUMN_LABELS[col_name.lower()]

    # Auto-generate: replace underscores, title-case
    label = col_name.replace("_", " ").strip()

    # Capitalize known acronyms
    acronyms = {"kva", "kwh", "sec", "pf", "fe", "mt", "id", "no"}
    words = label.split()
    result = []
    for word in words:
        if word.lower() in acronyms:
            result.append(word.upper())
        else:
            result.append(word.capitalize())

    return " ".join(result)


def _humanize_table(table_name: str) -> str:
    """Convert a technical table name to a human-readable label."""
    return table_name.replace("_", " ").title()


def _rewrite_for_readability(insight: Insight) -> None:
    """Rewrite insight title and description for business readability.

    - Replace technical column names with human labels
    - Add "So what?" context where possible
    - Format numbers with units
    """
    title = insight.title or ""
    description = insight.description or ""

    # Replace column names in title and description
    cols = insight.source_columns or []
    for col in cols:
        human = _humanize_column(col)
        if human != col:
            title = title.replace(col, human)
            description = description.replace(col, human)

    # Replace table names
    tables = insight.source_tables or []
    for table in tables:
        human_table = _humanize_table(table)
        # Only replace in format "in table_name" to avoid over-replacement
        title = title.replace(f"in {table}", f"in {human_table}")
        description = description.replace(f"in {table}", f"in {human_table}")
        # Also handle "Table table_name has..."
        description = description.replace(f"Table {table} ", f"Table {human_table} ")

    # Clean up the ↔ separator for correlations
    title = title.replace(" ↔ ", " vs ")

    # Add "So what?" to description for certain insight types
    evidence = insight.evidence or {}
    so_what = _generate_so_what(insight, evidence)
    if so_what and so_what not in description:
        description = description.rstrip(". ") + ". " + so_what

    insight.title = title
    insight.description = description


def _generate_so_what(insight: Insight, evidence: dict) -> str:
    """Generate a "So what?" sentence for actionability."""
    insight_type = insight.insight_type or ""

    if insight_type == "anomaly":
        severity = insight.severity or "info"
        if severity == "critical":
            return "⚠️ This requires immediate attention — it may indicate data corruption or a serious operational issue."
        elif severity == "warning":
            return "This warrants investigation to determine if it's a data issue or an operational concern."

    if insight_type == "correlation":
        est_r = evidence.get("estimated_correlation")
        if est_r and abs(est_r) >= 0.8:
            return "A strong relationship suggests one metric could predict the other — useful for forecasting."
        elif est_r and abs(est_r) >= 0.7:
            return "This relationship is worth monitoring — changes in one metric may signal changes in the other."

    if insight_type == "cross_event":
        return "This cross-table pattern could reveal hidden dependencies between different parts of your operation."

    if insight_type == "composite":
        return "Composite metrics combine multiple data points into a single actionable number for decision-making."

    if insight_type == "seasonality":
        return "Understanding this pattern can help optimize scheduling, staffing, and resource allocation."

    return ""
