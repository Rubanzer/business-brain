"""Domain knowledge registry — industry-specific intelligence for agents.

Contains terminology, benchmarks, red flags, and common leakage patterns
for each supported industry. Used by agents to provide expert-level analysis.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steel Manufacturing (Secondary — Induction Furnace)
# ---------------------------------------------------------------------------

STEEL_MANUFACTURING: dict = {
    "industry_name": "Secondary Steel Manufacturing (Induction Furnace)",

    "terminology": {
        "heat": "A single batch/cycle of steel production in the furnace. Each heat produces one ladle of molten steel.",
        "tap-to-tap": "Time between consecutive furnace taps. Target: 45-90 min. Shorter = more efficient.",
        "tapping": "Pouring molten steel from furnace into the ladle. Temperature at tapping is critical for quality.",
        "yield": "Output weight / Input weight × 100. Measures how much input scrap becomes usable output. Target: 88-94%.",
        "SEC": "Specific Energy Consumption = kWh per ton of output. Lower is better. Target: 450-600 for IF.",
        "Fe content": "Iron purity percentage in scrap. Higher = better quality scrap. Typical: 85-95%.",
        "power factor": "Ratio of real power to apparent power. Target: ≥0.95. Low PF incurs DISCOM penalties.",
        "KVA": "Kilovolt-Ampere — apparent power drawn by the furnace. Related to but different from actual energy (kWh).",
        "charge": "The scrap loaded into the furnace for melting. A heat may have multiple charges.",
        "billet": "Semi-finished steel product from continuous casting. Precursor to TMT bars.",
        "TMT bar": "Thermo-Mechanically Treated reinforcement bar. Primary finished product for construction.",
        "slag": "Non-metallic byproduct of steelmaking. Floats on top of molten steel.",
        "refractory": "Heat-resistant lining of the furnace. Degrades over time and needs periodic relining.",
        "alloy additions": "FeSi, FeMn, etc. added to adjust steel chemistry. Must be precisely controlled.",
        "DISCOM": "Distribution Company — the electricity provider. Imposes penalties for low power factor.",
        "weighbridge": "Industrial scale for weighing incoming scrap and outgoing products.",
        "ladle": "Container for holding and transporting molten steel from furnace to casting.",
        "shift": "Work period — typically 3 shifts per day (morning/afternoon/night, 8 hours each).",
        "furnace lining": "Refractory layer inside the furnace. Typical life: 200-400 heats before relining.",
    },

    "benchmarks": {
        "yield_pct": {
            "good": ">92%",
            "average": "88-92%",
            "poor": "<88%",
            "unit": "%",
            "context": "Output/Input weight ratio. Below 85% likely indicates measurement error.",
        },
        "sec_kwh_per_ton": {
            "good": "<500",
            "average": "500-600",
            "poor": ">600",
            "unit": "kWh/ton",
            "context": "Total energy per ton of output. Above 700 suggests furnace issues.",
        },
        "power_factor": {
            "good": ">0.95",
            "average": "0.90-0.95",
            "poor": "<0.90",
            "unit": "",
            "context": "Below 0.85 means capacitor bank failure and DISCOM penalty.",
        },
        "tap_to_tap_min": {
            "good": "<60",
            "average": "60-90",
            "poor": ">90",
            "unit": "minutes",
            "context": "Total cycle time. Above 120 min suggests operational problems.",
        },
        "furnace_temp": {
            "good": "1550-1620°C",
            "average": "1520-1650°C",
            "poor": "<1520 or >1700°C",
            "unit": "°C",
            "context": "Tapping too hot wastes energy; too cold causes quality issues.",
        },
        "rejection_rate": {
            "good": "<1.5%",
            "average": "1.5-3%",
            "poor": ">3%",
            "unit": "%",
            "context": "Percentage of output rejected due to quality issues.",
        },
        "electrode_consumption": {
            "good": "<3 kg/ton",
            "average": "3-5 kg/ton",
            "poor": ">5 kg/ton",
            "unit": "kg/ton",
            "context": "Electrode consumption per ton of steel produced.",
        },
        "scrap_cost_per_ton": {
            "good": "<₹28,000",
            "average": "₹28,000-32,000",
            "poor": ">₹32,000",
            "unit": "₹/ton",
            "context": "Cost varies by grade and market. HMS grade cheapest.",
        },
    },

    "red_flags": [
        "Yield below 85% → likely measurement error (weighbridge manipulation) or severe scrap quality issue",
        "SEC above 700 kWh/ton → furnace lining degradation, power system issue, or excessive idle time",
        "Power factor below 0.85 → capacitor bank failure, will incur DISCOM penalty (₹ per kVARh)",
        "Temperature above 1700°C → refractory damage risk, wasted energy",
        "Temperature below 1500°C → steel quality issues, cannot achieve target chemistry",
        "Slag rate above 8% → excessive oxidation, check scrap quality and melting practice",
        "Tap-to-tap above 120 min → investigate idle time, power interruptions, or scrap delays",
        "Electrode consumption above 6 kg/ton → check electrode quality and furnace operating practices",
        "Rejection rate above 5% → systemic quality issue, review chemistry and casting parameters",
        "Alloy addition above 15 kg/ton → over-alloying, review target chemistry and calculation",
        "Energy cost above 60% of total → power tariff issue or SEC problem",
        "Scrap yield variance >5% between shifts → possible weighbridge manipulation or reporting errors",
    ],

    "common_leakages": [
        {
            "name": "Weighbridge Manipulation",
            "description": "Input weight inflated or output weight deflated to show higher yield than actual",
            "detection": "Compare yield across shifts/operators, cross-check with power consumption",
            "impact": "2-5% revenue loss",
        },
        {
            "name": "Power Theft via Meter Bypass",
            "description": "Actual consumption higher than metered, causing incorrect SEC calculations",
            "detection": "Compare DISCOM bill with internal meter readings",
            "impact": "₹5-15 lakhs/month for a typical IF plant",
        },
        {
            "name": "Scrap Grade Mismatch",
            "description": "Paying for Grade A scrap but receiving Grade B or mixed",
            "detection": "Track Fe content per supplier, correlate with supplier invoices",
            "impact": "₹500-1500 per ton of scrap",
        },
        {
            "name": "Alloy Over-addition",
            "description": "Adding more FeSi/FeMn than required for target chemistry",
            "detection": "Compare actual alloy used vs calculated requirement per heat",
            "impact": "₹200-500 per ton of steel",
        },
        {
            "name": "Furnace Idle Time",
            "description": "Heat loss from furnace sitting idle between charges (no scrap ready, crane delays)",
            "detection": "Track gap between end-of-tap and start-of-next-charge",
            "impact": "50-100 kWh per idle hour",
        },
        {
            "name": "Refractory Over-consumption",
            "description": "Relining more frequently than needed or using excessive patching material",
            "detection": "Track refractory cost per heat, compare with lining life",
            "impact": "₹100-300 per heat",
        },
        {
            "name": "Casting Yield Loss",
            "description": "Excessive metal loss during continuous casting (crop ends, breakouts)",
            "detection": "Track casting yield separately from furnace yield",
            "impact": "2-4% of molten steel",
        },
    ],

    "key_ratios": {
        "energy_cost_ratio": {
            "formula": "Energy Cost / Total Production Cost × 100",
            "healthy_range": "35-50%",
            "context": "Energy is typically the largest cost component in IF steelmaking",
        },
        "scrap_cost_ratio": {
            "formula": "Scrap Cost / Total Production Cost × 100",
            "healthy_range": "55-70%",
            "context": "Scrap is the largest material cost",
        },
        "conversion_cost": {
            "formula": "Total Cost - Scrap Cost / Output Tonnage",
            "healthy_range": "₹3,000-5,000 per ton",
            "context": "All costs other than raw scrap per ton of output",
        },
        "power_per_heat": {
            "formula": "Total kWh / Number of Heats",
            "healthy_range": "1,500-3,000 kWh/heat",
            "context": "Depends on furnace capacity and scrap quality",
        },
    },
}


# ---------------------------------------------------------------------------
# Industry registry — maps industry name to knowledge base
# ---------------------------------------------------------------------------

_INDUSTRY_REGISTRY: dict[str, dict] = {
    "steel": STEEL_MANUFACTURING,
    "steel manufacturing": STEEL_MANUFACTURING,
    "secondary steel": STEEL_MANUFACTURING,
    "secondary steel manufacturing": STEEL_MANUFACTURING,
    "induction furnace": STEEL_MANUFACTURING,
    "manufacturing": STEEL_MANUFACTURING,  # default for now
}


def get_domain_knowledge(industry: str | None = None) -> dict | None:
    """Get the domain knowledge base for a given industry.

    Returns None if no matching knowledge base is found.
    """
    if not industry:
        return STEEL_MANUFACTURING  # default

    industry_lower = industry.lower().strip()
    return _INDUSTRY_REGISTRY.get(industry_lower)


def get_terminology(industry: str | None = None) -> dict[str, str]:
    """Get industry terminology as a dict of {term: definition}."""
    knowledge = get_domain_knowledge(industry)
    return knowledge.get("terminology", {}) if knowledge else {}


def get_benchmarks(industry: str | None = None) -> dict:
    """Get industry benchmarks for key metrics."""
    knowledge = get_domain_knowledge(industry)
    return knowledge.get("benchmarks", {}) if knowledge else {}


def get_red_flags(industry: str | None = None) -> list[str]:
    """Get industry red flags — conditions that always need attention."""
    knowledge = get_domain_knowledge(industry)
    return knowledge.get("red_flags", []) if knowledge else []


def get_leakage_patterns(industry: str | None = None) -> list[dict]:
    """Get common financial/operational leakage patterns."""
    knowledge = get_domain_knowledge(industry)
    return knowledge.get("common_leakages", []) if knowledge else []


def format_benchmarks_for_prompt(industry: str | None = None) -> str:
    """Format benchmarks as a text block suitable for LLM prompts."""
    benchmarks = get_benchmarks(industry)
    if not benchmarks:
        return ""

    lines = ["INDUSTRY BENCHMARKS (use these to evaluate metric values):"]
    for metric, info in benchmarks.items():
        name = metric.replace("_", " ").title()
        unit = info.get("unit", "")
        unit_str = f" ({unit})" if unit else ""
        lines.append(f"  {name}{unit_str}: Good={info['good']}, Average={info['average']}, Poor={info['poor']}")
        if "context" in info:
            lines.append(f"    → {info['context']}")
    return "\n".join(lines)


def format_red_flags_for_prompt(industry: str | None = None) -> str:
    """Format red flags as a text block suitable for LLM prompts."""
    flags = get_red_flags(industry)
    if not flags:
        return ""

    lines = ["RED FLAGS (always highlight these if detected in data):"]
    for flag in flags:
        lines.append(f"  ⚠️ {flag}")
    return "\n".join(lines)


def format_terminology_for_prompt(industry: str | None = None) -> str:
    """Format key terminology as a text block suitable for LLM prompts."""
    terms = get_terminology(industry)
    if not terms:
        return ""

    lines = ["INDUSTRY TERMINOLOGY:"]
    for term, definition in list(terms.items())[:20]:  # top 20 most important
        lines.append(f"  {term}: {definition}")
    return "\n".join(lines)


def format_leakages_for_prompt(industry: str | None = None) -> str:
    """Format leakage patterns as a text block suitable for LLM prompts."""
    leakages = get_leakage_patterns(industry)
    if not leakages:
        return ""

    lines = ["COMMON LEAKAGE PATTERNS (financial/operational losses to watch for):"]
    for leak in leakages:
        lines.append(f"  🔍 {leak['name']}: {leak['description']}")
        lines.append(f"     Detection: {leak['detection']}")
        lines.append(f"     Typical impact: {leak['impact']}")
    return "\n".join(lines)
