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


# ---------------------------------------------------------------------------
# Industry Setup Templates — pre-built configurations for Quick Setup
# ---------------------------------------------------------------------------

INDUSTRY_TEMPLATES: dict[str, dict] = {
    "steel": {
        "process_steps": [
            {"step_order": 1, "process_name": "Scrap Procurement & Sorting", "inputs": "Raw scrap (HMS1, HMS2, shredded, turnings), Sponge iron/DRI", "outputs": "Sorted scrap by grade, Weighed lots", "key_metric": "Fe content (%)", "target_range": "85-95%", "linked_table": "purchases"},
            {"step_order": 2, "process_name": "Furnace Charging", "inputs": "Sorted scrap, Sponge iron, Pig iron", "outputs": "Charged furnace (3-5 MT per heat)", "key_metric": "Charge weight (MT)", "target_range": "3-5 MT", "linked_table": "production"},
            {"step_order": 3, "process_name": "Induction Furnace Melting", "inputs": "Charged scrap, Electricity (kWh)", "outputs": "Molten steel at 1600-1650\u00b0C", "key_metric": "SEC (kWh/ton)", "target_range": "500-625 kWh/ton", "linked_table": "power_data"},
            {"step_order": 4, "process_name": "Chemistry Adjustment & Refining", "inputs": "Molten steel, FeSi, FeMn, Carbon raiser, Aluminum", "outputs": "Refined steel (target C, Mn, Si, S, P)", "key_metric": "Alloy consumption (kg/ton)", "target_range": "FeSi: 3-6, FeMn: 5-10 kg/ton", "linked_table": "alloy_data"},
            {"step_order": 5, "process_name": "Tapping to Ladle", "inputs": "Refined molten steel", "outputs": "Ladle with liquid steel", "key_metric": "Tapping temperature (\u00b0C)", "target_range": "1640-1660\u00b0C", "linked_table": "production"},
            {"step_order": 6, "process_name": "Continuous Casting (CCM)", "inputs": "Ladle steel, Tundish", "outputs": "Billets (various sections)", "key_metric": "Casting yield (%)", "target_range": "95-97%", "linked_table": "production"},
            {"step_order": 7, "process_name": "Reheating", "inputs": "Cold billets, Furnace oil", "outputs": "Hot billets at rolling temp", "key_metric": "Fuel consumption (litres/ton)", "target_range": "30-35 litres/ton", "linked_table": "fuel_data"},
            {"step_order": 8, "process_name": "Rolling Mill", "inputs": "Hot billets", "outputs": "TMT bars (various diameters)", "key_metric": "Rolling yield (%)", "target_range": "95-97%", "linked_table": "rolling_data"},
            {"step_order": 9, "process_name": "TMT Quenching (Thermex)", "inputs": "Hot rolled bars, Cooling water", "outputs": "Quenched & tempered TMT bars", "key_metric": "Mechanical properties (UTS, YS)", "target_range": "Fe500D spec", "linked_table": "quality_data"},
            {"step_order": 10, "process_name": "Shearing, Bundling & Dispatch", "inputs": "Finished TMT bars/billets", "outputs": "Bundled product, loaded trucks", "key_metric": "Dispatch weight (MT)", "target_range": "Match production \u00b10.5%", "linked_table": "dispatch"},
        ],
        "metrics": [
            {"metric_name": "Power Consumption (SEC)", "table_name": "power_data", "column_name": "sec_kwh_per_ton", "unit": "kWh/ton", "normal_min": 500, "normal_max": 625, "warning_min": 625, "warning_max": 750, "critical_min": 750, "critical_max": 9999},
            {"metric_name": "Melting Loss", "table_name": "production", "column_name": "melting_loss_pct", "unit": "%", "normal_min": 1.0, "normal_max": 2.0, "warning_min": 2.0, "warning_max": 4.0, "critical_min": 4.0, "critical_max": 100},
            {"metric_name": "Overall Yield", "table_name": "production", "column_name": "yield_pct", "unit": "%", "normal_min": 85, "normal_max": 95, "warning_min": 80, "warning_max": 85, "critical_min": 0, "critical_max": 80},
            {"metric_name": "Tap-to-Tap Time", "table_name": "production", "column_name": "tap_to_tap_min", "unit": "minutes", "normal_min": 60, "normal_max": 80, "warning_min": 80, "warning_max": 100, "critical_min": 100, "critical_max": 9999},
            {"metric_name": "Power Factor", "table_name": "power_data", "column_name": "power_factor", "unit": "ratio", "normal_min": 0.95, "normal_max": 1.0, "warning_min": 0.90, "warning_max": 0.95, "critical_min": 0, "critical_max": 0.90},
            {"metric_name": "Rolling Yield", "table_name": "rolling_data", "column_name": "rolling_yield_pct", "unit": "%", "normal_min": 95, "normal_max": 98, "warning_min": 93, "warning_max": 95, "critical_min": 0, "critical_max": 93},
            {"metric_name": "Refractory Consumption", "table_name": "refractory_data", "column_name": "consumption_kg_per_ton", "unit": "kg/ton", "normal_min": 3.0, "normal_max": 3.6, "warning_min": 3.6, "warning_max": 5.0, "critical_min": 5.0, "critical_max": 9999},
            {"metric_name": "Tapping Temperature", "table_name": "production", "column_name": "tapping_temp_c", "unit": "\u00b0C", "normal_min": 1620, "normal_max": 1660, "warning_min": 1660, "warning_max": 1690, "critical_min": 1690, "critical_max": 9999},
            {"metric_name": "Furnace Utilization", "table_name": "production", "column_name": "utilization_pct", "unit": "%", "normal_min": 75, "normal_max": 95, "warning_min": 60, "warning_max": 75, "critical_min": 0, "critical_max": 60},
            {"metric_name": "Receivables >90 days", "table_name": "finance", "column_name": "receivables_over_90_pct", "unit": "%", "normal_min": 0, "normal_max": 5, "warning_min": 5, "warning_max": 15, "critical_min": 15, "critical_max": 100},
            {"metric_name": "Customer Concentration (Top 3)", "table_name": "sales", "column_name": "top3_customer_pct", "unit": "%", "normal_min": 0, "normal_max": 30, "warning_min": 30, "warning_max": 50, "critical_min": 50, "critical_max": 100},
            {"metric_name": "Scrap Inventory Days", "table_name": "inventory", "column_name": "scrap_inventory_days", "unit": "days", "normal_min": 7, "normal_max": 15, "warning_min": 15, "warning_max": 25, "critical_min": 25, "critical_max": 9999},
        ],
        "inputs": [
            {"io_type": "input", "name": "Scrap Iron (HMS1/HMS2)", "source_or_destination": "Scrap dealers, importers", "unit": "MT", "typical_range": "3-5 per heat", "linked_table": "purchases"},
            {"io_type": "input", "name": "Sponge Iron / DRI", "source_or_destination": "DRI plants", "unit": "MT", "typical_range": "0.5-1.5 per heat", "linked_table": "purchases"},
            {"io_type": "input", "name": "Electricity", "source_or_destination": "State grid / DISCOM", "unit": "kWh", "typical_range": "1500-3000 per heat", "linked_table": "power_data"},
            {"io_type": "input", "name": "Ferro-Silicon (FeSi)", "source_or_destination": "Alloy suppliers", "unit": "kg", "typical_range": "3-6 per ton", "linked_table": "alloy_data"},
            {"io_type": "input", "name": "Ferro-Manganese (FeMn)", "source_or_destination": "Alloy suppliers", "unit": "kg", "typical_range": "5-10 per ton", "linked_table": "alloy_data"},
            {"io_type": "input", "name": "Carbon Raiser", "source_or_destination": "Carbon suppliers", "unit": "kg", "typical_range": "2-5 per ton", "linked_table": "alloy_data"},
            {"io_type": "input", "name": "Ramming Mass (refractory)", "source_or_destination": "Refractory suppliers", "unit": "kg", "typical_range": "3.4-3.6 per ton", "linked_table": "refractory_data"},
            {"io_type": "input", "name": "Furnace Oil (reheating)", "source_or_destination": "Fuel suppliers", "unit": "litres", "typical_range": "30-35 per ton", "linked_table": "fuel_data"},
        ],
        "outputs": [
            {"io_type": "output", "name": "TMT Bars (Fe500D)", "source_or_destination": "Dealers, builders, fabricators", "unit": "MT", "typical_range": "2.5-4.2 per heat", "linked_table": "dispatch"},
            {"io_type": "output", "name": "Billets", "source_or_destination": "Rolling mills, buyers", "unit": "MT", "typical_range": "2.7-4.5 per heat", "linked_table": "production"},
            {"io_type": "output", "name": "Slag", "source_or_destination": "Cement plants, road contractors", "unit": "MT", "typical_range": "0.05-0.25 per heat", "linked_table": "waste_data"},
            {"io_type": "output", "name": "Mill Scale", "source_or_destination": "Scale dealers", "unit": "MT", "typical_range": "0.03-0.07 per ton rolled", "linked_table": "waste_data"},
            {"io_type": "output", "name": "Crop Ends", "source_or_destination": "Internal recharge / scrap dealers", "unit": "MT", "typical_range": "1-1.5% of rolled output", "linked_table": "waste_data"},
        ],
    },
}


def get_industry_template(industry: str) -> dict | None:
    """Get setup template for a given industry."""
    return INDUSTRY_TEMPLATES.get(industry.lower().strip())

