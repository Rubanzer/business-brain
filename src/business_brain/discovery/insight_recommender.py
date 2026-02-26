"""Insight recommender — suggests what analyses to run next.

Two-tier cross-table intelligence:
  Tier 1: Hardcoded templates (fast, reliable) for common patterns
  Tier 2: LLM-generated suggestions (flexible, any industry) for novel data

Also provides domain-aware single-table recommendations.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    """A recommended analysis to run."""
    title: str
    description: str
    analysis_type: str  # benchmark, cohort, correlation, time_trend, anomaly, forecast + domain types
    target_table: str
    columns: list[str]
    priority: int  # 1-100
    reason: str


@dataclass
class EntityGroup:
    """A group of tables sharing a common entity type.

    This is the core unit for cross-table intelligence. If 3 tables all have
    a "buyer" column (or are linked by discovered relationships), they form
    one EntityGroup and can be analyzed together.
    """
    entity_type: str                              # "customer", "supplier", etc.
    tables: list[str] = field(default_factory=list)  # all tables in the group
    entity_columns: dict[str, list[str]] = field(default_factory=dict)  # table → entity column names


# ---------------------------------------------------------------------------
# Domain-specific recommendations (higher priority than generic)
# ---------------------------------------------------------------------------

_DOMAIN_RECOMMENDATIONS: dict[str, list[dict]] = {
    "manufacturing": [
        {
            "title": "Heat analysis on {table}",
            "type": "heat_analysis",
            "keywords": ["heat", "batch", "weight", "tonnage"],
            "priority": 92,
            "description": "Analyze heat-wise production — weight distribution, grade mix, cycle time variability",
            "reason": "Table has heat/batch and weight data suitable for production analysis.",
        },
        {
            "title": "Material balance for {table}",
            "type": "material_balance",
            "keywords": ["input", "output", "charge", "produced"],
            "priority": 90,
            "description": "Track material recovery — input vs output with loss quantification per stage",
            "reason": "Table has input/output columns for yield and loss tracking.",
        },
        {
            "title": "Shift performance comparison on {table}",
            "type": "production_scheduler",
            "keywords": ["shift", "output", "production"],
            "priority": 87,
            "description": "Compare output across shifts — find best/worst performing shifts with variance analysis",
            "reason": "Table has shift and output data for performance comparison.",
        },
        {
            "title": "OEE calculation for {table}",
            "type": "efficiency_metrics",
            "keywords": ["availability", "performance", "quality"],
            "priority": 85,
            "description": "Calculate Overall Equipment Effectiveness — availability × performance × quality",
            "reason": "Table has OEE component columns for equipment efficiency scoring.",
        },
        {
            "title": "Downtime Pareto analysis on {table}",
            "type": "downtime_analyzer",
            "keywords": ["machine", "downtime", "duration", "breakdown"],
            "priority": 83,
            "description": "Identify top downtime causes — worst machines, MTTR, recurring failures",
            "reason": "Table has machine and downtime data for reliability analysis.",
        },
    ],
    "energy": [
        {
            "title": "Power load profile analysis for {table}",
            "type": "power_monitor",
            "keywords": ["kw", "kva", "power", "consumption", "demand"],
            "priority": 90,
            "description": "Analyze power consumption pattern — load factor, peak demand, SEC per ton",
            "reason": "Table has power consumption data for energy optimization.",
        },
    ],
    "quality": [
        {
            "title": "Rejection rate & SPC analysis on {table}",
            "type": "quality_control",
            "keywords": ["defect", "reject", "rejection", "rework", "scrap"],
            "priority": 92,
            "description": "Defect rate analysis with SPC — entity-wise rejection rates, Pareto of defect types",
            "reason": "Table has defect/rejection data for quality control analysis.",
        },
    ],
    "procurement": [
        {
            "title": "Supplier scorecard for {table}",
            "type": "supplier_scorecard",
            "keywords": ["supplier", "vendor", "party"],
            "priority": 88,
            "description": "Score and rank suppliers — weighted performance across quality, delivery, and cost metrics",
            "reason": "Table has supplier data for performance scoring.",
        },
        {
            "title": "Rate comparison & savings potential in {table}",
            "type": "rate_analysis",
            "keywords": ["rate", "price", "cost", "amount"],
            "priority": 85,
            "description": "Compare procurement rates across suppliers — identify savings potential and best rates",
            "reason": "Table has rate/price data for cost optimization.",
        },
    ],
    "logistics": [
        {
            "title": "Dispatch & gate traffic analysis for {table}",
            "type": "dispatch_gate",
            "keywords": ["vehicle", "truck", "gate", "dispatch"],
            "priority": 83,
            "description": "Analyze vehicle traffic patterns — peak hours, throughput bottlenecks, turnaround time",
            "reason": "Table has vehicle/dispatch data for logistics optimization.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Cross-table intelligence templates
# ---------------------------------------------------------------------------

_ENTITY_KEYWORDS: dict[str, list[str]] = {
    "customer": [
        "customer", "buyer", "party", "client", "debtor",
        "consignee", "account_holder", "account", "dealer",
        "distributor", "retailer", "purchaser", "receiver",
    ],
    "supplier": [
        "supplier", "vendor", "seller", "creditor",
        "manufacturer", "provider", "source",
    ],
    "employee": [
        "employee", "worker", "staff", "operator", "person",
        "technician", "driver", "agent", "name",
    ],
    "product": [
        "product", "item", "sku", "material", "grade",
        "commodity", "goods", "article", "stock",
    ],
    "machine": [
        "machine", "equipment", "furnace", "line", "asset",
        "mill", "plant", "unit", "reactor", "kiln",
    ],
}

_CROSS_TABLE_TEMPLATES: dict[str, list[dict]] = {
    "customer": [
        {
            "title": "Build buyer credit score report from {tables}",
            "description": (
                "Combine booking/order data with dispatch records and payment history "
                "to create a creditworthiness score for each buyer. Score based on: "
                "payment timeliness (40%), order volume consistency (30%), "
                "dispatch fulfillment rate (30%). Then recommend a plan of action for each buyer."
            ),
            "priority": 95,
            "required_data": [
                {"table_keywords": ["booking", "order", "sales", "invoice"],
                 "col_keywords": ["order", "booking", "invoice", "quantity"]},
                {"table_keywords": ["payment", "receipt", "collection", "ledger"],
                 "col_keywords": ["payment", "receipt", "collection", "paid", "received"]},
            ],
            "reason": "Tables {tables} can be linked by {entity} — combine for credit scoring.",
        },
        {
            "title": "Customer profitability analysis across {tables}",
            "description": (
                "Combine sales revenue with cost/dispatch data to compute net profitability "
                "per customer. Identify high-revenue but low-profit customers and recommend "
                "pricing adjustments or service level changes."
            ),
            "priority": 88,
            "required_data": [
                {"table_keywords": ["sales", "revenue", "billing", "order"],
                 "col_keywords": ["revenue", "sales", "billing", "amount"]},
                {"table_keywords": ["cost", "expense", "freight", "dispatch", "transport"],
                 "col_keywords": ["cost", "expense", "freight", "dispatch"]},
            ],
            "reason": "Tables {tables} can be linked by {entity} — combine for profitability analysis.",
        },
    ],
    "supplier": [
        {
            "title": "Total cost of ownership report from {tables}",
            "description": (
                "Combine procurement rates, quality rejection data, and delivery performance "
                "to calculate the true cost of each supplier — not just the rate, but factoring in "
                "quality losses, delays, and rework costs."
            ),
            "priority": 93,
            "required_data": [
                {"table_keywords": ["purchase", "procurement", "po", "order", "rate"],
                 "col_keywords": ["rate", "price", "cost", "amount"]},
                {"table_keywords": ["quality", "reject", "inspection", "test"],
                 "col_keywords": ["quality", "reject", "defect", "grade"]},
            ],
            "reason": "Tables {tables} can be linked by {entity} — combine for total cost analysis.",
        },
    ],
    "product": [
        {
            "title": "Product lifecycle analysis across {tables}",
            "description": (
                "Combine production, quality, sales, and inventory data for each product/grade "
                "to track the full lifecycle — from manufacturing cost and defect rate through "
                "sales velocity and margin."
            ),
            "priority": 85,
            "required_data": [
                {"table_keywords": ["production", "manufacturing", "output", "batch"],
                 "col_keywords": ["production", "output", "manufactured", "quantity"]},
                {"table_keywords": ["sales", "dispatch", "order", "revenue"],
                 "col_keywords": ["sales", "revenue", "dispatch", "order"]},
            ],
            "reason": "Tables {tables} can be linked by {entity} — combine for lifecycle tracking.",
        },
    ],
    "machine": [
        {
            "title": "Equipment total performance report from {tables}",
            "description": (
                "Combine production output, downtime records, maintenance logs, and power "
                "consumption per machine to build a comprehensive equipment health and "
                "performance scorecard."
            ),
            "priority": 87,
            "required_data": [
                {"table_keywords": ["production", "output", "tonnage"],
                 "col_keywords": ["production", "output", "tonnage", "quantity"]},
                {"table_keywords": ["downtime", "breakdown", "maintenance"],
                 "col_keywords": ["downtime", "breakdown", "maintenance", "repair"]},
            ],
            "reason": "Tables {tables} can be linked by {entity} — combine for equipment scorecard.",
        },
    ],
    "employee": [
        {
            "title": "Employee performance 360° from {tables}",
            "description": (
                "Combine attendance, production output, quality metrics, and HR records "
                "to build a complete performance picture for each employee/operator."
            ),
            "priority": 82,
            "required_data": [
                {"table_keywords": ["attendance", "leave", "hr", "muster"],
                 "col_keywords": ["attendance", "leave", "present", "absent"]},
                {"table_keywords": ["production", "output", "performance", "target"],
                 "col_keywords": ["output", "production", "performance", "task"]},
            ],
            "reason": "Tables {tables} can be linked by {entity} — combine for 360° performance view.",
        },
    ],
}


# ---------------------------------------------------------------------------
# Tier 2 cache for LLM-generated suggestions
# ---------------------------------------------------------------------------

_dynamic_suggestion_cache: dict[str, tuple[float, list[Recommendation]]] = {}
_CACHE_TTL = 3600  # 1 hour


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------

def recommend_analyses(
    profiles: list[dict],
    insights: list[dict],
    relationships: list[dict],
    max_recommendations: int = 10,
) -> list[Recommendation]:
    """Generate analysis recommendations (sync — Tier 1 only).

    Backward-compatible sync entry point. For full Tier 1 + Tier 2 (LLM),
    use ``recommend_analyses_async()`` instead.
    """
    recs, _groups, _matched = _recommend_tier1(profiles, insights, relationships)
    return _finalize(recs, max_recommendations)


async def recommend_analyses_async(
    profiles: list[dict],
    insights: list[dict],
    relationships: list[dict],
    company_context: dict | None = None,
    max_recommendations: int = 10,
) -> list[Recommendation]:
    """Full recommendation pipeline: Tier 1 (templates) + Tier 2 (LLM-generated).

    Tier 2 always runs for ALL entity groups with 2+ tables — not just
    unmatched ones. This lets the LLM suggest COMPLEMENTARY analyses beyond
    what the hardcoded templates cover (e.g., a pharma-specific analysis
    even when a generic "customer profitability" template already matched).

    The LLM is told what Tier 1 already suggested, so it avoids duplicates
    and focuses on novel, industry-specific insights.

    Args:
        profiles: Table profiles.
        insights: Existing insights.
        relationships: Discovered relationships (with confidence, column_a, column_b).
        company_context: Company profile dict (industry, products, process_flow).
        max_recommendations: Max recs to return.
    """
    tier1_recs, all_groups, matched_analyses = _recommend_tier1(
        profiles, insights, relationships,
    )

    # Tier 2: LLM suggestions for ALL eligible entity groups
    # Even when templates matched, the LLM can suggest complementary analyses
    if all_groups and company_context:
        try:
            tier2_recs = await _generate_dynamic_cross_table_suggestions(
                all_groups, profiles, company_context, matched_analyses,
            )
            tier1_recs.extend(tier2_recs)
        except Exception:
            logger.exception("Tier 2 dynamic cross-table suggestions failed")

    return _finalize(tier1_recs, max_recommendations)


# ---------------------------------------------------------------------------
# Tier 1 — template-based recommendations (sync)
# ---------------------------------------------------------------------------

def _recommend_tier1(
    profiles: list[dict],
    insights: list[dict],
    relationships: list[dict],
) -> tuple[list[Recommendation], list[EntityGroup], dict[str, list[str]]]:
    """Core Tier 1 recommendation engine.

    Returns (recommendations, all_eligible_groups, matched_analyses):
      - recommendations: all generated Tier 1 recs
      - all_eligible_groups: every EntityGroup with >= 2 tables (for Tier 2)
      - matched_analyses: {entity_type: [matched template titles]} so Tier 2
        knows what Tier 1 already covered and can suggest COMPLEMENTARY analyses
    """
    recommendations: list[Recommendation] = []

    # Index insights by table
    insight_tables: dict[str, int] = {}
    insight_types: dict[str, set[str]] = {}
    for ins in insights:
        for t in (ins.get("source_tables") or []):
            insight_tables[t] = insight_tables.get(t, 0) + 1
            insight_types.setdefault(t, set()).add(ins.get("insight_type", ""))

    # Index relationships
    related_tables: dict[str, set[str]] = {}
    for rel in relationships:
        ta = rel.get("table_a", "")
        tb = rel.get("table_b", "")
        related_tables.setdefault(ta, set()).add(tb)
        related_tables.setdefault(tb, set()).add(ta)

    # --- Single-table recommendations ---
    for profile in profiles:
        table_name = profile.get("table_name", "")
        columns = _get_columns(profile)
        row_count = profile.get("row_count", 0) or 0

        if row_count < 10:
            continue

        existing_types = insight_types.get(table_name, set())
        insight_count = insight_tables.get(table_name, 0)

        # 0. Domain-specific recommendations (highest priority)
        domain = _get_domain(profile)
        if domain in _DOMAIN_RECOMMENDATIONS:
            col_names_lower = {c.lower() for c in columns}
            for rec in _DOMAIN_RECOMMENDATIONS[domain]:
                matched_keywords = sum(
                    1 for kw in rec["keywords"]
                    if any(kw in cn for cn in col_names_lower)
                )
                if matched_keywords >= 2 or (len(rec["keywords"]) <= 2 and matched_keywords >= 1):
                    recommendations.append(Recommendation(
                        title=rec["title"].format(table=table_name),
                        description=rec["description"],
                        analysis_type=rec["type"],
                        target_table=table_name,
                        columns=[],
                        priority=rec["priority"],
                        reason=rec["reason"],
                    ))

        # 1. Time trend
        temporal_cols = [c for c, info in columns.items() if info.get("semantic_type") == "temporal"]
        numeric_cols = [c for c, info in columns.items() if (info.get("semantic_type") or "").startswith("numeric")]

        if temporal_cols and numeric_cols and "trend" not in existing_types:
            recommendations.append(Recommendation(
                title=f"Time trend analysis on {table_name}",
                description=f"Analyze {numeric_cols[0]} trends over {temporal_cols[0]}",
                analysis_type="time_trend",
                target_table=table_name,
                columns=[temporal_cols[0], numeric_cols[0]],
                priority=_time_priority(row_count, insight_count),
                reason="Table has temporal and numeric columns but no trend analysis yet.",
            ))

        # 2. Benchmark
        cat_cols = [c for c, info in columns.items() if info.get("semantic_type") == "categorical"]
        if cat_cols and numeric_cols:
            recommendations.append(Recommendation(
                title=f"Benchmark {numeric_cols[0]} by {cat_cols[0]} in {table_name}",
                description=f"Compare {numeric_cols[0]} across different {cat_cols[0]} groups",
                analysis_type="benchmark",
                target_table=table_name,
                columns=[cat_cols[0], numeric_cols[0]],
                priority=_benchmark_priority(row_count, len(cat_cols), insight_count),
                reason="Table has categorical grouping column for metric comparison.",
            ))

        # 3. Correlation
        if len(numeric_cols) >= 2 and "correlation" not in existing_types:
            recommendations.append(Recommendation(
                title=f"Correlation analysis on {table_name}",
                description=f"Check correlations between {len(numeric_cols)} numeric columns",
                analysis_type="correlation",
                target_table=table_name,
                columns=numeric_cols[:5],
                priority=_correlation_priority(len(numeric_cols), insight_count),
                reason=f"Table has {len(numeric_cols)} numeric columns but no correlation analysis.",
            ))

        # 4. Anomaly scan
        if numeric_cols and "anomaly" not in existing_types:
            recommendations.append(Recommendation(
                title=f"Anomaly scan on {table_name}",
                description=f"Scan {len(numeric_cols)} numeric columns for outliers",
                analysis_type="anomaly",
                target_table=table_name,
                columns=numeric_cols[:5],
                priority=_anomaly_priority(row_count, insight_count),
                reason="No anomaly detection has been run on this table.",
            ))

        # 5. Cohort analysis
        if temporal_cols and cat_cols and numeric_cols:
            recommendations.append(Recommendation(
                title=f"Cohort analysis: {cat_cols[0]} over {temporal_cols[0]} in {table_name}",
                description=f"Track {numeric_cols[0]} for {cat_cols[0]} cohorts over {temporal_cols[0]}",
                analysis_type="cohort",
                target_table=table_name,
                columns=[cat_cols[0], temporal_cols[0], numeric_cols[0]],
                priority=_cohort_priority(row_count, insight_count),
                reason="Table has all three column types needed for cohort tracking.",
            ))

        # 6. Forecast
        if temporal_cols and numeric_cols and row_count >= 20:
            recommendations.append(Recommendation(
                title=f"Forecast {numeric_cols[0]} in {table_name}",
                description=f"Predict future values of {numeric_cols[0]} based on {temporal_cols[0]}",
                analysis_type="forecast",
                target_table=table_name,
                columns=[temporal_cols[0], numeric_cols[0]],
                priority=_forecast_priority(row_count, insight_count),
                reason="Sufficient temporal data for forecasting.",
            ))

    # --- Cross-table intelligence ---
    entity_groups = _build_entity_groups(profiles, relationships)
    all_eligible_groups: list[EntityGroup] = []
    matched_analyses: dict[str, list[str]] = {}  # entity_type → matched template titles

    for entity_type, group in entity_groups.items():
        if len(group.tables) < 2:
            continue

        all_eligible_groups.append(group)
        templates = _CROSS_TABLE_TEMPLATES.get(entity_type, [])

        for template in templates:
            if _check_template_coverage(group, template, profiles):
                table_str = " + ".join(sorted(group.tables)[:6])
                recommendations.append(Recommendation(
                    title=template["title"].format(tables=table_str),
                    description=template["description"],
                    analysis_type="cross_table_intelligence",
                    target_table=group.tables[0],
                    columns=[],
                    priority=template["priority"],
                    reason=template.get("reason", "Cross-table analysis").format(
                        tables=table_str, entity=entity_type,
                    ),
                ))
                matched_analyses.setdefault(entity_type, []).append(
                    template["title"].replace("{tables}", ", ".join(sorted(group.tables)[:4]))
                )

    # Fallback: generic cross-table for related tables with few insights
    for table_name, related in related_tables.items():
        if len(related) >= 2 and insight_tables.get(table_name, 0) < 3:
            recommendations.append(Recommendation(
                title=f"Cross-table analysis for {table_name}",
                description=f"Explore relationships between {table_name} and {', '.join(list(related)[:3])}",
                analysis_type="correlation",
                target_table=table_name,
                columns=[],
                priority=60,
                reason=f"Table has {len(related)} relationships but few insights.",
            ))

    return recommendations, all_eligible_groups, matched_analyses


def _finalize(recs: list[Recommendation], max_recs: int) -> list[Recommendation]:
    """Sort by priority, deduplicate by table+type, cap at max."""
    seen: set[tuple[str, str]] = set()
    unique: list[Recommendation] = []
    for r in sorted(recs, key=lambda x: -x.priority):
        key = (r.target_table, r.analysis_type)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:max_recs]


# ---------------------------------------------------------------------------
# Entity group building
# ---------------------------------------------------------------------------

def _build_entity_groups(
    profiles: list[dict],
    relationships: list[dict],
) -> dict[str, EntityGroup]:
    """Build groups of tables sharing each entity type.

    A table belongs to an entity group if:
      A. It has columns matching entity keywords (e.g., "buyer_name" → customer)
      B. It has a discovered relationship to a table in the group
         (even if the column name doesn't match any keyword)
      C. Both are inferred from relationship columns
    """
    groups: dict[str, EntityGroup] = {}

    # A. Column keyword scan
    for profile in profiles:
        tn = profile.get("table_name", "")
        cols = _get_columns(profile)
        for c, info in cols.items():
            cl = c.lower()
            for entity_type, keywords in _ENTITY_KEYWORDS.items():
                if any(kw in cl for kw in keywords):
                    if entity_type not in groups:
                        groups[entity_type] = EntityGroup(entity_type=entity_type)
                    g = groups[entity_type]
                    if tn not in g.tables:
                        g.tables.append(tn)
                    g.entity_columns.setdefault(tn, []).append(c)
                    break  # one entity type per column

    # B. Infer entity types from discovered relationships
    for rel in relationships:
        ta = rel.get("table_a", "")
        tb = rel.get("table_b", "")
        ca = rel.get("column_a", "")
        cb = rel.get("column_b", "")
        confidence = rel.get("confidence", 0)

        # Accept all relationships if confidence not provided (legacy api.py data)
        if confidence and confidence < 0.3:
            continue
        if not ta or not tb:
            continue

        # Try to infer entity type from column names
        inferred = (
            _infer_entity_type_from_column(ca)
            or _infer_entity_type_from_column(cb)
        )
        if not inferred:
            # Fallback: if both are categorical, assume "customer"
            cols_a = _get_columns(next((p for p in profiles if p.get("table_name") == ta), {}))
            cols_b = _get_columns(next((p for p in profiles if p.get("table_name") == tb), {}))
            if (cols_a.get(ca, {}).get("semantic_type") == "categorical"
                    and cols_b.get(cb, {}).get("semantic_type") == "categorical"):
                inferred = "customer"

        if inferred:
            if inferred not in groups:
                groups[inferred] = EntityGroup(entity_type=inferred)
            g = groups[inferred]
            for tbl, col in [(ta, ca), (tb, cb)]:
                if tbl not in g.tables:
                    g.tables.append(tbl)
                if col:
                    g.entity_columns.setdefault(tbl, []).append(col)

    # C. Expand groups via relationship links — if table A is in a group
    # and table B has a relationship with A, add B to the group too
    linked: dict[str, set[str]] = {}
    for rel in relationships:
        ta = rel.get("table_a", "")
        tb = rel.get("table_b", "")
        confidence = rel.get("confidence", 0)
        if confidence and confidence < 0.3:
            continue
        if ta and tb:
            linked.setdefault(ta, set()).add(tb)
            linked.setdefault(tb, set()).add(ta)

    for g in groups.values():
        expanded = set(g.tables)
        for t in list(g.tables):
            expanded.update(linked.get(t, set()))
        for t in expanded:
            if t not in g.tables:
                g.tables.append(t)

    return groups


# ---------------------------------------------------------------------------
# Template coverage check (entity-group-based — NOT per-table assignment)
# ---------------------------------------------------------------------------

def _check_template_coverage(
    group: EntityGroup,
    template: dict,
    profiles: list[dict],
) -> bool:
    """Check if the COMBINED data across ALL tables in the group covers the template.

    Key difference from the old greedy approach: a single table CAN satisfy
    multiple requirements. What matters is:
      1. At least 2 tables share the entity (the group already guarantees this)
      2. The UNION of data across all tables covers all required_data entries

    A requirement is covered if ANY table in the group matches it
    (via table name hints OR column keywords).
    """
    required_data = template.get("required_data", [])
    if not required_data:
        return True

    # Pre-compute table name hints and column sets
    table_info: dict[str, tuple[set[str], set[str]]] = {}  # table → (name_hints, col_lower)
    for tn in group.tables:
        hints = set(re.split(r"[_\s\-]+", tn.lower()))
        prof = next((p for p in profiles if p.get("table_name") == tn), {})
        cols = _get_columns(prof)
        col_lower = {c.lower() for c in cols}
        table_info[tn] = (hints, col_lower)

    for req in required_data:
        table_kws = req.get("table_keywords", [])
        col_kws = req.get("col_keywords", req.get("keywords", []))

        req_covered = False
        for tn in group.tables:
            hints, col_lower = table_info.get(tn, (set(), set()))

            # Table name match
            name_match = any(
                kw in hints or any(kw in h for h in hints)
                for kw in table_kws
            )

            # Column name match
            col_match = any(
                any(kw in cn for cn in col_lower)
                for kw in col_kws
            )

            if name_match or col_match:
                req_covered = True
                break

        if not req_covered:
            return False

    return True


# ---------------------------------------------------------------------------
# Tier 2 — LLM-generated cross-table suggestions
# ---------------------------------------------------------------------------

async def _generate_dynamic_cross_table_suggestions(
    entity_groups: list[EntityGroup],
    profiles: list[dict],
    company_context: dict,
    matched_analyses: dict[str, list[str]] | None = None,
    max_groups: int = 5,
) -> list[Recommendation]:
    """Tier 2: Use LLM to generate cross-table analysis suggestions.

    Runs for ALL entity groups (not just unmatched ones). When templates
    already matched, the LLM is told what's already covered so it suggests
    COMPLEMENTARY analyses — novel, industry-specific insights that the
    hardcoded templates can't anticipate.
    """
    try:
        from google import genai
        from config.settings import settings
    except ImportError:
        return []

    if not settings.gemini_api_key:
        return []

    recommendations: list[Recommendation] = []
    # Prioritize larger groups (more tables = more combination value)
    sorted_groups = sorted(entity_groups, key=lambda g: len(g.tables), reverse=True)[:max_groups]

    for group in sorted_groups:
        # Check cache
        cache_key = _cache_key(group, company_context)
        cached = _dynamic_suggestion_cache.get(cache_key)
        if cached:
            ts, cached_recs = cached
            if time.time() - ts < _CACHE_TTL:
                recommendations.extend(cached_recs)
                continue

        # Build prompt and call LLM
        existing = (matched_analyses or {}).get(group.entity_type, [])
        prompt = _build_cross_table_prompt(group, profiles, company_context, existing)

        try:
            client = genai.Client(api_key=settings.gemini_api_key)
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
            )
            group_recs = _parse_cross_table_suggestions(response.text or "", group)
            _dynamic_suggestion_cache[cache_key] = (time.time(), group_recs)
            recommendations.extend(group_recs)
        except Exception:
            logger.exception(
                "LLM cross-table suggestion failed for %s group (%s)",
                group.entity_type, group.tables,
            )

    return recommendations


def _cache_key(group: EntityGroup, company_context: dict) -> str:
    """Cache key for dynamic suggestions — keyed on tables + industry."""
    raw = f"{sorted(group.tables)}:{group.entity_type}:{company_context.get('industry', '')}"
    return hashlib.md5(raw.encode()).hexdigest()


def _build_cross_table_prompt(
    group: EntityGroup,
    profiles: list[dict],
    company_context: dict,
    existing_analyses: list[str] | None = None,
) -> str:
    """Build LLM prompt for cross-table analysis suggestions.

    Args:
        existing_analyses: Template-based analyses already generated by Tier 1.
            When provided, the LLM is asked for COMPLEMENTARY suggestions that
            go beyond what templates cover.
    """
    # Table summaries
    table_parts = []
    for tn in group.tables:
        prof = next((p for p in profiles if p.get("table_name") == tn), {})
        cols = _get_columns(prof)
        col_lines = []
        for cn, info in cols.items():
            sem = info.get("semantic_type", "unknown")
            samples = info.get("sample_values", [])[:3]
            line = f"    {cn} ({sem})"
            if samples:
                line += f": {samples}"
            col_lines.append(line)
        entity_cols = group.entity_columns.get(tn, [])
        entity_note = f" [shared entity column: {', '.join(entity_cols)}]" if entity_cols else ""
        table_parts.append(
            f"  Table: {tn} ({prof.get('row_count', '?')} rows){entity_note}\n"
            + "\n".join(col_lines)
        )

    tables_text = "\n\n".join(table_parts)
    industry = company_context.get("industry", "business")
    products = company_context.get("products", "")
    process_flow = company_context.get("process_flow", "")

    # Build the "already covered" section so the LLM avoids duplicates
    already_covered_section = ""
    if existing_analyses:
        covered_list = "\n".join(f"  - {a}" for a in existing_analyses)
        already_covered_section = (
            f"\nThese cross-table analyses have ALREADY been generated by templates:\n"
            f"{covered_list}\n\n"
            "Do NOT repeat these. Instead, suggest COMPLEMENTARY analyses that "
            "go beyond what these templates cover — think industry-specific, "
            "operational, or strategic insights that only a domain expert would suggest.\n"
        )

    return (
        f"You are a data analyst for a {industry} company"
        f"{' that makes ' + str(products) if products else ''}.\n"
        f"{'Process flow: ' + process_flow if process_flow else ''}\n\n"
        f"These tables share a common entity: '{group.entity_type}' "
        f"(linking columns: {group.entity_columns})\n\n"
        f"Tables:\n{tables_text}\n"
        f"{already_covered_section}\n"
        "Suggest 1-3 cross-table analyses that would provide ACTIONABLE business insights "
        "by combining data from 2+ of these tables.\n\n"
        "For each suggestion, output EXACTLY this format (one per line):\n"
        "SUGGESTION: <title> | <description of what the combined analysis reveals> | <priority 70-95> | <reason why this matters>\n\n"
        "Rules:\n"
        "- Focus on analyses that REQUIRE multiple tables (not single-table)\n"
        f"- Be specific to {industry} industry and these column types\n"
        "- Priority: 90-95 = critical operational insight, 80-89 = high value, 70-79 = nice-to-have\n"
        "- Description: explain what SPECIFIC business questions this answers\n"
        "- If these tables don't support meaningful cross-table analysis, output NO_SUGGESTIONS\n"
    )


def _parse_cross_table_suggestions(
    llm_response: str,
    group: EntityGroup,
) -> list[Recommendation]:
    """Parse SUGGESTION: lines from LLM response into Recommendations."""
    if not llm_response or "NO_SUGGESTIONS" in llm_response:
        return []

    recommendations: list[Recommendation] = []
    table_str = " + ".join(sorted(group.tables)[:4])

    for line in llm_response.split("\n"):
        line = line.strip()
        if not line.upper().startswith("SUGGESTION:"):
            continue

        content = line[len("SUGGESTION:"):].strip()
        parts = [p.strip() for p in content.split("|")]
        if len(parts) < 3:
            continue

        title = parts[0]
        description = parts[1]
        reason = parts[3] if len(parts) >= 4 else f"AI-identified pattern across {table_str}"

        try:
            priority = min(max(int(parts[2]), 65), 92)
        except (ValueError, IndexError):
            priority = 75

        if not title:
            continue

        recommendations.append(Recommendation(
            title=f"{title} across {table_str}",
            description=description,
            analysis_type="cross_table_intelligence",
            target_table=group.tables[0],
            columns=[],
            priority=priority,
            reason=f"[AI-suggested] {reason}",
        ))

    return recommendations


# ---------------------------------------------------------------------------
# Public coverage API
# ---------------------------------------------------------------------------

def compute_coverage(
    profiles: list[dict],
    insights: list[dict],
) -> dict:
    """Compute analysis coverage metrics."""
    all_tables = {p.get("table_name", "") for p in profiles}
    covered_tables = set()
    for ins in insights:
        for t in (ins.get("source_tables") or []):
            covered_tables.add(t)

    uncovered = all_tables - covered_tables
    coverage_pct = len(covered_tables) / len(all_tables) * 100 if all_tables else 0

    return {
        "total_tables": len(all_tables),
        "covered_tables": len(covered_tables),
        "uncovered_tables": sorted(uncovered),
        "coverage_pct": round(coverage_pct, 1),
        "total_insights": len(insights),
        "avg_insights_per_table": round(len(insights) / len(all_tables), 1) if all_tables else 0,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_columns(profile: dict) -> dict:
    """Extract columns from profile."""
    cls = profile.get("column_classification")
    if cls and isinstance(cls, dict) and "columns" in cls:
        return cls["columns"]
    return {}


def _get_domain(profile: dict) -> str:
    """Extract domain hint from profile."""
    domain = profile.get("domain_hint", "")
    if domain:
        return domain.lower()
    cls = profile.get("column_classification")
    if cls and isinstance(cls, dict):
        return (cls.get("domain_hint") or "general").lower()
    return "general"


def _infer_entity_type_from_column(col_name: str) -> str | None:
    """Try to infer which entity type a column represents from its name."""
    if not col_name:
        return None
    cl = col_name.lower()
    for entity_type, keywords in _ENTITY_KEYWORDS.items():
        if any(kw in cl for kw in keywords):
            return entity_type
    return None


def _time_priority(row_count: int, insight_count: int) -> int:
    base = 80
    if insight_count > 5:
        base -= 20
    if row_count > 100:
        base += 5
    return min(max(base, 10), 100)


def _benchmark_priority(row_count: int, cat_count: int, insight_count: int) -> int:
    base = 70
    if cat_count > 2:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)


def _correlation_priority(num_col_count: int, insight_count: int) -> int:
    base = 65
    if num_col_count > 4:
        base += 15
    if insight_count > 3:
        base -= 10
    return min(max(base, 10), 100)


def _anomaly_priority(row_count: int, insight_count: int) -> int:
    base = 75
    if row_count > 200:
        base += 5
    if insight_count > 5:
        base -= 20
    return min(max(base, 10), 100)


def _cohort_priority(row_count: int, insight_count: int) -> int:
    base = 60
    if row_count > 100:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)


def _forecast_priority(row_count: int, insight_count: int) -> int:
    base = 55
    if row_count > 50:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)
