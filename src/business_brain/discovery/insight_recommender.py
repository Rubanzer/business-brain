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

from business_brain.discovery.insight_quality_gate import (
    _COLUMN_LABELS,
    _humanize_column,
    _humanize_table,
)

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
    # Pre-computed backing (populated by _enrich_with_precomputed)
    precomputed_id: str | None = None
    precomputed_summary: dict | None = None
    confidence: str = "heuristic"  # "heuristic" | "pre-computed"


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
    precomputed: list[dict] | None = None,
    reinforcement_weights=None,
) -> list[Recommendation]:
    """Generate analysis recommendations (sync — Tier 1 only).

    Backward-compatible sync entry point. For full Tier 1 + Tier 2 (LLM),
    use ``recommend_analyses_async()`` instead.

    Args:
        precomputed: Optional list of pre-computed analysis dicts from
            PrecomputedAnalysis table. When provided, matching recs get
            enriched with real data (confidence="pre-computed", preview).
        reinforcement_weights: Optional ReinforcementWeights record. When
            provided, base priorities are modulated by learned multipliers.
    """
    recs, _groups, _matched = _recommend_tier1(
        profiles, insights, relationships,
        reinforcement_weights=reinforcement_weights,
    )
    if precomputed:
        recs = _enrich_with_precomputed(recs, precomputed)
    return _finalize(recs, max_recommendations)


async def recommend_analyses_async(
    profiles: list[dict],
    insights: list[dict],
    relationships: list[dict],
    company_context: dict | None = None,
    max_recommendations: int = 10,
    precomputed: list[dict] | None = None,
    reinforcement_weights=None,
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
        precomputed: Optional list of pre-computed analysis dicts. When provided,
            matching recs get enriched with real data (confidence, preview).
        reinforcement_weights: Optional ReinforcementWeights record. When
            provided, base priorities are modulated by learned multipliers.
    """
    tier1_recs, all_groups, matched_analyses = _recommend_tier1(
        profiles, insights, relationships,
        reinforcement_weights=reinforcement_weights,
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

    # Enrich with pre-computed results (attach real data to heuristic recs)
    if precomputed:
        tier1_recs = _enrich_with_precomputed(tier1_recs, precomputed)

    return _finalize(tier1_recs, max_recommendations)


# ---------------------------------------------------------------------------
# Tier 1 — template-based recommendations (sync)
# ---------------------------------------------------------------------------

def _recommend_tier1(
    profiles: list[dict],
    insights: list[dict],
    relationships: list[dict],
    reinforcement_weights=None,
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
        tbl_label = _table_label(table_name, profile)
        if domain in _DOMAIN_RECOMMENDATIONS:
            col_names_lower = {c.lower() for c in columns}
            for rec in _DOMAIN_RECOMMENDATIONS[domain]:
                matched_keywords = sum(
                    1 for kw in rec["keywords"]
                    if any(kw in cn for cn in col_names_lower)
                )
                if matched_keywords >= 2 or (len(rec["keywords"]) <= 2 and matched_keywords >= 1):
                    recommendations.append(Recommendation(
                        title=rec["title"].format(table=tbl_label),
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
            # Rank numeric columns by trend suitability (domain-aware, CV-based)
            num_subset = {c: columns[c] for c in numeric_cols}
            ranked_nums = _rank_columns(num_subset, row_count, profile, _score_numeric_for_trend, domain=domain)
            trend_num = ranked_nums[0] if ranked_nums else numeric_cols[0]
            trend_cols = [temporal_cols[0], trend_num]
            num_label = _col_label(trend_num, profile)
            time_label = _col_label(temporal_cols[0], profile)
            _time_mult = _get_rl_mult(reinforcement_weights, "time_trend")
            base_pri = _time_priority(row_count, insight_count, multiplier=_time_mult)
            priority = min(max(base_pri + _data_fitness_adjustment("time_trend", profile, trend_cols), 10), 100)
            recommendations.append(Recommendation(
                title=f"Track {num_label} over {time_label} in {tbl_label}",
                description=f"Analyze how {num_label} changes over {time_label} — spot trends, seasonality, and shifts",
                analysis_type="time_trend",
                target_table=table_name,
                columns=trend_cols,
                priority=priority,
                reason=_build_reason("time_trend", profile, trend_cols, domain),
            ))

        # 2. Benchmark
        cat_cols = [c for c, info in columns.items() if info.get("semantic_type") == "categorical"]
        if cat_cols and numeric_cols:
            # Rank categoricals and numerics by benchmark suitability
            cat_subset = {c: columns[c] for c in cat_cols}
            num_subset = {c: columns[c] for c in numeric_cols}
            ranked_cats = _rank_columns(cat_subset, row_count, profile, _score_categorical_for_benchmark)
            ranked_nums = _rank_columns(num_subset, row_count, profile, _score_numeric_for_benchmark)
            bench_cat = ranked_cats[0] if ranked_cats else cat_cols[0]
            bench_num = ranked_nums[0] if ranked_nums else numeric_cols[0]
            # Suppress if both featured columns are unnamed artifacts
            if not (_is_unnamed_column(bench_cat, profile) and _is_unnamed_column(bench_num, profile)):
                cat_label = _col_label(bench_cat, profile)
                num_label = _col_label(bench_num, profile)
                bench_cols = [bench_cat, bench_num]
                _bench_mult = _get_rl_mult(reinforcement_weights, "benchmark")
                base_pri = _benchmark_priority(row_count, len(cat_cols), insight_count, multiplier=_bench_mult)
                priority = min(max(base_pri + _data_fitness_adjustment("benchmark", profile, bench_cols), 10), 100)
                recommendations.append(Recommendation(
                    title=f"Compare {num_label} by {cat_label} in {tbl_label}",
                    description=f"Benchmark {num_label} across different {cat_label} groups — find best and worst performers",
                    analysis_type="benchmark",
                    target_table=table_name,
                    columns=bench_cols,
                    priority=priority,
                    reason=_build_reason("benchmark", profile, bench_cols, domain),
                ))

        # 3. Correlation
        if len(numeric_cols) >= 2 and "correlation" not in existing_types:
            # Rank by correlation suitability (high variance = more signal)
            num_subset = {c: columns[c] for c in numeric_cols}
            ranked_nums = _rank_columns(num_subset, row_count, profile, _score_numeric_for_correlation)
            corr_cols = ranked_nums[:5]
            corr_labels = [_col_label(c, profile) for c in corr_cols[:3]]
            _corr_mult = _get_rl_mult(reinforcement_weights, "correlation")
            base_pri = _correlation_priority(len(numeric_cols), insight_count, multiplier=_corr_mult)
            priority = min(max(base_pri + _data_fitness_adjustment("correlation", profile, corr_cols), 10), 100)
            recommendations.append(Recommendation(
                title=f"Find hidden relationships in {tbl_label}",
                description=f"Check correlations between {', '.join(corr_labels)} and {len(numeric_cols) - len(corr_labels)} more columns",
                analysis_type="correlation",
                target_table=table_name,
                columns=corr_cols,
                priority=priority,
                reason=_build_reason("correlation", profile, corr_cols, domain),
            ))

        # 4. Anomaly scan
        if numeric_cols and "anomaly" not in existing_types:
            # Rank by anomaly suitability (high CV, named, non-constant)
            num_subset = {c: columns[c] for c in numeric_cols}
            ranked_nums = _rank_columns(num_subset, row_count, profile, _score_numeric_for_anomaly)
            anomaly_cols = ranked_nums[:5]
            # Fallback if all scored 0 (all constant)
            if not anomaly_cols:
                anomaly_cols = numeric_cols[:5]
            headline_col = anomaly_cols[0]
            headline_label = _col_label(headline_col, profile)
            _anom_mult = _get_rl_mult(reinforcement_weights, "anomaly")
            base_pri = _anomaly_priority(row_count, insight_count, multiplier=_anom_mult)
            priority = min(max(base_pri + _data_fitness_adjustment("anomaly", profile, anomaly_cols), 10), 100)
            recommendations.append(Recommendation(
                title=f"Detect outliers in {headline_label} ({tbl_label})",
                description=f"Scan {len(numeric_cols)} numeric columns for anomalies — flag unusual values that may indicate data issues or opportunities",
                analysis_type="anomaly",
                target_table=table_name,
                columns=anomaly_cols,
                priority=priority,
                reason=_build_reason("anomaly", profile, anomaly_cols, domain),
            ))

        # 5. Cohort analysis
        if temporal_cols and cat_cols and numeric_cols:
            cat_subset = {c: columns[c] for c in cat_cols}
            num_subset = {c: columns[c] for c in numeric_cols}
            ranked_cats = _rank_columns(cat_subset, row_count, profile, _score_categorical_for_benchmark)
            ranked_nums = _rank_columns(num_subset, row_count, profile, _score_numeric_for_benchmark)
            cohort_cat = ranked_cats[0] if ranked_cats else cat_cols[0]
            cohort_num = ranked_nums[0] if ranked_nums else numeric_cols[0]
            if not (_is_unnamed_column(cohort_cat, profile) and _is_unnamed_column(cohort_num, profile)):
                cat_label = _col_label(cohort_cat, profile)
                num_label = _col_label(cohort_num, profile)
                time_label = _col_label(temporal_cols[0], profile)
                cohort_cols = [cohort_cat, temporal_cols[0], cohort_num]
                _cohort_mult = _get_rl_mult(reinforcement_weights, "cohort")
                base_pri = _cohort_priority(row_count, insight_count, multiplier=_cohort_mult)
                priority = min(max(base_pri + _data_fitness_adjustment("cohort", profile, cohort_cols), 10), 100)
                recommendations.append(Recommendation(
                    title=f"Track {num_label} by {cat_label} over {time_label} in {tbl_label}",
                    description=f"Cohort analysis — how does {num_label} evolve for each {cat_label} group over {time_label}?",
                    analysis_type="cohort",
                    target_table=table_name,
                    columns=cohort_cols,
                    priority=priority,
                    reason=_build_reason("cohort", profile, cohort_cols, domain),
                ))

        # 6. Forecast
        if temporal_cols and numeric_cols and row_count >= 20:
            num_subset = {c: columns[c] for c in numeric_cols}
            ranked_nums = _rank_columns(num_subset, row_count, profile, _score_numeric_for_trend, domain=domain)
            fc_num = ranked_nums[0] if ranked_nums else numeric_cols[0]
            num_label = _col_label(fc_num, profile)
            time_label = _col_label(temporal_cols[0], profile)
            fc_cols = [temporal_cols[0], fc_num]
            _fc_mult = _get_rl_mult(reinforcement_weights, "forecast")
            base_pri = _forecast_priority(row_count, insight_count, multiplier=_fc_mult)
            priority = min(max(base_pri + _data_fitness_adjustment("forecast", profile, fc_cols), 10), 100)
            recommendations.append(Recommendation(
                title=f"Forecast {num_label} in {tbl_label}",
                description=f"Predict future {num_label} based on {time_label} — plan ahead with data-driven projections",
                analysis_type="forecast",
                target_table=table_name,
                columns=fc_cols,
                priority=priority,
                reason=_build_reason("forecast", profile, fc_cols, domain),
            ))

        # 7. Insight-driven follow-up recommendations
        table_insights = [
            ins for ins in insights
            if table_name in (ins.get("source_tables") or [])
        ]
        if table_insights:
            followup_recs = _recommend_insight_followups(
                profile, table_insights, columns, domain,
            )
            recommendations.extend(followup_recs)

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
                # Humanize table names in cross-table titles
                table_labels = []
                for tn in sorted(group.tables)[:6]:
                    prof = next((p for p in profiles if p.get("table_name") == tn), {})
                    table_labels.append(_table_label(tn, prof))
                table_str_human = " + ".join(table_labels)
                table_str_raw = " + ".join(sorted(group.tables)[:6])
                recommendations.append(Recommendation(
                    title=template["title"].format(tables=table_str_human),
                    description=template["description"],
                    analysis_type="cross_table_intelligence",
                    target_table=group.tables[0],
                    columns=[],
                    priority=template["priority"],
                    reason=template.get("reason", "Cross-table analysis").format(
                        tables=table_str_human, entity=entity_type,
                    ),
                ))
                matched_analyses.setdefault(entity_type, []).append(
                    template["title"].replace("{tables}", ", ".join(sorted(group.tables)[:4]))
                )

    # Fallback: generic cross-table for related tables with few insights
    for table_name, related in related_tables.items():
        if len(related) >= 2 and insight_tables.get(table_name, 0) < 3:
            prof = next((p for p in profiles if p.get("table_name") == table_name), {})
            fb_tbl_label = _table_label(table_name, prof)
            related_labels = []
            for rtn in list(related)[:3]:
                rprof = next((p for p in profiles if p.get("table_name") == rtn), {})
                related_labels.append(_table_label(rtn, rprof))
            recommendations.append(Recommendation(
                title=f"Cross-table analysis for {fb_tbl_label}",
                description=f"Explore relationships between {fb_tbl_label} and {', '.join(related_labels)}",
                analysis_type="correlation",
                target_table=table_name,
                columns=[],
                priority=60,
                reason=f"Table has {len(related)} relationships but few insights.",
            ))

    return recommendations, all_eligible_groups, matched_analyses


# ---------------------------------------------------------------------------
# Pre-computed result enrichment
# ---------------------------------------------------------------------------

# Mapping from recommendation analysis_type → precomputed analysis_type.
# The recommender uses "time_trend" while precompute uses "trend", etc.
_REC_TYPE_TO_PRECOMPUTE_TYPE: dict[str, str] = {
    "benchmark": "benchmark",
    "correlation": "correlation",
    "anomaly": "anomaly",
    "time_trend": "trend",
    "forecast": "trend",
    "cohort": "benchmark",  # cohort recs can match benchmark pre-computation
}


def _enrich_with_precomputed(
    recommendations: list[Recommendation],
    precomputed: list[dict],
) -> list[Recommendation]:
    """Match recommendations to pre-computed results by (table, type, columns).

    When a recommendation matches a completed pre-computation:
      - confidence → "pre-computed"
      - precomputed_summary → result_summary from DB
      - precomputed_id → record ID
      - priority adjusted: quality_score > 0.5 → +15, < 0.2 → -10

    When matched but status = "failed":
      - priority -= 20 (analysis errored on real data)
    """
    if not precomputed:
        return recommendations

    # Build lookup index: (table, precompute_type, frozenset(columns)) → dict
    pc_index: dict[tuple, dict] = {}
    for pc in precomputed:
        key = (
            pc.get("table_name", ""),
            pc.get("analysis_type", ""),
            frozenset(pc.get("columns", [])),
        )
        # Keep the highest-quality match if duplicates exist
        existing = pc_index.get(key)
        if not existing or (pc.get("quality_score", 0) or 0) > (existing.get("quality_score", 0) or 0):
            pc_index[key] = pc

    for rec in recommendations:
        precompute_type = _REC_TYPE_TO_PRECOMPUTE_TYPE.get(rec.analysis_type)
        if not precompute_type:
            continue

        key = (rec.target_table, precompute_type, frozenset(rec.columns))
        pc = pc_index.get(key)
        if not pc:
            continue

        status = pc.get("status", "")

        if status == "completed":
            rec.confidence = "pre-computed"
            rec.precomputed_summary = pc.get("result_summary")
            rec.precomputed_id = pc.get("precomputed_id")

            quality = pc.get("quality_score", 0) or 0
            if quality > 0.5:
                rec.priority = min(100, rec.priority + 15)
            elif quality < 0.2:
                rec.priority = max(10, rec.priority - 10)

        elif status == "failed":
            rec.priority = max(10, rec.priority - 20)

    return recommendations


def _finalize(recs: list[Recommendation], max_recs: int) -> list[Recommendation]:
    """Sort by priority, deduplicate by table+type, cap at max.

    Applies a diversity pass: no single table dominates the top results.
    Each table gets a fair share of slots, then remaining slots fill by priority.
    """
    seen: set[tuple[str, str]] = set()
    unique: list[Recommendation] = []
    for r in sorted(recs, key=lambda x: -x.priority):
        key = (r.target_table, r.analysis_type)
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # Diversity pass: ensure no single table dominates
    if len(unique) > max_recs:
        distinct_tables = {r.target_table for r in unique}
        max_per_table = max(2, max_recs // max(len(distinct_tables), 1) + 1)

        table_counts: dict[str, int] = {}
        final: list[Recommendation] = []
        overflow: list[Recommendation] = []

        for r in unique:
            count = table_counts.get(r.target_table, 0)
            if count < max_per_table:
                final.append(r)
                table_counts[r.target_table] = count + 1
            else:
                overflow.append(r)

        # Fill remaining slots from overflow
        remaining = max_recs - len(final)
        if remaining > 0:
            final.extend(overflow[:remaining])

        return final[:max_recs]

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


def _col_label(col_name: str, profile: dict) -> str:
    """Best available human-readable label for a column.

    Priority: (1) Gemini description first sentence (≤40 chars),
              (2) _COLUMN_LABELS explicit mapping,
              (3) _humanize_column() auto-generation.
    """
    descs = profile.get("column_descriptions", {})
    desc = descs.get(col_name, "")
    if desc:
        first_sentence = desc.split(".")[0].strip()
        if 3 <= len(first_sentence) <= 40:
            return first_sentence

    if col_name.lower() in _COLUMN_LABELS:
        return _COLUMN_LABELS[col_name.lower()]

    return _humanize_column(col_name)


def _table_label(table_name: str, profile: dict) -> str:
    """Best available human-readable label for a table.

    Priority: (1) table_description first sentence (≤50 chars),
              (2) _humanize_table().
    """
    desc = profile.get("table_description", "")
    if desc:
        first_sentence = desc.split(".")[0].strip()
        if 3 <= len(first_sentence) <= 50:
            return first_sentence
    return _humanize_table(table_name)


def _is_unnamed_column(col_name: str, profile: dict) -> bool:
    """True if the column is a spreadsheet artifact like col_1, col_3.

    These produce useless recommendation titles and are suppressed.
    Columns with a Gemini description are allowed through.
    """
    if re.match(r"^col_?\d+$", col_name.lower()):
        descs = profile.get("column_descriptions", {})
        return col_name not in descs or not descs[col_name]
    return False


# ---------------------------------------------------------------------------
# Domain-flavored vocabulary for reason builder
# ---------------------------------------------------------------------------

_DOMAIN_VOCABULARY: dict[str, dict[str, str]] = {
    "manufacturing": {
        "benchmark": "benchmarking reveals best and worst performers across production",
        "anomaly": "outlier detection can flag process deviations or equipment issues",
        "correlation": "finding hidden relationships can reveal cost drivers or yield levers",
        "trend": "spot production trends, seasonality, and shift patterns",
        "cohort": "reveals which groups are improving or declining over time",
        "forecast": "predict future output to plan capacity and resources",
    },
    "energy": {
        "benchmark": "comparison identifies high and low consumption units",
        "anomaly": "detect power spikes or inefficient consumption patterns",
        "correlation": "understand what drives energy cost per unit of output",
        "trend": "track consumption patterns and peak demand periods",
        "cohort": "compare energy efficiency across equipment groups over time",
        "forecast": "predict future energy demand for load planning",
    },
    "procurement": {
        "benchmark": "comparison reveals which suppliers deliver the best value",
        "anomaly": "detect pricing anomalies or unusual purchase patterns",
        "correlation": "understand price-quality-delivery relationships",
        "trend": "track rate changes and purchasing patterns over time",
        "cohort": "compare supplier performance trajectories",
        "forecast": "predict future procurement costs and quantities",
    },
    "quality": {
        "benchmark": "comparison shows which entities have best and worst quality",
        "anomaly": "flag sudden quality drops or unusual defect patterns",
        "correlation": "link quality metrics to process parameters",
        "trend": "track defect rate progression and quality improvements",
        "cohort": "compare quality across product lines or batches over time",
        "forecast": "predict quality outcomes for proactive intervention",
    },
    "logistics": {
        "benchmark": "reveals fastest and slowest dispatch routes or carriers",
        "anomaly": "detect delays, bottlenecks, or unusual traffic patterns",
        "correlation": "understand what drives turnaround time and throughput",
        "trend": "track dispatch volumes and delivery performance over time",
        "cohort": "compare carrier or route performance trajectories",
        "forecast": "predict future dispatch volumes for capacity planning",
    },
}


def _domain_phrase(domain: str, rec_type: str) -> str:
    """Get domain-specific trailing phrase for a reason."""
    vocab = _DOMAIN_VOCABULARY.get(domain, {})
    return vocab.get(rec_type, {
        "benchmark": "comparison reveals performance differences across groups",
        "anomaly": "detect unusual values that warrant investigation",
        "correlation": "discover hidden relationships between measures",
        "trend": "spot patterns and inflection points over time",
        "cohort": "track how groups evolve over time",
        "forecast": "predict future values from historical patterns",
    }.get(rec_type, "analysis recommended based on data characteristics"))


def _build_reason(
    rec_type: str,
    profile: dict,
    columns: list[str],
    domain: str,
) -> str:
    """Generate a business-relevant reason from actual data characteristics.

    Uses column stats, sample values, cardinality, and domain vocabulary to
    produce specific reasons instead of generic system-status messages.
    """
    cls = _get_columns(profile)
    parts: list[str] = []

    if rec_type == "benchmark" and len(columns) >= 2:
        cat_col, num_col = columns[0], columns[1]
        cat_info = cls.get(cat_col, {})
        num_info = cls.get(num_col, {})
        stats = num_info.get("stats", {})
        cardinality = cat_info.get("cardinality", 0)
        samples = cat_info.get("sample_values", [])[:3]
        num_label = _col_label(num_col, profile)

        if stats and stats.get("min") is not None and stats.get("max") is not None:
            parts.append(f"{num_label} ranges from {stats['min']:g} to {stats['max']:g}")
        if cardinality:
            cat_label = _col_label(cat_col, profile)
            parts.append(f"across {cardinality} {cat_label} groups")
        if samples:
            sample_strs = [str(s).replace("_", " ").title() for s in samples[:3]]
            parts.append(f"(e.g., {', '.join(sample_strs)})")

    elif rec_type == "correlation" and columns:
        col_labels = [_col_label(c, profile) for c in columns[:3]]
        parts.append(f"Columns like {', '.join(col_labels)} may be interdependent")

    elif rec_type == "anomaly" and columns:
        col = columns[0]
        stats = cls.get(col, {}).get("stats", {})
        label = _col_label(col, profile)
        if stats and stats.get("stdev") is not None:
            parts.append(f"{label} has high variability (stdev {stats['stdev']:g})")
        else:
            parts.append(f"Scan {label} for unusual values")

    elif rec_type == "time_trend" and len(columns) >= 2:
        num_label = _col_label(columns[1], profile)
        time_label = _col_label(columns[0], profile)
        row_count = profile.get("row_count", 0)
        parts.append(f"Track how {num_label} changes over {time_label}")
        if row_count and row_count > 50:
            parts.append(f"({row_count} data points)")

    elif rec_type == "cohort" and len(columns) >= 3:
        cat_label = _col_label(columns[0], profile)
        num_label = _col_label(columns[2], profile)
        parts.append(f"Track {num_label} for each {cat_label} group over time")

    elif rec_type == "forecast" and len(columns) >= 2:
        num_label = _col_label(columns[1], profile)
        row_count = profile.get("row_count", 0)
        parts.append(f"Predict future {num_label}")
        if row_count:
            parts.append(f"from {row_count} historical data points")

    # Append domain-specific trailing phrase
    parts.append(f"— {_domain_phrase(domain, rec_type)}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Column scoring — rank columns by suitability for each analysis type
# ---------------------------------------------------------------------------

def _coefficient_of_variation(stats: dict) -> float:
    """Compute coefficient of variation from column stats. Returns 0.0 if not computable."""
    mean = stats.get("mean")
    stdev = stats.get("stdev")
    if mean is None or stdev is None or mean == 0:
        return 0.0
    return abs(stdev / mean)


def _column_completeness(col_info: dict, row_count: int) -> float:
    """Return fraction of non-null values (0.0 to 1.0). Returns 1.0 if unknown."""
    null_count = col_info.get("null_count", 0)
    if not row_count or row_count <= 0:
        return 1.0
    return max(0.0, 1.0 - null_count / row_count)


def _score_categorical_for_benchmark(
    col_name: str,
    col_info: dict,
    row_count: int,
    profile: dict,
) -> float:
    """Score a categorical column for benchmark suitability (0.0 - 1.0).

    Scoring: cardinality sweet spot [3-50] peak at ~15, named column bonus,
    description bonus, completeness, sample diversity.
    """
    score = 0.0
    cardinality = col_info.get("cardinality", 0)

    # Cardinality fitness: ideal 3-50, peak around 15
    if 3 <= cardinality <= 50:
        distance = abs(cardinality - 15) / 15
        score += 0.4 * max(0.0, 1.0 - distance)
    elif cardinality > 50:
        score += max(0.0, 0.2 - (cardinality - 50) * 0.002)
    elif cardinality == 2:
        score += 0.15

    # Named column bonus
    if not _is_unnamed_column(col_name, profile):
        score += 0.25

    # Has description bonus
    descs = profile.get("column_descriptions", {})
    if descs.get(col_name):
        score += 0.1

    # Completeness
    completeness = _column_completeness(col_info, row_count)
    score += 0.15 * completeness

    # Sample diversity
    samples = col_info.get("sample_values", [])
    if len(set(samples)) >= 3:
        score += 0.1

    return min(score, 1.0)


def _score_numeric_for_benchmark(
    col_name: str,
    col_info: dict,
    row_count: int,
    profile: dict,
) -> float:
    """Score a numeric column for benchmark suitability (0.0 - 1.0).

    Scoring: high CV, non-zero range, named column, completeness.
    """
    score = 0.0
    stats = col_info.get("stats", {})

    cv = _coefficient_of_variation(stats)
    if cv > 0:
        score += min(0.4, cv * 0.4)

    min_val = stats.get("min")
    max_val = stats.get("max")
    if min_val is not None and max_val is not None and max_val > min_val:
        score += 0.2

    if not _is_unnamed_column(col_name, profile):
        score += 0.2

    descs = profile.get("column_descriptions", {})
    if descs.get(col_name):
        score += 0.05

    completeness = _column_completeness(col_info, row_count)
    if completeness < 0.7:
        score -= 0.15
    else:
        score += 0.15 * completeness

    return max(0.0, min(score, 1.0))


def _score_numeric_for_correlation(
    col_name: str,
    col_info: dict,
    row_count: int,
    profile: dict,
) -> float:
    """Score a numeric column for correlation analysis (0.0 - 1.0).

    Higher variance = more likely to show correlation patterns.
    """
    score = 0.0
    stats = col_info.get("stats", {})
    stdev = stats.get("stdev", 0) or 0

    cv = _coefficient_of_variation(stats)
    score += min(0.4, cv * 0.5)

    if stdev > 0:
        score += 0.2

    if not _is_unnamed_column(col_name, profile):
        score += 0.2

    completeness = _column_completeness(col_info, row_count)
    score += 0.2 * completeness

    return max(0.0, min(score, 1.0))


def _score_numeric_for_anomaly(
    col_name: str,
    col_info: dict,
    row_count: int,
    profile: dict,
) -> float:
    """Score a numeric column for anomaly detection (0.0 - 1.0).

    High CV = most variable = most likely to have meaningful outliers.
    Constant columns (stdev=0) → hard 0.
    """
    stats = col_info.get("stats", {})
    if (stats.get("stdev") or 0) == 0:
        return 0.0

    score = 0.0
    cv = _coefficient_of_variation(stats)
    score += min(0.45, cv * 0.5)

    if not _is_unnamed_column(col_name, profile):
        score += 0.25

    completeness = _column_completeness(col_info, row_count)
    score += 0.15 * completeness

    descs = profile.get("column_descriptions", {})
    if descs.get(col_name):
        score += 0.1

    return max(0.0, min(score, 1.0))


_TREND_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "manufacturing": ["output", "yield", "tonnage", "weight", "production", "power", "energy", "cost"],
    "procurement": ["rate", "price", "cost", "amount", "quantity"],
    "quality": ["defect", "rejection", "scrap", "yield"],
    "energy": ["kwh", "kva", "power", "consumption", "demand"],
    "logistics": ["dispatch", "delivery", "turnaround"],
}


def _score_numeric_for_trend(
    col_name: str,
    col_info: dict,
    row_count: int,
    profile: dict,
    domain: str = "general",
) -> float:
    """Score a numeric column for time trend analysis (0.0 - 1.0).

    Prefers business-relevant numerics. Domain keywords boost score.
    """
    stats = col_info.get("stats", {})
    if (stats.get("stdev") or 0) == 0:
        return 0.0

    score = 0.0

    if not _is_unnamed_column(col_name, profile):
        score += 0.25

    descs = profile.get("column_descriptions", {})
    if descs.get(col_name):
        score += 0.15

    kws = _TREND_DOMAIN_KEYWORDS.get(domain, [])
    col_lower = col_name.lower()
    if any(kw in col_lower for kw in kws):
        score += 0.25

    cv = _coefficient_of_variation(stats)
    score += min(0.2, cv * 0.25)

    completeness = _column_completeness(col_info, row_count)
    score += 0.15 * completeness

    return max(0.0, min(score, 1.0))


def _rank_columns(
    columns: dict[str, dict],
    row_count: int,
    profile: dict,
    scorer,
    **scorer_kwargs,
) -> list[str]:
    """Rank columns by score using the given scorer function.

    Returns column names sorted descending by score.
    """
    scored = []
    for col_name, col_info in columns.items():
        s = scorer(col_name, col_info, row_count, profile, **scorer_kwargs)
        scored.append((col_name, s))
    scored.sort(key=lambda x: -x[1])
    return [name for name, _ in scored]


# ---------------------------------------------------------------------------
# Data fitness priority adjustment
# ---------------------------------------------------------------------------

def _data_fitness_adjustment(
    rec_type: str,
    profile: dict,
    columns: list[str],
) -> int:
    """Return a priority adjustment (-20 to +20) based on data fitness.

    Boosts high-quality column combinations, penalizes low-quality ones.
    """
    cls = _get_columns(profile)
    row_count = profile.get("row_count", 0) or 0
    adjustment = 0

    if rec_type == "benchmark" and len(columns) >= 2:
        cat_info = cls.get(columns[0], {})
        num_info = cls.get(columns[1], {})

        card = cat_info.get("cardinality", 0)
        if 5 <= card <= 30:
            adjustment += 10
        elif 3 <= card <= 50:
            adjustment += 5
        elif card > 100:
            adjustment -= 10
        elif card < 3:
            adjustment -= 5

        cv = _coefficient_of_variation(num_info.get("stats", {}))
        if cv > 0.5:
            adjustment += 8
        elif cv > 0.2:
            adjustment += 4
        elif cv < 0.05:
            adjustment -= 8

        for col in columns[:2]:
            info = cls.get(col, {})
            if _column_completeness(info, row_count) < 0.7:
                adjustment -= 5

    elif rec_type == "correlation":
        cvs = []
        for col in columns:
            info = cls.get(col, {})
            cvs.append(_coefficient_of_variation(info.get("stats", {})))
        avg_cv = sum(cvs) / len(cvs) if cvs else 0
        if avg_cv > 0.5:
            adjustment += 10
        elif avg_cv > 0.2:
            adjustment += 5
        elif avg_cv < 0.05:
            adjustment -= 10

    elif rec_type == "anomaly":
        if columns:
            info = cls.get(columns[0], {})
            cv = _coefficient_of_variation(info.get("stats", {}))
            if cv > 0.5:
                adjustment += 10
            elif cv > 0.2:
                adjustment += 5

    elif rec_type in ("time_trend", "forecast"):
        if len(columns) >= 2:
            info = cls.get(columns[1], {})
            cv = _coefficient_of_variation(info.get("stats", {}))
            if cv > 0.3:
                adjustment += 8
            if row_count > 200:
                adjustment += 5

    elif rec_type == "cohort":
        if len(columns) >= 3:
            cat_info = cls.get(columns[0], {})
            card = cat_info.get("cardinality", 0)
            if 3 <= card <= 20:
                adjustment += 8
            elif card > 50:
                adjustment -= 8

    return max(-20, min(20, adjustment))


# ---------------------------------------------------------------------------
# Insight-driven follow-up recommendations
# ---------------------------------------------------------------------------

def _recommend_insight_followups(
    profile: dict,
    table_insights: list[dict],
    columns: dict,
    domain: str,
) -> list[Recommendation]:
    """Generate follow-up recommendations based on existing insight results.

    Maps insight findings to targeted next-step analyses:
    - Critical/warning anomaly → correlate with categoricals + time to find root cause
    - Strong correlation → track relationship over time
    - Multiple insights → suggest composite/story analysis
    """
    recs: list[Recommendation] = []
    table_name = profile.get("table_name", "")
    tbl_label = _table_label(table_name, profile)
    row_count = profile.get("row_count", 0) or 0

    # Categorize insights by type and severity
    critical_anomalies: list[dict] = []
    strong_correlations: list[dict] = []

    for ins in table_insights:
        itype = ins.get("insight_type", "")
        severity = ins.get("severity", "info")
        evidence = ins.get("evidence") or {}

        if itype == "anomaly" and severity in ("critical", "warning"):
            critical_anomalies.append(ins)
        elif itype == "correlation":
            r_val = evidence.get("estimated_correlation") or evidence.get("r_value") or 0
            if abs(r_val) >= 0.7:
                strong_correlations.append(ins)

    cat_cols = [c for c, info in columns.items() if info.get("semantic_type") == "categorical"]
    temporal_cols = [c for c, info in columns.items() if info.get("semantic_type") == "temporal"]

    # Follow-up 1: Critical anomaly → investigate root cause
    for anom in critical_anomalies[:2]:
        anom_cols = anom.get("source_columns") or []
        if not anom_cols:
            continue
        anom_col = anom_cols[0]
        anom_label = _col_label(anom_col, profile)

        # Benchmark: break anomalous column down by categorical
        if cat_cols:
            # Pick best categorical using ranking
            cat_subset = {c: columns[c] for c in cat_cols}
            ranked_cats = _rank_columns(cat_subset, row_count, profile, _score_categorical_for_benchmark)
            best_cat = ranked_cats[0] if ranked_cats else cat_cols[0]
            cat_label = _col_label(best_cat, profile)
            recs.append(Recommendation(
                title=f"Investigate {anom_label} anomalies by {cat_label} in {tbl_label}",
                description=(
                    f"Critical anomalies detected in {anom_label}. "
                    f"Break down by {cat_label} to identify which group is driving the outliers."
                ),
                analysis_type="benchmark",
                target_table=table_name,
                columns=[best_cat, anom_col],
                priority=88,
                reason=f"Follow-up: {anom.get('severity', 'warning')} anomaly in {anom_label} needs root-cause analysis — {_domain_phrase(domain, 'benchmark')}",
            ))

        # Time trend: check if anomaly is increasing, seasonal, or one-time
        if temporal_cols:
            time_col = temporal_cols[0]
            time_label = _col_label(time_col, profile)
            recs.append(Recommendation(
                title=f"Track {anom_label} anomaly pattern over {time_label}",
                description=(
                    f"Check whether anomalies in {anom_label} are increasing, seasonal, or one-time events."
                ),
                analysis_type="time_trend",
                target_table=table_name,
                columns=[time_col, anom_col],
                priority=85,
                reason=f"Follow-up: track when {anom_label} outliers occur — {_domain_phrase(domain, 'trend')}",
            ))

    # Follow-up 2: Strong correlation → track over time
    for corr in strong_correlations[:2]:
        corr_cols = corr.get("source_columns") or []
        evidence = corr.get("evidence") or {}
        r_val = evidence.get("estimated_correlation") or evidence.get("r_value") or 0

        if len(corr_cols) >= 2 and temporal_cols:
            col_a_label = _col_label(corr_cols[0], profile)
            col_b_label = _col_label(corr_cols[1], profile)
            time_col = temporal_cols[0]
            recs.append(Recommendation(
                title=f"Track {col_a_label}–{col_b_label} relationship over time",
                description=(
                    f"Strong correlation (r={r_val:.2f}) found between {col_a_label} and {col_b_label}. "
                    f"Check if this relationship is stable or changing."
                ),
                analysis_type="time_trend",
                target_table=table_name,
                columns=[time_col, corr_cols[0], corr_cols[1]],
                priority=82,
                reason=f"Follow-up: r={r_val:.2f} correlation warrants temporal tracking — {_domain_phrase(domain, 'trend')}",
            ))

    # Follow-up 3: Multiple insights on same table → suggest story/composite
    if len(table_insights) >= 3:
        insight_types_present = {ins.get("insight_type") for ins in table_insights}
        if len(insight_types_present) >= 2:
            recs.append(Recommendation(
                title=f"Build data story for {tbl_label}",
                description=(
                    f"This table has {len(table_insights)} insights across {len(insight_types_present)} types. "
                    f"Combine into a narrative that explains the big picture."
                ),
                analysis_type="story",
                target_table=table_name,
                columns=[],
                priority=78,
                reason=f"Multiple findings ({', '.join(sorted(insight_types_present))}) suggest a composite narrative would be valuable.",
            ))

    return recs


def _infer_entity_type_from_column(col_name: str) -> str | None:
    """Try to infer which entity type a column represents from its name."""
    if not col_name:
        return None
    cl = col_name.lower()
    for entity_type, keywords in _ENTITY_KEYWORDS.items():
        if any(kw in cl for kw in keywords):
            return entity_type
    return None


def _get_rl_mult(reinforcement_weights, analysis_type: str) -> float:
    """Get reinforcement multiplier for an analysis type. Returns 1.0 if None."""
    if reinforcement_weights is None:
        return 1.0
    from business_brain.discovery.reinforcement_loop import get_multiplier
    return get_multiplier(reinforcement_weights, "analysis_type_multipliers", analysis_type)


def _time_priority(row_count: int, insight_count: int, multiplier: float = 1.0) -> int:
    base = int(80 * multiplier)
    if insight_count > 5:
        base -= 20
    if row_count > 100:
        base += 5
    return min(max(base, 10), 100)


def _benchmark_priority(row_count: int, cat_count: int, insight_count: int, multiplier: float = 1.0) -> int:
    base = int(70 * multiplier)
    if cat_count > 2:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)


def _correlation_priority(num_col_count: int, insight_count: int, multiplier: float = 1.0) -> int:
    base = int(65 * multiplier)
    if num_col_count > 4:
        base += 15
    if insight_count > 3:
        base -= 10
    return min(max(base, 10), 100)


def _anomaly_priority(row_count: int, insight_count: int, multiplier: float = 1.0) -> int:
    base = int(75 * multiplier)
    if row_count > 200:
        base += 5
    if insight_count > 5:
        base -= 20
    return min(max(base, 10), 100)


def _cohort_priority(row_count: int, insight_count: int, multiplier: float = 1.0) -> int:
    base = int(60 * multiplier)
    if row_count > 100:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)


def _forecast_priority(row_count: int, insight_count: int, multiplier: float = 1.0) -> int:
    base = int(55 * multiplier)
    if row_count > 50:
        base += 10
    if insight_count > 5:
        base -= 15
    return min(max(base, 10), 100)
