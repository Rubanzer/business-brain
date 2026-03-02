"""Domain-specific analysis dispatcher — bridges discovery engine to domain modules.

Two-tier approach:
  Tier 1: Pre-built module matching (fast, reliable) — 10 steel/mfg modules
  Tier 2: LLM-generated dynamic analysis (flexible, any industry) — fallback
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column matching helpers
# ---------------------------------------------------------------------------

@dataclass
class ColumnSpec:
    """Describes a required column using keywords + semantic types."""

    param_name: str  # function parameter name (e.g. "heat_column")
    keywords: list[str]  # substring matches against column names
    semantic_types: list[str] = field(default_factory=list)  # from column_classifier
    required: bool = True


@dataclass
class ModuleEntry:
    """Registry entry for a pre-built domain analysis module."""

    id: str
    domains: list[str]  # which domain_hint values trigger it
    columns: list[ColumnSpec]  # required columns to match
    module_path: str  # importable module path
    func_name: str  # function name to call
    result_to_insights: str  # name of converter function
    description: str  # human-readable description


# ---------------------------------------------------------------------------
# Pre-built module registry (Tier 1)
# ---------------------------------------------------------------------------

DOMAIN_MODULE_REGISTRY: list[ModuleEntry] = [
    ModuleEntry(
        id="heat_analysis",
        domains=["manufacturing"],
        columns=[
            ColumnSpec("heat_column", ["heat", "batch", "lot", "heat_no", "batch_no", "lot_no"],
                       ["identifier", "categorical"], required=True),
            ColumnSpec("weight_column", ["weight", "tonnage", "output", "production", "ton"],
                       ["numeric_metric"], required=True),
            ColumnSpec("grade_column", ["grade", "quality", "type", "specification"],
                       ["categorical"], required=False),
            ColumnSpec("time_column", ["date", "time", "created", "tap_time"],
                       ["temporal"], required=False),
        ],
        module_path="business_brain.discovery.heat_analysis",
        func_name="analyze_heats",
        result_to_insights="heat_analysis",
        description="Heat-wise production analysis — weight distribution, grade mix, cycle times",
    ),
    ModuleEntry(
        id="material_balance",
        domains=["manufacturing"],
        columns=[
            ColumnSpec("entity_column", ["furnace", "plant", "mill", "line", "stage", "section"],
                       ["categorical", "identifier"], required=True),
            ColumnSpec("input_column", ["input", "charge", "raw", "feed", "incoming"],
                       ["numeric_metric"], required=True),
            ColumnSpec("output_column", ["output", "produced", "finished", "yield", "product"],
                       ["numeric_metric"], required=True),
            ColumnSpec("loss_column", ["loss", "waste", "scrap", "slag", "reject"],
                       ["numeric_metric"], required=False),
        ],
        module_path="business_brain.discovery.material_balance",
        func_name="compute_material_balance",
        result_to_insights="material_balance",
        description="Material recovery tracking — input vs output with loss analysis",
    ),
    ModuleEntry(
        id="quality_control",
        domains=["manufacturing", "quality"],
        columns=[
            ColumnSpec("entity_column", ["machine", "line", "station", "furnace", "shift", "operator"],
                       ["categorical", "identifier"], required=True),
            ColumnSpec("defect_column", ["defect", "reject", "rework", "scrap", "failure", "rejection"],
                       ["numeric_metric", "numeric_percentage"], required=True),
            ColumnSpec("quantity_column", ["quantity", "total", "production", "output", "count"],
                       ["numeric_metric"], required=False),
        ],
        module_path="business_brain.discovery.quality_control",
        func_name="analyze_defects",
        result_to_insights="quality_control",
        description="Defect rate analysis — rejection rates by entity with SPC insights",
    ),
    ModuleEntry(
        id="power_monitor",
        domains=["manufacturing", "energy"],
        columns=[
            ColumnSpec("time_column", ["time", "date", "hour", "timestamp", "period"],
                       ["temporal"], required=True),
            ColumnSpec("power_column", ["kw", "kva", "power", "kwh", "consumption", "demand", "load"],
                       ["numeric_metric"], required=True),
            ColumnSpec("entity_column", ["furnace", "machine", "meter", "feeder", "section"],
                       ["categorical", "identifier"], required=False),
        ],
        module_path="business_brain.discovery.power_monitor",
        func_name="analyze_load_profile",
        result_to_insights="power_monitor",
        description="Power consumption patterns — load profile, peak demand, load factor",
    ),
    ModuleEntry(
        id="supplier_scorecard",
        domains=["procurement"],
        columns=[
            ColumnSpec("supplier_column", ["supplier", "vendor", "party", "seller"],
                       ["categorical", "identifier"], required=True),
            # Metrics are auto-detected from all numeric columns
        ],
        module_path="business_brain.discovery.supplier_scorecard",
        func_name="build_scorecard",
        result_to_insights="supplier_scorecard",
        description="Supplier performance scoring — weighted metrics with risk detection",
    ),
    ModuleEntry(
        id="production_scheduler",
        domains=["manufacturing"],
        columns=[
            ColumnSpec("shift_column", ["shift", "batch", "crew", "team"],
                       ["categorical"], required=True),
            ColumnSpec("output_column", ["output", "production", "tonnage", "quantity", "count"],
                       ["numeric_metric"], required=True),
            ColumnSpec("target_column", ["target", "plan", "budget", "expected"],
                       ["numeric_metric"], required=False),
            ColumnSpec("time_column", ["date", "time", "period"],
                       ["temporal"], required=False),
        ],
        module_path="business_brain.discovery.production_scheduler",
        func_name="analyze_shift_performance",
        result_to_insights="production_scheduler",
        description="Shift-wise performance comparison — best/worst shifts, variance analysis",
    ),
    ModuleEntry(
        id="downtime_analyzer",
        domains=["manufacturing"],
        columns=[
            ColumnSpec("machine_column", ["machine", "equipment", "furnace", "line", "asset"],
                       ["categorical", "identifier"], required=True),
            ColumnSpec("duration_column", ["duration", "downtime", "hours", "minutes", "time_lost"],
                       ["numeric_metric"], required=True),
            ColumnSpec("reason_column", ["reason", "cause", "type", "category", "failure"],
                       ["categorical"], required=False),
            ColumnSpec("time_column", ["date", "time", "timestamp", "reported"],
                       ["temporal"], required=False),
        ],
        module_path="business_brain.discovery.downtime_analyzer",
        func_name="analyze_downtime",
        result_to_insights="downtime_analyzer",
        description="Downtime Pareto analysis — worst machines, recurring failures, MTTR",
    ),
    ModuleEntry(
        id="dispatch_gate",
        domains=["logistics"],
        columns=[
            ColumnSpec("time_column", ["time", "date", "timestamp", "entry_time", "exit_time"],
                       ["temporal"], required=True),
            ColumnSpec("vehicle_column", ["vehicle", "truck", "lorry", "transporter"],
                       ["categorical", "identifier"], required=False),
            ColumnSpec("direction_column", ["direction", "type", "in_out", "entry", "exit"],
                       ["categorical"], required=False),
        ],
        module_path="business_brain.discovery.dispatch_gate",
        func_name="analyze_gate_traffic",
        result_to_insights="dispatch_gate",
        description="Gate traffic analysis — peak hours, vehicle patterns, throughput",
    ),
    ModuleEntry(
        id="rate_analysis",
        domains=["procurement"],
        columns=[
            ColumnSpec("supplier_column", ["supplier", "vendor", "party", "seller"],
                       ["categorical", "identifier"], required=True),
            ColumnSpec("rate_column", ["rate", "price", "cost", "unit_price", "amount"],
                       ["numeric_currency", "numeric_metric"], required=True),
            ColumnSpec("item_column", ["item", "material", "product", "grade", "sku"],
                       ["categorical"], required=False),
        ],
        module_path="business_brain.discovery.rate_analysis",
        func_name="compare_rates",
        result_to_insights="rate_analysis",
        description="Rate comparison across suppliers — savings potential, best/worst rates",
    ),
    ModuleEntry(
        id="efficiency_metrics",
        domains=["manufacturing"],
        columns=[
            ColumnSpec("entity_column", ["machine", "line", "furnace", "equipment", "section"],
                       ["categorical", "identifier"], required=True),
            ColumnSpec("availability_column", ["availability", "uptime", "running"],
                       ["numeric_metric", "numeric_percentage"], required=True),
            ColumnSpec("performance_column", ["performance", "speed", "efficiency", "throughput"],
                       ["numeric_metric", "numeric_percentage"], required=True),
            ColumnSpec("quality_column", ["quality", "yield", "pass_rate", "good"],
                       ["numeric_metric", "numeric_percentage"], required=True),
        ],
        module_path="business_brain.discovery.efficiency_metrics",
        func_name="compute_oee",
        result_to_insights="efficiency_metrics",
        description="OEE calculation — availability × performance × quality per entity",
    ),
]


# ---------------------------------------------------------------------------
# Core dispatch logic
# ---------------------------------------------------------------------------

async def run_domain_analysis(
    session: AsyncSession,
    profiles: list[TableProfile],
) -> list[Insight]:
    """Run domain-specific analysis on all profiled tables.

    For each table:
      Tier 1 — check DOMAIN_MODULE_REGISTRY for pre-built module match
      Tier 2 — if no match, try LLM-generated dynamic analysis (Change 1b)

    Returns list of domain-specific Insight objects.
    """
    all_insights: list[Insight] = []

    for profile in profiles:
        domain = (profile.domain_hint or "general").lower()
        cls = profile.column_classification or {}
        cols = cls.get("columns", {})

        if not cols:
            continue

        # Tier 1: Pre-built modules
        tier1_matched = False
        for entry in DOMAIN_MODULE_REGISTRY:
            # Check domain match
            if domain not in entry.domains and "general" not in entry.domains:
                continue

            # Try to match columns
            col_mapping = _match_columns(entry.columns, cols)
            if col_mapping is None:
                continue  # Required columns not found

            tier1_matched = True
            try:
                insights = await _run_module(
                    session, profile, entry, col_mapping,
                )
                if insights:
                    all_insights.extend(insights)
                    logger.info(
                        "Domain [%s] on %s: %d insights",
                        entry.id, profile.table_name, len(insights),
                    )
            except Exception:
                logger.exception(
                    "Domain module %s failed on %s",
                    entry.id, profile.table_name,
                )

        # Tier 2: Dynamic analysis fallback (if no pre-built module matched)
        if not tier1_matched and domain != "general":
            try:
                dynamic_insights = await _generate_dynamic_analysis(
                    session, profile,
                )
                if dynamic_insights:
                    all_insights.extend(dynamic_insights)
                    logger.info(
                        "Dynamic analysis on %s: %d insights",
                        profile.table_name, len(dynamic_insights),
                    )
            except Exception:
                logger.exception(
                    "Dynamic analysis failed on %s", profile.table_name,
                )

    return all_insights


# ---------------------------------------------------------------------------
# Column matching
# ---------------------------------------------------------------------------

def _match_columns(
    specs: list[ColumnSpec],
    classified_cols: dict[str, dict],
) -> dict[str, str] | None:
    """Match column specs against classified columns.

    Returns dict mapping param_name -> actual_column_name, or None if
    required columns can't be matched.
    """
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for spec in specs:
        best_col = _find_best_column(spec, classified_cols, used)
        if best_col:
            mapping[spec.param_name] = best_col
            used.add(best_col)
        elif spec.required:
            return None  # Required column not found

    return mapping


def _find_best_column(
    spec: ColumnSpec,
    classified_cols: dict[str, dict],
    used: set[str],
) -> str | None:
    """Find the best matching column for a spec.

    Scoring: keyword match in name (2pts) + semantic type match (1pt).
    """
    candidates: list[tuple[int, str]] = []

    for col_name, info in classified_cols.items():
        if col_name in used:
            continue

        score = 0
        col_lower = col_name.lower()

        # Keyword match (substring)
        for kw in spec.keywords:
            if kw in col_lower:
                score += 2
                break

        # Semantic type match
        sem_type = info.get("semantic_type", "")
        if spec.semantic_types and sem_type in spec.semantic_types:
            score += 1

        if score > 0:
            candidates.append((score, col_name))

    if not candidates:
        return None

    # Return highest scoring match
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Module execution
# ---------------------------------------------------------------------------

async def _run_module(
    session: AsyncSession,
    profile: TableProfile,
    entry: ModuleEntry,
    col_mapping: dict[str, str],
) -> list[Insight]:
    """Fetch data and run a domain module, converting results to Insights."""
    import importlib

    # Fetch rows from the table (limit 500 for performance)
    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", profile.table_name)
    result = await session.execute(
        sql_text(f'SELECT * FROM "{safe_table}" LIMIT 500')
    )
    rows = [dict(r._mapping) for r in result.fetchall()]

    if not rows:
        return []

    # Import and call the module function
    mod = importlib.import_module(entry.module_path)
    func = getattr(mod, entry.func_name)

    # Build kwargs from column mapping
    kwargs = {"rows": rows}
    for param_name, col_name in col_mapping.items():
        kwargs[param_name] = col_name

    # Special handling for supplier_scorecard — needs metrics list
    if entry.id == "supplier_scorecard":
        kwargs = _build_scorecard_kwargs(rows, col_mapping, profile)

    analysis_result = func(**kwargs)

    if analysis_result is None:
        return []

    # Convert result to insights
    converter = _RESULT_CONVERTERS.get(entry.result_to_insights)
    if converter:
        return converter(analysis_result, profile, entry)

    return []


def _build_scorecard_kwargs(
    rows: list[dict],
    col_mapping: dict[str, str],
    profile: TableProfile,
) -> dict:
    """Build kwargs for supplier_scorecard.build_scorecard()."""
    cls = profile.column_classification or {}
    cols = cls.get("columns", {})

    supplier_col = col_mapping["supplier_column"]
    metrics = []

    # Find all numeric columns as metrics (excluding the supplier column)
    for col_name, info in cols.items():
        if col_name == supplier_col:
            continue
        sem_type = info.get("semantic_type", "")
        if sem_type in ("numeric_metric", "numeric_currency", "numeric_percentage"):
            # Determine direction: lower is better for cost/rate, higher for quality/output
            col_lower = col_name.lower()
            direction = "lower"
            if any(kw in col_lower for kw in ["quality", "output", "yield", "score", "rating"]):
                direction = "higher"

            metrics.append({
                "column": col_name,
                "weight": 1.0,
                "direction": direction,
            })

    if not metrics:
        return {"rows": rows, "supplier_column": supplier_col, "metrics": []}

    # Normalize weights
    weight = round(1.0 / len(metrics), 2)
    for m in metrics:
        m["weight"] = weight

    return {
        "rows": rows,
        "supplier_column": supplier_col,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Result-to-Insight converters
# ---------------------------------------------------------------------------

def _convert_heat_analysis(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert HeatAnalysisResult to Insights."""
    insights = []
    table = profile.table_name

    # Main production summary insight
    title = (
        f"{result.total_heats} heats produced {result.total_weight:,.0f} MT — "
        f"avg {result.avg_weight_per_heat:,.1f} MT/heat in {table}"
    )
    desc = (
        f"Total production: {result.total_weight:,.0f} MT across {result.total_heats} heats. "
        f"Average heat weight: {result.avg_weight_per_heat:,.1f} MT "
        f"(range: {result.min_weight:,.1f} - {result.max_weight:,.1f} MT, "
        f"σ={result.weight_std:,.1f} MT)."
    )

    # Grade distribution adds context
    if result.grade_distribution:
        top_grades = sorted(result.grade_distribution.items(), key=lambda x: x[1], reverse=True)[:3]
        grade_str = ", ".join(f"{g}: {c} heats" for g, c in top_grades)
        desc += f" Grade mix: {grade_str}."

    severity = "info"
    impact = 50

    # Flag high variability
    if result.avg_weight_per_heat > 0:
        cv = result.weight_std / result.avg_weight_per_heat
        if cv > 0.25:
            severity = "warning"
            impact = 65
            title = (
                f"High heat weight variability ({cv:.0%} CV) in {table} — "
                f"range {result.min_weight:,.1f} to {result.max_weight:,.1f} MT"
            )
            desc += (
                f" Coefficient of variation is {cv:.0%}, indicating inconsistent "
                f"heat sizes that impact planning and energy efficiency."
            )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "heat_analysis",
            "total_heats": result.total_heats,
            "total_weight": result.total_weight,
            "avg_weight": result.avg_weight_per_heat,
            "weight_std": result.weight_std,
            "grade_distribution": result.grade_distribution,
        },
        suggested_actions=[
            f"Standardize heat sizes to reduce variability (current CV: {result.weight_std / max(result.avg_weight_per_heat, 1):.0%})",
            f"Review minimum heats ({result.min_weight:,.1f} MT) for cost efficiency",
        ],
    ))

    return insights


def _convert_material_balance(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert MaterialBalanceResult to Insights."""
    insights = []
    table = profile.table_name

    recovery = result.overall_recovery_pct
    loss = result.total_loss
    severity = "critical" if recovery < 85 else "warning" if recovery < 92 else "info"
    impact = 80 if recovery < 85 else 60 if recovery < 92 else 40

    # Build full entity breakdown for visibility
    entity_breakdown = []
    for ent in (result.entities or []):
        entity_breakdown.append({
            "entity": getattr(ent, "entity", ""),
            "input": getattr(ent, "total_input", 0),
            "output": getattr(ent, "total_output", 0),
            "loss": getattr(ent, "loss", 0),  # EntityBalance uses "loss" not "total_loss"
            "recovery_pct": getattr(ent, "recovery_pct", 0),
        })

    # Sort by recovery for the description
    entity_breakdown.sort(key=lambda x: x.get("recovery_pct", 0))

    ranking_str = "; ".join(
        f"{e['entity']}: {e.get('recovery_pct', 0):.1f}% recovery"
        for e in entity_breakdown
    )

    title = (
        f"Material recovery at {recovery:.1f}% in {table} — "
        f"{loss:,.0f} MT loss from {result.total_input:,.0f} MT input"
    )
    desc = (
        f"Input: {result.total_input:,.0f} MT, Output: {result.total_output:,.0f} MT, "
        f"Loss: {loss:,.0f} MT ({100 - recovery:.1f}%). "
        f"Worst: {result.worst_recovery_entity}, Best: {result.best_recovery_entity}. "
        f"Full breakdown: {ranking_str}."
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "material_balance",
            "total_input": result.total_input,
            "total_output": result.total_output,
            "total_loss": result.total_loss,
            "recovery_pct": recovery,
            "worst_entity": result.worst_recovery_entity,
            "best_entity": result.best_recovery_entity,
            "full_breakdown": entity_breakdown,
        },
        suggested_actions=[
            f"Investigate {result.worst_recovery_entity} — lowest recovery rate",
            f"Target {loss * 0.1:,.0f} MT loss reduction (10% improvement = ₹{loss * 0.1 * 35000:,.0f} saved at ₹35K/MT)",
        ],
    ))

    return insights


def _convert_quality_control(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert DefectResult to Insights."""
    insights = []
    table = profile.table_name

    rate = result.overall_defect_rate
    severity = "critical" if rate > 5 else "warning" if rate > 2 else "info"
    impact = 75 if rate > 5 else 55 if rate > 2 else 35

    # Build entity breakdown
    entity_breakdown = []
    for ent in (result.entities or []):
        entity_breakdown.append({
            "entity": getattr(ent, "entity", ""),
            "defects": getattr(ent, "defect_count", 0),
            "quantity": getattr(ent, "quantity", 0),
            "defect_rate": getattr(ent, "defect_rate", 0),
        })
    entity_breakdown.sort(key=lambda x: x.get("defect_rate", 0), reverse=True)

    ranking_str = "; ".join(
        f"{e['entity']}: {e.get('defect_rate', 0):.1f}%"
        for e in entity_breakdown
    )

    title = (
        f"Defect rate {rate:.1f}% across {result.total_quantity:,} units in {table} — "
        f"{result.total_defects:,} defects"
    )
    desc = (
        f"Overall defect rate: {rate:.1f}% ({result.total_defects:,} defects / "
        f"{result.total_quantity:,} units). "
        f"Worst: {result.worst_entity}, Best: {result.best_entity}. "
        f"All entities: {ranking_str}."
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "quality_control",
            "defect_rate": rate,
            "total_defects": result.total_defects,
            "total_quantity": result.total_quantity,
            "worst_entity": result.worst_entity,
            "best_entity": result.best_entity,
            "full_breakdown": entity_breakdown,
        },
        suggested_actions=[
            f"Focus quality improvement on {result.worst_entity} — highest defect rate",
            f"Root cause analysis on top defect types to reduce rejection by 50%",
        ],
    ))

    return insights


def _convert_power_monitor(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert LoadProfileResult to Insights."""
    insights = []
    table = profile.table_name

    lf = result.load_factor
    severity = "warning" if lf < 0.65 else "info"
    impact = 60 if lf < 0.65 else 40

    title = (
        f"Load factor {lf:.0%} — peak {result.peak_demand:,.0f} kW vs avg {result.avg_demand:,.0f} kW in {table}"
    )
    desc = (
        f"Peak demand: {result.peak_demand:,.0f} kW ({result.peak_period}), "
        f"Average: {result.avg_demand:,.0f} kW, Minimum: {result.min_demand:,.0f} kW "
        f"({result.off_peak_period}). "
        f"Load factor: {lf:.0%} — "
        f"{'poor load management, high demand charges likely' if lf < 0.65 else 'acceptable load profile'}."
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "power_monitor",
            "peak_demand": result.peak_demand,
            "avg_demand": result.avg_demand,
            "min_demand": result.min_demand,
            "load_factor": lf,
            "peak_period": result.peak_period,
            "off_peak_period": result.off_peak_period,
        },
        suggested_actions=[
            f"Shift non-critical loads away from peak period ({result.peak_period})",
            f"Improve load factor from {lf:.0%} to >80% to reduce demand charges",
        ],
    ))

    return insights


def _convert_supplier_scorecard(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert ScorecardResult to Insights."""
    insights = []
    table = profile.table_name

    severity = "warning" if result.worst_supplier else "info"
    impact = 55

    # Build full supplier ranking
    supplier_breakdown = []
    for s in (result.suppliers or []):
        # metric_scores is a list of MetricScore objects — convert to serializable form
        raw_metrics = getattr(s, "metric_scores", [])
        metrics = {}
        if isinstance(raw_metrics, list):
            for ms in raw_metrics:
                metrics[getattr(ms, "metric", "?")] = getattr(ms, "score", 0)
        elif isinstance(raw_metrics, dict):
            metrics = raw_metrics
        supplier_breakdown.append({
            "supplier": getattr(s, "supplier", ""),
            "score": getattr(s, "score", 0),
            "grade": getattr(s, "grade", ""),
            "rank": getattr(s, "rank", 0),
            "metrics": metrics,
            "strengths": getattr(s, "strengths", []),
            "weaknesses": getattr(s, "weaknesses", []),
        })
    supplier_breakdown.sort(key=lambda x: x.get("score", 0), reverse=True)

    ranking_str = "; ".join(
        f"{s['supplier']}: {s.get('score', 0):.0f} ({s.get('grade', '')})"
        for s in supplier_breakdown
    )

    title = (
        f"{result.supplier_count} suppliers scored — "
        f"best: {result.best_supplier}, worst: {result.worst_supplier} in {table}"
    )
    desc = (
        f"Evaluated {result.supplier_count} suppliers. "
        f"Mean score: {result.mean_score:.1f}. "
        f"Full ranking: {ranking_str}. "
        f"Grade distribution: {result.grade_distribution}."
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "supplier_scorecard",
            "supplier_count": result.supplier_count,
            "mean_score": result.mean_score,
            "best_supplier": result.best_supplier,
            "worst_supplier": result.worst_supplier,
            "grade_distribution": result.grade_distribution,
            "full_breakdown": supplier_breakdown,
        },
        suggested_actions=[
            f"Review {result.worst_supplier} — lowest performing supplier",
            f"Increase allocation to {result.best_supplier} to improve overall quality",
        ],
    ))

    return insights


def _convert_production_scheduler(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert ShiftPerformanceResult to Insights."""
    insights = []
    table = profile.table_name

    variance = result.variance_pct
    severity = "warning" if variance > 20 else "info"
    impact = 60 if variance > 20 else 40

    # Build full shift breakdown
    shift_breakdown = []
    for shift in (result.shifts or []):
        shift_breakdown.append({
            "shift": getattr(shift, "shift", ""),
            "total_output": getattr(shift, "total_output", 0),
            "avg_output": getattr(shift, "avg_output", 0),
            "event_count": getattr(shift, "event_count", 0),  # ShiftPerformance uses "event_count"
        })
    shift_breakdown.sort(key=lambda x: x.get("total_output", 0), reverse=True)

    ranking_str = "; ".join(
        f"{s['shift']}: {s.get('total_output', 0):,.0f} total ({s.get('event_count', 0)} entries)"
        for s in shift_breakdown
    )

    title = (
        f"{variance:.0f}% shift performance gap — "
        f"best: {result.best_shift}, worst: {result.worst_shift} in {table}"
    )
    desc = (
        f"Total output: {result.total_output:,.0f}. "
        f"Best shift: {result.best_shift}, Worst: {result.worst_shift}. "
        f"Performance variance: {variance:.1f}% between shifts. "
        f"Full breakdown: {ranking_str}. "
        f"{'Significant gap — investigate staffing, training, or equipment differences.' if variance > 20 else 'Shifts performing within acceptable range.'}"
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "production_scheduler",
            "best_shift": result.best_shift,
            "worst_shift": result.worst_shift,
            "variance_pct": variance,
            "total_output": result.total_output,
            "full_breakdown": shift_breakdown,
        },
        suggested_actions=[
            f"Audit {result.worst_shift} shift operations — lowest output",
            f"Replicate {result.best_shift} shift practices across all shifts",
        ],
    ))

    return insights


def _convert_downtime_analyzer(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert DowntimeResult to Insights."""
    insights = []
    table = profile.table_name

    severity = "warning" if result.total_downtime > 100 else "info"
    impact = 65 if result.total_downtime > 100 else 40

    # Top reasons string
    reason_str = ""
    if result.top_reasons:
        top_3 = result.top_reasons[:3]
        reason_parts = []
        for r in top_3:
            reason_parts.append(f"{r.reason}: {r.total_duration:,.0f} hrs ({r.event_count} events)")
        reason_str = f" Top causes: {'; '.join(reason_parts)}."

    title = (
        f"{result.total_downtime:,.0f} hrs downtime across {result.total_events} events in {table}"
    )
    desc = (
        f"Total downtime: {result.total_downtime:,.0f} hours over {result.total_events} events. "
        f"Worst machine: {result.worst_machine or 'N/A'}. "
        f"Best machine: {result.best_machine or 'N/A'}."
        f"{reason_str}"
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "downtime_analyzer",
            "total_downtime": result.total_downtime,
            "total_events": result.total_events,
            "worst_machine": result.worst_machine,
            "best_machine": result.best_machine,
            "top_reasons": [
                {"reason": r.reason, "duration": r.total_duration, "events": r.event_count}
                for r in (result.top_reasons or [])[:5]
            ],
        },
        suggested_actions=[
            f"Focus preventive maintenance on {result.worst_machine or 'worst-performing machine'}",
            f"Address top downtime cause to reduce unplanned stops by 30%",
        ],
    ))

    return insights


def _convert_dispatch_gate(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert GateTrafficResult to Insights."""
    insights = []
    table = profile.table_name

    title = (
        f"{result.total_vehicles} vehicles — peak at {result.peak_period}, "
        f"avg {result.avg_per_period:.0f}/period in {table}"
    )
    desc = (
        f"Total vehicles: {result.total_vehicles}. "
        f"Peak period: {result.peak_period}, Off-peak: {result.off_peak_period}. "
        f"Average: {result.avg_per_period:.0f} vehicles per period."
    )
    if result.direction_split:
        desc += f" Direction split: {result.direction_split}."

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity="info",
        impact_score=35,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "dispatch_gate",
            "total_vehicles": result.total_vehicles,
            "peak_period": result.peak_period,
            "off_peak_period": result.off_peak_period,
            "avg_per_period": result.avg_per_period,
        },
        suggested_actions=[
            f"Schedule dispatches outside peak period ({result.peak_period}) to reduce wait times",
            f"Consider adding weighbridge capacity during peak hours",
        ],
    ))

    return insights


def _convert_rate_analysis(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert RateComparisonResult to Insights."""
    insights = []
    table = profile.table_name

    savings = result.overall_savings_potential
    spread = result.rate_spread_pct
    severity = "warning" if savings > 0 else "info"
    impact = 65 if savings > 10000 else 45

    title = (
        f"₹{savings:,.0f} savings potential — "
        f"{spread:.0f}% rate spread between {result.best_rate_supplier} and {result.worst_rate_supplier}"
    )
    desc = (
        f"Rate spread: {spread:.1f}% between best ({result.best_rate_supplier}) "
        f"and worst ({result.worst_rate_supplier}). "
        f"Potential savings if all procurement at best rate: ₹{savings:,.0f}."
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "rate_analysis",
            "savings_potential": savings,
            "rate_spread_pct": spread,
            "best_supplier": result.best_rate_supplier,
            "worst_supplier": result.worst_rate_supplier,
        },
        suggested_actions=[
            f"Renegotiate rates with {result.worst_rate_supplier} — highest cost supplier",
            f"Increase procurement share with {result.best_rate_supplier} for immediate savings",
        ],
    ))

    return insights


def _convert_efficiency_metrics(result, profile: TableProfile, entry: ModuleEntry) -> list[Insight]:
    """Convert OEEResult to Insights."""
    insights = []
    table = profile.table_name

    oee = result.mean_oee
    severity = "critical" if oee < 60 else "warning" if oee < 75 else "info"
    impact = 75 if oee < 60 else 55 if oee < 75 else 40

    title = (
        f"OEE at {oee:.1f}% — "
        f"{'below world-class (85%)' if oee < 85 else 'world-class performance'} in {table}"
    )
    desc = (
        f"Mean OEE: {oee:.1f}%. "
        f"Best: {result.best_entity}, Worst: {result.worst_entity}. "
        f"{result.world_class_count} entities at world-class level (≥85%)."
    )

    insights.append(Insight(
        id=str(uuid.uuid4()),
        insight_type="domain_analysis",
        severity=severity,
        impact_score=impact,
        title=title,
        description=desc,
        source_tables=[table],
        source_columns=[],
        evidence={
            "module": "efficiency_metrics",
            "mean_oee": oee,
            "best_entity": result.best_entity,
            "worst_entity": result.worst_entity,
            "world_class_count": result.world_class_count,
        },
        suggested_actions=[
            f"Focus improvement on {result.worst_entity} — lowest OEE",
            f"Target {oee + 5:.0f}% OEE (+5pp improvement) across all entities",
        ],
    ))

    return insights


# Converter registry
_RESULT_CONVERTERS = {
    "heat_analysis": _convert_heat_analysis,
    "material_balance": _convert_material_balance,
    "quality_control": _convert_quality_control,
    "power_monitor": _convert_power_monitor,
    "supplier_scorecard": _convert_supplier_scorecard,
    "production_scheduler": _convert_production_scheduler,
    "downtime_analyzer": _convert_downtime_analyzer,
    "dispatch_gate": _convert_dispatch_gate,
    "rate_analysis": _convert_rate_analysis,
    "efficiency_metrics": _convert_efficiency_metrics,
}


# ---------------------------------------------------------------------------
# Tier 2: Dynamic analysis (LLM-generated code)
# ---------------------------------------------------------------------------

async def _generate_dynamic_analysis(
    session: AsyncSession,
    profile: TableProfile,
) -> list[Insight]:
    """Tier 2 fallback: Use LLM to generate custom analysis code for any industry.

    Only runs when no pre-built module matched. Uses company profile context
    to generate industry-specific analysis.
    """
    from business_brain.analysis.tools.llm_gateway import reason as _llm_reason

    # Get company profile for context
    company_context = await _get_company_context(session)

    # Get table data
    safe_table = re.sub(r"[^a-zA-Z0-9_]", "", profile.table_name)
    result = await session.execute(
        sql_text(f'SELECT * FROM "{safe_table}" LIMIT 500')
    )
    rows = [dict(r._mapping) for r in result.fetchall()]
    if not rows:
        return []

    # Build column summary from classification
    cls = profile.column_classification or {}
    cols = cls.get("columns", {})
    col_summary_parts = []
    for col_name, info in cols.items():
        sem = info.get("semantic_type", "unknown")
        stats = info.get("stats")
        samples = info.get("sample_values", [])[:3]
        part = f"  - {col_name}: {sem}"
        if stats:
            part += f" (mean={stats.get('mean')}, min={stats.get('min')}, max={stats.get('max')})"
        if samples:
            part += f" samples={samples}"
        col_summary_parts.append(part)
    col_summary = "\n".join(col_summary_parts)

    # Format sample rows
    sample_display = []
    for row in rows[:5]:
        sample_display.append(
            {k: str(v)[:50] for k, v in row.items()}
        )

    import json

    prompt = (
        f"You are a data analyst for a {company_context.get('industry', 'manufacturing')} company"
        f"{' that makes ' + company_context.get('products', '') if company_context.get('products') else ''}.\n"
        f"{'Process flow: ' + company_context.get('process_flow', '') if company_context.get('process_flow') else ''}\n\n"
        f"Table: {profile.table_name} ({profile.row_count or 0} rows)\n"
        f"Domain: {profile.domain_hint or 'general'}\n"
        f"Columns:\n{col_summary}\n\n"
        f"Sample data (first 5 rows):\n{json.dumps(sample_display, indent=2, default=str)}\n\n"
        "Write Python analysis code for this data. The variable `rows` contains ALL rows as list[dict].\n\n"
        "RULES:\n"
        "- Find the 2-3 MOST IMPORTANT business findings in this data\n"
        "- Every finding MUST include specific numbers (averages, percentages, comparisons)\n"
        "- Focus on: performance gaps, cost savings, quality issues, efficiency problems\n"
        "- Use print() to output each finding as: FINDING: <title> | <description with numbers>\n"
        "- Only use stdlib: statistics, collections, math, datetime, re, json\n"
        "- Keep under 60 lines. No functions or classes.\n"
        "- If the data doesn't support meaningful analysis, print NO_FINDINGS\n"
    )

    try:
        code = await _llm_reason(prompt)

        # Clean markdown fences
        if code.startswith("```"):
            lines = code.split("\n")
            code = "\n".join(lines[1:])
        if code.endswith("```"):
            code = code[:-3].strip()

    except Exception:
        logger.exception("LLM code generation failed for %s", profile.table_name)
        return []

    # Execute in sandbox
    try:
        from business_brain.cognitive.python_analyst_agent import execute_sandboxed
        exec_result = execute_sandboxed(code, {"rows": rows})
    except ImportError:
        logger.warning("Sandbox not available — skipping dynamic analysis")
        return []
    except Exception:
        logger.exception("Sandbox execution failed for %s", profile.table_name)
        return []

    # Parse FINDING: lines from stdout
    stdout = exec_result.get("stdout", "") if isinstance(exec_result, dict) else str(exec_result)
    return _parse_dynamic_findings(stdout, profile, code)


async def _get_company_context(session: AsyncSession) -> dict:
    """Get company profile context for LLM prompt."""
    try:
        from sqlalchemy import select
        from business_brain.db.v3_models import CompanyProfile
        result = await session.execute(select(CompanyProfile).limit(1))
        cp = result.scalar_one_or_none()
        if cp:
            return {
                "industry": getattr(cp, "industry", ""),
                "products": getattr(cp, "products", ""),
                "process_flow": getattr(cp, "process_flow", ""),
                "departments": getattr(cp, "departments", ""),
            }
    except Exception:
        logger.debug("Could not fetch company profile for dynamic analysis context")
    return {}


def _parse_dynamic_findings(
    stdout: str,
    profile: TableProfile,
    generated_code: str,
) -> list[Insight]:
    """Parse FINDING: lines from dynamic analysis output into Insights."""
    if not stdout or "NO_FINDINGS" in stdout:
        return []

    insights = []
    for line in stdout.split("\n"):
        line = line.strip()
        if not line.startswith("FINDING:"):
            continue

        content = line[len("FINDING:"):].strip()
        parts = content.split("|", 1)
        title = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else title

        if not title:
            continue

        # Infer severity from keywords
        severity = "info"
        lower = f"{title} {description}".lower()
        if any(kw in lower for kw in ["drop", "loss", "below", "critical", "fail", "reject", "deficit"]):
            severity = "warning"

        insights.append(Insight(
            id=str(uuid.uuid4()),
            insight_type="dynamic_analysis",
            severity=severity,
            impact_score=50,
            title=f"{title} in {profile.table_name}",
            description=description,
            source_tables=[profile.table_name],
            source_columns=[],
            evidence={
                "module": "dynamic_analysis",
                "generated_code": generated_code,
                "pattern_type": "llm_generated",
            },
            suggested_actions=[],
        ))

    return insights
