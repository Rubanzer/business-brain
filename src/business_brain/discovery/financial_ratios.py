"""Financial ratios and metrics from business data.

Pure functions for computing profitability, liquidity, efficiency ratios,
and financial trend analysis from tabular business data.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class EntityProfit:
    """Profitability metrics for a single entity."""
    entity: str
    revenue: float
    cost: float
    gross_profit: float
    gross_margin_pct: float
    cost_ratio: float


@dataclass
class ProfitabilityResult:
    """Complete profitability analysis result."""
    entities: list[EntityProfit]
    total_revenue: float
    total_cost: float
    overall_margin: float
    summary: str


@dataclass
class EntityLiquidity:
    """Liquidity metrics for a single entity."""
    entity: str
    current_assets: float
    current_liabilities: float
    current_ratio: float
    quick_ratio: float | None
    cash_ratio: float | None
    rating: str


@dataclass
class LiquidityResult:
    """Complete liquidity analysis result."""
    entities: list[EntityLiquidity]
    avg_current_ratio: float
    summary: str


@dataclass
class EfficiencyRatios:
    """Efficiency ratio values."""
    asset_turnover: float
    receivables_turnover: float | None
    dso: float | None
    payables_turnover: float | None
    dpo: float | None
    cash_conversion_cycle: float | None


@dataclass
class EfficiencyResult:
    """Complete efficiency analysis result."""
    ratios: EfficiencyRatios
    summary: str


@dataclass
class PeriodMetric:
    """Metric value for a single period."""
    period: str
    value: float
    change: float | None
    change_pct: float | None


@dataclass
class EntityTrend:
    """Trend analysis for a single entity."""
    entity: str
    periods: list[PeriodMetric]
    trend: str
    cagr: float | None
    best_period: str
    worst_period: str


@dataclass
class FinancialTrendResult:
    """Complete financial trend analysis result."""
    entities: list[EntityTrend]
    overall_trend: str
    summary: str


# ---------------------------------------------------------------------------
# 1. Profitability Ratios
# ---------------------------------------------------------------------------


def compute_profitability_ratios(
    rows: list[dict],
    revenue_column: str,
    cost_column: str,
    entity_column: str | None = None,
) -> ProfitabilityResult | None:
    """Compute profitability ratios from rows.

    Args:
        rows: Data rows as dicts.
        revenue_column: Column with revenue values.
        cost_column: Column with cost values.
        entity_column: Optional column to group by entity.

    Returns:
        ProfitabilityResult or None if insufficient data.
    """
    if not rows:
        return None

    # Group by entity (or use a single "Overall" entity)
    entity_data: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        rev = _safe_float(row.get(revenue_column))
        cost = _safe_float(row.get(cost_column))
        if rev is None or cost is None:
            continue
        entity = str(row.get(entity_column, "Overall")) if entity_column else "Overall"
        entity_data.setdefault(entity, []).append((rev, cost))

    if not entity_data:
        return None

    entities: list[EntityProfit] = []
    total_revenue = 0.0
    total_cost = 0.0

    for entity_name in sorted(entity_data):
        pairs = entity_data[entity_name]
        ent_rev = sum(r for r, _ in pairs)
        ent_cost = sum(c for _, c in pairs)
        gross_profit = ent_rev - ent_cost

        if ent_rev == 0:
            gross_margin_pct = 0.0
            cost_ratio = 0.0
        else:
            gross_margin_pct = round(gross_profit / ent_rev * 100, 2)
            cost_ratio = round(ent_cost / ent_rev * 100, 2)

        entities.append(EntityProfit(
            entity=entity_name,
            revenue=round(ent_rev, 2),
            cost=round(ent_cost, 2),
            gross_profit=round(gross_profit, 2),
            gross_margin_pct=gross_margin_pct,
            cost_ratio=cost_ratio,
        ))
        total_revenue += ent_rev
        total_cost += ent_cost

    if total_revenue == 0:
        overall_margin = 0.0
    else:
        overall_margin = round((total_revenue - total_cost) / total_revenue * 100, 2)

    n_entities = len(entities)
    best = max(entities, key=lambda e: e.gross_margin_pct)
    summary = (
        f"Profitability analysis across {n_entities} "
        f"{'entity' if n_entities == 1 else 'entities'}. "
        f"Overall margin: {overall_margin}%. "
        f"Best margin: {best.entity} at {best.gross_margin_pct}%."
    )

    return ProfitabilityResult(
        entities=entities,
        total_revenue=round(total_revenue, 2),
        total_cost=round(total_cost, 2),
        overall_margin=overall_margin,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. Liquidity Ratios
# ---------------------------------------------------------------------------


def _liquidity_rating(current_ratio: float) -> str:
    """Rate liquidity based on current ratio."""
    if current_ratio > 2:
        return "Strong"
    if current_ratio > 1:
        return "Adequate"
    return "Weak"


def compute_liquidity_ratios(
    rows: list[dict],
    current_assets_column: str,
    current_liabilities_column: str,
    cash_column: str | None = None,
    inventory_column: str | None = None,
    entity_column: str | None = None,
) -> LiquidityResult | None:
    """Compute liquidity ratios from rows.

    Args:
        rows: Data rows as dicts.
        current_assets_column: Column with current assets.
        current_liabilities_column: Column with current liabilities.
        cash_column: Optional column with cash values.
        inventory_column: Optional column with inventory values.
        entity_column: Optional column to group by entity.

    Returns:
        LiquidityResult or None if insufficient data.
    """
    if not rows:
        return None

    # Group by entity
    entity_data: dict[str, list[dict[str, float]]] = {}
    for row in rows:
        ca = _safe_float(row.get(current_assets_column))
        cl = _safe_float(row.get(current_liabilities_column))
        if ca is None or cl is None:
            continue

        entry: dict[str, float] = {"ca": ca, "cl": cl}
        if cash_column is not None:
            cash = _safe_float(row.get(cash_column))
            if cash is not None:
                entry["cash"] = cash
        if inventory_column is not None:
            inv = _safe_float(row.get(inventory_column))
            if inv is not None:
                entry["inventory"] = inv

        entity = str(row.get(entity_column, "Overall")) if entity_column else "Overall"
        entity_data.setdefault(entity, []).append(entry)

    if not entity_data:
        return None

    entities: list[EntityLiquidity] = []
    for entity_name in sorted(entity_data):
        entries = entity_data[entity_name]
        total_ca = sum(e["ca"] for e in entries)
        total_cl = sum(e["cl"] for e in entries)

        if total_cl == 0:
            current_ratio = 0.0
        else:
            current_ratio = round(total_ca / total_cl, 4)

        # Quick ratio
        quick_ratio: float | None = None
        if inventory_column is not None:
            total_inv = sum(e.get("inventory", 0.0) for e in entries)
            if total_cl != 0:
                quick_ratio = round((total_ca - total_inv) / total_cl, 4)
            else:
                quick_ratio = 0.0

        # Cash ratio
        cash_ratio: float | None = None
        if cash_column is not None:
            total_cash = sum(e.get("cash", 0.0) for e in entries)
            if total_cl != 0:
                cash_ratio = round(total_cash / total_cl, 4)
            else:
                cash_ratio = 0.0

        rating = _liquidity_rating(current_ratio)

        entities.append(EntityLiquidity(
            entity=entity_name,
            current_assets=round(total_ca, 2),
            current_liabilities=round(total_cl, 2),
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            cash_ratio=cash_ratio,
            rating=rating,
        ))

    ratios = [e.current_ratio for e in entities]
    avg_current_ratio = round(sum(ratios) / len(ratios), 4) if ratios else 0.0

    n_entities = len(entities)
    strong = sum(1 for e in entities if e.rating == "Strong")
    adequate = sum(1 for e in entities if e.rating == "Adequate")
    weak = sum(1 for e in entities if e.rating == "Weak")
    summary = (
        f"Liquidity analysis across {n_entities} "
        f"{'entity' if n_entities == 1 else 'entities'}. "
        f"Average current ratio: {avg_current_ratio}. "
        f"Ratings: {strong} Strong, {adequate} Adequate, {weak} Weak."
    )

    return LiquidityResult(
        entities=entities,
        avg_current_ratio=avg_current_ratio,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Efficiency Ratios
# ---------------------------------------------------------------------------


def compute_efficiency_ratios(
    rows: list[dict],
    revenue_column: str,
    assets_column: str,
    receivables_column: str | None = None,
    payables_column: str | None = None,
    cogs_column: str | None = None,
) -> EfficiencyResult | None:
    """Compute efficiency ratios from rows.

    Args:
        rows: Data rows as dicts.
        revenue_column: Column with revenue values.
        assets_column: Column with total assets values.
        receivables_column: Optional column with receivables.
        payables_column: Optional column with payables.
        cogs_column: Optional column with cost of goods sold.

    Returns:
        EfficiencyResult or None if insufficient data.
    """
    if not rows:
        return None

    total_revenue = 0.0
    total_assets = 0.0
    total_receivables = 0.0
    total_payables = 0.0
    total_cogs = 0.0
    has_receivables = receivables_column is not None
    has_payables = payables_column is not None and cogs_column is not None
    valid_count = 0

    for row in rows:
        rev = _safe_float(row.get(revenue_column))
        assets = _safe_float(row.get(assets_column))
        if rev is None or assets is None:
            continue
        valid_count += 1
        total_revenue += rev
        total_assets += assets

        if has_receivables:
            rec = _safe_float(row.get(receivables_column))
            if rec is not None:
                total_receivables += rec
        if has_payables:
            pay = _safe_float(row.get(payables_column))
            cogs = _safe_float(row.get(cogs_column))
            if pay is not None:
                total_payables += pay
            if cogs is not None:
                total_cogs += cogs

    if valid_count == 0 or total_assets == 0:
        return None

    asset_turnover = round(total_revenue / total_assets, 4)

    # Receivables turnover and DSO
    receivables_turnover: float | None = None
    dso: float | None = None
    if has_receivables and total_receivables > 0:
        receivables_turnover = round(total_revenue / total_receivables, 4)
        dso = round(365.0 / receivables_turnover, 2)

    # Payables turnover and DPO
    payables_turnover: float | None = None
    dpo: float | None = None
    if has_payables and total_payables > 0 and total_cogs > 0:
        payables_turnover = round(total_cogs / total_payables, 4)
        dpo = round(365.0 / payables_turnover, 2)

    # Cash conversion cycle
    cash_conversion_cycle: float | None = None
    if dso is not None and dpo is not None:
        cash_conversion_cycle = round(dso - dpo, 2)

    ratios = EfficiencyRatios(
        asset_turnover=asset_turnover,
        receivables_turnover=receivables_turnover,
        dso=dso,
        payables_turnover=payables_turnover,
        dpo=dpo,
        cash_conversion_cycle=cash_conversion_cycle,
    )

    parts = [f"Asset turnover: {asset_turnover}x"]
    if dso is not None:
        parts.append(f"DSO: {dso} days")
    if dpo is not None:
        parts.append(f"DPO: {dpo} days")
    if cash_conversion_cycle is not None:
        parts.append(f"Cash conversion cycle: {cash_conversion_cycle} days")
    summary = "Efficiency analysis. " + ". ".join(parts) + "."

    return EfficiencyResult(ratios=ratios, summary=summary)


# ---------------------------------------------------------------------------
# 4. Financial Trend Analysis
# ---------------------------------------------------------------------------


def analyze_financial_trends(
    rows: list[dict],
    metric_column: str,
    period_column: str,
    entity_column: str | None = None,
) -> FinancialTrendResult | None:
    """Analyze financial trends over periods.

    Args:
        rows: Data rows as dicts.
        metric_column: Numeric column to track.
        period_column: Column defining time periods.
        entity_column: Optional column to group by entity.

    Returns:
        FinancialTrendResult or None if insufficient data.
    """
    if not rows:
        return None

    # Group by entity and period
    entity_period_data: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        val = _safe_float(row.get(metric_column))
        period = row.get(period_column)
        if val is None or period is None:
            continue
        entity = str(row.get(entity_column, "Overall")) if entity_column else "Overall"
        entity_period_data.setdefault(entity, {}).setdefault(str(period), []).append(val)

    if not entity_period_data:
        return None

    entities: list[EntityTrend] = []
    for entity_name in sorted(entity_period_data):
        period_map = entity_period_data[entity_name]
        if len(period_map) < 2:
            continue

        sorted_periods = sorted(period_map.keys())
        period_metrics: list[PeriodMetric] = []

        for i, p in enumerate(sorted_periods):
            value = round(sum(period_map[p]), 4)
            if i == 0:
                change = None
                change_pct = None
            else:
                prev_value = period_metrics[i - 1].value
                change = round(value - prev_value, 4)
                if prev_value != 0:
                    change_pct = round(change / abs(prev_value) * 100, 2)
                else:
                    change_pct = None
            period_metrics.append(PeriodMetric(
                period=p,
                value=value,
                change=change,
                change_pct=change_pct,
            ))

        # CAGR if 3+ periods
        cagr: float | None = None
        if len(period_metrics) >= 3:
            first_val = period_metrics[0].value
            last_val = period_metrics[-1].value
            n_periods = len(period_metrics) - 1
            if first_val > 0 and last_val > 0:
                cagr = round(((last_val / first_val) ** (1.0 / n_periods) - 1) * 100, 2)

        # Trend direction
        changes = [pm.change for pm in period_metrics if pm.change is not None]
        if changes:
            positive = sum(1 for c in changes if c > 0)
            negative = sum(1 for c in changes if c < 0)
            if positive > negative:
                trend = "improving"
            elif negative > positive:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Best and worst periods
        best_pm = max(period_metrics, key=lambda pm: pm.value)
        worst_pm = min(period_metrics, key=lambda pm: pm.value)

        entities.append(EntityTrend(
            entity=entity_name,
            periods=period_metrics,
            trend=trend,
            cagr=cagr,
            best_period=best_pm.period,
            worst_period=worst_pm.period,
        ))

    if not entities:
        return None

    # Overall trend
    trend_counts = {"improving": 0, "declining": 0, "stable": 0}
    for e in entities:
        trend_counts[e.trend] += 1
    overall_trend = max(trend_counts, key=trend_counts.get)  # type: ignore[arg-type]

    n_entities = len(entities)
    summary = (
        f"Financial trend analysis across {n_entities} "
        f"{'entity' if n_entities == 1 else 'entities'}. "
        f"Overall trend: {overall_trend}. "
        f"Improving: {trend_counts['improving']}, "
        f"Declining: {trend_counts['declining']}, "
        f"Stable: {trend_counts['stable']}."
    )

    return FinancialTrendResult(
        entities=entities,
        overall_trend=overall_trend,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. Format Financial Report
# ---------------------------------------------------------------------------


def format_financial_report(
    profitability: ProfitabilityResult | None = None,
    liquidity: LiquidityResult | None = None,
    efficiency: EfficiencyResult | None = None,
    trends: FinancialTrendResult | None = None,
) -> str:
    """Combine financial analysis results into a text report.

    Args:
        profitability: Optional profitability result.
        liquidity: Optional liquidity result.
        efficiency: Optional efficiency result.
        trends: Optional trend result.

    Returns:
        Combined text report string.
    """
    sections: list[str] = []

    if profitability is not None:
        lines = ["=== Profitability ==="]
        lines.append(f"Total Revenue: {profitability.total_revenue:,.2f}")
        lines.append(f"Total Cost: {profitability.total_cost:,.2f}")
        lines.append(f"Overall Margin: {profitability.overall_margin}%")
        for ep in profitability.entities:
            lines.append(
                f"  {ep.entity}: revenue={ep.revenue:,.2f}, "
                f"cost={ep.cost:,.2f}, margin={ep.gross_margin_pct}%"
            )
        sections.append("\n".join(lines))

    if liquidity is not None:
        lines = ["=== Liquidity ==="]
        lines.append(f"Average Current Ratio: {liquidity.avg_current_ratio}")
        for el in liquidity.entities:
            parts = [
                f"  {el.entity}: current_ratio={el.current_ratio}",
                f"rating={el.rating}",
            ]
            if el.quick_ratio is not None:
                parts.append(f"quick_ratio={el.quick_ratio}")
            if el.cash_ratio is not None:
                parts.append(f"cash_ratio={el.cash_ratio}")
            lines.append(", ".join(parts))
        sections.append("\n".join(lines))

    if efficiency is not None:
        lines = ["=== Efficiency ==="]
        r = efficiency.ratios
        lines.append(f"Asset Turnover: {r.asset_turnover}x")
        if r.receivables_turnover is not None:
            lines.append(f"Receivables Turnover: {r.receivables_turnover}x")
        if r.dso is not None:
            lines.append(f"Days Sales Outstanding: {r.dso}")
        if r.payables_turnover is not None:
            lines.append(f"Payables Turnover: {r.payables_turnover}x")
        if r.dpo is not None:
            lines.append(f"Days Payable Outstanding: {r.dpo}")
        if r.cash_conversion_cycle is not None:
            lines.append(f"Cash Conversion Cycle: {r.cash_conversion_cycle} days")
        sections.append("\n".join(lines))

    if trends is not None:
        lines = ["=== Financial Trends ==="]
        lines.append(f"Overall Trend: {trends.overall_trend}")
        for et in trends.entities:
            line = f"  {et.entity}: trend={et.trend}"
            if et.cagr is not None:
                line += f", CAGR={et.cagr}%"
            line += f", best={et.best_period}, worst={et.worst_period}"
            lines.append(line)
        sections.append("\n".join(lines))

    if not sections:
        return "No financial data available for report."

    return "\n\n".join(sections)
