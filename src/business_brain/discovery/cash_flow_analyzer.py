"""Cash flow analysis and forecasting.

Pure functions for analyzing cash flows, computing working capital,
burn rates, and forecasting cash positions from tabular business data.
No DB, async, or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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


def _parse_date(val) -> datetime | None:
    """Attempt to parse a date from string or datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(str(val), fmt)
        except (ValueError, TypeError):
            continue
    return None


def _month_key(dt: datetime) -> str:
    """Return 'YYYY-MM' string for a datetime."""
    return dt.strftime("%Y-%m")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PeriodFlow:
    """Cash flow for a single period."""
    period: str
    inflow: float
    outflow: float
    net_flow: float


@dataclass
class CategoryFlow:
    """Cash flow for a single category."""
    category: str
    total_inflow: float
    total_outflow: float
    net_flow: float


@dataclass
class CashFlowResult:
    """Complete cash flow analysis result."""
    total_inflow: float
    total_outflow: float
    net_flow: float
    cash_flow_ratio: float
    negative_periods_count: int
    period_flows: list[PeriodFlow] | None
    category_flows: list[CategoryFlow] | None
    summary: str


@dataclass
class PeriodWC:
    """Working capital for a single period."""
    period: str
    receivables: float
    payables: float
    inventory: float
    working_capital: float


@dataclass
class WorkingCapitalResult:
    """Complete working capital analysis result."""
    avg_working_capital: float
    min_wc: float
    max_wc: float
    health: str
    periods: list[PeriodWC]
    summary: str


@dataclass
class MonthlyExpense:
    """Expense for a single month."""
    month: str
    amount: float
    is_highest: bool


@dataclass
class BurnRateResult:
    """Burn rate analysis result."""
    gross_burn_rate: float
    net_burn_rate: float | None
    months_analyzed: int
    total_expenses: float
    monthly_expenses: list[MonthlyExpense]
    trend: str
    summary: str


@dataclass
class ProjectedPeriod:
    """A single projected future period."""
    period_label: str
    projected_inflow: float
    projected_outflow: float
    projected_net: float


@dataclass
class CashForecastResult:
    """Complete cash forecast result."""
    current_net_monthly: float
    trend_direction: str
    trend_magnitude: float
    projections: list[ProjectedPeriod]
    confidence: str
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_cash_flow
# ---------------------------------------------------------------------------


def analyze_cash_flow(
    rows: list[dict],
    inflow_column: str,
    outflow_column: str,
    date_column: str | None = None,
    category_column: str | None = None,
) -> CashFlowResult | None:
    """Analyze cash inflows and outflows.

    Computes total inflows, total outflows, net cash flow, and cash flow ratio.
    If date_column: compute monthly/period-level cash flow.
    If category_column: breakdown by category.

    Args:
        rows: Data rows as dicts.
        inflow_column: Column with inflow amounts.
        outflow_column: Column with outflow amounts.
        date_column: Optional column with date values for period analysis.
        category_column: Optional column for category breakdown.

    Returns:
        CashFlowResult or None if insufficient data.
    """
    if not rows:
        return None

    total_inflow = 0.0
    total_outflow = 0.0
    valid_count = 0

    # For period analysis
    period_data: dict[str, dict[str, float]] = {}
    # For category analysis
    category_data: dict[str, dict[str, float]] = {}

    for row in rows:
        inf = _safe_float(row.get(inflow_column))
        out = _safe_float(row.get(outflow_column))
        if inf is None or out is None:
            continue

        valid_count += 1
        total_inflow += inf
        total_outflow += out

        # Period grouping
        if date_column is not None:
            dt = _parse_date(row.get(date_column))
            if dt is not None:
                mk = _month_key(dt)
                if mk not in period_data:
                    period_data[mk] = {"inflow": 0.0, "outflow": 0.0}
                period_data[mk]["inflow"] += inf
                period_data[mk]["outflow"] += out

        # Category grouping
        if category_column is not None:
            cat = row.get(category_column)
            if cat is not None:
                cat_key = str(cat)
                if cat_key not in category_data:
                    category_data[cat_key] = {"inflow": 0.0, "outflow": 0.0}
                category_data[cat_key]["inflow"] += inf
                category_data[cat_key]["outflow"] += out

    if valid_count == 0:
        return None

    net_flow = total_inflow - total_outflow
    cash_flow_ratio = round(total_inflow / total_outflow, 4) if total_outflow != 0 else 0.0

    # Period flows
    period_flows: list[PeriodFlow] | None = None
    negative_periods_count = 0
    if date_column is not None and period_data:
        period_flows = []
        for period_key in sorted(period_data.keys()):
            vals = period_data[period_key]
            p_net = vals["inflow"] - vals["outflow"]
            if p_net < 0:
                negative_periods_count += 1
            period_flows.append(PeriodFlow(
                period=period_key,
                inflow=round(vals["inflow"], 2),
                outflow=round(vals["outflow"], 2),
                net_flow=round(p_net, 2),
            ))
    elif date_column is None:
        # Without date_column, check overall net flow for negative count
        if net_flow < 0:
            negative_periods_count = 1

    # Category flows
    category_flows: list[CategoryFlow] | None = None
    if category_column is not None and category_data:
        category_flows = []
        for cat_key in sorted(category_data.keys()):
            vals = category_data[cat_key]
            c_net = vals["inflow"] - vals["outflow"]
            category_flows.append(CategoryFlow(
                category=cat_key,
                total_inflow=round(vals["inflow"], 2),
                total_outflow=round(vals["outflow"], 2),
                net_flow=round(c_net, 2),
            ))

    summary = (
        f"Cash flow analysis: total inflow={total_inflow:,.2f}, "
        f"total outflow={total_outflow:,.2f}, net={net_flow:,.2f}. "
        f"Cash flow ratio: {cash_flow_ratio}. "
        f"Negative periods: {negative_periods_count}."
    )

    return CashFlowResult(
        total_inflow=round(total_inflow, 2),
        total_outflow=round(total_outflow, 2),
        net_flow=round(net_flow, 2),
        cash_flow_ratio=cash_flow_ratio,
        negative_periods_count=negative_periods_count,
        period_flows=period_flows,
        category_flows=category_flows,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. compute_working_capital
# ---------------------------------------------------------------------------


def compute_working_capital(
    rows: list[dict],
    receivables_column: str,
    payables_column: str,
    inventory_column: str | None = None,
    period_column: str | None = None,
) -> WorkingCapitalResult | None:
    """Compute working capital metrics.

    Working capital = receivables - payables (+ inventory if provided).
    If period_column: compute per-period working capital trend.

    Args:
        rows: Data rows as dicts.
        receivables_column: Column with receivables.
        payables_column: Column with payables.
        inventory_column: Optional column with inventory.
        period_column: Optional column for period grouping.

    Returns:
        WorkingCapitalResult or None if insufficient data.
    """
    if not rows:
        return None

    # Group by period (or single "Overall")
    period_data: dict[str, dict[str, float]] = {}

    for row in rows:
        rec = _safe_float(row.get(receivables_column))
        pay = _safe_float(row.get(payables_column))
        if rec is None or pay is None:
            continue

        inv = 0.0
        if inventory_column is not None:
            inv_val = _safe_float(row.get(inventory_column))
            if inv_val is not None:
                inv = inv_val

        period_key = str(row.get(period_column, "Overall")) if period_column else "Overall"

        if period_key not in period_data:
            period_data[period_key] = {"receivables": 0.0, "payables": 0.0, "inventory": 0.0}
        period_data[period_key]["receivables"] += rec
        period_data[period_key]["payables"] += pay
        period_data[period_key]["inventory"] += inv

    if not period_data:
        return None

    periods: list[PeriodWC] = []
    wc_values: list[float] = []

    for pk in sorted(period_data.keys()):
        vals = period_data[pk]
        wc = vals["receivables"] - vals["payables"] + vals["inventory"]
        wc_values.append(wc)
        periods.append(PeriodWC(
            period=pk,
            receivables=round(vals["receivables"], 2),
            payables=round(vals["payables"], 2),
            inventory=round(vals["inventory"], 2),
            working_capital=round(wc, 2),
        ))

    avg_wc = sum(wc_values) / len(wc_values)
    min_wc = min(wc_values)
    max_wc = max(wc_values)

    # Classify health
    total_receivables = sum(v["receivables"] for v in period_data.values())
    avg_receivables = total_receivables / len(period_data) if period_data else 1.0
    if avg_receivables == 0:
        avg_receivables = 1.0  # avoid division by zero

    if avg_wc > 0:
        health = "Healthy"
    elif avg_wc >= -0.10 * avg_receivables:
        health = "Strained"
    else:
        health = "Critical"

    summary = (
        f"Working capital analysis: avg={avg_wc:,.2f}, "
        f"min={min_wc:,.2f}, max={max_wc:,.2f}. "
        f"Health: {health}."
    )

    return WorkingCapitalResult(
        avg_working_capital=round(avg_wc, 2),
        min_wc=round(min_wc, 2),
        max_wc=round(max_wc, 2),
        health=health,
        periods=periods,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. analyze_burn_rate
# ---------------------------------------------------------------------------


def analyze_burn_rate(
    rows: list[dict],
    expense_column: str,
    date_column: str,
    revenue_column: str | None = None,
) -> BurnRateResult | None:
    """Analyze monthly burn rate from expense data.

    Computes monthly burn rate (total expenses / number of months).
    If revenue_column: compute net burn rate (expenses - revenue per month).

    Args:
        rows: Data rows as dicts.
        expense_column: Column with expense amounts.
        date_column: Column with date values.
        revenue_column: Optional column with revenue amounts.

    Returns:
        BurnRateResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate by month
    monthly_expenses: dict[str, float] = {}
    monthly_revenue: dict[str, float] = {}

    for row in rows:
        exp = _safe_float(row.get(expense_column))
        dt = _parse_date(row.get(date_column))
        if exp is None or dt is None:
            continue
        mk = _month_key(dt)
        monthly_expenses[mk] = monthly_expenses.get(mk, 0.0) + exp

        if revenue_column is not None:
            rev = _safe_float(row.get(revenue_column))
            if rev is not None:
                monthly_revenue[mk] = monthly_revenue.get(mk, 0.0) + rev

    if not monthly_expenses:
        return None

    sorted_months = sorted(monthly_expenses.keys())
    months_analyzed = len(sorted_months)
    expense_values = [monthly_expenses[m] for m in sorted_months]
    total_expenses = sum(expense_values)

    # Gross burn rate = total expenses / number of months
    gross_burn_rate = total_expenses / months_analyzed

    # Net burn rate (expenses - revenue per month)
    net_burn_rate: float | None = None
    if revenue_column is not None and monthly_revenue:
        total_revenue = sum(monthly_revenue.values())
        net_burn_rate = round((total_expenses - total_revenue) / months_analyzed, 2)

    # Identify highest-expense month
    max_expense = max(expense_values)
    monthly_expense_list: list[MonthlyExpense] = []
    for m in sorted_months:
        amt = monthly_expenses[m]
        monthly_expense_list.append(MonthlyExpense(
            month=m,
            amount=round(amt, 2),
            is_highest=(amt == max_expense),
        ))

    # Trend detection: compare first half average vs second half average
    if months_analyzed >= 2:
        mid = months_analyzed // 2
        first_half_avg = sum(expense_values[:mid]) / mid
        second_half_avg = sum(expense_values[mid:]) / (months_analyzed - mid)

        if first_half_avg == 0:
            if second_half_avg > 0:
                trend = "increasing"
            else:
                trend = "stable"
        else:
            change_pct = (second_half_avg - first_half_avg) / abs(first_half_avg) * 100
            if change_pct > 10:
                trend = "increasing"
            elif change_pct < -10:
                trend = "decreasing"
            else:
                trend = "stable"
    else:
        trend = "stable"

    parts = [
        f"Burn rate analysis over {months_analyzed} months. ",
        f"Gross burn rate: {gross_burn_rate:,.2f}/month. ",
    ]
    if net_burn_rate is not None:
        parts.append(f"Net burn rate: {net_burn_rate:,.2f}/month. ")
    parts.append(f"Trend: {trend}.")

    summary = "".join(parts)

    return BurnRateResult(
        gross_burn_rate=round(gross_burn_rate, 2),
        net_burn_rate=net_burn_rate,
        months_analyzed=months_analyzed,
        total_expenses=round(total_expenses, 2),
        monthly_expenses=monthly_expense_list,
        trend=trend,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. forecast_cash_position
# ---------------------------------------------------------------------------


def forecast_cash_position(
    rows: list[dict],
    amount_column: str,
    date_column: str,
    type_column: str | None = None,
    periods_ahead: int = 3,
) -> CashForecastResult | None:
    """Forecast future cash position using linear projection.

    Simple linear projection based on historical monthly net flows.
    If type_column: classify rows as inflow/outflow based on keywords.
    Otherwise: positive amounts are inflows, negative are outflows.

    Args:
        rows: Data rows as dicts.
        amount_column: Column with amount values.
        date_column: Column with date values.
        type_column: Optional column to classify inflow/outflow by keywords.
        periods_ahead: Number of future periods to project.

    Returns:
        CashForecastResult or None if insufficient data.
    """
    if not rows or periods_ahead <= 0:
        return None

    inflow_keywords = {"in", "receipt", "revenue", "income", "credit"}

    # Aggregate monthly inflows and outflows
    monthly_inflows: dict[str, float] = {}
    monthly_outflows: dict[str, float] = {}

    for row in rows:
        amt = _safe_float(row.get(amount_column))
        dt = _parse_date(row.get(date_column))
        if amt is None or dt is None:
            continue

        mk = _month_key(dt)
        if mk not in monthly_inflows:
            monthly_inflows[mk] = 0.0
            monthly_outflows[mk] = 0.0

        if type_column is not None:
            type_val = str(row.get(type_column, "")).lower()
            is_inflow = any(kw in type_val for kw in inflow_keywords)
            if is_inflow:
                monthly_inflows[mk] += abs(amt)
            else:
                monthly_outflows[mk] += abs(amt)
        else:
            if amt >= 0:
                monthly_inflows[mk] += amt
            else:
                monthly_outflows[mk] += abs(amt)

    if not monthly_inflows:
        return None

    sorted_months = sorted(monthly_inflows.keys())
    n = len(sorted_months)

    inflow_values = [monthly_inflows[m] for m in sorted_months]
    outflow_values = [monthly_outflows[m] for m in sorted_months]
    net_values = [monthly_inflows[m] - monthly_outflows[m] for m in sorted_months]

    current_net_monthly = sum(net_values) / n

    # Linear regression on net values: y = intercept + slope * x
    if n >= 2:
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = sum(net_values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, net_values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        else:
            slope = 0.0
            intercept = y_mean
    else:
        slope = 0.0
        intercept = net_values[0]

    # Trend direction
    if slope > 0.01:
        trend_direction = "improving"
    elif slope < -0.01:
        trend_direction = "declining"
    else:
        trend_direction = "stable"

    trend_magnitude = round(abs(slope), 2)

    # Linear regression on inflows and outflows separately
    if n >= 2:
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n

        # Inflow regression
        in_mean = sum(inflow_values) / n
        in_num = sum((x - x_mean) * (y - in_mean) for x, y in zip(x_vals, inflow_values))
        in_slope = in_num / denominator if denominator != 0 else 0.0
        in_intercept = in_mean - in_slope * x_mean

        # Outflow regression
        out_mean = sum(outflow_values) / n
        out_num = sum((x - x_mean) * (y - out_mean) for x, y in zip(x_vals, outflow_values))
        out_slope = out_num / denominator if denominator != 0 else 0.0
        out_intercept = out_mean - out_slope * x_mean
    else:
        in_slope = 0.0
        in_intercept = inflow_values[0]
        out_slope = 0.0
        out_intercept = outflow_values[0]

    # R-squared for confidence
    if n >= 2:
        ss_res = sum((net_values[i] - (intercept + slope * i)) ** 2 for i in range(n))
        ss_tot = sum((y - y_mean) ** 2 for y in net_values)
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = 1.0  # perfect fit (all values the same)
    else:
        r_squared = 0.0

    if r_squared >= 0.7:
        confidence = "High"
    elif r_squared >= 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Generate projections
    last_month_str = sorted_months[-1]
    last_dt = datetime.strptime(last_month_str + "-01", "%Y-%m-%d")

    projections: list[ProjectedPeriod] = []
    for i in range(1, periods_ahead + 1):
        x_idx = n - 1 + i

        proj_inflow = max(in_intercept + in_slope * x_idx, 0.0)
        proj_outflow = max(out_intercept + out_slope * x_idx, 0.0)
        proj_net = proj_inflow - proj_outflow

        # Compute future month
        future_month = last_dt.month + i
        future_year = last_dt.year
        while future_month > 12:
            future_month -= 12
            future_year += 1
        period_label = f"{future_year:04d}-{future_month:02d}"

        projections.append(ProjectedPeriod(
            period_label=period_label,
            projected_inflow=round(proj_inflow, 2),
            projected_outflow=round(proj_outflow, 2),
            projected_net=round(proj_net, 2),
        ))

    summary = (
        f"Cash forecast: current avg net monthly={current_net_monthly:,.2f}. "
        f"Trend: {trend_direction} (magnitude={trend_magnitude}). "
        f"Confidence: {confidence}. "
        f"Projected {periods_ahead} periods ahead."
    )

    return CashForecastResult(
        current_net_monthly=round(current_net_monthly, 2),
        trend_direction=trend_direction,
        trend_magnitude=trend_magnitude,
        projections=projections,
        confidence=confidence,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_cash_flow_report
# ---------------------------------------------------------------------------


def format_cash_flow_report(
    cash_flow: CashFlowResult | None = None,
    working_capital: WorkingCapitalResult | None = None,
    burn_rate: BurnRateResult | None = None,
    forecast: CashForecastResult | None = None,
) -> str:
    """Combine cash flow analysis results into a text report.

    Args:
        cash_flow: Optional cash flow result.
        working_capital: Optional working capital result.
        burn_rate: Optional burn rate result.
        forecast: Optional forecast result.

    Returns:
        Combined text report string.
    """
    sections: list[str] = []

    if cash_flow is not None:
        lines = ["=== Cash Flow Analysis ==="]
        lines.append(f"Total Inflow: {cash_flow.total_inflow:,.2f}")
        lines.append(f"Total Outflow: {cash_flow.total_outflow:,.2f}")
        lines.append(f"Net Cash Flow: {cash_flow.net_flow:,.2f}")
        lines.append(f"Cash Flow Ratio: {cash_flow.cash_flow_ratio}")
        lines.append(f"Negative Periods: {cash_flow.negative_periods_count}")
        if cash_flow.period_flows:
            lines.append("Period Breakdown:")
            for pf in cash_flow.period_flows:
                lines.append(
                    f"  {pf.period}: inflow={pf.inflow:,.2f}, "
                    f"outflow={pf.outflow:,.2f}, net={pf.net_flow:,.2f}"
                )
        if cash_flow.category_flows:
            lines.append("Category Breakdown:")
            for cf in cash_flow.category_flows:
                lines.append(
                    f"  {cf.category}: inflow={cf.total_inflow:,.2f}, "
                    f"outflow={cf.total_outflow:,.2f}, net={cf.net_flow:,.2f}"
                )
        sections.append("\n".join(lines))

    if working_capital is not None:
        lines = ["=== Working Capital ==="]
        lines.append(f"Average Working Capital: {working_capital.avg_working_capital:,.2f}")
        lines.append(f"Min: {working_capital.min_wc:,.2f}")
        lines.append(f"Max: {working_capital.max_wc:,.2f}")
        lines.append(f"Health: {working_capital.health}")
        if working_capital.periods:
            lines.append("Period Details:")
            for pw in working_capital.periods:
                lines.append(
                    f"  {pw.period}: receivables={pw.receivables:,.2f}, "
                    f"payables={pw.payables:,.2f}, inventory={pw.inventory:,.2f}, "
                    f"WC={pw.working_capital:,.2f}"
                )
        sections.append("\n".join(lines))

    if burn_rate is not None:
        lines = ["=== Burn Rate ==="]
        lines.append(f"Gross Burn Rate: {burn_rate.gross_burn_rate:,.2f}/month")
        if burn_rate.net_burn_rate is not None:
            lines.append(f"Net Burn Rate: {burn_rate.net_burn_rate:,.2f}/month")
        lines.append(f"Months Analyzed: {burn_rate.months_analyzed}")
        lines.append(f"Total Expenses: {burn_rate.total_expenses:,.2f}")
        lines.append(f"Trend: {burn_rate.trend}")
        if burn_rate.monthly_expenses:
            lines.append("Monthly Details:")
            for me in burn_rate.monthly_expenses:
                flag = " [HIGHEST]" if me.is_highest else ""
                lines.append(f"  {me.month}: {me.amount:,.2f}{flag}")
        sections.append("\n".join(lines))

    if forecast is not None:
        lines = ["=== Cash Forecast ==="]
        lines.append(f"Current Net Monthly: {forecast.current_net_monthly:,.2f}")
        lines.append(f"Trend: {forecast.trend_direction} (magnitude={forecast.trend_magnitude})")
        lines.append(f"Confidence: {forecast.confidence}")
        if forecast.projections:
            lines.append("Projections:")
            for pp in forecast.projections:
                lines.append(
                    f"  {pp.period_label}: inflow={pp.projected_inflow:,.2f}, "
                    f"outflow={pp.projected_outflow:,.2f}, net={pp.projected_net:,.2f}"
                )
        sections.append("\n".join(lines))

    if not sections:
        return "No cash flow data available for report."

    return "\n\n".join(sections)
