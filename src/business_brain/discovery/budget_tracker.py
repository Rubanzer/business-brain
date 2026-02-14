"""Budget tracking and financial planning.

Pure functions for analyzing budget vs actual spending, computing burn rates,
analyzing spending patterns, forecasting budgets, and generating reports.
No DB, async, or LLM dependencies.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Convert a value to float safely, returning None on failure."""
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
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CategoryVariance:
    """Budget vs actual for a single category."""

    category: str
    budget: float
    actual: float
    variance: float  # actual - budget
    variance_pct: float
    over_budget: bool


@dataclass
class PeriodVariance:
    """Per-period variance breakdown."""

    period: str
    budget: float
    actual: float
    variance: float
    variance_pct: float


@dataclass
class BudgetResult:
    """Complete budget vs actual analysis result."""

    categories: list[CategoryVariance]
    total_budget: float
    total_actual: float
    overall_variance: float
    overall_variance_pct: float
    over_budget_count: int
    under_budget_count: int
    periods: list[PeriodVariance] | None
    summary: str


@dataclass
class BurnRateResult:
    """Burn rate computation result."""

    total_spend: float
    num_days: int
    daily_burn_rate: float
    monthly_burn_rate: float
    remaining_budget: float | None
    days_until_exhaustion: float | None
    projected_end_date: str | None
    trend: str  # "accelerating", "decelerating", "stable"
    first_half_rate: float
    second_half_rate: float
    summary: str


@dataclass
class CategorySpend:
    """Spending in a single category."""

    category: str
    amount: float
    pct_of_total: float


@dataclass
class VendorSpend:
    """Spending from a single vendor."""

    vendor: str
    amount: float
    pct_of_total: float


@dataclass
class MonthChange:
    """Month-over-month spending change for a category."""

    category: str
    month: str
    amount: float
    prev_amount: float | None
    change: float | None
    change_pct: float | None


@dataclass
class SpendingResult:
    """Spending pattern analysis result."""

    category_breakdown: list[CategorySpend]
    total_spend: float
    top_categories: list[CategorySpend]
    month_over_month: list[MonthChange] | None
    top_vendors: list[VendorSpend] | None
    summary: str


@dataclass
class ForecastEntry:
    """A single forecasted future period."""

    period: str
    projected_amount: float
    cumulative: float
    confidence: str  # "high", "moderate", "low"


# ---------------------------------------------------------------------------
# 1. budget_vs_actual
# ---------------------------------------------------------------------------


def budget_vs_actual(
    rows: list[dict],
    category_column: str,
    budget_column: str,
    actual_column: str,
    period_column: str | None = None,
) -> BudgetResult | None:
    """Analyze budget versus actual spending per category.

    Per category: budget, actual, variance (actual - budget), variance_pct.
    Flags over-budget items (actual > budget).
    Overall: total budget, total actual, overall variance pct.
    If period_column: per-period breakdown of variance.

    Args:
        rows: Data rows as dicts.
        category_column: Column identifying the budget category.
        budget_column: Column with budgeted amount.
        actual_column: Column with actual amount.
        period_column: Optional column for time period grouping.

    Returns:
        BudgetResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate by category
    cat_agg: dict[str, dict[str, float]] = {}
    for row in rows:
        cat = row.get(category_column)
        bgt = _safe_float(row.get(budget_column))
        act = _safe_float(row.get(actual_column))
        if cat is None or bgt is None or act is None:
            continue
        key = str(cat)
        if key not in cat_agg:
            cat_agg[key] = {"budget": 0.0, "actual": 0.0}
        cat_agg[key]["budget"] += bgt
        cat_agg[key]["actual"] += act

    if not cat_agg:
        return None

    categories: list[CategoryVariance] = []
    for name, vals in sorted(cat_agg.items(), key=lambda x: -(x[1]["actual"] - x[1]["budget"])):
        bgt_val = vals["budget"]
        act_val = vals["actual"]
        variance = act_val - bgt_val
        variance_pct = (variance / abs(bgt_val) * 100) if bgt_val != 0 else 0.0
        over = act_val > bgt_val
        categories.append(
            CategoryVariance(
                category=name,
                budget=round(bgt_val, 4),
                actual=round(act_val, 4),
                variance=round(variance, 4),
                variance_pct=round(variance_pct, 2),
                over_budget=over,
            )
        )

    total_budget = sum(v["budget"] for v in cat_agg.values())
    total_actual = sum(v["actual"] for v in cat_agg.values())
    overall_variance = total_actual - total_budget
    overall_variance_pct = (
        (overall_variance / abs(total_budget) * 100) if total_budget != 0 else 0.0
    )

    over_budget_count = sum(1 for c in categories if c.over_budget)
    under_budget_count = len(categories) - over_budget_count

    # Per-period breakdown
    periods: list[PeriodVariance] | None = None
    if period_column is not None:
        period_agg: dict[str, dict[str, float]] = {}
        for row in rows:
            period = row.get(period_column)
            bgt = _safe_float(row.get(budget_column))
            act = _safe_float(row.get(actual_column))
            if period is None or bgt is None or act is None:
                continue
            key = str(period)
            if key not in period_agg:
                period_agg[key] = {"budget": 0.0, "actual": 0.0}
            period_agg[key]["budget"] += bgt
            period_agg[key]["actual"] += act

        if period_agg:
            periods = []
            for pname in sorted(period_agg.keys()):
                vals = period_agg[pname]
                var = vals["actual"] - vals["budget"]
                var_pct = (var / abs(vals["budget"]) * 100) if vals["budget"] != 0 else 0.0
                periods.append(
                    PeriodVariance(
                        period=pname,
                        budget=round(vals["budget"], 4),
                        actual=round(vals["actual"], 4),
                        variance=round(var, 4),
                        variance_pct=round(var_pct, 2),
                    )
                )

    summary = (
        f"Budget analysis across {len(categories)} categories: "
        f"Total budget = {total_budget:,.2f}, Total actual = {total_actual:,.2f}. "
        f"Overall variance = {overall_variance:+,.2f} ({overall_variance_pct:+.1f}%). "
        f"Over-budget: {over_budget_count}, Under-budget: {under_budget_count}."
    )

    return BudgetResult(
        categories=categories,
        total_budget=round(total_budget, 4),
        total_actual=round(total_actual, 4),
        overall_variance=round(overall_variance, 4),
        overall_variance_pct=round(overall_variance_pct, 2),
        over_budget_count=over_budget_count,
        under_budget_count=under_budget_count,
        periods=periods,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. compute_burn_rate
# ---------------------------------------------------------------------------


def compute_burn_rate(
    rows: list[dict],
    amount_column: str,
    date_column: str,
    total_budget: float | None = None,
) -> BurnRateResult | None:
    """Compute daily/monthly burn rate from spending data.

    Calculates daily and monthly burn rate (total spend / number of days).
    If total_budget: remaining budget, days_until_exhaustion, projected_end_date.
    Trend: compares first half vs second half burn rate.

    Args:
        rows: Data rows as dicts.
        amount_column: Column with spend amount.
        date_column: Column with date value.
        total_budget: Optional total budget to compute remaining and projections.

    Returns:
        BurnRateResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect valid (date, amount) pairs
    entries: list[tuple[datetime, float]] = []
    for row in rows:
        amt = _safe_float(row.get(amount_column))
        dt = _parse_date(row.get(date_column))
        if amt is None or dt is None:
            continue
        entries.append((dt, amt))

    if not entries:
        return None

    entries.sort(key=lambda x: x[0])
    total_spend = sum(amt for _, amt in entries)

    min_date = entries[0][0]
    max_date = entries[-1][0]
    num_days = max((max_date - min_date).days, 1)

    daily_burn = total_spend / num_days
    monthly_burn = daily_burn * 30

    # Trend: compare first half vs second half
    mid = len(entries) // 2
    first_half = entries[:mid] if mid > 0 else entries
    second_half = entries[mid:] if mid > 0 else entries

    first_spend = sum(amt for _, amt in first_half)
    second_spend = sum(amt for _, amt in second_half)

    if len(first_half) > 0:
        first_days = max(
            (first_half[-1][0] - first_half[0][0]).days, 1
        )
    else:
        first_days = 1

    if len(second_half) > 0:
        second_days = max(
            (second_half[-1][0] - second_half[0][0]).days, 1
        )
    else:
        second_days = 1

    first_rate = first_spend / first_days
    second_rate = second_spend / second_days

    # Determine trend
    if first_rate == 0:
        trend = "accelerating" if second_rate > 0 else "stable"
    else:
        rate_change_pct = (second_rate - first_rate) / abs(first_rate) * 100
        if rate_change_pct > 10:
            trend = "accelerating"
        elif rate_change_pct < -10:
            trend = "decelerating"
        else:
            trend = "stable"

    # Budget projections
    remaining_budget: float | None = None
    days_until_exhaustion: float | None = None
    projected_end_date: str | None = None

    if total_budget is not None:
        remaining_budget = total_budget - total_spend
        if daily_burn > 0 and remaining_budget > 0:
            days_until_exhaustion = remaining_budget / daily_burn
            projected_end = max_date + timedelta(days=days_until_exhaustion)
            projected_end_date = projected_end.strftime("%Y-%m-%d")
        elif remaining_budget <= 0:
            days_until_exhaustion = 0.0
            projected_end_date = max_date.strftime("%Y-%m-%d")

    # Summary
    parts = [
        f"Burn rate: {daily_burn:,.2f}/day ({monthly_burn:,.2f}/month). ",
        f"Total spend: {total_spend:,.2f} over {num_days} days. ",
        f"Trend: {trend}.",
    ]
    if remaining_budget is not None:
        parts.append(f" Remaining budget: {remaining_budget:,.2f}.")
    if days_until_exhaustion is not None:
        parts.append(f" Days until exhaustion: {days_until_exhaustion:,.0f}.")

    return BurnRateResult(
        total_spend=round(total_spend, 4),
        num_days=num_days,
        daily_burn_rate=round(daily_burn, 4),
        monthly_burn_rate=round(monthly_burn, 4),
        remaining_budget=round(remaining_budget, 4) if remaining_budget is not None else None,
        days_until_exhaustion=(
            round(days_until_exhaustion, 2) if days_until_exhaustion is not None else None
        ),
        projected_end_date=projected_end_date,
        trend=trend,
        first_half_rate=round(first_rate, 4),
        second_half_rate=round(second_rate, 4),
        summary="".join(parts),
    )


# ---------------------------------------------------------------------------
# 3. analyze_spending_patterns
# ---------------------------------------------------------------------------


def analyze_spending_patterns(
    rows: list[dict],
    category_column: str,
    amount_column: str,
    date_column: str | None = None,
    vendor_column: str | None = None,
) -> SpendingResult | None:
    """Analyze spending patterns by category, time, and vendor.

    Category-wise spend breakdown (amount, pct of total).
    Top 5 categories by spend.
    If date_column: month-over-month change per category.
    If vendor_column: top 5 vendors by spend.

    Args:
        rows: Data rows as dicts.
        category_column: Column identifying the spending category.
        amount_column: Column with spend amount.
        date_column: Optional column with date for MoM analysis.
        vendor_column: Optional column with vendor name.

    Returns:
        SpendingResult or None if no valid data.
    """
    if not rows:
        return None

    # Category aggregation
    cat_totals: dict[str, float] = {}
    for row in rows:
        cat = row.get(category_column)
        amt = _safe_float(row.get(amount_column))
        if cat is None or amt is None:
            continue
        key = str(cat)
        cat_totals[key] = cat_totals.get(key, 0.0) + amt

    if not cat_totals:
        return None

    grand_total = sum(cat_totals.values())
    if grand_total == 0:
        return None

    # Build sorted category breakdown
    sorted_cats = sorted(cat_totals.items(), key=lambda x: -x[1])
    category_breakdown: list[CategorySpend] = []
    for name, amount in sorted_cats:
        pct = amount / grand_total * 100
        category_breakdown.append(
            CategorySpend(
                category=name,
                amount=round(amount, 4),
                pct_of_total=round(pct, 2),
            )
        )

    top_categories = category_breakdown[:5]

    # Month-over-month analysis
    month_over_month: list[MonthChange] | None = None
    if date_column is not None:
        # Collect (category, month) -> amount
        cat_month: dict[str, dict[str, float]] = {}
        for row in rows:
            cat = row.get(category_column)
            amt = _safe_float(row.get(amount_column))
            dt = _parse_date(row.get(date_column))
            if cat is None or amt is None or dt is None:
                continue
            key = str(cat)
            mk = _month_key(dt)
            if key not in cat_month:
                cat_month[key] = {}
            cat_month[key][mk] = cat_month[key].get(mk, 0.0) + amt

        if cat_month:
            # Get sorted list of all months
            all_months = sorted({m for months in cat_month.values() for m in months})
            mom_entries: list[MonthChange] = []
            for cat_name in sorted(cat_month.keys()):
                months_data = cat_month[cat_name]
                prev_amt: float | None = None
                for month in all_months:
                    current_amt = months_data.get(month, 0.0)
                    change: float | None = None
                    change_pct: float | None = None
                    if prev_amt is not None:
                        change = current_amt - prev_amt
                        change_pct = (
                            (change / abs(prev_amt) * 100) if prev_amt != 0 else 0.0
                        )
                    mom_entries.append(
                        MonthChange(
                            category=cat_name,
                            month=month,
                            amount=round(current_amt, 4),
                            prev_amount=round(prev_amt, 4) if prev_amt is not None else None,
                            change=round(change, 4) if change is not None else None,
                            change_pct=round(change_pct, 2) if change_pct is not None else None,
                        )
                    )
                    prev_amt = current_amt
            month_over_month = mom_entries

    # Vendor analysis
    top_vendors: list[VendorSpend] | None = None
    if vendor_column is not None:
        vendor_totals: dict[str, float] = {}
        for row in rows:
            vendor = row.get(vendor_column)
            amt = _safe_float(row.get(amount_column))
            if vendor is None or amt is None:
                continue
            vkey = str(vendor)
            vendor_totals[vkey] = vendor_totals.get(vkey, 0.0) + amt

        if vendor_totals:
            sorted_vendors = sorted(vendor_totals.items(), key=lambda x: -x[1])[:5]
            top_vendors = [
                VendorSpend(
                    vendor=v,
                    amount=round(a, 4),
                    pct_of_total=round(a / grand_total * 100, 2),
                )
                for v, a in sorted_vendors
            ]

    top_names = ", ".join(c.category for c in top_categories[:3])
    summary = (
        f"Spending analysis across {len(category_breakdown)} categories. "
        f"Total spend: {grand_total:,.2f}. "
        f"Top categories: {top_names}."
    )

    return SpendingResult(
        category_breakdown=category_breakdown,
        total_spend=round(grand_total, 4),
        top_categories=top_categories,
        month_over_month=month_over_month,
        top_vendors=top_vendors,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. forecast_budget
# ---------------------------------------------------------------------------


def forecast_budget(
    rows: list[dict],
    amount_column: str,
    date_column: str,
    periods_ahead: int = 3,
) -> list[ForecastEntry]:
    """Forecast future budget periods using linear projection.

    Simple linear projection based on historical monthly totals.
    For each future period: projected amount, cumulative.
    Confidence based on data volatility (cv < 20% = high, < 50% = moderate, else low).

    Args:
        rows: Data rows as dicts.
        amount_column: Column with spend amount.
        date_column: Column with date value.
        periods_ahead: Number of future months to forecast.

    Returns:
        List of ForecastEntry. Empty if insufficient data.
    """
    if not rows or periods_ahead <= 0:
        return []

    # Aggregate spending by month
    monthly: dict[str, float] = {}
    for row in rows:
        amt = _safe_float(row.get(amount_column))
        dt = _parse_date(row.get(date_column))
        if amt is None or dt is None:
            continue
        mk = _month_key(dt)
        monthly[mk] = monthly.get(mk, 0.0) + amt

    if not monthly:
        return []

    sorted_months = sorted(monthly.keys())
    values = [monthly[m] for m in sorted_months]

    if len(values) < 2:
        # With only one month of data, project flat
        avg = values[0]
        mean_val = avg
        std_val = 0.0
    else:
        mean_val = sum(values) / len(values)
        std_val = statistics.stdev(values)

    # Compute linear trend: use simple regression y = a + b*x
    n = len(values)
    if n >= 2:
        x_vals = list(range(n))
        x_mean = sum(x_vals) / n
        y_mean = mean_val

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, values))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)

        if denominator != 0:
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
        else:
            slope = 0.0
            intercept = y_mean
    else:
        slope = 0.0
        intercept = values[0]

    # Coefficient of variation for confidence
    cv = (std_val / abs(mean_val) * 100) if mean_val != 0 else 0.0

    if cv < 20:
        confidence = "high"
    elif cv < 50:
        confidence = "moderate"
    else:
        confidence = "low"

    # Generate last month's date, then project forward
    last_month_str = sorted_months[-1]
    last_dt = datetime.strptime(last_month_str + "-01", "%Y-%m-%d")

    forecasts: list[ForecastEntry] = []
    cumulative = 0.0
    for i in range(1, periods_ahead + 1):
        x_idx = n - 1 + i  # continuing the x index
        projected = intercept + slope * x_idx
        projected = max(projected, 0.0)  # don't allow negative forecast
        cumulative += projected

        # Compute the future month
        future_month = last_dt.month + i
        future_year = last_dt.year
        while future_month > 12:
            future_month -= 12
            future_year += 1
        period_str = f"{future_year:04d}-{future_month:02d}"

        forecasts.append(
            ForecastEntry(
                period=period_str,
                projected_amount=round(projected, 2),
                cumulative=round(cumulative, 2),
                confidence=confidence,
            )
        )

    return forecasts


# ---------------------------------------------------------------------------
# 5. format_budget_report
# ---------------------------------------------------------------------------


def format_budget_report(
    budget_result: BudgetResult | None = None,
    burn_rate: BurnRateResult | None = None,
    spending: SpendingResult | None = None,
    forecast: list[ForecastEntry] | None = None,
) -> str:
    """Generate a combined text report from available budget analyses.

    Args:
        budget_result: Budget vs actual analysis result.
        burn_rate: Burn rate computation result.
        spending: Spending pattern analysis result.
        forecast: List of forecast entries.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Budget Tracking Report")
    sections.append("=" * 40)

    if budget_result is not None:
        lines = ["", "Budget vs Actual", "-" * 38]
        for c in budget_result.categories:
            flag = " [OVER]" if c.over_budget else ""
            lines.append(
                f"  {c.category}: budget={c.budget:,.2f} actual={c.actual:,.2f} "
                f"variance={c.variance:+,.2f} ({c.variance_pct:+.1f}%){flag}"
            )
        lines.append(
            f"  Total: budget={budget_result.total_budget:,.2f} "
            f"actual={budget_result.total_actual:,.2f} "
            f"variance={budget_result.overall_variance:+,.2f} "
            f"({budget_result.overall_variance_pct:+.1f}%)"
        )
        lines.append(
            f"  Over-budget: {budget_result.over_budget_count} | "
            f"Under-budget: {budget_result.under_budget_count}"
        )
        if budget_result.periods:
            lines.append("")
            lines.append("  Period Breakdown:")
            for p in budget_result.periods:
                lines.append(
                    f"    {p.period}: budget={p.budget:,.2f} actual={p.actual:,.2f} "
                    f"variance={p.variance:+,.2f} ({p.variance_pct:+.1f}%)"
                )
        sections.append("\n".join(lines))

    if burn_rate is not None:
        lines = ["", "Burn Rate", "-" * 38]
        lines.append(f"  Daily burn rate: {burn_rate.daily_burn_rate:,.2f}")
        lines.append(f"  Monthly burn rate: {burn_rate.monthly_burn_rate:,.2f}")
        lines.append(f"  Total spend: {burn_rate.total_spend:,.2f} over {burn_rate.num_days} days")
        lines.append(f"  Trend: {burn_rate.trend}")
        if burn_rate.remaining_budget is not None:
            lines.append(f"  Remaining budget: {burn_rate.remaining_budget:,.2f}")
        if burn_rate.days_until_exhaustion is not None:
            lines.append(
                f"  Days until exhaustion: {burn_rate.days_until_exhaustion:,.0f}"
            )
        if burn_rate.projected_end_date is not None:
            lines.append(f"  Projected end date: {burn_rate.projected_end_date}")
        sections.append("\n".join(lines))

    if spending is not None:
        lines = ["", "Spending Patterns", "-" * 38]
        lines.append(f"  Total spend: {spending.total_spend:,.2f}")
        lines.append("  Top categories:")
        for c in spending.top_categories:
            lines.append(
                f"    {c.category}: {c.amount:,.2f} ({c.pct_of_total:.1f}%)"
            )
        if spending.top_vendors:
            lines.append("  Top vendors:")
            for v in spending.top_vendors:
                lines.append(
                    f"    {v.vendor}: {v.amount:,.2f} ({v.pct_of_total:.1f}%)"
                )
        sections.append("\n".join(lines))

    if forecast:
        lines = ["", "Budget Forecast", "-" * 38]
        for f in forecast:
            lines.append(
                f"  {f.period}: projected={f.projected_amount:,.2f} "
                f"cumulative={f.cumulative:,.2f} [{f.confidence} confidence]"
            )
        sections.append("\n".join(lines))

    if budget_result is None and burn_rate is None and spending is None and not forecast:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
