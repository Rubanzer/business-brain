"""Sales analytics — performance, product mix, velocity, and discounts.

Pure functions for analyzing sales data across multiple dimensions:
rep performance, regional breakdowns, product revenue mix, pipeline
velocity, and discount impact analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


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
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PeriodSales:
    """Sales for a single time period."""
    period: str
    amount: float


@dataclass
class RepSales:
    """Sales per rep with ranking."""
    rep: str
    total: float
    rank: int


@dataclass
class RegionSales:
    """Sales per region with share."""
    region: str
    total: float
    pct: float


@dataclass
class SalesResult:
    """Overall sales performance result."""
    total_sales: float
    period_count: int
    avg_per_period: float
    growth_rate: float  # first-half vs second-half growth %
    best_period: PeriodSales | None
    worst_period: PeriodSales | None
    by_rep: list[RepSales]
    by_region: list[RegionSales]
    summary: str


@dataclass
class ProductRevenue:
    """Revenue details for a single product."""
    product: str
    revenue: float
    pct_of_total: float
    transaction_count: int
    avg_price: float | None  # only if quantity_column provided


@dataclass
class ProductMixResult:
    """Product mix analysis result."""
    products: list[ProductRevenue]
    total_revenue: float
    hhi: float  # Herfindahl-Hirschman Index (0-10000)
    concentration_risk: str  # "low", "moderate", "high"
    top_products: list[ProductRevenue]
    bottom_products: list[ProductRevenue]
    summary: str


@dataclass
class VelocityResult:
    """Sales velocity / pipeline result."""
    total_deals: int
    total_value: float
    avg_deal_size: float
    win_rate: float | None  # % of deals won
    avg_days_to_close: float | None
    pipeline_velocity: float | None
    funnel: list[tuple[str, int]]  # (stage, count) pairs
    summary: str


@dataclass
class DiscountBucket:
    """Distribution bucket for discount ranges."""
    range_label: str
    count: int
    pct: float


@dataclass
class ProductDiscount:
    """Average discount by product."""
    product: str
    avg_discount: float
    deal_count: int


@dataclass
class DiscountResult:
    """Discount impact analysis result."""
    avg_discount: float
    max_discount: float
    revenue_impact: float  # estimated revenue lost to discounts
    distribution: list[DiscountBucket]
    by_product: list[ProductDiscount]
    discount_volume_correlation: float | None
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_sales_performance
# ---------------------------------------------------------------------------


def analyze_sales_performance(
    rows: list[dict],
    amount_column: str,
    date_column: str,
    rep_column: str | None = None,
    region_column: str | None = None,
) -> SalesResult | None:
    """Analyze overall sales performance.

    Args:
        rows: Data rows as dicts.
        amount_column: Column containing sale amounts.
        date_column: Column containing date / period identifiers.
        rep_column: Optional column for sales rep.
        region_column: Optional column for region.

    Returns:
        SalesResult or None if insufficient data.
    """
    if not rows:
        return None

    # Collect valid entries
    entries: list[tuple[str, float]] = []
    for row in rows:
        amount = _safe_float(row.get(amount_column))
        period = row.get(date_column)
        if amount is None or period is None:
            continue
        entries.append((str(period), amount))

    if not entries:
        return None

    total_sales = sum(a for _, a in entries)

    # Aggregate by period
    period_totals: dict[str, float] = {}
    for period, amount in entries:
        period_totals[period] = period_totals.get(period, 0.0) + amount

    periods_list = sorted(period_totals.keys())
    period_count = len(periods_list)
    avg_per_period = total_sales / period_count if period_count > 0 else 0.0

    # Growth rate: first half vs second half
    growth_rate = _compute_growth_rate(periods_list, period_totals)

    # Best / worst period
    period_sales = [PeriodSales(p, round(period_totals[p], 2)) for p in periods_list]
    best_period = max(period_sales, key=lambda ps: ps.amount) if period_sales else None
    worst_period = min(period_sales, key=lambda ps: ps.amount) if period_sales else None

    # By rep
    by_rep: list[RepSales] = []
    if rep_column:
        rep_totals: dict[str, float] = {}
        for row in rows:
            amount = _safe_float(row.get(amount_column))
            rep = row.get(rep_column)
            if amount is None or rep is None:
                continue
            rep_totals[str(rep)] = rep_totals.get(str(rep), 0.0) + amount

        sorted_reps = sorted(rep_totals.items(), key=lambda x: x[1], reverse=True)
        by_rep = [
            RepSales(rep=name, total=round(total, 2), rank=i + 1)
            for i, (name, total) in enumerate(sorted_reps)
        ]

    # By region
    by_region: list[RegionSales] = []
    if region_column:
        region_totals: dict[str, float] = {}
        for row in rows:
            amount = _safe_float(row.get(amount_column))
            region = row.get(region_column)
            if amount is None or region is None:
                continue
            region_totals[str(region)] = region_totals.get(str(region), 0.0) + amount

        for name, total in sorted(region_totals.items(), key=lambda x: x[1], reverse=True):
            pct = (total / total_sales * 100) if total_sales > 0 else 0.0
            by_region.append(RegionSales(region=name, total=round(total, 2), pct=round(pct, 1)))

    summary = (
        f"Total sales: {total_sales:,.2f} across {period_count} periods. "
        f"Avg/period: {avg_per_period:,.2f}. Growth: {growth_rate:+.1f}%."
    )
    if by_rep:
        summary += f" {len(by_rep)} reps tracked."
    if by_region:
        summary += f" {len(by_region)} regions."

    return SalesResult(
        total_sales=round(total_sales, 2),
        period_count=period_count,
        avg_per_period=round(avg_per_period, 2),
        growth_rate=round(growth_rate, 2),
        best_period=best_period,
        worst_period=worst_period,
        by_rep=by_rep,
        by_region=by_region,
        summary=summary,
    )


def _compute_growth_rate(
    periods: list[str], totals: dict[str, float]
) -> float:
    """Compute growth rate as change from first half to second half."""
    if len(periods) < 2:
        return 0.0
    mid = len(periods) // 2
    first_half = sum(totals[p] for p in periods[:mid])
    second_half = sum(totals[p] for p in periods[mid:])
    if first_half == 0:
        return 0.0
    return (second_half - first_half) / abs(first_half) * 100


# ---------------------------------------------------------------------------
# 2. analyze_product_mix
# ---------------------------------------------------------------------------


def analyze_product_mix(
    rows: list[dict],
    product_column: str,
    amount_column: str,
    quantity_column: str | None = None,
) -> ProductMixResult | None:
    """Analyze revenue distribution across products.

    Args:
        rows: Data rows as dicts.
        product_column: Column for product name.
        amount_column: Column for revenue / amount.
        quantity_column: Optional column for unit quantity.

    Returns:
        ProductMixResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per product
    product_data: dict[str, dict] = {}
    for row in rows:
        product = row.get(product_column)
        amount = _safe_float(row.get(amount_column))
        if product is None or amount is None:
            continue

        key = str(product)
        if key not in product_data:
            product_data[key] = {"revenue": 0.0, "count": 0, "total_qty": 0.0}

        product_data[key]["revenue"] += amount
        product_data[key]["count"] += 1

        if quantity_column:
            qty = _safe_float(row.get(quantity_column))
            if qty is not None:
                product_data[key]["total_qty"] += qty

    if not product_data:
        return None

    total_revenue = sum(d["revenue"] for d in product_data.values())

    products: list[ProductRevenue] = []
    for name, data in product_data.items():
        pct = (data["revenue"] / total_revenue * 100) if total_revenue > 0 else 0.0
        avg_price = None
        if quantity_column and data["total_qty"] > 0:
            avg_price = round(data["revenue"] / data["total_qty"], 2)
        products.append(ProductRevenue(
            product=name,
            revenue=round(data["revenue"], 2),
            pct_of_total=round(pct, 2),
            transaction_count=data["count"],
            avg_price=avg_price,
        ))

    # Sort by revenue descending
    products.sort(key=lambda p: p.revenue, reverse=True)

    # HHI (Herfindahl-Hirschman Index)
    hhi = sum(p.pct_of_total ** 2 for p in products)
    if hhi > 2500:
        concentration_risk = "high"
    elif hhi > 1500:
        concentration_risk = "moderate"
    else:
        concentration_risk = "low"

    top_products = products[:5]
    bottom_products = products[-5:] if len(products) > 5 else products[:]

    summary = (
        f"Product mix: {len(products)} products, total revenue {total_revenue:,.2f}. "
        f"HHI: {hhi:,.0f} ({concentration_risk} concentration). "
        f"Top product: {products[0].product} ({products[0].pct_of_total:.1f}%)."
    )

    return ProductMixResult(
        products=products,
        total_revenue=round(total_revenue, 2),
        hhi=round(hhi, 2),
        concentration_risk=concentration_risk,
        top_products=top_products,
        bottom_products=bottom_products,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. compute_sales_velocity
# ---------------------------------------------------------------------------


def compute_sales_velocity(
    rows: list[dict],
    deal_column: str,
    amount_column: str,
    date_column: str,
    stage_column: str | None = None,
) -> VelocityResult | None:
    """Compute sales pipeline velocity metrics.

    Args:
        rows: Data rows as dicts.
        deal_column: Column identifying unique deals.
        amount_column: Column for deal value.
        date_column: Column for date (used to compute cycle time).
        stage_column: Optional column for pipeline stage.

    Returns:
        VelocityResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per deal
    deal_data: dict[str, dict] = {}
    for row in rows:
        deal = row.get(deal_column)
        amount = _safe_float(row.get(amount_column))
        if deal is None or amount is None:
            continue

        key = str(deal)
        if key not in deal_data:
            deal_data[key] = {
                "amount": amount,
                "dates": [],
                "stages": set(),
            }
        else:
            # Take the max amount for the deal
            if amount > deal_data[key]["amount"]:
                deal_data[key]["amount"] = amount

        dt = _parse_date(row.get(date_column))
        if dt is not None:
            deal_data[key]["dates"].append(dt)

        if stage_column:
            stage = row.get(stage_column)
            if stage is not None:
                deal_data[key]["stages"].add(str(stage).lower())

    if not deal_data:
        return None

    total_deals = len(deal_data)
    total_value = sum(d["amount"] for d in deal_data.values())
    avg_deal_size = total_value / total_deals if total_deals > 0 else 0.0

    # Win rate
    win_rate: float | None = None
    if stage_column:
        won_keywords = {"won", "closed", "closed won", "closed-won"}
        won_count = sum(
            1 for d in deal_data.values()
            if d["stages"] & won_keywords
        )
        win_rate = (won_count / total_deals * 100) if total_deals > 0 else 0.0

    # Avg days to close
    avg_days_to_close: float | None = None
    cycle_days: list[float] = []
    for d in deal_data.values():
        if len(d["dates"]) >= 2:
            sorted_dates = sorted(d["dates"])
            delta = (sorted_dates[-1] - sorted_dates[0]).days
            cycle_days.append(float(delta))

    if cycle_days:
        avg_days_to_close = sum(cycle_days) / len(cycle_days)

    # Pipeline velocity = (deals * avg_deal * win_rate) / avg_cycle
    pipeline_velocity: float | None = None
    if win_rate is not None and avg_days_to_close is not None and avg_days_to_close > 0:
        pipeline_velocity = (total_deals * avg_deal_size * (win_rate / 100)) / avg_days_to_close

    # Stage funnel
    funnel: list[tuple[str, int]] = []
    if stage_column:
        stage_counts: dict[str, int] = {}
        for d in deal_data.values():
            for stage in d["stages"]:
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        funnel = sorted(stage_counts.items(), key=lambda x: x[1], reverse=True)

    # Summary
    summary_parts = [
        f"Pipeline: {total_deals} deals, total value {total_value:,.2f}.",
        f"Avg deal size: {avg_deal_size:,.2f}.",
    ]
    if win_rate is not None:
        summary_parts.append(f"Win rate: {win_rate:.1f}%.")
    if avg_days_to_close is not None:
        summary_parts.append(f"Avg cycle: {avg_days_to_close:.1f} days.")
    if pipeline_velocity is not None:
        summary_parts.append(f"Velocity: {pipeline_velocity:,.2f}/day.")

    return VelocityResult(
        total_deals=total_deals,
        total_value=round(total_value, 2),
        avg_deal_size=round(avg_deal_size, 2),
        win_rate=round(win_rate, 2) if win_rate is not None else None,
        avg_days_to_close=round(avg_days_to_close, 2) if avg_days_to_close is not None else None,
        pipeline_velocity=round(pipeline_velocity, 2) if pipeline_velocity is not None else None,
        funnel=funnel,
        summary=" ".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 4. analyze_discount_impact
# ---------------------------------------------------------------------------


_DISCOUNT_BUCKETS = [
    ("0%", 0.0, 0.0),
    ("1-5%", 0.01, 0.05),
    ("5-10%", 0.05, 0.10),
    ("10-20%", 0.10, 0.20),
    ("20%+", 0.20, float("inf")),
]


def analyze_discount_impact(
    rows: list[dict],
    amount_column: str,
    discount_column: str,
    quantity_column: str | None = None,
    product_column: str | None = None,
) -> DiscountResult | None:
    """Analyze the impact of discounts on revenue.

    The discount_column should contain a fractional value (e.g., 0.10 for 10%)
    or a percentage value (e.g., 10 for 10%).  Values > 1 are treated as
    percentages and divided by 100.

    Args:
        rows: Data rows as dicts.
        amount_column: Column for sale amount (after discount).
        discount_column: Column for discount rate.
        quantity_column: Optional column for units sold.
        product_column: Optional column for product name.

    Returns:
        DiscountResult or None if insufficient data.
    """
    if not rows:
        return None

    entries: list[dict] = []
    for row in rows:
        amount = _safe_float(row.get(amount_column))
        discount = _safe_float(row.get(discount_column))
        if amount is None or discount is None:
            continue

        # Normalize: treat values > 1 as percentages
        if discount > 1:
            discount = discount / 100.0

        # Clamp to [0, 1)
        discount = max(0.0, min(discount, 0.9999))

        entry: dict = {"amount": amount, "discount": discount}

        if quantity_column:
            qty = _safe_float(row.get(quantity_column))
            entry["quantity"] = qty

        if product_column:
            product = row.get(product_column)
            entry["product"] = str(product) if product is not None else None

        entries.append(entry)

    if not entries:
        return None

    discounts = [e["discount"] for e in entries]
    avg_discount = sum(discounts) / len(discounts)
    max_discount = max(discounts)

    # Revenue impact: for each transaction, the "lost" revenue is
    # amount * discount / (1 - discount), which is the amount that would
    # have been received at full price minus the discounted price.
    revenue_impact = 0.0
    for e in entries:
        d = e["discount"]
        if d < 1.0:
            revenue_impact += e["amount"] * d / (1.0 - d)

    # Distribution
    bucket_counts = {label: 0 for label, _, _ in _DISCOUNT_BUCKETS}
    for d in discounts:
        for label, lo, hi in _DISCOUNT_BUCKETS:
            if label == "0%":
                if d == 0.0:
                    bucket_counts[label] += 1
                    break
            elif label == "20%+":
                if d > 0.20 - 1e-9:
                    bucket_counts[label] += 1
                    break
            else:
                if lo < d <= hi:
                    bucket_counts[label] += 1
                    break
        else:
            # Edge case: nonzero discount <= 0.01 that doesn't fit 0%
            # Put in 1-5% bucket
            if 0 < d <= 0.05:
                bucket_counts["1-5%"] += 1

    total_entries = len(entries)
    distribution = [
        DiscountBucket(
            range_label=label,
            count=bucket_counts[label],
            pct=round(bucket_counts[label] / total_entries * 100, 1) if total_entries > 0 else 0.0,
        )
        for label, _, _ in _DISCOUNT_BUCKETS
    ]

    # By product
    by_product: list[ProductDiscount] = []
    if product_column:
        product_discounts: dict[str, list[float]] = {}
        for e in entries:
            p = e.get("product")
            if p is not None:
                product_discounts.setdefault(p, []).append(e["discount"])

        for name, disc_list in sorted(product_discounts.items()):
            by_product.append(ProductDiscount(
                product=name,
                avg_discount=round(sum(disc_list) / len(disc_list), 4),
                deal_count=len(disc_list),
            ))

    # Correlation between discount and volume
    discount_volume_correlation: float | None = None
    if quantity_column:
        pairs = [
            (e["discount"], e["quantity"])
            for e in entries
            if e.get("quantity") is not None
        ]
        if len(pairs) >= 2:
            discount_volume_correlation = _pearson_correlation(
                [p[0] for p in pairs], [p[1] for p in pairs]
            )

    summary = (
        f"Discounts: avg {avg_discount * 100:.1f}%, max {max_discount * 100:.1f}%. "
        f"Revenue impact (est. lost): {revenue_impact:,.2f}. "
        f"{len(entries)} transactions analyzed."
    )
    if discount_volume_correlation is not None:
        summary += f" Discount-volume correlation: {discount_volume_correlation:.3f}."

    return DiscountResult(
        avg_discount=round(avg_discount, 4),
        max_discount=round(max_discount, 4),
        revenue_impact=round(revenue_impact, 2),
        distribution=distribution,
        by_product=by_product,
        discount_volume_correlation=discount_volume_correlation,
        summary=summary,
    )


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    """Compute Pearson correlation coefficient between two lists."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)

    denom = (var_x * var_y) ** 0.5
    if denom == 0:
        return None
    return round(cov / denom, 4)


# ---------------------------------------------------------------------------
# 5. format_sales_report
# ---------------------------------------------------------------------------


def format_sales_report(
    performance: SalesResult | None = None,
    product_mix: ProductMixResult | None = None,
    velocity: VelocityResult | None = None,
    discounts: DiscountResult | None = None,
) -> str:
    """Combine analysis results into a formatted text report.

    Args:
        performance: Sales performance result.
        product_mix: Product mix result.
        velocity: Sales velocity result.
        discounts: Discount impact result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("SALES ANALYTICS REPORT")
    sections.append("=" * 60)

    if performance:
        lines = [
            "",
            "SALES PERFORMANCE",
            "-" * 40,
            f"  Total Sales:     {performance.total_sales:>15,.2f}",
            f"  Periods:         {performance.period_count:>15}",
            f"  Avg per Period:  {performance.avg_per_period:>15,.2f}",
            f"  Growth Rate:     {performance.growth_rate:>14.1f}%",
        ]
        if performance.best_period:
            lines.append(
                f"  Best Period:     {performance.best_period.period} "
                f"({performance.best_period.amount:,.2f})"
            )
        if performance.worst_period:
            lines.append(
                f"  Worst Period:    {performance.worst_period.period} "
                f"({performance.worst_period.amount:,.2f})"
            )
        if performance.by_rep:
            lines.append("")
            lines.append("  Top Reps:")
            for rep in performance.by_rep[:5]:
                lines.append(f"    {rep.rank}. {rep.rep:<20} {rep.total:>12,.2f}")
        if performance.by_region:
            lines.append("")
            lines.append("  Regions:")
            for reg in performance.by_region:
                lines.append(f"    {reg.region:<20} {reg.total:>12,.2f} ({reg.pct:.1f}%)")
        sections.append("\n".join(lines))

    if product_mix:
        lines = [
            "",
            "PRODUCT MIX",
            "-" * 40,
            f"  Products:        {len(product_mix.products):>15}",
            f"  Total Revenue:   {product_mix.total_revenue:>15,.2f}",
            f"  HHI:             {product_mix.hhi:>15,.0f}",
            f"  Concentration:   {product_mix.concentration_risk:>15}",
            "",
            "  Top Products:",
        ]
        for p in product_mix.top_products:
            lines.append(
                f"    {p.product:<20} {p.revenue:>12,.2f} ({p.pct_of_total:.1f}%)"
            )
        sections.append("\n".join(lines))

    if velocity:
        lines = [
            "",
            "SALES VELOCITY",
            "-" * 40,
            f"  Total Deals:     {velocity.total_deals:>15}",
            f"  Total Value:     {velocity.total_value:>15,.2f}",
            f"  Avg Deal Size:   {velocity.avg_deal_size:>15,.2f}",
        ]
        if velocity.win_rate is not None:
            lines.append(f"  Win Rate:        {velocity.win_rate:>14.1f}%")
        if velocity.avg_days_to_close is not None:
            lines.append(f"  Avg Cycle:       {velocity.avg_days_to_close:>12.1f} days")
        if velocity.pipeline_velocity is not None:
            lines.append(f"  Velocity:        {velocity.pipeline_velocity:>12,.2f}/day")
        if velocity.funnel:
            lines.append("")
            lines.append("  Stage Funnel:")
            for stage, count in velocity.funnel:
                lines.append(f"    {stage:<20} {count:>6}")
        sections.append("\n".join(lines))

    if discounts:
        lines = [
            "",
            "DISCOUNT IMPACT",
            "-" * 40,
            f"  Avg Discount:    {discounts.avg_discount * 100:>14.1f}%",
            f"  Max Discount:    {discounts.max_discount * 100:>14.1f}%",
            f"  Revenue Impact:  {discounts.revenue_impact:>15,.2f}",
            "",
            "  Distribution:",
        ]
        for bucket in discounts.distribution:
            lines.append(f"    {bucket.range_label:<10} {bucket.count:>6} ({bucket.pct:.1f}%)")
        if discounts.by_product:
            lines.append("")
            lines.append("  By Product:")
            for pd in discounts.by_product:
                lines.append(
                    f"    {pd.product:<20} avg {pd.avg_discount * 100:.1f}% ({pd.deal_count} deals)"
                )
        if discounts.discount_volume_correlation is not None:
            lines.append(
                f"\n  Discount-Volume Correlation: {discounts.discount_volume_correlation:.3f}"
            )
        sections.append("\n".join(lines))

    if not any([performance, product_mix, velocity, discounts]):
        sections.append("\nNo data available for report.")

    sections.append("\n" + "=" * 60)
    return "\n".join(sections)
