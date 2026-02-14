"""Pricing analysis — distribution, elasticity, competitive positioning, and margins.

Pure functions for pricing analysis including price distribution profiling,
price elasticity of demand computation, competitive pricing gap analysis,
price-margin analysis, and combined reporting.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Try to convert a value to float; return None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Dataclasses — Price Distribution
# ---------------------------------------------------------------------------


@dataclass
class ProductPrice:
    """Price statistics for a single product."""

    product: str
    min_price: float
    max_price: float
    avg_price: float
    count: int


@dataclass
class CategoryPrice:
    """Price statistics for a single category."""

    category: str
    avg_price: float
    count: int


@dataclass
class PriceDistributionResult:
    """Complete price distribution analysis result."""

    mean_price: float
    median_price: float
    std_price: float
    min_price: float
    max_price: float
    outlier_count: int
    outlier_pct: float
    by_product: list[ProductPrice]
    by_category: list[CategoryPrice]
    summary: str


# ---------------------------------------------------------------------------
# Dataclasses — Price Elasticity
# ---------------------------------------------------------------------------


@dataclass
class ProductElasticity:
    """Elasticity result for a single product."""

    product: str
    elasticity: float
    elasticity_type: str


@dataclass
class PriceElasticityResult:
    """Complete price elasticity analysis result."""

    overall_elasticity: float
    elasticity_type: str
    by_product: list[ProductElasticity]
    summary: str


# ---------------------------------------------------------------------------
# Dataclasses — Competitive Pricing
# ---------------------------------------------------------------------------


@dataclass
class ProductGap:
    """Competitive price gap for a single product."""

    product: str
    our_price: float
    competitor_avg_price: float
    price_gap_pct: float
    position: str


@dataclass
class CompetitorGap:
    """Aggregated price gap for a single competitor."""

    competitor: str
    avg_gap_pct: float
    product_count: int


@dataclass
class CompetitivePricingResult:
    """Complete competitive pricing analysis result."""

    avg_price_gap: float
    premium_count: int
    competitive_count: int
    discount_count: int
    by_product: list[ProductGap]
    by_competitor: list[CompetitorGap]
    summary: str


# ---------------------------------------------------------------------------
# Dataclasses — Price Margin
# ---------------------------------------------------------------------------


@dataclass
class ProductMargin:
    """Margin statistics for a single product."""

    product: str
    avg_price: float
    avg_cost: float
    margin_pct: float
    volume: int


@dataclass
class MarginBucket:
    """A single margin distribution bucket."""

    range_label: str
    count: int
    pct: float


@dataclass
class PriceMarginResult:
    """Complete price margin analysis result."""

    avg_margin: float
    weighted_margin: float | None
    min_margin: float
    max_margin: float
    negative_margin_count: int
    by_product: list[ProductMargin]
    margin_distribution: list[MarginBucket]
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_price_distribution
# ---------------------------------------------------------------------------


def analyze_price_distribution(
    rows: list[dict],
    price_column: str,
    product_column: str | None = None,
    category_column: str | None = None,
) -> PriceDistributionResult | None:
    """Analyze the distribution of prices across data rows.

    Computes mean, median, standard deviation, min, max, IQR, and identifies
    outliers beyond 1.5 * IQR.  Optionally breaks down by product and/or
    category.

    Args:
        rows: Data rows as dicts.
        price_column: Column with price values.
        product_column: Optional column identifying the product.
        category_column: Optional column identifying the category.

    Returns:
        PriceDistributionResult or None if no valid data.
    """
    if not rows:
        return None

    prices: list[float] = []
    for row in rows:
        p = _safe_float(row.get(price_column))
        if p is not None:
            prices.append(p)

    if not prices:
        return None

    mean_price = sum(prices) / len(prices)
    sorted_prices = sorted(prices)
    median_price = statistics.median(sorted_prices)
    std_price = statistics.stdev(prices) if len(prices) >= 2 else 0.0
    min_price = sorted_prices[0]
    max_price = sorted_prices[-1]

    # IQR and outliers
    n = len(sorted_prices)
    q1_idx = n // 4
    q3_idx = (3 * n) // 4
    q1 = sorted_prices[q1_idx]
    q3 = sorted_prices[q3_idx]
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    outlier_count = sum(1 for p in prices if p < lower_fence or p > upper_fence)
    outlier_pct = round(outlier_count / len(prices) * 100, 2)

    # Per-product breakdown
    by_product: list[ProductPrice] = []
    if product_column is not None:
        product_prices: dict[str, list[float]] = {}
        for row in rows:
            prod = row.get(product_column)
            p = _safe_float(row.get(price_column))
            if prod is None or p is None:
                continue
            product_prices.setdefault(str(prod), []).append(p)

        for prod_name in sorted(product_prices.keys()):
            pp = product_prices[prod_name]
            by_product.append(ProductPrice(
                product=prod_name,
                min_price=round(min(pp), 4),
                max_price=round(max(pp), 4),
                avg_price=round(sum(pp) / len(pp), 4),
                count=len(pp),
            ))

    # Per-category breakdown
    by_category: list[CategoryPrice] = []
    if category_column is not None:
        cat_prices: dict[str, list[float]] = {}
        for row in rows:
            cat = row.get(category_column)
            p = _safe_float(row.get(price_column))
            if cat is None or p is None:
                continue
            cat_prices.setdefault(str(cat), []).append(p)

        for cat_name in sorted(cat_prices.keys()):
            cp = cat_prices[cat_name]
            by_category.append(CategoryPrice(
                category=cat_name,
                avg_price=round(sum(cp) / len(cp), 4),
                count=len(cp),
            ))

    summary = (
        f"Price distribution across {len(prices)} observations: "
        f"Mean = {mean_price:,.2f}, Median = {median_price:,.2f}, "
        f"Std = {std_price:,.2f}. "
        f"Range: [{min_price:,.2f}, {max_price:,.2f}]. "
        f"Outliers: {outlier_count} ({outlier_pct:.1f}%)."
    )

    return PriceDistributionResult(
        mean_price=round(mean_price, 4),
        median_price=round(median_price, 4),
        std_price=round(std_price, 4),
        min_price=round(min_price, 4),
        max_price=round(max_price, 4),
        outlier_count=outlier_count,
        outlier_pct=outlier_pct,
        by_product=by_product,
        by_category=by_category,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. compute_price_elasticity
# ---------------------------------------------------------------------------


def _classify_elasticity(e: float) -> str:
    """Classify elasticity value into Elastic, Unit Elastic, or Inelastic."""
    abs_e = abs(e)
    if abs(abs_e - 1.0) <= 0.1:
        return "Unit Elastic"
    elif abs_e > 1.0:
        return "Elastic"
    else:
        return "Inelastic"


def _compute_point_elasticities(pairs: list[tuple[float, float]]) -> list[float]:
    """Compute point elasticities from (price, quantity) pairs.

    Uses midpoint method: (DeltaQ / Q_avg) / (DeltaP / P_avg) for each
    consecutive pair.
    """
    elasticities: list[float] = []
    for i in range(1, len(pairs)):
        p1, q1 = pairs[i - 1]
        p2, q2 = pairs[i]
        p_avg = (p1 + p2) / 2
        q_avg = (q1 + q2) / 2
        delta_p = p2 - p1
        delta_q = q2 - q1

        if p_avg == 0 or q_avg == 0 or delta_p == 0:
            continue

        e = (delta_q / q_avg) / (delta_p / p_avg)
        elasticities.append(e)

    return elasticities


def compute_price_elasticity(
    rows: list[dict],
    price_column: str,
    quantity_column: str,
    product_column: str | None = None,
) -> PriceElasticityResult | None:
    """Compute price elasticity of demand.

    Uses point elasticity (midpoint method) for consecutive observations.
    Groups by product if product_column is provided.

    Args:
        rows: Data rows as dicts.
        price_column: Column with price values.
        quantity_column: Column with quantity / demand values.
        product_column: Optional column identifying the product.

    Returns:
        PriceElasticityResult or None if insufficient data.
    """
    if not rows:
        return None

    # Collect (price, quantity) pairs, optionally grouped by product
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        price = _safe_float(row.get(price_column))
        qty = _safe_float(row.get(quantity_column))
        if price is None or qty is None:
            continue
        key = str(row.get(product_column)) if product_column and row.get(product_column) is not None else "_overall_"
        groups.setdefault(key, []).append((price, qty))

    if not groups:
        return None

    # Compute per-product elasticities
    by_product: list[ProductElasticity] = []
    all_elasticities: list[float] = []

    for prod_name in sorted(groups.keys()):
        pairs = groups[prod_name]
        if len(pairs) < 2:
            continue
        elist = _compute_point_elasticities(pairs)
        if not elist:
            continue
        avg_e = sum(elist) / len(elist)
        all_elasticities.extend(elist)
        if prod_name != "_overall_":
            by_product.append(ProductElasticity(
                product=prod_name,
                elasticity=round(avg_e, 4),
                elasticity_type=_classify_elasticity(avg_e),
            ))

    if not all_elasticities:
        return None

    overall_elasticity = sum(all_elasticities) / len(all_elasticities)
    elasticity_type = _classify_elasticity(overall_elasticity)

    summary = (
        f"Price elasticity of demand: {overall_elasticity:.2f} ({elasticity_type}). "
        f"Based on {len(all_elasticities)} observation pair(s)."
    )
    if by_product:
        summary += f" Analyzed across {len(by_product)} product(s)."

    return PriceElasticityResult(
        overall_elasticity=round(overall_elasticity, 4),
        elasticity_type=elasticity_type,
        by_product=by_product,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. analyze_competitive_pricing
# ---------------------------------------------------------------------------


def _classify_position(gap_pct: float) -> str:
    """Classify competitive position based on price gap percentage."""
    if gap_pct > 10:
        return "Premium"
    elif gap_pct < -10:
        return "Discount"
    else:
        return "Competitive"


def analyze_competitive_pricing(
    rows: list[dict],
    product_column: str,
    our_price_column: str,
    competitor_price_column: str,
    competitor_column: str | None = None,
) -> CompetitivePricingResult | None:
    """Analyze competitive pricing gaps.

    Computes the price gap as (our_price - competitor_price) / competitor_price * 100
    and classifies each product's position as Premium (>10%), Competitive (+-10%),
    or Discount (<-10%).

    Args:
        rows: Data rows as dicts.
        product_column: Column identifying the product.
        our_price_column: Column with our price.
        competitor_price_column: Column with competitor price.
        competitor_column: Optional column identifying the competitor.

    Returns:
        CompetitivePricingResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect valid rows
    valid_rows: list[dict] = []
    for row in rows:
        product = row.get(product_column)
        our_p = _safe_float(row.get(our_price_column))
        comp_p = _safe_float(row.get(competitor_price_column))
        if product is None or our_p is None or comp_p is None:
            continue
        valid_rows.append(row)

    if not valid_rows:
        return None

    # Per-product analysis: aggregate competitor prices per product
    product_our: dict[str, list[float]] = {}
    product_comp: dict[str, list[float]] = {}
    for row in valid_rows:
        prod = str(row.get(product_column))
        our_p = float(row.get(our_price_column))
        comp_p = float(row.get(competitor_price_column))
        product_our.setdefault(prod, []).append(our_p)
        product_comp.setdefault(prod, []).append(comp_p)

    by_product: list[ProductGap] = []
    all_gaps: list[float] = []
    premium_count = 0
    competitive_count = 0
    discount_count = 0

    for prod_name in sorted(product_our.keys()):
        our_avg = sum(product_our[prod_name]) / len(product_our[prod_name])
        comp_avg = sum(product_comp[prod_name]) / len(product_comp[prod_name])

        if comp_avg == 0:
            gap_pct = 0.0
        else:
            gap_pct = (our_avg - comp_avg) / comp_avg * 100

        position = _classify_position(gap_pct)
        if position == "Premium":
            premium_count += 1
        elif position == "Discount":
            discount_count += 1
        else:
            competitive_count += 1

        all_gaps.append(gap_pct)

        by_product.append(ProductGap(
            product=prod_name,
            our_price=round(our_avg, 4),
            competitor_avg_price=round(comp_avg, 4),
            price_gap_pct=round(gap_pct, 2),
            position=position,
        ))

    avg_price_gap = sum(all_gaps) / len(all_gaps) if all_gaps else 0.0

    # Per-competitor breakdown
    by_competitor: list[CompetitorGap] = []
    if competitor_column is not None:
        comp_gaps: dict[str, list[float]] = {}
        for row in valid_rows:
            comp_name = row.get(competitor_column)
            if comp_name is None:
                continue
            our_p = float(row.get(our_price_column))
            comp_p = float(row.get(competitor_price_column))
            if comp_p == 0:
                gap = 0.0
            else:
                gap = (our_p - comp_p) / comp_p * 100
            comp_gaps.setdefault(str(comp_name), []).append(gap)

        for cname in sorted(comp_gaps.keys()):
            gaps = comp_gaps[cname]
            by_competitor.append(CompetitorGap(
                competitor=cname,
                avg_gap_pct=round(sum(gaps) / len(gaps), 2),
                product_count=len(gaps),
            ))

    summary = (
        f"Competitive pricing across {len(by_product)} product(s): "
        f"Avg gap = {avg_price_gap:+.1f}%. "
        f"Premium: {premium_count}, Competitive: {competitive_count}, "
        f"Discount: {discount_count}."
    )

    return CompetitivePricingResult(
        avg_price_gap=round(avg_price_gap, 2),
        premium_count=premium_count,
        competitive_count=competitive_count,
        discount_count=discount_count,
        by_product=by_product,
        by_competitor=by_competitor,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. compute_price_margin_analysis
# ---------------------------------------------------------------------------


def compute_price_margin_analysis(
    rows: list[dict],
    price_column: str,
    cost_column: str,
    product_column: str | None = None,
    quantity_column: str | None = None,
) -> PriceMarginResult | None:
    """Compute gross margin analysis.

    Gross margin = (price - cost) / price * 100 per item.

    Args:
        rows: Data rows as dicts.
        price_column: Column with selling price.
        cost_column: Column with cost.
        product_column: Optional column identifying the product.
        quantity_column: Optional column with quantity for volume weighting.

    Returns:
        PriceMarginResult or None if no valid data.
    """
    if not rows:
        return None

    margins: list[float] = []
    quantities: list[float] = []
    revenues: list[float] = []

    # Per-product accumulators
    prod_prices: dict[str, list[float]] = {}
    prod_costs: dict[str, list[float]] = {}
    prod_qtys: dict[str, list[float]] = {}

    for row in rows:
        price = _safe_float(row.get(price_column))
        cost = _safe_float(row.get(cost_column))
        if price is None or cost is None:
            continue

        if price == 0:
            margin = 0.0
        else:
            margin = (price - cost) / price * 100

        margins.append(margin)

        qty = _safe_float(row.get(quantity_column)) if quantity_column else None
        if qty is not None:
            quantities.append(qty)
            revenues.append(price * qty)
        else:
            quantities.append(1.0)
            revenues.append(price)

        if product_column is not None:
            prod = row.get(product_column)
            if prod is not None:
                key = str(prod)
                prod_prices.setdefault(key, []).append(price)
                prod_costs.setdefault(key, []).append(cost)
                if qty is not None:
                    prod_qtys.setdefault(key, []).append(qty)
                else:
                    prod_qtys.setdefault(key, []).append(1.0)

    if not margins:
        return None

    avg_margin = sum(margins) / len(margins)
    min_margin = min(margins)
    max_margin = max(margins)
    negative_margin_count = sum(1 for m in margins if m < 0)

    # Weighted average margin by revenue
    weighted_margin: float | None = None
    if quantity_column is not None:
        total_revenue = sum(revenues)
        if total_revenue > 0:
            # weighted margin = sum(margin_i * revenue_i) / sum(revenue_i)
            weighted_sum = sum(m * r for m, r in zip(margins, revenues))
            weighted_margin = round(weighted_sum / total_revenue, 4)

    # Per-product margins
    by_product: list[ProductMargin] = []
    if product_column is not None:
        for prod_name in sorted(prod_prices.keys()):
            pp = prod_prices[prod_name]
            cc = prod_costs[prod_name]
            qq = prod_qtys[prod_name]
            avg_p = sum(pp) / len(pp)
            avg_c = sum(cc) / len(cc)
            total_vol = int(sum(qq))
            if avg_p == 0:
                m_pct = 0.0
            else:
                m_pct = (avg_p - avg_c) / avg_p * 100

            by_product.append(ProductMargin(
                product=prod_name,
                avg_price=round(avg_p, 4),
                avg_cost=round(avg_c, 4),
                margin_pct=round(m_pct, 2),
                volume=total_vol,
            ))

    # Margin distribution buckets
    bucket_defs = [
        ("<0%", lambda m: m < 0),
        ("0-10%", lambda m: 0 <= m < 10),
        ("10-20%", lambda m: 10 <= m < 20),
        ("20-30%", lambda m: 20 <= m < 30),
        ("30-50%", lambda m: 30 <= m < 50),
        (">50%", lambda m: m >= 50),
    ]
    margin_distribution: list[MarginBucket] = []
    total_items = len(margins)
    for label, pred in bucket_defs:
        cnt = sum(1 for m in margins if pred(m))
        pct = round(cnt / total_items * 100, 2) if total_items > 0 else 0.0
        margin_distribution.append(MarginBucket(
            range_label=label,
            count=cnt,
            pct=pct,
        ))

    summary = (
        f"Margin analysis across {len(margins)} item(s): "
        f"Avg margin = {avg_margin:.1f}%, "
        f"Range: [{min_margin:.1f}%, {max_margin:.1f}%]. "
        f"Negative margins: {negative_margin_count}."
    )
    if weighted_margin is not None:
        summary += f" Volume-weighted margin = {weighted_margin:.1f}%."

    return PriceMarginResult(
        avg_margin=round(avg_margin, 4),
        weighted_margin=weighted_margin,
        min_margin=round(min_margin, 4),
        max_margin=round(max_margin, 4),
        negative_margin_count=negative_margin_count,
        by_product=by_product,
        margin_distribution=margin_distribution,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_pricing_report
# ---------------------------------------------------------------------------


def format_pricing_report(
    distribution: PriceDistributionResult | None = None,
    elasticity: PriceElasticityResult | None = None,
    competitive: CompetitivePricingResult | None = None,
    margins: PriceMarginResult | None = None,
) -> str:
    """Generate a combined text report from available pricing analyses.

    Each section is only included if the corresponding parameter is not None.

    Args:
        distribution: Price distribution result.
        elasticity: Price elasticity result.
        competitive: Competitive pricing result.
        margins: Price margin result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Pricing Analysis Report")
    sections.append("=" * 50)

    if distribution is not None:
        lines = ["", "Price Distribution", "-" * 48]
        lines.append(
            f"  Mean: {distribution.mean_price:,.2f} | "
            f"Median: {distribution.median_price:,.2f} | "
            f"Std: {distribution.std_price:,.2f}"
        )
        lines.append(
            f"  Range: [{distribution.min_price:,.2f}, {distribution.max_price:,.2f}]"
        )
        lines.append(
            f"  Outliers: {distribution.outlier_count} ({distribution.outlier_pct:.1f}%)"
        )
        if distribution.by_product:
            lines.append("  By Product:")
            for pp in distribution.by_product:
                lines.append(
                    f"    {pp.product}: avg={pp.avg_price:,.2f} "
                    f"[{pp.min_price:,.2f} - {pp.max_price:,.2f}] "
                    f"(n={pp.count})"
                )
        if distribution.by_category:
            lines.append("  By Category:")
            for cp in distribution.by_category:
                lines.append(
                    f"    {cp.category}: avg={cp.avg_price:,.2f} (n={cp.count})"
                )
        sections.append("\n".join(lines))

    if elasticity is not None:
        lines = ["", "Price Elasticity", "-" * 48]
        lines.append(
            f"  Overall: {elasticity.overall_elasticity:.4f} "
            f"({elasticity.elasticity_type})"
        )
        if elasticity.by_product:
            lines.append("  By Product:")
            for pe in elasticity.by_product:
                lines.append(
                    f"    {pe.product}: {pe.elasticity:.4f} ({pe.elasticity_type})"
                )
        sections.append("\n".join(lines))

    if competitive is not None:
        lines = ["", "Competitive Pricing", "-" * 48]
        lines.append(f"  Avg Price Gap: {competitive.avg_price_gap:+.1f}%")
        lines.append(
            f"  Premium: {competitive.premium_count} | "
            f"Competitive: {competitive.competitive_count} | "
            f"Discount: {competitive.discount_count}"
        )
        if competitive.by_product:
            lines.append("  By Product:")
            for pg in competitive.by_product:
                lines.append(
                    f"    {pg.product}: our={pg.our_price:,.2f} "
                    f"vs comp={pg.competitor_avg_price:,.2f} "
                    f"({pg.price_gap_pct:+.1f}%) [{pg.position}]"
                )
        if competitive.by_competitor:
            lines.append("  By Competitor:")
            for cg in competitive.by_competitor:
                lines.append(
                    f"    {cg.competitor}: gap={cg.avg_gap_pct:+.1f}% "
                    f"({cg.product_count} products)"
                )
        sections.append("\n".join(lines))

    if margins is not None:
        lines = ["", "Price Margins", "-" * 48]
        lines.append(
            f"  Avg Margin: {margins.avg_margin:.1f}% | "
            f"Range: [{margins.min_margin:.1f}%, {margins.max_margin:.1f}%]"
        )
        if margins.weighted_margin is not None:
            lines.append(f"  Volume-Weighted Margin: {margins.weighted_margin:.1f}%")
        lines.append(f"  Negative Margins: {margins.negative_margin_count}")
        if margins.by_product:
            lines.append("  By Product:")
            for pm in margins.by_product:
                lines.append(
                    f"    {pm.product}: price={pm.avg_price:,.2f} "
                    f"cost={pm.avg_cost:,.2f} margin={pm.margin_pct:.1f}% "
                    f"(vol={pm.volume})"
                )
        if margins.margin_distribution:
            lines.append("  Margin Distribution:")
            for mb in margins.margin_distribution:
                lines.append(
                    f"    {mb.range_label}: {mb.count} ({mb.pct:.1f}%)"
                )
        sections.append("\n".join(lines))

    if distribution is None and elasticity is None and competitive is None and margins is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
