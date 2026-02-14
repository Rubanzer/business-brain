"""Market basket analysis — product associations, cross-sell, and basket metrics.

Pure functions for analyzing transactional data to discover product
co-occurrence patterns, basket size distributions, cross-sell opportunities,
and product frequency rankings.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


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


def _median(values: list[int | float]) -> float:
    """Compute the median of a list of numeric values."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 0:
        return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
    return float(sorted_vals[mid])


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ProductPair:
    """Association metrics for a pair of products."""
    product_a: str
    product_b: str
    support: float
    confidence_a_to_b: float
    confidence_b_to_a: float
    lift: float
    co_occurrence_count: int


@dataclass
class AssociationResult:
    """Result of product association analysis."""
    pairs: list[ProductPair]
    total_transactions: int
    unique_products: int
    avg_basket_size: float
    summary: str


@dataclass
class SizeBucket:
    """A single bucket in the basket size distribution."""
    size: int
    count: int
    pct: float


@dataclass
class BasketSizeResult:
    """Result of basket size analysis."""
    avg_size: float
    median_size: float
    max_size: int
    min_size: int
    distribution: list[SizeBucket]
    avg_value: float | None
    summary: str


@dataclass
class CrossSellProduct:
    """A product recommended for cross-selling with a target product."""
    product: str
    co_purchase_rate: float
    lift: float
    co_occurrence_count: int


@dataclass
class CrossSellResult:
    """Result of cross-sell opportunity analysis."""
    target_product: str
    target_transactions: int
    recommendations: list[CrossSellProduct]
    summary: str


@dataclass
class ProductFreq:
    """Frequency information for a single product."""
    product: str
    frequency: int
    pct_of_transactions: float
    unique_customers: int | None
    rank: int


@dataclass
class ProductFrequencyResult:
    """Result of product frequency analysis."""
    products: list[ProductFreq]
    total_transactions: int
    total_products: int
    most_popular: str
    least_popular: str
    summary: str


# ---------------------------------------------------------------------------
# Internal: basket building
# ---------------------------------------------------------------------------


def _build_baskets(
    rows: list[dict],
    transaction_column: str,
    product_column: str,
) -> dict[str, set[str]] | None:
    """Group products by transaction, returning {txn_id: {products}} or None."""
    baskets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        txn = row.get(transaction_column)
        product = row.get(product_column)
        if txn is None or product is None:
            continue
        baskets[str(txn)].add(str(product))

    if not baskets:
        return None
    return dict(baskets)


# ---------------------------------------------------------------------------
# 1. find_product_associations
# ---------------------------------------------------------------------------


def find_product_associations(
    rows: list[dict],
    transaction_column: str,
    product_column: str,
    min_support: float = 0.01,
) -> AssociationResult | None:
    """Find pairs of products that frequently appear together in transactions.

    Args:
        rows: Data rows as dicts.
        transaction_column: Column identifying unique transactions.
        product_column: Column containing product names.
        min_support: Minimum support threshold for a pair (default 1%).

    Returns:
        AssociationResult or None if insufficient data.
    """
    if not rows:
        return None

    baskets = _build_baskets(rows, transaction_column, product_column)
    if baskets is None:
        return None

    total_transactions = len(baskets)
    all_products: set[str] = set()
    for products in baskets.values():
        all_products.update(products)

    unique_products = len(all_products)
    if unique_products < 2:
        # Need at least 2 distinct products for pairs
        basket_sizes = [len(b) for b in baskets.values()]
        avg_basket = sum(basket_sizes) / len(basket_sizes) if basket_sizes else 0.0
        return AssociationResult(
            pairs=[],
            total_transactions=total_transactions,
            unique_products=unique_products,
            avg_basket_size=round(avg_basket, 2),
            summary=(
                f"Only {unique_products} unique product(s) found across "
                f"{total_transactions} transactions. Need at least 2 for associations."
            ),
        )

    # Compute individual product support (count of transactions containing product)
    product_counts: Counter[str] = Counter()
    for products in baskets.values():
        for p in products:
            product_counts[p] += 1

    # Compute pair co-occurrence counts
    pair_counts: Counter[tuple[str, str]] = Counter()
    for products in baskets.values():
        product_list = sorted(products)
        for i in range(len(product_list)):
            for j in range(i + 1, len(product_list)):
                pair_counts[(product_list[i], product_list[j])] += 1

    # Build ProductPair objects for pairs meeting min_support
    pairs: list[ProductPair] = []
    for (a, b), co_count in pair_counts.items():
        support = co_count / total_transactions
        if support < min_support:
            continue

        support_a = product_counts[a] / total_transactions
        support_b = product_counts[b] / total_transactions

        confidence_a_to_b = co_count / product_counts[a] if product_counts[a] > 0 else 0.0
        confidence_b_to_a = co_count / product_counts[b] if product_counts[b] > 0 else 0.0

        lift = confidence_a_to_b / support_b if support_b > 0 else 0.0

        pairs.append(ProductPair(
            product_a=a,
            product_b=b,
            support=round(support, 4),
            confidence_a_to_b=round(confidence_a_to_b, 4),
            confidence_b_to_a=round(confidence_b_to_a, 4),
            lift=round(lift, 4),
            co_occurrence_count=co_count,
        ))

    # Sort by lift descending, take top 20
    pairs.sort(key=lambda p: p.lift, reverse=True)
    pairs = pairs[:20]

    basket_sizes = [len(b) for b in baskets.values()]
    avg_basket = sum(basket_sizes) / len(basket_sizes) if basket_sizes else 0.0

    summary = (
        f"Found {len(pairs)} product pair(s) across {total_transactions} transactions "
        f"with {unique_products} unique products. "
        f"Avg basket size: {avg_basket:.1f}."
    )
    if pairs:
        top = pairs[0]
        summary += (
            f" Strongest association: {top.product_a} & {top.product_b} "
            f"(lift={top.lift:.2f})."
        )

    return AssociationResult(
        pairs=pairs,
        total_transactions=total_transactions,
        unique_products=unique_products,
        avg_basket_size=round(avg_basket, 2),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. analyze_basket_size
# ---------------------------------------------------------------------------


def analyze_basket_size(
    rows: list[dict],
    transaction_column: str,
    product_column: str,
    value_column: str | None = None,
) -> BasketSizeResult | None:
    """Analyze the distribution of basket sizes (items per transaction).

    Args:
        rows: Data rows as dicts.
        transaction_column: Column identifying unique transactions.
        product_column: Column containing product names.
        value_column: Optional column for item value (to compute avg basket value).

    Returns:
        BasketSizeResult or None if insufficient data.
    """
    if not rows:
        return None

    baskets = _build_baskets(rows, transaction_column, product_column)
    if baskets is None:
        return None

    sizes = [len(products) for products in baskets.values()]
    avg_size = sum(sizes) / len(sizes)
    median_size = _median(sizes)
    max_size = max(sizes)
    min_size = min(sizes)

    # Distribution of basket sizes
    size_counts: Counter[int] = Counter(sizes)
    total_baskets = len(sizes)
    distribution = [
        SizeBucket(
            size=s,
            count=c,
            pct=round(c / total_baskets * 100, 1),
        )
        for s, c in sorted(size_counts.items())
    ]

    # Average basket value
    avg_value: float | None = None
    if value_column is not None:
        txn_values: dict[str, float] = defaultdict(float)
        has_any_value = False
        for row in rows:
            txn = row.get(transaction_column)
            val = _safe_float(row.get(value_column))
            if txn is None or val is None:
                continue
            txn_values[str(txn)] += val
            has_any_value = True

        if has_any_value and txn_values:
            avg_value = round(sum(txn_values.values()) / len(txn_values), 2)

    summary = (
        f"Basket size: avg {avg_size:.1f}, median {median_size:.1f}, "
        f"range {min_size}-{max_size} across {total_baskets} transactions."
    )
    if avg_value is not None:
        summary += f" Avg basket value: {avg_value:,.2f}."

    return BasketSizeResult(
        avg_size=round(avg_size, 2),
        median_size=round(median_size, 2),
        max_size=max_size,
        min_size=min_size,
        distribution=distribution,
        avg_value=avg_value,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. find_cross_sell_opportunities
# ---------------------------------------------------------------------------


def find_cross_sell_opportunities(
    rows: list[dict],
    transaction_column: str,
    product_column: str,
    target_product: str,
) -> CrossSellResult | None:
    """Find products most frequently purchased alongside a target product.

    Args:
        rows: Data rows as dicts.
        transaction_column: Column identifying unique transactions.
        product_column: Column containing product names.
        target_product: The product to find cross-sell partners for.

    Returns:
        CrossSellResult or None if insufficient data.
    """
    if not rows:
        return None

    baskets = _build_baskets(rows, transaction_column, product_column)
    if baskets is None:
        return None

    total_transactions = len(baskets)

    # Find transactions containing the target product
    target_baskets = {
        txn: products
        for txn, products in baskets.items()
        if target_product in products
    }
    target_transactions = len(target_baskets)

    if target_transactions == 0:
        return None

    # Count co-occurrences of other products with the target
    co_counts: Counter[str] = Counter()
    for products in target_baskets.values():
        for p in products:
            if p != target_product:
                co_counts[p] += 1

    if not co_counts:
        return CrossSellResult(
            target_product=target_product,
            target_transactions=target_transactions,
            recommendations=[],
            summary=(
                f"No cross-sell opportunities found for '{target_product}'. "
                f"It appears in {target_transactions} transactions but always alone."
            ),
        )

    # Compute product-level support across all transactions
    product_counts: Counter[str] = Counter()
    for products in baskets.values():
        for p in products:
            product_counts[p] += 1

    recommendations: list[CrossSellProduct] = []
    for product, co_count in co_counts.most_common():
        co_purchase_rate = co_count / target_transactions
        # Lift: P(product | target) / P(product)
        product_support = product_counts[product] / total_transactions
        lift = co_purchase_rate / product_support if product_support > 0 else 0.0

        recommendations.append(CrossSellProduct(
            product=product,
            co_purchase_rate=round(co_purchase_rate, 4),
            lift=round(lift, 4),
            co_occurrence_count=co_count,
        ))

    # Sort by co_purchase_rate descending
    recommendations.sort(key=lambda r: r.co_purchase_rate, reverse=True)

    summary = (
        f"Cross-sell for '{target_product}': {len(recommendations)} candidate(s) "
        f"from {target_transactions} transactions."
    )
    if recommendations:
        top = recommendations[0]
        summary += (
            f" Top recommendation: {top.product} "
            f"(co-purchase rate={top.co_purchase_rate:.0%}, lift={top.lift:.2f})."
        )

    return CrossSellResult(
        target_product=target_product,
        target_transactions=target_transactions,
        recommendations=recommendations,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. analyze_product_frequency
# ---------------------------------------------------------------------------


def analyze_product_frequency(
    rows: list[dict],
    transaction_column: str,
    product_column: str,
    customer_column: str | None = None,
) -> ProductFrequencyResult | None:
    """Analyze how often each product appears across transactions.

    Args:
        rows: Data rows as dicts.
        transaction_column: Column identifying unique transactions.
        product_column: Column containing product names.
        customer_column: Optional column for customer identifier.

    Returns:
        ProductFrequencyResult or None if insufficient data.
    """
    if not rows:
        return None

    baskets = _build_baskets(rows, transaction_column, product_column)
    if baskets is None:
        return None

    total_transactions = len(baskets)

    # Count transactions per product
    product_txn_counts: Counter[str] = Counter()
    for products in baskets.values():
        for p in products:
            product_txn_counts[p] += 1

    # Unique customers per product (if customer_column provided)
    product_customers: dict[str, set[str]] | None = None
    if customer_column is not None:
        product_customers = defaultdict(set)
        for row in rows:
            product = row.get(product_column)
            customer = row.get(customer_column)
            if product is None or customer is None:
                continue
            product_customers[str(product)].add(str(customer))

    # Build ranked list
    ranked = product_txn_counts.most_common()
    products: list[ProductFreq] = []
    for rank_idx, (product, freq) in enumerate(ranked, start=1):
        pct = (freq / total_transactions * 100) if total_transactions > 0 else 0.0
        unique_cust: int | None = None
        if product_customers is not None:
            unique_cust = len(product_customers.get(product, set()))
        products.append(ProductFreq(
            product=product,
            frequency=freq,
            pct_of_transactions=round(pct, 2),
            unique_customers=unique_cust,
            rank=rank_idx,
        ))

    total_products = len(products)
    most_popular = products[0].product if products else ""
    least_popular = products[-1].product if products else ""

    summary = (
        f"{total_products} products across {total_transactions} transactions. "
        f"Most popular: {most_popular} ({products[0].frequency} transactions). "
        f"Least popular: {least_popular} ({products[-1].frequency} transactions)."
    )

    return ProductFrequencyResult(
        products=products,
        total_transactions=total_transactions,
        total_products=total_products,
        most_popular=most_popular,
        least_popular=least_popular,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_basket_report
# ---------------------------------------------------------------------------


def format_basket_report(
    associations: AssociationResult | None = None,
    basket_size: BasketSizeResult | None = None,
    cross_sell: CrossSellResult | None = None,
    frequency: ProductFrequencyResult | None = None,
) -> str:
    """Combine market basket analyses into a formatted text report.

    Args:
        associations: Product association result.
        basket_size: Basket size distribution result.
        cross_sell: Cross-sell opportunity result.
        frequency: Product frequency result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("MARKET BASKET ANALYSIS REPORT")
    sections.append("=" * 60)

    if associations:
        lines = [
            "",
            "PRODUCT ASSOCIATIONS",
            "-" * 40,
            f"  Total Transactions: {associations.total_transactions:>10}",
            f"  Unique Products:    {associations.unique_products:>10}",
            f"  Avg Basket Size:    {associations.avg_basket_size:>10.1f}",
            f"  Pairs Found:        {len(associations.pairs):>10}",
        ]
        if associations.pairs:
            lines.append("")
            lines.append("  Top Pairs (by lift):")
            for pair in associations.pairs[:10]:
                lines.append(
                    f"    {pair.product_a} & {pair.product_b:<20} "
                    f"lift={pair.lift:.2f}  support={pair.support:.4f}  "
                    f"count={pair.co_occurrence_count}"
                )
        sections.append("\n".join(lines))

    if basket_size:
        lines = [
            "",
            "BASKET SIZE DISTRIBUTION",
            "-" * 40,
            f"  Avg Size:           {basket_size.avg_size:>10.1f}",
            f"  Median Size:        {basket_size.median_size:>10.1f}",
            f"  Min Size:           {basket_size.min_size:>10}",
            f"  Max Size:           {basket_size.max_size:>10}",
        ]
        if basket_size.avg_value is not None:
            lines.append(f"  Avg Basket Value:   {basket_size.avg_value:>10,.2f}")
        lines.append("")
        lines.append("  Distribution:")
        for bucket in basket_size.distribution:
            lines.append(f"    Size {bucket.size:<5} {bucket.count:>6} ({bucket.pct:.1f}%)")
        sections.append("\n".join(lines))

    if cross_sell:
        lines = [
            "",
            "CROSS-SELL OPPORTUNITIES",
            "-" * 40,
            f"  Target Product:     {cross_sell.target_product}",
            f"  Target Transactions:{cross_sell.target_transactions:>10}",
            f"  Candidates Found:   {len(cross_sell.recommendations):>10}",
        ]
        if cross_sell.recommendations:
            lines.append("")
            lines.append("  Recommendations:")
            for rec in cross_sell.recommendations[:10]:
                lines.append(
                    f"    {rec.product:<20} rate={rec.co_purchase_rate:.0%}  "
                    f"lift={rec.lift:.2f}  count={rec.co_occurrence_count}"
                )
        sections.append("\n".join(lines))

    if frequency:
        lines = [
            "",
            "PRODUCT FREQUENCY",
            "-" * 40,
            f"  Total Products:     {frequency.total_products:>10}",
            f"  Total Transactions: {frequency.total_transactions:>10}",
            f"  Most Popular:       {frequency.most_popular}",
            f"  Least Popular:      {frequency.least_popular}",
            "",
            "  Product Rankings:",
        ]
        for pf in frequency.products[:15]:
            cust_info = ""
            if pf.unique_customers is not None:
                cust_info = f"  customers={pf.unique_customers}"
            lines.append(
                f"    {pf.rank:>3}. {pf.product:<20} "
                f"{pf.frequency:>6} txns ({pf.pct_of_transactions:.1f}%)"
                f"{cust_info}"
            )
        sections.append("\n".join(lines))

    if not any([associations, basket_size, cross_sell, frequency]):
        sections.append("\nNo data available for report.")

    sections.append("\n" + "=" * 60)
    return "\n".join(sections)
