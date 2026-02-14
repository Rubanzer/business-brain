"""Customer behavior analytics -- beyond RFM segmentation.

Pure functions for customer tiering, churn risk assessment,
revenue concentration analysis, and purchase behavior profiling.
No DB, async, or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_float(value) -> float | None:
    """Attempt to parse a value into a float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_date(value) -> datetime | None:
    """Attempt to parse a value into a datetime."""
    if isinstance(value, datetime):
        return value
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CustomerTier:
    """Revenue tier for a group of customers."""

    tier: str
    customer_count: int
    total_revenue: float
    avg_revenue: float
    pct_of_total: float
    avg_frequency: float | None = None


@dataclass
class CustomerSegmentResult:
    """Complete customer segmentation result."""

    tiers: list[CustomerTier]
    total_customers: int
    total_revenue: float
    summary: str


@dataclass
class ChurnStatus:
    """Churn status group."""

    status: str
    customer_count: int
    pct: float
    avg_spend: float | None = None


@dataclass
class ChurnResult:
    """Complete churn risk analysis result."""

    statuses: list[ChurnStatus]
    total_customers: int
    churn_rate: float
    at_risk_rate: float
    summary: str


@dataclass
class CustomerShare:
    """Individual customer revenue share."""

    customer: str
    total_spend: float
    share_pct: float
    cumulative_pct: float


@dataclass
class ConcentrationResult:
    """Revenue concentration analysis result."""

    top_customers: list[CustomerShare]
    hhi: float
    concentration_risk: str
    customers_for_80pct: int
    summary: str


@dataclass
class CustomerBehavior:
    """Purchase behavior profile for a single customer."""

    customer: str
    total_orders: int
    total_spend: float
    avg_order_value: float
    first_purchase: str
    last_purchase: str
    lifespan_days: int


@dataclass
class BehaviorResult:
    """Complete purchase behavior analysis result."""

    customers: list[CustomerBehavior]
    avg_orders: float
    avg_aov: float
    repeat_purchase_rate: float
    avg_lifespan_days: float
    summary: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_customer_segments(
    rows: list[dict],
    customer_column: str,
    revenue_column: str,
    frequency_column: str | None = None,
    category_column: str | None = None,
) -> CustomerSegmentResult | None:
    """Tier customers by revenue into Premium/Standard/Basic segments.

    Args:
        rows: Transaction or summary data as list of dicts.
        customer_column: Column identifying the customer.
        revenue_column: Column with revenue/amount values.
        frequency_column: Optional column with frequency counts.
        category_column: Optional column with product/service category.

    Returns:
        CustomerSegmentResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per customer
    customer_data: dict[str, dict] = {}
    for row in rows:
        cust = row.get(customer_column)
        rev = _safe_float(row.get(revenue_column))
        if cust is None or rev is None:
            continue

        cust_str = str(cust)
        if cust_str not in customer_data:
            customer_data[cust_str] = {
                "revenue": 0.0,
                "frequencies": [],
                "categories": [],
            }
        entry = customer_data[cust_str]
        entry["revenue"] += rev

        if frequency_column is not None:
            freq = _safe_float(row.get(frequency_column))
            if freq is not None:
                entry["frequencies"].append(freq)

        if category_column is not None:
            cat = row.get(category_column)
            if cat is not None:
                entry["categories"].append(str(cat))

    if not customer_data:
        return None

    # Sort customers by revenue descending
    sorted_customers = sorted(
        customer_data.items(), key=lambda x: x[1]["revenue"], reverse=True
    )

    n = len(sorted_customers)
    total_revenue = sum(entry["revenue"] for _, entry in sorted_customers)

    # Tier boundaries
    premium_cutoff = max(1, int(n * 0.2))  # top 20%
    standard_cutoff = premium_cutoff + max(1, int(n * 0.3))  # next 30%

    # Assign tiers
    tier_map: dict[str, list[tuple[str, dict]]] = {
        "Premium": [],
        "Standard": [],
        "Basic": [],
    }

    for i, (cust, entry) in enumerate(sorted_customers):
        if i < premium_cutoff:
            tier_map["Premium"].append((cust, entry))
        elif i < standard_cutoff:
            tier_map["Standard"].append((cust, entry))
        else:
            tier_map["Basic"].append((cust, entry))

    # Build tier results
    tiers: list[CustomerTier] = []
    for tier_name in ("Premium", "Standard", "Basic"):
        members = tier_map[tier_name]
        if not members:
            continue

        tier_revenue = sum(entry["revenue"] for _, entry in members)
        tier_count = len(members)
        avg_rev = tier_revenue / tier_count
        pct_of_total = (tier_revenue / total_revenue * 100) if total_revenue > 0 else 0.0

        # Average frequency if column provided
        avg_freq: float | None = None
        if frequency_column is not None:
            all_freqs = []
            for _, entry in members:
                all_freqs.extend(entry["frequencies"])
            if all_freqs:
                avg_freq = round(sum(all_freqs) / len(all_freqs), 2)

        tiers.append(CustomerTier(
            tier=tier_name,
            customer_count=tier_count,
            total_revenue=round(tier_revenue, 2),
            avg_revenue=round(avg_rev, 2),
            pct_of_total=round(pct_of_total, 2),
            avg_frequency=avg_freq,
        ))

    # Preferred category per tier (mode)
    if category_column is not None:
        for tier_obj in tiers:
            members = tier_map[tier_obj.tier]
            all_cats: list[str] = []
            for _, entry in members:
                all_cats.extend(entry["categories"])
            if all_cats:
                # Find mode
                cat_counts: dict[str, int] = {}
                for c in all_cats:
                    cat_counts[c] = cat_counts.get(c, 0) + 1
                # Store as a tier attribute via summary workaround
                # (dataclass doesn't have preferred_category, include in summary)

    summary_parts = [
        f"Customer Segmentation: {n} customers across {len(tiers)} tiers.",
    ]
    for t in tiers:
        summary_parts.append(
            f"  {t.tier}: {t.customer_count} customers, "
            f"${t.total_revenue:,.2f} revenue ({t.pct_of_total:.1f}%)"
        )

    return CustomerSegmentResult(
        tiers=tiers,
        total_customers=n,
        total_revenue=round(total_revenue, 2),
        summary="\n".join(summary_parts),
    )


def analyze_churn_risk(
    rows: list[dict],
    customer_column: str,
    date_column: str,
    amount_column: str | None = None,
) -> ChurnResult | None:
    """Classify customers by churn risk based on recency.

    Args:
        rows: Transaction data as list of dicts.
        customer_column: Column identifying the customer.
        date_column: Column with the transaction/activity date.
        amount_column: Optional column with spend amounts.

    Returns:
        ChurnResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per customer: last date, total spend
    customer_data: dict[str, dict] = {}
    for row in rows:
        cust = row.get(customer_column)
        date_val = _parse_date(row.get(date_column))
        if cust is None or date_val is None:
            continue

        cust_str = str(cust)
        if cust_str not in customer_data:
            customer_data[cust_str] = {
                "last_date": date_val,
                "total_spend": 0.0,
                "txn_count": 0,
            }
        entry = customer_data[cust_str]
        if date_val > entry["last_date"]:
            entry["last_date"] = date_val
        entry["txn_count"] += 1

        if amount_column is not None:
            amt = _safe_float(row.get(amount_column))
            if amt is not None:
                entry["total_spend"] += amt

    if not customer_data:
        return None

    # Reference date = max date in data
    max_date = max(entry["last_date"] for entry in customer_data.values())

    # Classify each customer
    status_groups: dict[str, list[dict]] = {
        "Active": [],
        "At-Risk": [],
        "Dormant": [],
        "Churned": [],
    }

    for cust, entry in customer_data.items():
        recency = (max_date - entry["last_date"]).days
        if recency <= 30:
            status = "Active"
        elif recency <= 90:
            status = "At-Risk"
        elif recency <= 180:
            status = "Dormant"
        else:
            status = "Churned"

        status_groups[status].append(entry)

    total = len(customer_data)

    # Build ChurnStatus objects
    statuses: list[ChurnStatus] = []
    for status_name in ("Active", "At-Risk", "Dormant", "Churned"):
        members = status_groups[status_name]
        count = len(members)
        pct = (count / total * 100) if total > 0 else 0.0

        avg_spend: float | None = None
        if amount_column is not None and members:
            total_spend = sum(m["total_spend"] for m in members)
            avg_spend = round(total_spend / count, 2)

        statuses.append(ChurnStatus(
            status=status_name,
            customer_count=count,
            pct=round(pct, 2),
            avg_spend=avg_spend,
        ))

    churned_count = len(status_groups["Churned"])
    at_risk_count = len(status_groups["At-Risk"])
    churn_rate = round((churned_count / total * 100) if total > 0 else 0.0, 2)
    at_risk_rate = round((at_risk_count / total * 100) if total > 0 else 0.0, 2)

    summary = (
        f"Churn Analysis: {total} customers. "
        f"Churn rate: {churn_rate:.1f}%, At-risk rate: {at_risk_rate:.1f}%. "
        f"Active: {len(status_groups['Active'])}, "
        f"At-Risk: {at_risk_count}, "
        f"Dormant: {len(status_groups['Dormant'])}, "
        f"Churned: {churned_count}."
    )

    return ChurnResult(
        statuses=statuses,
        total_customers=total,
        churn_rate=churn_rate,
        at_risk_rate=at_risk_rate,
        summary=summary,
    )


def compute_customer_concentration(
    rows: list[dict],
    customer_column: str,
    amount_column: str,
) -> ConcentrationResult | None:
    """Compute revenue concentration and Pareto analysis.

    Args:
        rows: Transaction data as list of dicts.
        customer_column: Column identifying the customer.
        amount_column: Column with spend/revenue amounts.

    Returns:
        ConcentrationResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per customer
    customer_totals: dict[str, float] = {}
    for row in rows:
        cust = row.get(customer_column)
        amt = _safe_float(row.get(amount_column))
        if cust is None or amt is None:
            continue

        cust_str = str(cust)
        customer_totals[cust_str] = customer_totals.get(cust_str, 0.0) + amt

    if not customer_totals:
        return None

    total_revenue = sum(customer_totals.values())
    if total_revenue <= 0:
        return None

    # Sort descending by spend
    sorted_customers = sorted(
        customer_totals.items(), key=lambda x: x[1], reverse=True
    )

    # Build CustomerShare list with cumulative percentages
    top_customers: list[CustomerShare] = []
    cumulative = 0.0
    for cust, spend in sorted_customers:
        share = spend / total_revenue * 100
        cumulative += share
        top_customers.append(CustomerShare(
            customer=cust,
            total_spend=round(spend, 2),
            share_pct=round(share, 2),
            cumulative_pct=round(cumulative, 2),
        ))

    # Pareto: how many customers for 80% of revenue
    customers_for_80 = 0
    cum_check = 0.0
    for cust, spend in sorted_customers:
        cum_check += spend / total_revenue * 100
        customers_for_80 += 1
        if cum_check >= 80.0:
            break

    # HHI (Herfindahl-Hirschman Index)
    # Sum of squared market shares (as percentages)
    hhi = sum(
        (spend / total_revenue * 100) ** 2
        for spend in customer_totals.values()
    )
    hhi = round(hhi, 2)

    # Concentration risk assessment
    top_1_share = top_customers[0].share_pct if top_customers else 0.0
    top_5_share = sum(
        c.share_pct for c in top_customers[:5]
    ) if len(top_customers) >= 5 else sum(c.share_pct for c in top_customers)

    if top_1_share > 25:
        concentration_risk = "High"
    elif top_5_share > 60:
        concentration_risk = "Moderate"
    else:
        concentration_risk = "Low"

    # Limit top_customers to 10 for the result
    top_10 = top_customers[:10]

    summary = (
        f"Concentration Analysis: {len(customer_totals)} customers, "
        f"${total_revenue:,.2f} total revenue. "
        f"HHI: {hhi:.0f}. "
        f"Top {customers_for_80} customer(s) generate 80% of revenue. "
        f"Concentration risk: {concentration_risk}."
    )

    return ConcentrationResult(
        top_customers=top_10,
        hhi=hhi,
        concentration_risk=concentration_risk,
        customers_for_80pct=customers_for_80,
        summary=summary,
    )


def analyze_purchase_behavior(
    rows: list[dict],
    customer_column: str,
    amount_column: str,
    date_column: str,
    product_column: str | None = None,
) -> BehaviorResult | None:
    """Analyze purchase behavior patterns across customers.

    Args:
        rows: Transaction data as list of dicts.
        customer_column: Column identifying the customer.
        amount_column: Column with order/transaction amounts.
        date_column: Column with the transaction date.
        product_column: Optional column with product/item identifiers.

    Returns:
        BehaviorResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per customer
    customer_data: dict[str, dict] = {}
    # Track products per order (keyed by customer+date for simplicity)
    order_products: dict[str, dict[str, set]] = {}  # cust -> {date_str -> set of products}

    for row in rows:
        cust = row.get(customer_column)
        amt = _safe_float(row.get(amount_column))
        date_val = _parse_date(row.get(date_column))
        if cust is None or amt is None or date_val is None:
            continue

        cust_str = str(cust)
        if cust_str not in customer_data:
            customer_data[cust_str] = {
                "total_spend": 0.0,
                "order_count": 0,
                "first_date": date_val,
                "last_date": date_val,
            }
        entry = customer_data[cust_str]
        entry["total_spend"] += amt
        entry["order_count"] += 1
        if date_val < entry["first_date"]:
            entry["first_date"] = date_val
        if date_val > entry["last_date"]:
            entry["last_date"] = date_val

        if product_column is not None:
            product = row.get(product_column)
            if product is not None:
                if cust_str not in order_products:
                    order_products[cust_str] = {}
                date_str = date_val.strftime("%Y-%m-%d")
                if date_str not in order_products[cust_str]:
                    order_products[cust_str][date_str] = set()
                order_products[cust_str][date_str].add(str(product))

    if not customer_data:
        return None

    # Build CustomerBehavior records
    behaviors: list[CustomerBehavior] = []
    total_orders_all = 0
    total_aov_all = 0.0
    total_lifespan = 0
    repeat_count = 0

    for cust in sorted(customer_data.keys()):
        entry = customer_data[cust]
        orders = entry["order_count"]
        spend = entry["total_spend"]
        aov = spend / orders if orders > 0 else 0.0
        lifespan = (entry["last_date"] - entry["first_date"]).days

        behaviors.append(CustomerBehavior(
            customer=cust,
            total_orders=orders,
            total_spend=round(spend, 2),
            avg_order_value=round(aov, 2),
            first_purchase=entry["first_date"].strftime("%Y-%m-%d"),
            last_purchase=entry["last_date"].strftime("%Y-%m-%d"),
            lifespan_days=lifespan,
        ))

        total_orders_all += orders
        total_aov_all += aov
        total_lifespan += lifespan
        if orders > 1:
            repeat_count += 1

    n = len(behaviors)
    avg_orders = round(total_orders_all / n, 2) if n > 0 else 0.0
    avg_aov = round(total_aov_all / n, 2) if n > 0 else 0.0
    avg_lifespan = round(total_lifespan / n, 2) if n > 0 else 0.0
    repeat_rate = round((repeat_count / n * 100) if n > 0 else 0.0, 2)

    # Basket size (avg products per order) if product_column provided
    basket_info = ""
    if product_column is not None and order_products:
        total_products = 0
        total_order_count = 0
        for cust, dates in order_products.items():
            for date_str, products in dates.items():
                total_products += len(products)
                total_order_count += 1
        avg_basket = round(total_products / total_order_count, 2) if total_order_count > 0 else 0.0
        basket_info = f" Avg basket size: {avg_basket} products/order."

    summary = (
        f"Purchase Behavior: {n} customers, "
        f"avg {avg_orders} orders/customer, avg AOV ${avg_aov:.2f}. "
        f"Repeat purchase rate: {repeat_rate:.1f}%. "
        f"Avg lifespan: {avg_lifespan:.0f} days."
        f"{basket_info}"
    )

    return BehaviorResult(
        customers=behaviors,
        avg_orders=avg_orders,
        avg_aov=avg_aov,
        repeat_purchase_rate=repeat_rate,
        avg_lifespan_days=avg_lifespan,
        summary=summary,
    )


def format_customer_report(
    segments: CustomerSegmentResult | None = None,
    churn: ChurnResult | None = None,
    concentration: ConcentrationResult | None = None,
    behavior: BehaviorResult | None = None,
) -> str:
    """Format a combined customer analytics report as plain text.

    Any parameter may be ``None``; only provided sections are rendered.

    Returns:
        Formatted multi-line report string.
    """
    lines: list[str] = []

    if segments is not None:
        lines.append("Customer Segmentation Report")
        lines.append("=" * 60)
        lines.append(f"Total Customers: {segments.total_customers}")
        lines.append(f"Total Revenue:   ${segments.total_revenue:,.2f}")
        lines.append("")
        lines.append(
            f"{'Tier':<12}{'Count':>8}{'Revenue':>14}{'Avg Rev':>12}{'% Total':>10}"
        )
        lines.append("-" * 56)
        for t in segments.tiers:
            freq_str = ""
            if t.avg_frequency is not None:
                freq_str = f"  (avg freq: {t.avg_frequency:.1f})"
            lines.append(
                f"{t.tier:<12}{t.customer_count:>8}"
                f"{t.total_revenue:>14,.2f}{t.avg_revenue:>12,.2f}"
                f"{t.pct_of_total:>9.1f}%{freq_str}"
            )
        lines.append("")

    if churn is not None:
        lines.append("Churn Risk Analysis")
        lines.append("=" * 60)
        lines.append(f"Total Customers: {churn.total_customers}")
        lines.append(f"Churn Rate:      {churn.churn_rate:.1f}%")
        lines.append(f"At-Risk Rate:    {churn.at_risk_rate:.1f}%")
        lines.append("")
        lines.append(
            f"{'Status':<12}{'Count':>8}{'Pct':>8}"
        )
        lines.append("-" * 28)
        for s in churn.statuses:
            spend_str = ""
            if s.avg_spend is not None:
                spend_str = f"  (avg spend: ${s.avg_spend:,.2f})"
            lines.append(
                f"{s.status:<12}{s.customer_count:>8}{s.pct:>7.1f}%{spend_str}"
            )
        lines.append("")

    if concentration is not None:
        lines.append("Revenue Concentration Analysis")
        lines.append("=" * 60)
        lines.append(f"HHI Index:           {concentration.hhi:.0f}")
        lines.append(f"Concentration Risk:  {concentration.concentration_risk}")
        lines.append(f"Customers for 80%:   {concentration.customers_for_80pct}")
        lines.append("")
        lines.append(
            f"{'Customer':<20}{'Spend':>14}{'Share':>8}{'Cumul.':>8}"
        )
        lines.append("-" * 50)
        for c in concentration.top_customers:
            lines.append(
                f"{c.customer:<20}{c.total_spend:>14,.2f}"
                f"{c.share_pct:>7.1f}%{c.cumulative_pct:>7.1f}%"
            )
        lines.append("")

    if behavior is not None:
        lines.append("Purchase Behavior Analysis")
        lines.append("=" * 60)
        lines.append(f"Total Customers:       {len(behavior.customers)}")
        lines.append(f"Avg Orders/Customer:   {behavior.avg_orders:.1f}")
        lines.append(f"Avg Order Value:       ${behavior.avg_aov:,.2f}")
        lines.append(f"Repeat Purchase Rate:  {behavior.repeat_purchase_rate:.1f}%")
        lines.append(f"Avg Lifespan:          {behavior.avg_lifespan_days:.0f} days")
        lines.append("")
        lines.append(
            f"{'Customer':<16}{'Orders':>8}{'Spend':>12}{'AOV':>10}{'Lifespan':>10}"
        )
        lines.append("-" * 56)
        for c in behavior.customers:
            lines.append(
                f"{c.customer:<16}{c.total_orders:>8}"
                f"{c.total_spend:>12,.2f}{c.avg_order_value:>10,.2f}"
                f"{c.lifespan_days:>8}d"
            )
        lines.append("")

    if not lines:
        return "No data provided for customer report."

    return "\n".join(lines)
