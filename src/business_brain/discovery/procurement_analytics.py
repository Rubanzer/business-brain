"""Procurement analytics — PO analysis, price variance, vendor performance, spend categories.

Pure functions for manufacturing procurement analysis including purchase order
aggregation by vendor with monthly trends, purchase price variance (PPV)
computation, vendor delivery performance scoring, and spend-by-category
concentration analysis.
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
    """Parse a date value (datetime or ISO-format string) to datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    try:
        return datetime.fromisoformat(str(val))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VendorOrderSummary:
    """Aggregated PO stats for a single vendor."""
    vendor: str
    order_count: int
    total_value: float
    avg_value: float


@dataclass
class MonthlyPOTrend:
    """PO volume and value for a single month."""
    month: str
    order_count: int
    total_value: float


@dataclass
class PurchaseOrderResult:
    """Complete purchase order analysis result."""
    total_orders: int
    total_value: float
    vendors: list[VendorOrderSummary]
    monthly_trends: list[MonthlyPOTrend]
    status_breakdown: dict[str, int]
    summary: str


@dataclass
class ItemVariance:
    """PPV detail for a single item."""
    item: str
    quantity: float
    avg_actual_price: float
    standard_price: float | None
    total_variance: float | None
    variance_type: str | None  # "favorable", "unfavorable", or None


@dataclass
class PriceVarianceResult:
    """Complete purchase price variance result."""
    items: list[ItemVariance]
    total_variance: float
    favorable_count: int
    unfavorable_count: int
    summary: str


@dataclass
class VendorDelivery:
    """Delivery performance for a single vendor."""
    vendor: str
    total_deliveries: int
    on_time_count: int
    late_count: int
    on_time_pct: float
    avg_days_late: float
    avg_quality: float | None
    total_quantity: float | None


@dataclass
class VendorPerfResult:
    """Complete vendor performance result."""
    vendors: list[VendorDelivery]
    overall_on_time_pct: float
    avg_quality: float | None
    summary: str


@dataclass
class SpendCategory:
    """Spend breakdown for a single category."""
    category: str
    total_spend: float
    pct_of_total: float
    transaction_count: int
    vendor_count: int | None


@dataclass
class SpendCategoryResult:
    """Complete spend-by-category analysis result."""
    categories: list[SpendCategory]
    total_spend: float
    hhi: float
    concentration_risk: str  # "low", "moderate", "high"
    top_categories: list[str]
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_purchase_orders
# ---------------------------------------------------------------------------


def analyze_purchase_orders(
    rows: list[dict],
    po_column: str,
    vendor_column: str,
    amount_column: str,
    date_column: str | None = None,
    status_column: str | None = None,
) -> PurchaseOrderResult | None:
    """Analyze purchase orders grouped by vendor with optional trends and status.

    Args:
        rows: Data rows as dicts.
        po_column: Column identifying the purchase order.
        vendor_column: Column identifying the vendor.
        amount_column: Column with the PO monetary value.
        date_column: Optional column with PO date for monthly trends.
        status_column: Optional column with PO status (open/closed/cancelled).

    Returns:
        PurchaseOrderResult or None if no valid data.
    """
    if not rows:
        return None

    # --- Aggregate by vendor ------------------------------------------------
    vendor_orders: dict[str, list[float]] = {}
    monthly_agg: dict[str, dict] = {}  # month_key -> {count, value}
    status_counts: dict[str, int] = {}
    total_value = 0.0
    valid_count = 0

    for row in rows:
        po = row.get(po_column)
        vendor = row.get(vendor_column)
        amt_raw = row.get(amount_column)
        if po is None or vendor is None:
            continue
        amt = _safe_float(amt_raw)
        if amt is None:
            continue

        vendor_key = str(vendor)
        vendor_orders.setdefault(vendor_key, []).append(amt)
        total_value += amt
        valid_count += 1

        # Monthly trend
        if date_column is not None:
            dt = _parse_date(row.get(date_column))
            if dt is not None:
                month_key = dt.strftime("%Y-%m")
                if month_key not in monthly_agg:
                    monthly_agg[month_key] = {"count": 0, "value": 0.0}
                monthly_agg[month_key]["count"] += 1
                monthly_agg[month_key]["value"] += amt

        # Status breakdown
        if status_column is not None:
            status = row.get(status_column)
            if status is not None:
                s_key = str(status).lower()
                status_counts[s_key] = status_counts.get(s_key, 0) + 1

    if not vendor_orders:
        return None

    # --- Build vendor summaries ---------------------------------------------
    vendors: list[VendorOrderSummary] = []
    for v, amounts in vendor_orders.items():
        total_v = sum(amounts)
        count_v = len(amounts)
        vendors.append(
            VendorOrderSummary(
                vendor=v,
                order_count=count_v,
                total_value=round(total_v, 4),
                avg_value=round(total_v / count_v, 4) if count_v > 0 else 0.0,
            )
        )

    # Sort by total_value descending
    vendors.sort(key=lambda v: -v.total_value)

    # --- Monthly trends (sorted by month) -----------------------------------
    monthly_trends: list[MonthlyPOTrend] = []
    for mk in sorted(monthly_agg.keys()):
        monthly_trends.append(
            MonthlyPOTrend(
                month=mk,
                order_count=monthly_agg[mk]["count"],
                total_value=round(monthly_agg[mk]["value"], 4),
            )
        )

    # --- Top 5 vendors by value ---------------------------------------------
    top5_names = [v.vendor for v in vendors[:5]]

    summary = (
        f"{valid_count} purchase orders totalling {total_value:,.2f} "
        f"across {len(vendors)} vendors. "
        f"Top vendors by value: {', '.join(top5_names)}."
    )

    return PurchaseOrderResult(
        total_orders=valid_count,
        total_value=round(total_value, 4),
        vendors=vendors,
        monthly_trends=monthly_trends,
        status_breakdown=status_counts,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. compute_purchase_price_variance
# ---------------------------------------------------------------------------


def compute_purchase_price_variance(
    rows: list[dict],
    item_column: str,
    quantity_column: str,
    actual_price_column: str,
    standard_price_column: str | None = None,
) -> PriceVarianceResult | None:
    """Compute purchase price variance (PPV) per item.

    PPV per row = (actual_price - standard_price) * quantity.
    Favorable = actual < standard (negative variance).

    Args:
        rows: Data rows as dicts.
        item_column: Column identifying the item/material.
        quantity_column: Column with the purchased quantity.
        actual_price_column: Column with the actual unit price paid.
        standard_price_column: Optional column with the standard/budgeted price.

    Returns:
        PriceVarianceResult or None if no valid data.
    """
    if not rows:
        return None

    # Per-item accumulators: {item: {qty, spend, variances, std_price_vals}}
    item_agg: dict[str, dict] = {}

    for row in rows:
        item = row.get(item_column)
        qty_raw = row.get(quantity_column)
        actual_raw = row.get(actual_price_column)
        if item is None:
            continue
        qty = _safe_float(qty_raw)
        actual = _safe_float(actual_raw)
        if qty is None or actual is None:
            continue

        item_key = str(item)
        if item_key not in item_agg:
            item_agg[item_key] = {
                "qty": 0.0,
                "spend": 0.0,
                "actual_prices": [],
                "variance": 0.0,
                "has_std": False,
                "std_prices": [],
            }

        item_agg[item_key]["qty"] += qty
        item_agg[item_key]["spend"] += actual * qty
        item_agg[item_key]["actual_prices"].append(actual)

        if standard_price_column is not None:
            std_raw = row.get(standard_price_column)
            std = _safe_float(std_raw)
            if std is not None:
                ppv = (actual - std) * qty
                item_agg[item_key]["variance"] += ppv
                item_agg[item_key]["has_std"] = True
                item_agg[item_key]["std_prices"].append(std)

    if not item_agg:
        return None

    # --- Build item variance list -------------------------------------------
    items: list[ItemVariance] = []
    total_variance = 0.0
    favorable_count = 0
    unfavorable_count = 0

    for item_key, agg in item_agg.items():
        avg_actual = agg["spend"] / agg["qty"] if agg["qty"] != 0 else 0.0
        std_price: float | None = None
        item_variance: float | None = None
        vtype: str | None = None

        if agg["has_std"] and agg["std_prices"]:
            std_price = sum(agg["std_prices"]) / len(agg["std_prices"])
            item_variance = agg["variance"]
            total_variance += item_variance
            if item_variance < 0:
                vtype = "favorable"
                favorable_count += 1
            elif item_variance > 0:
                vtype = "unfavorable"
                unfavorable_count += 1
            else:
                vtype = "neutral"

        items.append(
            ItemVariance(
                item=item_key,
                quantity=round(agg["qty"], 4),
                avg_actual_price=round(avg_actual, 4),
                standard_price=round(std_price, 4) if std_price is not None else None,
                total_variance=round(item_variance, 4) if item_variance is not None else None,
                variance_type=vtype,
            )
        )

    # Sort by absolute variance descending (highest impact first)
    items.sort(key=lambda it: -abs(it.total_variance or 0.0))

    var_label = "favorable" if total_variance < 0 else "unfavorable" if total_variance > 0 else "neutral"
    summary = (
        f"Price variance across {len(items)} items: "
        f"total variance = {total_variance:+,.2f} ({var_label}). "
        f"Favorable: {favorable_count}, Unfavorable: {unfavorable_count}."
    )

    return PriceVarianceResult(
        items=items,
        total_variance=round(total_variance, 4),
        favorable_count=favorable_count,
        unfavorable_count=unfavorable_count,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. analyze_vendor_performance
# ---------------------------------------------------------------------------


def analyze_vendor_performance(
    rows: list[dict],
    vendor_column: str,
    delivery_date_column: str,
    promised_date_column: str,
    quality_column: str | None = None,
    quantity_column: str | None = None,
) -> VendorPerfResult | None:
    """Analyze vendor delivery performance (on-time vs late).

    A delivery is on-time if delivery_date <= promised_date.

    Args:
        rows: Data rows as dicts.
        vendor_column: Column identifying the vendor.
        delivery_date_column: Column with the actual delivery date.
        promised_date_column: Column with the promised delivery date.
        quality_column: Optional column with a numeric quality score.
        quantity_column: Optional column with delivered quantity.

    Returns:
        VendorPerfResult or None if no valid data.
    """
    if not rows:
        return None

    # Per-vendor accumulators
    vendor_agg: dict[str, dict] = {}

    for row in rows:
        vendor = row.get(vendor_column)
        if vendor is None:
            continue
        delivery_dt = _parse_date(row.get(delivery_date_column))
        promised_dt = _parse_date(row.get(promised_date_column))
        if delivery_dt is None or promised_dt is None:
            continue

        v_key = str(vendor)
        if v_key not in vendor_agg:
            vendor_agg[v_key] = {
                "total": 0,
                "on_time": 0,
                "late": 0,
                "late_days": [],
                "quality_scores": [],
                "quantity": 0.0,
            }

        vendor_agg[v_key]["total"] += 1
        diff_days = (delivery_dt - promised_dt).days
        if diff_days <= 0:
            vendor_agg[v_key]["on_time"] += 1
        else:
            vendor_agg[v_key]["late"] += 1
            vendor_agg[v_key]["late_days"].append(diff_days)

        if quality_column is not None:
            q = _safe_float(row.get(quality_column))
            if q is not None:
                vendor_agg[v_key]["quality_scores"].append(q)

        if quantity_column is not None:
            qty = _safe_float(row.get(quantity_column))
            if qty is not None:
                vendor_agg[v_key]["quantity"] += qty

    if not vendor_agg:
        return None

    # --- Build vendor delivery list -----------------------------------------
    vendors: list[VendorDelivery] = []
    total_deliveries_all = 0
    total_on_time_all = 0
    all_quality: list[float] = []

    for v_key, agg in vendor_agg.items():
        total_d = agg["total"]
        on_time = agg["on_time"]
        late = agg["late"]
        on_time_pct = (on_time / total_d * 100) if total_d > 0 else 0.0
        avg_late = (
            sum(agg["late_days"]) / len(agg["late_days"])
            if agg["late_days"]
            else 0.0
        )
        avg_q: float | None = None
        if agg["quality_scores"]:
            avg_q = sum(agg["quality_scores"]) / len(agg["quality_scores"])
            all_quality.extend(agg["quality_scores"])

        total_qty: float | None = None
        if quantity_column is not None:
            total_qty = round(agg["quantity"], 4)

        total_deliveries_all += total_d
        total_on_time_all += on_time

        vendors.append(
            VendorDelivery(
                vendor=v_key,
                total_deliveries=total_d,
                on_time_count=on_time,
                late_count=late,
                on_time_pct=round(on_time_pct, 2),
                avg_days_late=round(avg_late, 2),
                avg_quality=round(avg_q, 4) if avg_q is not None else None,
                total_quantity=total_qty,
            )
        )

    # Sort by on_time_pct descending
    vendors.sort(key=lambda v: -v.on_time_pct)

    overall_on_time_pct = (
        (total_on_time_all / total_deliveries_all * 100)
        if total_deliveries_all > 0
        else 0.0
    )
    overall_avg_quality: float | None = None
    if all_quality:
        overall_avg_quality = round(sum(all_quality) / len(all_quality), 4)

    summary = (
        f"Vendor performance across {len(vendors)} vendors, "
        f"{total_deliveries_all} deliveries. "
        f"Overall on-time: {overall_on_time_pct:.1f}%."
    )
    if overall_avg_quality is not None:
        summary += f" Avg quality: {overall_avg_quality:.2f}."

    return VendorPerfResult(
        vendors=vendors,
        overall_on_time_pct=round(overall_on_time_pct, 2),
        avg_quality=overall_avg_quality,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. analyze_spend_by_category
# ---------------------------------------------------------------------------


def analyze_spend_by_category(
    rows: list[dict],
    category_column: str,
    amount_column: str,
    vendor_column: str | None = None,
) -> SpendCategoryResult | None:
    """Analyze procurement spend by category with concentration measurement.

    HHI = sum of squared share percentages across categories.
    - HHI < 1500 => low concentration
    - 1500 <= HHI < 2500 => moderate concentration
    - HHI >= 2500 => high concentration

    Args:
        rows: Data rows as dicts.
        category_column: Column identifying the spend category.
        amount_column: Column with the spend amount.
        vendor_column: Optional column to count unique vendors per category.

    Returns:
        SpendCategoryResult or None if no valid data.
    """
    if not rows:
        return None

    # Per-category accumulators
    cat_agg: dict[str, dict] = {}

    for row in rows:
        cat = row.get(category_column)
        amt_raw = row.get(amount_column)
        if cat is None:
            continue
        amt = _safe_float(amt_raw)
        if amt is None:
            continue

        cat_key = str(cat)
        if cat_key not in cat_agg:
            cat_agg[cat_key] = {"spend": 0.0, "count": 0, "vendors": set()}
        cat_agg[cat_key]["spend"] += amt
        cat_agg[cat_key]["count"] += 1

        if vendor_column is not None:
            vendor = row.get(vendor_column)
            if vendor is not None:
                cat_agg[cat_key]["vendors"].add(str(vendor))

    if not cat_agg:
        return None

    total_spend = sum(v["spend"] for v in cat_agg.values())
    if total_spend == 0:
        return None

    # --- Build category list ------------------------------------------------
    categories: list[SpendCategory] = []
    for cat_key, agg in cat_agg.items():
        pct = (agg["spend"] / total_spend * 100) if total_spend != 0 else 0.0
        vendor_count: int | None = None
        if vendor_column is not None:
            vendor_count = len(agg["vendors"])

        categories.append(
            SpendCategory(
                category=cat_key,
                total_spend=round(agg["spend"], 4),
                pct_of_total=round(pct, 2),
                transaction_count=agg["count"],
                vendor_count=vendor_count,
            )
        )

    # Sort by total_spend descending
    categories.sort(key=lambda c: -c.total_spend)

    # --- HHI ----------------------------------------------------------------
    hhi = sum(c.pct_of_total ** 2 for c in categories)
    hhi = round(hhi, 2)

    if hhi < 1500:
        concentration_risk = "low"
    elif hhi < 2500:
        concentration_risk = "moderate"
    else:
        concentration_risk = "high"

    top3 = [c.category for c in categories[:3]]

    summary = (
        f"Spend analysis across {len(categories)} categories, "
        f"total spend = {total_spend:,.2f}. "
        f"HHI = {hhi:.0f} ({concentration_risk} concentration). "
        f"Top categories: {', '.join(top3)}."
    )

    return SpendCategoryResult(
        categories=categories,
        total_spend=round(total_spend, 4),
        hhi=hhi,
        concentration_risk=concentration_risk,
        top_categories=top3,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_procurement_report
# ---------------------------------------------------------------------------


def format_procurement_report(
    purchase_orders: PurchaseOrderResult | None = None,
    price_variance: PriceVarianceResult | None = None,
    vendor_perf: VendorPerfResult | None = None,
    spend_category: SpendCategoryResult | None = None,
) -> str:
    """Generate a combined text report from available procurement analyses.

    Args:
        purchase_orders: Result from analyze_purchase_orders.
        price_variance: Result from compute_purchase_price_variance.
        vendor_perf: Result from analyze_vendor_performance.
        spend_category: Result from analyze_spend_by_category.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Procurement Analytics Report")
    sections.append("=" * 40)

    if purchase_orders is not None:
        lines = ["", "Purchase Orders", "-" * 38]
        lines.append(
            f"  Total orders: {purchase_orders.total_orders} | "
            f"Total value: {purchase_orders.total_value:,.2f}"
        )
        lines.append("")
        lines.append("  Top Vendors:")
        for v in purchase_orders.vendors[:5]:
            lines.append(
                f"    {v.vendor}: {v.order_count} orders, "
                f"value = {v.total_value:,.2f} (avg {v.avg_value:,.2f})"
            )
        if purchase_orders.monthly_trends:
            lines.append("")
            lines.append("  Monthly Trends:")
            for mt in purchase_orders.monthly_trends:
                lines.append(
                    f"    {mt.month}: {mt.order_count} orders, "
                    f"value = {mt.total_value:,.2f}"
                )
        if purchase_orders.status_breakdown:
            lines.append("")
            lines.append("  Status Breakdown:")
            for status, cnt in sorted(purchase_orders.status_breakdown.items()):
                lines.append(f"    {status}: {cnt}")
        sections.append("\n".join(lines))

    if price_variance is not None:
        lines = ["", "Purchase Price Variance", "-" * 38]
        lines.append(
            f"  Total variance: {price_variance.total_variance:+,.2f} | "
            f"Favorable: {price_variance.favorable_count} | "
            f"Unfavorable: {price_variance.unfavorable_count}"
        )
        lines.append("")
        lines.append("  Items (by impact):")
        for it in price_variance.items:
            var_str = f"{it.total_variance:+,.2f}" if it.total_variance is not None else "N/A"
            vtype_str = f" [{it.variance_type}]" if it.variance_type else ""
            lines.append(
                f"    {it.item}: qty={it.quantity:,.0f}, "
                f"avg_price={it.avg_actual_price:,.2f}, "
                f"variance={var_str}{vtype_str}"
            )
        sections.append("\n".join(lines))

    if vendor_perf is not None:
        lines = ["", "Vendor Performance", "-" * 38]
        lines.append(
            f"  Overall on-time: {vendor_perf.overall_on_time_pct:.1f}%"
        )
        if vendor_perf.avg_quality is not None:
            lines.append(f"  Overall avg quality: {vendor_perf.avg_quality:.2f}")
        lines.append("")
        lines.append("  Vendors:")
        for vd in vendor_perf.vendors:
            q_str = f", quality={vd.avg_quality:.2f}" if vd.avg_quality is not None else ""
            qty_str = f", qty={vd.total_quantity:,.0f}" if vd.total_quantity is not None else ""
            lines.append(
                f"    {vd.vendor}: {vd.total_deliveries} deliveries, "
                f"on-time={vd.on_time_pct:.1f}%, "
                f"late={vd.late_count} (avg {vd.avg_days_late:.1f} days)"
                f"{q_str}{qty_str}"
            )
        sections.append("\n".join(lines))

    if spend_category is not None:
        lines = ["", "Spend by Category", "-" * 38]
        lines.append(
            f"  Total spend: {spend_category.total_spend:,.2f} | "
            f"HHI: {spend_category.hhi:.0f} ({spend_category.concentration_risk})"
        )
        lines.append("")
        lines.append("  Categories:")
        for sc in spend_category.categories:
            vc_str = f", {sc.vendor_count} vendors" if sc.vendor_count is not None else ""
            lines.append(
                f"    {sc.category}: {sc.total_spend:,.2f} "
                f"({sc.pct_of_total:.1f}%, {sc.transaction_count} txns{vc_str})"
            )
        sections.append("\n".join(lines))

    if (
        purchase_orders is None
        and price_variance is None
        and vendor_perf is None
        and spend_category is None
    ):
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
