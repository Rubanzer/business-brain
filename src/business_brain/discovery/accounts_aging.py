"""Accounts aging analysis for receivables and payables.

Pure functions for analyzing aging of receivables and payables,
computing Days Sales Outstanding (DSO), and evaluating collection
effectiveness from tabular business data.
No DB, async, or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, date, timedelta
from collections import defaultdict


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
    """Attempt to parse a date from string, datetime, or date."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    if isinstance(val, date):
        return datetime(val.year, val.month, val.day)
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
class AgingBucket:
    """A single aging bucket with count and amount."""
    label: str
    min_days: int
    max_days: int | None
    count: int
    amount: float
    pct_of_total: float


@dataclass
class CustomerAging:
    """Aging breakdown for a single customer."""
    customer: str
    total: float
    current: float
    days_31_60: float
    days_61_90: float
    days_91_120: float
    over_120: float


@dataclass
class ReceivablesAgingResult:
    """Complete receivables aging analysis result."""
    buckets: list[AgingBucket]
    by_customer: list[CustomerAging]
    total_outstanding: float
    total_overdue: float
    avg_days_outstanding: float
    worst_customers: list[str]
    summary: str


@dataclass
class VendorAging:
    """Aging breakdown for a single vendor."""
    vendor: str
    total: float
    current: float
    days_31_60: float
    days_61_90: float
    days_91_120: float
    over_120: float


@dataclass
class PayablesAgingResult:
    """Complete payables aging analysis result."""
    buckets: list[AgingBucket]
    by_vendor: list[VendorAging]
    total_outstanding: float
    total_overdue: float
    avg_days_outstanding: float
    summary: str


@dataclass
class PeriodDSO:
    """DSO for a single period."""
    period: str
    dso: float
    revenue: float
    receivables: float


@dataclass
class DSOResult:
    """Complete DSO analysis result."""
    overall_dso: float
    periods: list[PeriodDSO]
    trend: str
    benchmark_status: str
    summary: str


@dataclass
class CustomerCollection:
    """Collection metrics for a single customer."""
    customer: str
    invoiced: float
    collected: float
    collection_rate: float
    outstanding: float


@dataclass
class CollectionResult:
    """Complete collection effectiveness result."""
    overall_rate: float
    total_invoiced: float
    total_collected: float
    total_outstanding: float
    by_customer: list[CustomerCollection]
    best_collectors: list[str]
    worst_collectors: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Bucket definitions
# ---------------------------------------------------------------------------

_BUCKET_DEFS = [
    ("Current (0-30)", 0, 30),
    ("31-60 days", 31, 60),
    ("61-90 days", 61, 90),
    ("91-120 days", 91, 120),
    ("120+ days", 121, None),
]


def _classify_days(days_outstanding: int) -> int:
    """Return bucket index (0-4) for a given number of days outstanding."""
    if days_outstanding <= 30:
        return 0
    elif days_outstanding <= 60:
        return 1
    elif days_outstanding <= 90:
        return 2
    elif days_outstanding <= 120:
        return 3
    else:
        return 4


# ---------------------------------------------------------------------------
# 1. analyze_receivables_aging
# ---------------------------------------------------------------------------


def analyze_receivables_aging(
    rows: list[dict],
    customer_column: str,
    amount_column: str,
    invoice_date_column: str,
    due_date_column: str | None = None,
    reference_date: datetime | None = None,
) -> ReceivablesAgingResult | None:
    """Analyze receivables aging by customer.

    Classifies invoices into aging buckets: Current (0-30 days), 31-60,
    61-90, 91-120, and 120+ days past due.

    Args:
        rows: Data rows as dicts.
        customer_column: Column with customer names.
        amount_column: Column with invoice amounts.
        invoice_date_column: Column with invoice dates.
        due_date_column: Optional column with due dates. If not provided,
            due date defaults to invoice_date + 30 days.
        reference_date: Date to compute aging from. Defaults to the
            maximum date found in the data.

    Returns:
        ReceivablesAgingResult or None if insufficient data.
    """
    if not rows:
        return None

    # First pass: parse valid rows and determine reference_date
    parsed: list[tuple[str, float, datetime]] = []
    max_dt: datetime | None = None

    for row in rows:
        customer = row.get(customer_column)
        if customer is None:
            continue
        customer = str(customer)

        amt = _safe_float(row.get(amount_column))
        if amt is None:
            continue

        inv_dt = _parse_date(row.get(invoice_date_column))
        if inv_dt is None:
            continue

        if due_date_column is not None:
            due_dt = _parse_date(row.get(due_date_column))
            if due_dt is None:
                due_dt = inv_dt + timedelta(days=30)
        else:
            due_dt = inv_dt + timedelta(days=30)

        # Track max date for default reference_date
        if max_dt is None or due_dt > max_dt:
            max_dt = due_dt
        if inv_dt > (max_dt or inv_dt):
            max_dt = inv_dt

        parsed.append((customer, amt, due_dt))

    if not parsed:
        return None

    if reference_date is None:
        reference_date = max_dt

    # Build buckets and per-customer data
    bucket_counts = [0] * 5
    bucket_amounts = [0.0] * 5
    total_outstanding = 0.0
    total_overdue = 0.0
    total_days = 0.0
    count_for_avg = 0

    # Per-customer accumulators: {customer: [current, 31-60, 61-90, 91-120, 120+]}
    cust_buckets: dict[str, list[float]] = defaultdict(lambda: [0.0] * 5)

    for customer, amt, due_dt in parsed:
        days_past = (reference_date - due_dt).days
        if days_past < 0:
            days_past = 0

        bucket_idx = _classify_days(days_past)
        bucket_counts[bucket_idx] += 1
        bucket_amounts[bucket_idx] += amt
        total_outstanding += amt

        if days_past > 30:
            total_overdue += amt

        total_days += days_past
        count_for_avg += 1

        cust_buckets[customer][bucket_idx] += amt

    avg_days = total_days / count_for_avg if count_for_avg > 0 else 0.0

    # Build AgingBucket list
    buckets: list[AgingBucket] = []
    for i, (label, min_d, max_d) in enumerate(_BUCKET_DEFS):
        pct = (bucket_amounts[i] / total_outstanding * 100.0) if total_outstanding > 0 else 0.0
        buckets.append(AgingBucket(
            label=label,
            min_days=min_d,
            max_days=max_d,
            count=bucket_counts[i],
            amount=round(bucket_amounts[i], 2),
            pct_of_total=round(pct, 2),
        ))

    # Build CustomerAging list
    by_customer: list[CustomerAging] = []
    for cust in sorted(cust_buckets.keys()):
        b = cust_buckets[cust]
        total = sum(b)
        by_customer.append(CustomerAging(
            customer=cust,
            total=round(total, 2),
            current=round(b[0], 2),
            days_31_60=round(b[1], 2),
            days_61_90=round(b[2], 2),
            days_91_120=round(b[3], 2),
            over_120=round(b[4], 2),
        ))

    # Worst customers: those with highest overdue (>30 days) amounts
    cust_overdue: list[tuple[str, float]] = []
    for cust in cust_buckets:
        b = cust_buckets[cust]
        overdue = b[1] + b[2] + b[3] + b[4]
        if overdue > 0:
            cust_overdue.append((cust, overdue))
    cust_overdue.sort(key=lambda x: x[1], reverse=True)
    worst_customers = [c[0] for c in cust_overdue[:5]]

    summary = (
        f"Receivables aging: total outstanding={total_outstanding:,.2f}, "
        f"total overdue={total_overdue:,.2f}, "
        f"avg days outstanding={avg_days:.1f}. "
        f"Worst customers: {', '.join(worst_customers) if worst_customers else 'none'}."
    )

    return ReceivablesAgingResult(
        buckets=buckets,
        by_customer=by_customer,
        total_outstanding=round(total_outstanding, 2),
        total_overdue=round(total_overdue, 2),
        avg_days_outstanding=round(avg_days, 1),
        worst_customers=worst_customers,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. analyze_payables_aging
# ---------------------------------------------------------------------------


def analyze_payables_aging(
    rows: list[dict],
    vendor_column: str,
    amount_column: str,
    invoice_date_column: str,
    due_date_column: str | None = None,
    reference_date: datetime | None = None,
) -> PayablesAgingResult | None:
    """Analyze payables aging by vendor.

    Same bucket structure as receivables but for payables/vendors.

    Args:
        rows: Data rows as dicts.
        vendor_column: Column with vendor names.
        amount_column: Column with invoice amounts.
        invoice_date_column: Column with invoice dates.
        due_date_column: Optional column with due dates. If not provided,
            due date defaults to invoice_date + 30 days.
        reference_date: Date to compute aging from. Defaults to the
            maximum date found in the data.

    Returns:
        PayablesAgingResult or None if insufficient data.
    """
    if not rows:
        return None

    # First pass: parse valid rows and determine reference_date
    parsed: list[tuple[str, float, datetime]] = []
    max_dt: datetime | None = None

    for row in rows:
        vendor = row.get(vendor_column)
        if vendor is None:
            continue
        vendor = str(vendor)

        amt = _safe_float(row.get(amount_column))
        if amt is None:
            continue

        inv_dt = _parse_date(row.get(invoice_date_column))
        if inv_dt is None:
            continue

        if due_date_column is not None:
            due_dt = _parse_date(row.get(due_date_column))
            if due_dt is None:
                due_dt = inv_dt + timedelta(days=30)
        else:
            due_dt = inv_dt + timedelta(days=30)

        if max_dt is None or due_dt > max_dt:
            max_dt = due_dt
        if inv_dt > (max_dt or inv_dt):
            max_dt = inv_dt

        parsed.append((vendor, amt, due_dt))

    if not parsed:
        return None

    if reference_date is None:
        reference_date = max_dt

    # Build buckets and per-vendor data
    bucket_counts = [0] * 5
    bucket_amounts = [0.0] * 5
    total_outstanding = 0.0
    total_overdue = 0.0
    total_days = 0.0
    count_for_avg = 0

    vendor_buckets: dict[str, list[float]] = defaultdict(lambda: [0.0] * 5)

    for vendor, amt, due_dt in parsed:
        days_past = (reference_date - due_dt).days
        if days_past < 0:
            days_past = 0

        bucket_idx = _classify_days(days_past)
        bucket_counts[bucket_idx] += 1
        bucket_amounts[bucket_idx] += amt
        total_outstanding += amt

        if days_past > 30:
            total_overdue += amt

        total_days += days_past
        count_for_avg += 1

        vendor_buckets[vendor][bucket_idx] += amt

    avg_days = total_days / count_for_avg if count_for_avg > 0 else 0.0

    # Build AgingBucket list
    buckets: list[AgingBucket] = []
    for i, (label, min_d, max_d) in enumerate(_BUCKET_DEFS):
        pct = (bucket_amounts[i] / total_outstanding * 100.0) if total_outstanding > 0 else 0.0
        buckets.append(AgingBucket(
            label=label,
            min_days=min_d,
            max_days=max_d,
            count=bucket_counts[i],
            amount=round(bucket_amounts[i], 2),
            pct_of_total=round(pct, 2),
        ))

    # Build VendorAging list
    by_vendor: list[VendorAging] = []
    for vnd in sorted(vendor_buckets.keys()):
        b = vendor_buckets[vnd]
        total = sum(b)
        by_vendor.append(VendorAging(
            vendor=vnd,
            total=round(total, 2),
            current=round(b[0], 2),
            days_31_60=round(b[1], 2),
            days_61_90=round(b[2], 2),
            days_91_120=round(b[3], 2),
            over_120=round(b[4], 2),
        ))

    summary = (
        f"Payables aging: total outstanding={total_outstanding:,.2f}, "
        f"total overdue={total_overdue:,.2f}, "
        f"avg days outstanding={avg_days:.1f}."
    )

    return PayablesAgingResult(
        buckets=buckets,
        by_vendor=by_vendor,
        total_outstanding=round(total_outstanding, 2),
        total_overdue=round(total_overdue, 2),
        avg_days_outstanding=round(avg_days, 1),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. compute_dso
# ---------------------------------------------------------------------------


def compute_dso(
    rows: list[dict],
    revenue_column: str,
    receivables_column: str,
    date_column: str | None = None,
) -> DSOResult | None:
    """Compute Days Sales Outstanding (DSO).

    DSO = (Avg Receivables / Total Revenue) * Days in period.
    If date_column provided, compute DSO per period (month).

    Args:
        rows: Data rows as dicts.
        revenue_column: Column with revenue amounts.
        receivables_column: Column with receivables amounts.
        date_column: Optional column for period-level DSO.

    Returns:
        DSOResult or None if insufficient data.
    """
    if not rows:
        return None

    total_revenue = 0.0
    total_receivables = 0.0
    valid_count = 0

    # For period analysis
    period_data: dict[str, dict[str, list[float]]] = {}

    for row in rows:
        rev = _safe_float(row.get(revenue_column))
        rec = _safe_float(row.get(receivables_column))
        if rev is None or rec is None:
            continue

        valid_count += 1
        total_revenue += rev
        total_receivables += rec

        if date_column is not None:
            dt = _parse_date(row.get(date_column))
            if dt is not None:
                mk = _month_key(dt)
                if mk not in period_data:
                    period_data[mk] = {"revenue": [], "receivables": []}
                period_data[mk]["revenue"].append(rev)
                period_data[mk]["receivables"].append(rec)

    if valid_count == 0 or total_revenue == 0:
        return None

    avg_receivables = total_receivables / valid_count

    # Determine days in period
    if date_column is not None and period_data:
        sorted_periods = sorted(period_data.keys())
        num_periods = len(sorted_periods)
        # Approximate days = num_periods * 30
        days_in_period = num_periods * 30
    else:
        # Without date info, assume data covers 30 days per row on average
        # but use 365 as a sensible default for annual data
        days_in_period = valid_count * 30

    overall_dso = (avg_receivables / total_revenue) * days_in_period

    # Per-period DSO
    periods: list[PeriodDSO] = []
    if date_column is not None and period_data:
        for mk in sorted(period_data.keys()):
            p_revs = period_data[mk]["revenue"]
            p_recs = period_data[mk]["receivables"]
            p_total_rev = sum(p_revs)
            p_avg_rec = sum(p_recs) / len(p_recs)
            if p_total_rev > 0:
                p_dso = (p_avg_rec / p_total_rev) * 30
            else:
                p_dso = 0.0
            periods.append(PeriodDSO(
                period=mk,
                dso=round(p_dso, 2),
                revenue=round(p_total_rev, 2),
                receivables=round(p_avg_rec, 2),
            ))

    # Trend detection
    if len(periods) >= 2:
        first_half = periods[: len(periods) // 2]
        second_half = periods[len(periods) // 2 :]
        avg_first = sum(p.dso for p in first_half) / len(first_half)
        avg_second = sum(p.dso for p in second_half) / len(second_half)
        if avg_first == 0:
            trend = "stable" if avg_second == 0 else "increasing"
        else:
            change_pct = (avg_second - avg_first) / abs(avg_first) * 100
            if change_pct > 10:
                trend = "increasing"
            elif change_pct < -10:
                trend = "decreasing"
            else:
                trend = "stable"
    else:
        trend = "stable"

    # Benchmark
    if overall_dso < 30:
        benchmark_status = "excellent"
    elif overall_dso < 45:
        benchmark_status = "good"
    elif overall_dso < 60:
        benchmark_status = "fair"
    else:
        benchmark_status = "needs attention"

    summary = (
        f"DSO analysis: overall DSO={overall_dso:.1f} days. "
        f"Benchmark: {benchmark_status}. Trend: {trend}."
    )

    return DSOResult(
        overall_dso=round(overall_dso, 2),
        periods=periods,
        trend=trend,
        benchmark_status=benchmark_status,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. analyze_collection_effectiveness
# ---------------------------------------------------------------------------


def analyze_collection_effectiveness(
    rows: list[dict],
    customer_column: str,
    amount_column: str,
    paid_amount_column: str,
    date_column: str | None = None,
) -> CollectionResult | None:
    """Analyze collection effectiveness by customer.

    Collection rate = total paid / total invoiced * 100.

    Args:
        rows: Data rows as dicts.
        customer_column: Column with customer names.
        amount_column: Column with invoiced amounts.
        paid_amount_column: Column with paid/collected amounts.
        date_column: Optional column with dates (reserved for future use).

    Returns:
        CollectionResult or None if insufficient data.
    """
    if not rows:
        return None

    total_invoiced = 0.0
    total_collected = 0.0
    valid_count = 0

    cust_data: dict[str, dict[str, float]] = defaultdict(
        lambda: {"invoiced": 0.0, "collected": 0.0}
    )

    for row in rows:
        customer = row.get(customer_column)
        if customer is None:
            continue
        customer = str(customer)

        amt = _safe_float(row.get(amount_column))
        paid = _safe_float(row.get(paid_amount_column))
        if amt is None or paid is None:
            continue

        valid_count += 1
        total_invoiced += amt
        total_collected += paid

        cust_data[customer]["invoiced"] += amt
        cust_data[customer]["collected"] += paid

    if valid_count == 0 or total_invoiced == 0:
        return None

    overall_rate = (total_collected / total_invoiced) * 100.0
    total_outstanding = total_invoiced - total_collected

    # Per-customer collection
    by_customer: list[CustomerCollection] = []
    for cust in sorted(cust_data.keys()):
        d = cust_data[cust]
        inv = d["invoiced"]
        col = d["collected"]
        rate = (col / inv * 100.0) if inv > 0 else 0.0
        out = inv - col
        by_customer.append(CustomerCollection(
            customer=cust,
            invoiced=round(inv, 2),
            collected=round(col, 2),
            collection_rate=round(rate, 2),
            outstanding=round(out, 2),
        ))

    # Best/worst collectors (by rate)
    sorted_by_rate = sorted(by_customer, key=lambda c: c.collection_rate, reverse=True)
    best_collectors = [c.customer for c in sorted_by_rate[:3] if c.collection_rate >= overall_rate]
    worst_collectors = [c.customer for c in sorted_by_rate[-3:] if c.collection_rate < overall_rate]

    summary = (
        f"Collection analysis: overall rate={overall_rate:.1f}%, "
        f"total invoiced={total_invoiced:,.2f}, "
        f"total collected={total_collected:,.2f}, "
        f"outstanding={total_outstanding:,.2f}."
    )

    return CollectionResult(
        overall_rate=round(overall_rate, 2),
        total_invoiced=round(total_invoiced, 2),
        total_collected=round(total_collected, 2),
        total_outstanding=round(total_outstanding, 2),
        by_customer=by_customer,
        best_collectors=best_collectors,
        worst_collectors=worst_collectors,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_aging_report
# ---------------------------------------------------------------------------


def format_aging_report(
    receivables: ReceivablesAgingResult | None = None,
    payables: PayablesAgingResult | None = None,
    dso: DSOResult | None = None,
    collection: CollectionResult | None = None,
) -> str:
    """Combine all aging analyses into a text report.

    Args:
        receivables: Optional receivables aging result.
        payables: Optional payables aging result.
        dso: Optional DSO result.
        collection: Optional collection effectiveness result.

    Returns:
        Combined text report string.
    """
    sections: list[str] = []

    if receivables is not None:
        lines = ["=== Receivables Aging ==="]
        lines.append(f"Total Outstanding: {receivables.total_outstanding:,.2f}")
        lines.append(f"Total Overdue: {receivables.total_overdue:,.2f}")
        lines.append(f"Avg Days Outstanding: {receivables.avg_days_outstanding:.1f}")
        lines.append("Aging Buckets:")
        for b in receivables.buckets:
            lines.append(
                f"  {b.label}: count={b.count}, "
                f"amount={b.amount:,.2f}, pct={b.pct_of_total:.1f}%"
            )
        if receivables.by_customer:
            lines.append("By Customer:")
            for ca in receivables.by_customer:
                lines.append(
                    f"  {ca.customer}: total={ca.total:,.2f}, "
                    f"current={ca.current:,.2f}, 31-60={ca.days_31_60:,.2f}, "
                    f"61-90={ca.days_61_90:,.2f}, 91-120={ca.days_91_120:,.2f}, "
                    f"120+={ca.over_120:,.2f}"
                )
        if receivables.worst_customers:
            lines.append(f"Worst Customers: {', '.join(receivables.worst_customers)}")
        sections.append("\n".join(lines))

    if payables is not None:
        lines = ["=== Payables Aging ==="]
        lines.append(f"Total Outstanding: {payables.total_outstanding:,.2f}")
        lines.append(f"Total Overdue: {payables.total_overdue:,.2f}")
        lines.append(f"Avg Days Outstanding: {payables.avg_days_outstanding:.1f}")
        lines.append("Aging Buckets:")
        for b in payables.buckets:
            lines.append(
                f"  {b.label}: count={b.count}, "
                f"amount={b.amount:,.2f}, pct={b.pct_of_total:.1f}%"
            )
        if payables.by_vendor:
            lines.append("By Vendor:")
            for va in payables.by_vendor:
                lines.append(
                    f"  {va.vendor}: total={va.total:,.2f}, "
                    f"current={va.current:,.2f}, 31-60={va.days_31_60:,.2f}, "
                    f"61-90={va.days_61_90:,.2f}, 91-120={va.days_91_120:,.2f}, "
                    f"120+={va.over_120:,.2f}"
                )
        sections.append("\n".join(lines))

    if dso is not None:
        lines = ["=== Days Sales Outstanding ==="]
        lines.append(f"Overall DSO: {dso.overall_dso:.2f} days")
        lines.append(f"Benchmark: {dso.benchmark_status}")
        lines.append(f"Trend: {dso.trend}")
        if dso.periods:
            lines.append("Period Details:")
            for p in dso.periods:
                lines.append(
                    f"  {p.period}: DSO={p.dso:.2f}, "
                    f"revenue={p.revenue:,.2f}, receivables={p.receivables:,.2f}"
                )
        sections.append("\n".join(lines))

    if collection is not None:
        lines = ["=== Collection Effectiveness ==="]
        lines.append(f"Overall Rate: {collection.overall_rate:.2f}%")
        lines.append(f"Total Invoiced: {collection.total_invoiced:,.2f}")
        lines.append(f"Total Collected: {collection.total_collected:,.2f}")
        lines.append(f"Total Outstanding: {collection.total_outstanding:,.2f}")
        if collection.by_customer:
            lines.append("By Customer:")
            for cc in collection.by_customer:
                lines.append(
                    f"  {cc.customer}: invoiced={cc.invoiced:,.2f}, "
                    f"collected={cc.collected:,.2f}, "
                    f"rate={cc.collection_rate:.2f}%, "
                    f"outstanding={cc.outstanding:,.2f}"
                )
        if collection.best_collectors:
            lines.append(f"Best Collectors: {', '.join(collection.best_collectors)}")
        if collection.worst_collectors:
            lines.append(f"Worst Collectors: {', '.join(collection.worst_collectors)}")
        sections.append("\n".join(lines))

    if not sections:
        return "No aging data available for report."

    return "\n\n".join(sections)
