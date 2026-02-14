"""RFM analysis -- Recency, Frequency, Monetary customer segmentation.

Pure functions for computing RFM scores, segmenting customers, and
estimating customer lifetime value.  Common in sales and marketing
analytics for identifying high-value customers, at-risk segments,
and growth opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CustomerRFM:
    """RFM profile for a single customer."""

    customer: str
    recency_days: int
    frequency: int
    monetary: float
    r_score: int  # 1-5
    f_score: int  # 1-5
    m_score: int  # 1-5
    rfm_score: float  # weighted average 1-5
    segment: str  # e.g. "Champions", "At Risk"


@dataclass
class RFMResult:
    """Complete RFM analysis result."""

    customers: list[CustomerRFM]
    segment_distribution: dict[str, int]
    total_customers: int
    avg_recency: float
    avg_frequency: float
    avg_monetary: float
    summary: str


@dataclass
class CustomerCLV:
    """Lifetime value profile for a single customer."""

    customer: str
    total_spent: float
    purchase_count: int
    avg_purchase: float
    first_purchase: str
    last_purchase: str
    lifespan_days: int
    estimated_clv: float


@dataclass
class CLVResult:
    """Complete CLV analysis result."""

    customers: list[CustomerCLV]
    avg_clv: float
    top_customers: list[str]
    summary: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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


def _parse_float(value) -> float | None:
    """Attempt to parse a value into a float."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _quintile_scores(values: list[float]) -> list[int]:
    """Assign quintile scores 1-5 to a list of values.

    Higher values get higher scores.  Ties at bucket boundaries are
    resolved by assigning the lower bucket.
    """
    n = len(values)
    if n == 0:
        return []

    # Rank the values (indices sorted by value ascending)
    indexed = sorted(enumerate(values), key=lambda x: x[1])

    scores = [0] * n
    for rank, (orig_idx, _) in enumerate(indexed):
        # rank goes from 0..n-1; map to 1..5
        bucket = int(rank * 5 / n) + 1
        bucket = min(bucket, 5)
        scores[orig_idx] = bucket

    return scores


def _quintile_scores_reverse(values: list[float]) -> list[int]:
    """Assign quintile scores 1-5 where *lower* values get *higher* scores.

    Used for recency: fewer days since last purchase = better = higher score.
    """
    n = len(values)
    if n == 0:
        return []

    indexed = sorted(enumerate(values), key=lambda x: x[1])

    scores = [0] * n
    for rank, (orig_idx, _) in enumerate(indexed):
        # rank 0 = smallest value = best recency = score 5
        bucket = 5 - int(rank * 5 / n)
        bucket = max(bucket, 1)
        scores[orig_idx] = bucket

    return scores


def _assign_segment(r: int, f: int, m: int) -> str:
    """Assign a marketing segment name based on R, F, M scores."""
    avg = (r + f + m) / 3.0

    if avg >= 4.5:
        return "Champions"
    if f >= 4:
        return "Loyal"
    if r >= 4 and f >= 2:
        return "Potential Loyalists"
    if r >= 4 and f == 1:
        return "New Customers"
    if r <= 2 and f >= 3:
        return "At Risk"
    if r == 1 and f == 1:
        return "Lost"
    if r >= 3 and f >= 3:
        return "Promising"
    if r <= 2 and f >= 1 and m >= 3:
        return "Need Attention"
    if r >= 3 and f <= 2:
        return "About to Sleep"
    return "Hibernating"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_rfm(
    rows: list[dict],
    customer_column: str,
    date_column: str,
    amount_column: str,
    reference_date: datetime | None = None,
) -> RFMResult | None:
    """Compute RFM scores for each customer.

    Args:
        rows: Transaction data as list of dicts.
        customer_column: Column identifying the customer.
        date_column: Column with the transaction date.
        amount_column: Column with the transaction amount.
        reference_date: Date to compute recency from.  Defaults to
            one day after the latest transaction date found.

    Returns:
        RFMResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per customer: last date, count, total amount
    customer_data: dict[str, dict] = {}
    for row in rows:
        cust = row.get(customer_column)
        date_val = _parse_date(row.get(date_column))
        amount_val = _parse_float(row.get(amount_column))
        if cust is None or date_val is None or amount_val is None:
            continue

        cust_str = str(cust)
        if cust_str not in customer_data:
            customer_data[cust_str] = {
                "last_date": date_val,
                "count": 0,
                "total": 0.0,
            }
        entry = customer_data[cust_str]
        if date_val > entry["last_date"]:
            entry["last_date"] = date_val
        entry["count"] += 1
        entry["total"] += amount_val

    if len(customer_data) < 2:
        return None

    # Determine reference date
    if reference_date is None:
        max_date = max(d["last_date"] for d in customer_data.values())
        reference_date = max_date + timedelta(days=1)

    # Build raw R, F, M lists
    customers_list = sorted(customer_data.keys())
    recency_values: list[float] = []
    frequency_values: list[float] = []
    monetary_values: list[float] = []

    for cust in customers_list:
        entry = customer_data[cust]
        recency = (reference_date - entry["last_date"]).days
        recency_values.append(float(recency))
        frequency_values.append(float(entry["count"]))
        monetary_values.append(float(entry["total"]))

    # Score using quintiles
    r_scores = _quintile_scores_reverse(recency_values)
    f_scores = _quintile_scores(frequency_values)
    m_scores = _quintile_scores(monetary_values)

    # Build CustomerRFM objects
    customer_rfms: list[CustomerRFM] = []
    segment_counts: dict[str, int] = {}

    for i, cust in enumerate(customers_list):
        r, f, m = r_scores[i], f_scores[i], m_scores[i]
        rfm_score = round((r + f + m) / 3.0, 2)
        segment = _assign_segment(r, f, m)

        customer_rfms.append(CustomerRFM(
            customer=cust,
            recency_days=int(recency_values[i]),
            frequency=int(frequency_values[i]),
            monetary=round(monetary_values[i], 2),
            r_score=r,
            f_score=f,
            m_score=m,
            rfm_score=rfm_score,
            segment=segment,
        ))

        segment_counts[segment] = segment_counts.get(segment, 0) + 1

    n = len(customer_rfms)
    avg_r = round(sum(recency_values) / n, 2)
    avg_f = round(sum(frequency_values) / n, 2)
    avg_m = round(sum(monetary_values) / n, 2)

    summary = (
        f"RFM Analysis: {n} customers analysed. "
        f"Avg recency: {avg_r} days, avg frequency: {avg_f}, "
        f"avg monetary: {avg_m:.2f}. "
        f"Segments: {', '.join(f'{seg} ({cnt})' for seg, cnt in sorted(segment_counts.items()))}."
    )

    return RFMResult(
        customers=customer_rfms,
        segment_distribution=segment_counts,
        total_customers=n,
        avg_recency=avg_r,
        avg_frequency=avg_f,
        avg_monetary=avg_m,
        summary=summary,
    )


def segment_customers(rfm_result: RFMResult) -> dict[str, dict]:
    """Group customers into named segments based on RFM scores.

    Args:
        rfm_result: Output from :func:`compute_rfm`.

    Returns:
        Dict mapping segment name to ``{"customers": [...], "count": N}``.
    """
    if not rfm_result or not rfm_result.customers:
        return {}

    groups: dict[str, list[str]] = {}
    for c in rfm_result.customers:
        groups.setdefault(c.segment, []).append(c.customer)

    return {
        seg: {"customers": sorted(names), "count": len(names)}
        for seg, names in sorted(groups.items())
    }


def customer_lifetime_value(
    rows: list[dict],
    customer_column: str,
    amount_column: str,
    date_column: str,
) -> CLVResult | None:
    """Estimate simple customer lifetime value.

    CLV = avg_purchase * purchase_frequency_per_year * avg_lifespan_years.

    Args:
        rows: Transaction data as list of dicts.
        customer_column: Column identifying the customer.
        amount_column: Column with the transaction amount.
        date_column: Column with the transaction date.

    Returns:
        CLVResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate per customer
    customer_data: dict[str, dict] = {}
    for row in rows:
        cust = row.get(customer_column)
        amount_val = _parse_float(row.get(amount_column))
        date_val = _parse_date(row.get(date_column))
        if cust is None or amount_val is None or date_val is None:
            continue

        cust_str = str(cust)
        if cust_str not in customer_data:
            customer_data[cust_str] = {
                "total": 0.0,
                "count": 0,
                "first": date_val,
                "last": date_val,
            }
        entry = customer_data[cust_str]
        entry["total"] += amount_val
        entry["count"] += 1
        if date_val < entry["first"]:
            entry["first"] = date_val
        if date_val > entry["last"]:
            entry["last"] = date_val

    if not customer_data:
        return None

    # Global average lifespan across all customers (in years)
    lifespans: list[float] = []
    for entry in customer_data.values():
        days = (entry["last"] - entry["first"]).days
        lifespans.append(days)

    avg_lifespan_days = sum(lifespans) / len(lifespans) if lifespans else 0.0
    avg_lifespan_years = avg_lifespan_days / 365.0 if avg_lifespan_days > 0 else 1.0

    # Build CustomerCLV records
    clv_list: list[CustomerCLV] = []
    for cust in sorted(customer_data.keys()):
        entry = customer_data[cust]
        total = entry["total"]
        count = entry["count"]
        avg_purchase = total / count if count > 0 else 0.0
        lifespan_days = (entry["last"] - entry["first"]).days

        # Customer-level lifespan for CLV estimation
        cust_lifespan_years = lifespan_days / 365.0 if lifespan_days > 0 else avg_lifespan_years
        # Frequency per year
        freq_per_year = count / cust_lifespan_years if cust_lifespan_years > 0 else float(count)

        estimated_clv = round(avg_purchase * freq_per_year * cust_lifespan_years, 2)

        clv_list.append(CustomerCLV(
            customer=cust,
            total_spent=round(total, 2),
            purchase_count=count,
            avg_purchase=round(avg_purchase, 2),
            first_purchase=entry["first"].strftime("%Y-%m-%d"),
            last_purchase=entry["last"].strftime("%Y-%m-%d"),
            lifespan_days=lifespan_days,
            estimated_clv=estimated_clv,
        ))

    # Stats
    clv_values = [c.estimated_clv for c in clv_list]
    avg_clv = round(sum(clv_values) / len(clv_values), 2) if clv_values else 0.0

    # Top customers by CLV
    sorted_by_clv = sorted(clv_list, key=lambda c: c.estimated_clv, reverse=True)
    top_n = min(5, len(sorted_by_clv))
    top_customers = [c.customer for c in sorted_by_clv[:top_n]]

    summary = (
        f"CLV Analysis: {len(clv_list)} customers. "
        f"Avg CLV: {avg_clv:.2f}. "
        f"Top customers: {', '.join(top_customers)}."
    )

    return CLVResult(
        customers=clv_list,
        avg_clv=avg_clv,
        top_customers=top_customers,
        summary=summary,
    )


def format_rfm_report(
    rfm: RFMResult | None = None,
    segments: dict | None = None,
    clv: CLVResult | None = None,
) -> str:
    """Format a combined RFM / segmentation / CLV report as plain text.

    Any parameter may be ``None``; only provided sections are rendered.

    Args:
        rfm: Output from :func:`compute_rfm`.
        segments: Output from :func:`segment_customers`.
        clv: Output from :func:`customer_lifetime_value`.

    Returns:
        Formatted multi-line report string.
    """
    lines: list[str] = []

    if rfm is not None:
        lines.append("RFM Analysis Report")
        lines.append("=" * 60)
        lines.append(f"Total Customers: {rfm.total_customers}")
        lines.append(f"Avg Recency:     {rfm.avg_recency} days")
        lines.append(f"Avg Frequency:   {rfm.avg_frequency}")
        lines.append(f"Avg Monetary:    {rfm.avg_monetary:.2f}")
        lines.append("")
        lines.append(
            f"{'Customer':<20}{'R':>4}{'F':>4}{'M':>4}{'RFM':>6}{'Segment':<20}"
        )
        lines.append("-" * 58)
        for c in rfm.customers:
            lines.append(
                f"{c.customer:<20}{c.r_score:>4}{c.f_score:>4}{c.m_score:>4}"
                f"{c.rfm_score:>6.2f}  {c.segment}"
            )
        lines.append("")

    if segments is not None:
        lines.append("Customer Segments")
        lines.append("=" * 60)
        for seg_name, info in sorted(segments.items()):
            count = info["count"]
            names = ", ".join(info["customers"][:5])
            suffix = "..." if len(info["customers"]) > 5 else ""
            lines.append(f"  {seg_name} ({count}): {names}{suffix}")
        lines.append("")

    if clv is not None:
        lines.append("Customer Lifetime Value")
        lines.append("=" * 60)
        lines.append(f"Avg CLV: {clv.avg_clv:.2f}")
        lines.append("")
        lines.append(
            f"{'Customer':<20}{'Spent':>12}{'Purchases':>10}{'CLV':>12}"
        )
        lines.append("-" * 54)
        for c in clv.customers:
            lines.append(
                f"{c.customer:<20}{c.total_spent:>12.2f}{c.purchase_count:>10}"
                f"{c.estimated_clv:>12.2f}"
            )
        lines.append("")

    if not lines:
        return "No data provided for RFM report."

    return "\n".join(lines)
