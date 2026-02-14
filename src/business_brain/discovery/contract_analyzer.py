"""Contract analysis — expiry tracking, value analysis, vendor concentration, renewal patterns.

Pure functions for procurement contract analysis including overall contract
portfolio analysis, expiring-contract detection with urgency levels, vendor
concentration measurement (HHI), and renewal-pattern detection with price
escalation tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Attempt to convert *val* to float, returning None on failure."""
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
class VendorSummary:
    """Summary of one vendor's contracts."""
    vendor: str
    contract_count: int
    total_value: float
    avg_value: float


@dataclass
class ContractResult:
    """Overall contract portfolio analysis."""
    total_contracts: int
    total_value: float
    avg_value: float
    vendors: list[VendorSummary]
    status_counts: dict[str, int] | None
    avg_duration_days: float | None
    summary: str


@dataclass
class ExpiringContract:
    """A contract that is expiring within the detection horizon."""
    contract: str
    vendor: str | None
    value: float | None
    end_date: datetime
    days_until_expiry: int
    urgency: str  # "critical", "warning", "upcoming"


@dataclass
class VendorShare:
    """A vendor's share of total contract value."""
    vendor: str
    total_value: float
    share_pct: float


@dataclass
class ConcentrationResult:
    """Vendor concentration analysis using HHI."""
    hhi: float
    risk_rating: str  # "High", "Moderate", "Low"
    top_vendors: list[VendorShare]
    single_vendor_dependency: bool
    dependency_vendors: list[str]
    summary: str


@dataclass
class VendorRenewal:
    """Renewal information for a single vendor."""
    vendor: str
    contract_count: int
    has_renewals: bool
    renewal_count: int
    avg_value_change_pct: float | None


@dataclass
class RenewalResult:
    """Renewal pattern analysis across vendors."""
    total_vendors: int
    vendors_with_renewals: int
    renewal_rate: float
    vendor_details: list[VendorRenewal]
    avg_price_escalation_pct: float | None
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_contracts
# ---------------------------------------------------------------------------


def analyze_contracts(
    rows: list[dict],
    contract_column: str,
    vendor_column: str,
    value_column: str,
    start_column: str | None = None,
    end_column: str | None = None,
    status_column: str | None = None,
) -> ContractResult | None:
    """Analyse a portfolio of contracts.

    Args:
        rows: Data rows as dicts.
        contract_column: Column identifying the contract.
        vendor_column: Column identifying the vendor.
        value_column: Column with contract monetary value.
        start_column: Optional column with contract start date.
        end_column: Optional column with contract end date.
        status_column: Optional column with contract status string.

    Returns:
        ContractResult or None if no valid data.
    """
    if not rows:
        return None

    # --- Gather valid rows ---------------------------------------------------
    contracts: list[dict] = []
    for row in rows:
        contract = row.get(contract_column)
        vendor = row.get(vendor_column)
        value = _safe_float(row.get(value_column))
        if contract is None or vendor is None or value is None:
            continue
        entry: dict = {
            "contract": str(contract),
            "vendor": str(vendor),
            "value": value,
        }
        if status_column is not None:
            status = row.get(status_column)
            if status is not None:
                entry["status"] = str(status)
        if start_column is not None and end_column is not None:
            start_dt = _parse_date(row.get(start_column))
            end_dt = _parse_date(row.get(end_column))
            if start_dt is not None and end_dt is not None:
                entry["start"] = start_dt
                entry["end"] = end_dt
        contracts.append(entry)

    if not contracts:
        return None

    total_contracts = len(contracts)
    total_value = sum(c["value"] for c in contracts)
    avg_value = total_value / total_contracts

    # --- Per-vendor aggregation -----------------------------------------------
    vendor_agg: dict[str, dict] = {}
    for c in contracts:
        v = c["vendor"]
        if v not in vendor_agg:
            vendor_agg[v] = {"count": 0, "total": 0.0}
        vendor_agg[v]["count"] += 1
        vendor_agg[v]["total"] += c["value"]

    vendors = [
        VendorSummary(
            vendor=v,
            contract_count=agg["count"],
            total_value=round(agg["total"], 4),
            avg_value=round(agg["total"] / agg["count"], 4),
        )
        for v, agg in sorted(vendor_agg.items(), key=lambda x: -x[1]["total"])
    ]

    # --- Status counts --------------------------------------------------------
    status_counts: dict[str, int] | None = None
    if status_column is not None:
        status_counts = {}
        for c in contracts:
            s = c.get("status", "unknown")
            status_counts[s] = status_counts.get(s, 0) + 1

    # --- Average duration -----------------------------------------------------
    avg_duration_days: float | None = None
    if start_column is not None and end_column is not None:
        durations: list[float] = []
        for c in contracts:
            if "start" in c and "end" in c:
                delta = (c["end"] - c["start"]).total_seconds() / 86400.0
                durations.append(delta)
        if durations:
            avg_duration_days = round(sum(durations) / len(durations), 2)

    summary_parts = [
        f"{total_contracts} contracts across {len(vendors)} vendors.",
        f"Total value: {total_value:,.2f}, Avg value: {avg_value:,.2f}.",
    ]
    if status_counts:
        status_str = ", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items()))
        summary_parts.append(f"Status: {status_str}.")
    if avg_duration_days is not None:
        summary_parts.append(f"Avg duration: {avg_duration_days:.1f} days.")

    return ContractResult(
        total_contracts=total_contracts,
        total_value=round(total_value, 4),
        avg_value=round(avg_value, 4),
        vendors=vendors,
        status_counts=status_counts,
        avg_duration_days=avg_duration_days,
        summary=" ".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 2. detect_expiring_contracts
# ---------------------------------------------------------------------------


def detect_expiring_contracts(
    rows: list[dict],
    contract_column: str,
    end_column: str,
    vendor_column: str | None = None,
    value_column: str | None = None,
    reference_date: datetime | None = None,
    horizon_days: int = 90,
) -> list[ExpiringContract]:
    """Find contracts expiring within *horizon_days* of *reference_date*.

    Args:
        rows: Data rows as dicts.
        contract_column: Column identifying the contract.
        end_column: Column with contract end/expiry date.
        vendor_column: Optional column for vendor name.
        value_column: Optional column for contract value.
        reference_date: Anchor date; defaults to max end date in data.
        horizon_days: How many days ahead to look.

    Returns:
        List of ExpiringContract sorted by end_date ascending (most urgent first).
    """
    if not rows:
        return []

    # --- Parse all valid rows ------------------------------------------------
    parsed: list[dict] = []
    all_dates: list[datetime] = []
    for row in rows:
        contract = row.get(contract_column)
        end_dt = _parse_date(row.get(end_column))
        if contract is None or end_dt is None:
            continue
        entry: dict = {"contract": str(contract), "end": end_dt}
        if vendor_column is not None:
            v = row.get(vendor_column)
            entry["vendor"] = str(v) if v is not None else None
        else:
            entry["vendor"] = None
        if value_column is not None:
            entry["value"] = _safe_float(row.get(value_column))
        else:
            entry["value"] = None
        parsed.append(entry)
        all_dates.append(end_dt)

    if not parsed:
        return []

    # Default reference date = max end date
    if reference_date is None:
        reference_date = max(all_dates)

    horizon_end = reference_date + timedelta(days=horizon_days)

    # --- Filter contracts expiring within the window -------------------------
    expiring: list[ExpiringContract] = []
    for p in parsed:
        end_dt = p["end"]
        if reference_date <= end_dt <= horizon_end:
            days_until = (end_dt - reference_date).days
            if days_until <= 30:
                urgency = "critical"
            elif days_until <= 60:
                urgency = "warning"
            else:
                urgency = "upcoming"
            expiring.append(ExpiringContract(
                contract=p["contract"],
                vendor=p["vendor"],
                value=p["value"],
                end_date=end_dt,
                days_until_expiry=days_until,
                urgency=urgency,
            ))

    # Sort by end_date ascending (most urgent first)
    expiring.sort(key=lambda e: e.end_date)
    return expiring


# ---------------------------------------------------------------------------
# 3. vendor_contract_concentration
# ---------------------------------------------------------------------------


def vendor_contract_concentration(
    rows: list[dict],
    vendor_column: str,
    value_column: str,
) -> ConcentrationResult | None:
    """Measure vendor concentration using the Herfindahl-Hirschman Index.

    HHI = sum of squared market-share percentages.

    Risk rating:
        - "High"     if HHI > 2500
        - "Moderate"  if HHI > 1500
        - "Low"       otherwise

    Args:
        rows: Data rows as dicts.
        vendor_column: Column identifying the vendor.
        value_column: Column with contract monetary value.

    Returns:
        ConcentrationResult or None if no valid data.
    """
    if not rows:
        return None

    totals: dict[str, float] = {}
    for row in rows:
        vendor = row.get(vendor_column)
        value = _safe_float(row.get(value_column))
        if vendor is None or value is None:
            continue
        key = str(vendor)
        totals[key] = totals.get(key, 0.0) + value

    if not totals:
        return None

    grand_total = sum(totals.values())
    if grand_total == 0:
        return None

    # Build share list
    vendor_shares: list[VendorShare] = []
    for vendor, val in totals.items():
        share = val / grand_total * 100
        vendor_shares.append(VendorShare(
            vendor=vendor,
            total_value=round(val, 4),
            share_pct=round(share, 4),
        ))
    vendor_shares.sort(key=lambda v: -v.share_pct)

    # HHI
    hhi = sum(v.share_pct ** 2 for v in vendor_shares)
    hhi = round(hhi, 2)

    if hhi > 2500:
        risk_rating = "High"
    elif hhi > 1500:
        risk_rating = "Moderate"
    else:
        risk_rating = "Low"

    # Single-vendor dependency (>40% share)
    dependency_vendors = [v.vendor for v in vendor_shares if v.share_pct > 40]
    single_vendor_dependency = len(dependency_vendors) > 0

    summary = (
        f"Vendor concentration HHI = {hhi:.0f} ({risk_rating} risk). "
        f"{len(vendor_shares)} vendors. "
        f"Top vendor: {vendor_shares[0].vendor} ({vendor_shares[0].share_pct:.1f}%)."
    )
    if single_vendor_dependency:
        summary += (
            f" Dependency alert: {', '.join(dependency_vendors)} "
            f"exceed{'s' if len(dependency_vendors) == 1 else ''} 40% share."
        )

    return ConcentrationResult(
        hhi=hhi,
        risk_rating=risk_rating,
        top_vendors=vendor_shares,
        single_vendor_dependency=single_vendor_dependency,
        dependency_vendors=dependency_vendors,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. analyze_renewal_patterns
# ---------------------------------------------------------------------------


def analyze_renewal_patterns(
    rows: list[dict],
    contract_column: str,
    vendor_column: str,
    start_column: str,
    end_column: str,
    value_column: str | None = None,
) -> RenewalResult | None:
    """Detect contract renewal patterns across vendors.

    A renewal is detected when a vendor has two or more contracts whose date
    ranges are consecutive (next start <= previous end + 30 days gap) or
    overlapping.

    Args:
        rows: Data rows as dicts.
        contract_column: Column identifying the contract.
        vendor_column: Column identifying the vendor.
        start_column: Column with contract start date.
        end_column: Column with contract end date.
        value_column: Optional column for value; used to detect price escalation.

    Returns:
        RenewalResult or None if no valid data.
    """
    if not rows:
        return None

    # --- Parse valid entries --------------------------------------------------
    entries: list[dict] = []
    for row in rows:
        contract = row.get(contract_column)
        vendor = row.get(vendor_column)
        start_dt = _parse_date(row.get(start_column))
        end_dt = _parse_date(row.get(end_column))
        if contract is None or vendor is None or start_dt is None or end_dt is None:
            continue
        entry: dict = {
            "contract": str(contract),
            "vendor": str(vendor),
            "start": start_dt,
            "end": end_dt,
        }
        if value_column is not None:
            entry["value"] = _safe_float(row.get(value_column))
        else:
            entry["value"] = None
        entries.append(entry)

    if not entries:
        return None

    # --- Group by vendor, sort by start date ----------------------------------
    vendor_contracts: dict[str, list[dict]] = {}
    for e in entries:
        v = e["vendor"]
        vendor_contracts.setdefault(v, []).append(e)

    for v in vendor_contracts:
        vendor_contracts[v].sort(key=lambda c: c["start"])

    total_vendors = len(vendor_contracts)
    vendors_with_renewals = 0
    vendor_details: list[VendorRenewal] = []
    all_value_changes: list[float] = []

    for vendor, contracts in sorted(vendor_contracts.items()):
        count = len(contracts)
        renewal_count = 0
        value_changes: list[float] = []

        if count >= 2:
            for i in range(1, count):
                prev = contracts[i - 1]
                curr = contracts[i]
                # Consecutive/overlapping: curr.start <= prev.end + 30 days
                gap_days = (curr["start"] - prev["end"]).days
                if gap_days <= 30:
                    renewal_count += 1
                    # Value change
                    if (value_column is not None
                            and prev["value"] is not None
                            and curr["value"] is not None
                            and prev["value"] != 0):
                        change_pct = (
                            (curr["value"] - prev["value"]) / abs(prev["value"]) * 100
                        )
                        value_changes.append(change_pct)

        has_renewals = renewal_count > 0
        if has_renewals:
            vendors_with_renewals += 1

        avg_vc: float | None = None
        if value_changes:
            avg_vc = round(sum(value_changes) / len(value_changes), 2)
            all_value_changes.extend(value_changes)

        vendor_details.append(VendorRenewal(
            vendor=vendor,
            contract_count=count,
            has_renewals=has_renewals,
            renewal_count=renewal_count,
            avg_value_change_pct=avg_vc,
        ))

    renewal_rate = (
        round(vendors_with_renewals / total_vendors * 100, 2)
        if total_vendors > 0 else 0.0
    )

    avg_price_escalation: float | None = None
    if all_value_changes:
        avg_price_escalation = round(
            sum(all_value_changes) / len(all_value_changes), 2
        )

    summary_parts = [
        f"Renewal analysis across {total_vendors} vendors.",
        f"{vendors_with_renewals} vendor(s) show renewal patterns "
        f"(rate: {renewal_rate:.1f}%).",
    ]
    if avg_price_escalation is not None:
        summary_parts.append(
            f"Avg price escalation on renewals: {avg_price_escalation:+.1f}%."
        )

    return RenewalResult(
        total_vendors=total_vendors,
        vendors_with_renewals=vendors_with_renewals,
        renewal_rate=renewal_rate,
        vendor_details=vendor_details,
        avg_price_escalation_pct=avg_price_escalation,
        summary=" ".join(summary_parts),
    )


# ---------------------------------------------------------------------------
# 5. format_contract_report
# ---------------------------------------------------------------------------


def format_contract_report(
    contracts: ContractResult | None = None,
    expiring: list[ExpiringContract] | None = None,
    concentration: ConcentrationResult | None = None,
    renewals: RenewalResult | None = None,
) -> str:
    """Generate a combined text report from available contract analyses.

    Args:
        contracts: Overall contract analysis result.
        expiring: List of expiring contracts.
        concentration: Vendor concentration result.
        renewals: Renewal pattern result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Contract Analysis Report")
    sections.append("=" * 60)

    if contracts is not None:
        lines = ["", "Portfolio Overview", "-" * 58]
        lines.append(f"  Total contracts: {contracts.total_contracts}")
        lines.append(f"  Total value: {contracts.total_value:,.2f}")
        lines.append(f"  Avg value: {contracts.avg_value:,.2f}")
        lines.append(f"  Vendors: {len(contracts.vendors)}")
        if contracts.status_counts:
            status_str = ", ".join(
                f"{k}: {v}" for k, v in sorted(contracts.status_counts.items())
            )
            lines.append(f"  Status: {status_str}")
        if contracts.avg_duration_days is not None:
            lines.append(f"  Avg duration: {contracts.avg_duration_days:.1f} days")
        lines.append("")
        lines.append(f"  {'Vendor':<25}{'Count':>8}{'Total Value':>16}{'Avg Value':>14}")
        lines.append(f"  {'-' * 63}")
        for v in contracts.vendors:
            lines.append(
                f"  {v.vendor:<25}{v.contract_count:>8}"
                f"{v.total_value:>16,.2f}{v.avg_value:>14,.2f}"
            )
        sections.append("\n".join(lines))

    if expiring is not None and expiring:
        lines = ["", "Expiring Contracts", "-" * 58]
        lines.append(f"  {len(expiring)} contract(s) expiring soon:")
        lines.append("")
        for ec in expiring:
            vendor_str = f" ({ec.vendor})" if ec.vendor else ""
            value_str = f" value={ec.value:,.2f}" if ec.value is not None else ""
            lines.append(
                f"  [{ec.urgency.upper():>8}] {ec.contract}{vendor_str}"
                f" expires {ec.end_date.strftime('%Y-%m-%d')}"
                f" ({ec.days_until_expiry}d){value_str}"
            )
        sections.append("\n".join(lines))

    if concentration is not None:
        lines = ["", "Vendor Concentration", "-" * 58]
        lines.append(f"  HHI: {concentration.hhi:.0f} ({concentration.risk_rating} risk)")
        lines.append(f"  Single-vendor dependency: {'Yes' if concentration.single_vendor_dependency else 'No'}")
        if concentration.dependency_vendors:
            lines.append(f"  Dependency vendors: {', '.join(concentration.dependency_vendors)}")
        lines.append("")
        lines.append(f"  {'Vendor':<25}{'Value':>14}{'Share':>10}")
        lines.append(f"  {'-' * 49}")
        for v in concentration.top_vendors:
            lines.append(
                f"  {v.vendor:<25}{v.total_value:>14,.2f}{v.share_pct:>9.1f}%"
            )
        sections.append("\n".join(lines))

    if renewals is not None:
        lines = ["", "Renewal Patterns", "-" * 58]
        lines.append(f"  Total vendors: {renewals.total_vendors}")
        lines.append(f"  Vendors with renewals: {renewals.vendors_with_renewals}")
        lines.append(f"  Renewal rate: {renewals.renewal_rate:.1f}%")
        if renewals.avg_price_escalation_pct is not None:
            lines.append(
                f"  Avg price escalation: {renewals.avg_price_escalation_pct:+.1f}%"
            )
        lines.append("")
        for vd in renewals.vendor_details:
            renewal_str = (
                f"  {vd.vendor}: {vd.contract_count} contracts, "
                f"{vd.renewal_count} renewal(s)"
            )
            if vd.avg_value_change_pct is not None:
                renewal_str += f", value change: {vd.avg_value_change_pct:+.1f}%"
            lines.append(renewal_str)
        sections.append("\n".join(lines))

    if contracts is None and not expiring and concentration is None and renewals is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
