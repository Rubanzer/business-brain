"""Inventory optimizer — EOQ, reorder points, safety stock, and turnover analysis.

Pure functions for inventory analysis and optimization in manufacturing.
Computes Economic Order Quantity, reorder points, safety stock levels,
inventory turnover ratios, and overall inventory health assessments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ItemTurnover:
    """Turnover metrics for a single inventory item."""
    item: str
    cogs: float
    avg_inventory: float
    turnover_ratio: float
    days_of_inventory: float
    category: str  # "fast", "normal", "slow"


@dataclass
class TurnoverResult:
    """Aggregate inventory turnover analysis."""
    items: list[ItemTurnover]
    mean_turnover: float
    best_item: str
    worst_item: str
    slow_movers: list[str]   # turnover < 2
    fast_movers: list[str]   # turnover > 12
    summary: str


@dataclass
class EOQResult:
    """Economic Order Quantity calculation result."""
    eoq: float
    annual_orders: float
    order_interval_days: float
    total_ordering_cost: float
    total_holding_cost: float
    total_cost: float


@dataclass
class ReorderPoint:
    """Reorder point calculation result."""
    rop: float
    daily_demand: float
    lead_time_days: float
    safety_stock: float


@dataclass
class ItemHealth:
    """Health status for a single inventory item."""
    item: str
    current_qty: float
    min_qty: float | None
    max_qty: float | None
    reorder_point: float | None
    status: str  # "overstocked", "understocked", "reorder", "healthy"


@dataclass
class HealthResult:
    """Aggregate inventory health analysis."""
    items: list[ItemHealth]
    overstocked_count: int
    understocked_count: int
    at_reorder_count: int
    healthy_count: int
    summary: str


# ---------------------------------------------------------------------------
# Z-score lookup for common service levels
# ---------------------------------------------------------------------------

_Z_SCORES: dict[float, float] = {
    0.90: 1.282,
    0.95: 1.645,
    0.99: 2.326,
}


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def compute_inventory_turnover(
    rows: list[dict],
    item_column: str,
    cost_of_goods_column: str,
    avg_inventory_column: str,
) -> TurnoverResult | None:
    """Compute inventory turnover ratio for each item.

    Args:
        rows: Data rows as dicts.
        item_column: Column identifying the item/SKU.
        cost_of_goods_column: Column with cost of goods sold.
        avg_inventory_column: Column with average inventory value.

    Returns:
        TurnoverResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate COGS and average inventory per item
    cogs_totals: dict[str, float] = {}
    inv_totals: dict[str, float] = {}
    inv_counts: dict[str, int] = {}

    for row in rows:
        item = row.get(item_column)
        cogs_val = row.get(cost_of_goods_column)
        inv_val = row.get(avg_inventory_column)

        if item is None or cogs_val is None or inv_val is None:
            continue

        try:
            cogs_f = float(cogs_val)
            inv_f = float(inv_val)
        except (TypeError, ValueError):
            continue

        item_str = str(item)
        cogs_totals[item_str] = cogs_totals.get(item_str, 0) + cogs_f
        inv_totals[item_str] = inv_totals.get(item_str, 0) + inv_f
        inv_counts[item_str] = inv_counts.get(item_str, 0) + 1

    if not cogs_totals:
        return None

    # Build item turnover list
    item_turnovers: list[ItemTurnover] = []
    for item_str in cogs_totals:
        cogs = cogs_totals[item_str]
        avg_inv = inv_totals[item_str] / inv_counts[item_str]

        if avg_inv <= 0:
            turnover_ratio = 0.0
            days = 0.0
        else:
            turnover_ratio = cogs / avg_inv
            days = 365.0 / turnover_ratio if turnover_ratio > 0 else 0.0

        if turnover_ratio > 12:
            category = "fast"
        elif turnover_ratio < 2:
            category = "slow"
        else:
            category = "normal"

        item_turnovers.append(ItemTurnover(
            item=item_str,
            cogs=round(cogs, 4),
            avg_inventory=round(avg_inv, 4),
            turnover_ratio=round(turnover_ratio, 4),
            days_of_inventory=round(days, 2),
            category=category,
        ))

    if not item_turnovers:
        return None

    # Sort by turnover descending
    item_turnovers.sort(key=lambda it: -it.turnover_ratio)

    mean_turnover = sum(it.turnover_ratio for it in item_turnovers) / len(item_turnovers)
    best = item_turnovers[0].item
    worst = item_turnovers[-1].item
    slow = [it.item for it in item_turnovers if it.turnover_ratio < 2]
    fast = [it.item for it in item_turnovers if it.turnover_ratio > 12]

    summary = (
        f"Inventory turnover analysis: {len(item_turnovers)} items, "
        f"mean turnover {mean_turnover:.2f}x. "
        f"Best: {best}, Worst: {worst}. "
        f"{len(slow)} slow movers (<2x), {len(fast)} fast movers (>12x)."
    )

    return TurnoverResult(
        items=item_turnovers,
        mean_turnover=round(mean_turnover, 4),
        best_item=best,
        worst_item=worst,
        slow_movers=slow,
        fast_movers=fast,
        summary=summary,
    )


def compute_eoq(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_per_unit: float,
) -> EOQResult | None:
    """Compute Economic Order Quantity (EOQ).

    Uses the Wilson EOQ formula: EOQ = sqrt(2 * D * S / H)

    Args:
        annual_demand: Annual demand in units (D).
        ordering_cost: Cost per order placed (S).
        holding_cost_per_unit: Annual holding cost per unit (H).

    Returns:
        EOQResult or None if inputs are invalid (zero or negative).
    """
    if annual_demand <= 0 or ordering_cost <= 0 or holding_cost_per_unit <= 0:
        return None

    eoq = math.sqrt(2 * annual_demand * ordering_cost / holding_cost_per_unit)
    annual_orders = annual_demand / eoq
    order_interval_days = 365.0 / annual_orders
    total_ordering_cost = annual_orders * ordering_cost
    total_holding_cost = (eoq / 2) * holding_cost_per_unit
    total_cost = total_ordering_cost + total_holding_cost

    return EOQResult(
        eoq=round(eoq, 4),
        annual_orders=round(annual_orders, 4),
        order_interval_days=round(order_interval_days, 2),
        total_ordering_cost=round(total_ordering_cost, 4),
        total_holding_cost=round(total_holding_cost, 4),
        total_cost=round(total_cost, 4),
    )


def compute_reorder_point(
    daily_demand: float,
    lead_time_days: float,
    safety_stock: float = 0,
) -> ReorderPoint:
    """Compute reorder point.

    ROP = daily_demand * lead_time_days + safety_stock

    Args:
        daily_demand: Average daily demand in units.
        lead_time_days: Lead time in days.
        safety_stock: Safety stock buffer (default 0).

    Returns:
        ReorderPoint with computed ROP.
    """
    rop = daily_demand * lead_time_days + safety_stock

    return ReorderPoint(
        rop=round(rop, 4),
        daily_demand=daily_demand,
        lead_time_days=lead_time_days,
        safety_stock=safety_stock,
    )


def compute_safety_stock(
    daily_demand_std: float,
    lead_time_days: float,
    service_level: float = 0.95,
) -> float:
    """Compute safety stock level.

    Safety stock = Z * sigma_demand * sqrt(lead_time)

    Args:
        daily_demand_std: Standard deviation of daily demand.
        lead_time_days: Lead time in days.
        service_level: Desired service level (0.90, 0.95, or 0.99).

    Returns:
        Safety stock quantity (float). Returns 0.0 for invalid inputs.
    """
    if daily_demand_std <= 0 or lead_time_days <= 0:
        return 0.0

    z = _Z_SCORES.get(service_level)
    if z is None:
        # Default to 0.95 if unrecognized service level
        z = _Z_SCORES[0.95]

    return round(z * daily_demand_std * math.sqrt(lead_time_days), 4)


def analyze_inventory_health(
    rows: list[dict],
    item_column: str,
    quantity_column: str,
    min_column: str | None = None,
    max_column: str | None = None,
    reorder_column: str | None = None,
) -> HealthResult | None:
    """Analyze inventory health — overstocked, understocked, at reorder point.

    Args:
        rows: Data rows as dicts.
        item_column: Column identifying the item/SKU.
        quantity_column: Column with current on-hand quantity.
        min_column: Optional column with minimum stock level.
        max_column: Optional column with maximum stock level.
        reorder_column: Optional column with reorder point.

    Returns:
        HealthResult or None if no valid data.
    """
    if not rows:
        return None

    items: list[ItemHealth] = []

    for row in rows:
        item = row.get(item_column)
        qty_val = row.get(quantity_column)

        if item is None or qty_val is None:
            continue

        try:
            qty = float(qty_val)
        except (TypeError, ValueError):
            continue

        min_qty: float | None = None
        max_qty: float | None = None
        reorder_pt: float | None = None

        if min_column is not None:
            raw = row.get(min_column)
            if raw is not None:
                try:
                    min_qty = float(raw)
                except (TypeError, ValueError):
                    pass

        if max_column is not None:
            raw = row.get(max_column)
            if raw is not None:
                try:
                    max_qty = float(raw)
                except (TypeError, ValueError):
                    pass

        if reorder_column is not None:
            raw = row.get(reorder_column)
            if raw is not None:
                try:
                    reorder_pt = float(raw)
                except (TypeError, ValueError):
                    pass

        # Determine status
        status = _classify_health(qty, min_qty, max_qty, reorder_pt)

        items.append(ItemHealth(
            item=str(item),
            current_qty=qty,
            min_qty=min_qty,
            max_qty=max_qty,
            reorder_point=reorder_pt,
            status=status,
        ))

    if not items:
        return None

    overstocked = sum(1 for it in items if it.status == "overstocked")
    understocked = sum(1 for it in items if it.status == "understocked")
    at_reorder = sum(1 for it in items if it.status == "reorder")
    healthy = sum(1 for it in items if it.status == "healthy")

    summary = (
        f"Inventory health: {len(items)} items analyzed. "
        f"{healthy} healthy, {overstocked} overstocked, "
        f"{understocked} understocked, {at_reorder} at reorder point."
    )

    return HealthResult(
        items=items,
        overstocked_count=overstocked,
        understocked_count=understocked,
        at_reorder_count=at_reorder,
        healthy_count=healthy,
        summary=summary,
    )


def _classify_health(
    qty: float,
    min_qty: float | None,
    max_qty: float | None,
    reorder_pt: float | None,
) -> str:
    """Classify a single item's inventory health status."""
    if max_qty is not None and qty > max_qty:
        return "overstocked"
    if min_qty is not None and qty < min_qty:
        return "understocked"
    if reorder_pt is not None and qty <= reorder_pt:
        return "reorder"
    return "healthy"


def format_inventory_report(
    turnover: TurnoverResult | None = None,
    health: HealthResult | None = None,
) -> str:
    """Format a combined text report from turnover and/or health results.

    Args:
        turnover: Optional TurnoverResult from compute_inventory_turnover.
        health: Optional HealthResult from analyze_inventory_health.

    Returns:
        Human-readable text summary.
    """
    if turnover is None and health is None:
        return "No inventory data available for report."

    sections: list[str] = []
    sections.append("=== Inventory Report ===")
    sections.append("")

    if turnover is not None:
        sections.append("--- Turnover Analysis ---")
        sections.append(turnover.summary)
        sections.append("")
        sections.append(f"  Mean Turnover: {turnover.mean_turnover:.2f}x")
        sections.append(f"  Best Item:     {turnover.best_item}")
        sections.append(f"  Worst Item:    {turnover.worst_item}")

        if turnover.slow_movers:
            sections.append(f"  Slow Movers:   {', '.join(turnover.slow_movers)}")
        if turnover.fast_movers:
            sections.append(f"  Fast Movers:   {', '.join(turnover.fast_movers)}")

        sections.append("")
        sections.append("  Item Details:")
        for it in turnover.items:
            sections.append(
                f"    {it.item}: turnover={it.turnover_ratio:.2f}x, "
                f"days={it.days_of_inventory:.0f}, category={it.category}"
            )
        sections.append("")

    if health is not None:
        sections.append("--- Inventory Health ---")
        sections.append(health.summary)
        sections.append("")
        sections.append(f"  Healthy:       {health.healthy_count}")
        sections.append(f"  Overstocked:   {health.overstocked_count}")
        sections.append(f"  Understocked:  {health.understocked_count}")
        sections.append(f"  At Reorder:    {health.at_reorder_count}")

        sections.append("")
        sections.append("  Item Details:")
        for it in health.items:
            sections.append(
                f"    {it.item}: qty={it.current_qty:.0f}, status={it.status}"
            )
        sections.append("")

    return "\n".join(sections)
