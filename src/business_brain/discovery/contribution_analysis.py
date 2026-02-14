"""Contribution analysis — determine what's driving changes in a total.

Pure functions for analyzing how individual components contribute to
overall changes between two periods.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContributionItem:
    """One component's contribution to the overall change."""
    name: str
    value_before: float
    value_after: float
    absolute_change: float
    pct_change: float  # own change as %
    contribution_pct: float  # % of total change this item explains
    direction: str  # "positive", "negative", "unchanged"


@dataclass
class ContributionResult:
    """Complete contribution analysis result."""
    items: list[ContributionItem]
    total_before: float
    total_after: float
    total_change: float
    total_change_pct: float
    top_positive_driver: str
    top_negative_driver: str
    concentration: float  # what % of change is explained by top item
    summary: str


def analyze_contributions(
    before: dict[str, float],
    after: dict[str, float],
) -> ContributionResult | None:
    """Analyze what drove the change from before to after.

    Args:
        before: Component name -> value in period 1.
        after: Component name -> value in period 2.

    Returns:
        ContributionResult or None if both empty.
    """
    all_keys = set(before.keys()) | set(after.keys())
    if not all_keys:
        return None

    total_before = sum(before.values())
    total_after = sum(after.values())
    total_change = total_after - total_before

    items: list[ContributionItem] = []
    for key in sorted(all_keys):
        val_before = before.get(key, 0.0)
        val_after = after.get(key, 0.0)
        abs_change = val_after - val_before
        pct_change = (abs_change / abs(val_before) * 100) if val_before != 0 else (100.0 if abs_change != 0 else 0.0)
        contrib_pct = (abs_change / abs(total_change) * 100) if total_change != 0 else 0.0

        if abs_change > 0.001:
            direction = "positive"
        elif abs_change < -0.001:
            direction = "negative"
        else:
            direction = "unchanged"

        items.append(ContributionItem(
            name=key,
            value_before=round(val_before, 4),
            value_after=round(val_after, 4),
            absolute_change=round(abs_change, 4),
            pct_change=round(pct_change, 2),
            contribution_pct=round(contrib_pct, 2),
            direction=direction,
        ))

    # Sort by absolute contribution descending
    items.sort(key=lambda x: -abs(x.absolute_change))

    total_change_pct = (total_change / abs(total_before) * 100) if total_before != 0 else 0.0

    positive = [i for i in items if i.direction == "positive"]
    negative = [i for i in items if i.direction == "negative"]
    top_pos = positive[0].name if positive else ""
    top_neg = negative[0].name if negative else ""

    concentration = abs(items[0].contribution_pct) if items else 0.0

    summary = (
        f"Total changed from {total_before:,.2f} to {total_after:,.2f} "
        f"({total_change:+,.2f}, {total_change_pct:+.1f}%). "
    )
    if top_pos:
        summary += f"Top positive driver: {top_pos}. "
    if top_neg:
        summary += f"Top negative driver: {top_neg}. "
    summary += f"Top item explains {concentration:.0f}% of change."

    return ContributionResult(
        items=items,
        total_before=round(total_before, 4),
        total_after=round(total_after, 4),
        total_change=round(total_change, 4),
        total_change_pct=round(total_change_pct, 2),
        top_positive_driver=top_pos,
        top_negative_driver=top_neg,
        concentration=round(concentration, 1),
        summary=summary,
    )


def contribution_from_rows(
    rows_before: list[dict],
    rows_after: list[dict],
    name_column: str,
    value_column: str,
) -> ContributionResult | None:
    """Build contribution analysis from row data.

    Aggregates values by name_column in each period.
    """
    def _aggregate(rows):
        agg = {}
        for row in rows:
            name = row.get(name_column)
            val = row.get(value_column)
            if name is None or val is None:
                continue
            try:
                agg[str(name)] = agg.get(str(name), 0.0) + float(val)
            except (TypeError, ValueError):
                continue
        return agg

    before = _aggregate(rows_before)
    after = _aggregate(rows_after)

    if not before and not after:
        return None

    return analyze_contributions(before, after)


def waterfall_data(result: ContributionResult) -> list[dict]:
    """Generate waterfall chart data from contribution analysis.

    Returns list of bars: starting value, each component's contribution, ending value.
    """
    data = [{"label": "Start", "value": result.total_before, "type": "total"}]

    running = result.total_before
    for item in result.items:
        if abs(item.absolute_change) < 0.01:
            continue
        data.append({
            "label": item.name,
            "value": item.absolute_change,
            "start": running,
            "end": running + item.absolute_change,
            "type": "positive" if item.absolute_change > 0 else "negative",
        })
        running += item.absolute_change

    data.append({"label": "End", "value": result.total_after, "type": "total"})
    return data
