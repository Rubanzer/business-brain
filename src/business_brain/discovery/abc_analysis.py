"""ABC analysis — inventory classification based on cumulative value contribution.

Pure functions for classifying items into A (vital), B (important), C (trivial)
categories based on their contribution to total value. Common in inventory
management, supplier analysis, and customer segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ABCItem:
    """A single item with its ABC classification."""
    name: str
    value: float
    pct_of_total: float
    cumulative_pct: float
    rank: int
    category: str  # "A", "B", or "C"


@dataclass
class ABCResult:
    """Complete ABC analysis result."""
    items: list[ABCItem]
    total_value: float
    a_count: int
    b_count: int
    c_count: int
    a_value_pct: float
    b_value_pct: float
    c_value_pct: float
    a_item_pct: float
    b_item_pct: float
    c_item_pct: float
    summary: str


def abc_analysis(
    rows: list[dict],
    name_column: str,
    value_column: str,
    a_threshold: float = 80.0,
    b_threshold: float = 95.0,
) -> ABCResult | None:
    """Perform ABC classification on items.

    Args:
        rows: Data rows as dicts.
        name_column: Column identifying each item.
        value_column: Numeric column to classify by.
        a_threshold: Cumulative % threshold for A class (default 80%).
        b_threshold: Cumulative % threshold for A+B classes (default 95%).

    Returns:
        ABCResult or None if insufficient data.
    """
    if not rows:
        return None

    # Aggregate by name
    aggregated: dict[str, float] = {}
    for row in rows:
        name = row.get(name_column)
        val = row.get(value_column)
        if name is None or val is None:
            continue
        try:
            fval = abs(float(val))
        except (TypeError, ValueError):
            continue
        aggregated[str(name)] = aggregated.get(str(name), 0) + fval

    if len(aggregated) < 2:
        return None

    total = sum(aggregated.values())
    if total == 0:
        return None

    # Sort descending
    sorted_items = sorted(aggregated.items(), key=lambda x: -x[1])

    items: list[ABCItem] = []
    cumulative = 0.0
    a_count = b_count = c_count = 0
    a_value = b_value = c_value = 0.0

    for i, (name, value) in enumerate(sorted_items):
        pct = value / total * 100
        cumulative += pct

        if cumulative <= a_threshold or (i == 0 and cumulative > a_threshold):
            category = "A"
            a_count += 1
            a_value += pct
        elif cumulative <= b_threshold:
            category = "B"
            b_count += 1
            b_value += pct
        else:
            category = "C"
            c_count += 1
            c_value += pct

        items.append(ABCItem(
            name=name,
            value=round(value, 4),
            pct_of_total=round(pct, 2),
            cumulative_pct=round(cumulative, 2),
            rank=i + 1,
            category=category,
        ))

    n = len(items)
    summary = (
        f"ABC Analysis: {n} items classified. "
        f"A: {a_count} items ({a_count/n*100:.0f}%) = {a_value:.1f}% of value. "
        f"B: {b_count} items ({b_count/n*100:.0f}%) = {b_value:.1f}% of value. "
        f"C: {c_count} items ({c_count/n*100:.0f}%) = {c_value:.1f}% of value."
    )

    return ABCResult(
        items=items,
        total_value=round(total, 4),
        a_count=a_count,
        b_count=b_count,
        c_count=c_count,
        a_value_pct=round(a_value, 1),
        b_value_pct=round(b_value, 1),
        c_value_pct=round(c_value, 1),
        a_item_pct=round(a_count / n * 100, 1),
        b_item_pct=round(b_count / n * 100, 1),
        c_item_pct=round(c_count / n * 100, 1),
        summary=summary,
    )


def get_category_items(result: ABCResult, category: str) -> list[ABCItem]:
    """Get all items in a specific category."""
    return [item for item in result.items if item.category == category]


def abc_matrix(
    rows: list[dict],
    name_column: str,
    value_column: str,
    volume_column: str,
) -> list[dict]:
    """Create a 2D ABC matrix classifying by both value AND volume.

    Useful for inventory: high-value+high-volume = critical.
    Returns list of dicts with dual classification.
    """
    value_result = abc_analysis(rows, name_column, value_column)
    volume_result = abc_analysis(rows, name_column, volume_column)

    if value_result is None or volume_result is None:
        return []

    value_map = {item.name: item.category for item in value_result.items}
    volume_map = {item.name: item.category for item in volume_result.items}

    matrix = []
    for name in value_map:
        val_cat = value_map.get(name, "C")
        vol_cat = volume_map.get(name, "C")

        # Priority: AA=critical, AB/BA=important, rest=routine
        if val_cat == "A" and vol_cat == "A":
            priority = "critical"
        elif val_cat == "A" or vol_cat == "A":
            priority = "important"
        elif val_cat == "B" and vol_cat == "B":
            priority = "moderate"
        else:
            priority = "routine"

        matrix.append({
            "name": name,
            "value_category": val_cat,
            "volume_category": vol_cat,
            "combined": f"{val_cat}{vol_cat}",
            "priority": priority,
        })

    return sorted(matrix, key=lambda x: ("critical", "important", "moderate", "routine").index(x["priority"]))


def format_abc_table(result: ABCResult) -> str:
    """Format ABC analysis as a text table."""
    lines = [
        f"ABC Analysis",
        f"{'=' * 60}",
        f"Total Value: {result.total_value:,.2f}",
        f"",
        f"{'Rank':<6}{'Item':<25}{'Value':>12}{'%':>8}{'Cum%':>8}{'Class':>6}",
        f"{'-'*65}",
    ]
    for item in result.items:
        lines.append(
            f"{item.rank:<6}{item.name:<25}{item.value:>12,.2f}{item.pct_of_total:>7.1f}%{item.cumulative_pct:>7.1f}%{item.category:>6}"
        )
    lines.append(f"{'-'*65}")
    lines.append(f"A: {result.a_count} items ({result.a_value_pct:.1f}% value)")
    lines.append(f"B: {result.b_count} items ({result.b_value_pct:.1f}% value)")
    lines.append(f"C: {result.c_count} items ({result.c_value_pct:.1f}% value)")
    return "\n".join(lines)
