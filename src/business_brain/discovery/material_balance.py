"""Material balance and mass flow analysis for manufacturing.

Pure functions for tracking input vs output, losses, material efficiency,
leakage detection through processing stages, and raw material mix analysis.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Material Balance
# ---------------------------------------------------------------------------


@dataclass
class EntityBalance:
    """Material balance for a single entity."""

    entity: str
    total_input: float
    total_output: float
    loss: float
    recovery_pct: float
    loss_pct: float


@dataclass
class MaterialBalanceResult:
    """Aggregated material balance across all entities."""

    entities: list[EntityBalance]
    total_input: float
    total_output: float
    total_loss: float
    overall_recovery_pct: float
    worst_recovery_entity: str
    best_recovery_entity: str
    summary: str


def compute_material_balance(
    rows: list[dict],
    entity_column: str,
    input_column: str,
    output_column: str,
    loss_column: str | None = None,
) -> MaterialBalanceResult | None:
    """Compute material balance per entity from row data.

    For each entity the function totals input and output quantities. Loss is
    either taken from a dedicated loss column or computed as input - output.
    Recovery percentage is output / input * 100.

    Args:
        rows: Data rows as dicts.
        entity_column: Column identifying the entity (plant, line, etc.).
        input_column: Column with input material quantity.
        output_column: Column with output material quantity.
        loss_column: Optional column with explicit loss quantity. When absent
            loss is computed as total_input - total_output.

    Returns:
        MaterialBalanceResult or None if no valid data.
    """
    if not rows:
        return None

    acc: dict[str, dict[str, float]] = {}
    for row in rows:
        entity = row.get(entity_column)
        inp = row.get(input_column)
        out = row.get(output_column)
        if entity is None or inp is None or out is None:
            continue
        try:
            inp_val = float(inp)
            out_val = float(out)
        except (TypeError, ValueError):
            continue

        loss_val = 0.0
        if loss_column is not None:
            raw = row.get(loss_column)
            if raw is not None:
                try:
                    loss_val = float(raw)
                except (TypeError, ValueError):
                    loss_val = 0.0

        key = str(entity)
        if key not in acc:
            acc[key] = {"input": 0.0, "output": 0.0, "loss": 0.0}
        acc[key]["input"] += inp_val
        acc[key]["output"] += out_val
        acc[key]["loss"] += loss_val

    if not acc:
        return None

    entities: list[EntityBalance] = []
    for name, vals in acc.items():
        total_inp = vals["input"]
        total_out = vals["output"]
        if loss_column is not None:
            loss = vals["loss"]
        else:
            loss = total_inp - total_out

        if total_inp == 0:
            recovery_pct = 0.0
            loss_pct = 0.0
        else:
            recovery_pct = total_out / total_inp * 100
            loss_pct = loss / total_inp * 100

        entities.append(
            EntityBalance(
                entity=name,
                total_input=round(total_inp, 4),
                total_output=round(total_out, 4),
                loss=round(loss, 4),
                recovery_pct=round(recovery_pct, 2),
                loss_pct=round(loss_pct, 2),
            )
        )

    # Sort by recovery_pct descending (best first)
    entities.sort(key=lambda e: e.recovery_pct, reverse=True)

    total_input = sum(e.total_input for e in entities)
    total_output = sum(e.total_output for e in entities)
    total_loss = sum(e.loss for e in entities)
    overall_recovery = (total_output / total_input * 100) if total_input != 0 else 0.0

    best = entities[0]
    worst = entities[-1]

    summary = (
        f"Material balance across {len(entities)} entities: "
        f"Total input = {total_input:,.2f}, Total output = {total_output:,.2f}, "
        f"Total loss = {total_loss:,.2f}. "
        f"Overall recovery = {overall_recovery:.1f}%. "
        f"Best = {best.entity} ({best.recovery_pct:.1f}%), "
        f"Worst = {worst.entity} ({worst.recovery_pct:.1f}%)."
    )

    return MaterialBalanceResult(
        entities=entities,
        total_input=round(total_input, 4),
        total_output=round(total_output, 4),
        total_loss=round(total_loss, 4),
        overall_recovery_pct=round(overall_recovery, 2),
        worst_recovery_entity=worst.entity,
        best_recovery_entity=best.entity,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Leakage Detection
# ---------------------------------------------------------------------------


@dataclass
class LeakagePoint:
    """A point of material loss between two processing stages."""

    from_stage: str
    to_stage: str
    input_qty: float
    output_qty: float
    loss: float
    loss_pct: float
    severity: str  # "critical" if >10%, "warning" if >5%, "ok"


def _leakage_severity(loss_pct: float) -> str:
    """Classify leakage severity based on loss percentage."""
    if loss_pct > 10:
        return "critical"
    if loss_pct > 5:
        return "warning"
    return "ok"


def detect_material_leakage(
    rows: list[dict],
    stage_column: str,
    quantity_column: str,
    sequence: list[str] | None = None,
) -> list[LeakagePoint]:
    """Track material through processing stages and identify losses.

    Each stage should have a total quantity. The function compares consecutive
    stages to determine where losses occur.

    Args:
        rows: Data rows as dicts.
        stage_column: Column identifying the processing stage.
        quantity_column: Column with material quantity at that stage.
        sequence: Optional ordered list of stage names. When absent stages
            are sorted alphabetically.

    Returns:
        List of LeakagePoint objects for each stage transition.
    """
    if not rows:
        return []

    # Aggregate quantity per stage
    stage_totals: dict[str, float] = {}
    for row in rows:
        stage = row.get(stage_column)
        qty = row.get(quantity_column)
        if stage is None or qty is None:
            continue
        try:
            qty_val = float(qty)
        except (TypeError, ValueError):
            continue
        key = str(stage)
        stage_totals[key] = stage_totals.get(key, 0.0) + qty_val

    if len(stage_totals) < 2:
        return []

    # Determine stage ordering
    if sequence is not None:
        ordered_stages = [s for s in sequence if s in stage_totals]
    else:
        ordered_stages = sorted(stage_totals.keys())

    if len(ordered_stages) < 2:
        return []

    results: list[LeakagePoint] = []
    for i in range(len(ordered_stages) - 1):
        from_stage = ordered_stages[i]
        to_stage = ordered_stages[i + 1]
        input_qty = stage_totals[from_stage]
        output_qty = stage_totals[to_stage]
        loss = input_qty - output_qty
        loss_pct = (loss / input_qty * 100) if input_qty != 0 else 0.0

        results.append(
            LeakagePoint(
                from_stage=from_stage,
                to_stage=to_stage,
                input_qty=round(input_qty, 4),
                output_qty=round(output_qty, 4),
                loss=round(loss, 4),
                loss_pct=round(loss_pct, 2),
                severity=_leakage_severity(loss_pct),
            )
        )

    return results


# ---------------------------------------------------------------------------
# Mix Analysis
# ---------------------------------------------------------------------------


@dataclass
class ComponentMix:
    """A single component in the material mix."""

    component: str
    quantity: float
    actual_ratio: float
    target_ratio: float | None  # None when no recipe given
    deviation_pct: float | None  # None when no recipe given


@dataclass
class MixResult:
    """Complete mix analysis result."""

    components: list[ComponentMix]
    total_quantity: float
    deviation_from_recipe: float | None  # mean absolute deviation; None if no recipe
    summary: str


def compute_mix_analysis(
    rows: list[dict],
    component_column: str,
    quantity_column: str,
    recipe: dict[str, float] | None = None,
) -> MixResult | None:
    """Analyze raw material mix ratios and optionally compare to a target recipe.

    Args:
        rows: Data rows as dicts.
        component_column: Column identifying the component/raw material.
        quantity_column: Column with quantity of that component.
        recipe: Optional dict mapping component names to target ratios
            (should sum to 1.0), e.g. {"iron_ore": 0.6, "coal": 0.3, "flux": 0.1}.

    Returns:
        MixResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate quantity per component
    comp_totals: dict[str, float] = {}
    for row in rows:
        comp = row.get(component_column)
        qty = row.get(quantity_column)
        if comp is None or qty is None:
            continue
        try:
            qty_val = float(qty)
        except (TypeError, ValueError):
            continue
        key = str(comp)
        comp_totals[key] = comp_totals.get(key, 0.0) + qty_val

    if not comp_totals:
        return None

    total_qty = sum(comp_totals.values())
    if total_qty == 0:
        return None

    components: list[ComponentMix] = []
    deviations: list[float] = []

    for name in sorted(comp_totals.keys()):
        qty = comp_totals[name]
        actual_ratio = qty / total_qty

        target_ratio: float | None = None
        deviation_pct: float | None = None

        if recipe is not None:
            target_ratio = recipe.get(name, 0.0)
            if target_ratio != 0:
                deviation_pct = (actual_ratio - target_ratio) / target_ratio * 100
            else:
                # Target is 0 but we have material — deviation is the actual ratio
                # expressed as a percentage (infinite relative, but cap at actual * 100)
                deviation_pct = actual_ratio * 100
            deviations.append(abs(actual_ratio - target_ratio))

        components.append(
            ComponentMix(
                component=name,
                quantity=round(qty, 4),
                actual_ratio=round(actual_ratio, 4),
                target_ratio=round(target_ratio, 4) if target_ratio is not None else None,
                deviation_pct=round(deviation_pct, 2) if deviation_pct is not None else None,
            )
        )

    # If recipe given, also add components in recipe that are missing from data
    if recipe is not None:
        existing_names = {c.component for c in components}
        for name, target in sorted(recipe.items()):
            if name not in existing_names:
                deviation_pct = -100.0  # completely missing
                deviations.append(abs(target))
                components.append(
                    ComponentMix(
                        component=name,
                        quantity=0.0,
                        actual_ratio=0.0,
                        target_ratio=round(target, 4),
                        deviation_pct=round(deviation_pct, 2),
                    )
                )

    # Mean absolute deviation from recipe
    mean_deviation: float | None = None
    if recipe is not None and deviations:
        mean_deviation = sum(deviations) / len(deviations)

    # Build summary
    comp_parts = ", ".join(
        f"{c.component} {c.actual_ratio:.1%}" for c in components if c.quantity > 0
    )
    summary = (
        f"Mix analysis: {len([c for c in components if c.quantity > 0])} components, "
        f"total quantity = {total_qty:,.2f}. "
        f"Ratios: {comp_parts}."
    )
    if mean_deviation is not None:
        summary += f" Mean absolute deviation from recipe = {mean_deviation:.4f}."

    return MixResult(
        components=components,
        total_quantity=round(total_qty, 4),
        deviation_from_recipe=round(mean_deviation, 4) if mean_deviation is not None else None,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Report Formatter
# ---------------------------------------------------------------------------


def format_balance_report(
    result: MaterialBalanceResult,
    leakage: list[LeakagePoint] | None = None,
) -> str:
    """Generate a human-readable text report from material balance results.

    Args:
        result: A MaterialBalanceResult from compute_material_balance.
        leakage: Optional list of LeakagePoint from detect_material_leakage.

    Returns:
        Multi-line formatted report string.
    """
    sections: list[str] = []
    sections.append("Material Balance Report")
    sections.append("=" * 40)

    # Overall summary
    lines = [
        "",
        "Overall",
        "-" * 38,
        f"  Total Input:    {result.total_input:>12,.2f}",
        f"  Total Output:   {result.total_output:>12,.2f}",
        f"  Total Loss:     {result.total_loss:>12,.2f}",
        f"  Recovery:       {result.overall_recovery_pct:>11.1f}%",
    ]
    sections.append("\n".join(lines))

    # Per-entity breakdown
    lines = ["", "Entity Breakdown", "-" * 38]
    for e in result.entities:
        lines.append(
            f"  {e.entity}: input={e.total_input:,.2f} output={e.total_output:,.2f} "
            f"loss={e.loss:,.2f} recovery={e.recovery_pct:.1f}% loss={e.loss_pct:.1f}%"
        )
    lines.append(f"  Best:  {result.best_recovery_entity}")
    lines.append(f"  Worst: {result.worst_recovery_entity}")
    sections.append("\n".join(lines))

    # Leakage section
    if leakage is not None and leakage:
        lines = ["", "Leakage Analysis", "-" * 38]
        for lp in leakage:
            lines.append(
                f"  {lp.from_stage} -> {lp.to_stage}: "
                f"in={lp.input_qty:,.2f} out={lp.output_qty:,.2f} "
                f"loss={lp.loss:,.2f} ({lp.loss_pct:.1f}%) [{lp.severity}]"
            )
        critical_count = sum(1 for lp in leakage if lp.severity == "critical")
        warning_count = sum(1 for lp in leakage if lp.severity == "warning")
        lines.append(
            f"  Critical: {critical_count} | Warning: {warning_count}"
        )
        sections.append("\n".join(lines))

    return "\n".join(sections)
