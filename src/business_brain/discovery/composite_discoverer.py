"""Template-based composite metric discovery across profiled tables."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from business_brain.db.discovery_models import (
    DiscoveredRelationship,
    Insight,
    TableProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class CompositeTemplate:
    """Definition of a composite metric template."""

    name: str
    required_signals: list[str]  # keywords that must match column names
    entity_keywords: list[str]  # what entity type this applies to
    description: str
    formula_hint: str

    def match_columns(
        self, all_columns: list[tuple[str, str, dict]],
    ) -> list[tuple[str, str]]:
        """Return matched (table, column) pairs for each required signal.

        all_columns: list of (table_name, column_name, column_info)
        Returns list of (table_name, column_name) matches — one per signal.
        """
        matched: list[tuple[str, str]] = []
        used: set[tuple[str, str]] = set()

        for signal in self.required_signals:
            for table, col, info in all_columns:
                if (table, col) in used:
                    continue
                if signal.lower() in col.lower():
                    matched.append((table, col))
                    used.add((table, col))
                    break

        return matched if len(matched) == len(self.required_signals) else []


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

TEMPLATES: list[CompositeTemplate] = [
    CompositeTemplate(
        name="Buyer Credit Score",
        required_signals=["payment", "order", "total"],
        entity_keywords=["customer", "buyer", "party", "client"],
        description="Creditworthiness score from payment history, order volume, and spend",
        formula_hint="NORMALIZE(payment_delay)*0.4 + NORMALIZE(order_count)*0.3 + NORMALIZE(total_spend)*0.3",
    ),
    CompositeTemplate(
        name="Employee Performance Index",
        required_signals=["task", "attend"],
        entity_keywords=["employee", "staff", "worker"],
        description="Performance index combining productivity and attendance",
        formula_hint="NORMALIZE(tasks_completed)*0.6 + NORMALIZE(attendance)*0.4",
    ),
    CompositeTemplate(
        name="Supplier Risk Score",
        required_signals=["rate", "quality"],
        entity_keywords=["supplier", "vendor", "party"],
        description="Supplier risk assessment from cost and quality metrics",
        formula_hint="NORMALIZE(rate)*0.5 + NORMALIZE(quality)*0.5",
    ),
    CompositeTemplate(
        name="Product Profitability",
        required_signals=["revenue", "cost"],
        entity_keywords=["product", "item", "sku"],
        description="Profitability scoring per product from revenue and cost",
        formula_hint="(revenue - cost) / revenue * 100",
    ),
    CompositeTemplate(
        name="Customer Churn Risk",
        required_signals=["last_order", "order_count"],
        entity_keywords=["customer", "client"],
        description="RFM-based churn prediction from recency and frequency",
        formula_hint="NORMALIZE(days_since_last_order)*0.5 + (1-NORMALIZE(order_count))*0.5",
    ),
    # --- Manufacturing-specific templates ---
    CompositeTemplate(
        name="Furnace Efficiency Score",
        required_signals=["kva", "output", "temperature"],
        entity_keywords=["furnace", "heat", "melt", "casting"],
        description="Furnace efficiency combining power consumption, output tonnage, and operating temperature",
        formula_hint="output_tonnage / (kva * hours) — lower kWh/ton = higher efficiency",
    ),
    CompositeTemplate(
        name="Material Yield Tracker",
        required_signals=["input", "output"],
        entity_keywords=["furnace", "heat", "production", "melt", "rolling"],
        description="Tracks material yield ratio from input to output weight",
        formula_hint="(output_weight / input_weight) * 100 — target > 90%",
    ),
    CompositeTemplate(
        name="Equipment Health Index",
        required_signals=["breakdown", "runtime"],
        entity_keywords=["machine", "equipment", "furnace", "rolling"],
        description="Equipment health from breakdown frequency vs runtime hours",
        formula_hint="runtime_hours / (runtime_hours + breakdown_hours) * 100 — OEE component",
    ),
    CompositeTemplate(
        name="Power Consumption per Ton",
        required_signals=["power", "tonnage"],
        entity_keywords=["production", "furnace", "plant", "mill"],
        description="Specific energy consumption metric — kWh per ton of output",
        formula_hint="total_power_kwh / total_tonnage — lower is better",
    ),
    CompositeTemplate(
        name="Supplier Quality Score",
        required_signals=["fe", "rejection"],
        entity_keywords=["supplier", "vendor", "party", "material"],
        description="Supplier quality assessment from Fe content and rejection rates",
        formula_hint="NORMALIZE(fe_content)*0.6 + (1-NORMALIZE(rejection_rate))*0.4",
    ),
    # --- Heat-level tracking templates ---
    CompositeTemplate(
        name="Heat Cycle Time",
        required_signals=["tap", "charge"],
        entity_keywords=["heat", "furnace", "melt", "casting"],
        description="Tap-to-tap cycle time tracking per heat — measures furnace turnaround",
        formula_hint="tap_time - charge_time — lower cycle time = higher throughput",
    ),
    CompositeTemplate(
        name="Alloy Addition Efficiency",
        required_signals=["alloy", "grade"],
        entity_keywords=["heat", "furnace", "melt", "ladle"],
        description="Alloy addition efficiency — input alloys vs achieved grade",
        formula_hint="grade_achieved / target_grade * 100 — higher = less rework",
    ),
    CompositeTemplate(
        name="Slag Rate Monitor",
        required_signals=["slag", "metal"],
        entity_keywords=["heat", "furnace", "melt", "casting"],
        description="Slag-to-metal ratio monitoring per heat — critical for quality",
        formula_hint="slag_weight / metal_weight * 100 — target < 5%",
    ),
    CompositeTemplate(
        name="Electrode Consumption Tracker",
        required_signals=["electrode", "tonnage"],
        entity_keywords=["furnace", "heat", "eaf", "melt"],
        description="Electrode consumption per ton of steel produced",
        formula_hint="electrode_kg / tonnage — lower is better, typical: 1.5-3.0 kg/ton",
    ),
]


def discover_composites(
    profiles: list[TableProfile],
    relationships: list[DiscoveredRelationship],
) -> list[Insight]:
    """Match composite metric templates against all profiled columns."""
    insights: list[Insight] = []

    # Build flat list of (table, column, info)
    all_columns: list[tuple[str, str, dict]] = []
    for profile in profiles:
        cls = profile.column_classification or {}
        cols = cls.get("columns", {})
        for col_name, col_info in cols.items():
            all_columns.append((profile.table_name, col_name, col_info))

    # Track which tables have relationships
    related_tables: dict[str, set[str]] = {}
    for rel in relationships:
        related_tables.setdefault(rel.table_a, set()).add(rel.table_b)
        related_tables.setdefault(rel.table_b, set()).add(rel.table_a)

    for template in TEMPLATES:
        try:
            matched = template.match_columns(all_columns)
            if not matched:
                continue

            # Determine if cross-table
            matched_tables = {t for t, c in matched}
            is_cross_table = len(matched_tables) > 1

            # Check entity relevance
            entity_match = _check_entity(template.entity_keywords, all_columns)

            # Build the insight
            source_tables = list(matched_tables)
            source_columns = [c for _, c in matched]

            # Build a suggested SQL query
            query = _build_composite_query(template, matched, profiles)

            impact = 60 if is_cross_table else 45
            if entity_match:
                impact += 10

            insights.append(Insight(
                id=str(uuid.uuid4()),
                insight_type="composite",
                severity="info",
                impact_score=min(impact, 100),
                title=f"{template.name} available",
                description=(
                    f"Columns {', '.join(source_columns)} "
                    f"{'across tables ' + ', '.join(source_tables) if is_cross_table else 'in ' + source_tables[0]} "
                    f"can create a {template.name.lower()}. {template.description}."
                ),
                source_tables=source_tables,
                source_columns=source_columns,
                evidence={
                    "query": query,
                    "formula": template.formula_hint,
                    "matched_columns": [{"table": t, "column": c} for t, c in matched],
                    "is_cross_table": is_cross_table,
                    "chart_spec": {
                        "type": "bar",
                        "x": source_columns[0],
                        "y": source_columns[1:] if len(source_columns) > 1 else source_columns,
                        "title": template.name,
                    },
                },
                composite_template=template.name,
                suggested_actions=[
                    f"Compute {template.name} using formula: {template.formula_hint}",
                    f"Create a dashboard widget for {template.name}",
                ],
            ))
        except Exception:
            logger.exception("Template matching failed for %s", template.name)

    return insights


def _check_entity(entity_keywords: list[str], all_columns: list[tuple[str, str, dict]]) -> bool:
    """Check if any column or table name matches entity keywords."""
    for table, col, _ in all_columns:
        combined = f"{table} {col}".lower()
        for kw in entity_keywords:
            if kw in combined:
                return True
    return False


def _build_composite_query(
    template: CompositeTemplate,
    matched: list[tuple[str, str]],
    profiles: list[TableProfile],
) -> str:
    """Build a SQL query to compute the composite metric."""
    if len(set(t for t, c in matched)) == 1:
        # Single table
        table = matched[0][0]
        cols = ", ".join(f'"{c}"' for _, c in matched)
        return f'SELECT {cols} FROM "{table}" LIMIT 100'

    # Cross-table: simple select from first table
    tables_cols: dict[str, list[str]] = {}
    for t, c in matched:
        tables_cols.setdefault(t, []).append(c)

    parts = []
    for t, cs in tables_cols.items():
        col_str = ", ".join(f'"{t}"."{c}"' for c in cs)
        parts.append(col_str)

    tables = list(tables_cols.keys())
    select_str = ", ".join(parts)
    from_str = f'"{tables[0]}"'

    return f"SELECT {select_str} FROM {from_str} LIMIT 100"
