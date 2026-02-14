"""Data lineage tracker — maps upstream/downstream dependencies.

Tracks which insights came from which tables, which reports depend on
which insights, and provides dependency graph traversal.
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.discovery_models import (
    DeployedReport,
    DiscoveredRelationship,
    Insight,
    TableProfile,
)

logger = logging.getLogger(__name__)


def build_lineage_graph(
    profiles: list,
    relationships: list,
    insights: list,
    reports: list,
) -> dict[str, Any]:
    """Build a complete lineage graph from discovery artifacts.

    Pure function — takes lists of objects and returns a dependency dict.

    Returns:
        {
            "tables": { "table_name": {"insights": [...], "reports": [...], "related_tables": [...]} },
            "insights": { "insight_id": {"source_tables": [...], "reports": [...]} },
            "reports": { "report_id": {"insight_id": ..., "source_tables": [...]} },
            "edges": [{"from": ..., "to": ..., "type": ...}, ...]
        }
    """
    graph: dict[str, Any] = {
        "tables": {},
        "insights": {},
        "reports": {},
        "edges": [],
    }

    # Index profiles
    for profile in profiles:
        table_name = _get_attr(profile, "table_name", "")
        if not table_name:
            continue
        graph["tables"][table_name] = {
            "insights": [],
            "reports": [],
            "related_tables": [],
            "row_count": _get_attr(profile, "row_count", 0),
            "domain": _get_attr(profile, "domain_hint", "general"),
        }

    # Index relationships
    for rel in relationships:
        table_a = _get_attr(rel, "table_a", "")
        table_b = _get_attr(rel, "table_b", "")
        if not table_a or not table_b:
            continue

        if table_a in graph["tables"]:
            if table_b not in graph["tables"][table_a]["related_tables"]:
                graph["tables"][table_a]["related_tables"].append(table_b)
        if table_b in graph["tables"]:
            if table_a not in graph["tables"][table_b]["related_tables"]:
                graph["tables"][table_b]["related_tables"].append(table_a)

        graph["edges"].append({
            "from": table_a,
            "to": table_b,
            "type": "relationship",
            "confidence": _get_attr(rel, "confidence", 0),
        })

    # Index insights
    for insight in insights:
        insight_id = _get_attr(insight, "id", "")
        if not insight_id:
            continue

        source_tables = _get_attr(insight, "source_tables", []) or []
        graph["insights"][insight_id] = {
            "source_tables": list(source_tables),
            "reports": [],
            "type": _get_attr(insight, "insight_type", ""),
            "title": _get_attr(insight, "title", ""),
        }

        # Link tables → insights
        for table_name in source_tables:
            if table_name in graph["tables"]:
                if insight_id not in graph["tables"][table_name]["insights"]:
                    graph["tables"][table_name]["insights"].append(insight_id)
            graph["edges"].append({
                "from": table_name,
                "to": insight_id,
                "type": "source",
            })

    # Index reports
    for report in reports:
        report_id = _get_attr(report, "id", "")
        insight_id = _get_attr(report, "insight_id", "")
        if not report_id:
            continue

        source_tables = []
        if insight_id and insight_id in graph["insights"]:
            source_tables = graph["insights"][insight_id]["source_tables"]
            graph["insights"][insight_id]["reports"].append(report_id)

        graph["reports"][report_id] = {
            "insight_id": insight_id,
            "source_tables": source_tables,
            "name": _get_attr(report, "name", ""),
        }

        # Link insight → report
        if insight_id:
            graph["edges"].append({
                "from": insight_id,
                "to": report_id,
                "type": "deployed",
            })

        # Link tables → reports
        for table_name in source_tables:
            if table_name in graph["tables"]:
                if report_id not in graph["tables"][table_name]["reports"]:
                    graph["tables"][table_name]["reports"].append(report_id)

    return graph


def get_table_lineage(graph: dict, table_name: str) -> dict[str, Any]:
    """Get the lineage for a specific table — what depends on it.

    Pure function.

    Returns:
        {
            "table": table_name,
            "upstream": [related tables feeding into this one],
            "downstream_insights": [insight IDs derived from this table],
            "downstream_reports": [report IDs that depend on this table],
            "impact": number of downstream dependencies
        }
    """
    tables = graph.get("tables", {})
    table_info = tables.get(table_name)

    if not table_info:
        return {
            "table": table_name,
            "upstream": [],
            "downstream_insights": [],
            "downstream_reports": [],
            "impact": 0,
        }

    return {
        "table": table_name,
        "upstream": table_info.get("related_tables", []),
        "downstream_insights": table_info.get("insights", []),
        "downstream_reports": table_info.get("reports", []),
        "impact": len(table_info.get("insights", [])) + len(table_info.get("reports", [])),
    }


def get_impact_ranking(graph: dict) -> list[dict]:
    """Rank tables by their downstream impact (most dependencies first).

    Pure function.

    Returns:
        List of dicts sorted by impact descending.
    """
    tables = graph.get("tables", {})
    rankings = []

    for table_name, info in tables.items():
        impact = len(info.get("insights", [])) + len(info.get("reports", []))
        rankings.append({
            "table": table_name,
            "insight_count": len(info.get("insights", [])),
            "report_count": len(info.get("reports", [])),
            "relationship_count": len(info.get("related_tables", [])),
            "impact": impact,
        })

    return sorted(rankings, key=lambda r: r["impact"], reverse=True)


def find_orphaned_tables(graph: dict) -> list[str]:
    """Find tables with no insights, no reports, and no relationships.

    Pure function. These tables may be unused or need attention.
    """
    tables = graph.get("tables", {})
    orphans = []

    for table_name, info in tables.items():
        if (
            not info.get("insights")
            and not info.get("reports")
            and not info.get("related_tables")
        ):
            orphans.append(table_name)

    return sorted(orphans)


async def get_lineage_for_table(session: AsyncSession, table_name: str) -> dict:
    """Load all discovery artifacts and return lineage for a specific table."""
    profiles = list((await session.execute(select(TableProfile))).scalars().all())
    relationships = list((await session.execute(select(DiscoveredRelationship))).scalars().all())
    insights = list((await session.execute(select(Insight))).scalars().all())
    reports = list((await session.execute(select(DeployedReport))).scalars().all())

    graph = build_lineage_graph(profiles, relationships, insights, reports)
    return get_table_lineage(graph, table_name)


def _get_attr(obj: Any, attr: str, default: Any = None) -> Any:
    """Safe attribute getter that works with both objects and dicts."""
    if isinstance(obj, dict):
        return obj.get(attr, default)
    return getattr(obj, attr, default)
