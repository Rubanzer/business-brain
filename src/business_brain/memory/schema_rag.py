"""Retrieve relevant table schemas, business context, relationships, and thresholds
for a natural-language query.  This is the central RAG orchestrator — it collects ALL
context that downstream agents need to make informed decisions.
"""
from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import func, select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.ingestion.embeddings import embed_text
from business_brain.memory import metadata_store, vector_store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def retrieve_relevant_tables(
    session: AsyncSession,
    query: str,
    top_k: int = 8,
    allowed_tables: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Given a natural language query, return the top-k most relevant table schemas
    and matching business context snippets.

    Returns:
        Tuple of (ranked_tables, business_contexts) where business_contexts is
        a list of ``{"content": str, "source": str}`` dicts from the vector store,
        enriched with company profile, metric thresholds, and discovered relationships.

    Strategy:
      1. Embed the query and search business_contexts for semantic matches.
      2. Keyword-match against metadata_store descriptions.
      3. Boost tables that have discovered relationships with already-matched tables.
      4. Merge and return enriched schema info.
    """
    results: dict[str, dict] = {}
    context_snippets: list[dict] = []

    # 1. Semantic search against business_contexts
    try:
        query_embedding = embed_text(query)
        context_hits = await vector_store.search(session, query_embedding, top_k=top_k + 3)
        # Context hits give us hints about relevant tables via their content
        context_keywords = " ".join(hit.content for hit in context_hits).lower()
        context_snippets = [
            {"content": hit.content, "source": hit.source or "unknown"}
            for hit in context_hits
        ]
    except Exception:
        logger.exception("Vector search failed, falling back to keyword-only")
        await session.rollback()
        context_keywords = ""

    # 2. Keyword + context matching against metadata entries (filtered by focus scope)
    all_entries = await metadata_store.get_filtered(session, allowed_tables)
    query_lower = query.lower()

    # Build a keyword set for smarter matching (ignore stopwords)
    _STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "for", "of", "in", "on", "to", "and", "or", "not", "with",
        "from", "by", "at", "as", "this", "that", "it", "its", "my",
        "our", "what", "which", "how", "why", "when", "where", "who",
        "all", "each", "every", "any", "do", "does", "did", "can",
        "could", "should", "would", "will", "shall", "may", "might",
        "me", "we", "us", "you", "they", "them", "i", "he", "she",
        "show", "give", "tell", "get", "find", "list", "display",
    })
    query_words = [w for w in query_lower.split() if len(w) > 2 and w not in _STOPWORDS]

    for entry in all_entries:
        score = 0.0
        table_lower = entry.table_name.lower()
        desc_lower = (entry.description or "").lower()

        # Also build searchable text from column names and descriptions
        col_text = ""
        if entry.columns_metadata:
            col_names = [c.get("name", "").lower() for c in entry.columns_metadata]
            col_descs = [c.get("description", "").lower() for c in entry.columns_metadata]
            col_text = " ".join(col_names + col_descs)

        # Direct table name mention in query
        if table_lower in query_lower or table_lower.rstrip("s") in query_lower:
            score += 5.0

        # Keywords from query appear in description
        for word in query_words:
            if word in desc_lower:
                score += 1.0
            if word in table_lower:
                score += 2.0
            # Column name match — very important for finding relevant tables
            if word in col_text:
                score += 1.5

        # Table name mentioned in semantically similar contexts
        if table_lower in context_keywords:
            score += 2.5

        # Column names mentioned in contexts
        if entry.columns_metadata:
            for col in entry.columns_metadata:
                col_name = col.get("name", "").lower()
                if len(col_name) > 3 and col_name in context_keywords:
                    score += 1.0
                    break  # one match is enough for boost

        if score > 0:
            results[entry.table_name] = {
                "table_name": entry.table_name,
                "description": entry.description,
                "columns": entry.columns_metadata,
                "score": score,
            }

    # 3. Aggressively expand via discovered relationships
    #    For EVERY matched table, pull in ALL related tables — even if they
    #    scored 0 on keywords.  This is critical for multi-table JOIN queries
    #    (e.g. "which customer paid on time?" needs both sales + bank_statements).
    if results:
        try:
            relationships = await _get_relationships(session)
            # Iterate over a snapshot of current matches so we can mutate `results`
            seed_tables = set(results.keys())
            for rel in relationships:
                confidence = rel.get("confidence", 0.5)
                # table_a matched → unconditionally include table_b
                if rel["table_a"] in seed_tables and rel["table_b"] not in results:
                    entry = _find_entry(all_entries, rel["table_b"])
                    if entry:
                        results[rel["table_b"]] = {
                            "table_name": entry.table_name,
                            "description": entry.description,
                            "columns": entry.columns_metadata,
                            "score": 2.0 * confidence,
                        }
                # table_b matched → unconditionally include table_a
                if rel["table_b"] in seed_tables and rel["table_a"] not in results:
                    entry = _find_entry(all_entries, rel["table_a"])
                    if entry:
                        results[rel["table_a"]] = {
                            "table_name": entry.table_name,
                            "description": entry.description,
                            "columns": entry.columns_metadata,
                            "score": 2.0 * confidence,
                        }
                # Boost already-matched related tables
                if rel["table_a"] in results and rel["table_b"] in results:
                    results[rel["table_a"]]["score"] += 0.5
                    results[rel["table_b"]]["score"] += 0.5
        except Exception:
            logger.debug("Relationship expansion failed, continuing without it")

    # If no matches, return all entries as fallback
    if not results:
        for entry in all_entries[:top_k]:
            results[entry.table_name] = {
                "table_name": entry.table_name,
                "description": entry.description,
                "columns": entry.columns_metadata,
                "score": 0.0,
            }

    # Sort by score descending, return top_k
    ranked = sorted(results.values(), key=lambda r: r["score"], reverse=True)[:top_k]

    # Enrich tables with relationship info
    try:
        relationships = await _get_relationships(session)
        ranked_names = {r["table_name"] for r in ranked}
        for table_info in ranked:
            related = []
            for rel in relationships:
                if rel["table_a"] == table_info["table_name"] and rel["table_b"] in ranked_names:
                    related.append(f"{rel['table_a']}.{rel['column_a']} → {rel['table_b']}.{rel['column_b']}")
                elif rel["table_b"] == table_info["table_name"] and rel["table_a"] in ranked_names:
                    related.append(f"{rel['table_a']}.{rel['column_a']} → {rel['table_b']}.{rel['column_b']}")
            if related:
                table_info["relationships"] = related
    except Exception:
        logger.debug("Relationship enrichment failed")

    # Drop score from output
    tables = [{k: v for k, v in r.items() if k != "score"} for r in ranked]

    # 4. Enrich context with company profile and thresholds
    try:
        profile_context = await _get_company_context(session)
        if profile_context:
            context_snippets.insert(0, profile_context)
    except Exception:
        logger.debug("Company profile context retrieval failed")

    try:
        threshold_context = await _get_threshold_context(session)
        if threshold_context:
            context_snippets.insert(1, threshold_context)
    except Exception:
        logger.debug("Threshold context retrieval failed")

    # 5. Feed relevant discovery insights into context
    try:
        matched_table_names = [t["table_name"] for t in ranked]
        insight_context = await _get_relevant_insights(session, matched_table_names)
        if insight_context:
            context_snippets.append(insight_context)
    except Exception:
        logger.debug("Discovery insight retrieval failed")

    # 6. Add domain knowledge (benchmarks, red flags, terminology)
    try:
        domain_context = await _get_domain_context(session)
        if domain_context:
            context_snippets.append(domain_context)
    except Exception:
        logger.debug("Domain knowledge context retrieval failed")

    return tables, context_snippets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_entry(entries: list, table_name: str):
    """Find a metadata entry by table name."""
    for e in entries:
        if e.table_name == table_name:
            return e
    return None


async def _get_relationships(session: AsyncSession) -> list[dict]:
    """Fetch discovered relationships from the database."""
    try:
        from business_brain.db.discovery_models import DiscoveredRelationship
        result = await session.execute(
            select(DiscoveredRelationship).where(
                DiscoveredRelationship.confidence >= 0.4
            ).order_by(DiscoveredRelationship.confidence.desc())
            .limit(80)
        )
        rels = result.scalars().all()
        return [
            {
                "table_a": r.table_a,
                "column_a": r.column_a,
                "table_b": r.table_b,
                "column_b": r.column_b,
                "relationship_type": r.relationship_type,
                "confidence": r.confidence,
            }
            for r in rels
        ]
    except Exception:
        return []


async def _get_company_context(session: AsyncSession) -> Optional[dict]:
    """Fetch the company profile and format it as a context snippet."""
    try:
        from business_brain.db.v3_models import CompanyProfile
        result = await session.execute(select(CompanyProfile).limit(1))
        profile = result.scalar_one_or_none()
        if not profile or not profile.name:
            return None

        parts = [f"COMPANY PROFILE: {profile.name}"]
        if profile.industry:
            parts.append(f"Industry: {profile.industry}")
        if profile.products:
            products = profile.products if isinstance(profile.products, list) else [str(profile.products)]
            parts.append(f"Products: {', '.join(products)}")
        if profile.process_flow:
            parts.append(f"Process flow: {profile.process_flow}")
        if profile.departments:
            dept_names = []
            for d in profile.departments:
                if isinstance(d, dict):
                    dept_names.append(d.get("name", ""))
            if dept_names:
                parts.append(f"Departments: {', '.join(dept_names)}")
        if profile.systems:
            sys_names = []
            for s in profile.systems:
                if isinstance(s, dict):
                    sys_names.append(s.get("name", ""))
            if sys_names:
                parts.append(f"Systems: {', '.join(sys_names)}")

        return {
            "content": " | ".join(parts),
            "source": "company_profile",
        }
    except Exception:
        return None


async def _get_threshold_context(session: AsyncSession) -> Optional[dict]:
    """Fetch metric thresholds and format them as context for agents."""
    try:
        from business_brain.db.v3_models import MetricThreshold
        result = await session.execute(select(MetricThreshold).limit(30))
        thresholds = result.scalars().all()
        if not thresholds:
            return None

        parts = ["METRIC THRESHOLDS (use these to judge if values are normal/warning/critical):"]
        for t in thresholds:
            line = f"  {t.metric_name}"
            if t.unit:
                line += f" ({t.unit})"
            line += ":"
            ranges = []
            if t.normal_min is not None or t.normal_max is not None:
                ranges.append(f"normal={t.normal_min}-{t.normal_max}")
            if t.warning_min is not None or t.warning_max is not None:
                ranges.append(f"warning={t.warning_min}-{t.warning_max}")
            if t.critical_min is not None or t.critical_max is not None:
                ranges.append(f"critical={t.critical_min}-{t.critical_max}")
            if ranges:
                line += " " + ", ".join(ranges)
                parts.append(line)

        if len(parts) <= 1:
            return None

        return {
            "content": "\n".join(parts),
            "source": "metric_thresholds",
        }
    except Exception:
        return None


async def _get_relevant_insights(
    session: AsyncSession,
    table_names: list[str],
    limit: int = 5,
) -> Optional[dict]:
    """Fetch recent high-scoring insights relevant to the queried tables.

    This feeds discovery knowledge back into the query pipeline so agents
    know about anomalies, trends, and patterns already found.
    """
    try:
        from business_brain.db.discovery_models import Insight

        # Find insights where source_tables overlap with queried tables
        result = await session.execute(
            select(Insight)
            .where(
                Insight.status != "dismissed",
                func.coalesce(Insight.quality_score, 0) >= 30,
            )
            .order_by(Insight.impact_score.desc(), Insight.discovered_at.desc())
            .limit(limit * 3)  # fetch extra to filter by table overlap
        )
        all_insights = result.scalars().all()

        # Filter to insights relevant to the queried tables
        relevant = []
        for insight in all_insights:
            source_tables = insight.source_tables or []
            if any(t in table_names for t in source_tables):
                relevant.append(insight)
            if len(relevant) >= limit:
                break

        if not relevant:
            return None

        parts = ["PREVIOUSLY DISCOVERED INSIGHTS (from automated analysis):"]
        for ins in relevant:
            severity_icon = {"critical": "🔴", "warning": "🟡"}.get(ins.severity, "🔵")
            parts.append(f"  {severity_icon} [{ins.insight_type}] {ins.title}")
            if ins.description:
                # Truncate long descriptions
                desc = ins.description[:200] + "..." if len(ins.description) > 200 else ins.description
                parts.append(f"    {desc}")

        return {
            "content": "\n".join(parts),
            "source": "discovery_insights",
        }
    except Exception:
        return None


async def _get_domain_context(session: AsyncSession) -> Optional[dict]:
    """Get domain knowledge (benchmarks, red flags) based on company industry."""
    try:
        from business_brain.cognitive.domain_knowledge import (
            format_benchmarks_for_prompt,
            format_red_flags_for_prompt,
            get_domain_knowledge,
        )
        from business_brain.db.v3_models import CompanyProfile

        # Get the company's industry
        result = await session.execute(select(CompanyProfile).limit(1))
        profile = result.scalar_one_or_none()
        industry = profile.industry if profile else None

        knowledge = get_domain_knowledge(industry)
        if not knowledge:
            return None

        # Combine benchmarks and red flags into a single context snippet
        benchmarks_text = format_benchmarks_for_prompt(industry)
        red_flags_text = format_red_flags_for_prompt(industry)

        parts = []
        if benchmarks_text:
            parts.append(benchmarks_text)
        if red_flags_text:
            parts.append(red_flags_text)

        if not parts:
            return None

        return {
            "content": "\n\n".join(parts),
            "source": "domain_knowledge",
        }
    except Exception:
        return None
