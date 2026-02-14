"""Structured onboarding — company profile collection and context generation."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.v3_models import CompanyProfile, MetricThreshold

logger = logging.getLogger(__name__)


async def get_company_profile(session: AsyncSession) -> CompanyProfile | None:
    """Get the company profile (there should be at most one)."""
    result = await session.execute(select(CompanyProfile).limit(1))
    return result.scalar_one_or_none()


async def save_company_profile(
    session: AsyncSession,
    data: dict[str, Any],
) -> CompanyProfile:
    """Create or update the company profile from onboarding data.

    Also generates natural language context for RAG.
    """
    existing = await get_company_profile(session)

    if existing:
        if "name" in data:
            existing.name = data["name"]
        if "industry" in data:
            existing.industry = data["industry"]
        if "products" in data:
            existing.products = data["products"]
        if "departments" in data:
            existing.departments = data["departments"]
        if "process_flow" in data:
            existing.process_flow = data["process_flow"]
        if "systems" in data:
            existing.systems = data["systems"]
        if "known_relationships" in data:
            existing.known_relationships = data["known_relationships"]
        profile = existing
    else:
        profile = CompanyProfile(
            name=data.get("name", ""),
            industry=data.get("industry"),
            products=data.get("products"),
            departments=data.get("departments"),
            process_flow=data.get("process_flow"),
            systems=data.get("systems"),
            known_relationships=data.get("known_relationships"),
        )
        session.add(profile)

    await session.commit()
    await session.refresh(profile)

    # Generate natural language context and store it for RAG
    try:
        context_text = _generate_context_text(profile)
        if context_text:
            from business_brain.ingestion.context_ingestor import ingest_context
            await ingest_context(context_text, session, source="onboarding:company_profile")
            logger.info("Generated company context (%d chars)", len(context_text))
    except Exception:
        logger.exception("Failed to generate context from company profile")

    return profile


def _generate_context_text(profile: CompanyProfile) -> str:
    """Convert structured company profile into natural language for RAG."""
    parts = []

    if profile.name:
        parts.append(f"Company: {profile.name}")
    if profile.industry:
        parts.append(f"Industry: {profile.industry}")
    if profile.products:
        products = profile.products if isinstance(profile.products, list) else [str(profile.products)]
        parts.append(f"Products manufactured: {', '.join(products)}")

    if profile.departments:
        dept_lines = []
        for dept in profile.departments:
            if isinstance(dept, dict):
                name = dept.get("name", "Unknown")
                head = dept.get("head", "")
                line = f"- {name}" + (f" (Head: {head})" if head else "")
                dept_lines.append(line)
        if dept_lines:
            parts.append("Departments:\n" + "\n".join(dept_lines))

    if profile.process_flow:
        parts.append(f"Production process: {profile.process_flow}")

    if profile.systems:
        sys_lines = []
        for sys in profile.systems:
            if isinstance(sys, dict):
                name = sys.get("name", "Unknown")
                desc = sys.get("description", "")
                sys_lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        if sys_lines:
            parts.append("Data systems:\n" + "\n".join(sys_lines))

    if profile.known_relationships:
        parts.append(f"Known data relationships: {profile.known_relationships}")

    return "\n\n".join(parts)


def compute_profile_completeness(profile: CompanyProfile) -> int:
    """Compute profile completeness percentage (0-100)."""
    fields = [
        profile.name,
        profile.industry,
        profile.products,
        profile.departments,
        profile.process_flow,
        profile.systems,
    ]
    filled = sum(1 for f in fields if f)
    return int(filled / len(fields) * 100)


# ---------------------------------------------------------------------------
# Metric Thresholds
# ---------------------------------------------------------------------------


async def get_all_thresholds(session: AsyncSession) -> list[MetricThreshold]:
    """Get all metric thresholds."""
    result = await session.execute(select(MetricThreshold).order_by(MetricThreshold.created_at.desc()))
    return list(result.scalars().all())


async def create_threshold(
    session: AsyncSession,
    data: dict[str, Any],
) -> MetricThreshold:
    """Create a new metric threshold."""
    profile = await get_company_profile(session)

    threshold = MetricThreshold(
        company_id=profile.id if profile else None,
        metric_name=data["metric_name"],
        table_name=data.get("table_name"),
        column_name=data.get("column_name"),
        unit=data.get("unit"),
        normal_min=data.get("normal_min"),
        normal_max=data.get("normal_max"),
        warning_min=data.get("warning_min"),
        warning_max=data.get("warning_max"),
        critical_min=data.get("critical_min"),
        critical_max=data.get("critical_max"),
    )
    session.add(threshold)
    await session.commit()
    await session.refresh(threshold)
    return threshold


async def update_threshold(
    session: AsyncSession,
    threshold_id: int,
    data: dict[str, Any],
) -> MetricThreshold | None:
    """Update a metric threshold."""
    result = await session.execute(
        select(MetricThreshold).where(MetricThreshold.id == threshold_id)
    )
    threshold = result.scalar_one_or_none()
    if not threshold:
        return None

    for field in ("metric_name", "table_name", "column_name", "unit",
                  "normal_min", "normal_max", "warning_min", "warning_max",
                  "critical_min", "critical_max"):
        if field in data:
            setattr(threshold, field, data[field])

    await session.commit()
    return threshold


async def delete_threshold(session: AsyncSession, threshold_id: int) -> bool:
    """Delete a metric threshold."""
    result = await session.execute(
        select(MetricThreshold).where(MetricThreshold.id == threshold_id)
    )
    threshold = result.scalar_one_or_none()
    if not threshold:
        return False
    await session.delete(threshold)
    await session.commit()
    return True
