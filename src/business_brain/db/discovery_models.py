"""ORM models for the proactive discovery engine."""

import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON

from business_brain.db.models import Base


def _uuid() -> str:
    return str(uuid.uuid4())


class DiscoveryRun(Base):
    """Tracks each discovery sweep."""

    __tablename__ = "discovery_runs"

    id = Column(String(36), primary_key=True, default=_uuid)
    status = Column(String(20), nullable=False, default="running")  # running/completed/failed
    trigger = Column(String(100), nullable=False, default="manual")  # upload:<table>/manual
    tables_scanned = Column(Integer, default=0)
    insights_found = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)


class TableProfile(Base):
    """Cached column classification per table."""

    __tablename__ = "table_profiles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(255), nullable=False, unique=True)
    row_count = Column(Integer, default=0)
    column_classification = Column(JSON, nullable=True)
    domain_hint = Column(String(50), nullable=True)
    profiled_at = Column(DateTime(timezone=True), server_default=func.now())
    data_hash = Column(String(64), nullable=True)  # detect data changes


class DiscoveredRelationship(Base):
    """Cross-table join keys and correlations."""

    __tablename__ = "discovered_relationships"

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_a = Column(String(255), nullable=False)
    column_a = Column(String(255), nullable=False)
    table_b = Column(String(255), nullable=False)
    column_b = Column(String(255), nullable=False)
    relationship_type = Column(String(50), nullable=False)  # join_key/value_overlap/semantic_match
    confidence = Column(Float, default=0.0)
    overlap_count = Column(Integer, default=0)
    discovered_at = Column(DateTime(timezone=True), server_default=func.now())


class Insight(Base):
    """All discovered insights (anomalies, composites, stories, etc.)."""

    __tablename__ = "insights"

    id = Column(String(36), primary_key=True, default=_uuid)
    insight_type = Column(String(30), nullable=False)  # anomaly/trend/correlation/composite/cross_event/story
    severity = Column(String(20), nullable=False, default="info")  # critical/warning/info
    impact_score = Column(Integer, default=0)  # 0-100
    title = Column(String(500), nullable=False)
    description = Column(Text, nullable=False)
    narrative = Column(Text, nullable=True)
    source_tables = Column(JSON, nullable=True)  # ["table1", "table2"]
    source_columns = Column(JSON, nullable=True)  # ["col1", "col2"]
    evidence = Column(JSON, nullable=True)  # {query, sample_rows, chart_spec}
    related_insights = Column(JSON, nullable=True)  # [insight_id, ...]
    suggested_actions = Column(JSON, nullable=True)  # ["action1", ...]
    composite_template = Column(String(100), nullable=True)
    discovered_at = Column(DateTime(timezone=True), server_default=func.now())
    discovery_run_id = Column(String(36), nullable=True)
    status = Column(String(20), nullable=False, default="new")  # new/seen/deployed/dismissed
    session_id = Column(String(64), nullable=True)


class DeployedReport(Base):
    """Persistent reports created from insights."""

    __tablename__ = "deployed_reports"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(500), nullable=False)
    insight_id = Column(String(36), nullable=False)
    query = Column(Text, nullable=True)
    chart_spec = Column(JSON, nullable=True)
    parameters = Column(JSON, nullable=True)
    last_result = Column(JSON, nullable=True)
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    session_id = Column(String(64), nullable=True)
    active = Column(Boolean, default=True)
