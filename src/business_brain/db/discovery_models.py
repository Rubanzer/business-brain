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
    impact_score = Column(Integer, default=0)  # 0-100 (business value score from quality gate)
    quality_score = Column(Integer, default=0)  # 0-100 (mirrors impact_score after quality gate)
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


# ---------------------------------------------------------------------------
# Deep Tier — Analysis Task Queue
# ---------------------------------------------------------------------------


class AnalysisTask(Base):
    """Task queue for Deep Tier analysis (Claude API).

    Created automatically when Fast Tier confidence < threshold,
    or manually via the 'Investigate Deeper' button.
    """

    __tablename__ = "analysis_tasks"

    id = Column(String(36), primary_key=True, default=_uuid)
    question = Column(Text, nullable=False)
    source_tier = Column(String(10), nullable=False, default="fast")  # fast / manual
    status = Column(String(20), nullable=False, default="pending")
    # pending / running / completed / failed
    priority = Column(Integer, default=0)  # higher = more important
    # Input context from Fast Tier
    fast_tier_result = Column(JSON, nullable=True)  # summary of fast tier analysis
    sql_query = Column(Text, nullable=True)  # SQL used by fast tier
    sql_data = Column(JSON, nullable=True)  # rows from fast tier (anonymized)
    tables_used = Column(JSON, nullable=True)  # ["table1", "table2"]
    fast_confidence = Column(Float, nullable=True)  # confidence from router
    session_id = Column(String(64), nullable=True)
    # Deep Tier output
    result = Column(JSON, nullable=True)  # full Claude analysis
    error = Column(Text, nullable=True)
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    # Audit
    requested_by = Column(String(255), nullable=True)  # user_id or "auto"


# ---------------------------------------------------------------------------
# Pre-computed Analysis Results (Background Intelligence Engine)
# ---------------------------------------------------------------------------


class PrecomputedAnalysis(Base):
    """Background-computed analysis results backing recommendations.

    After discovery profiles tables, the pre-compute engine scores ALL column
    combinations, picks the top-N most promising ones, and runs actual SQL
    queries (GROUP BY, CORR, z-score, regression) to get real results.

    Recommendations matched to a completed PrecomputedAnalysis get
    confidence="pre-computed" and include a result preview.
    """

    __tablename__ = "precomputed_analyses"

    id = Column(String(36), primary_key=True, default=_uuid)
    table_name = Column(String(255), nullable=False, index=True)
    analysis_type = Column(String(30), nullable=False)  # benchmark/correlation/anomaly/trend
    columns = Column(JSON, nullable=False)  # ["supplier", "rate"]
    column_scores = Column(JSON, nullable=True)  # {"supplier": 0.82, "rate": 0.91}

    status = Column(String(20), default="pending")  # pending/running/completed/failed/stale
    result_summary = Column(JSON, nullable=True)  # compact result (type-specific shape)
    result_detail = Column(JSON, nullable=True)  # full result (chart_spec, sample rows)
    quality_score = Column(Float, default=0.0)  # 0.0-1.0 how interesting the result was
    error = Column(Text, nullable=True)

    computed_at = Column(DateTime(timezone=True), nullable=True)
    discovery_run_id = Column(String(36), nullable=True)
    data_hash = Column(String(64), nullable=True)  # from TableProfile — stale detection


# ---------------------------------------------------------------------------
# Engagement Tracking (Background Intelligence Engine — Phase 2)
# ---------------------------------------------------------------------------


class EngagementEvent(Base):
    """Tracks user interactions with insights and recommendations.

    Every time a user views the feed, deploys an insight, dismisses insights,
    or views recommendations, an event is recorded here. These events feed
    the Phase 3 reinforcement loop for scoring weight adjustment.
    """

    __tablename__ = "engagement_events"

    id = Column(String(36), primary_key=True, default=_uuid)
    event_type = Column(String(30), nullable=False)
    # Events: insight_shown, insight_seen, insight_deployed,
    #         insight_dismissed, insights_dismissed_all,
    #         recommendation_shown
    entity_type = Column(String(20), nullable=False)  # "insight" | "recommendation"
    entity_id = Column(String(36), nullable=True)  # insight_id or rec hash
    analysis_type = Column(String(30), nullable=True)  # benchmark/correlation/anomaly/etc
    table_name = Column(String(255), nullable=True)  # primary table involved
    columns = Column(JSON, nullable=True)  # columns involved
    severity = Column(String(20), nullable=True)  # insight severity if applicable
    impact_score = Column(Integer, nullable=True)  # insight score if applicable
    extra_metadata = Column(JSON, nullable=True)  # extra context (count, report_name, etc)
    session_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Reinforcement Weights (Background Intelligence Engine — Phase 3)
# ---------------------------------------------------------------------------


class ReinforcementWeights(Base):
    """Versioned weight adjustments from the engagement reinforcement loop.

    Each row is a full snapshot of all multipliers computed from engagement
    data. Only the most recent row (highest version) is active. Previous
    rows are kept for audit/history.

    All multipliers default to 1.0 (no change from hardcoded base values).
    Multipliers are clamped to [0.8, 1.2] to prevent wild swings.
    """

    __tablename__ = "reinforcement_weights"

    id = Column(String(36), primary_key=True, default=_uuid)
    version = Column(Integer, nullable=False, default=1)

    # Multipliers applied to recommendation base priorities
    analysis_type_multipliers = Column(JSON, nullable=False, default=dict)
    # Multipliers applied to quality gate severity weights
    severity_multipliers = Column(JSON, nullable=False, default=dict)
    # Multipliers applied to quality gate novelty scores
    insight_type_multipliers = Column(JSON, nullable=False, default=dict)

    # Metadata
    engagement_summary = Column(JSON, nullable=True)  # input data snapshot
    computed_at = Column(DateTime(timezone=True), server_default=func.now())
    discovery_run_id = Column(String(36), nullable=True)
    period_days = Column(Integer, default=30)
    total_events = Column(Integer, default=0)


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
    refresh_frequency = Column(String(20), default="manual")  # manual/hourly/daily/weekly
