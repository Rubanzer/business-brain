"""ORM models for the three-track analysis engine."""

import uuid

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON
from pgvector.sqlalchemy import Vector

from business_brain.db.models import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Analysis Run — top-level orchestration record
# ---------------------------------------------------------------------------


class AnalysisRun(Base):
    """A single analysis execution (explore / diagnose / monitor)."""

    __tablename__ = "analysis_runs"

    id = Column(String(36), primary_key=True, default=_uuid)
    situation_type = Column(String(20), nullable=False)  # exploratory/diagnostic/monitoring
    trigger = Column(String(100), nullable=False, default="manual")  # manual/post_sync/scheduled
    status = Column(String(20), nullable=False, default="running")  # running/completed/failed
    config = Column(JSON, nullable=True)  # {table_names, budget overrides, ...}
    time_scope = Column(JSON, nullable=True)  # {column, window, compare_to}
    summary = Column(JSON, nullable=True)  # post-run summary stats
    started_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Analysis Result — one finding per row (N-ary, cross-table, composable)
# ---------------------------------------------------------------------------


class AnalysisResult(Base):
    """A single analysis finding from Track 1.

    Supports N-ary columns via target/segmenters/controls (Gap #1),
    cross-table analysis via join_spec (Gap #2),
    incremental caching via data_hash (Gap #4),
    and finding composability via parent_result_id (Gap #7).
    """

    __tablename__ = "analysis_results"

    id = Column(String(36), primary_key=True, default=_uuid)
    run_id = Column(String(36), nullable=False, index=True)
    operation_type = Column(String(30), nullable=False)  # DESCRIBE/COMPARE/CORRELATE/RANK/DETECT_ANOMALY/FORECAST/ATTRIBUTE
    table_name = Column(String(255), nullable=False, index=True)
    data_hash = Column(String(64), nullable=True)  # incremental caching key (Gap #4)
    tier = Column(Integer, nullable=False, default=0)  # 0-4 enumeration tier

    # N-ary column structure (Gap #1)
    target = Column(JSON, nullable=False)  # ["measure1", ...] — columns being analyzed
    segmenters = Column(JSON, nullable=True)  # ["dim1", ...] — GROUP BY dimensions
    controls = Column(JSON, nullable=True)  # ["dim1", ...] — WHERE conditions

    # Cross-table support (Gap #2)
    join_spec = Column(JSON, nullable=True)  # {table, local_col, remote_col, join_type}

    # Canonical dedup key (Gap #6) — for incremental cache lookup
    dedup_key = Column(String(64), nullable=True, index=True)

    # Result payload
    result_data = Column(JSON, nullable=True)  # operation-specific result

    # Track 1: algorithmic scores
    interestingness_score = Column(Float, default=0.0)
    interestingness_breakdown = Column(JSON, nullable=True)  # {surprise, magnitude, variance, stability, coverage}

    # Track 2: agent enrichment
    quality_verdict = Column(String(20), nullable=True)  # RELIABLE/CAUTIONARY/UNRELIABLE (VETO)
    domain_relevance = Column(Float, nullable=True)  # 0.0-1.0 from domain agent
    temporal_context = Column(JSON, nullable=True)  # {seasonality_adjusted, trend_status, novelty}

    # Track 3: delta classification
    delta_type = Column(String(30), nullable=True)  # FILTERED_BY_CONTEXT/EXPECTED_BUT_ABSENT/MAGNITUDE_DISAGREEMENT/UNEXPLAINED_SIGNAL/SEGMENT_REVERSAL

    # Final composite score
    final_score = Column(Float, default=0.0)

    # Composability (Gap #7) — follow-up findings link to their parent
    parent_result_id = Column(String(36), nullable=True, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Agent Output — per-agent evaluation of a finding
# ---------------------------------------------------------------------------


class AgentOutput(Base):
    """Output from a single agent evaluating a finding."""

    __tablename__ = "agent_outputs"

    id = Column(String(36), primary_key=True, default=_uuid)
    run_id = Column(String(36), nullable=False, index=True)
    result_id = Column(String(36), nullable=False, index=True)
    agent_id = Column(String(30), nullable=False)  # quality/domain/temporal
    output = Column(JSON, nullable=True)  # agent-specific structured output
    confidence = Column(Float, default=0.0)
    duration_ms = Column(Integer, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Analysis Delta — Track 3 disagreements between algorithmic and contextual
# ---------------------------------------------------------------------------


class AnalysisDelta(Base):
    """A delta between algorithmic (Track 1) and contextual (Track 2) views."""

    __tablename__ = "analysis_deltas"

    id = Column(String(36), primary_key=True, default=_uuid)
    run_id = Column(String(36), nullable=False, index=True)
    result_id = Column(String(36), nullable=False, index=True)
    delta_type = Column(String(30), nullable=False)  # FILTERED_BY_CONTEXT/EXPECTED_BUT_ABSENT/MAGNITUDE_DISAGREEMENT/UNEXPLAINED_SIGNAL/SEGMENT_REVERSAL
    description = Column(Text, nullable=True)
    algorithmic_view = Column(JSON, nullable=True)  # what Track 1 says
    contextual_view = Column(JSON, nullable=True)  # what Track 2 says
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Analysis Feedback — user signals on finding quality
# ---------------------------------------------------------------------------


class AnalysisFeedback(Base):
    """User feedback on an analysis finding."""

    __tablename__ = "analysis_feedback"

    id = Column(String(36), primary_key=True, default=_uuid)
    result_id = Column(String(36), nullable=False, index=True)
    feedback_type = Column(String(20), nullable=False)  # useful/not_useful/wrong/expected
    comment = Column(Text, nullable=True)
    session_id = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Learning State — tunable weights for the reinforcement loop
# ---------------------------------------------------------------------------


class LearningState(Base):
    """Versioned learning parameters adjusted by feedback."""

    __tablename__ = "analysis_learning_state"

    id = Column(String(36), primary_key=True, default=_uuid)
    version = Column(Integer, nullable=False, default=1)
    interestingness_weights = Column(JSON, nullable=True)  # {surprise: 0.3, magnitude: 0.25, ...}
    agent_calibration = Column(JSON, nullable=True)  # per-agent confidence adjustments
    operation_preferences = Column(JSON, nullable=True)  # per-operation boost/penalty
    tier_budgets = Column(JSON, nullable=True)  # learned tier budget allocations
    feedback_count = Column(Integer, default=0)
    computed_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Analysis History Embeddings — RAG store for past analysis findings
# ---------------------------------------------------------------------------


class AnalysisHistoryEmbedding(Base):
    """Vector embeddings of past analysis findings for RAG retrieval."""

    __tablename__ = "analysis_history_embeddings"

    id = Column(String(36), primary_key=True, default=_uuid)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=True)  # Gemini embedding-001 dimension
    source = Column(String(255), nullable=True)  # result_id or run_id reference
    metadata_ = Column("metadata", JSON, nullable=True)  # operation_type, table, score, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())
