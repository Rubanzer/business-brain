"""Tests for analysis/models.py — verify all 7 ORM models."""

from business_brain.analysis.models import (
    AgentOutput,
    AnalysisDelta,
    AnalysisFeedback,
    AnalysisHistoryEmbedding,
    AnalysisResult,
    AnalysisRun,
    LearningState,
    _uuid,
)
from business_brain.db.models import Base


class TestModels:
    def test_uuid_generates_unique(self):
        ids = {_uuid() for _ in range(100)}
        assert len(ids) == 100

    def test_uuid_format(self):
        uid = _uuid()
        assert len(uid) == 36
        assert uid.count("-") == 4

    def test_all_models_inherit_base(self):
        for model in (AnalysisRun, AnalysisResult, AgentOutput, AnalysisDelta,
                      AnalysisFeedback, LearningState, AnalysisHistoryEmbedding):
            assert issubclass(model, Base)

    def test_tablenames(self):
        assert AnalysisRun.__tablename__ == "analysis_runs"
        assert AnalysisResult.__tablename__ == "analysis_results"
        assert AgentOutput.__tablename__ == "agent_outputs"
        assert AnalysisDelta.__tablename__ == "analysis_deltas"
        assert AnalysisFeedback.__tablename__ == "analysis_feedback"
        assert LearningState.__tablename__ == "analysis_learning_state"
        assert AnalysisHistoryEmbedding.__tablename__ == "analysis_history_embeddings"

    def test_analysis_result_has_nary_columns(self):
        """Gap #1: N-ary column support via target/segmenters/controls."""
        cols = {c.key for c in AnalysisResult.__table__.columns}
        assert "target" in cols
        assert "segmenters" in cols
        assert "controls" in cols

    def test_analysis_result_has_data_hash(self):
        """Gap #4: Incremental caching via data_hash."""
        cols = {c.key for c in AnalysisResult.__table__.columns}
        assert "data_hash" in cols

    def test_analysis_result_has_join_spec(self):
        """Gap #2: Cross-table analysis."""
        cols = {c.key for c in AnalysisResult.__table__.columns}
        assert "join_spec" in cols

    def test_analysis_result_has_parent_result_id(self):
        """Gap #7: Finding composability."""
        cols = {c.key for c in AnalysisResult.__table__.columns}
        assert "parent_result_id" in cols

    def test_analysis_run_has_time_scope(self):
        """Gap #5: Time-scoped analysis."""
        cols = {c.key for c in AnalysisRun.__table__.columns}
        assert "time_scope" in cols

    def test_analysis_result_has_dedup_key(self):
        """Gap #6: Canonical deduplication."""
        cols = {c.key for c in AnalysisResult.__table__.columns}
        assert "dedup_key" in cols

    def test_learning_state_has_tier_budgets(self):
        cols = {c.key for c in LearningState.__table__.columns}
        assert "tier_budgets" in cols

    def test_embedding_dimension(self):
        """Verify 3072-dim embedding column."""
        col = AnalysisHistoryEmbedding.__table__.columns["embedding"]
        assert col.type.dim == 3072

    def test_base_metadata_includes_analysis_tables(self):
        """All analysis tables registered via Base.metadata."""
        table_names = set(Base.metadata.tables.keys())
        assert "analysis_runs" in table_names
        assert "analysis_results" in table_names
        assert "agent_outputs" in table_names
        assert "analysis_deltas" in table_names
        assert "analysis_feedback" in table_names
        assert "analysis_learning_state" in table_names
        assert "analysis_history_embeddings" in table_names
