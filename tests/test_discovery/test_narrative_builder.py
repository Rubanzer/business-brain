"""Tests for narrative_builder helper functions."""

import pytest

from business_brain.db.discovery_models import Insight
from business_brain.discovery.narrative_builder import _group_by_tables, build_narratives


def _make_insight(insight_id, tables, insight_type="anomaly"):
    """Create a minimal Insight-like object for testing."""
    ins = Insight()
    ins.id = insight_id
    ins.insight_type = insight_type
    ins.severity = "info"
    ins.impact_score = 50
    ins.title = f"Insight {insight_id}"
    ins.description = f"Description for {insight_id}"
    ins.source_tables = tables
    ins.source_columns = []
    ins.evidence = {}
    ins.suggested_actions = []
    return ins


# ---------------------------------------------------------------------------
# _group_by_tables
# ---------------------------------------------------------------------------


class TestGroupByTables:
    def test_single_insight_one_group(self):
        ins = _make_insight("1", ["sales"])
        groups = _group_by_tables([ins])
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_shared_table_same_group(self):
        a = _make_insight("1", ["sales"])
        b = _make_insight("2", ["sales", "products"])
        groups = _group_by_tables([a, b])
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_no_shared_tables_separate_groups(self):
        a = _make_insight("1", ["sales"])
        b = _make_insight("2", ["hr"])
        groups = _group_by_tables([a, b])
        assert len(groups) == 2

    def test_transitive_grouping(self):
        """A shares table with B, B shares table with C → all in one group."""
        a = _make_insight("1", ["sales"])
        b = _make_insight("2", ["sales", "products"])
        c = _make_insight("3", ["products", "inventory"])
        groups = _group_by_tables([a, b, c])
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_empty_source_tables(self):
        a = _make_insight("1", [])
        b = _make_insight("2", [])
        groups = _group_by_tables([a, b])
        # Empty tables don't overlap, so separate groups
        assert len(groups) == 2

    def test_none_source_tables(self):
        a = _make_insight("1", None)
        groups = _group_by_tables([a])
        assert len(groups) == 1

    def test_multiple_separate_groups(self):
        a = _make_insight("1", ["sales"])
        b = _make_insight("2", ["sales"])
        c = _make_insight("3", ["hr"])
        d = _make_insight("4", ["hr"])
        groups = _group_by_tables([a, b, c, d])
        assert len(groups) == 2
        group_sizes = sorted(len(g) for g in groups)
        assert group_sizes == [2, 2]

    def test_no_duplicate_insights_in_groups(self):
        a = _make_insight("1", ["sales", "products"])
        b = _make_insight("2", ["products", "inventory"])
        c = _make_insight("3", ["sales", "inventory"])
        groups = _group_by_tables([a, b, c])
        all_ids = []
        for g in groups:
            for ins in g:
                all_ids.append(ins.id)
        assert len(all_ids) == len(set(all_ids))


# ---------------------------------------------------------------------------
# build_narratives filtering logic
# ---------------------------------------------------------------------------


class TestBuildNarrativesFiltering:
    @pytest.mark.asyncio
    async def test_fewer_than_2_returns_empty(self):
        """A single non-story insight should not trigger narratives."""
        a = _make_insight("1", ["sales"])
        result = await build_narratives([a])
        assert result == []

    @pytest.mark.asyncio
    async def test_stories_excluded_from_count(self):
        """If all 3 insights are stories, the non-story count is 0."""
        a = _make_insight("1", ["sales"], insight_type="story")
        b = _make_insight("2", ["sales"], insight_type="story")
        c = _make_insight("3", ["sales"], insight_type="story")
        result = await build_narratives([a, b, c])
        assert result == []

    @pytest.mark.asyncio
    async def test_mixed_with_1_non_story_returns_empty(self):
        """Only 1 non-story insight (even with stories) should not trigger."""
        a = _make_insight("1", ["sales"], insight_type="anomaly")
        b = _make_insight("2", ["sales"], insight_type="story")
        c = _make_insight("3", ["sales"], insight_type="story")
        result = await build_narratives([a, b, c])
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_list_returns_empty(self):
        result = await build_narratives([])
        assert result == []
