"""Tests for lineage tracker pure functions."""

from business_brain.discovery.lineage_tracker import (
    build_lineage_graph,
    find_orphaned_tables,
    get_impact_ranking,
    get_table_lineage,
)


class _Prof:
    def __init__(self, table_name, row_count=100, domain_hint="general"):
        self.table_name = table_name
        self.row_count = row_count
        self.domain_hint = domain_hint


class _Rel:
    def __init__(self, table_a, table_b, confidence=0.9):
        self.table_a = table_a
        self.table_b = table_b
        self.confidence = confidence


class _Insight:
    def __init__(self, id, source_tables, insight_type="anomaly", title="Test"):
        self.id = id
        self.source_tables = source_tables
        self.insight_type = insight_type
        self.title = title


class _Report:
    def __init__(self, id, insight_id, name="Report"):
        self.id = id
        self.insight_id = insight_id
        self.name = name


class TestBuildLineageGraph:
    def test_empty_inputs(self):
        graph = build_lineage_graph([], [], [], [])
        assert graph["tables"] == {}
        assert graph["insights"] == {}
        assert graph["reports"] == {}
        assert graph["edges"] == []

    def test_profiles_indexed(self):
        profiles = [_Prof("orders"), _Prof("products")]
        graph = build_lineage_graph(profiles, [], [], [])
        assert "orders" in graph["tables"]
        assert "products" in graph["tables"]
        assert graph["tables"]["orders"]["row_count"] == 100

    def test_relationships_create_edges(self):
        profiles = [_Prof("orders"), _Prof("products")]
        rels = [_Rel("orders", "products", 0.85)]
        graph = build_lineage_graph(profiles, rels, [], [])
        assert "products" in graph["tables"]["orders"]["related_tables"]
        assert "orders" in graph["tables"]["products"]["related_tables"]
        assert len(graph["edges"]) == 1
        assert graph["edges"][0]["type"] == "relationship"

    def test_insights_linked_to_tables(self):
        profiles = [_Prof("orders"), _Prof("products")]
        insights = [_Insight("i1", ["orders", "products"])]
        graph = build_lineage_graph(profiles, [], insights, [])
        assert "i1" in graph["tables"]["orders"]["insights"]
        assert "i1" in graph["tables"]["products"]["insights"]
        assert graph["insights"]["i1"]["source_tables"] == ["orders", "products"]

    def test_reports_linked_to_insights(self):
        profiles = [_Prof("orders")]
        insights = [_Insight("i1", ["orders"])]
        reports = [_Report("r1", "i1", "Sales Report")]
        graph = build_lineage_graph(profiles, [], insights, reports)
        assert "r1" in graph["insights"]["i1"]["reports"]
        assert graph["reports"]["r1"]["insight_id"] == "i1"
        assert graph["reports"]["r1"]["source_tables"] == ["orders"]
        assert "r1" in graph["tables"]["orders"]["reports"]

    def test_full_chain(self):
        """Table → Relationship → Insight → Report."""
        profiles = [_Prof("a"), _Prof("b")]
        rels = [_Rel("a", "b")]
        insights = [_Insight("i1", ["a", "b"])]
        reports = [_Report("r1", "i1")]
        graph = build_lineage_graph(profiles, rels, insights, reports)

        # Edges: 1 relationship + 2 source (i1→a, i1→b) + 1 deployed (i1→r1)
        assert len(graph["edges"]) == 4
        edge_types = [e["type"] for e in graph["edges"]]
        assert "relationship" in edge_types
        assert "source" in edge_types
        assert "deployed" in edge_types

    def test_insight_with_unknown_table(self):
        """Insight referencing a non-profiled table."""
        profiles = [_Prof("orders")]
        insights = [_Insight("i1", ["orders", "unknown_table"])]
        graph = build_lineage_graph(profiles, [], insights, [])
        assert "i1" in graph["tables"]["orders"]["insights"]
        assert "unknown_table" not in graph["tables"]

    def test_report_with_unknown_insight(self):
        """Report referencing a non-existent insight."""
        profiles = [_Prof("orders")]
        reports = [_Report("r1", "nonexistent")]
        graph = build_lineage_graph(profiles, [], [], reports)
        assert graph["reports"]["r1"]["source_tables"] == []

    def test_no_duplicate_related_tables(self):
        profiles = [_Prof("a"), _Prof("b")]
        rels = [_Rel("a", "b"), _Rel("a", "b")]
        graph = build_lineage_graph(profiles, rels, [], [])
        assert graph["tables"]["a"]["related_tables"].count("b") == 1

    def test_no_duplicate_insights_in_table(self):
        profiles = [_Prof("a")]
        insights = [_Insight("i1", ["a"]), _Insight("i1", ["a"])]
        graph = build_lineage_graph(profiles, [], insights, [])
        assert graph["tables"]["a"]["insights"].count("i1") == 1

    def test_dict_inputs(self):
        """Test that dict-based inputs work via _get_attr."""
        profiles = [{"table_name": "t1", "row_count": 50, "domain_hint": "hr"}]
        rels = [{"table_a": "t1", "table_b": "t2", "confidence": 0.8}]
        insights = [{"id": "i1", "source_tables": ["t1"], "insight_type": "anomaly", "title": "X"}]
        reports = [{"id": "r1", "insight_id": "i1", "name": "R"}]
        graph = build_lineage_graph(profiles, rels, insights, reports)
        assert "t1" in graph["tables"]
        assert "i1" in graph["insights"]


class TestGetTableLineage:
    def _make_graph(self):
        profiles = [_Prof("orders"), _Prof("products"), _Prof("orphan")]
        rels = [_Rel("orders", "products")]
        insights = [
            _Insight("i1", ["orders"]),
            _Insight("i2", ["orders", "products"]),
        ]
        reports = [_Report("r1", "i1")]
        return build_lineage_graph(profiles, rels, insights, reports)

    def test_orders_lineage(self):
        graph = self._make_graph()
        lineage = get_table_lineage(graph, "orders")
        assert lineage["table"] == "orders"
        assert "products" in lineage["upstream"]
        assert "i1" in lineage["downstream_insights"]
        assert "i2" in lineage["downstream_insights"]
        assert "r1" in lineage["downstream_reports"]
        assert lineage["impact"] == 3  # 2 insights + 1 report

    def test_orphan_lineage(self):
        graph = self._make_graph()
        lineage = get_table_lineage(graph, "orphan")
        assert lineage["impact"] == 0
        assert lineage["downstream_insights"] == []
        assert lineage["downstream_reports"] == []

    def test_unknown_table(self):
        graph = self._make_graph()
        lineage = get_table_lineage(graph, "nonexistent")
        assert lineage["impact"] == 0
        assert lineage["upstream"] == []


class TestGetImpactRanking:
    def test_ranking_order(self):
        profiles = [_Prof("high"), _Prof("medium"), _Prof("low")]
        insights = [
            _Insight("i1", ["high"]),
            _Insight("i2", ["high"]),
            _Insight("i3", ["medium"]),
        ]
        reports = [_Report("r1", "i1")]
        graph = build_lineage_graph(profiles, [], insights, reports)
        ranking = get_impact_ranking(graph)
        assert ranking[0]["table"] == "high"
        assert ranking[0]["impact"] == 3  # 2 insights + 1 report
        assert ranking[1]["table"] == "medium"
        assert ranking[1]["impact"] == 1
        assert ranking[2]["table"] == "low"
        assert ranking[2]["impact"] == 0

    def test_empty_graph(self):
        graph = build_lineage_graph([], [], [], [])
        assert get_impact_ranking(graph) == []

    def test_includes_relationship_count(self):
        profiles = [_Prof("a"), _Prof("b")]
        rels = [_Rel("a", "b")]
        graph = build_lineage_graph(profiles, rels, [], [])
        ranking = get_impact_ranking(graph)
        for r in ranking:
            assert r["relationship_count"] == 1


class TestFindOrphanedTables:
    def test_finds_orphans(self):
        profiles = [_Prof("connected"), _Prof("orphan1"), _Prof("orphan2")]
        insights = [_Insight("i1", ["connected"])]
        graph = build_lineage_graph(profiles, [], insights, [])
        orphans = find_orphaned_tables(graph)
        assert orphans == ["orphan1", "orphan2"]

    def test_no_orphans(self):
        profiles = [_Prof("a"), _Prof("b")]
        rels = [_Rel("a", "b")]
        graph = build_lineage_graph(profiles, rels, [], [])
        assert find_orphaned_tables(graph) == []

    def test_empty_graph(self):
        graph = build_lineage_graph([], [], [], [])
        assert find_orphaned_tables(graph) == []

    def test_relationship_prevents_orphan(self):
        profiles = [_Prof("a"), _Prof("b")]
        rels = [_Rel("a", "b")]
        graph = build_lineage_graph(profiles, rels, [], [])
        orphans = find_orphaned_tables(graph)
        assert "a" not in orphans
        assert "b" not in orphans

    def test_sorted_alphabetically(self):
        profiles = [_Prof("z_table"), _Prof("a_table"), _Prof("m_table")]
        graph = build_lineage_graph(profiles, [], [], [])
        orphans = find_orphaned_tables(graph)
        assert orphans == ["a_table", "m_table", "z_table"]
