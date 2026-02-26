"""Tests for insight recommender module."""

import pytest

from business_brain.discovery.insight_recommender import (
    EntityGroup,
    Recommendation,
    _build_entity_groups,
    _build_cross_table_prompt,
    _build_reason,
    _check_template_coverage,
    _coefficient_of_variation,
    _col_label,
    _column_completeness,
    _data_fitness_adjustment,
    _finalize,
    _is_unnamed_column,
    _parse_cross_table_suggestions,
    _rank_columns,
    _recommend_insight_followups,
    _recommend_tier1,
    _score_categorical_for_benchmark,
    _score_numeric_for_anomaly,
    _score_numeric_for_benchmark,
    _table_label,
    compute_coverage,
    recommend_analyses,
)


def _profile(name, row_count=100, columns=None, domain_hint=None,
             table_description=None, column_descriptions=None):
    """Helper to create a profile dict."""
    d = {
        "table_name": name,
        "row_count": row_count,
        "column_classification": {"columns": columns or {}},
    }
    if domain_hint:
        d["domain_hint"] = domain_hint
    if table_description:
        d["table_description"] = table_description
    if column_descriptions:
        d["column_descriptions"] = column_descriptions
    return d


# ---------------------------------------------------------------------------
# recommend_analyses (sync — backward-compat)
# ---------------------------------------------------------------------------


class TestRecommendAnalyses:
    def test_empty_input(self):
        assert recommend_analyses([], [], []) == []

    def test_tiny_table_skipped(self):
        profiles = [_profile("tiny", row_count=5)]
        recs = recommend_analyses(profiles, [], [])
        assert all(r.target_table != "tiny" for r in recs)

    def test_temporal_numeric_suggests_time_trend(self):
        cols = {
            "date": {"semantic_type": "temporal"},
            "revenue": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("sales", 200, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "time_trend" for r in recs)

    def test_categorical_numeric_suggests_benchmark(self):
        cols = {
            "dept": {"semantic_type": "categorical"},
            "score": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("employees", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "benchmark" for r in recs)

    def test_multiple_numerics_suggest_correlation(self):
        cols = {
            "a": {"semantic_type": "numeric_metric"},
            "b": {"semantic_type": "numeric_metric"},
            "c": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("metrics", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "correlation" for r in recs)

    def test_numeric_suggests_anomaly(self):
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profiles = [_profile("data", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "anomaly" for r in recs)

    def test_cohort_analysis_suggested(self):
        cols = {
            "supplier": {"semantic_type": "categorical"},
            "month": {"semantic_type": "temporal"},
            "cost": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("purchases", 200, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "cohort" for r in recs)

    def test_forecast_suggested_with_enough_data(self):
        cols = {
            "date": {"semantic_type": "temporal"},
            "val": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("time_series", 50, cols)]
        recs = recommend_analyses(profiles, [], [])
        assert any(r.analysis_type == "forecast" for r in recs)

    def test_existing_insights_reduce_priority(self):
        cols = {
            "val": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile("data", 100, cols)]
        # No insights
        recs_empty = recommend_analyses(profiles, [], [])
        # With existing anomaly insight
        insights = [{"insight_type": "anomaly", "source_tables": ["data"]}] * 10
        recs_full = recommend_analyses(profiles, insights, [])
        # Should have fewer anomaly recommendations when already covered
        anomaly_empty = [r for r in recs_empty if r.analysis_type == "anomaly"]
        anomaly_full = [r for r in recs_full if r.analysis_type == "anomaly"]
        assert len(anomaly_full) <= len(anomaly_empty)

    def test_max_recommendations(self):
        cols = {
            "cat": {"semantic_type": "categorical"},
            "date": {"semantic_type": "temporal"},
            "num1": {"semantic_type": "numeric_metric"},
            "num2": {"semantic_type": "numeric_metric"},
        }
        profiles = [_profile(f"table_{i}", 100, cols) for i in range(10)]
        recs = recommend_analyses(profiles, [], [], max_recommendations=5)
        assert len(recs) <= 5

    def test_cross_table_recommendation(self):
        profiles = [_profile("A", 100), _profile("B", 100)]
        relationships = [
            {"table_a": "A", "table_b": "B"},
            {"table_a": "A", "table_b": "C"},
        ]
        recs = recommend_analyses(profiles, [], relationships)
        # Should suggest cross-table analysis for A (has 2 relationships)
        assert any(r.target_table == "A" for r in recs)

    def test_deduplication(self):
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profiles = [_profile("data", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        # Should not have duplicate (table, analysis_type) pairs
        keys = [(r.target_table, r.analysis_type) for r in recs]
        assert len(keys) == len(set(keys))

    def test_recommendation_fields(self):
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profiles = [_profile("data", 100, cols)]
        recs = recommend_analyses(profiles, [], [])
        if recs:
            r = recs[0]
            assert r.title
            assert r.description
            assert r.analysis_type
            assert r.target_table == "data"
            assert 1 <= r.priority <= 100


# ---------------------------------------------------------------------------
# Entity group building
# ---------------------------------------------------------------------------


class TestBuildEntityGroups:
    def test_column_keyword_scan(self):
        """Tables with entity keyword columns form entity groups."""
        profiles = [
            _profile("booking_register", columns={
                "buyer_name": {"semantic_type": "categorical"},
                "order_qty": {"semantic_type": "numeric_metric"},
            }),
            _profile("payment_ledger", columns={
                "customer_id": {"semantic_type": "identifier"},
                "amount": {"semantic_type": "numeric_metric"},
            }),
        ]
        groups = _build_entity_groups(profiles, [])
        assert "customer" in groups
        assert "booking_register" in groups["customer"].tables
        assert "payment_ledger" in groups["customer"].tables

    def test_relationship_based_entity_inference(self):
        """Tables linked by relationships with entity-like column names form groups."""
        profiles = [
            _profile("table_a", columns={
                "party_code": {"semantic_type": "categorical"},
                "amount": {"semantic_type": "numeric_metric"},
            }),
            _profile("table_b", columns={
                "party_code": {"semantic_type": "categorical"},
                "qty": {"semantic_type": "numeric_metric"},
            }),
        ]
        relationships = [
            {
                "table_a": "table_a", "table_b": "table_b",
                "column_a": "party_code", "column_b": "party_code",
                "confidence": 0.8,
            },
        ]
        groups = _build_entity_groups(profiles, relationships)
        # "party" is in customer keywords
        assert "customer" in groups
        g = groups["customer"]
        assert "table_a" in g.tables
        assert "table_b" in g.tables

    def test_relationship_expansion(self):
        """Tables linked by relationships to group members get added."""
        profiles = [
            _profile("orders", columns={
                "customer_id": {"semantic_type": "identifier"},
            }),
            _profile("dispatch", columns={
                "ship_ref": {"semantic_type": "identifier"},
            }),
        ]
        # orders is in customer group via keyword; dispatch linked to orders
        relationships = [
            {
                "table_a": "orders", "table_b": "dispatch",
                "column_a": "customer_id", "column_b": "ship_ref",
                "confidence": 0.7,
            },
        ]
        groups = _build_entity_groups(profiles, relationships)
        assert "customer" in groups
        # dispatch should be expanded into customer group via link
        assert "dispatch" in groups["customer"].tables

    def test_low_confidence_ignored(self):
        """Relationships with very low confidence are ignored."""
        profiles = [
            _profile("a", columns={"x": {"semantic_type": "identifier"}}),
            _profile("b", columns={"y": {"semantic_type": "identifier"}}),
        ]
        relationships = [
            {
                "table_a": "a", "table_b": "b",
                "column_a": "x", "column_b": "y",
                "confidence": 0.1,
            },
        ]
        groups = _build_entity_groups(profiles, relationships)
        # Should not form any group from low-confidence relationship
        for g in groups.values():
            assert not ({"a", "b"} <= set(g.tables))

    def test_empty_profiles(self):
        """Empty input returns no groups."""
        assert _build_entity_groups([], []) == {}

    def test_supplier_group(self):
        """Supplier keywords detected correctly."""
        profiles = [
            _profile("purchase_orders", columns={
                "vendor_name": {"semantic_type": "categorical"},
            }),
            _profile("quality_checks", columns={
                "supplier_code": {"semantic_type": "identifier"},
            }),
        ]
        groups = _build_entity_groups(profiles, [])
        assert "supplier" in groups
        assert len(groups["supplier"].tables) == 2

    def test_legacy_rel_dicts_no_confidence(self):
        """Relationships without confidence field are still accepted."""
        profiles = [
            _profile("a", columns={"buyer": {"semantic_type": "categorical"}}),
            _profile("b", columns={"buyer": {"semantic_type": "categorical"}}),
        ]
        # Legacy format — no confidence, no column_a/column_b
        relationships = [{"table_a": "a", "table_b": "b"}]
        groups = _build_entity_groups(profiles, relationships)
        # Both tables in customer group via keyword; relationship shouldn't crash
        assert "customer" in groups


# ---------------------------------------------------------------------------
# Template coverage check
# ---------------------------------------------------------------------------


class TestCheckTemplateCoverage:
    def test_basic_coverage(self):
        """Two tables covering two requirements → True."""
        group = EntityGroup(
            entity_type="customer",
            tables=["booking_register", "payment_ledger"],
        )
        template = {
            "required_data": [
                {"table_keywords": ["booking", "order"], "col_keywords": ["order", "booking"]},
                {"table_keywords": ["payment", "receipt"], "col_keywords": ["payment", "amount"]},
            ],
        }
        profiles = [
            _profile("booking_register", columns={
                "order_no": {"semantic_type": "identifier"},
                "qty": {"semantic_type": "numeric_metric"},
            }),
            _profile("payment_ledger", columns={
                "payment_date": {"semantic_type": "temporal"},
                "amount": {"semantic_type": "numeric_metric"},
            }),
        ]
        assert _check_template_coverage(group, template, profiles) is True

    def test_single_table_satisfies_multiple_requirements(self):
        """One ERP table has both order AND payment data — should still match."""
        group = EntityGroup(
            entity_type="customer",
            tables=["erp_master", "dispatch_log"],
        )
        template = {
            "required_data": [
                {"table_keywords": ["booking", "order"], "col_keywords": ["order", "booking"]},
                {"table_keywords": ["payment", "receipt"], "col_keywords": ["payment", "amount"]},
            ],
        }
        profiles = [
            _profile("erp_master", columns={
                "order_no": {"semantic_type": "identifier"},
                "payment_status": {"semantic_type": "categorical"},
                "amount": {"semantic_type": "numeric_metric"},
            }),
            _profile("dispatch_log", columns={
                "dispatch_id": {"semantic_type": "identifier"},
            }),
        ]
        # erp_master has both "order" and "payment"/"amount" columns
        assert _check_template_coverage(group, template, profiles) is True

    def test_no_coverage(self):
        """Tables don't match any template requirements → False."""
        group = EntityGroup(
            entity_type="customer",
            tables=["weather_data", "expense_journal"],
        )
        template = {
            "required_data": [
                {"table_keywords": ["booking"], "col_keywords": ["order"]},
                {"table_keywords": ["payment"], "col_keywords": ["payment"]},
            ],
        }
        profiles = [
            _profile("weather_data", columns={"temp": {"semantic_type": "numeric_metric"}}),
            _profile("expense_journal", columns={"category": {"semantic_type": "categorical"}}),
        ]
        assert _check_template_coverage(group, template, profiles) is False

    def test_empty_required_data(self):
        """Template with no requirements always matches."""
        group = EntityGroup(entity_type="any", tables=["a", "b"])
        assert _check_template_coverage(group, {"required_data": []}, []) is True
        assert _check_template_coverage(group, {}, []) is True

    def test_table_name_match(self):
        """Template matches via table name hints even without matching columns."""
        group = EntityGroup(
            entity_type="supplier",
            tables=["purchase_orders", "quality_inspections"],
        )
        template = {
            "required_data": [
                {"table_keywords": ["purchase", "procurement"], "col_keywords": ["rate"]},
                {"table_keywords": ["quality", "inspection"], "col_keywords": ["defect"]},
            ],
        }
        profiles = [
            _profile("purchase_orders", columns={"amount": {"semantic_type": "numeric_metric"}}),
            _profile("quality_inspections", columns={"result": {"semantic_type": "categorical"}}),
        ]
        # Table names match even though columns don't have exact keyword matches
        assert _check_template_coverage(group, template, profiles) is True


# ---------------------------------------------------------------------------
# Tier 1 returns all groups + matched analyses
# ---------------------------------------------------------------------------


class TestRecommendTier1:
    def test_returns_three_tuple(self):
        """_recommend_tier1 returns (recs, all_groups, matched_analyses)."""
        result = _recommend_tier1([], [], [])
        assert len(result) == 3
        recs, groups, matched = result
        assert isinstance(recs, list)
        assert isinstance(groups, list)
        assert isinstance(matched, dict)

    def test_all_eligible_groups_returned(self):
        """Entity groups with 2+ tables are returned regardless of template match."""
        profiles = [
            _profile("booking_register", columns={
                "buyer_name": {"semantic_type": "categorical"},
                "order_qty": {"semantic_type": "numeric_metric"},
            }),
            _profile("payment_ledger", columns={
                "customer_id": {"semantic_type": "identifier"},
                "amount": {"semantic_type": "numeric_metric"},
            }),
        ]
        recs, groups, matched = _recommend_tier1(profiles, [], [])
        # Both tables share "customer" entity
        assert len(groups) >= 1
        customer_groups = [g for g in groups if g.entity_type == "customer"]
        assert len(customer_groups) == 1

    def test_matched_analyses_populated(self):
        """When a template matches, matched_analyses contains its title."""
        profiles = [
            _profile("booking_register", columns={
                "buyer_name": {"semantic_type": "categorical"},
                "booking_no": {"semantic_type": "identifier"},
            }),
            _profile("payment_ledger", columns={
                "customer_id": {"semantic_type": "identifier"},
                "payment_amount": {"semantic_type": "numeric_metric"},
            }),
        ]
        recs, groups, matched = _recommend_tier1(profiles, [], [])
        # The customer credit score template should match
        cross_table_recs = [r for r in recs if r.analysis_type == "cross_table_intelligence"]
        if cross_table_recs:
            # matched_analyses should have customer entry
            assert "customer" in matched
            assert len(matched["customer"]) > 0

    def test_unmatched_groups_still_returned(self):
        """Entity groups with no matching template still appear in all_groups."""
        profiles = [
            _profile("batch_records", columns={
                "product_code": {"semantic_type": "identifier"},
                "yield_pct": {"semantic_type": "numeric_metric"},
            }),
            _profile("stability_data", columns={
                "item_code": {"semantic_type": "identifier"},
                "shelf_life": {"semantic_type": "numeric_metric"},
            }),
        ]
        recs, groups, matched = _recommend_tier1(profiles, [], [])
        # product entity should be detected
        product_groups = [g for g in groups if g.entity_type == "product"]
        assert len(product_groups) >= 1
        # Whether templates matched or not, the group is in all_groups


# ---------------------------------------------------------------------------
# Dynamic suggestion parsing
# ---------------------------------------------------------------------------


class TestParseCrossTableSuggestions:
    def test_valid_suggestions(self):
        group = EntityGroup(
            entity_type="supplier",
            tables=["purchase_orders", "quality_data"],
        )
        llm_response = (
            "SUGGESTION: Vendor Risk Score | Combine purchase volume with rejection rate "
            "to identify high-risk vendors | 88 | Critical for supply chain resilience\n"
            "SUGGESTION: Price-Quality Tradeoff | Correlate purchase price with incoming "
            "quality metrics | 82 | Helps negotiate better contracts\n"
        )
        recs = _parse_cross_table_suggestions(llm_response, group)
        assert len(recs) == 2
        assert recs[0].analysis_type == "cross_table_intelligence"
        assert "Vendor Risk Score" in recs[0].title
        assert 65 <= recs[0].priority <= 92
        assert "[AI-suggested]" in recs[0].reason

    def test_no_suggestions(self):
        group = EntityGroup(entity_type="x", tables=["a", "b"])
        assert _parse_cross_table_suggestions("NO_SUGGESTIONS", group) == []
        assert _parse_cross_table_suggestions("", group) == []

    def test_malformed_lines_skipped(self):
        group = EntityGroup(entity_type="x", tables=["a", "b"])
        llm_response = (
            "Some random text\n"
            "SUGGESTION: Only title\n"  # Too few parts
            "SUGGESTION: Valid | Description | 80 | Reason\n"
        )
        recs = _parse_cross_table_suggestions(llm_response, group)
        assert len(recs) == 1
        assert "Valid" in recs[0].title

    def test_priority_clamped(self):
        group = EntityGroup(entity_type="x", tables=["a", "b"])
        llm_response = "SUGGESTION: Title | Desc | 99 | Reason\n"
        recs = _parse_cross_table_suggestions(llm_response, group)
        assert recs[0].priority <= 92  # Clamped

    def test_invalid_priority_defaults(self):
        group = EntityGroup(entity_type="x", tables=["a", "b"])
        llm_response = "SUGGESTION: Title | Desc | abc | Reason\n"
        recs = _parse_cross_table_suggestions(llm_response, group)
        assert recs[0].priority == 75  # Default


# ---------------------------------------------------------------------------
# Cross-table prompt building
# ---------------------------------------------------------------------------


class TestBuildCrossTablePrompt:
    def test_basic_prompt(self):
        group = EntityGroup(
            entity_type="supplier",
            tables=["purchases", "quality"],
            entity_columns={"purchases": ["vendor_name"], "quality": ["supplier_code"]},
        )
        profiles = [
            _profile("purchases", columns={
                "vendor_name": {"semantic_type": "categorical", "sample_values": ["A", "B"]},
                "rate": {"semantic_type": "numeric_metric"},
            }),
            _profile("quality", columns={
                "supplier_code": {"semantic_type": "identifier"},
                "defect_rate": {"semantic_type": "numeric_metric"},
            }),
        ]
        context = {"industry": "manufacturing", "products": ["TMT bars"]}
        prompt = _build_cross_table_prompt(group, profiles, context)

        assert "manufacturing" in prompt
        assert "supplier" in prompt
        assert "purchases" in prompt
        assert "quality" in prompt
        assert "SUGGESTION:" in prompt

    def test_prompt_includes_existing_analyses(self):
        """When Tier 1 templates already matched, the prompt tells the LLM."""
        group = EntityGroup(
            entity_type="customer",
            tables=["orders", "payments"],
            entity_columns={"orders": ["buyer"], "payments": ["customer"]},
        )
        profiles = [
            _profile("orders", columns={"buyer": {"semantic_type": "categorical"}}),
            _profile("payments", columns={"customer": {"semantic_type": "categorical"}}),
        ]
        context = {"industry": "steel"}
        existing = [
            "Build buyer credit score report",
            "Customer profitability analysis",
        ]
        prompt = _build_cross_table_prompt(group, profiles, context, existing)

        # Should include the already-covered section
        assert "ALREADY been generated" in prompt
        assert "credit score" in prompt
        assert "profitability" in prompt
        assert "COMPLEMENTARY" in prompt

    def test_prompt_no_existing_analyses(self):
        """Without existing analyses, no 'already covered' section."""
        group = EntityGroup(entity_type="x", tables=["a", "b"])
        profiles = [_profile("a"), _profile("b")]
        context = {"industry": "pharma"}

        prompt_no_existing = _build_cross_table_prompt(group, profiles, context, [])
        prompt_none = _build_cross_table_prompt(group, profiles, context, None)

        assert "ALREADY been generated" not in prompt_no_existing
        assert "ALREADY been generated" not in prompt_none


# ---------------------------------------------------------------------------
# Cross-table intelligence end-to-end (sync)
# ---------------------------------------------------------------------------


class TestCrossTableIntelligence:
    def test_customer_credit_score_template(self):
        """booking_register + payment_ledger → credit score recommendation."""
        profiles = [
            _profile("booking_register", columns={
                "buyer_name": {"semantic_type": "categorical"},
                "booking_no": {"semantic_type": "identifier"},
                "order_qty": {"semantic_type": "numeric_metric"},
            }),
            _profile("payment_ledger", columns={
                "customer_id": {"semantic_type": "identifier"},
                "payment_amount": {"semantic_type": "numeric_metric"},
                "payment_date": {"semantic_type": "temporal"},
            }),
        ]
        recs = recommend_analyses(profiles, [], [])
        cross_recs = [r for r in recs if r.analysis_type == "cross_table_intelligence"]
        assert len(cross_recs) >= 1
        assert any("credit" in r.title.lower() for r in cross_recs)

    def test_supplier_tco_template(self):
        """purchase_orders + quality_inspections → total cost of ownership."""
        profiles = [
            _profile("purchase_orders", columns={
                "vendor_name": {"semantic_type": "categorical"},
                "rate": {"semantic_type": "numeric_metric"},
                "amount": {"semantic_type": "numeric_metric"},
            }),
            _profile("quality_rejections", columns={
                "supplier_code": {"semantic_type": "identifier"},
                "reject_count": {"semantic_type": "numeric_metric"},
                "defect_type": {"semantic_type": "categorical"},
            }),
        ]
        recs = recommend_analyses(profiles, [], [])
        cross_recs = [r for r in recs if r.analysis_type == "cross_table_intelligence"]
        assert len(cross_recs) >= 1
        assert any("cost" in r.title.lower() or "ownership" in r.title.lower() for r in cross_recs)

    def test_relationship_linked_tables_get_suggestions(self):
        """Tables linked only by discovered relationships still get cross-table recs."""
        profiles = [
            _profile("table_x", columns={
                "party_code": {"semantic_type": "categorical"},
                "qty": {"semantic_type": "numeric_metric"},
            }),
            _profile("table_y", columns={
                "party_code": {"semantic_type": "categorical"},
                "amount": {"semantic_type": "numeric_metric"},
            }),
        ]
        relationships = [
            {
                "table_a": "table_x", "table_b": "table_y",
                "column_a": "party_code", "column_b": "party_code",
                "confidence": 0.9,
            },
        ]
        recs = recommend_analyses(profiles, [], relationships)
        # Both tables should be linked via "party" (customer keyword) entity group
        # At minimum, they should have some cross-table recommendation
        all_tables = {r.target_table for r in recs}
        assert "table_x" in all_tables or "table_y" in all_tables

    def test_unrelated_tables_no_cross_table(self):
        """Tables with no shared entity produce no cross-table intelligence."""
        profiles = [
            _profile("weather_data", columns={
                "temperature": {"semantic_type": "numeric_metric"},
                "date": {"semantic_type": "temporal"},
            }),
            _profile("expense_journal", columns={
                "category": {"semantic_type": "categorical"},
                "value": {"semantic_type": "numeric_metric"},
            }),
        ]
        recs = recommend_analyses(profiles, [], [])
        cross_recs = [r for r in recs if r.analysis_type == "cross_table_intelligence"]
        assert len(cross_recs) == 0


# ---------------------------------------------------------------------------
# compute_coverage
# ---------------------------------------------------------------------------


class TestComputeCoverage:
    def test_full_coverage(self):
        profiles = [_profile("A"), _profile("B")]
        insights = [
            {"source_tables": ["A"]},
            {"source_tables": ["B"]},
        ]
        cov = compute_coverage(profiles, insights)
        assert cov["coverage_pct"] == 100.0
        assert cov["uncovered_tables"] == []

    def test_partial_coverage(self):
        profiles = [_profile("A"), _profile("B"), _profile("C")]
        insights = [{"source_tables": ["A"]}]
        cov = compute_coverage(profiles, insights)
        assert cov["covered_tables"] == 1
        assert len(cov["uncovered_tables"]) == 2

    def test_no_insights(self):
        profiles = [_profile("A")]
        cov = compute_coverage(profiles, [])
        assert cov["coverage_pct"] == 0.0

    def test_empty(self):
        cov = compute_coverage([], [])
        assert cov["total_tables"] == 0
        assert cov["coverage_pct"] == 0.0


# ---------------------------------------------------------------------------
# Humanized names & data-aware reasons
# ---------------------------------------------------------------------------


class TestHumanizedRecommendations:
    def test_benchmark_uses_humanized_names(self):
        """When column_descriptions are provided, titles use human labels."""
        cols = {
            "gajkesari_steels___alloys_pvt__ltd": {"semantic_type": "categorical",
                                                    "cardinality": 12,
                                                    "sample_values": ["Gajkesari", "Tata", "JSW"]},
            "rate_per_mt": {"semantic_type": "numeric_metric",
                           "stats": {"min": 3200, "max": 5800, "stdev": 450}},
        }
        profile = _profile(
            "sponge_purchase_3",
            row_count=200,
            columns=cols,
            table_description="Sponge iron purchase register for FY 2018-19",
            column_descriptions={
                "gajkesari_steels___alloys_pvt__ltd": "Supplier name",
                "rate_per_mt": "Purchase rate per metric ton",
            },
        )
        recs = recommend_analyses([profile], [], [])
        bench = [r for r in recs if r.analysis_type == "benchmark"]
        assert bench, "Expected a benchmark recommendation"
        r = bench[0]
        # Title should use human labels, not raw column names
        assert "gajkesari_steels___alloys_pvt__ltd" not in r.title
        assert "sponge_purchase_3" not in r.title
        # Should contain human-readable labels
        assert "Purchase rate per metric ton" in r.title or "Supplier name" in r.title

    def test_unnamed_columns_suppressed(self):
        """Columns like col_3 without descriptions are not featured in titles."""
        cols = {
            "col_3": {"semantic_type": "categorical"},
            "col_5": {"semantic_type": "numeric_metric"},
            "supplier": {"semantic_type": "categorical"},
            "amount": {"semantic_type": "numeric_metric"},
        }
        profile = _profile("sale_register", row_count=100, columns=cols)
        recs = recommend_analyses([profile], [], [])
        bench = [r for r in recs if r.analysis_type == "benchmark"]
        assert bench, "Expected a benchmark recommendation"
        # Benchmark should prefer named columns over col_N artifacts
        assert "col_3" not in bench[0].title
        assert "col_5" not in bench[0].title

    def test_reason_includes_stats(self):
        """Benchmark reason includes min/max from column stats."""
        cols = {
            "dept": {"semantic_type": "categorical", "cardinality": 5,
                     "sample_values": ["HR", "Sales", "Ops"]},
            "revenue": {"semantic_type": "numeric_metric",
                        "stats": {"min": 1000, "max": 50000, "stdev": 8000}},
        }
        profile = _profile("departments", row_count=100, columns=cols)
        reason = _build_reason("benchmark", profile, ["dept", "revenue"], "general")
        # Should include actual min/max numbers
        assert "1000" in reason
        assert "50000" in reason
        # Should include cardinality
        assert "5" in reason

    def test_backward_compat_no_enrichment(self):
        """Profiles without enrichment still produce valid recommendations."""
        cols = {
            "date": {"semantic_type": "temporal"},
            "val": {"semantic_type": "numeric_metric"},
            "dept": {"semantic_type": "categorical"},
        }
        # No table_description, no column_descriptions — classic format
        profile = _profile("basic_table", row_count=100, columns=cols)
        recs = recommend_analyses([profile], [], [])
        # Should still generate recs without crashing
        assert len(recs) >= 3  # time_trend, benchmark, correlation, anomaly, cohort, forecast
        for r in recs:
            assert r.title
            assert r.reason
            assert r.target_table == "basic_table"


# ---------------------------------------------------------------------------
# Column scoring system
# ---------------------------------------------------------------------------


class TestColumnScoring:
    def test_cv_computation(self):
        assert _coefficient_of_variation({"mean": 10, "stdev": 5}) == 0.5
        assert _coefficient_of_variation({"mean": 0, "stdev": 5}) == 0.0
        assert _coefficient_of_variation({}) == 0.0

    def test_completeness(self):
        assert _column_completeness({"null_count": 0}, 100) == 1.0
        assert _column_completeness({"null_count": 50}, 100) == 0.5
        assert _column_completeness({"null_count": 0}, 0) == 1.0

    def test_cat_benchmark_ideal_cardinality(self):
        """Cardinality 10 scores higher than cardinality 200."""
        profile = _profile("t", 100)
        good = _score_categorical_for_benchmark(
            "supplier", {"cardinality": 10, "sample_values": ["A", "B", "C"]}, 100, profile,
        )
        bad = _score_categorical_for_benchmark(
            "supplier", {"cardinality": 200, "sample_values": ["A"]}, 100, profile,
        )
        assert good > bad

    def test_cat_benchmark_unnamed_penalized(self):
        """col_3 scores lower than supplier for benchmark categorical."""
        profile = _profile("t", 100)
        named = _score_categorical_for_benchmark("supplier", {"cardinality": 10}, 100, profile)
        unnamed = _score_categorical_for_benchmark("col_3", {"cardinality": 10}, 100, profile)
        assert named > unnamed

    def test_num_benchmark_high_cv_preferred(self):
        """High CV column scores higher for benchmark."""
        profile = _profile("t", 100)
        high_cv = _score_numeric_for_benchmark(
            "amount", {"stats": {"mean": 100, "stdev": 80, "min": 5, "max": 500}}, 100, profile,
        )
        low_cv = _score_numeric_for_benchmark(
            "amount", {"stats": {"mean": 100, "stdev": 1, "min": 99, "max": 101}}, 100, profile,
        )
        assert high_cv > low_cv

    def test_num_anomaly_constant_zero(self):
        """Constant column (stdev=0) scores 0 for anomaly."""
        profile = _profile("t", 100)
        score = _score_numeric_for_anomaly(
            "x", {"stats": {"mean": 5, "stdev": 0, "min": 5, "max": 5}}, 100, profile,
        )
        assert score == 0.0

    def test_rank_columns_returns_sorted(self):
        """_rank_columns returns columns sorted by descending score."""
        cols = {
            "bad": {"stats": {"mean": 100, "stdev": 0.1, "min": 99, "max": 101}},
            "good": {"stats": {"mean": 100, "stdev": 80, "min": 5, "max": 500}},
        }
        profile = _profile("t", 100, cols)
        ranked = _rank_columns(cols, 100, profile, _score_numeric_for_benchmark)
        assert ranked[0] == "good"
        assert ranked[1] == "bad"


# ---------------------------------------------------------------------------
# Data fitness priority adjustment
# ---------------------------------------------------------------------------


class TestDataFitnessAdjustment:
    def test_benchmark_sweet_spot_boost(self):
        """Benchmark with cardinality=10 and high CV gets positive adjustment."""
        cols = {
            "dept": {"semantic_type": "categorical", "cardinality": 10, "null_count": 0},
            "revenue": {"semantic_type": "numeric_metric", "null_count": 0,
                        "stats": {"mean": 1000, "stdev": 800, "min": 50, "max": 5000}},
        }
        profile = _profile("t", 100, cols)
        adj = _data_fitness_adjustment("benchmark", profile, ["dept", "revenue"])
        assert adj > 0

    def test_benchmark_high_cardinality_penalty(self):
        """Benchmark with cardinality=200 and low CV gets negative adjustment."""
        cols = {
            "item": {"semantic_type": "categorical", "cardinality": 200, "null_count": 0},
            "val": {"semantic_type": "numeric_metric", "null_count": 0,
                    "stats": {"mean": 100, "stdev": 1, "min": 99, "max": 101}},
        }
        profile = _profile("t", 100, cols)
        adj = _data_fitness_adjustment("benchmark", profile, ["item", "val"])
        assert adj < 0

    def test_correlation_high_cv_boost(self):
        """Correlation columns with high average CV get boosted."""
        cols = {
            "a": {"semantic_type": "numeric_metric", "stats": {"mean": 10, "stdev": 8}},
            "b": {"semantic_type": "numeric_metric", "stats": {"mean": 20, "stdev": 15}},
        }
        profile = _profile("t", 100, cols)
        adj = _data_fitness_adjustment("correlation", profile, ["a", "b"])
        assert adj > 0


# ---------------------------------------------------------------------------
# Insight-driven follow-up recommendations
# ---------------------------------------------------------------------------


class TestInsightFollowups:
    def test_critical_anomaly_generates_benchmark_followup(self):
        """Critical anomaly with categorical columns suggests root-cause benchmark."""
        cols = {
            "supplier": {"semantic_type": "categorical", "cardinality": 8},
            "rate": {"semantic_type": "numeric_metric",
                     "stats": {"mean": 100, "stdev": 50, "min": 30, "max": 250}},
            "date": {"semantic_type": "temporal"},
        }
        profile = _profile("purchases", 200, cols)
        insights = [{
            "insight_type": "anomaly",
            "severity": "critical",
            "source_tables": ["purchases"],
            "source_columns": ["rate"],
            "evidence": {},
        }]
        recs = _recommend_insight_followups(profile, insights, cols, "procurement")
        assert any(r.analysis_type == "benchmark" for r in recs)
        assert any("anomal" in r.title.lower() or "investigate" in r.title.lower() for r in recs)

    def test_strong_correlation_generates_trend_followup(self):
        """Strong correlation suggests tracking the relationship over time."""
        cols = {
            "cost": {"semantic_type": "numeric_metric"},
            "weight": {"semantic_type": "numeric_metric"},
            "date": {"semantic_type": "temporal"},
        }
        profile = _profile("production", 200, cols)
        insights = [{
            "insight_type": "correlation",
            "severity": "info",
            "source_tables": ["production"],
            "source_columns": ["cost", "weight"],
            "evidence": {"estimated_correlation": 0.92},
        }]
        recs = _recommend_insight_followups(profile, insights, cols, "manufacturing")
        assert any("relationship" in r.title.lower() or "track" in r.title.lower() for r in recs)

    def test_multiple_insights_suggest_story(self):
        """3+ insights of different types suggest a data story."""
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profile = _profile("data", 200, cols)
        insights = [
            {"insight_type": "anomaly", "severity": "warning", "source_tables": ["data"],
             "source_columns": ["val"], "evidence": {}},
            {"insight_type": "correlation", "severity": "info", "source_tables": ["data"],
             "source_columns": ["val"], "evidence": {}},
            {"insight_type": "trend", "severity": "info", "source_tables": ["data"],
             "source_columns": ["val"], "evidence": {}},
        ]
        recs = _recommend_insight_followups(profile, insights, cols, "general")
        assert any(r.analysis_type == "story" for r in recs)

    def test_no_insights_no_followups(self):
        """No insights means no follow-up recommendations."""
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profile = _profile("data", 200, cols)
        recs = _recommend_insight_followups(profile, [], cols, "general")
        assert recs == []

    def test_info_anomaly_no_followup(self):
        """Info-level anomalies do not trigger follow-ups."""
        cols = {"val": {"semantic_type": "numeric_metric"}, "cat": {"semantic_type": "categorical"}}
        profile = _profile("data", 200, cols)
        insights = [{
            "insight_type": "anomaly", "severity": "info",
            "source_tables": ["data"], "source_columns": ["val"], "evidence": {},
        }]
        recs = _recommend_insight_followups(profile, insights, cols, "general")
        assert not any(r.analysis_type == "benchmark" and "anomal" in r.title.lower() for r in recs)


# ---------------------------------------------------------------------------
# Finalize diversity
# ---------------------------------------------------------------------------


class TestFinalizeDiversity:
    def test_diversity_prevents_single_table_domination(self):
        """When one table has much higher priorities, other tables still appear."""
        recs = [
            Recommendation("R1", "", "anomaly", "table_A", [], 95, ""),
            Recommendation("R2", "", "benchmark", "table_A", [], 93, ""),
            Recommendation("R3", "", "correlation", "table_A", [], 91, ""),
            Recommendation("R4", "", "time_trend", "table_A", [], 89, ""),
            Recommendation("R5", "", "cohort", "table_A", [], 87, ""),
            Recommendation("R6", "", "anomaly", "table_B", [], 70, ""),
            Recommendation("R7", "", "benchmark", "table_B", [], 68, ""),
        ]
        result = _finalize(recs, 5)
        tables = {r.target_table for r in result}
        # table_B should appear in top 5 despite lower priority
        assert "table_B" in tables


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_profiles_without_stats_still_work(self):
        """Profiles with minimal column info (no stats, no cardinality) still generate recs."""
        cols = {
            "cat": {"semantic_type": "categorical"},
            "num": {"semantic_type": "numeric_metric"},
        }
        profile = _profile("t", 100, cols)
        recs = recommend_analyses([profile], [], [])
        assert len(recs) >= 1

    def test_insight_dicts_without_new_fields_still_work(self):
        """Legacy insight dicts (only insight_type + source_tables) work with new code."""
        cols = {"val": {"semantic_type": "numeric_metric"}}
        profile = _profile("data", 100, cols)
        insights = [{"insight_type": "anomaly", "source_tables": ["data"]}]
        recs = recommend_analyses([profile], insights, [])
        assert isinstance(recs, list)
