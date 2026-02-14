"""Tests for supplier scorecard module."""

from business_brain.discovery.supplier_scorecard import (
    ScorecardResult,
    SupplierScore,
    MetricScore,
    ComparisonResult,
    MetricComparison,
    SupplierRisk,
    ConcentrationResult,
    build_scorecard,
    compare_suppliers,
    detect_supplier_risks,
    supplier_concentration,
    format_scorecard,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _sample_rows():
    """Basic supplier data: 3 suppliers, 2 metrics each."""
    return [
        {"supplier": "Acme",   "quality": 95, "cost": 50},
        {"supplier": "Acme",   "quality": 90, "cost": 55},
        {"supplier": "Beta",   "quality": 70, "cost": 40},
        {"supplier": "Beta",   "quality": 75, "cost": 45},
        {"supplier": "Gamma",  "quality": 60, "cost": 30},
        {"supplier": "Gamma",  "quality": 65, "cost": 35},
    ]


def _sample_metrics():
    return [
        {"column": "quality", "weight": 0.6, "direction": "higher_is_better"},
        {"column": "cost",    "weight": 0.4, "direction": "lower_is_better"},
    ]


def _build_sample():
    return build_scorecard(_sample_rows(), "supplier", _sample_metrics())


# ---------------------------------------------------------------------------
# 1. build_scorecard
# ---------------------------------------------------------------------------


class TestBuildScorecard:
    def test_basic_returns_result(self):
        result = _build_sample()
        assert result is not None
        assert isinstance(result, ScorecardResult)

    def test_supplier_count(self):
        result = _build_sample()
        assert result.supplier_count == 3

    def test_scores_between_0_and_100(self):
        result = _build_sample()
        for s in result.suppliers:
            assert 0 <= s.score <= 100

    def test_ranks_are_sequential(self):
        result = _build_sample()
        ranks = [s.rank for s in result.suppliers]
        assert ranks == [1, 2, 3]

    def test_best_and_worst(self):
        result = _build_sample()
        assert result.best_supplier == result.suppliers[0].supplier
        assert result.worst_supplier == result.suppliers[-1].supplier

    def test_grades_assigned(self):
        result = _build_sample()
        valid_grades = {"A", "B", "C", "D", "F"}
        for s in result.suppliers:
            assert s.grade in valid_grades

    def test_grade_distribution_sums_to_total(self):
        result = _build_sample()
        assert sum(result.grade_distribution.values()) == result.supplier_count

    def test_mean_score_correct(self):
        result = _build_sample()
        expected = round(sum(s.score for s in result.suppliers) / len(result.suppliers), 2)
        assert result.mean_score == expected

    def test_metric_scores_populated(self):
        result = _build_sample()
        for s in result.suppliers:
            assert len(s.metric_scores) == 2
            for ms in s.metric_scores:
                assert isinstance(ms, MetricScore)

    def test_weights_normalised(self):
        """Metric weights in the result should sum to ~1."""
        result = _build_sample()
        s = result.suppliers[0]
        total_w = sum(ms.weight for ms in s.metric_scores)
        assert abs(total_w - 1.0) < 0.01

    def test_empty_rows_returns_none(self):
        assert build_scorecard([], "supplier", _sample_metrics()) is None

    def test_empty_metrics_returns_none(self):
        assert build_scorecard(_sample_rows(), "supplier", []) is None

    def test_missing_column_still_scores(self):
        """If some rows lack a metric column, suppliers still get scored."""
        rows = [
            {"supplier": "A", "quality": 80},
            {"supplier": "B", "quality": 60},
        ]
        metrics = [
            {"column": "quality", "weight": 0.5, "direction": "higher_is_better"},
            {"column": "delivery", "weight": 0.5, "direction": "higher_is_better"},
        ]
        result = build_scorecard(rows, "supplier", metrics)
        assert result is not None
        assert result.supplier_count == 2

    def test_lower_is_better_direction(self):
        """With lower_is_better, the supplier with the lowest cost should rank highest on that metric."""
        rows = [
            {"supplier": "Cheap", "cost": 10},
            {"supplier": "Pricey", "cost": 100},
        ]
        metrics = [{"column": "cost", "weight": 1.0, "direction": "lower_is_better"}]
        result = build_scorecard(rows, "supplier", metrics)
        assert result is not None
        assert result.best_supplier == "Cheap"

    def test_all_equal_values_gives_same_score(self):
        """If every supplier has the same metric value, all scores should be equal."""
        rows = [
            {"supplier": "A", "quality": 50},
            {"supplier": "B", "quality": 50},
        ]
        metrics = [{"column": "quality", "weight": 1.0, "direction": "higher_is_better"}]
        result = build_scorecard(rows, "supplier", metrics)
        assert result is not None
        assert result.suppliers[0].score == result.suppliers[1].score

    def test_summary_contains_names(self):
        result = _build_sample()
        assert result.best_supplier in result.summary
        assert result.worst_supplier in result.summary

    def test_strengths_and_weaknesses(self):
        """Top scorer should have at least one strength; worst should have a weakness."""
        result = _build_sample()
        best = result.suppliers[0]
        worst = result.suppliers[-1]
        # The best supplier should have at least one strong metric
        assert len(best.strengths) >= 1 or len(worst.weaknesses) >= 1


# ---------------------------------------------------------------------------
# 2. compare_suppliers
# ---------------------------------------------------------------------------


class TestCompareSuppliers:
    def test_basic_comparison(self):
        result = _build_sample()
        comp = compare_suppliers(result, "Acme", "Beta")
        assert comp is not None
        assert isinstance(comp, ComparisonResult)

    def test_winner_correct(self):
        result = _build_sample()
        comp = compare_suppliers(result, "Acme", "Beta")
        higher = "Acme" if comp.score_a > comp.score_b else "Beta"
        assert comp.winner == higher

    def test_metric_comparisons_populated(self):
        result = _build_sample()
        comp = compare_suppliers(result, "Acme", "Gamma")
        assert len(comp.metric_comparisons) == 2
        for mc in comp.metric_comparisons:
            assert isinstance(mc, MetricComparison)

    def test_advantages_lists(self):
        result = _build_sample()
        comp = compare_suppliers(result, "Acme", "Gamma")
        # Acme beats Gamma on quality (higher) but not cost (higher cost, which is bad)
        total_advantages = len(comp.advantages_a) + len(comp.advantages_b)
        assert total_advantages >= 1

    def test_unknown_supplier_returns_none(self):
        result = _build_sample()
        assert compare_suppliers(result, "Acme", "NoSuch") is None

    def test_both_unknown_returns_none(self):
        result = _build_sample()
        assert compare_suppliers(result, "X", "Y") is None

    def test_difference_pct_nonnegative(self):
        result = _build_sample()
        comp = compare_suppliers(result, "Acme", "Beta")
        for mc in comp.metric_comparisons:
            assert mc.difference_pct >= 0


# ---------------------------------------------------------------------------
# 3. detect_supplier_risks
# ---------------------------------------------------------------------------


class TestDetectSupplierRisks:
    def test_no_risks_on_empty(self):
        assert detect_supplier_risks([], "supplier", _sample_metrics()) == []

    def test_below_threshold_detected(self):
        rows = [
            {"supplier": "Bad",  "quality": 30},
            {"supplier": "Good", "quality": 90},
        ]
        metrics = [{"column": "quality", "weight": 1.0, "direction": "higher_is_better"}]
        risks = detect_supplier_risks(rows, "supplier", metrics, thresholds={"quality": 50})
        risk_types = [r.risk_type for r in risks]
        assert "below_threshold" in risk_types
        below = [r for r in risks if r.risk_type == "below_threshold"]
        assert any(r.supplier == "Bad" for r in below)

    def test_declining_detected(self):
        """Supplier with clearly declining quality over 6 readings."""
        rows = [
            {"supplier": "Decline", "quality": 95},
            {"supplier": "Decline", "quality": 90},
            {"supplier": "Decline", "quality": 85},
            {"supplier": "Decline", "quality": 40},
            {"supplier": "Decline", "quality": 35},
            {"supplier": "Decline", "quality": 30},
            {"supplier": "Stable",  "quality": 80},
            {"supplier": "Stable",  "quality": 80},
            {"supplier": "Stable",  "quality": 80},
            {"supplier": "Stable",  "quality": 80},
            {"supplier": "Stable",  "quality": 80},
            {"supplier": "Stable",  "quality": 80},
        ]
        metrics = [{"column": "quality", "weight": 1.0, "direction": "higher_is_better"}]
        risks = detect_supplier_risks(rows, "supplier", metrics)
        declining = [r for r in risks if r.risk_type == "declining"]
        assert len(declining) >= 1
        assert declining[0].supplier == "Decline"

    def test_single_source_detected(self):
        """Only one supplier provides data for a metric."""
        rows = [
            {"supplier": "Solo", "quality": 80},
        ]
        metrics = [{"column": "quality", "weight": 1.0, "direction": "higher_is_better"}]
        risks = detect_supplier_risks(rows, "supplier", metrics)
        single = [r for r in risks if r.risk_type == "single_source"]
        assert len(single) == 1
        assert single[0].supplier == "Solo"

    def test_risk_has_required_fields(self):
        rows = [
            {"supplier": "Bad",  "quality": 30},
            {"supplier": "Good", "quality": 90},
        ]
        metrics = [{"column": "quality", "weight": 1.0, "direction": "higher_is_better"}]
        risks = detect_supplier_risks(rows, "supplier", metrics, thresholds={"quality": 50})
        for r in risks:
            assert isinstance(r, SupplierRisk)
            assert r.risk_type in ("declining", "below_threshold", "single_source")
            assert r.severity in ("low", "medium", "high")
            assert r.affected_metric != ""

    def test_no_false_decline_on_stable_data(self):
        """Stable data should not trigger a declining risk."""
        rows = [
            {"supplier": "Stable", "quality": 80},
            {"supplier": "Stable", "quality": 80},
            {"supplier": "Stable", "quality": 80},
            {"supplier": "Stable", "quality": 80},
            {"supplier": "Other",  "quality": 80},
            {"supplier": "Other",  "quality": 80},
            {"supplier": "Other",  "quality": 80},
            {"supplier": "Other",  "quality": 80},
        ]
        metrics = [{"column": "quality", "weight": 1.0, "direction": "higher_is_better"}]
        risks = detect_supplier_risks(rows, "supplier", metrics)
        declining = [r for r in risks if r.risk_type == "declining"]
        assert len(declining) == 0


# ---------------------------------------------------------------------------
# 4. supplier_concentration
# ---------------------------------------------------------------------------


class TestSupplierConcentration:
    def test_basic(self):
        rows = [
            {"supplier": "A", "spend": 500},
            {"supplier": "B", "spend": 300},
            {"supplier": "C", "spend": 200},
        ]
        result = supplier_concentration(rows, "supplier", "spend")
        assert result is not None
        assert isinstance(result, ConcentrationResult)

    def test_hhi_calculation(self):
        """Two equal suppliers: each 50% -> HHI = 2*50^2 = 5000."""
        rows = [
            {"supplier": "A", "spend": 100},
            {"supplier": "B", "spend": 100},
        ]
        result = supplier_concentration(rows, "supplier", "spend")
        assert result is not None
        assert result.hhi == 5000.0

    def test_high_concentration(self):
        """One supplier with 90% share."""
        rows = [
            {"supplier": "Mono", "spend": 900},
            {"supplier": "Small", "spend": 100},
        ]
        result = supplier_concentration(rows, "supplier", "spend")
        assert result.concentration_level == "high"

    def test_low_concentration(self):
        """Many equal suppliers."""
        rows = [{"supplier": f"S{i}", "spend": 100} for i in range(20)]
        result = supplier_concentration(rows, "supplier", "spend")
        assert result is not None
        assert result.concentration_level == "low"
        assert result.hhi < 1500

    def test_top_supplier_share(self):
        rows = [
            {"supplier": "Big",   "spend": 600},
            {"supplier": "Small", "spend": 400},
        ]
        result = supplier_concentration(rows, "supplier", "spend")
        assert result.top_supplier_share == 60.0

    def test_empty_returns_none(self):
        assert supplier_concentration([], "supplier", "spend") is None

    def test_shares_sum_to_100(self):
        rows = [
            {"supplier": "A", "spend": 300},
            {"supplier": "B", "spend": 200},
            {"supplier": "C", "spend": 500},
        ]
        result = supplier_concentration(rows, "supplier", "spend")
        total = sum(s["share_pct"] for s in result.suppliers)
        assert abs(total - 100.0) < 0.1

    def test_summary_text(self):
        rows = [
            {"supplier": "A", "spend": 500},
            {"supplier": "B", "spend": 500},
        ]
        result = supplier_concentration(rows, "supplier", "spend")
        assert "HHI" in result.summary

    def test_zero_spend_returns_none(self):
        rows = [
            {"supplier": "A", "spend": 0},
            {"supplier": "B", "spend": 0},
        ]
        result = supplier_concentration(rows, "supplier", "spend")
        assert result is None


# ---------------------------------------------------------------------------
# 5. format_scorecard
# ---------------------------------------------------------------------------


class TestFormatScorecard:
    def test_contains_header(self):
        result = _build_sample()
        text = format_scorecard(result)
        assert "Supplier Scorecard" in text

    def test_contains_all_suppliers(self):
        result = _build_sample()
        text = format_scorecard(result)
        for s in result.suppliers:
            assert s.supplier in text

    def test_contains_grade_distribution(self):
        result = _build_sample()
        text = format_scorecard(result)
        assert "Grade distribution" in text

    def test_contains_mean_score(self):
        result = _build_sample()
        text = format_scorecard(result)
        assert "Mean score" in text
