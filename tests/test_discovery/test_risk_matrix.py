"""Tests for risk_matrix discovery module."""

from business_brain.discovery.risk_matrix import (
    CategoryRiskCount,
    HeatmapCell,
    OwnerRiskCount,
    PeriodRiskTrend,
    RiskAssessmentResult,
    RiskExposureItem,
    RiskExposureResult,
    RiskHeatmapResult,
    RiskItem,
    RiskTrendResult,
    _classify_risk,
    _parse_level,
    _safe_float,
    analyze_risk_trends,
    assess_risks,
    compute_risk_exposure,
    compute_risk_heatmap,
    format_risk_report,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("99.5") == 99.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None


class TestParseLevel:
    def test_numeric_int(self):
        assert _parse_level(3) == 3

    def test_numeric_float(self):
        assert _parse_level(2.7) == 3

    def test_numeric_string(self):
        assert _parse_level("4") == 4

    def test_clamp_low(self):
        assert _parse_level(0) == 1

    def test_clamp_high(self):
        assert _parse_level(10) == 5

    def test_text_low(self):
        assert _parse_level("Low") == 1

    def test_text_medium(self):
        assert _parse_level("Medium") == 2

    def test_text_high(self):
        assert _parse_level("High") == 3

    def test_text_very_high(self):
        assert _parse_level("Very High") == 4

    def test_text_critical(self):
        assert _parse_level("Critical") == 5

    def test_text_case_insensitive(self):
        assert _parse_level("HIGH") == 3

    def test_text_with_spaces(self):
        assert _parse_level("  very high  ") == 4

    def test_none(self):
        assert _parse_level(None) is None

    def test_invalid_text(self):
        assert _parse_level("unknown") is None


class TestClassifyRisk:
    def test_critical_at_20(self):
        assert _classify_risk(20) == "Critical"

    def test_critical_at_25(self):
        assert _classify_risk(25) == "Critical"

    def test_high_at_12(self):
        assert _classify_risk(12) == "High"

    def test_high_at_19(self):
        assert _classify_risk(19) == "High"

    def test_medium_at_6(self):
        assert _classify_risk(6) == "Medium"

    def test_medium_at_11(self):
        assert _classify_risk(11) == "Medium"

    def test_low_at_5(self):
        assert _classify_risk(5) == "Low"

    def test_low_at_1(self):
        assert _classify_risk(1) == "Low"


# ---------------------------------------------------------------------------
# Test assess_risks
# ---------------------------------------------------------------------------


class TestAssessRisks:
    def test_empty_rows(self):
        assert assess_risks([], "risk", "likelihood", "impact") is None

    def test_all_null_data(self):
        rows = [
            {"risk": None, "likelihood": None, "impact": None},
            {"risk": None, "likelihood": None, "impact": None},
        ]
        assert assess_risks(rows, "risk", "likelihood", "impact") is None

    def test_single_row(self):
        rows = [{"risk": "R1", "likelihood": 3, "impact": 4}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result is not None
        assert result.total_count == 1
        assert result.risks[0].risk_score == 12
        assert result.risks[0].risk_level == "High"

    def test_multiple_risks(self):
        rows = [
            {"risk": "R1", "likelihood": 5, "impact": 5},  # 25 -> Critical
            {"risk": "R2", "likelihood": 3, "impact": 4},  # 12 -> High
            {"risk": "R3", "likelihood": 2, "impact": 3},  # 6  -> Medium
            {"risk": "R4", "likelihood": 1, "impact": 1},  # 1  -> Low
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result is not None
        assert result.total_count == 4
        assert result.critical_count == 1
        assert result.high_count == 1
        assert result.medium_count == 1
        assert result.low_count == 1

    def test_sorted_by_score_desc(self):
        rows = [
            {"risk": "Low", "likelihood": 1, "impact": 1},
            {"risk": "High", "likelihood": 5, "impact": 5},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result.risks[0].name == "High"
        assert result.risks[1].name == "Low"

    def test_text_likelihood_and_impact(self):
        rows = [
            {"risk": "R1", "likelihood": "High", "impact": "Critical"},
            {"risk": "R2", "likelihood": "Low", "impact": "Medium"},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result is not None
        assert result.risks[0].likelihood == 3
        assert result.risks[0].impact == 5
        assert result.risks[0].risk_score == 15

    def test_mixed_text_and_numeric(self):
        rows = [
            {"risk": "R1", "likelihood": 4, "impact": "Very High"},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result is not None
        assert result.risks[0].likelihood == 4
        assert result.risks[0].impact == 4
        assert result.risks[0].risk_score == 16

    def test_with_category_column(self):
        rows = [
            {"risk": "R1", "likelihood": 5, "impact": 5, "cat": "Ops"},
            {"risk": "R2", "likelihood": 3, "impact": 3, "cat": "Ops"},
            {"risk": "R3", "likelihood": 2, "impact": 2, "cat": "Finance"},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact", category_column="cat")
        assert result is not None
        assert len(result.by_category) == 2
        cats = {c.category for c in result.by_category}
        assert "Ops" in cats
        assert "Finance" in cats

    def test_category_avg_score(self):
        rows = [
            {"risk": "R1", "likelihood": 4, "impact": 4, "cat": "A"},
            {"risk": "R2", "likelihood": 2, "impact": 2, "cat": "A"},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact", category_column="cat")
        cat_a = [c for c in result.by_category if c.category == "A"][0]
        # (16 + 4) / 2 = 10.0
        assert cat_a.avg_score == 10.0
        assert cat_a.count == 2

    def test_with_owner_column(self):
        rows = [
            {"risk": "R1", "likelihood": 5, "impact": 5, "owner": "Alice"},   # 25 -> Critical
            {"risk": "R2", "likelihood": 3, "impact": 3, "owner": "Alice"},   # 9  -> Medium
            {"risk": "R3", "likelihood": 2, "impact": 2, "owner": "Bob"},     # 4  -> Low
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact", owner_column="owner")
        assert result is not None
        assert len(result.by_owner) == 2
        alice = [o for o in result.by_owner if o.owner == "Alice"][0]
        assert alice.count == 2
        assert alice.critical_count == 1

    def test_with_both_category_and_owner(self):
        rows = [
            {"risk": "R1", "likelihood": 5, "impact": 5, "cat": "Ops", "owner": "Alice"},
            {"risk": "R2", "likelihood": 2, "impact": 2, "cat": "Fin", "owner": "Bob"},
        ]
        result = assess_risks(
            rows, "risk", "likelihood", "impact",
            category_column="cat", owner_column="owner"
        )
        assert len(result.by_category) == 2
        assert len(result.by_owner) == 2

    def test_skips_missing_name(self):
        rows = [
            {"risk": None, "likelihood": 5, "impact": 5},
            {"risk": "R2", "likelihood": 3, "impact": 3},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result.total_count == 1

    def test_skips_invalid_likelihood(self):
        rows = [
            {"risk": "R1", "likelihood": "bad", "impact": 3},
            {"risk": "R2", "likelihood": 3, "impact": 3},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result.total_count == 1

    def test_summary_text(self):
        rows = [{"risk": "R1", "likelihood": 3, "impact": 3}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert "Risk assessment" in result.summary
        assert "1 risks evaluated" in result.summary

    def test_all_critical_risks(self):
        rows = [
            {"risk": "R1", "likelihood": 5, "impact": 4},  # 20 -> Critical
            {"risk": "R2", "likelihood": 4, "impact": 5},  # 20 -> Critical
            {"risk": "R3", "likelihood": 5, "impact": 5},  # 25 -> Critical
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        # 5*4=20 -> Critical, 4*5=20 -> Critical, 5*5=25 -> Critical
        assert result.critical_count == 3
        assert result.high_count == 0

    def test_boundary_score_19(self):
        # 19 should be High, not Critical
        rows = [{"risk": "R1", "likelihood": 4, "impact": 4}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        # 4*4=16, which is High
        assert result.risks[0].risk_level == "High"

    def test_boundary_score_20(self):
        rows = [{"risk": "R1", "likelihood": 5, "impact": 4}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        # 5*4=20 -> Critical
        assert result.risks[0].risk_level == "Critical"

    def test_boundary_score_12(self):
        rows = [{"risk": "R1", "likelihood": 3, "impact": 4}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        # 3*4=12 -> High
        assert result.risks[0].risk_level == "High"

    def test_boundary_score_11(self):
        rows = [{"risk": "R1", "likelihood": 3, "impact": 3}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        # 3*3=9 -> Medium
        assert result.risks[0].risk_level == "Medium"

    def test_boundary_score_6(self):
        rows = [{"risk": "R1", "likelihood": 2, "impact": 3}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        # 2*3=6 -> Medium
        assert result.risks[0].risk_level == "Medium"

    def test_boundary_score_5(self):
        rows = [{"risk": "R1", "likelihood": 1, "impact": 5}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        # 1*5=5 -> Low
        assert result.risks[0].risk_level == "Low"

    def test_uncategorized_when_category_none(self):
        rows = [
            {"risk": "R1", "likelihood": 3, "impact": 3, "cat": None},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact", category_column="cat")
        assert result.by_category[0].category == "Uncategorized"

    def test_unassigned_when_owner_none(self):
        rows = [
            {"risk": "R1", "likelihood": 3, "impact": 3, "owner": None},
        ]
        result = assess_risks(rows, "risk", "likelihood", "impact", owner_column="owner")
        assert result.by_owner[0].owner == "Unassigned"

    def test_no_category_or_owner_by_default(self):
        rows = [{"risk": "R1", "likelihood": 3, "impact": 3}]
        result = assess_risks(rows, "risk", "likelihood", "impact")
        assert result.by_category == []
        assert result.by_owner == []


# ---------------------------------------------------------------------------
# Test compute_risk_heatmap
# ---------------------------------------------------------------------------


class TestComputeRiskHeatmap:
    def test_empty_rows(self):
        assert compute_risk_heatmap([], "likelihood", "impact") is None

    def test_all_null_data(self):
        rows = [{"likelihood": None, "impact": None}]
        assert compute_risk_heatmap(rows, "likelihood", "impact") is None

    def test_single_risk(self):
        rows = [{"likelihood": 3, "impact": 4}]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert result is not None
        assert result.total_risks == 1
        assert result.matrix[2][3] == 1  # likelihood=3 -> index 2, impact=4 -> index 3

    def test_matrix_dimensions(self):
        rows = [{"likelihood": 1, "impact": 1}]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert len(result.matrix) == 5
        for row in result.matrix:
            assert len(row) == 5

    def test_multiple_risks_same_cell(self):
        rows = [
            {"likelihood": 2, "impact": 3},
            {"likelihood": 2, "impact": 3},
            {"likelihood": 2, "impact": 3},
        ]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert result.matrix[1][2] == 3  # likelihood=2 -> index 1, impact=3 -> index 2
        assert result.total_risks == 3

    def test_multiple_risks_different_cells(self):
        rows = [
            {"likelihood": 1, "impact": 1},
            {"likelihood": 5, "impact": 5},
        ]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert result.matrix[0][0] == 1
        assert result.matrix[4][4] == 1
        assert result.total_risks == 2

    def test_hotspots_sorted_by_score(self):
        rows = [
            {"likelihood": 1, "impact": 1},
            {"likelihood": 5, "impact": 5},
            {"likelihood": 3, "impact": 3},
        ]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert result.hotspots[0].risk_score == 25  # 5*5
        assert result.hotspots[-1].risk_score == 1  # 1*1

    def test_hotspots_only_nonzero_cells(self):
        rows = [
            {"likelihood": 2, "impact": 2},
            {"likelihood": 4, "impact": 4},
        ]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert len(result.hotspots) == 2
        for h in result.hotspots:
            assert h.count > 0

    def test_text_values_converted(self):
        rows = [
            {"likelihood": "Low", "impact": "High"},
        ]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert result is not None
        # Low=1, High=3
        assert result.matrix[0][2] == 1

    def test_summary_present(self):
        rows = [{"likelihood": 3, "impact": 3}]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert "heatmap" in result.summary.lower()

    def test_hotspot_cell_fields(self):
        rows = [{"likelihood": 4, "impact": 5}]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        cell = result.hotspots[0]
        assert cell.likelihood == 4
        assert cell.impact == 5
        assert cell.count == 1
        assert cell.risk_score == 20

    def test_all_cells_empty_except_parsed(self):
        rows = [{"likelihood": 3, "impact": 3}]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        total_in_matrix = sum(sum(row) for row in result.matrix)
        assert total_in_matrix == 1

    def test_skips_invalid_values(self):
        rows = [
            {"likelihood": "bad", "impact": "bad"},
            {"likelihood": 2, "impact": 3},
        ]
        result = compute_risk_heatmap(rows, "likelihood", "impact")
        assert result.total_risks == 1


# ---------------------------------------------------------------------------
# Test analyze_risk_trends
# ---------------------------------------------------------------------------


class TestAnalyzeRiskTrends:
    def test_empty_rows(self):
        assert analyze_risk_trends([], "risk", "score", "date") is None

    def test_all_null_data(self):
        rows = [{"risk": None, "score": None, "date": None}]
        assert analyze_risk_trends(rows, "risk", "score", "date") is None

    def test_single_period(self):
        rows = [
            {"risk": "R1", "score": 10, "date": "2024-01-15"},
            {"risk": "R2", "score": 20, "date": "2024-01-20"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert result is not None
        assert len(result.periods) == 1
        assert result.periods[0].period == "2024-01"
        assert result.periods[0].avg_score == 15.0
        assert result.periods[0].max_score == 20.0
        assert result.periods[0].risk_count == 2
        assert result.trend_direction == "stable"

    def test_worsening_trend(self):
        rows = [
            {"risk": "R1", "score": 5, "date": "2024-01-01"},
            {"risk": "R2", "score": 6, "date": "2024-01-15"},
            {"risk": "R3", "score": 15, "date": "2024-03-01"},
            {"risk": "R4", "score": 20, "date": "2024-03-15"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert result.trend_direction == "worsening"
        assert result.avg_score_change > 0

    def test_improving_trend(self):
        rows = [
            {"risk": "R1", "score": 20, "date": "2024-01-01"},
            {"risk": "R2", "score": 18, "date": "2024-01-15"},
            {"risk": "R3", "score": 5, "date": "2024-03-01"},
            {"risk": "R4", "score": 3, "date": "2024-03-15"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert result.trend_direction == "improving"
        assert result.avg_score_change < 0

    def test_stable_trend(self):
        rows = [
            {"risk": "R1", "score": 10, "date": "2024-01-01"},
            {"risk": "R2", "score": 10, "date": "2024-02-01"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert result.trend_direction == "stable"

    def test_periods_sorted(self):
        rows = [
            {"risk": "R1", "score": 10, "date": "2024-03-01"},
            {"risk": "R2", "score": 10, "date": "2024-01-01"},
            {"risk": "R3", "score": 10, "date": "2024-02-01"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        periods = [p.period for p in result.periods]
        assert periods == ["2024-01", "2024-02", "2024-03"]

    def test_with_category_column(self):
        rows = [
            {"risk": "R1", "score": 5, "date": "2024-01-01", "cat": "Ops"},
            {"risk": "R2", "score": 15, "date": "2024-03-01", "cat": "Ops"},
            {"risk": "R3", "score": 20, "date": "2024-01-01", "cat": "Fin"},
            {"risk": "R4", "score": 5, "date": "2024-03-01", "cat": "Fin"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date", category_column="cat")
        assert "Ops" in result.by_category_trend
        assert "Fin" in result.by_category_trend
        assert result.by_category_trend["Ops"] == "worsening"
        assert result.by_category_trend["Fin"] == "improving"

    def test_category_stable(self):
        rows = [
            {"risk": "R1", "score": 10, "date": "2024-01-01", "cat": "A"},
            {"risk": "R2", "score": 10, "date": "2024-03-01", "cat": "A"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date", category_column="cat")
        assert result.by_category_trend["A"] == "stable"

    def test_summary_present(self):
        rows = [{"risk": "R1", "score": 10, "date": "2024-01-01"}]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert "trend" in result.summary.lower()

    def test_skips_invalid_dates(self):
        rows = [
            {"risk": "R1", "score": 10, "date": "not-a-date"},
            {"risk": "R2", "score": 15, "date": "2024-01-01"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert result is not None
        assert result.periods[0].risk_count == 1

    def test_skips_invalid_scores(self):
        rows = [
            {"risk": "R1", "score": "bad", "date": "2024-01-01"},
            {"risk": "R2", "score": 15, "date": "2024-01-01"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert result.periods[0].risk_count == 1

    def test_no_category_trend_by_default(self):
        rows = [{"risk": "R1", "score": 10, "date": "2024-01-01"}]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        assert result.by_category_trend == {}

    def test_avg_score_change_value(self):
        rows = [
            {"risk": "R1", "score": 10, "date": "2024-01-01"},
            {"risk": "R2", "score": 20, "date": "2024-02-01"},
        ]
        result = analyze_risk_trends(rows, "risk", "score", "date")
        # first avg = 10, last avg = 20, change = 10
        assert result.avg_score_change == 10.0


# ---------------------------------------------------------------------------
# Test compute_risk_exposure
# ---------------------------------------------------------------------------


class TestComputeRiskExposure:
    def test_empty_rows(self):
        assert compute_risk_exposure([], "risk", "impact_val", "prob") is None

    def test_all_null_data(self):
        rows = [{"risk": None, "impact_val": None, "prob": None}]
        assert compute_risk_exposure(rows, "risk", "impact_val", "prob") is None

    def test_single_risk(self):
        rows = [{"risk": "R1", "impact_val": 100000, "prob": 0.5}]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert result is not None
        assert result.total_expected_loss == 50000.0
        assert result.max_expected_loss == 50000.0
        assert result.avg_expected_loss == 50000.0
        assert len(result.top_risks) == 1

    def test_multiple_risks(self):
        rows = [
            {"risk": "R1", "impact_val": 100000, "prob": 0.8},
            {"risk": "R2", "impact_val": 50000, "prob": 0.5},
            {"risk": "R3", "impact_val": 200000, "prob": 0.1},
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert result is not None
        # R1: 80000, R2: 25000, R3: 20000
        assert result.total_expected_loss == 125000.0
        assert result.max_expected_loss == 80000.0

    def test_top_5_limit(self):
        rows = [
            {"risk": f"R{i}", "impact_val": 1000 * (10 - i), "prob": 0.5}
            for i in range(8)
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert len(result.top_risks) == 5

    def test_top_risks_sorted_desc(self):
        rows = [
            {"risk": "Low", "impact_val": 1000, "prob": 0.1},
            {"risk": "High", "impact_val": 100000, "prob": 0.9},
            {"risk": "Med", "impact_val": 50000, "prob": 0.3},
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert result.top_risks[0].name == "High"

    def test_risk_concentration(self):
        rows = [
            {"risk": "R1", "impact_val": 100000, "prob": 0.9},  # 90000
            {"risk": "R2", "impact_val": 100000, "prob": 0.8},  # 80000
            {"risk": "R3", "impact_val": 100000, "prob": 0.7},  # 70000
            {"risk": "R4", "impact_val": 10000, "prob": 0.1},   # 1000
            {"risk": "R5", "impact_val": 10000, "prob": 0.1},   # 1000
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        # total = 242000, top3 = 240000 -> 99.17%
        assert result.risk_concentration_pct > 90

    def test_skips_invalid_impact(self):
        rows = [
            {"risk": "R1", "impact_val": "bad", "prob": 0.5},
            {"risk": "R2", "impact_val": 10000, "prob": 0.5},
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert result.total_expected_loss == 5000.0

    def test_skips_missing_risk_name(self):
        rows = [
            {"risk": None, "impact_val": 10000, "prob": 0.5},
            {"risk": "R2", "impact_val": 10000, "prob": 0.5},
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert len(result.top_risks) == 1

    def test_summary_present(self):
        rows = [{"risk": "R1", "impact_val": 100000, "prob": 0.5}]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert "exposure" in result.summary.lower()

    def test_zero_probability(self):
        rows = [{"risk": "R1", "impact_val": 100000, "prob": 0}]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert result.total_expected_loss == 0.0

    def test_full_probability(self):
        rows = [{"risk": "R1", "impact_val": 100000, "prob": 1.0}]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert result.total_expected_loss == 100000.0

    def test_avg_expected_loss(self):
        rows = [
            {"risk": "R1", "impact_val": 10000, "prob": 0.5},
            {"risk": "R2", "impact_val": 20000, "prob": 0.5},
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        # R1: 5000, R2: 10000; avg = 7500
        assert result.avg_expected_loss == 7500.0

    def test_concentration_with_few_risks(self):
        # When there are fewer than 3 risks, top 3 = all of them
        rows = [
            {"risk": "R1", "impact_val": 10000, "prob": 0.5},
            {"risk": "R2", "impact_val": 20000, "prob": 0.5},
        ]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        assert result.risk_concentration_pct == 100.0

    def test_exposure_item_fields(self):
        rows = [{"risk": "R1", "impact_val": 50000, "prob": 0.3}]
        result = compute_risk_exposure(rows, "risk", "impact_val", "prob")
        item = result.top_risks[0]
        assert item.name == "R1"
        assert item.impact_value == 50000.0
        assert item.probability == 0.3
        assert item.expected_loss == 15000.0


# ---------------------------------------------------------------------------
# Test format_risk_report
# ---------------------------------------------------------------------------


class TestFormatRiskReport:
    def _make_assessment(self):
        risks = [
            RiskItem("R1", 5, 5, 25, "Critical", "Ops", "Alice"),
            RiskItem("R2", 2, 2, 4, "Low", None, None),
        ]
        return RiskAssessmentResult(
            risks=risks,
            total_count=2,
            critical_count=1,
            high_count=0,
            medium_count=0,
            low_count=1,
            by_category=[CategoryRiskCount("Ops", 1, 25.0)],
            by_owner=[OwnerRiskCount("Alice", 1, 1)],
            summary="test summary",
        )

    def _make_heatmap(self):
        matrix = [[0] * 5 for _ in range(5)]
        matrix[4][4] = 1
        return RiskHeatmapResult(
            matrix=matrix,
            hotspots=[HeatmapCell(5, 5, 1, 25)],
            total_risks=1,
            summary="heatmap summary",
        )

    def _make_trends(self):
        return RiskTrendResult(
            periods=[PeriodRiskTrend("2024-01", 10.0, 15.0, 3)],
            trend_direction="stable",
            avg_score_change=0.0,
            by_category_trend={"Ops": "stable"},
            summary="trend summary",
        )

    def _make_exposure(self):
        return RiskExposureResult(
            total_expected_loss=50000.0,
            top_risks=[RiskExposureItem("R1", 100000, 0.5, 50000.0)],
            risk_concentration_pct=100.0,
            avg_expected_loss=50000.0,
            max_expected_loss=50000.0,
            summary="exposure summary",
        )

    def test_no_sections(self):
        report = format_risk_report()
        assert "Risk Assessment Report" in report
        assert "No analysis data provided" in report

    def test_assessment_only(self):
        report = format_risk_report(assessment=self._make_assessment())
        assert "Risk Assessment" in report
        assert "Total risks: 2" in report
        assert "R1" in report
        assert "By category:" in report
        assert "By owner:" in report
        assert "No analysis data provided" not in report

    def test_heatmap_only(self):
        report = format_risk_report(heatmap=self._make_heatmap())
        assert "Heatmap" in report
        assert "Total risks mapped: 1" in report
        assert "No analysis data provided" not in report

    def test_trends_only(self):
        report = format_risk_report(trends=self._make_trends())
        assert "Risk Trends" in report
        assert "stable" in report
        assert "No analysis data provided" not in report

    def test_exposure_only(self):
        report = format_risk_report(exposure=self._make_exposure())
        assert "Risk Exposure" in report
        assert "50,000.00" in report
        assert "No analysis data provided" not in report

    def test_all_sections(self):
        report = format_risk_report(
            assessment=self._make_assessment(),
            heatmap=self._make_heatmap(),
            trends=self._make_trends(),
            exposure=self._make_exposure(),
        )
        assert "Risk Assessment" in report
        assert "Heatmap" in report
        assert "Risk Trends" in report
        assert "Risk Exposure" in report
        assert "No analysis data provided" not in report

    def test_report_header(self):
        report = format_risk_report()
        lines = report.split("\n")
        assert lines[0] == "Risk Assessment Report"
        assert "=" in lines[1]

    def test_assessment_risk_detail(self):
        report = format_risk_report(assessment=self._make_assessment())
        assert "L=5 I=5 score=25 [Critical] [Ops] (owner: Alice)" in report

    def test_heatmap_matrix_display(self):
        report = format_risk_report(heatmap=self._make_heatmap())
        assert "Impact ->" in report

    def test_trends_category_display(self):
        report = format_risk_report(trends=self._make_trends())
        assert "Ops: stable" in report

    def test_exposure_top_risks_display(self):
        report = format_risk_report(exposure=self._make_exposure())
        assert "R1" in report
        assert "100,000.00" in report


# ---------------------------------------------------------------------------
# Test dataclass instantiation
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_risk_item(self):
        r = RiskItem("R1", 3, 4, 12, "High", "Ops", "Alice")
        assert r.name == "R1"
        assert r.likelihood == 3
        assert r.impact == 4
        assert r.risk_score == 12
        assert r.risk_level == "High"
        assert r.category == "Ops"
        assert r.owner == "Alice"

    def test_category_risk_count(self):
        c = CategoryRiskCount("Ops", 5, 12.5)
        assert c.category == "Ops"
        assert c.count == 5
        assert c.avg_score == 12.5

    def test_owner_risk_count(self):
        o = OwnerRiskCount("Alice", 3, 1)
        assert o.owner == "Alice"
        assert o.count == 3
        assert o.critical_count == 1

    def test_heatmap_cell(self):
        h = HeatmapCell(3, 4, 2, 12)
        assert h.likelihood == 3
        assert h.impact == 4
        assert h.count == 2
        assert h.risk_score == 12

    def test_period_risk_trend(self):
        p = PeriodRiskTrend("2024-01", 10.5, 20.0, 5)
        assert p.period == "2024-01"
        assert p.avg_score == 10.5
        assert p.max_score == 20.0
        assert p.risk_count == 5

    def test_risk_exposure_item(self):
        e = RiskExposureItem("R1", 100000, 0.5, 50000)
        assert e.name == "R1"
        assert e.impact_value == 100000
        assert e.probability == 0.5
        assert e.expected_loss == 50000

    def test_risk_assessment_result(self):
        r = RiskAssessmentResult([], 0, 0, 0, 0, 0, [], [], "")
        assert r.total_count == 0

    def test_risk_heatmap_result(self):
        r = RiskHeatmapResult([[0]*5]*5, [], 0, "")
        assert r.total_risks == 0

    def test_risk_trend_result(self):
        r = RiskTrendResult([], "stable", 0.0, {}, "")
        assert r.trend_direction == "stable"

    def test_risk_exposure_result(self):
        r = RiskExposureResult(0.0, [], 0.0, 0.0, 0.0, "")
        assert r.total_expected_loss == 0.0
