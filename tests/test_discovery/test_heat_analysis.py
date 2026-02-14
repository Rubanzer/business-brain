"""Tests for heat-by-heat analysis module for steel manufacturing."""

import math
import statistics

import pytest

from business_brain.discovery.heat_analysis import (
    ChemistryResult,
    ElementStats,
    GradeAnomaly,
    GradeInfo,
    GradeWiseResult,
    HeatAnalysisResult,
    analyze_chemistry,
    analyze_heats,
    detect_grade_anomalies,
    format_heat_report,
    grade_wise_analysis,
)


# ===================================================================
# analyze_heats Tests
# ===================================================================


class TestAnalyzeHeats:
    """Tests for the analyze_heats function."""

    # --- None / empty returns ---

    def test_empty_rows_returns_none(self):
        assert analyze_heats([], "heat_id", "weight") is None

    def test_all_none_heat_ids_returns_none(self):
        rows = [{"heat_id": None, "weight": 100}]
        assert analyze_heats(rows, "heat_id", "weight") is None

    def test_all_none_weights_returns_none(self):
        rows = [{"heat_id": "H1", "weight": None}]
        assert analyze_heats(rows, "heat_id", "weight") is None

    def test_both_none_returns_none(self):
        rows = [{"heat_id": None, "weight": None}]
        assert analyze_heats(rows, "heat_id", "weight") is None

    def test_missing_columns_returns_none(self):
        """Rows that lack the heat or weight column entirely."""
        rows = [{"other_col": "X"}]
        assert analyze_heats(rows, "heat_id", "weight") is None

    # --- Basic heat counting ---

    def test_basic_heat_counting_three_heats(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 120},
            {"heat_id": "H3", "weight": 80},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result is not None
        assert result.total_heats == 3

    def test_single_heat(self):
        rows = [{"heat_id": "H1", "weight": 150}]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result is not None
        assert result.total_heats == 1
        assert result.total_weight == 150.0
        assert result.avg_weight_per_heat == 150.0
        assert result.min_weight == 150.0
        assert result.max_weight == 150.0

    def test_single_heat_std_is_zero(self):
        rows = [{"heat_id": "H1", "weight": 100}]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.weight_std == 0.0

    def test_aggregates_multiple_rows_per_heat(self):
        """Multiple rows with the same heat ID should sum weights."""
        rows = [
            {"heat_id": "H1", "weight": 50},
            {"heat_id": "H1", "weight": 50},
            {"heat_id": "H2", "weight": 80},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.total_heats == 2
        assert result.total_weight == 180.0
        # H1=100, H2=80 -> avg=90
        assert result.avg_weight_per_heat == 90.0

    def test_skips_non_numeric_weight(self):
        rows = [
            {"heat_id": "H1", "weight": "bad"},
            {"heat_id": "H2", "weight": 100},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result is not None
        assert result.total_heats == 1
        assert result.total_weight == 100.0

    def test_skips_rows_with_none_heat_id_but_keeps_valid(self):
        rows = [
            {"heat_id": None, "weight": 200},
            {"heat_id": "H2", "weight": 100},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result is not None
        assert result.total_heats == 1
        assert result.total_weight == 100.0

    # --- Weight statistics ---

    def test_weight_stats_total(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 120},
            {"heat_id": "H3", "weight": 80},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.total_weight == 300.0

    def test_weight_stats_avg(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 120},
            {"heat_id": "H3", "weight": 80},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.avg_weight_per_heat == 100.0

    def test_weight_stats_min_max(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 120},
            {"heat_id": "H3", "weight": 80},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.min_weight == 80.0
        assert result.max_weight == 120.0

    def test_weight_stats_std(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 120},
            {"heat_id": "H3", "weight": 80},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        expected_std = statistics.stdev([100.0, 120.0, 80.0])
        assert result.weight_std == round(expected_std, 4)

    def test_weight_values_rounded_to_four_decimals(self):
        rows = [
            {"heat_id": "H1", "weight": 33.33333333},
            {"heat_id": "H2", "weight": 66.66666666},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.total_weight == round(33.33333333 + 66.66666666, 4)
        assert result.avg_weight_per_heat == round((33.33333333 + 66.66666666) / 2, 4)

    def test_numeric_string_weights_are_converted(self):
        """Weights provided as string numbers should be parsed."""
        rows = [
            {"heat_id": "H1", "weight": "100.5"},
            {"heat_id": "H2", "weight": "200.5"},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result is not None
        assert result.total_heats == 2
        assert result.total_weight == 301.0

    # --- Grade distribution ---

    def test_grade_distribution(self):
        rows = [
            {"heat_id": "H1", "weight": 100, "grade": "A"},
            {"heat_id": "H2", "weight": 110, "grade": "A"},
            {"heat_id": "H3", "weight": 90, "grade": "B"},
        ]
        result = analyze_heats(rows, "heat_id", "weight", grade_column="grade")
        assert result is not None
        assert result.grade_distribution == {"A": 2, "B": 1}

    def test_no_grade_column_returns_none_distribution(self):
        rows = [{"heat_id": "H1", "weight": 100}]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.grade_distribution is None

    def test_grade_column_specified_but_value_none(self):
        """Grade column is specified but all values are None."""
        rows = [
            {"heat_id": "H1", "weight": 100, "grade": None},
            {"heat_id": "H2", "weight": 200, "grade": None},
        ]
        result = analyze_heats(rows, "heat_id", "weight", grade_column="grade")
        assert result is not None
        # No valid grades found, so distribution should be None
        assert result.grade_distribution is None

    def test_grade_distribution_with_mixed_none(self):
        """Some rows have grade, some do not."""
        rows = [
            {"heat_id": "H1", "weight": 100, "grade": "A"},
            {"heat_id": "H2", "weight": 200, "grade": None},
            {"heat_id": "H3", "weight": 150, "grade": "A"},
        ]
        result = analyze_heats(rows, "heat_id", "weight", grade_column="grade")
        assert result is not None
        # H1 and H3 are grade A; H2 has None grade so not counted in distribution
        assert result.grade_distribution == {"A": 2}

    def test_grade_last_value_wins_for_heat(self):
        """When a heat appears multiple times, the last grade assignment wins."""
        rows = [
            {"heat_id": "H1", "weight": 50, "grade": "A"},
            {"heat_id": "H1", "weight": 50, "grade": "B"},
        ]
        result = analyze_heats(rows, "heat_id", "weight", grade_column="grade")
        assert result is not None
        assert result.total_heats == 1
        # Last grade assignment for H1 should be "B"
        assert result.grade_distribution == {"B": 1}

    # --- Heats per period ---

    def test_heats_per_period(self):
        rows = [
            {"heat_id": "H1", "weight": 100, "shift": "morning"},
            {"heat_id": "H2", "weight": 110, "shift": "morning"},
            {"heat_id": "H3", "weight": 90, "shift": "night"},
        ]
        result = analyze_heats(rows, "heat_id", "weight", time_column="shift")
        assert result is not None
        assert result.heats_per_period == {"morning": 2, "night": 1}

    def test_no_time_column_returns_none_periods(self):
        rows = [{"heat_id": "H1", "weight": 100}]
        result = analyze_heats(rows, "heat_id", "weight")
        assert result.heats_per_period is None

    def test_heats_per_period_with_none_time_values(self):
        """Time column specified but all None values."""
        rows = [
            {"heat_id": "H1", "weight": 100, "shift": None},
        ]
        result = analyze_heats(rows, "heat_id", "weight", time_column="shift")
        assert result is not None
        assert result.heats_per_period is None

    def test_heats_per_period_last_period_wins(self):
        """Multiple rows for same heat: last time period overwrites."""
        rows = [
            {"heat_id": "H1", "weight": 50, "shift": "morning"},
            {"heat_id": "H1", "weight": 50, "shift": "evening"},
        ]
        result = analyze_heats(rows, "heat_id", "weight", time_column="shift")
        assert result.heats_per_period == {"evening": 1}

    # --- Summary ---

    def test_summary_contains_key_info(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 200},
        ]
        result = analyze_heats(rows, "heat_id", "weight")
        assert "2 heats" in result.summary
        assert "300" in result.summary

    def test_summary_mentions_dominant_grade_when_present(self):
        rows = [
            {"heat_id": "H1", "weight": 100, "grade": "A"},
            {"heat_id": "H2", "weight": 200, "grade": "A"},
            {"heat_id": "H3", "weight": 150, "grade": "B"},
        ]
        result = analyze_heats(rows, "heat_id", "weight", grade_column="grade")
        assert "Dominant grade: A" in result.summary

    # --- Return type ---

    def test_returns_heat_analysis_result_instance(self):
        rows = [{"heat_id": "H1", "weight": 100}]
        result = analyze_heats(rows, "heat_id", "weight")
        assert isinstance(result, HeatAnalysisResult)

    # --- Both grade and time ---

    def test_grade_and_time_together(self):
        rows = [
            {"heat_id": "H1", "weight": 100, "grade": "A", "month": "Jan"},
            {"heat_id": "H2", "weight": 120, "grade": "B", "month": "Jan"},
            {"heat_id": "H3", "weight": 90, "grade": "A", "month": "Feb"},
        ]
        result = analyze_heats(
            rows, "heat_id", "weight", grade_column="grade", time_column="month"
        )
        assert result is not None
        assert result.grade_distribution == {"A": 2, "B": 1}
        assert result.heats_per_period == {"Jan": 2, "Feb": 1}


# ===================================================================
# analyze_chemistry Tests
# ===================================================================


class TestAnalyzeChemistry:
    """Tests for the analyze_chemistry function."""

    # --- None / empty returns ---

    def test_empty_rows_returns_none(self):
        assert analyze_chemistry([], "heat", ["C"]) is None

    def test_empty_elements_returns_none(self):
        rows = [{"heat": "H1", "C": 0.05}]
        assert analyze_chemistry(rows, "heat", []) is None

    def test_no_valid_heat_ids_returns_none(self):
        rows = [{"heat": None, "C": 0.05}]
        assert analyze_chemistry(rows, "heat", ["C"]) is None

    def test_all_element_values_none_returns_none(self):
        rows = [{"heat": "H1", "C": None}]
        assert analyze_chemistry(rows, "heat", ["C"]) is None

    def test_missing_element_column_in_rows_returns_none(self):
        """Element column specified but not present in any row."""
        rows = [{"heat": "H1", "other": 0.05}]
        assert analyze_chemistry(rows, "heat", ["C"]) is None

    # --- Basic chemistry stats ---

    def test_basic_chemistry_stats_mean(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
            {"heat": "H3", "C": 0.04},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert result is not None
        c_stats = result.elements[0]
        expected_mean = round(statistics.mean([0.05, 0.06, 0.04]), 6)
        assert c_stats.mean == expected_mean

    def test_basic_chemistry_stats_std(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
            {"heat": "H3", "C": 0.04},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        c_stats = result.elements[0]
        expected_std = round(statistics.stdev([0.05, 0.06, 0.04]), 6)
        assert c_stats.std == expected_std

    def test_basic_chemistry_stats_min_max(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
            {"heat": "H3", "C": 0.04},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        c_stats = result.elements[0]
        assert c_stats.min == 0.04
        assert c_stats.max == 0.06

    def test_cv_pct_calculation(self):
        rows = [
            {"heat": "H1", "C": 0.10},
            {"heat": "H2", "C": 0.20},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        c_stats = result.elements[0]
        expected_mean = statistics.mean([0.10, 0.20])
        expected_std = statistics.stdev([0.10, 0.20])
        expected_cv = round(expected_std / expected_mean * 100, 2)
        assert c_stats.cv_pct == expected_cv

    def test_cv_pct_zero_std(self):
        """When all values are identical, CV should be 0."""
        rows = [
            {"heat": "H1", "C": 0.10},
            {"heat": "H2", "C": 0.10},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert result.elements[0].cv_pct == 0.0

    def test_single_heat_std_is_zero(self):
        rows = [{"heat": "H1", "C": 0.05}]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert result.elements[0].std == 0.0

    def test_heat_count(self):
        rows = [
            {"heat": "H1", "C": 0.05, "Mn": 1.2},
            {"heat": "H2", "C": 0.06, "Mn": 1.3},
            {"heat": "H3", "C": 0.04, "Mn": 1.1},
        ]
        result = analyze_chemistry(rows, "heat", ["C", "Mn"])
        assert result.heat_count == 3

    def test_multiple_elements(self):
        rows = [
            {"heat": "H1", "C": 0.05, "Mn": 1.2},
            {"heat": "H2", "C": 0.06, "Mn": 1.3},
        ]
        result = analyze_chemistry(rows, "heat", ["C", "Mn"])
        assert len(result.elements) == 2
        element_names = [e.element for e in result.elements]
        assert "C" in element_names
        assert "Mn" in element_names

    def test_element_order_matches_input(self):
        """Elements should appear in the same order as element_columns."""
        rows = [
            {"heat": "H1", "C": 0.05, "Mn": 1.2, "Si": 0.3},
            {"heat": "H2", "C": 0.06, "Mn": 1.3, "Si": 0.4},
        ]
        result = analyze_chemistry(rows, "heat", ["Mn", "C", "Si"])
        assert [e.element for e in result.elements] == ["Mn", "C", "Si"]

    def test_multiple_rows_per_heat_averaged(self):
        """When a heat has multiple rows, per-heat average is computed first."""
        rows = [
            {"heat": "H1", "C": 0.04},
            {"heat": "H1", "C": 0.06},  # H1 avg = 0.05
            {"heat": "H2", "C": 0.10},  # H2 avg = 0.10
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        c_stats = result.elements[0]
        # Overall mean should be mean([0.05, 0.10]) = 0.075
        expected_mean = round(statistics.mean([0.05, 0.10]), 6)
        assert c_stats.mean == expected_mean

    # --- With specs ---

    def test_with_specs_all_in_spec(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
            {"heat": "H3", "C": 0.04},
        ]
        specs = {"C": (0.03, 0.07)}
        result = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        c_stats = result.elements[0]
        assert c_stats.in_spec_pct == 100.0
        assert result.off_spec_heats == []

    def test_with_specs_some_off_spec(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.10},  # off spec
        ]
        specs = {"C": (0.03, 0.07)}
        result = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        c_stats = result.elements[0]
        assert c_stats.in_spec_pct == 50.0
        assert "H2" in result.off_spec_heats

    def test_with_specs_boundary_value_in_spec(self):
        """Values exactly on spec boundary should be in spec (lo <= val <= hi)."""
        rows = [
            {"heat": "H1", "C": 0.03},  # exactly at lower bound
            {"heat": "H2", "C": 0.07},  # exactly at upper bound
        ]
        specs = {"C": (0.03, 0.07)}
        result = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        assert result.elements[0].in_spec_pct == 100.0
        assert result.off_spec_heats == []

    def test_with_specs_only_for_some_elements(self):
        """Specs provided for one element but not another."""
        rows = [
            {"heat": "H1", "C": 0.05, "Mn": 1.2},
            {"heat": "H2", "C": 0.06, "Mn": 1.3},
        ]
        specs = {"C": (0.03, 0.07)}  # no spec for Mn
        result = analyze_chemistry(rows, "heat", ["C", "Mn"], specs=specs)
        c_stats = [e for e in result.elements if e.element == "C"][0]
        mn_stats = [e for e in result.elements if e.element == "Mn"][0]
        assert c_stats.in_spec_pct == 100.0
        assert mn_stats.in_spec_pct is None  # no spec for Mn

    def test_off_spec_heats_sorted(self):
        """Off-spec heats should be returned sorted."""
        rows = [
            {"heat": "H3", "C": 0.20},
            {"heat": "H1", "C": 0.20},
            {"heat": "H2", "C": 0.05},
        ]
        specs = {"C": (0.03, 0.07)}
        result = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        assert result.off_spec_heats == ["H1", "H3"]

    # --- Without specs ---

    def test_no_specs_in_spec_pct_is_none(self):
        rows = [{"heat": "H1", "C": 0.05}]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert result.elements[0].in_spec_pct is None

    def test_no_specs_off_spec_heats_empty(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert result.off_spec_heats == []

    # --- Partial / skipped data ---

    def test_skips_none_element_values(self):
        rows = [
            {"heat": "H1", "C": 0.05, "Mn": None},
            {"heat": "H2", "C": 0.06, "Mn": 1.3},
        ]
        result = analyze_chemistry(rows, "heat", ["C", "Mn"])
        assert result.heat_count == 2
        mn_stats = [e for e in result.elements if e.element == "Mn"][0]
        assert mn_stats.mean == 1.3

    def test_skips_non_numeric_element_values(self):
        rows = [
            {"heat": "H1", "C": "not_a_number"},
            {"heat": "H2", "C": 0.06},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert result is not None
        assert result.elements[0].mean == 0.06

    # --- Summary ---

    def test_summary_mentions_heat_count(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert "2 heats" in result.summary

    def test_summary_mentions_off_spec(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.20},
        ]
        specs = {"C": (0.03, 0.07)}
        result = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        assert "off-spec" in result.summary

    def test_summary_no_off_spec_when_all_in_spec(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        specs = {"C": (0.03, 0.07)}
        result = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        assert "off-spec" not in result.summary

    # --- Return type ---

    def test_returns_chemistry_result_instance(self):
        rows = [{"heat": "H1", "C": 0.05}]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert isinstance(result, ChemistryResult)

    def test_element_stats_are_element_stats_instances(self):
        rows = [{"heat": "H1", "C": 0.05}]
        result = analyze_chemistry(rows, "heat", ["C"])
        assert isinstance(result.elements[0], ElementStats)


# ===================================================================
# grade_wise_analysis Tests
# ===================================================================


class TestGradeWiseAnalysis:
    """Tests for the grade_wise_analysis function."""

    # --- None / empty returns ---

    def test_empty_rows_returns_none(self):
        assert grade_wise_analysis([], "grade", "weight") is None

    def test_all_none_returns_none(self):
        rows = [{"grade": None, "weight": None}]
        assert grade_wise_analysis(rows, "grade", "weight") is None

    def test_all_none_grades_returns_none(self):
        rows = [{"grade": None, "weight": 100}]
        assert grade_wise_analysis(rows, "grade", "weight") is None

    def test_all_none_weights_returns_none(self):
        rows = [{"grade": "A", "weight": None}]
        assert grade_wise_analysis(rows, "grade", "weight") is None

    def test_missing_columns_returns_none(self):
        rows = [{"other": "X"}]
        assert grade_wise_analysis(rows, "grade", "weight") is None

    # --- Basic grade breakdown ---

    def test_basic_grade_counts(self):
        rows = [
            {"grade": "TMT500", "weight": 100},
            {"grade": "TMT500", "weight": 120},
            {"grade": "TMT550", "weight": 80},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert result is not None
        assert result.grade_count == 2

        tmt500 = [g for g in result.grades if g.grade == "TMT500"][0]
        assert tmt500.heat_count == 2
        assert tmt500.total_weight == 220.0

        tmt550 = [g for g in result.grades if g.grade == "TMT550"][0]
        assert tmt550.heat_count == 1
        assert tmt550.total_weight == 80.0

    def test_total_weight(self):
        rows = [
            {"grade": "TMT500", "weight": 100},
            {"grade": "TMT500", "weight": 120},
            {"grade": "TMT550", "weight": 80},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert result.total_weight == 300.0

    def test_avg_weight_per_grade(self):
        rows = [
            {"grade": "A", "weight": 100},
            {"grade": "A", "weight": 200},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        a_info = result.grades[0]
        assert a_info.avg_weight == 150.0

    def test_share_pct(self):
        rows = [
            {"grade": "A", "weight": 75},
            {"grade": "B", "weight": 25},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        a_info = [g for g in result.grades if g.grade == "A"][0]
        b_info = [g for g in result.grades if g.grade == "B"][0]
        assert a_info.share_pct == 75.0
        assert b_info.share_pct == 25.0

    def test_share_pct_sums_to_100(self):
        rows = [
            {"grade": "A", "weight": 33},
            {"grade": "B", "weight": 33},
            {"grade": "C", "weight": 34},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        total_share = sum(g.share_pct for g in result.grades)
        assert abs(total_share - 100.0) < 0.5

    # --- Value column ---

    def test_with_value_column(self):
        rows = [
            {"grade": "A", "weight": 100, "revenue": 5000},
            {"grade": "A", "weight": 100, "revenue": 6000},
        ]
        result = grade_wise_analysis(rows, "grade", "weight", value_column="revenue")
        a_info = result.grades[0]
        assert a_info.avg_value == 5500.0

    def test_without_value_column_avg_value_is_none(self):
        rows = [{"grade": "A", "weight": 100}]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert result.grades[0].avg_value is None

    def test_value_column_with_none_values(self):
        """When value column specified but some values are None."""
        rows = [
            {"grade": "A", "weight": 100, "revenue": 5000},
            {"grade": "A", "weight": 100, "revenue": None},
        ]
        result = grade_wise_analysis(rows, "grade", "weight", value_column="revenue")
        a_info = result.grades[0]
        # Only one valid revenue value: 5000
        assert a_info.avg_value == 5000.0

    def test_value_column_with_all_none_for_grade(self):
        """Value column specified but all values for a grade are None."""
        rows = [
            {"grade": "A", "weight": 100, "revenue": None},
        ]
        result = grade_wise_analysis(rows, "grade", "weight", value_column="revenue")
        assert result.grades[0].avg_value is None

    def test_value_column_multiple_grades(self):
        rows = [
            {"grade": "A", "weight": 100, "revenue": 5000},
            {"grade": "B", "weight": 200, "revenue": 10000},
        ]
        result = grade_wise_analysis(rows, "grade", "weight", value_column="revenue")
        a_info = [g for g in result.grades if g.grade == "A"][0]
        b_info = [g for g in result.grades if g.grade == "B"][0]
        assert a_info.avg_value == 5000.0
        assert b_info.avg_value == 10000.0

    # --- Dominant grade ---

    def test_dominant_grade_is_heaviest(self):
        rows = [
            {"grade": "A", "weight": 500},
            {"grade": "B", "weight": 200},
            {"grade": "B", "weight": 200},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        # A: 500, B: 400 -> A is dominant (first in sorted-by-weight-desc)
        assert result.dominant_grade == "A"

    def test_dominant_grade_single_grade(self):
        rows = [
            {"grade": "OnlyGrade", "weight": 100},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert result.dominant_grade == "OnlyGrade"

    # --- Sorting by weight descending ---

    def test_sorted_by_total_weight_descending(self):
        rows = [
            {"grade": "Small", "weight": 10},
            {"grade": "Big", "weight": 100},
            {"grade": "Mid", "weight": 50},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        weights = [g.total_weight for g in result.grades]
        assert weights == sorted(weights, reverse=True)
        assert result.grades[0].grade == "Big"
        assert result.grades[1].grade == "Mid"
        assert result.grades[2].grade == "Small"

    # --- Summary ---

    def test_summary_mentions_dominant_grade(self):
        rows = [
            {"grade": "TMT500", "weight": 200},
            {"grade": "TMT550", "weight": 100},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert "TMT500" in result.summary

    def test_summary_mentions_grade_count(self):
        rows = [
            {"grade": "A", "weight": 100},
            {"grade": "B", "weight": 200},
            {"grade": "C", "weight": 300},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert "3 grades" in result.summary

    # --- Return type ---

    def test_returns_grade_wise_result_instance(self):
        rows = [{"grade": "A", "weight": 100}]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert isinstance(result, GradeWiseResult)

    def test_grade_info_instances(self):
        rows = [{"grade": "A", "weight": 100}]
        result = grade_wise_analysis(rows, "grade", "weight")
        assert isinstance(result.grades[0], GradeInfo)

    # --- Rounding ---

    def test_values_are_rounded(self):
        rows = [
            {"grade": "A", "weight": 33.333333},
            {"grade": "A", "weight": 66.666666},
        ]
        result = grade_wise_analysis(rows, "grade", "weight")
        a_info = result.grades[0]
        assert a_info.total_weight == round(33.333333 + 66.666666, 4)
        assert a_info.avg_weight == round((33.333333 + 66.666666) / 2, 4)


# ===================================================================
# detect_grade_anomalies Tests
# ===================================================================


class TestDetectGradeAnomalies:
    """Tests for the detect_grade_anomalies function."""

    # --- Empty / edge returns ---

    def test_empty_rows_returns_empty(self):
        assert detect_grade_anomalies([], "heat", "grade", ["C"]) == []

    def test_empty_elements_returns_empty(self):
        rows = [{"heat": "H1", "grade": "A", "C": 0.05}]
        assert detect_grade_anomalies(rows, "heat", "grade", []) == []

    def test_no_valid_data_returns_empty(self):
        rows = [{"heat": None, "grade": None}]
        assert detect_grade_anomalies(rows, "heat", "grade", ["C"]) == []

    def test_missing_element_data_returns_empty(self):
        """Heats have no data for the requested elements."""
        rows = [{"heat": "H1", "grade": "A", "other_elem": 0.05}]
        specs = {"A": {"C": (0.03, 0.07)}}
        anomalies = detect_grade_anomalies(
            rows, "heat", "grade", ["C"], specs=specs
        )
        assert anomalies == []

    # --- With explicit specs ---

    def test_no_anomalies_when_all_in_spec(self):
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.05},
            {"heat": "H2", "grade": "A", "C": 0.06},
        ]
        specs = {"A": {"C": (0.03, 0.10)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert anomalies == []

    def test_detects_anomaly_above_upper_spec(self):
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.05},
            {"heat": "H2", "grade": "A", "C": 0.20},  # above upper spec
        ]
        specs = {"A": {"C": (0.03, 0.07)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        assert anomalies[0].heat == "H2"
        assert anomalies[0].element == "C"
        assert anomalies[0].grade == "A"
        assert anomalies[0].expected_range == (0.03, 0.07)

    def test_detects_anomaly_below_lower_spec(self):
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.01},  # below lower spec
            {"heat": "H2", "grade": "A", "C": 0.05},
        ]
        specs = {"A": {"C": (0.03, 0.07)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        assert anomalies[0].heat == "H1"
        assert anomalies[0].value == 0.01

    def test_boundary_values_not_anomalous(self):
        """Values exactly at spec boundaries should NOT be flagged."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.03},
            {"heat": "H2", "grade": "A", "C": 0.07},
        ]
        specs = {"A": {"C": (0.03, 0.07)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert anomalies == []

    def test_multiple_elements_anomaly(self):
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.20, "Mn": 5.0},
        ]
        specs = {"A": {"C": (0.03, 0.07), "Mn": (1.0, 1.5)}}
        anomalies = detect_grade_anomalies(
            rows, "heat", "grade", ["C", "Mn"], specs=specs
        )
        assert len(anomalies) == 2
        elements = {a.element for a in anomalies}
        assert elements == {"C", "Mn"}

    def test_multiple_grades_specs(self):
        """Different grades have different specs."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.05},
            {"heat": "H2", "grade": "B", "C": 0.05},  # off-spec for grade B
        ]
        specs = {
            "A": {"C": (0.03, 0.07)},
            "B": {"C": (0.06, 0.10)},  # H2's 0.05 is below lower bound
        }
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        assert anomalies[0].heat == "H2"
        assert anomalies[0].grade == "B"

    def test_grade_not_in_specs_ignored(self):
        """Heats with grades not in the spec dict should be skipped."""
        rows = [
            {"heat": "H1", "grade": "Unknown", "C": 999.0},
        ]
        specs = {"A": {"C": (0.03, 0.07)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert anomalies == []

    # --- Severity levels ---

    def test_severity_high(self):
        """Deviation > 1.0 * spec_range -> high severity."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.15},
        ]
        specs = {"A": {"C": (0.04, 0.06)}}
        # deviation = (0.15 - 0.06) / (0.06 - 0.04) = 0.09 / 0.02 = 4.5 -> high
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        assert anomalies[0].severity == "high"

    def test_severity_medium(self):
        """Deviation > 0.5 but <= 1.0 -> medium severity."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.075},
        ]
        specs = {"A": {"C": (0.04, 0.06)}}
        # deviation = (0.075 - 0.06) / (0.06 - 0.04) = 0.015 / 0.02 = 0.75 -> medium
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        assert anomalies[0].severity == "medium"

    def test_severity_low(self):
        """Deviation <= 0.5 -> low severity."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.065},
        ]
        specs = {"A": {"C": (0.04, 0.06)}}
        # deviation = (0.065 - 0.06) / (0.06 - 0.04) = 0.005 / 0.02 = 0.25 -> low
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        assert anomalies[0].severity == "low"

    def test_severity_with_zero_spec_range(self):
        """When spec range is 0, deviation is computed as abs(value - lo)."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.10},
        ]
        # spec range = 0.05 - 0.05 = 0
        specs = {"A": {"C": (0.05, 0.05)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        # deviation = abs(0.10 - 0.05) = 0.05 -> low (<=0.5 threshold)
        assert anomalies[0].severity == "low"

    def test_severity_high_with_zero_spec_range(self):
        """With zero spec range, large abs diff -> high."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 5.0},
        ]
        specs = {"A": {"C": (0.05, 0.05)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert len(anomalies) == 1
        # deviation = abs(5.0 - 0.05) = 4.95 -> high
        assert anomalies[0].severity == "high"

    # --- Auto-derived specs ---

    def test_auto_derive_specs_detects_outlier(self):
        """With many tight values, an extreme outlier should be detected."""
        # Need enough tight data points so the outlier doesn't inflate std
        # enough to include itself within mean +/- 2*std.
        rows = [
            {"heat": f"H{i}", "grade": "A", "C": 0.050}
            for i in range(1, 11)
        ]
        rows.append({"heat": "H_outlier", "grade": "A", "C": 0.500})
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"])
        heat_ids = [a.heat for a in anomalies]
        assert "H_outlier" in heat_ids

    def test_auto_derive_specs_no_anomaly_in_tight_data(self):
        """When all values are tight, no anomaly should be found with auto specs."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.050},
            {"heat": "H2", "grade": "A", "C": 0.051},
            {"heat": "H3", "grade": "A", "C": 0.049},
            {"heat": "H4", "grade": "A", "C": 0.050},
        ]
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"])
        # mean+/-2*std should encompass all values
        assert anomalies == []

    def test_auto_derive_with_single_value_tight_tolerance(self):
        """Single value per grade uses (v, v) as spec, so exact match is not anomalous."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.05},
        ]
        # With a single value, the spec is (0.05, 0.05), and the heat avg is 0.05
        # So it should NOT be an anomaly (lo <= val <= hi)
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"])
        assert anomalies == []

    def test_auto_derive_multiple_grades(self):
        """Auto-derive per grade, not globally."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.05},
            {"heat": "H2", "grade": "A", "C": 0.05},
            {"heat": "H3", "grade": "A", "C": 0.05},
            {"heat": "H4", "grade": "B", "C": 0.50},
            {"heat": "H5", "grade": "B", "C": 0.50},
            {"heat": "H6", "grade": "B", "C": 0.50},
        ]
        # All values within each grade are identical, so no anomalies
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"])
        assert anomalies == []

    # --- Anomaly value ---

    def test_anomaly_value_is_rounded(self):
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.123456789},
        ]
        specs = {"A": {"C": (0.01, 0.02)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert anomalies[0].value == round(0.123456789, 6)

    # --- Multiple rows per heat ---

    def test_multiple_rows_per_heat_uses_average(self):
        """Multiple samples for a heat should be averaged before comparison."""
        rows = [
            {"heat": "H1", "grade": "A", "C": 0.04},  # avg -> 0.055
            {"heat": "H1", "grade": "A", "C": 0.07},
            {"heat": "H2", "grade": "A", "C": 0.05},
        ]
        specs = {"A": {"C": (0.045, 0.06)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        # H1 avg = 0.055, in spec; H2 = 0.05, in spec
        assert anomalies == []

    # --- Sorted by heat ---

    def test_anomalies_sorted_by_heat(self):
        rows = [
            {"heat": "H3", "grade": "A", "C": 0.20},
            {"heat": "H1", "grade": "A", "C": 0.20},
            {"heat": "H2", "grade": "A", "C": 0.05},
        ]
        specs = {"A": {"C": (0.03, 0.07)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        heats = [a.heat for a in anomalies]
        assert heats == sorted(heats)

    # --- Return type ---

    def test_returns_list(self):
        result = detect_grade_anomalies([], "heat", "grade", ["C"])
        assert isinstance(result, list)

    def test_anomaly_is_grade_anomaly_instance(self):
        rows = [{"heat": "H1", "grade": "A", "C": 0.20}]
        specs = {"A": {"C": (0.03, 0.07)}}
        anomalies = detect_grade_anomalies(rows, "heat", "grade", ["C"], specs=specs)
        assert isinstance(anomalies[0], GradeAnomaly)


# ===================================================================
# format_heat_report Tests
# ===================================================================


class TestFormatHeatReport:
    """Tests for the format_heat_report function."""

    # --- No data ---

    def test_all_none_reports_no_data(self):
        report = format_heat_report()
        assert "No analysis data provided" in report

    def test_explicit_none_args(self):
        report = format_heat_report(analysis=None, chemistry=None, grade_wise=None)
        assert "No analysis data provided" in report

    # --- Header ---

    def test_header_always_present(self):
        report = format_heat_report()
        assert "Heat Analysis Report" in report
        assert "=" * 50 in report

    def test_header_present_with_data(self):
        rows = [{"heat_id": "H1", "weight": 100}]
        analysis = analyze_heats(rows, "heat_id", "weight")
        report = format_heat_report(analysis=analysis)
        assert "Heat Analysis Report" in report
        assert "=" * 50 in report

    # --- Analysis section only ---

    def test_with_analysis_section(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 120},
        ]
        analysis = analyze_heats(rows, "heat_id", "weight")
        report = format_heat_report(analysis=analysis)
        assert "Heat Overview" in report
        assert "Total Heats" in report
        assert "2" in report
        assert "Total Weight" in report
        assert "Avg Weight/Heat" in report
        assert "Min Weight" in report
        assert "Max Weight" in report
        assert "Weight Std Dev" in report
        assert "No analysis data provided" not in report

    def test_analysis_section_values(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 200},
        ]
        analysis = analyze_heats(rows, "heat_id", "weight")
        report = format_heat_report(analysis=analysis)
        assert "300.00" in report  # total weight
        assert "150.00" in report  # avg weight
        assert "100.00" in report  # min weight
        assert "200.00" in report  # max weight

    def test_analysis_with_grade_distribution(self):
        rows = [
            {"heat_id": "H1", "weight": 100, "grade": "A"},
            {"heat_id": "H2", "weight": 120, "grade": "B"},
        ]
        analysis = analyze_heats(rows, "heat_id", "weight", grade_column="grade")
        report = format_heat_report(analysis=analysis)
        assert "Grade Distribution" in report
        assert "A: 1 heats" in report
        assert "B: 1 heats" in report

    def test_analysis_with_heats_per_period(self):
        rows = [
            {"heat_id": "H1", "weight": 100, "shift": "morning"},
            {"heat_id": "H2", "weight": 120, "shift": "night"},
        ]
        analysis = analyze_heats(rows, "heat_id", "weight", time_column="shift")
        report = format_heat_report(analysis=analysis)
        assert "Heats per Period" in report
        assert "morning: 1 heats" in report
        assert "night: 1 heats" in report

    def test_analysis_without_grade_or_period(self):
        rows = [{"heat_id": "H1", "weight": 100}]
        analysis = analyze_heats(rows, "heat_id", "weight")
        report = format_heat_report(analysis=analysis)
        assert "Grade Distribution" not in report
        assert "Heats per Period" not in report

    # --- Chemistry section only ---

    def test_with_chemistry_section(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        chem = analyze_chemistry(rows, "heat", ["C"])
        report = format_heat_report(chemistry=chem)
        assert "Chemistry Analysis" in report
        assert "Heats analyzed: 2" in report
        assert "C:" in report
        assert "No analysis data provided" not in report

    def test_chemistry_with_specs(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        specs = {"C": (0.03, 0.07)}
        chem = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        report = format_heat_report(chemistry=chem)
        assert "in-spec=" in report

    def test_chemistry_without_specs(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        chem = analyze_chemistry(rows, "heat", ["C"])
        report = format_heat_report(chemistry=chem)
        assert "in-spec=" not in report

    def test_chemistry_with_off_spec_heats(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.20},
        ]
        specs = {"C": (0.03, 0.07)}
        chem = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        report = format_heat_report(chemistry=chem)
        assert "Off-spec heats" in report
        assert "H2" in report

    def test_chemistry_no_off_spec_section_when_all_in_spec(self):
        rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        specs = {"C": (0.03, 0.07)}
        chem = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        report = format_heat_report(chemistry=chem)
        assert "Off-spec heats" not in report

    def test_chemistry_multiple_elements_in_report(self):
        rows = [
            {"heat": "H1", "C": 0.05, "Mn": 1.2},
            {"heat": "H2", "C": 0.06, "Mn": 1.3},
        ]
        chem = analyze_chemistry(rows, "heat", ["C", "Mn"])
        report = format_heat_report(chemistry=chem)
        assert "C:" in report
        assert "Mn:" in report

    # --- Grade-wise section only ---

    def test_with_grade_wise_section(self):
        rows = [
            {"grade": "TMT500", "weight": 100},
            {"grade": "TMT550", "weight": 80},
        ]
        gw = grade_wise_analysis(rows, "grade", "weight")
        report = format_heat_report(grade_wise=gw)
        assert "Grade-Wise Analysis" in report
        assert "TMT500" in report
        assert "TMT550" in report
        assert "Dominant Grade:" in report
        assert "Grade Count:" in report
        assert "No analysis data provided" not in report

    def test_grade_wise_with_value(self):
        rows = [
            {"grade": "A", "weight": 100, "revenue": 5000},
            {"grade": "B", "weight": 200, "revenue": 10000},
        ]
        gw = grade_wise_analysis(rows, "grade", "weight", value_column="revenue")
        report = format_heat_report(grade_wise=gw)
        assert "avg_value=" in report

    def test_grade_wise_without_value(self):
        rows = [
            {"grade": "A", "weight": 100},
        ]
        gw = grade_wise_analysis(rows, "grade", "weight")
        report = format_heat_report(grade_wise=gw)
        assert "avg_value=" not in report

    # --- Combined sections ---

    def test_combined_all_sections(self):
        heat_rows = [
            {"heat_id": "H1", "weight": 100, "grade": "A"},
            {"heat_id": "H2", "weight": 120, "grade": "B"},
        ]
        chem_rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        grade_rows = [
            {"grade": "A", "weight": 100},
            {"grade": "B", "weight": 120},
        ]
        analysis = analyze_heats(heat_rows, "heat_id", "weight", grade_column="grade")
        chemistry = analyze_chemistry(chem_rows, "heat", ["C"])
        gw = grade_wise_analysis(grade_rows, "grade", "weight")

        report = format_heat_report(analysis=analysis, chemistry=chemistry, grade_wise=gw)
        assert "Heat Overview" in report
        assert "Chemistry Analysis" in report
        assert "Grade-Wise Analysis" in report
        assert "No analysis data provided" not in report

    def test_analysis_and_chemistry_only(self):
        heat_rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 120},
        ]
        chem_rows = [
            {"heat": "H1", "C": 0.05},
            {"heat": "H2", "C": 0.06},
        ]
        analysis = analyze_heats(heat_rows, "heat_id", "weight")
        chemistry = analyze_chemistry(chem_rows, "heat", ["C"])

        report = format_heat_report(analysis=analysis, chemistry=chemistry)
        assert "Heat Overview" in report
        assert "Chemistry Analysis" in report
        assert "Grade-Wise Analysis" not in report
        assert "No analysis data provided" not in report

    def test_grade_wise_only(self):
        grade_rows = [
            {"grade": "A", "weight": 100},
        ]
        gw = grade_wise_analysis(grade_rows, "grade", "weight")

        report = format_heat_report(grade_wise=gw)
        assert "Heat Overview" not in report
        assert "Chemistry Analysis" not in report
        assert "Grade-Wise Analysis" in report
        assert "No analysis data provided" not in report

    # --- Report format ---

    def test_report_contains_section_separators(self):
        rows = [
            {"heat_id": "H1", "weight": 100},
            {"heat_id": "H2", "weight": 200},
        ]
        analysis = analyze_heats(rows, "heat_id", "weight")
        report = format_heat_report(analysis=analysis)
        assert "-" * 48 in report

    def test_report_is_string(self):
        report = format_heat_report()
        assert isinstance(report, str)

    def test_report_off_spec_truncated_at_ten(self):
        """When more than 10 off-spec heats, only first 10 shown + '... and N more'."""
        rows = [{"heat": f"H{i:02d}", "C": 0.50} for i in range(15)]
        specs = {"C": (0.03, 0.07)}
        chem = analyze_chemistry(rows, "heat", ["C"], specs=specs)
        report = format_heat_report(chemistry=chem)
        assert "... and 5 more" in report


# ===================================================================
# Dataclass sanity tests
# ===================================================================


class TestDataclasses:
    """Verify dataclass instantiation and field access."""

    def test_heat_analysis_result_fields(self):
        result = HeatAnalysisResult(
            total_heats=5,
            total_weight=1000.0,
            avg_weight_per_heat=200.0,
            min_weight=150.0,
            max_weight=250.0,
            weight_std=25.0,
            grade_distribution={"A": 3, "B": 2},
            heats_per_period={"Jan": 3, "Feb": 2},
            summary="test summary",
        )
        assert result.total_heats == 5
        assert result.total_weight == 1000.0
        assert result.avg_weight_per_heat == 200.0
        assert result.min_weight == 150.0
        assert result.max_weight == 250.0
        assert result.weight_std == 25.0
        assert result.grade_distribution == {"A": 3, "B": 2}
        assert result.heats_per_period == {"Jan": 3, "Feb": 2}
        assert result.summary == "test summary"

    def test_element_stats_fields(self):
        stats = ElementStats(
            element="C",
            mean=0.05,
            std=0.01,
            min=0.03,
            max=0.07,
            cv_pct=20.0,
            in_spec_pct=95.0,
        )
        assert stats.element == "C"
        assert stats.mean == 0.05
        assert stats.std == 0.01
        assert stats.min == 0.03
        assert stats.max == 0.07
        assert stats.cv_pct == 20.0
        assert stats.in_spec_pct == 95.0

    def test_chemistry_result_fields(self):
        result = ChemistryResult(
            elements=[],
            heat_count=10,
            off_spec_heats=["H1", "H2"],
            summary="test",
        )
        assert result.heat_count == 10
        assert result.off_spec_heats == ["H1", "H2"]

    def test_grade_info_fields(self):
        info = GradeInfo(
            grade="TMT500",
            heat_count=10,
            total_weight=500.0,
            avg_weight=50.0,
            share_pct=30.0,
            avg_value=100.0,
        )
        assert info.grade == "TMT500"
        assert info.heat_count == 10
        assert info.total_weight == 500.0
        assert info.avg_weight == 50.0
        assert info.share_pct == 30.0
        assert info.avg_value == 100.0

    def test_grade_wise_result_fields(self):
        result = GradeWiseResult(
            grades=[],
            total_weight=1000.0,
            dominant_grade="TMT500",
            grade_count=3,
            summary="test",
        )
        assert result.total_weight == 1000.0
        assert result.dominant_grade == "TMT500"
        assert result.grade_count == 3

    def test_grade_anomaly_fields(self):
        anomaly = GradeAnomaly(
            heat="H1",
            grade="A",
            element="C",
            value=0.15,
            expected_range=(0.03, 0.07),
            severity="high",
        )
        assert anomaly.heat == "H1"
        assert anomaly.grade == "A"
        assert anomaly.element == "C"
        assert anomaly.value == 0.15
        assert anomaly.expected_range == (0.03, 0.07)
        assert anomaly.severity == "high"
