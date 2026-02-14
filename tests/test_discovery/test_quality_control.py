"""Tests for quality control analytics module."""

from business_brain.discovery.quality_control import (
    CapabilityResult,
    ControlChartResult,
    DefectResult,
    EntityDefect,
    EntityGrade,
    EntityRejection,
    GradeResult,
    RejectionResult,
    analyze_defects,
    analyze_rejections,
    compute_process_capability,
    control_chart_data,
    format_quality_report,
    grade_analysis,
)


# ===================================================================
# compute_process_capability Tests
# ===================================================================


class TestComputeProcessCapability:
    def test_basic_capability(self):
        # Well-centered process with known values
        values = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0]
        result = compute_process_capability(values, lsl=9.0, usl=11.0)
        assert result is not None
        assert result.cp > 0
        assert result.cpk > 0
        assert result.lsl == 9.0
        assert result.usl == 11.0
        assert isinstance(result.summary, str)

    def test_excellent_process(self):
        # Very tight process, wide spec limits -> high Cpk
        values = [50.0, 50.01, 49.99, 50.0, 50.02, 49.98] * 10
        result = compute_process_capability(values, lsl=49.0, usl=51.0)
        assert result is not None
        assert result.process_grade == "Excellent"
        assert result.cpk >= 1.67

    def test_poor_process(self):
        # Wide spread relative to spec -> low Cpk
        values = [10.0, 12.0, 8.0, 14.0, 6.0, 11.0, 9.0, 13.0, 7.0, 15.0]
        result = compute_process_capability(values, lsl=9.0, usl=11.0)
        assert result is not None
        assert result.process_grade == "Poor"
        assert result.cpk < 1.0

    def test_good_process(self):
        # Cpk between 1.33 and 1.67
        import statistics as st
        values = [100.0, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100.3, 99.7, 100.0]
        mean = st.mean(values)
        std = st.stdev(values)
        # Set spec limits to give Cpk ~ 1.5
        usl = mean + 1.5 * 3 * std
        lsl = mean - 1.5 * 3 * std
        result = compute_process_capability(values, lsl=lsl, usl=usl)
        assert result is not None
        assert result.process_grade == "Good"

    def test_adequate_process(self):
        # Cpk between 1.0 and 1.33
        import statistics as st
        values = [50.0, 50.5, 49.5, 50.2, 49.8, 50.1, 49.9, 50.3, 49.7, 50.0]
        mean = st.mean(values)
        std = st.stdev(values)
        # Set spec limits to give Cpk ~ 1.15
        usl = mean + 1.15 * 3 * std
        lsl = mean - 1.15 * 3 * std
        result = compute_process_capability(values, lsl=lsl, usl=usl)
        assert result is not None
        assert result.process_grade == "Adequate"

    def test_centered_process(self):
        # Symmetric around mean -> Cp should be close to Cpk
        values = [10.0, 10.1, 9.9, 10.0, 10.05, 9.95, 10.0, 10.02, 9.98, 10.0]
        result = compute_process_capability(values, lsl=9.5, usl=10.5)
        assert result is not None
        assert result.centered is True
        assert abs(result.cp - result.cpk) < 0.1

    def test_off_center_process(self):
        # Mean shifted toward USL -> Cpk << Cp
        values = [10.8, 10.9, 10.7, 10.85, 10.75, 10.8, 10.82, 10.78, 10.81, 10.79]
        result = compute_process_capability(values, lsl=9.0, usl=11.0)
        assert result is not None
        assert result.centered is False
        assert result.cpk < result.cp

    def test_empty_values_returns_none(self):
        assert compute_process_capability([], lsl=0, usl=10) is None

    def test_single_value_returns_none(self):
        assert compute_process_capability([5.0], lsl=0, usl=10) is None

    def test_none_values_filtered(self):
        values = [10.0, None, 10.1, None, 9.9, None, 10.0, 10.2, 9.8, 10.0]
        result = compute_process_capability(values, lsl=9.0, usl=11.0)
        assert result is not None
        # Should have 6 valid values contributing
        assert result.mean > 0

    def test_all_none_values_returns_none(self):
        assert compute_process_capability([None, None, None], lsl=0, usl=10) is None

    def test_zero_std_within_spec(self):
        # All identical values within spec
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = compute_process_capability(values, lsl=0, usl=10)
        assert result is not None
        assert result.std == 0
        assert result.cp == 999.99
        assert result.cpk == 999.99
        assert result.ppm_out_of_spec == 0.0

    def test_zero_std_outside_spec(self):
        # All identical values outside spec
        values = [15.0, 15.0, 15.0, 15.0, 15.0]
        result = compute_process_capability(values, lsl=0, usl=10)
        assert result is not None
        assert result.cp == 0.0
        assert result.cpk == 0.0
        assert result.ppm_out_of_spec == 1_000_000.0

    def test_ppm_reasonable(self):
        # Well-centered process should have low PPM
        values = [10.0, 10.01, 9.99, 10.0, 10.005, 9.995] * 10
        result = compute_process_capability(values, lsl=9.0, usl=11.0)
        assert result is not None
        assert result.ppm_out_of_spec < 100  # should be very low


# ===================================================================
# control_chart_data Tests
# ===================================================================


class TestControlChartData:
    def test_basic_control_chart(self):
        values = [10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.0, 10.1, 9.9, 10.0]
        result = control_chart_data(values)
        assert result is not None
        assert len(result.values) == 10
        assert result.ucl > result.mean
        assert result.lcl < result.mean
        assert isinstance(result.out_of_control_indices, list)
        assert isinstance(result.summary, str)

    def test_all_in_control(self):
        # Tight, uniform data - all in control
        values = [10.0, 10.01, 9.99, 10.0, 10.005, 9.995, 10.0, 10.002, 9.998, 10.0]
        result = control_chart_data(values)
        assert result is not None
        assert result.out_of_control_count == 0
        assert result.in_control_pct == 100.0

    def test_out_of_control_detected(self):
        # Many tight values with one extreme outlier far beyond 3-sigma
        values = [10.0] * 30 + [10.01, 9.99, 10.02, 9.98] + [500.0]
        result = control_chart_data(values)
        assert result is not None
        assert result.out_of_control_count >= 1
        assert (len(values) - 1) in result.out_of_control_indices  # the outlier at last index

    def test_subgroup_size(self):
        values = list(range(1, 21))  # 20 values
        result = control_chart_data(values, subgroup_size=5)
        assert result is not None
        # 20 values / 5 per subgroup = 4 subgroup means
        assert len(result.values) == 4

    def test_subgroup_size_one(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = control_chart_data(values, subgroup_size=1)
        assert result is not None
        assert len(result.values) == 5

    def test_empty_returns_none(self):
        assert control_chart_data([]) is None

    def test_single_value_returns_none(self):
        assert control_chart_data([5.0]) is None

    def test_none_values_filtered(self):
        values = [10.0, None, 10.1, None, 9.9, 10.0, 10.2, 9.8]
        result = control_chart_data(values)
        assert result is not None
        assert len(result.values) == 6

    def test_all_same_values(self):
        values = [5.0, 5.0, 5.0, 5.0, 5.0]
        result = control_chart_data(values)
        assert result is not None
        assert result.mean == 5.0
        assert result.ucl == 5.0  # std=0 => UCL=LCL=mean
        assert result.lcl == 5.0
        assert result.out_of_control_count == 0

    def test_in_control_pct_calculation(self):
        # Many stable points + extreme outliers beyond 3-sigma
        values = [10.0] * 50 + [1000.0, 1000.0]
        result = control_chart_data(values)
        assert result is not None
        assert result.out_of_control_count > 0
        assert result.in_control_pct < 100.0
        # in_control_pct = (total - ooc) / total * 100
        expected_pct = (len(values) - result.out_of_control_count) / len(values) * 100
        assert abs(result.in_control_pct - expected_pct) < 0.01

    def test_subgroup_size_larger_than_data(self):
        values = [1.0, 2.0, 3.0]
        result = control_chart_data(values, subgroup_size=10)
        assert result is None  # only 1 subgroup, need >= 2


# ===================================================================
# analyze_defects Tests
# ===================================================================


class TestAnalyzeDefects:
    def test_basic_defects(self):
        rows = [
            {"line": "L1", "defects": 5, "qty": 1000},
            {"line": "L2", "defects": 15, "qty": 1000},
            {"line": "L1", "defects": 3, "qty": 500},
        ]
        result = analyze_defects(rows, "line", "defects", "qty")
        assert result is not None
        assert result.total_defects == 23
        assert result.total_quantity == 2500
        assert result.worst_entity == "L2"
        assert result.best_entity == "L1"

    def test_defects_without_quantity(self):
        rows = [
            {"line": "L1", "defects": 5},
            {"line": "L2", "defects": 15},
        ]
        result = analyze_defects(rows, "line", "defects")
        assert result is not None
        assert result.total_defects == 20
        assert result.total_quantity == 0
        assert result.overall_defect_rate == 0.0

    def test_dpmo_calculation(self):
        rows = [
            {"line": "L1", "defects": 100, "qty": 1_000_000},
        ]
        result = analyze_defects(rows, "line", "defects", "qty")
        assert result is not None
        entity = result.entities[0]
        assert entity.dpmo == 100.0  # 100 / 1_000_000 * 1_000_000

    def test_empty_returns_none(self):
        assert analyze_defects([], "line", "defects") is None

    def test_all_none_returns_none(self):
        rows = [{"line": None, "defects": None}]
        assert analyze_defects(rows, "line", "defects") is None

    def test_none_defect_values_skipped(self):
        rows = [
            {"line": "L1", "defects": 5, "qty": 100},
            {"line": "L1", "defects": None, "qty": 100},
        ]
        result = analyze_defects(rows, "line", "defects", "qty")
        assert result is not None
        assert result.total_defects == 5

    def test_summary_includes_entities(self):
        rows = [
            {"line": "L1", "defects": 2, "qty": 100},
            {"line": "L2", "defects": 8, "qty": 100},
        ]
        result = analyze_defects(rows, "line", "defects", "qty")
        assert result is not None
        assert "2 entities" in result.summary
        assert "L2" in result.summary  # worst entity


# ===================================================================
# analyze_rejections Tests
# ===================================================================


class TestAnalyzeRejections:
    def test_basic_rejections(self):
        rows = [
            {"line": "L1", "accepted": 950, "rejected": 50},
            {"line": "L2", "accepted": 800, "rejected": 200},
        ]
        result = analyze_rejections(rows, "line", "accepted", "rejected")
        assert result is not None
        assert result.total_accepted == 1750
        assert result.total_rejected == 250
        assert result.worst_entity == "L2"
        assert result.best_entity == "L1"

    def test_rejection_rate(self):
        rows = [
            {"line": "L1", "accepted": 90, "rejected": 10},
        ]
        result = analyze_rejections(rows, "line", "accepted", "rejected")
        assert result is not None
        entity = result.entities[0]
        assert entity.total == 100
        assert abs(entity.rejection_rate - 10.0) < 0.01

    def test_empty_returns_none(self):
        assert analyze_rejections([], "line", "accepted", "rejected") is None

    def test_all_none_returns_none(self):
        rows = [{"line": None, "accepted": None, "rejected": None}]
        assert analyze_rejections(rows, "line", "accepted", "rejected") is None

    def test_zero_total(self):
        rows = [{"line": "L1", "accepted": 0, "rejected": 0}]
        result = analyze_rejections(rows, "line", "accepted", "rejected")
        assert result is not None
        assert result.entities[0].rejection_rate == 0.0

    def test_multiple_rows_per_entity(self):
        rows = [
            {"line": "L1", "accepted": 100, "rejected": 5},
            {"line": "L1", "accepted": 200, "rejected": 10},
        ]
        result = analyze_rejections(rows, "line", "accepted", "rejected")
        assert result is not None
        entity = result.entities[0]
        assert entity.accepted == 300
        assert entity.rejected == 15
        assert entity.total == 315

    def test_overall_rejection_rate(self):
        rows = [
            {"line": "L1", "accepted": 90, "rejected": 10},
            {"line": "L2", "accepted": 80, "rejected": 20},
        ]
        result = analyze_rejections(rows, "line", "accepted", "rejected")
        assert result is not None
        # Overall: 30 / 200 = 15%
        assert abs(result.overall_rejection_rate - 15.0) < 0.01


# ===================================================================
# grade_analysis Tests
# ===================================================================


class TestGradeAnalysis:
    def test_basic_grades(self):
        rows = [
            {"line": "L1", "grade": "A"},
            {"line": "L1", "grade": "A"},
            {"line": "L1", "grade": "B"},
            {"line": "L2", "grade": "B"},
            {"line": "L2", "grade": "C"},
        ]
        result = grade_analysis(rows, "line", "grade")
        assert result is not None
        assert result.most_common_grade in ("A", "B")  # A and B both appear twice
        assert result.grade_distribution["A"] == 2
        assert result.grade_distribution["B"] == 2
        assert result.grade_distribution["C"] == 1

    def test_entity_primary_grade(self):
        rows = [
            {"line": "L1", "grade": "A"},
            {"line": "L1", "grade": "A"},
            {"line": "L1", "grade": "B"},
        ]
        result = grade_analysis(rows, "line", "grade")
        assert result is not None
        entity = result.entity_grades[0]
        assert entity.primary_grade == "A"
        assert entity.total_items == 3

    def test_empty_returns_none(self):
        assert grade_analysis([], "line", "grade") is None

    def test_all_none_returns_none(self):
        rows = [{"line": None, "grade": None}]
        assert grade_analysis(rows, "line", "grade") is None

    def test_single_grade(self):
        rows = [
            {"line": "L1", "grade": "A"},
            {"line": "L2", "grade": "A"},
        ]
        result = grade_analysis(rows, "line", "grade")
        assert result is not None
        assert result.most_common_grade == "A"
        assert result.grade_distribution == {"A": 2}

    def test_with_value_column(self):
        rows = [
            {"line": "L1", "grade": "A", "score": 95},
            {"line": "L1", "grade": "B", "score": 82},
        ]
        result = grade_analysis(rows, "line", "grade", "score")
        assert result is not None
        # value_column doesn't change grade aggregation
        assert result.grade_distribution["A"] == 1
        assert result.grade_distribution["B"] == 1

    def test_summary_content(self):
        rows = [
            {"line": "L1", "grade": "A"},
            {"line": "L2", "grade": "B"},
        ]
        result = grade_analysis(rows, "line", "grade")
        assert result is not None
        assert "2 entities" in result.summary
        assert "2 total items" in result.summary


# ===================================================================
# format_quality_report Tests
# ===================================================================


class TestFormatQualityReport:
    def test_no_data(self):
        report = format_quality_report()
        assert "Quality Control Report" in report
        assert "No analysis data provided." in report

    def test_with_capability(self):
        cap = CapabilityResult(
            cp=1.5,
            cpk=1.4,
            mean=10.0,
            std=0.5,
            lsl=8.0,
            usl=12.0,
            ppm_out_of_spec=27.0,
            process_grade="Good",
            centered=True,
            summary="test summary",
        )
        report = format_quality_report(capability=cap)
        assert "Process Capability" in report
        assert "1.5" in report
        assert "Good" in report
        assert "No analysis data provided." not in report

    def test_with_all_sections(self):
        cap = CapabilityResult(
            cp=1.5, cpk=1.4, mean=10.0, std=0.5,
            lsl=8.0, usl=12.0, ppm_out_of_spec=27.0,
            process_grade="Good", centered=True, summary="cap summary",
        )
        defects = DefectResult(
            entities=[EntityDefect("L1", 5, 100, 5.0, 50000.0)],
            total_defects=5, total_quantity=100,
            overall_defect_rate=5.0,
            worst_entity="L1", best_entity="L1",
            summary="defect summary",
        )
        rejections = RejectionResult(
            entities=[EntityRejection("L1", 90, 10, 100, 10.0)],
            total_accepted=90, total_rejected=10,
            overall_rejection_rate=10.0,
            worst_entity="L1", best_entity="L1",
            summary="rejection summary",
        )
        grades = GradeResult(
            grade_distribution={"A": 3, "B": 2},
            entity_grades=[EntityGrade("L1", {"A": 3, "B": 2}, "A", 5)],
            most_common_grade="A",
            summary="grade summary",
        )
        report = format_quality_report(
            capability=cap,
            defects=defects,
            rejections=rejections,
            grades=grades,
        )
        assert "Process Capability" in report
        assert "Defect Analysis" in report
        assert "Rejection Analysis" in report
        assert "Grade Distribution" in report

    def test_with_only_rejections(self):
        rejections = RejectionResult(
            entities=[EntityRejection("L1", 90, 10, 100, 10.0)],
            total_accepted=90, total_rejected=10,
            overall_rejection_rate=10.0,
            worst_entity="L1", best_entity="L1",
            summary="rejection summary",
        )
        report = format_quality_report(rejections=rejections)
        assert "Rejection Analysis" in report
        assert "Process Capability" not in report
        assert "Defect Analysis" not in report
        assert "No analysis data provided." not in report
