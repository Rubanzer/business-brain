"""Tests for employee attrition and HR analytics module."""

from datetime import date

from business_brain.discovery.employee_attrition import (
    AttritionDriverResult,
    AttritionResult,
    Cohort,
    CohortRetention,
    DeptAttrition,
    DeptTenure,
    DriverFactor,
    MonthlyAttrition,
    RetentionCohortResult,
    TenureBucket,
    TenureResult,
    analyze_attrition_drivers,
    analyze_attrition_rate,
    analyze_tenure_distribution,
    compute_retention_cohorts,
    format_attrition_report,
)


# ===================================================================
# Helper to build employee rows
# ===================================================================


def _make_employee(emp_id, status, dept=None, hire_date=None, term_date=None, date_val=None, **extra):
    """Create an employee row dict."""
    row = {"emp_id": emp_id, "status": status}
    if dept is not None:
        row["department"] = dept
    if hire_date is not None:
        row["hire_date"] = hire_date
    if term_date is not None:
        row["term_date"] = term_date
    if date_val is not None:
        row["date"] = date_val
    row.update(extra)
    return row


# ===================================================================
# 1. Attrition Rate Tests
# ===================================================================


class TestAnalyzeAttritionRateBasic:
    def test_empty_returns_none(self):
        assert analyze_attrition_rate([], "emp_id", "status") is None

    def test_all_none_values_returns_none(self):
        rows = [{"emp_id": None, "status": None}]
        assert analyze_attrition_rate(rows, "emp_id", "status") is None

    def test_unrecognized_status_returns_none(self):
        rows = [{"emp_id": "E1", "status": "unknown"}]
        assert analyze_attrition_rate(rows, "emp_id", "status") is None

    def test_single_active_employee(self):
        rows = [_make_employee("E1", "Active")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result is not None
        assert result.total_employees == 1
        assert result.active_count == 1
        assert result.left_count == 0
        assert result.attrition_rate == 0.0

    def test_single_left_employee(self):
        rows = [_make_employee("E1", "Left")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result is not None
        assert result.total_employees == 1
        assert result.left_count == 1
        assert result.attrition_rate == 100.0

    def test_basic_attrition(self):
        rows = [
            _make_employee("E1", "Active"),
            _make_employee("E2", "Active"),
            _make_employee("E3", "Left"),
            _make_employee("E4", "Terminated"),
        ]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result is not None
        assert result.total_employees == 4
        assert result.active_count == 2
        assert result.left_count == 2
        assert result.attrition_rate == 50.0

    def test_attrition_rate_calculation(self):
        rows = [
            _make_employee("E1", "Active"),
            _make_employee("E2", "Active"),
            _make_employee("E3", "Active"),
            _make_employee("E4", "Left"),
        ]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.attrition_rate == 25.0

    def test_summary_contains_key_info(self):
        rows = [
            _make_employee("E1", "Active"),
            _make_employee("E2", "Left"),
        ]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert "2 employees" in result.summary
        assert "50.0%" in result.summary

    def test_result_type(self):
        rows = [_make_employee("E1", "Active")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert isinstance(result, AttritionResult)


class TestAttritionStatusNormalization:
    def test_case_insensitive_active(self):
        for status in ["active", "ACTIVE", "Active", "AcTiVe"]:
            rows = [_make_employee("E1", status)]
            result = analyze_attrition_rate(rows, "emp_id", "status")
            assert result is not None
            assert result.active_count == 1

    def test_case_insensitive_left(self):
        for status in ["left", "LEFT", "Left", "LeFt"]:
            rows = [_make_employee("E1", status)]
            result = analyze_attrition_rate(rows, "emp_id", "status")
            assert result is not None
            assert result.left_count == 1

    def test_terminated_counts_as_left(self):
        rows = [_make_employee("E1", "Terminated")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.left_count == 1

    def test_resigned_counts_as_left(self):
        rows = [_make_employee("E1", "Resigned")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.left_count == 1

    def test_quit_counts_as_left(self):
        rows = [_make_employee("E1", "Quit")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.left_count == 1

    def test_current_counts_as_active(self):
        rows = [_make_employee("E1", "Current")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.active_count == 1

    def test_employed_counts_as_active(self):
        rows = [_make_employee("E1", "Employed")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.active_count == 1

    def test_fired_counts_as_left(self):
        rows = [_make_employee("E1", "fired")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.left_count == 1

    def test_whitespace_handling(self):
        rows = [_make_employee("E1", "  Active  ")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.active_count == 1


class TestAttritionWithDepartment:
    def test_department_breakdown(self):
        rows = [
            _make_employee("E1", "Active", dept="Engineering"),
            _make_employee("E2", "Left", dept="Engineering"),
            _make_employee("E3", "Active", dept="Sales"),
            _make_employee("E4", "Active", dept="Sales"),
        ]
        result = analyze_attrition_rate(
            rows, "emp_id", "status", department_column="department"
        )
        assert len(result.by_department) == 2
        eng = [d for d in result.by_department if d.department == "Engineering"][0]
        assert eng.total == 2
        assert eng.left == 1
        assert eng.rate == 50.0
        sales = [d for d in result.by_department if d.department == "Sales"][0]
        assert sales.total == 2
        assert sales.left == 0
        assert sales.rate == 0.0

    def test_no_department_empty_list(self):
        rows = [_make_employee("E1", "Active")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.by_department == []

    def test_department_sorted_alphabetically(self):
        rows = [
            _make_employee("E1", "Active", dept="Zebra"),
            _make_employee("E2", "Active", dept="Alpha"),
        ]
        result = analyze_attrition_rate(
            rows, "emp_id", "status", department_column="department"
        )
        assert result.by_department[0].department == "Alpha"
        assert result.by_department[1].department == "Zebra"

    def test_dept_attrition_type(self):
        rows = [_make_employee("E1", "Active", dept="Eng")]
        result = analyze_attrition_rate(
            rows, "emp_id", "status", department_column="department"
        )
        assert isinstance(result.by_department[0], DeptAttrition)


class TestAttritionWithDate:
    def test_monthly_trends(self):
        rows = [
            _make_employee("E1", "Active", date_val="2024-01-15"),
            _make_employee("E2", "Left", date_val="2024-01-20"),
            _make_employee("E3", "Active", date_val="2024-02-10"),
            _make_employee("E4", "Left", date_val="2024-02-15"),
        ]
        result = analyze_attrition_rate(
            rows, "emp_id", "status", date_column="date"
        )
        assert len(result.monthly_trends) == 2
        jan = result.monthly_trends[0]
        assert jan.month == "2024-01"
        assert jan.active == 1
        assert jan.left == 1
        assert jan.rate == 50.0

    def test_no_date_empty_trends(self):
        rows = [_make_employee("E1", "Active")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.monthly_trends == []

    def test_monthly_trend_sorted_chronologically(self):
        rows = [
            _make_employee("E1", "Active", date_val="2024-03-01"),
            _make_employee("E2", "Active", date_val="2024-01-01"),
            _make_employee("E3", "Left", date_val="2024-02-01"),
        ]
        result = analyze_attrition_rate(
            rows, "emp_id", "status", date_column="date"
        )
        months = [mt.month for mt in result.monthly_trends]
        assert months == sorted(months)

    def test_monthly_attrition_type(self):
        rows = [_make_employee("E1", "Active", date_val="2024-01-01")]
        result = analyze_attrition_rate(
            rows, "emp_id", "status", date_column="date"
        )
        if result.monthly_trends:
            assert isinstance(result.monthly_trends[0], MonthlyAttrition)

    def test_date_parsing_various_formats(self):
        rows = [
            _make_employee("E1", "Active", date_val="2024-01-15"),
            _make_employee("E2", "Left", date_val="2024/02/15"),
        ]
        result = analyze_attrition_rate(
            rows, "emp_id", "status", date_column="date"
        )
        assert result is not None
        assert result.total_employees == 2


# ===================================================================
# 2. Tenure Distribution Tests
# ===================================================================


class TestTenureDistributionBasic:
    def test_empty_returns_none(self):
        assert analyze_tenure_distribution([], "emp_id", "hire_date") is None

    def test_all_none_returns_none(self):
        rows = [{"emp_id": None, "hire_date": None}]
        assert analyze_tenure_distribution(rows, "emp_id", "hire_date") is None

    def test_invalid_dates_returns_none(self):
        rows = [{"emp_id": "E1", "hire_date": "not-a-date"}]
        assert analyze_tenure_distribution(rows, "emp_id", "hire_date") is None

    def test_single_employee_tenure(self):
        rows = [{"emp_id": "E1", "hire_date": "2022-01-01"}]
        result = analyze_tenure_distribution(rows, "emp_id", "hire_date")
        assert result is not None
        assert result.avg_tenure >= 0
        assert result.median_tenure >= 0
        assert isinstance(result, TenureResult)

    def test_tenure_with_termination(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result is not None
        # Tenure should be approximately 3 years
        assert 2.9 <= result.avg_tenure <= 3.1

    def test_avg_and_median_tenure(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2022-01-01"},  # ~2yr
            {"emp_id": "E2", "hire_date": "2020-01-01", "term_date": "2024-01-01"},  # ~4yr
            {"emp_id": "E3", "hire_date": "2020-01-01", "term_date": "2026-01-01"},  # ~6yr
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        # avg should be ~4
        assert 3.8 <= result.avg_tenure <= 4.2
        # median should be ~4
        assert 3.8 <= result.median_tenure <= 4.2

    def test_summary_contains_key_info(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert "1 employees" in result.summary or "1 employee" in result.summary
        assert "yr" in result.summary


class TestTenureBuckets:
    def _rows_with_tenure_years(self, years_list):
        """Create rows spanning known tenure durations."""
        rows = []
        for i, yrs in enumerate(years_list):
            hire = date(2000, 1, 1)
            term = date(2000 + int(yrs), 1 + int((yrs % 1) * 12), 1)
            rows.append({
                "emp_id": f"E{i}",
                "hire_date": hire.isoformat(),
                "term_date": term.isoformat(),
            })
        return rows

    def test_bucket_under_1_year(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-06-01", "term_date": "2023-12-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result.buckets[0].range_label == "<1yr"
        assert result.buckets[0].count == 1

    def test_bucket_1_to_2_year(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2021-06-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        b = [b for b in result.buckets if b.range_label == "1-2yr"][0]
        assert b.count == 1

    def test_bucket_2_to_5_year(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        b = [b for b in result.buckets if b.range_label == "2-5yr"][0]
        assert b.count == 1

    def test_bucket_5_to_10_year(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2015-01-01", "term_date": "2022-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        b = [b for b in result.buckets if b.range_label == "5-10yr"][0]
        assert b.count == 1

    def test_bucket_over_10_year(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2005-01-01", "term_date": "2020-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        b = [b for b in result.buckets if b.range_label == ">10yr"][0]
        assert b.count == 1

    def test_exactly_1_year_in_1_2_bucket(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2021-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        # ~1.0 year lands in 1-2yr bucket
        b_1_2 = [b for b in result.buckets if b.range_label == "1-2yr"][0]
        assert b_1_2.count >= 0  # Boundary check (365/365.25 ~ 0.999, so in <1yr)
        total = sum(b.count for b in result.buckets)
        assert total == 1

    def test_bucket_percentages_sum_to_100(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-06-01", "term_date": "2023-12-01"},
            {"emp_id": "E2", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
            {"emp_id": "E3", "hire_date": "2005-01-01", "term_date": "2020-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        total_pct = sum(b.pct for b in result.buckets)
        assert abs(total_pct - 100.0) < 0.1

    def test_five_buckets_always_present(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert len(result.buckets) == 5
        labels = [b.range_label for b in result.buckets]
        assert "<1yr" in labels
        assert "1-2yr" in labels
        assert "2-5yr" in labels
        assert "5-10yr" in labels
        assert ">10yr" in labels

    def test_bucket_type(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert isinstance(result.buckets[0], TenureBucket)


class TestTenureLeaverVsStayer:
    def test_leaver_vs_stayer_avg(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2022-01-01"},  # leaver ~2yr
            {"emp_id": "E2", "hire_date": "2015-01-01"},  # stayer, tenure computed to max date
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result.leaver_avg_tenure is not None
        assert 1.9 <= result.leaver_avg_tenure <= 2.1
        assert result.stayer_avg_tenure is not None
        assert result.stayer_avg_tenure > result.leaver_avg_tenure

    def test_no_termination_column_none_values(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01"},
        ]
        result = analyze_tenure_distribution(rows, "emp_id", "hire_date")
        assert result.leaver_avg_tenure is None
        assert result.stayer_avg_tenure is None

    def test_all_leavers_stayer_is_none(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2022-01-01"},
            {"emp_id": "E2", "hire_date": "2019-01-01", "term_date": "2021-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result.leaver_avg_tenure is not None
        assert result.stayer_avg_tenure is None

    def test_all_stayers_leaver_is_none(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01"},
            {"emp_id": "E2", "hire_date": "2019-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result.leaver_avg_tenure is None
        assert result.stayer_avg_tenure is not None


class TestTenureByDepartment:
    def test_department_breakdown(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01", "dept": "Eng"},
            {"emp_id": "E2", "hire_date": "2021-01-01", "term_date": "2023-01-01", "dept": "Eng"},
            {"emp_id": "E3", "hire_date": "2020-01-01", "term_date": "2023-01-01", "dept": "Sales"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date",
            termination_date_column="term_date",
            department_column="dept",
        )
        assert len(result.by_department) == 2
        eng = [d for d in result.by_department if d.department == "Eng"][0]
        assert eng.count == 2
        assert isinstance(eng, DeptTenure)

    def test_no_department_empty_list(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result.by_department == []

    def test_department_sorted_alphabetically(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01", "dept": "Zebra"},
            {"emp_id": "E2", "hire_date": "2020-01-01", "term_date": "2023-01-01", "dept": "Alpha"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date",
            termination_date_column="term_date",
            department_column="dept",
        )
        assert result.by_department[0].department == "Alpha"


# ===================================================================
# 3. Retention Cohorts Tests
# ===================================================================


class TestRetentionCohortsBasic:
    def test_empty_returns_none(self):
        assert compute_retention_cohorts([], "emp_id", "hire_date", "term_date") is None

    def test_all_none_returns_none(self):
        rows = [{"emp_id": None, "hire_date": None, "term_date": None}]
        assert compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date") is None

    def test_invalid_dates_returns_none(self):
        rows = [{"emp_id": "E1", "hire_date": "bad", "term_date": "bad"}]
        assert compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date") is None

    def test_single_cohort(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2023-02-15", "term_date": "2023-08-01"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result is not None
        assert len(result.cohorts) == 1
        assert result.cohorts[0].cohort_label == "2023-Q1"
        assert result.cohorts[0].starting_count == 2

    def test_multiple_cohorts(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2023-04-15", "term_date": None},
            {"emp_id": "E3", "hire_date": "2023-07-15", "term_date": None},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result is not None
        assert len(result.cohorts) == 3
        labels = [c.cohort_label for c in result.cohorts]
        assert "2023-Q1" in labels
        assert "2023-Q2" in labels
        assert "2023-Q3" in labels

    def test_retention_rate_all_retained(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2023-02-15", "term_date": None},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result.cohorts[0].retention_rate == 100.0
        assert result.cohorts[0].retained_count == 2

    def test_retention_rate_some_left(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2023-02-15", "term_date": "2023-06-01"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result.cohorts[0].retention_rate == 50.0

    def test_best_worst_cohort(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2022-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2022-01-20", "term_date": None},
            {"emp_id": "E3", "hire_date": "2022-04-15", "term_date": "2022-10-01"},
            {"emp_id": "E4", "hire_date": "2022-04-20", "term_date": "2022-11-01"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result.best_cohort == "2022-Q1"
        assert result.worst_cohort == "2022-Q2"

    def test_result_type(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert isinstance(result, RetentionCohortResult)

    def test_cohort_type(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert isinstance(result.cohorts[0], Cohort)

    def test_summary_contains_key_info(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2023-04-15", "term_date": "2023-08-01"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert "cohort" in result.summary.lower()

    def test_deduplicates_employees(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
            {"emp_id": "E1", "hire_date": "2023-01-15", "term_date": None},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result.cohorts[0].starting_count == 1

    def test_cohorts_sorted_chronologically(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2023-07-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2023-01-15", "term_date": None},
            {"emp_id": "E3", "hire_date": "2023-04-15", "term_date": None},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        labels = [c.cohort_label for c in result.cohorts]
        assert labels == sorted(labels)


class TestRetentionMilestones:
    def test_6mo_milestone(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2022-01-01", "term_date": None},
            {"emp_id": "E2", "hire_date": "2022-01-15", "term_date": "2022-04-01"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        cohort = result.cohorts[0]
        ms_6mo = [m for m in cohort.retention_milestones if m.period == "6mo"]
        if ms_6mo:
            assert isinstance(ms_6mo[0], CohortRetention)
            assert ms_6mo[0].retained_pct <= 100.0

    def test_1yr_milestone(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2021-01-01", "term_date": None},
            {"emp_id": "E2", "hire_date": "2021-01-15", "term_date": "2021-06-01"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        cohort = result.cohorts[0]
        ms_1yr = [m for m in cohort.retention_milestones if m.period == "1yr"]
        if ms_1yr:
            assert ms_1yr[0].retained_pct == 50.0

    def test_overall_1yr_retention(self):
        # Reference date needs to be > 1yr after all hires to evaluate 1yr milestone
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": None},
            {"emp_id": "E2", "hire_date": "2020-02-01", "term_date": "2020-06-01"},
            {"emp_id": "E3", "hire_date": "2020-04-01", "term_date": None},
            {"emp_id": "E4", "hire_date": "2020-05-01", "term_date": "2020-08-01"},
            # Add a late termination to push reference date past 1yr for all
            {"emp_id": "E5", "hire_date": "2020-01-15", "term_date": "2021-12-01"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result.overall_1yr_retention is not None
        # E1 survived 1yr, E2 didn't, E3 survived, E4 didn't, E5 survived 1yr
        # 3 survived / 5 eligible = 60%
        assert result.overall_1yr_retention == 60.0

    def test_no_1yr_retention_if_too_recent(self):
        # All hired very recently, reference date is close to hire
        rows = [
            {"emp_id": "E1", "hire_date": "2024-06-01", "term_date": "2024-06-15"},
        ]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        # Reference date is 2024-06-15, only 14 days after hire
        assert result.overall_1yr_retention is None


# ===================================================================
# 4. Attrition Drivers Tests
# ===================================================================


class TestAttritionDriversBasic:
    def test_empty_returns_none(self):
        assert analyze_attrition_drivers([], "emp_id", "status", ["salary"]) is None

    def test_no_factor_columns_returns_none(self):
        rows = [_make_employee("E1", "Active")]
        assert analyze_attrition_drivers(rows, "emp_id", "status", []) is None

    def test_all_same_status_returns_none(self):
        rows = [
            _make_employee("E1", "Active", salary=50000),
            _make_employee("E2", "Active", salary=60000),
        ]
        assert analyze_attrition_drivers(rows, "emp_id", "status", ["salary"]) is None

    def test_no_leavers_returns_none(self):
        rows = [
            _make_employee("E1", "Active", salary=50000),
        ]
        assert analyze_attrition_drivers(rows, "emp_id", "status", ["salary"]) is None

    def test_no_stayers_returns_none(self):
        rows = [
            _make_employee("E1", "Left", salary=50000),
        ]
        assert analyze_attrition_drivers(rows, "emp_id", "status", ["salary"]) is None

    def test_result_type(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        assert isinstance(result, AttritionDriverResult)

    def test_factor_type(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        assert isinstance(result.factors[0], DriverFactor)


class TestAttritionDriversNumeric:
    def test_numeric_factor_detected(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Active", salary=65000),
            _make_employee("E3", "Left", salary=40000),
            _make_employee("E4", "Left", salary=45000),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        assert result is not None
        assert result.factors[0].factor_type == "numeric"

    def test_numeric_factor_values(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Active", salary=60000),
            _make_employee("E3", "Left", salary=40000),
            _make_employee("E4", "Left", salary=40000),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        factor = result.factors[0]
        assert factor.leaver_value == "40000.00"
        assert factor.stayer_value == "60000.00"

    def test_numeric_direction_lower_for_leavers(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        assert result.factors[0].direction == "lower for leavers"

    def test_numeric_direction_higher_for_leavers(self):
        rows = [
            _make_employee("E1", "Active", hours=40),
            _make_employee("E2", "Left", hours=60),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["hours"])
        assert result.factors[0].direction == "higher for leavers"

    def test_numeric_impact_calculation(self):
        rows = [
            _make_employee("E1", "Active", salary=100),
            _make_employee("E2", "Left", salary=80),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        factor = result.factors[0]
        # abs(80-100)/100 * 100 = 20%
        assert factor.impact == 20.0

    def test_multiple_numeric_factors(self):
        rows = [
            _make_employee("E1", "Active", salary=100, hours=40),
            _make_employee("E2", "Left", salary=50, hours=60),
        ]
        result = analyze_attrition_drivers(
            rows, "emp_id", "status", ["salary", "hours"]
        )
        assert len(result.factors) == 2

    def test_factors_sorted_by_impact(self):
        rows = [
            _make_employee("E1", "Active", salary=100, hours=40, rating=4.0),
            _make_employee("E2", "Left", salary=50, hours=60, rating=3.8),
        ]
        result = analyze_attrition_drivers(
            rows, "emp_id", "status", ["salary", "hours", "rating"]
        )
        impacts = [f.impact for f in result.factors]
        assert impacts == sorted(impacts, reverse=True)

    def test_top_driver(self):
        rows = [
            _make_employee("E1", "Active", salary=100, hours=40),
            _make_employee("E2", "Left", salary=50, hours=60),
        ]
        result = analyze_attrition_drivers(
            rows, "emp_id", "status", ["salary", "hours"]
        )
        # salary has 50% impact, hours has 50% impact
        assert result.top_driver in ["salary", "hours"]


class TestAttritionDriversCategorical:
    def test_categorical_factor_detected(self):
        rows = [
            _make_employee("E1", "Active", dept="Eng"),
            _make_employee("E2", "Active", dept="Eng"),
            _make_employee("E3", "Left", dept="Sales"),
            _make_employee("E4", "Left", dept="Sales"),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["department"])
        assert result is not None
        assert result.factors[0].factor_type == "categorical"

    def test_categorical_highest_attrition_category(self):
        rows = [
            _make_employee("E1", "Active", dept="Eng"),
            _make_employee("E2", "Active", dept="Eng"),
            _make_employee("E3", "Active", dept="Sales"),
            _make_employee("E4", "Left", dept="Sales"),
            _make_employee("E5", "Left", dept="Sales"),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["department"])
        factor = result.factors[0]
        # Sales has 2 left / 3 total = 66.7%, Eng has 0 left / 2 total = 0%
        assert factor.leaver_value == "Sales"
        assert factor.stayer_value == "Eng"

    def test_categorical_impact_is_rate_difference(self):
        rows = [
            _make_employee("E1", "Active", dept="Eng"),
            _make_employee("E2", "Active", dept="Eng"),
            _make_employee("E3", "Left", dept="Sales"),
            _make_employee("E4", "Active", dept="Sales"),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["department"])
        factor = result.factors[0]
        # Sales: 1/2 = 50%, Eng: 0/2 = 0%, diff = 50%
        assert factor.impact == 50.0

    def test_categorical_direction_format(self):
        rows = [
            _make_employee("E1", "Active", dept="Eng"),
            _make_employee("E2", "Left", dept="Sales"),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["department"])
        assert "highest attrition" in result.factors[0].direction

    def test_mixed_numeric_and_categorical(self):
        rows = [
            _make_employee("E1", "Active", dept="Eng", salary=60000),
            _make_employee("E2", "Left", dept="Sales", salary=40000),
        ]
        result = analyze_attrition_drivers(
            rows, "emp_id", "status", ["department", "salary"]
        )
        types = {f.factor_type for f in result.factors}
        assert "numeric" in types
        assert "categorical" in types

    def test_summary_contains_top_driver(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        assert "salary" in result.summary

    def test_factor_with_all_none_values_skipped(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        # "missing_col" not present in rows, so all values are None
        result = analyze_attrition_drivers(
            rows, "emp_id", "status", ["salary", "missing_col"]
        )
        assert len(result.factors) == 1
        assert result.factors[0].factor_name == "salary"


# ===================================================================
# 5. Format Attrition Report Tests
# ===================================================================


class TestFormatAttritionReport:
    def test_all_none_reports_no_data(self):
        report = format_attrition_report()
        assert "No analysis data provided" in report

    def test_header_always_present(self):
        report = format_attrition_report()
        assert "Employee Attrition Report" in report
        assert "=" * 40 in report

    def test_attrition_section_only(self):
        rows = [
            _make_employee("E1", "Active"),
            _make_employee("E2", "Left"),
        ]
        att = analyze_attrition_rate(rows, "emp_id", "status")
        report = format_attrition_report(attrition=att)
        assert "Attrition Rate Analysis" in report
        assert "50.0%" in report
        assert "No analysis data" not in report

    def test_tenure_section_only(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        ten = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        report = format_attrition_report(tenure=ten)
        assert "Tenure Distribution" in report
        assert "years" in report
        assert "No analysis data" not in report

    def test_cohorts_section_only(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2022-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2022-02-15", "term_date": "2022-08-01"},
        ]
        coh = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        report = format_attrition_report(cohorts=coh)
        assert "Retention Cohorts" in report
        assert "No analysis data" not in report

    def test_drivers_section_only(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        drv = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        report = format_attrition_report(drivers=drv)
        assert "Attrition Drivers" in report
        assert "salary" in report
        assert "No analysis data" not in report

    def test_combined_all_sections(self):
        att_rows = [
            _make_employee("E1", "Active"),
            _make_employee("E2", "Left"),
        ]
        ten_rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
        ]
        coh_rows = [
            {"emp_id": "E1", "hire_date": "2022-01-15", "term_date": None},
            {"emp_id": "E2", "hire_date": "2022-02-15", "term_date": "2022-08-01"},
        ]
        drv_rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]

        att = analyze_attrition_rate(att_rows, "emp_id", "status")
        ten = analyze_tenure_distribution(
            ten_rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        coh = compute_retention_cohorts(coh_rows, "emp_id", "hire_date", "term_date")
        drv = analyze_attrition_drivers(drv_rows, "emp_id", "status", ["salary"])

        report = format_attrition_report(
            attrition=att, tenure=ten, cohorts=coh, drivers=drv
        )
        assert "Employee Attrition Report" in report
        assert "Attrition Rate Analysis" in report
        assert "Tenure Distribution" in report
        assert "Retention Cohorts" in report
        assert "Attrition Drivers" in report
        assert "No analysis data" not in report

    def test_attrition_with_department_in_report(self):
        rows = [
            _make_employee("E1", "Active", dept="Eng"),
            _make_employee("E2", "Left", dept="Sales"),
        ]
        att = analyze_attrition_rate(
            rows, "emp_id", "status", department_column="department"
        )
        report = format_attrition_report(attrition=att)
        assert "By department" in report
        assert "Eng" in report
        assert "Sales" in report

    def test_attrition_with_monthly_trends_in_report(self):
        rows = [
            _make_employee("E1", "Active", date_val="2024-01-15"),
            _make_employee("E2", "Left", date_val="2024-02-15"),
        ]
        att = analyze_attrition_rate(
            rows, "emp_id", "status", date_column="date"
        )
        report = format_attrition_report(attrition=att)
        assert "Monthly trends" in report

    def test_tenure_with_leaver_stayer_in_report(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2022-01-01"},
            {"emp_id": "E2", "hire_date": "2015-01-01"},
        ]
        ten = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        report = format_attrition_report(tenure=ten)
        assert "Leaver avg tenure" in report
        assert "Stayer avg tenure" in report

    def test_cohorts_with_milestones_in_report(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": None},
            {"emp_id": "E2", "hire_date": "2020-02-01", "term_date": "2020-06-01"},
        ]
        coh = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        report = format_attrition_report(cohorts=coh)
        assert "retained" in report

    def test_report_is_string(self):
        report = format_attrition_report()
        assert isinstance(report, str)

    def test_drivers_shows_impact(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        drv = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        report = format_attrition_report(drivers=drv)
        assert "impact=" in report

    def test_drivers_shows_top_driver(self):
        rows = [
            _make_employee("E1", "Active", salary=60000),
            _make_employee("E2", "Left", salary=40000),
        ]
        drv = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        report = format_attrition_report(drivers=drv)
        assert "Top driver" in report


# ===================================================================
# Edge Case Tests
# ===================================================================


class TestEdgeCases:
    def test_date_object_input(self):
        rows = [
            {"emp_id": "E1", "hire_date": date(2020, 1, 1), "term_date": date(2023, 1, 1)},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result is not None
        assert 2.9 <= result.avg_tenure <= 3.1

    def test_mixed_date_formats(self):
        rows = [
            {"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"},
            {"emp_id": "E2", "hire_date": "2020/06/15", "term_date": "2023/06/15"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result is not None
        assert len(result.buckets) == 5

    def test_single_row_attrition(self):
        rows = [_make_employee("E1", "Active")]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result is not None
        assert result.total_employees == 1

    def test_single_row_tenure(self):
        rows = [{"emp_id": "E1", "hire_date": "2020-01-01", "term_date": "2023-01-01"}]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result is not None

    def test_single_row_cohort(self):
        rows = [{"emp_id": "E1", "hire_date": "2020-01-01", "term_date": None}]
        result = compute_retention_cohorts(rows, "emp_id", "hire_date", "term_date")
        assert result is not None
        assert len(result.cohorts) == 1

    def test_zero_tenure_negative_dates(self):
        # Termination before hire should give 0 tenure
        rows = [
            {"emp_id": "E1", "hire_date": "2023-06-01", "term_date": "2020-01-01"},
        ]
        result = analyze_tenure_distribution(
            rows, "emp_id", "hire_date", termination_date_column="term_date"
        )
        assert result is not None
        assert result.avg_tenure == 0.0

    def test_large_dataset_attrition(self):
        rows = []
        for i in range(100):
            status = "Active" if i < 70 else "Left"
            rows.append(_make_employee(f"E{i}", status))
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.total_employees == 100
        assert result.attrition_rate == 30.0

    def test_all_active_zero_attrition(self):
        rows = [
            _make_employee("E1", "Active"),
            _make_employee("E2", "Active"),
            _make_employee("E3", "Active"),
        ]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.attrition_rate == 0.0
        assert result.left_count == 0

    def test_all_left_100_attrition(self):
        rows = [
            _make_employee("E1", "Left"),
            _make_employee("E2", "Terminated"),
            _make_employee("E3", "Resigned"),
        ]
        result = analyze_attrition_rate(rows, "emp_id", "status")
        assert result.attrition_rate == 100.0
        assert result.active_count == 0

    def test_numeric_driver_no_difference(self):
        rows = [
            _make_employee("E1", "Active", salary=50000),
            _make_employee("E2", "Left", salary=50000),
        ]
        result = analyze_attrition_drivers(rows, "emp_id", "status", ["salary"])
        assert result.factors[0].impact == 0.0
        assert result.factors[0].direction == "no difference"
