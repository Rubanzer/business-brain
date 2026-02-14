"""Tests for workforce and HR analytics module."""

from business_brain.discovery.workforce_analytics import (
    AttendanceResult,
    DeptHeadcount,
    EmployeeAttendance,
    EmployeeOvertime,
    EntityProductivity,
    HeadcountResult,
    OvertimeResult,
    ProductivityResult,
    analyze_attendance,
    analyze_overtime,
    compute_labor_productivity,
    format_workforce_report,
    headcount_analysis,
)


# ===================================================================
# Attendance Tests
# ===================================================================


class TestAnalyzeAttendance:
    def test_basic_attendance(self):
        rows = [
            {"emp": "Alice", "status": "present"},
            {"emp": "Alice", "status": "present"},
            {"emp": "Alice", "status": "absent"},
            {"emp": "Bob", "status": "present"},
            {"emp": "Bob", "status": "leave"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        assert result is not None
        assert result.total_employees == 2
        alice = [e for e in result.employees if e.employee == "Alice"][0]
        assert alice.present_days == 2
        assert alice.absent_days == 1
        assert alice.total_days == 3
        assert abs(alice.attendance_rate - 66.67) < 0.01

    def test_empty_returns_none(self):
        assert analyze_attendance([], "emp", "status") is None

    def test_all_none_values_returns_none(self):
        rows = [{"emp": None, "status": None}]
        assert analyze_attendance(rows, "emp", "status") is None

    def test_perfect_attendance(self):
        rows = [
            {"emp": "Alice", "status": "present"},
            {"emp": "Alice", "status": "present"},
            {"emp": "Alice", "status": "present"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        assert result is not None
        assert "Alice" in result.perfect_attendance
        assert result.employees[0].attendance_rate == 100.0

    def test_chronic_absentee(self):
        rows = [
            {"emp": "Lazy", "status": "present"},
            {"emp": "Lazy", "status": "absent"},
            {"emp": "Lazy", "status": "absent"},
            {"emp": "Lazy", "status": "absent"},
            {"emp": "Lazy", "status": "absent"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        assert result is not None
        assert "Lazy" in result.chronic_absentees
        assert result.employees[0].attendance_rate == 20.0

    def test_leave_status_variants(self):
        rows = [
            {"emp": "A", "status": "leave"},
            {"emp": "B", "status": "sick"},
            {"emp": "C", "status": "vacation"},
            {"emp": "D", "status": "holiday"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        assert result is not None
        for e in result.employees:
            assert e.leave_days == 1
            assert e.present_days == 0

    def test_present_status_variants(self):
        rows = [
            {"emp": "A", "status": "present"},
            {"emp": "B", "status": "P"},
            {"emp": "C", "status": "1"},
            {"emp": "D", "status": "yes"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        assert result is not None
        for e in result.employees:
            assert e.present_days == 1
            assert e.attendance_rate == 100.0

    def test_sorted_by_rate_descending(self):
        rows = [
            {"emp": "Low", "status": "absent"},
            {"emp": "Low", "status": "absent"},
            {"emp": "Low", "status": "present"},
            {"emp": "High", "status": "present"},
            {"emp": "High", "status": "present"},
            {"emp": "High", "status": "present"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        rates = [e.attendance_rate for e in result.employees]
        assert rates == sorted(rates, reverse=True)

    def test_summary_contains_key_info(self):
        rows = [
            {"emp": "Alice", "status": "present"},
            {"emp": "Bob", "status": "absent"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        assert "2 employees" in result.summary
        assert "%" in result.summary

    def test_unknown_status_counts_as_absent(self):
        rows = [
            {"emp": "Alice", "status": "unknown_xyz"},
        ]
        result = analyze_attendance(rows, "emp", "status")
        assert result is not None
        assert result.employees[0].absent_days == 1
        assert result.employees[0].present_days == 0


# ===================================================================
# Labor Productivity Tests
# ===================================================================


class TestComputeLaborProductivity:
    def test_basic_productivity(self):
        rows = [
            {"worker": "A", "output": 100, "hours": 8},
            {"worker": "B", "output": 80, "hours": 8},
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        assert result is not None
        a = [e for e in result.entities if e.entity == "A"][0]
        assert a.productivity == 12.5  # 100/8
        b = [e for e in result.entities if e.entity == "B"][0]
        assert b.productivity == 10.0  # 80/8

    def test_empty_returns_none(self):
        assert compute_labor_productivity([], "w", "o", "h") is None

    def test_all_none_returns_none(self):
        rows = [{"worker": None, "output": None, "hours": None}]
        assert compute_labor_productivity(rows, "worker", "output", "hours") is None

    def test_productivity_index_best_is_100(self):
        rows = [
            {"worker": "Best", "output": 200, "hours": 8},
            {"worker": "Avg", "output": 100, "hours": 8},
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        best = [e for e in result.entities if e.entity == "Best"][0]
        avg = [e for e in result.entities if e.entity == "Avg"][0]
        assert best.productivity_index == 100.0
        assert abs(avg.productivity_index - 50.0) < 0.01

    def test_best_worst_entity(self):
        rows = [
            {"worker": "Star", "output": 500, "hours": 8},
            {"worker": "Slow", "output": 50, "hours": 8},
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        assert result.best_entity == "Star"
        assert result.worst_entity == "Slow"

    def test_spread_ratio(self):
        rows = [
            {"worker": "A", "output": 200, "hours": 10},  # 20
            {"worker": "B", "output": 100, "hours": 10},  # 10
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        assert result.spread_ratio == 2.0

    def test_zero_hours_zero_productivity(self):
        rows = [
            {"worker": "A", "output": 100, "hours": 0},
            {"worker": "B", "output": 100, "hours": 10},
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        a = [e for e in result.entities if e.entity == "A"][0]
        assert a.productivity == 0.0

    def test_aggregates_multiple_rows(self):
        rows = [
            {"worker": "A", "output": 50, "hours": 4},
            {"worker": "A", "output": 50, "hours": 4},
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        a = result.entities[0]
        assert a.total_output == 100.0
        assert a.total_hours == 8.0
        assert a.productivity == 12.5

    def test_sorted_by_productivity_descending(self):
        rows = [
            {"worker": "Low", "output": 10, "hours": 10},
            {"worker": "High", "output": 100, "hours": 10},
            {"worker": "Mid", "output": 50, "hours": 10},
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        prods = [e.productivity for e in result.entities]
        assert prods == sorted(prods, reverse=True)

    def test_summary_contains_key_info(self):
        rows = [
            {"worker": "A", "output": 100, "hours": 8},
        ]
        result = compute_labor_productivity(rows, "worker", "output", "hours")
        assert "1 entities" in result.summary or "1 entit" in result.summary
        assert "units/hr" in result.summary


# ===================================================================
# Overtime Tests
# ===================================================================


class TestAnalyzeOvertime:
    def test_basic_overtime(self):
        rows = [
            {"emp": "Alice", "regular": 8, "actual": 10},
            {"emp": "Bob", "regular": 8, "actual": 8},
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        assert result is not None
        alice = [e for e in result.employees if e.employee == "Alice"][0]
        assert alice.overtime_hours == 2.0
        bob = [e for e in result.employees if e.employee == "Bob"][0]
        assert bob.overtime_hours == 0.0

    def test_empty_returns_none(self):
        assert analyze_overtime([], "emp", "r", "a") is None

    def test_all_none_returns_none(self):
        rows = [{"emp": None, "regular": None, "actual": None}]
        assert analyze_overtime(rows, "emp", "regular", "actual") is None

    def test_with_rate_column(self):
        rows = [
            {"emp": "Alice", "regular": 8, "actual": 10, "rate": 25.0},
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual", rate_column="rate")
        alice = result.employees[0]
        assert alice.overtime_cost == 50.0  # 2 OT hours * 25.0

    def test_no_rate_column_cost_is_none(self):
        rows = [
            {"emp": "Alice", "regular": 8, "actual": 10},
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        assert result.employees[0].overtime_cost is None

    def test_overtime_pct(self):
        rows = [
            {"emp": "Alice", "regular": 8, "actual": 10},  # 2 OT / 8 reg = 25%
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        assert result.employees[0].overtime_pct == 25.0

    def test_no_overtime_when_actual_less_than_regular(self):
        rows = [
            {"emp": "Alice", "regular": 8, "actual": 6},
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        assert result.employees[0].overtime_hours == 0.0

    def test_top_overtime_employees(self):
        rows = [
            {"emp": "Low", "regular": 8, "actual": 9},
            {"emp": "High", "regular": 8, "actual": 14},
            {"emp": "Mid", "regular": 8, "actual": 11},
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        assert result.top_overtime_employees[0] == "High"

    def test_total_overtime_hours(self):
        rows = [
            {"emp": "Alice", "regular": 8, "actual": 10},  # 2 OT
            {"emp": "Bob", "regular": 8, "actual": 12},    # 4 OT
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        assert result.total_overtime_hours == 6.0

    def test_overall_overtime_pct(self):
        rows = [
            {"emp": "A", "regular": 10, "actual": 12},  # 2 OT
            {"emp": "B", "regular": 10, "actual": 13},  # 3 OT
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        # total_ot=5, total_reg=20, pct=25%
        assert result.overtime_pct == 25.0

    def test_summary_contains_key_info(self):
        rows = [
            {"emp": "Alice", "regular": 8, "actual": 10},
        ]
        result = analyze_overtime(rows, "emp", "regular", "actual")
        assert "1 employees" in result.summary or "1 employee" in result.summary
        assert "OT" in result.summary


# ===================================================================
# Headcount Tests
# ===================================================================


class TestHeadcountAnalysis:
    def test_basic_headcount(self):
        rows = [
            {"dept": "Engineering", "emp": "Alice"},
            {"dept": "Engineering", "emp": "Bob"},
            {"dept": "Sales", "emp": "Charlie"},
        ]
        result = headcount_analysis(rows, "dept", "emp")
        assert result is not None
        assert result.total_headcount == 3
        eng = [d for d in result.departments if d.department == "Engineering"][0]
        assert eng.headcount == 2
        sales = [d for d in result.departments if d.department == "Sales"][0]
        assert sales.headcount == 1

    def test_empty_returns_none(self):
        assert headcount_analysis([], "dept", "emp") is None

    def test_all_none_returns_none(self):
        rows = [{"dept": None, "emp": None}]
        assert headcount_analysis(rows, "dept", "emp") is None

    def test_unique_employee_count(self):
        rows = [
            {"dept": "Eng", "emp": "Alice"},
            {"dept": "Eng", "emp": "Alice"},  # duplicate
            {"dept": "Eng", "emp": "Bob"},
        ]
        result = headcount_analysis(rows, "dept", "emp")
        eng = result.departments[0]
        assert eng.headcount == 2  # Alice counted once

    def test_with_output_column(self):
        rows = [
            {"dept": "Eng", "emp": "Alice", "output": 100},
            {"dept": "Eng", "emp": "Bob", "output": 200},
            {"dept": "Sales", "emp": "Charlie", "output": 50},
        ]
        result = headcount_analysis(rows, "dept", "emp", output_column="output")
        eng = [d for d in result.departments if d.department == "Eng"][0]
        assert eng.output_total == 300.0
        assert eng.output_per_head == 150.0  # 300 / 2

    def test_no_output_column_nulls(self):
        rows = [
            {"dept": "Eng", "emp": "Alice"},
        ]
        result = headcount_analysis(rows, "dept", "emp")
        assert result.departments[0].output_total is None
        assert result.departments[0].output_per_head is None
        assert result.output_per_head is None

    def test_largest_smallest_dept(self):
        rows = [
            {"dept": "Big", "emp": "A"},
            {"dept": "Big", "emp": "B"},
            {"dept": "Big", "emp": "C"},
            {"dept": "Small", "emp": "D"},
        ]
        result = headcount_analysis(rows, "dept", "emp")
        assert result.largest_dept == "Big"
        assert result.smallest_dept == "Small"

    def test_share_pct(self):
        rows = [
            {"dept": "A", "emp": "E1"},
            {"dept": "A", "emp": "E2"},
            {"dept": "A", "emp": "E3"},
            {"dept": "B", "emp": "E4"},
        ]
        result = headcount_analysis(rows, "dept", "emp")
        dept_a = [d for d in result.departments if d.department == "A"][0]
        assert dept_a.share_pct == 75.0

    def test_sorted_by_headcount_descending(self):
        rows = [
            {"dept": "Small", "emp": "E1"},
            {"dept": "Big", "emp": "E2"},
            {"dept": "Big", "emp": "E3"},
            {"dept": "Big", "emp": "E4"},
            {"dept": "Mid", "emp": "E5"},
            {"dept": "Mid", "emp": "E6"},
        ]
        result = headcount_analysis(rows, "dept", "emp")
        counts = [d.headcount for d in result.departments]
        assert counts == sorted(counts, reverse=True)

    def test_summary_contains_key_info(self):
        rows = [
            {"dept": "Eng", "emp": "Alice"},
            {"dept": "Sales", "emp": "Bob"},
        ]
        result = headcount_analysis(rows, "dept", "emp")
        assert "2 departments" in result.summary
        assert "Total = 2" in result.summary


# ===================================================================
# Combined Report Tests
# ===================================================================


class TestFormatWorkforceReport:
    def test_all_none_reports_no_data(self):
        report = format_workforce_report()
        assert "No analysis data provided" in report

    def test_header_always_present(self):
        report = format_workforce_report()
        assert "Workforce Analytics Report" in report
        assert "=" * 40 in report

    def test_attendance_only(self):
        rows = [
            {"emp": "Alice", "status": "present"},
            {"emp": "Bob", "status": "absent"},
        ]
        att = analyze_attendance(rows, "emp", "status")
        report = format_workforce_report(attendance=att)
        assert "Attendance" in report
        assert "Alice" in report

    def test_productivity_only(self):
        rows = [{"worker": "A", "output": 100, "hours": 8}]
        prod = compute_labor_productivity(rows, "worker", "output", "hours")
        report = format_workforce_report(productivity=prod)
        assert "Productivity" in report
        assert "units/hr" in report

    def test_overtime_only(self):
        rows = [{"emp": "Alice", "regular": 8, "actual": 10}]
        ot = analyze_overtime(rows, "emp", "regular", "actual")
        report = format_workforce_report(overtime=ot)
        assert "Overtime" in report

    def test_headcount_only(self):
        rows = [
            {"dept": "Eng", "emp": "Alice"},
            {"dept": "Sales", "emp": "Bob"},
        ]
        hc = headcount_analysis(rows, "dept", "emp")
        report = format_workforce_report(headcount=hc)
        assert "Headcount" in report
        assert "Eng" in report

    def test_combined_all_sections(self):
        att_rows = [{"emp": "Alice", "status": "present"}]
        prod_rows = [{"worker": "A", "output": 100, "hours": 8}]
        ot_rows = [{"emp": "Alice", "regular": 8, "actual": 10}]
        hc_rows = [{"dept": "Eng", "emp": "Alice"}]

        att = analyze_attendance(att_rows, "emp", "status")
        prod = compute_labor_productivity(prod_rows, "worker", "output", "hours")
        ot = analyze_overtime(ot_rows, "emp", "regular", "actual")
        hc = headcount_analysis(hc_rows, "dept", "emp")

        report = format_workforce_report(
            attendance=att,
            productivity=prod,
            overtime=ot,
            headcount=hc,
        )
        assert "Workforce Analytics Report" in report
        assert "Attendance" in report
        assert "Productivity" in report
        assert "Overtime" in report
        assert "Headcount" in report

    def test_overtime_with_cost_in_report(self):
        rows = [{"emp": "Alice", "regular": 8, "actual": 10, "rate": 25.0}]
        ot = analyze_overtime(rows, "emp", "regular", "actual", rate_column="rate")
        report = format_workforce_report(overtime=ot)
        assert "cost=" in report
