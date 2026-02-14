"""Tests for production scheduler pure functions."""

from business_brain.discovery.production_scheduler import (
    ShiftPerformance,
    ShiftPerformanceResult,
    BatchInfo,
    BatchResult,
    TaktResult,
    PlanEntity,
    PlanResult,
    analyze_shift_performance,
    analyze_batch_efficiency,
    compute_takt_time,
    plan_vs_actual,
    format_schedule_report,
)


# ---------------------------------------------------------------------------
# analyze_shift_performance
# ---------------------------------------------------------------------------


class TestAnalyzeShiftPerformance:
    def test_basic_two_shifts(self):
        rows = [
            {"shift": "A", "output": 100},
            {"shift": "A", "output": 120},
            {"shift": "B", "output": 80},
            {"shift": "B", "output": 90},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is not None
        assert len(result.shifts) == 2
        assert result.best_shift == "A"
        assert result.worst_shift == "B"
        assert result.total_output == 390.0

    def test_single_shift(self):
        rows = [
            {"shift": "Morning", "output": 200},
            {"shift": "Morning", "output": 250},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is not None
        assert len(result.shifts) == 1
        assert result.best_shift == "Morning"
        assert result.worst_shift == "Morning"
        assert result.variance_pct == 0.0

    def test_with_target_column(self):
        rows = [
            {"shift": "A", "output": 90, "target": 100},
            {"shift": "A", "output": 95, "target": 100},
            {"shift": "B", "output": 110, "target": 100},
            {"shift": "B", "output": 105, "target": 100},
        ]
        result = analyze_shift_performance(rows, "shift", "output", target_column="target")
        assert result is not None
        shift_a = [s for s in result.shifts if s.shift == "A"][0]
        shift_b = [s for s in result.shifts if s.shift == "B"][0]
        assert shift_a.achievement_pct == 92.5  # 185/200 * 100
        assert shift_b.achievement_pct == 107.5  # 215/200 * 100

    def test_empty_rows(self):
        result = analyze_shift_performance([], "shift", "output")
        assert result is None

    def test_all_none_values(self):
        rows = [
            {"shift": "A", "output": None},
            {"shift": None, "output": 100},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is None

    def test_consistency_grade_a(self):
        # CV < 10%: very consistent values
        rows = [
            {"shift": "X", "output": 100},
            {"shift": "X", "output": 102},
            {"shift": "X", "output": 98},
            {"shift": "X", "output": 101},
            {"shift": "X", "output": 99},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is not None
        assert result.shifts[0].consistency_grade == "A"

    def test_consistency_grade_d(self):
        # CV >= 30%: very inconsistent
        rows = [
            {"shift": "X", "output": 10},
            {"shift": "X", "output": 50},
            {"shift": "X", "output": 100},
            {"shift": "X", "output": 5},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is not None
        assert result.shifts[0].consistency_grade == "D"

    def test_single_event_per_shift(self):
        rows = [
            {"shift": "A", "output": 100},
            {"shift": "B", "output": 200},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is not None
        shift_a = [s for s in result.shifts if s.shift == "A"][0]
        assert shift_a.event_count == 1
        assert shift_a.std_dev == 0.0

    def test_mixed_valid_and_invalid_rows(self):
        rows = [
            {"shift": "A", "output": 100},
            {"shift": "A", "output": "bad_value"},
            {"shift": "A", "output": 150},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is not None
        assert result.shifts[0].event_count == 2
        assert result.shifts[0].total_output == 250.0

    def test_summary_contains_key_info(self):
        rows = [
            {"shift": "Day", "output": 500},
            {"shift": "Night", "output": 300},
        ]
        result = analyze_shift_performance(rows, "shift", "output")
        assert result is not None
        assert "Day" in result.summary
        assert "Night" in result.summary


# ---------------------------------------------------------------------------
# analyze_batch_efficiency
# ---------------------------------------------------------------------------


class TestAnalyzeBatchEfficiency:
    def test_basic_batches(self):
        rows = [
            {"batch": "B001", "input": 100, "output": 90},
            {"batch": "B002", "input": 100, "output": 85},
            {"batch": "B003", "input": 100, "output": 95},
        ]
        result = analyze_batch_efficiency(rows, "batch", "input", "output")
        assert result is not None
        assert len(result.batches) == 3
        assert result.best_batch == "B003"
        assert result.worst_batch == "B002"
        assert result.mean_yield == 90.0

    def test_with_duration(self):
        rows = [
            {"batch": "B001", "input": 100, "output": 90, "duration": 60},
            {"batch": "B002", "input": 100, "output": 80, "duration": 40},
        ]
        result = analyze_batch_efficiency(rows, "batch", "input", "output", duration_column="duration")
        assert result is not None
        b1 = [b for b in result.batches if b.batch_id == "B001"][0]
        b2 = [b for b in result.batches if b.batch_id == "B002"][0]
        assert b1.throughput == 1.5  # 90/60
        assert b2.throughput == 2.0  # 80/40
        assert result.mean_throughput is not None

    def test_empty_rows(self):
        result = analyze_batch_efficiency([], "batch", "input", "output")
        assert result is None

    def test_all_none_values(self):
        rows = [
            {"batch": "B001", "input": None, "output": 90},
            {"batch": "B002", "input": 100, "output": None},
        ]
        result = analyze_batch_efficiency(rows, "batch", "input", "output")
        assert result is None

    def test_zero_input(self):
        rows = [
            {"batch": "B001", "input": 0, "output": 0},
            {"batch": "B002", "input": 100, "output": 90},
        ]
        result = analyze_batch_efficiency(rows, "batch", "input", "output")
        assert result is not None
        b1 = [b for b in result.batches if b.batch_id == "B001"][0]
        assert b1.yield_pct == 0.0

    def test_optimal_batch_size(self):
        rows = [
            {"batch": "B001", "input": 50, "output": 48},    # 96%
            {"batch": "B002", "input": 100, "output": 92},   # 92%
            {"batch": "B003", "input": 200, "output": 170},  # 85%
        ]
        result = analyze_batch_efficiency(rows, "batch", "input", "output")
        assert result is not None
        # B001 has best yield (96%), so optimal is 50
        assert result.optimal_batch_size == 50.0

    def test_no_duration_throughput_is_none(self):
        rows = [
            {"batch": "B001", "input": 100, "output": 90},
        ]
        result = analyze_batch_efficiency(rows, "batch", "input", "output")
        assert result is not None
        assert result.batches[0].throughput is None
        assert result.mean_throughput is None

    def test_aggregates_multiple_rows_per_batch(self):
        rows = [
            {"batch": "B001", "input": 50, "output": 45},
            {"batch": "B001", "input": 50, "output": 44},
        ]
        result = analyze_batch_efficiency(rows, "batch", "input", "output")
        assert result is not None
        assert result.batches[0].input_qty == 100.0
        assert result.batches[0].output_qty == 89.0


# ---------------------------------------------------------------------------
# compute_takt_time
# ---------------------------------------------------------------------------


class TestComputeTaktTime:
    def test_basic_takt(self):
        rows = [
            {"period": "Day1", "output": 480},
            {"period": "Day2", "output": 480},
        ]
        result = compute_takt_time(rows, "output", "period", available_minutes_per_period=480)
        assert result is not None
        assert result.takt_time_minutes == 1.0  # 480/480
        assert result.total_output == 960.0
        assert result.total_periods == 2

    def test_empty_rows(self):
        result = compute_takt_time([], "output", "period")
        assert result is None

    def test_zero_output(self):
        rows = [
            {"period": "Day1", "output": 0},
            {"period": "Day2", "output": 0},
        ]
        result = compute_takt_time(rows, "output", "period")
        assert result is not None
        assert result.takt_time_minutes == 0.0
        assert result.cycle_time_estimate == 0.0
        assert result.efficiency_pct == 0.0

    def test_all_none_output(self):
        rows = [
            {"period": "Day1", "output": None},
            {"period": "Day2", "output": None},
        ]
        result = compute_takt_time(rows, "output", "period")
        assert result is None

    def test_custom_available_minutes(self):
        rows = [
            {"period": "Day1", "output": 120},
        ]
        result = compute_takt_time(rows, "output", "period", available_minutes_per_period=600)
        assert result is not None
        assert result.takt_time_minutes == 5.0  # 600/120

    def test_efficiency_100_pct(self):
        # When takt == cycle, efficiency = 100%
        rows = [
            {"period": "D1", "output": 240},
            {"period": "D2", "output": 240},
        ]
        result = compute_takt_time(rows, "output", "period", available_minutes_per_period=480)
        assert result is not None
        assert result.efficiency_pct == 100.0

    def test_aggregates_within_period(self):
        rows = [
            {"period": "D1", "output": 100},
            {"period": "D1", "output": 100},
            {"period": "D2", "output": 200},
        ]
        result = compute_takt_time(rows, "output", "period", available_minutes_per_period=480)
        assert result is not None
        assert result.total_output == 400.0
        assert result.total_periods == 2
        assert result.avg_output_per_period == 200.0


# ---------------------------------------------------------------------------
# plan_vs_actual
# ---------------------------------------------------------------------------


class TestPlanVsActual:
    def test_basic_plan_vs_actual(self):
        rows = [
            {"line": "L1", "plan": 100, "actual": 110},
            {"line": "L2", "plan": 100, "actual": 90},
            {"line": "L3", "plan": 100, "actual": 100},
        ]
        result = plan_vs_actual(rows, "line", "plan", "actual")
        assert result is not None
        assert len(result.entities) == 3
        assert "L1" in result.over_achievers
        assert "L2" in result.under_achievers
        assert result.overall_achievement_pct == 100.0

    def test_empty_rows(self):
        result = plan_vs_actual([], "line", "plan", "actual")
        assert result is None

    def test_all_none_values(self):
        rows = [
            {"line": "L1", "plan": None, "actual": 100},
            {"line": "L2", "plan": 100, "actual": None},
        ]
        result = plan_vs_actual(rows, "line", "plan", "actual")
        assert result is None

    def test_on_target_within_5_percent(self):
        rows = [
            {"line": "L1", "plan": 100, "actual": 103},
        ]
        result = plan_vs_actual(rows, "line", "plan", "actual")
        assert result is not None
        assert result.entities[0].status == "on_target"
        assert len(result.over_achievers) == 0
        assert len(result.under_achievers) == 0

    def test_zero_plan(self):
        rows = [
            {"line": "L1", "plan": 0, "actual": 50},
        ]
        result = plan_vs_actual(rows, "line", "plan", "actual")
        assert result is not None
        assert result.entities[0].achievement_pct == 0.0

    def test_aggregates_multiple_rows(self):
        rows = [
            {"line": "L1", "plan": 50, "actual": 55},
            {"line": "L1", "plan": 50, "actual": 60},
        ]
        result = plan_vs_actual(rows, "line", "plan", "actual")
        assert result is not None
        assert result.entities[0].planned == 100.0
        assert result.entities[0].actual == 115.0
        assert result.entities[0].status == "over"

    def test_overall_achievement_pct(self):
        rows = [
            {"line": "L1", "plan": 200, "actual": 180},
            {"line": "L2", "plan": 300, "actual": 330},
        ]
        result = plan_vs_actual(rows, "line", "plan", "actual")
        assert result is not None
        # total plan=500, total actual=510 -> 102%
        assert result.overall_achievement_pct == 102.0

    def test_variance_calculation(self):
        rows = [
            {"line": "L1", "plan": 100, "actual": 80},
        ]
        result = plan_vs_actual(rows, "line", "plan", "actual")
        assert result is not None
        assert result.entities[0].variance == -20.0


# ---------------------------------------------------------------------------
# format_schedule_report
# ---------------------------------------------------------------------------


class TestFormatScheduleReport:
    def test_empty_report(self):
        report = format_schedule_report()
        assert "Production Schedule Report" in report
        assert "No analysis results provided." in report

    def test_shift_section(self):
        rows = [
            {"shift": "A", "output": 100},
            {"shift": "B", "output": 80},
        ]
        shift_result = analyze_shift_performance(rows, "shift", "output")
        report = format_schedule_report(shift=shift_result)
        assert "SHIFT PERFORMANCE" in report
        assert "Best shift" in report

    def test_all_sections(self):
        shift_rows = [
            {"shift": "A", "output": 100},
            {"shift": "B", "output": 80},
        ]
        batch_rows = [
            {"batch": "B1", "input": 100, "output": 90},
        ]
        takt_rows = [
            {"period": "D1", "output": 240},
        ]
        plan_rows = [
            {"line": "L1", "plan": 100, "actual": 110},
        ]
        shift_r = analyze_shift_performance(shift_rows, "shift", "output")
        batch_r = analyze_batch_efficiency(batch_rows, "batch", "input", "output")
        takt_r = compute_takt_time(takt_rows, "output", "period")
        plan_r = plan_vs_actual(plan_rows, "line", "plan", "actual")

        report = format_schedule_report(shift=shift_r, batch=batch_r, takt=takt_r, plan=plan_r)
        assert "SHIFT PERFORMANCE" in report
        assert "BATCH EFFICIENCY" in report
        assert "TAKT TIME" in report
        assert "PLAN VS ACTUAL" in report

    def test_partial_report_takt_only(self):
        takt_rows = [
            {"period": "D1", "output": 240},
        ]
        takt_r = compute_takt_time(takt_rows, "output", "period")
        report = format_schedule_report(takt=takt_r)
        assert "TAKT TIME" in report
        assert "SHIFT PERFORMANCE" not in report
        assert "No analysis results provided." not in report
