"""Tests for equipment/machine downtime pattern analysis."""

from __future__ import annotations

from datetime import datetime, timedelta

from business_brain.discovery.downtime_analyzer import (
    DowntimeParetoItem,
    DowntimeResult,
    MachineDowntime,
    RecurringFailure,
    ReasonSummary,
    ShiftDowntime,
    ShiftResult,
    analyze_downtime,
    detect_recurring_failures,
    downtime_pareto,
    format_downtime_report,
    shift_analysis,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(
    data: list[tuple],
    columns: list[str],
) -> list[dict]:
    """Build list of dicts from tuples and column names."""
    return [dict(zip(columns, row)) for row in data]


def _ts(day: int, hour: int = 0) -> str:
    """Create a timestamp string for 2024-01-{day} {hour}:00:00."""
    return f"2024-01-{day:02d} {hour:02d}:00:00"


# ---------------------------------------------------------------------------
# 1. analyze_downtime — basic
# ---------------------------------------------------------------------------

class TestAnalyzeDowntimeBasic:
    def test_basic_single_machine(self):
        rows = [
            {"machine": "CNC-1", "duration": 30},
            {"machine": "CNC-1", "duration": 45},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert isinstance(result, DowntimeResult)
        assert result.total_events == 2
        assert result.total_downtime == 75.0
        assert len(result.machines) == 1
        assert result.machines[0].machine == "CNC-1"
        assert result.machines[0].mttr == 37.5

    def test_basic_multiple_machines(self):
        rows = [
            {"machine": "CNC-1", "duration": 30},
            {"machine": "CNC-2", "duration": 60},
            {"machine": "CNC-1", "duration": 20},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert result.total_events == 3
        assert result.total_downtime == 110.0
        assert len(result.machines) == 2
        assert result.worst_machine == "CNC-2"
        assert result.best_machine == "CNC-1"

    def test_empty_rows(self):
        result = analyze_downtime([], "machine", "duration")
        assert result.total_events == 0
        assert result.total_downtime == 0.0
        assert result.machines == []
        assert result.worst_machine is None
        assert result.best_machine is None
        assert "No downtime data" in result.summary

    def test_missing_columns_skipped(self):
        rows = [
            {"machine": "CNC-1", "duration": 30},
            {"machine": None, "duration": 20},
            {"machine": "CNC-1", "duration": None},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert result.total_events == 1
        assert result.total_downtime == 30.0

    def test_non_numeric_duration_skipped(self):
        rows = [
            {"machine": "CNC-1", "duration": 30},
            {"machine": "CNC-1", "duration": "bad"},
            {"machine": "CNC-2", "duration": 20},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert result.total_events == 2
        assert result.total_downtime == 50.0


# ---------------------------------------------------------------------------
# 2. analyze_downtime — with reasons
# ---------------------------------------------------------------------------

class TestAnalyzeDowntimeReasons:
    def test_top_reasons(self):
        rows = [
            {"machine": "M1", "duration": 30, "reason": "Bearing"},
            {"machine": "M1", "duration": 45, "reason": "Motor"},
            {"machine": "M2", "duration": 60, "reason": "Bearing"},
            {"machine": "M2", "duration": 15, "reason": "Bearing"},
        ]
        result = analyze_downtime(rows, "machine", "duration", reason_column="reason")
        assert len(result.top_reasons) == 2
        # Bearing: 30+60+15=105, Motor: 45
        assert result.top_reasons[0].reason == "Bearing"
        assert result.top_reasons[0].total_duration == 105.0
        assert result.top_reasons[0].event_count == 3

    def test_machine_top_reason(self):
        rows = [
            {"machine": "M1", "duration": 10, "reason": "A"},
            {"machine": "M1", "duration": 10, "reason": "A"},
            {"machine": "M1", "duration": 10, "reason": "B"},
        ]
        result = analyze_downtime(rows, "machine", "duration", reason_column="reason")
        m1 = result.machines[0]
        assert m1.top_reason == "A"

    def test_no_reason_column(self):
        rows = [
            {"machine": "M1", "duration": 30},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert result.top_reasons == []
        assert result.machines[0].top_reason is None

    def test_reason_with_none_values(self):
        rows = [
            {"machine": "M1", "duration": 30, "reason": "Bearing"},
            {"machine": "M1", "duration": 20, "reason": None},
        ]
        result = analyze_downtime(rows, "machine", "duration", reason_column="reason")
        assert len(result.top_reasons) == 1
        assert result.top_reasons[0].reason == "Bearing"


# ---------------------------------------------------------------------------
# 3. analyze_downtime — with time_column
# ---------------------------------------------------------------------------

class TestAnalyzeDowntimeTime:
    def test_availability_computed_with_time(self):
        rows = [
            {"machine": "M1", "duration": 100, "ts": "2024-01-01 00:00:00"},
            {"machine": "M1", "duration": 100, "ts": "2024-01-02 00:00:00"},
        ]
        result = analyze_downtime(rows, "machine", "duration", time_column="ts")
        m1 = result.machines[0]
        assert m1.availability_pct is not None
        # span = 86400s, downtime = 200, availability = (86400-200)/86400*100
        expected = (86400 - 200) / 86400 * 100
        assert abs(m1.availability_pct - round(expected, 2)) < 0.1

    def test_availability_none_without_time(self):
        rows = [
            {"machine": "M1", "duration": 100},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert result.machines[0].availability_pct is None

    def test_single_timestamp_no_availability(self):
        rows = [
            {"machine": "M1", "duration": 100, "ts": "2024-01-01 00:00:00"},
        ]
        result = analyze_downtime(rows, "machine", "duration", time_column="ts")
        assert result.machines[0].availability_pct is None


# ---------------------------------------------------------------------------
# 4. analyze_downtime — summary
# ---------------------------------------------------------------------------

class TestAnalyzeDowntimeSummary:
    def test_summary_contains_key_info(self):
        rows = [
            {"machine": "M1", "duration": 30, "reason": "Bearing"},
            {"machine": "M2", "duration": 60, "reason": "Motor"},
        ]
        result = analyze_downtime(rows, "machine", "duration", reason_column="reason")
        assert "2 downtime events" in result.summary
        assert "2 machine" in result.summary
        assert "Worst" in result.summary
        assert "Top reason" in result.summary


# ---------------------------------------------------------------------------
# 5. detect_recurring_failures
# ---------------------------------------------------------------------------

class TestDetectRecurringFailures:
    def test_basic_recurring(self):
        base = datetime(2024, 1, 1)
        rows = [
            {"machine": "M1", "reason": "Bearing", "ts": (base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")}
            for i in range(5)
        ]
        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert len(result) == 1
        assert result[0].machine == "M1"
        assert result[0].reason == "Bearing"
        assert result[0].occurrence_count == 5

    def test_below_min_occurrences_excluded(self):
        rows = [
            {"machine": "M1", "reason": "Bearing", "ts": "2024-01-01 00:00:00"},
            {"machine": "M1", "reason": "Bearing", "ts": "2024-01-02 00:00:00"},
        ]
        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert result == []

    def test_empty_rows(self):
        result = detect_recurring_failures([], "machine", "reason", "ts")
        assert result == []

    def test_avg_interval_computed(self):
        rows = [
            {"machine": "M1", "reason": "Bearing", "ts": "2024-01-01 00:00:00"},
            {"machine": "M1", "reason": "Bearing", "ts": "2024-01-03 00:00:00"},
            {"machine": "M1", "reason": "Bearing", "ts": "2024-01-05 00:00:00"},
        ]
        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert len(result) == 1
        # Intervals: 2 days, 2 days -> avg = 2 days = 172800 seconds
        assert result[0].avg_interval_between == 172800.0

    def test_trend_stable(self):
        # Evenly spaced events
        base = datetime(2024, 1, 1)
        rows = [
            {"machine": "M1", "reason": "X", "ts": (base + timedelta(days=i * 10)).strftime("%Y-%m-%d %H:%M:%S")}
            for i in range(6)
        ]
        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert result[0].trend == "stable"

    def test_trend_increasing(self):
        # Intervals shrinking: failures becoming more frequent
        rows = [
            {"machine": "M1", "reason": "X", "ts": "2024-01-01 00:00:00"},
            {"machine": "M1", "reason": "X", "ts": "2024-01-11 00:00:00"},  # +10 days
            {"machine": "M1", "reason": "X", "ts": "2024-01-21 00:00:00"},  # +10 days
            {"machine": "M1", "reason": "X", "ts": "2024-01-24 00:00:00"},  # +3 days
            {"machine": "M1", "reason": "X", "ts": "2024-01-27 00:00:00"},  # +3 days
        ]
        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert result[0].trend == "increasing"

    def test_multiple_combos(self):
        base = datetime(2024, 1, 1)
        rows = []
        for i in range(4):
            rows.append({"machine": "M1", "reason": "A", "ts": (base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")})
            rows.append({"machine": "M2", "reason": "B", "ts": (base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")})

        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert len(result) == 2
        machines = {r.machine for r in result}
        assert machines == {"M1", "M2"}

    def test_none_values_skipped(self):
        rows = [
            {"machine": "M1", "reason": "A", "ts": "2024-01-01 00:00:00"},
            {"machine": "M1", "reason": None, "ts": "2024-01-02 00:00:00"},
            {"machine": None, "reason": "A", "ts": "2024-01-03 00:00:00"},
            {"machine": "M1", "reason": "A", "ts": None},
            {"machine": "M1", "reason": "A", "ts": "2024-01-04 00:00:00"},
            {"machine": "M1", "reason": "A", "ts": "2024-01-05 00:00:00"},
        ]
        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert len(result) == 1
        assert result[0].occurrence_count == 3


# ---------------------------------------------------------------------------
# 6. downtime_pareto
# ---------------------------------------------------------------------------

class TestDowntimePareto:
    def test_basic_pareto(self):
        rows = [
            {"reason": "Bearing", "duration": 500},
            {"reason": "Motor", "duration": 200},
            {"reason": "Electrical", "duration": 100},
            {"reason": "Software", "duration": 50},
            {"reason": "Other", "duration": 30},
        ]
        result = downtime_pareto(rows, "reason", "duration")
        assert len(result) == 5
        assert result[0].reason == "Bearing"
        assert result[0].category == "A"
        assert result[-1].cumulative_pct == 100.0

    def test_empty_rows(self):
        result = downtime_pareto([], "reason", "duration")
        assert result == []

    def test_single_reason(self):
        rows = [{"reason": "Bearing", "duration": 100}]
        result = downtime_pareto(rows, "reason", "duration")
        assert len(result) == 1
        assert result[0].pct_of_total == 100.0
        assert result[0].cumulative_pct == 100.0
        assert result[0].category == "A"

    def test_abc_classification(self):
        # A = top 80%, B = next 15% (80-95%), C = remaining
        rows = [
            {"reason": "Big", "duration": 800},
            {"reason": "Medium", "duration": 120},
            {"reason": "Small1", "duration": 40},
            {"reason": "Small2", "duration": 30},
            {"reason": "Tiny", "duration": 10},
        ]
        result = downtime_pareto(rows, "reason", "duration")
        categories = {item.reason: item.category for item in result}
        assert categories["Big"] == "A"
        # Big=80%, Medium would push to ~92%, so it's B
        assert categories["Medium"] == "B"

    def test_aggregates_duplicate_reasons(self):
        rows = [
            {"reason": "Bearing", "duration": 50},
            {"reason": "Bearing", "duration": 50},
            {"reason": "Motor", "duration": 100},
        ]
        result = downtime_pareto(rows, "reason", "duration")
        assert len(result) == 2
        bearing = next(i for i in result if i.reason == "Bearing")
        assert bearing.total_duration == 100.0

    def test_none_values_skipped(self):
        rows = [
            {"reason": "Bearing", "duration": 100},
            {"reason": None, "duration": 50},
            {"reason": "Motor", "duration": None},
        ]
        result = downtime_pareto(rows, "reason", "duration")
        assert len(result) == 1
        assert result[0].reason == "Bearing"

    def test_cumulative_pct_increasing(self):
        rows = [
            {"reason": "A", "duration": 50},
            {"reason": "B", "duration": 30},
            {"reason": "C", "duration": 20},
        ]
        result = downtime_pareto(rows, "reason", "duration")
        for i in range(1, len(result)):
            assert result[i].cumulative_pct >= result[i - 1].cumulative_pct


# ---------------------------------------------------------------------------
# 7. shift_analysis
# ---------------------------------------------------------------------------

class TestShiftAnalysis:
    def test_basic_shift(self):
        rows = [
            {"shift": "Day", "duration": 30},
            {"shift": "Day", "duration": 20},
            {"shift": "Night", "duration": 60},
        ]
        result = shift_analysis(rows, "shift", "duration")
        assert isinstance(result, ShiftResult)
        assert len(result.shifts) == 2
        assert result.worst_shift == "Night"
        assert result.best_shift == "Day"

    def test_empty_rows(self):
        result = shift_analysis([], "shift", "duration")
        assert result.shifts == []
        assert result.worst_shift is None
        assert result.best_shift is None
        assert "No shift data" in result.summary

    def test_single_shift(self):
        rows = [
            {"shift": "Day", "duration": 30},
            {"shift": "Day", "duration": 20},
        ]
        result = shift_analysis(rows, "shift", "duration")
        assert len(result.shifts) == 1
        assert result.worst_shift == "Day"
        assert result.best_shift == "Day"
        assert result.variance_ratio == 1.0

    def test_variance_ratio(self):
        rows = [
            {"shift": "Day", "duration": 100},
            {"shift": "Night", "duration": 200},
        ]
        result = shift_analysis(rows, "shift", "duration")
        assert result.variance_ratio == 2.0

    def test_avg_duration_computed(self):
        rows = [
            {"shift": "Day", "duration": 30},
            {"shift": "Day", "duration": 50},
        ]
        result = shift_analysis(rows, "shift", "duration")
        day = result.shifts[0]
        assert day.avg_duration == 40.0

    def test_none_values_skipped(self):
        rows = [
            {"shift": "Day", "duration": 30},
            {"shift": None, "duration": 20},
            {"shift": "Night", "duration": None},
        ]
        result = shift_analysis(rows, "shift", "duration")
        assert len(result.shifts) == 1
        assert result.shifts[0].shift == "Day"

    def test_summary_content(self):
        rows = [
            {"shift": "Day", "duration": 30},
            {"shift": "Night", "duration": 60},
        ]
        result = shift_analysis(rows, "shift", "duration")
        assert "2 shift" in result.summary
        assert "Worst" in result.summary
        assert "Best" in result.summary


# ---------------------------------------------------------------------------
# 8. format_downtime_report
# ---------------------------------------------------------------------------

class TestFormatDowntimeReport:
    def test_basic_report(self):
        result = DowntimeResult(
            machines=[
                MachineDowntime(
                    machine="CNC-1", total_downtime=100, event_count=5,
                    mttr=20, availability_pct=95.0, top_reason="Bearing",
                ),
            ],
            total_downtime=100,
            total_events=5,
            top_reasons=[
                ReasonSummary(reason="Bearing", total_duration=80, event_count=4, pct_of_total=80.0),
            ],
            worst_machine="CNC-1",
            best_machine="CNC-1",
            summary="5 events, 100 total downtime.",
        )
        report = format_downtime_report(result)
        assert "DOWNTIME ANALYSIS REPORT" in report
        assert "CNC-1" in report
        assert "Bearing" in report
        assert "OVERVIEW" in report
        assert "MACHINE BREAKDOWN" in report

    def test_report_with_pareto(self):
        result = DowntimeResult(
            machines=[], total_downtime=0, total_events=0,
            top_reasons=[], worst_machine=None, best_machine=None,
            summary="Empty.",
        )
        pareto = [
            DowntimeParetoItem(
                reason="Bearing", total_duration=500,
                pct_of_total=50.0, cumulative_pct=50.0, category="A",
            ),
        ]
        report = format_downtime_report(result, pareto=pareto)
        assert "PARETO ANALYSIS" in report
        assert "[A]" in report

    def test_report_with_recurring(self):
        result = DowntimeResult(
            machines=[], total_downtime=0, total_events=0,
            top_reasons=[], worst_machine=None, best_machine=None,
            summary="Empty.",
        )
        recurring = [
            RecurringFailure(
                machine="M1", reason="Bearing", occurrence_count=5,
                total_duration=250, avg_interval_between=86400.0, trend="increasing",
            ),
        ]
        report = format_downtime_report(result, recurring=recurring)
        assert "RECURRING FAILURES" in report
        assert "M1" in report
        assert "increasing" in report

    def test_report_empty_result(self):
        result = DowntimeResult(
            machines=[], total_downtime=0, total_events=0,
            top_reasons=[], worst_machine=None, best_machine=None,
            summary="No data.",
        )
        report = format_downtime_report(result)
        assert "DOWNTIME ANALYSIS REPORT" in report
        assert "Total Events:    0" in report
        assert "No data." in report


# ---------------------------------------------------------------------------
# 9. Edge cases — all functions
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_analyze_all_none_rows(self):
        rows = [
            {"machine": None, "duration": None},
            {"machine": None, "duration": None},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert result.total_events == 0
        assert "No valid downtime records" in result.summary

    def test_pareto_all_zero_durations(self):
        rows = [
            {"reason": "A", "duration": 0},
            {"reason": "B", "duration": 0},
        ]
        result = downtime_pareto(rows, "reason", "duration")
        assert result == []

    def test_shift_all_non_numeric(self):
        rows = [
            {"shift": "Day", "duration": "bad"},
            {"shift": "Night", "duration": "worse"},
        ]
        result = shift_analysis(rows, "shift", "duration")
        assert result.shifts == []
        assert "No valid shift records" in result.summary

    def test_recurring_invalid_timestamps(self):
        rows = [
            {"machine": "M1", "reason": "A", "ts": "not-a-date"},
            {"machine": "M1", "reason": "A", "ts": "also-not-a-date"},
            {"machine": "M1", "reason": "A", "ts": "nope"},
        ]
        result = detect_recurring_failures(rows, "machine", "reason", "ts", min_occurrences=3)
        assert result == []

    def test_analyze_string_durations_converted(self):
        rows = [
            {"machine": "M1", "duration": "30.5"},
            {"machine": "M1", "duration": "19.5"},
        ]
        result = analyze_downtime(rows, "machine", "duration")
        assert result.total_downtime == 50.0
        assert result.machines[0].mttr == 25.0

    def test_pareto_pct_of_total_sums_to_100(self):
        rows = [
            {"reason": "A", "duration": 60},
            {"reason": "B", "duration": 30},
            {"reason": "C", "duration": 10},
        ]
        result = downtime_pareto(rows, "reason", "duration")
        total_pct = sum(item.pct_of_total for item in result)
        assert abs(total_pct - 100.0) < 0.1
