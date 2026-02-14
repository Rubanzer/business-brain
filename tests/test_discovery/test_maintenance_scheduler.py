"""Tests for maintenance scheduling and predictive maintenance analytics."""

from __future__ import annotations

from datetime import datetime, timedelta

from business_brain.discovery.maintenance_scheduler import (
    EquipmentSummary,
    MaintenanceResult,
    PartSummary,
    ReliabilityMetric,
    ScheduleEntry,
    SparePartsResult,
    analyze_maintenance_history,
    analyze_spare_parts,
    compute_mtbf_mttr,
    format_maintenance_report,
    generate_maintenance_schedule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(day: int, month: int = 1) -> str:
    """Create a timestamp string for 2024-{month}-{day}."""
    return f"2024-{month:02d}-{day:02d} 00:00:00"


# ---------------------------------------------------------------------------
# 1. analyze_maintenance_history — basic
# ---------------------------------------------------------------------------


class TestAnalyzeMaintenanceHistoryBasic:
    def test_single_equipment(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "preventive"},
            {"equip": "Pump-1", "date": "2024-01-15", "type": "corrective"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        assert result.total_events == 2
        assert len(result.equipment_summaries) == 1
        es = result.equipment_summaries[0]
        assert es.equipment == "Pump-1"
        assert es.total_events == 2
        assert es.type_breakdown["preventive"] == 1
        assert es.type_breakdown["corrective"] == 1

    def test_multiple_equipment(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "preventive"},
            {"equip": "Pump-2", "date": "2024-01-02", "type": "corrective"},
            {"equip": "Pump-1", "date": "2024-01-10", "type": "breakdown"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        assert result.total_events == 3
        assert len(result.equipment_summaries) == 2
        assert result.most_maintained_equipment == "Pump-1"

    def test_empty_rows(self):
        result = analyze_maintenance_history([], "equip", "date", "type")
        assert result is None

    def test_all_none_values(self):
        rows = [
            {"equip": None, "date": "2024-01-01", "type": "preventive"},
            {"equip": "Pump-1", "date": None, "type": "corrective"},
            {"equip": "Pump-1", "date": "2024-01-01", "type": None},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is None

    def test_corrective_ratio(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "preventive"},
            {"equip": "Pump-1", "date": "2024-01-02", "type": "corrective"},
            {"equip": "Pump-1", "date": "2024-01-03", "type": "preventive"},
            {"equip": "Pump-1", "date": "2024-01-04", "type": "preventive"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.corrective_ratio == 0.25  # 1 corrective / 4 total

    def test_overall_corrective_ratio(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "corrective"},
            {"equip": "Pump-1", "date": "2024-01-02", "type": "corrective"},
            {"equip": "Pump-2", "date": "2024-01-03", "type": "preventive"},
            {"equip": "Pump-2", "date": "2024-01-04", "type": "preventive"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        assert result.overall_corrective_ratio == 0.5  # 2 corrective / 4 total

    def test_most_common_type(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "A", "date": "2024-01-02", "type": "preventive"},
            {"equip": "A", "date": "2024-01-03", "type": "corrective"},
            {"equip": "B", "date": "2024-01-04", "type": "preventive"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        assert result.most_common_type == "preventive"

    def test_type_normalization(self):
        """Types should be lowercased and stripped."""
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "  Preventive  "},
            {"equip": "A", "date": "2024-01-02", "type": "PREVENTIVE"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.type_breakdown["preventive"] == 2


# ---------------------------------------------------------------------------
# 2. analyze_maintenance_history — with duration
# ---------------------------------------------------------------------------


class TestAnalyzeMaintenanceHistoryDuration:
    def test_duration_stats(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "preventive", "dur": 2.0},
            {"equip": "Pump-1", "date": "2024-01-15", "type": "corrective", "dur": 8.0},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type", duration_column="dur")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.total_downtime == 10.0
        assert es.avg_downtime == 5.0

    def test_duration_none_ignored(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive", "dur": 4.0},
            {"equip": "A", "date": "2024-01-02", "type": "preventive", "dur": None},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type", duration_column="dur")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.total_downtime == 4.0
        assert es.avg_downtime == 4.0

    def test_no_duration_column(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.total_downtime is None
        assert es.avg_downtime is None


# ---------------------------------------------------------------------------
# 3. analyze_maintenance_history — with cost
# ---------------------------------------------------------------------------


class TestAnalyzeMaintenanceHistoryCost:
    def test_cost_stats(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive", "cost": 100},
            {"equip": "A", "date": "2024-01-02", "type": "corrective", "cost": 500},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type", cost_column="cost")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.total_cost == 600.0
        assert es.avg_cost == 300.0

    def test_cost_string_conversion(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive", "cost": "250.50"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type", cost_column="cost")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.total_cost == 250.50

    def test_no_cost_column(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        es = result.equipment_summaries[0]
        assert es.total_cost is None
        assert es.avg_cost is None


# ---------------------------------------------------------------------------
# 4. analyze_maintenance_history — summary
# ---------------------------------------------------------------------------


class TestAnalyzeMaintenanceHistorySummary:
    def test_summary_content(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "B", "date": "2024-01-02", "type": "corrective"},
        ]
        result = analyze_maintenance_history(rows, "equip", "date", "type")
        assert result is not None
        assert "2 maintenance events" in result.summary
        assert "2 equipment" in result.summary
        assert "Most maintained" in result.summary
        assert "corrective ratio" in result.summary


# ---------------------------------------------------------------------------
# 5. compute_mtbf_mttr — basic
# ---------------------------------------------------------------------------


class TestComputeMtbfMttrBasic:
    def test_basic_two_failures(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "breakdown", "dur": 4.0},
            {"equip": "Pump-1", "date": "2024-01-11", "type": "breakdown", "dur": 6.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 1
        rm = result[0]
        assert rm.equipment == "Pump-1"
        assert rm.mtbf_days == 10.0  # 10 days between failures
        assert rm.mttr_hours == 5.0  # avg(4, 6)
        assert rm.failure_count == 2

    def test_corrective_also_counts_as_failure(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "corrective", "dur": 3.0},
            {"equip": "A", "date": "2024-01-06", "type": "corrective", "dur": 5.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 1
        assert result[0].mtbf_days == 5.0

    def test_preventive_excluded(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive", "dur": 2.0},
            {"equip": "A", "date": "2024-01-05", "type": "preventive", "dur": 2.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert result == []

    def test_single_failure_no_mtbf(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "breakdown", "dur": 4.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 1
        rm = result[0]
        assert rm.mtbf_days is None  # need >= 2 to compute MTBF
        assert rm.mttr_hours == 4.0

    def test_empty_rows(self):
        result = compute_mtbf_mttr([], "equip", "date", "type", "dur")
        assert result == []

    def test_availability_computation(self):
        # MTBF = 30 days, MTTR = 24 hours = 1 day
        # Availability = 30 / (30 + 1) * 100 = 96.77%
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "breakdown", "dur": 24.0},
            {"equip": "A", "date": "2024-01-31", "type": "breakdown", "dur": 24.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 1
        rm = result[0]
        assert rm.mtbf_days == 30.0
        assert rm.mttr_hours == 24.0
        expected_avail = 30.0 / (30.0 + 1.0) * 100
        assert abs(rm.availability - expected_avail) < 0.01

    def test_multiple_equipment(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "breakdown", "dur": 2.0},
            {"equip": "A", "date": "2024-01-11", "type": "breakdown", "dur": 4.0},
            {"equip": "B", "date": "2024-01-01", "type": "corrective", "dur": 1.0},
            {"equip": "B", "date": "2024-01-06", "type": "corrective", "dur": 3.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 2
        a = next(r for r in result if r.equipment == "A")
        b = next(r for r in result if r.equipment == "B")
        assert a.mtbf_days == 10.0
        assert b.mtbf_days == 5.0

    def test_mixed_types_only_failures(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive", "dur": 1.0},
            {"equip": "A", "date": "2024-01-05", "type": "breakdown", "dur": 4.0},
            {"equip": "A", "date": "2024-01-10", "type": "calibration", "dur": 1.0},
            {"equip": "A", "date": "2024-01-15", "type": "corrective", "dur": 6.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 1
        rm = result[0]
        # Only breakdown (Jan 5) and corrective (Jan 15) count
        assert rm.failure_count == 2
        assert rm.mtbf_days == 10.0
        assert rm.mttr_hours == 5.0  # avg(4, 6)


# ---------------------------------------------------------------------------
# 6. compute_mtbf_mttr — edge cases
# ---------------------------------------------------------------------------


class TestComputeMtbfMttrEdgeCases:
    def test_none_duration_defaults_to_zero(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "breakdown", "dur": None},
            {"equip": "A", "date": "2024-01-11", "type": "breakdown", "dur": None},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 1
        assert result[0].mttr_hours == 0.0

    def test_invalid_date_skipped(self):
        rows = [
            {"equip": "A", "date": "not-a-date", "type": "breakdown", "dur": 4.0},
            {"equip": "A", "date": "2024-01-01", "type": "breakdown", "dur": 4.0},
            {"equip": "A", "date": "2024-01-11", "type": "breakdown", "dur": 6.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert len(result) == 1
        assert result[0].failure_count == 2

    def test_first_and_last_failure_dates(self):
        rows = [
            {"equip": "A", "date": "2024-01-05", "type": "breakdown", "dur": 2.0},
            {"equip": "A", "date": "2024-01-01", "type": "breakdown", "dur": 2.0},
            {"equip": "A", "date": "2024-01-10", "type": "breakdown", "dur": 2.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        rm = result[0]
        assert rm.first_failure == datetime(2024, 1, 1)
        assert rm.last_failure == datetime(2024, 1, 10)

    def test_sorted_by_equipment_name(self):
        rows = [
            {"equip": "Z", "date": "2024-01-01", "type": "breakdown", "dur": 1.0},
            {"equip": "Z", "date": "2024-01-11", "type": "breakdown", "dur": 1.0},
            {"equip": "A", "date": "2024-01-01", "type": "breakdown", "dur": 1.0},
            {"equip": "A", "date": "2024-01-11", "type": "breakdown", "dur": 1.0},
        ]
        result = compute_mtbf_mttr(rows, "equip", "date", "type", "dur")
        assert result[0].equipment == "A"
        assert result[1].equipment == "Z"


# ---------------------------------------------------------------------------
# 7. analyze_spare_parts — basic
# ---------------------------------------------------------------------------


class TestAnalyzeSparePartsBasic:
    def test_basic_parts(self):
        rows = [
            {"part": "Bearing-6205", "qty": 10},
            {"part": "O-Ring", "qty": 50},
            {"part": "Bearing-6205", "qty": 5},
        ]
        result = analyze_spare_parts(rows, "part", "qty")
        assert result is not None
        assert result.total_unique_parts == 2
        assert result.total_quantity == 65.0
        assert result.total_spend is None

    def test_empty_rows(self):
        result = analyze_spare_parts([], "part", "qty")
        assert result is None

    def test_all_none(self):
        rows = [
            {"part": None, "qty": 10},
            {"part": "Bearing", "qty": None},
        ]
        result = analyze_spare_parts(rows, "part", "qty")
        assert result is None

    def test_top_5_parts(self):
        rows = [
            {"part": f"Part-{i}", "qty": (10 - i) * 10}
            for i in range(8)
        ]
        result = analyze_spare_parts(rows, "part", "qty")
        assert result is not None
        assert len(result.top_parts) == 5
        # Sorted by quantity descending
        assert result.top_parts[0].total_quantity >= result.top_parts[1].total_quantity

    def test_fewer_than_5_parts(self):
        rows = [
            {"part": "A", "qty": 10},
            {"part": "B", "qty": 5},
        ]
        result = analyze_spare_parts(rows, "part", "qty")
        assert result is not None
        assert len(result.top_parts) == 2


# ---------------------------------------------------------------------------
# 8. analyze_spare_parts — with cost
# ---------------------------------------------------------------------------


class TestAnalyzeSparePartsCost:
    def test_cost_stats(self):
        rows = [
            {"part": "Bearing", "qty": 10, "cost": 500},
            {"part": "O-Ring", "qty": 100, "cost": 200},
        ]
        result = analyze_spare_parts(rows, "part", "qty", cost_column="cost")
        assert result is not None
        assert result.total_spend == 700.0
        bearing = next(p for p in result.top_parts if p.part == "Bearing")
        assert bearing.total_cost == 500.0
        assert bearing.avg_cost_per_unit == 50.0

    def test_abc_classification(self):
        rows = [
            {"part": "Expensive", "qty": 1, "cost": 8000},
            {"part": "Medium", "qty": 5, "cost": 1500},
            {"part": "Cheap1", "qty": 50, "cost": 300},
            {"part": "Cheap2", "qty": 100, "cost": 150},
            {"part": "Tiny", "qty": 200, "cost": 50},
        ]
        result = analyze_spare_parts(rows, "part", "qty", cost_column="cost")
        assert result is not None
        abc = {p.part: p.abc_category for p in result.abc_parts}
        assert abc["Expensive"] == "A"  # 80% of cost
        # Medium should be B (80-95%)
        assert abc["Medium"] == "B"

    def test_abc_single_part(self):
        rows = [{"part": "Only", "qty": 10, "cost": 100}]
        result = analyze_spare_parts(rows, "part", "qty", cost_column="cost")
        assert result is not None
        assert len(result.abc_parts) == 1
        assert result.abc_parts[0].abc_category == "A"


# ---------------------------------------------------------------------------
# 9. analyze_spare_parts — with equipment
# ---------------------------------------------------------------------------


class TestAnalyzeSparePartsEquipment:
    def test_parts_by_equipment(self):
        rows = [
            {"part": "Bearing", "qty": 10, "equip": "Pump-1"},
            {"part": "O-Ring", "qty": 5, "equip": "Pump-1"},
            {"part": "Bearing", "qty": 8, "equip": "Pump-2"},
        ]
        result = analyze_spare_parts(rows, "part", "qty", equipment_column="equip")
        assert result is not None
        assert result.parts_by_equipment is not None
        assert "Pump-1" in result.parts_by_equipment
        assert sorted(result.parts_by_equipment["Pump-1"]) == ["Bearing", "O-Ring"]
        assert result.parts_by_equipment["Pump-2"] == ["Bearing"]

    def test_no_equipment_column(self):
        rows = [{"part": "Bearing", "qty": 10}]
        result = analyze_spare_parts(rows, "part", "qty")
        assert result is not None
        assert result.parts_by_equipment is None


# ---------------------------------------------------------------------------
# 10. analyze_spare_parts — summary
# ---------------------------------------------------------------------------


class TestAnalyzeSparePartsSummary:
    def test_summary_content(self):
        rows = [
            {"part": "Bearing", "qty": 10, "cost": 500},
            {"part": "O-Ring", "qty": 100, "cost": 200},
        ]
        result = analyze_spare_parts(rows, "part", "qty", cost_column="cost")
        assert result is not None
        assert "2 unique parts" in result.summary
        assert "Total spend" in result.summary
        assert "Top part" in result.summary


# ---------------------------------------------------------------------------
# 11. generate_maintenance_schedule — basic
# ---------------------------------------------------------------------------


class TestGenerateMaintenanceScheduleBasic:
    def test_basic_schedule(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "preventive"},
            {"equip": "Pump-1", "date": "2024-01-31", "type": "preventive"},
            {"equip": "Pump-1", "date": "2024-03-01", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        assert len(result) == 1
        se = result[0]
        assert se.equipment == "Pump-1"
        assert se.next_maintenance_date is not None
        assert se.avg_interval_days is not None

    def test_empty_rows(self):
        result = generate_maintenance_schedule([], "equip", "date", "type")
        assert result == []

    def test_fixed_interval(self):
        rows = [
            {"equip": "Pump-1", "date": "2024-01-01", "type": "preventive"},
            {"equip": "Pump-1", "date": "2024-02-01", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type", interval_days=30)
        assert len(result) == 1
        se = result[0]
        assert se.avg_interval_days == 30.0
        # Last date is Feb 1, so next = Mar 2
        expected_next = datetime(2024, 2, 1) + timedelta(days=30)
        assert se.next_maintenance_date == expected_next

    def test_multiple_equipment(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "A", "date": "2024-01-31", "type": "preventive"},
            {"equip": "B", "date": "2024-01-01", "type": "preventive"},
            {"equip": "B", "date": "2024-02-15", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        assert len(result) == 2
        equipment_names = {se.equipment for se in result}
        assert equipment_names == {"A", "B"}


# ---------------------------------------------------------------------------
# 12. generate_maintenance_schedule — overdue detection
# ---------------------------------------------------------------------------


class TestGenerateMaintenanceScheduleOverdue:
    def test_overdue_detection(self):
        # Equipment A: last on Jan 1, interval 10 days => next Jan 11
        # Equipment B: last on Jan 20, interval 10 days => next Jan 30
        # Max date in data = Jan 20
        # A's next (Jan 11) < Jan 20 => overdue
        # B's next (Jan 30) > Jan 20 => not overdue
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "B", "date": "2024-01-10", "type": "preventive"},
            {"equip": "B", "date": "2024-01-20", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type", interval_days=10)
        a = next(s for s in result if s.equipment == "A")
        b = next(s for s in result if s.equipment == "B")
        assert a.is_overdue is True  # next = Jan 11 < Jan 20
        assert b.is_overdue is False  # next = Jan 30 > Jan 20

    def test_not_overdue_when_recent(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "A", "date": "2024-01-15", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type", interval_days=30)
        se = result[0]
        # Last: Jan 15, next: Feb 14, max date: Jan 15 => not overdue
        assert se.is_overdue is False


# ---------------------------------------------------------------------------
# 13. generate_maintenance_schedule — interval computation
# ---------------------------------------------------------------------------


class TestGenerateMaintenanceScheduleInterval:
    def test_interval_from_preventive_events(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "A", "date": "2024-01-11", "type": "preventive"},
            {"equip": "A", "date": "2024-01-21", "type": "preventive"},
            {"equip": "A", "date": "2024-01-05", "type": "corrective"},  # not used for interval
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        se = result[0]
        assert se.avg_interval_days == 10.0

    def test_fallback_to_all_events(self):
        """When no preventive events, falls back to all events for interval."""
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "corrective"},
            {"equip": "A", "date": "2024-01-11", "type": "breakdown"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        se = result[0]
        assert se.avg_interval_days == 10.0

    def test_single_event_no_interval(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        se = result[0]
        assert se.avg_interval_days is None
        assert se.next_maintenance_date is None
        assert se.is_overdue is False

    def test_sorted_by_next_date(self):
        rows = [
            {"equip": "Late", "date": "2024-01-01", "type": "preventive"},
            {"equip": "Late", "date": "2024-02-01", "type": "preventive"},
            {"equip": "Soon", "date": "2024-01-01", "type": "preventive"},
            {"equip": "Soon", "date": "2024-01-10", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        assert len(result) == 2
        # "Soon" has 9-day interval, next ~Jan 19; "Late" has 31-day interval, next ~Mar 3
        assert result[0].equipment == "Soon"
        assert result[1].equipment == "Late"


# ---------------------------------------------------------------------------
# 14. generate_maintenance_schedule — edge cases
# ---------------------------------------------------------------------------


class TestGenerateMaintenanceScheduleEdgeCases:
    def test_invalid_dates_skipped(self):
        rows = [
            {"equip": "A", "date": "not-a-date", "type": "preventive"},
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "A", "date": "2024-01-11", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        assert len(result) == 1
        assert result[0].events_in_history == 2

    def test_all_invalid_dates(self):
        rows = [
            {"equip": "A", "date": "garbage", "type": "preventive"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        assert result == []

    def test_events_in_history_count(self):
        rows = [
            {"equip": "A", "date": "2024-01-01", "type": "preventive"},
            {"equip": "A", "date": "2024-01-05", "type": "corrective"},
            {"equip": "A", "date": "2024-01-10", "type": "breakdown"},
        ]
        result = generate_maintenance_schedule(rows, "equip", "date", "type")
        assert result[0].events_in_history == 3


# ---------------------------------------------------------------------------
# 15. format_maintenance_report — basic
# ---------------------------------------------------------------------------


class TestFormatMaintenanceReportBasic:
    def test_report_with_history(self):
        history = MaintenanceResult(
            equipment_summaries=[
                EquipmentSummary(
                    equipment="Pump-1", total_events=5,
                    type_breakdown={"preventive": 3, "corrective": 2},
                    corrective_ratio=0.4, avg_downtime=3.0,
                    total_downtime=15.0, total_cost=2000.0, avg_cost=400.0,
                ),
            ],
            total_events=5,
            most_maintained_equipment="Pump-1",
            most_common_type="preventive",
            overall_corrective_ratio=0.4,
            summary="5 events across 1 equipment.",
        )
        report = format_maintenance_report(history=history)
        assert "MAINTENANCE ANALYSIS REPORT" in report
        assert "MAINTENANCE HISTORY" in report
        assert "Pump-1" in report
        assert "Total Events:" in report
        assert "Corrective Ratio:" in report

    def test_report_with_reliability(self):
        reliability = [
            ReliabilityMetric(
                equipment="Pump-1", mtbf_days=30.0, mttr_hours=4.0,
                availability=99.45, failure_count=3,
                first_failure=datetime(2024, 1, 1),
                last_failure=datetime(2024, 3, 1),
            ),
        ]
        report = format_maintenance_report(reliability=reliability)
        assert "RELIABILITY METRICS" in report
        assert "MTBF" in report
        assert "Pump-1" in report

    def test_report_with_spare_parts(self):
        spare_parts = SparePartsResult(
            total_unique_parts=3, total_quantity=100.0,
            total_spend=5000.0, top_parts=[
                PartSummary(part="Bearing", total_quantity=50.0,
                            total_cost=3000.0, avg_cost_per_unit=60.0, abc_category="A"),
            ],
            parts_by_equipment=None,
            abc_parts=[
                PartSummary(part="Bearing", total_quantity=50.0,
                            total_cost=3000.0, avg_cost_per_unit=60.0, abc_category="A"),
            ],
            summary="3 parts, 100 qty.",
        )
        report = format_maintenance_report(spare_parts=spare_parts)
        assert "SPARE PARTS ANALYSIS" in report
        assert "Bearing" in report
        assert "Total Spend" in report

    def test_report_with_schedule(self):
        schedule = [
            ScheduleEntry(
                equipment="Pump-1",
                last_maintenance_date=datetime(2024, 1, 15),
                avg_interval_days=30.0,
                next_maintenance_date=datetime(2024, 2, 14),
                is_overdue=False,
                events_in_history=5,
            ),
            ScheduleEntry(
                equipment="Pump-2",
                last_maintenance_date=datetime(2024, 1, 1),
                avg_interval_days=15.0,
                next_maintenance_date=datetime(2024, 1, 16),
                is_overdue=True,
                events_in_history=3,
            ),
        ]
        report = format_maintenance_report(schedule=schedule)
        assert "MAINTENANCE SCHEDULE" in report
        assert "OVERDUE" in report
        assert "Pump-2" in report

    def test_empty_report(self):
        report = format_maintenance_report()
        assert "MAINTENANCE ANALYSIS REPORT" in report
        assert "SUMMARY" in report

    def test_full_report(self):
        history = MaintenanceResult(
            equipment_summaries=[], total_events=0,
            most_maintained_equipment=None, most_common_type=None,
            overall_corrective_ratio=0.0, summary="No data.",
        )
        reliability = [
            ReliabilityMetric(
                equipment="A", mtbf_days=10.0, mttr_hours=2.0,
                availability=99.17, failure_count=2,
                first_failure=datetime(2024, 1, 1),
                last_failure=datetime(2024, 1, 11),
            ),
        ]
        report = format_maintenance_report(history=history, reliability=reliability)
        assert "MAINTENANCE HISTORY" in report
        assert "RELIABILITY METRICS" in report
        assert "SUMMARY" in report

    def test_report_schedule_no_overdue(self):
        schedule = [
            ScheduleEntry(
                equipment="A",
                last_maintenance_date=datetime(2024, 1, 15),
                avg_interval_days=30.0,
                next_maintenance_date=datetime(2024, 2, 14),
                is_overdue=False,
                events_in_history=2,
            ),
        ]
        report = format_maintenance_report(schedule=schedule)
        assert "MAINTENANCE SCHEDULE" in report
        assert "OVERDUE" not in report

    def test_report_reliability_none_values(self):
        reliability = [
            ReliabilityMetric(
                equipment="A", mtbf_days=None, mttr_hours=None,
                availability=None, failure_count=1,
                first_failure=datetime(2024, 1, 1),
                last_failure=datetime(2024, 1, 1),
            ),
        ]
        report = format_maintenance_report(reliability=reliability)
        assert "N/A" in report

    def test_report_schedule_no_next_date(self):
        schedule = [
            ScheduleEntry(
                equipment="A",
                last_maintenance_date=datetime(2024, 1, 1),
                avg_interval_days=None,
                next_maintenance_date=None,
                is_overdue=False,
                events_in_history=1,
            ),
        ]
        report = format_maintenance_report(schedule=schedule)
        assert "next=N/A" in report
        assert "interval=N/A" in report
