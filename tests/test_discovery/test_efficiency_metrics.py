"""Tests for manufacturing efficiency metrics module."""

from business_brain.discovery.efficiency_metrics import (
    EntityEnergy,
    EntityOEE,
    EntityWaste,
    EntityYield,
    EnergyResult,
    OEEResult,
    WasteResult,
    YieldResult,
    compute_energy_efficiency,
    compute_oee,
    compute_waste_analysis,
    compute_yield_analysis,
    efficiency_report,
)


# ===================================================================
# OEE Tests
# ===================================================================


class TestComputeOEE:
    def test_basic_oee(self):
        rows = [
            {"line": "L1", "avail": 0.90, "perf": 0.95, "qual": 0.99},
            {"line": "L2", "avail": 0.80, "perf": 0.85, "qual": 0.90},
        ]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result is not None
        assert len(result.entities) == 2
        # L1: 0.90 * 0.95 * 0.99 = 0.84645
        l1 = [e for e in result.entities if e.entity == "L1"][0]
        assert abs(l1.oee - 0.8465) < 0.01
        # L2: 0.80 * 0.85 * 0.90 = 0.612
        l2 = [e for e in result.entities if e.entity == "L2"][0]
        assert abs(l2.oee - 0.612) < 0.01

    def test_empty_returns_none(self):
        assert compute_oee([], "line", "a", "p", "q") is None

    def test_all_none_values_returns_none(self):
        rows = [{"line": None, "avail": None, "perf": None, "qual": None}]
        assert compute_oee(rows, "line", "avail", "perf", "qual") is None

    def test_world_class_grade(self):
        rows = [{"line": "L1", "avail": 0.95, "perf": 0.95, "qual": 0.99}]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result is not None
        assert result.entities[0].oee_grade == "World Class"
        assert result.world_class_count == 1

    def test_good_grade(self):
        rows = [{"line": "L1", "avail": 0.85, "perf": 0.85, "qual": 0.85}]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        # 0.85^3 = 0.614125
        assert result.entities[0].oee_grade == "Good"

    def test_poor_grade(self):
        rows = [{"line": "L1", "avail": 0.70, "perf": 0.70, "qual": 0.70}]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        # 0.70^3 = 0.343
        assert result.entities[0].oee_grade == "Poor"

    def test_limiting_factor_availability(self):
        rows = [{"line": "L1", "avail": 0.50, "perf": 0.90, "qual": 0.95}]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result.entities[0].limiting_factor == "availability"

    def test_limiting_factor_performance(self):
        rows = [{"line": "L1", "avail": 0.90, "perf": 0.50, "qual": 0.95}]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result.entities[0].limiting_factor == "performance"

    def test_limiting_factor_quality(self):
        rows = [{"line": "L1", "avail": 0.90, "perf": 0.95, "qual": 0.50}]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result.entities[0].limiting_factor == "quality"

    def test_best_worst_entity(self):
        rows = [
            {"line": "Best", "avail": 0.95, "perf": 0.95, "qual": 0.99},
            {"line": "Worst", "avail": 0.50, "perf": 0.50, "qual": 0.50},
        ]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result.best_entity == "Best"
        assert result.worst_entity == "Worst"

    def test_sorted_descending_by_oee(self):
        rows = [
            {"line": "Low", "avail": 0.60, "perf": 0.60, "qual": 0.60},
            {"line": "High", "avail": 0.95, "perf": 0.95, "qual": 0.99},
            {"line": "Mid", "avail": 0.80, "perf": 0.80, "qual": 0.80},
        ]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        oees = [e.oee for e in result.entities]
        assert oees == sorted(oees, reverse=True)

    def test_averages_multiple_rows_per_entity(self):
        rows = [
            {"line": "L1", "avail": 0.80, "perf": 0.80, "qual": 0.80},
            {"line": "L1", "avail": 1.00, "perf": 1.00, "qual": 1.00},
        ]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        e = result.entities[0]
        assert abs(e.availability - 0.90) < 0.01
        assert abs(e.performance - 0.90) < 0.01
        assert abs(e.quality - 0.90) < 0.01

    def test_summary_contains_key_info(self):
        rows = [
            {"line": "L1", "avail": 0.90, "perf": 0.90, "qual": 0.90},
            {"line": "L2", "avail": 0.80, "perf": 0.80, "qual": 0.80},
        ]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert "OEE" in result.summary
        assert "L1" in result.summary or "L2" in result.summary

    def test_skips_invalid_non_numeric(self):
        rows = [
            {"line": "L1", "avail": "bad", "perf": 0.90, "qual": 0.90},
            {"line": "L2", "avail": 0.90, "perf": 0.90, "qual": 0.90},
        ]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result is not None
        assert len(result.entities) == 1

    def test_single_entity(self):
        rows = [{"line": "Solo", "avail": 0.90, "perf": 0.85, "qual": 0.95}]
        result = compute_oee(rows, "line", "avail", "perf", "qual")
        assert result is not None
        assert result.best_entity == "Solo"
        assert result.worst_entity == "Solo"
        assert len(result.entities) == 1


# ===================================================================
# Yield Analysis Tests
# ===================================================================


class TestComputeYieldAnalysis:
    def test_basic_yield(self):
        rows = [
            {"line": "L1", "inp": 1000, "out": 950},
            {"line": "L2", "inp": 1000, "out": 800},
        ]
        result = compute_yield_analysis(rows, "line", "inp", "out")
        assert result is not None
        l1 = [e for e in result.entities if e.entity == "L1"][0]
        assert l1.yield_pct == 95.0
        assert l1.waste_pct == 5.0

    def test_empty_returns_none(self):
        assert compute_yield_analysis([], "line", "inp", "out") is None

    def test_all_none_returns_none(self):
        rows = [{"line": None, "inp": None, "out": None}]
        assert compute_yield_analysis(rows, "line", "inp", "out") is None

    def test_with_defect_column(self):
        rows = [{"line": "L1", "inp": 1000, "out": 900, "defect": 80}]
        result = compute_yield_analysis(rows, "line", "inp", "out", defect_column="defect")
        e = result.entities[0]
        assert e.defect_rate == 8.0
        assert e.yield_pct == 90.0

    def test_no_defect_column_default_zero(self):
        rows = [{"line": "L1", "inp": 1000, "out": 950}]
        result = compute_yield_analysis(rows, "line", "inp", "out")
        assert result.entities[0].defect_rate == 0.0

    def test_zero_input_no_division_error(self):
        rows = [{"line": "L1", "inp": 0, "out": 0}]
        result = compute_yield_analysis(rows, "line", "inp", "out")
        assert result is not None
        assert result.entities[0].yield_pct == 0.0

    def test_best_worst_entity(self):
        rows = [
            {"line": "Good", "inp": 100, "out": 98},
            {"line": "Bad", "inp": 100, "out": 50},
        ]
        result = compute_yield_analysis(rows, "line", "inp", "out")
        assert result.best_entity == "Good"
        assert result.worst_entity == "Bad"

    def test_aggregates_multiple_rows(self):
        rows = [
            {"line": "L1", "inp": 500, "out": 480},
            {"line": "L1", "inp": 500, "out": 470},
        ]
        result = compute_yield_analysis(rows, "line", "inp", "out")
        e = result.entities[0]
        assert e.input_total == 1000
        assert e.output_total == 950
        assert e.yield_pct == 95.0

    def test_summary_text(self):
        rows = [
            {"line": "L1", "inp": 100, "out": 90},
            {"line": "L2", "inp": 100, "out": 85},
        ]
        result = compute_yield_analysis(rows, "line", "inp", "out")
        assert "Yield" in result.summary
        assert "L1" in result.summary

    def test_sorted_by_yield_descending(self):
        rows = [
            {"line": "Low", "inp": 100, "out": 50},
            {"line": "High", "inp": 100, "out": 99},
            {"line": "Mid", "inp": 100, "out": 75},
        ]
        result = compute_yield_analysis(rows, "line", "inp", "out")
        yields = [e.yield_pct for e in result.entities]
        assert yields == sorted(yields, reverse=True)


# ===================================================================
# Energy Efficiency Tests
# ===================================================================


class TestComputeEnergyEfficiency:
    def test_basic_energy(self):
        rows = [
            {"line": "L1", "output": 1000, "energy": 500},
            {"line": "L2", "output": 1000, "energy": 800},
        ]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        assert result is not None
        l1 = [e for e in result.entities if e.entity == "L1"][0]
        assert l1.specific_energy == 0.5

    def test_empty_returns_none(self):
        assert compute_energy_efficiency([], "line", "output", "energy") is None

    def test_all_none_returns_none(self):
        rows = [{"line": None, "output": None, "energy": None}]
        assert compute_energy_efficiency(rows, "line", "output", "energy") is None

    def test_zero_output_infinite_sec(self):
        rows = [
            {"line": "L1", "output": 0, "energy": 100},
            {"line": "L2", "output": 100, "energy": 50},
        ]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        assert result is not None
        l1 = [e for e in result.entities if e.entity == "L1"][0]
        assert l1.specific_energy == float("inf")
        assert l1.efficiency_grade == "Poor"

    def test_best_worst_entity(self):
        rows = [
            {"line": "Efficient", "output": 1000, "energy": 200},
            {"line": "Wasteful", "output": 1000, "energy": 1000},
        ]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        assert result.best_entity == "Efficient"
        assert result.worst_entity == "Wasteful"

    def test_sorted_ascending_by_sec(self):
        rows = [
            {"line": "High", "output": 100, "energy": 500},
            {"line": "Low", "output": 100, "energy": 100},
            {"line": "Mid", "output": 100, "energy": 250},
        ]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        secs = [e.specific_energy for e in result.entities]
        assert secs == sorted(secs)

    def test_potential_savings(self):
        rows = [
            {"line": "Best", "output": 100, "energy": 100},  # SEC=1.0
            {"line": "Worst", "output": 100, "energy": 200},  # SEC=2.0
        ]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        # Savings = (1 - 1.0/2.0) * 100 = 50%
        assert abs(result.potential_savings_pct - 50.0) < 0.1

    def test_efficiency_grades(self):
        rows = [
            {"line": "A", "output": 100, "energy": 100},  # SEC=1.0 (best)
            {"line": "B", "output": 100, "energy": 105},  # SEC=1.05 (within 10%)
            {"line": "C", "output": 100, "energy": 120},  # SEC=1.20 (within 25%)
            {"line": "D", "output": 100, "energy": 145},  # SEC=1.45 (within 50%)
            {"line": "E", "output": 100, "energy": 200},  # SEC=2.0 (>50%)
        ]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        grades = {e.entity: e.efficiency_grade for e in result.entities}
        assert grades["A"] == "Excellent"
        assert grades["B"] == "Excellent"
        assert grades["C"] == "Good"
        assert grades["D"] == "Average"
        assert grades["E"] == "Poor"

    def test_summary_text(self):
        rows = [
            {"line": "L1", "output": 100, "energy": 50},
            {"line": "L2", "output": 100, "energy": 80},
        ]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        assert "Energy" in result.summary or "SEC" in result.summary

    def test_single_entity(self):
        rows = [{"line": "Solo", "output": 200, "energy": 100}]
        result = compute_energy_efficiency(rows, "line", "output", "energy")
        assert result is not None
        assert result.best_entity == "Solo"
        assert result.worst_entity == "Solo"


# ===================================================================
# Waste Analysis Tests
# ===================================================================


class TestComputeWasteAnalysis:
    def test_basic_waste(self):
        rows = [
            {"line": "L1", "total": 1000, "waste": 50},
            {"line": "L2", "total": 1000, "waste": 100},
        ]
        result = compute_waste_analysis(rows, "line", "total", "waste")
        assert result is not None
        assert result.total_waste == 150
        assert result.total_production == 2000
        assert result.waste_pct == 7.5

    def test_empty_returns_none(self):
        assert compute_waste_analysis([], "line", "total", "waste") is None

    def test_all_none_returns_none(self):
        rows = [{"line": None, "total": None, "waste": None}]
        assert compute_waste_analysis(rows, "line", "total", "waste") is None

    def test_zero_production_no_division_error(self):
        rows = [{"line": "L1", "total": 0, "waste": 0}]
        result = compute_waste_analysis(rows, "line", "total", "waste")
        assert result is not None
        assert result.entities[0].waste_pct == 0.0

    def test_worst_wasters_ordering(self):
        rows = [
            {"line": "Clean", "total": 1000, "waste": 10},
            {"line": "Messy", "total": 1000, "waste": 200},
            {"line": "Average", "total": 1000, "waste": 50},
        ]
        result = compute_waste_analysis(rows, "line", "total", "waste")
        assert result.worst_wasters[0] == "Messy"

    def test_rank_assigned(self):
        rows = [
            {"line": "A", "total": 100, "waste": 5},
            {"line": "B", "total": 100, "waste": 20},
            {"line": "C", "total": 100, "waste": 10},
        ]
        result = compute_waste_analysis(rows, "line", "total", "waste")
        ranks = {e.entity: e.rank for e in result.entities}
        assert ranks["B"] == 1  # worst waste %
        assert ranks["C"] == 2
        assert ranks["A"] == 3

    def test_aggregates_multiple_rows(self):
        rows = [
            {"line": "L1", "total": 500, "waste": 20},
            {"line": "L1", "total": 500, "waste": 30},
        ]
        result = compute_waste_analysis(rows, "line", "total", "waste")
        e = result.entities[0]
        assert e.total_production == 1000
        assert e.total_waste == 50
        assert e.waste_pct == 5.0

    def test_summary_text(self):
        rows = [
            {"line": "L1", "total": 1000, "waste": 50},
            {"line": "L2", "total": 1000, "waste": 100},
        ]
        result = compute_waste_analysis(rows, "line", "total", "waste")
        assert "Waste" in result.summary or "waste" in result.summary

    def test_worst_wasters_capped_at_three(self):
        rows = [{"line": f"L{i}", "total": 100, "waste": i * 5} for i in range(10)]
        result = compute_waste_analysis(rows, "line", "total", "waste")
        assert len(result.worst_wasters) == 3


# ===================================================================
# Combined Report Tests
# ===================================================================


class TestEfficiencyReport:
    def test_all_none_reports_no_data(self):
        report = efficiency_report()
        assert "No analysis data provided" in report

    def test_oee_only(self):
        rows = [{"line": "L1", "avail": 0.90, "perf": 0.90, "qual": 0.90}]
        oee = compute_oee(rows, "line", "avail", "perf", "qual")
        report = efficiency_report(oee=oee)
        assert "OEE" in report
        assert "L1" in report

    def test_yield_only(self):
        rows = [{"line": "L1", "inp": 100, "out": 90}]
        yr = compute_yield_analysis(rows, "line", "inp", "out")
        report = efficiency_report(yield_result=yr)
        assert "Yield" in report

    def test_energy_only(self):
        rows = [{"line": "L1", "output": 100, "energy": 50}]
        er = compute_energy_efficiency(rows, "line", "output", "energy")
        report = efficiency_report(energy=er)
        assert "Energy" in report

    def test_waste_only(self):
        rows = [{"line": "L1", "total": 100, "waste": 5}]
        wr = compute_waste_analysis(rows, "line", "total", "waste")
        report = efficiency_report(waste=wr)
        assert "Waste" in report

    def test_combined_all_sections(self):
        oee_rows = [{"line": "L1", "avail": 0.90, "perf": 0.90, "qual": 0.90}]
        yield_rows = [{"line": "L1", "inp": 100, "out": 90}]
        energy_rows = [{"line": "L1", "output": 100, "energy": 50}]
        waste_rows = [{"line": "L1", "total": 100, "waste": 5}]

        oee = compute_oee(oee_rows, "line", "avail", "perf", "qual")
        yr = compute_yield_analysis(yield_rows, "line", "inp", "out")
        er = compute_energy_efficiency(energy_rows, "line", "output", "energy")
        wr = compute_waste_analysis(waste_rows, "line", "total", "waste")

        report = efficiency_report(oee=oee, yield_result=yr, energy=er, waste=wr)
        assert "Manufacturing Efficiency Report" in report
        assert "OEE" in report
        assert "Yield" in report
        assert "Energy" in report
        assert "Waste" in report

    def test_report_header_always_present(self):
        report = efficiency_report()
        assert "Manufacturing Efficiency Report" in report
        assert "=" * 40 in report
