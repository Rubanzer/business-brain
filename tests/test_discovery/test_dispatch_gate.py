"""Comprehensive tests for gate register and dispatch analytics module."""

import math

import pytest

from business_brain.discovery.dispatch_gate import (
    DispatchAnomaly,
    GateTrafficResult,
    MaterialMovement,
    MovementResult,
    PeriodTraffic,
    WeighbridgeResult,
    WeighEntry,
    analyze_gate_traffic,
    detect_dispatch_anomalies,
    format_dispatch_report,
    track_material_movement,
    weighbridge_analysis,
)


# ===================================================================
# Gate Traffic Tests
# ===================================================================


class TestAnalyzeGateTraffic:
    """Tests for analyze_gate_traffic."""

    # --- empty / None returns ---

    def test_empty_rows_returns_none(self):
        assert analyze_gate_traffic([], "time") is None

    def test_all_none_time_returns_none(self):
        rows = [{"time": None}, {"time": None}]
        assert analyze_gate_traffic(rows, "time") is None

    def test_missing_time_column_key_returns_none(self):
        """Rows that lack the time_column key entirely yield None."""
        rows = [{"other": "value"}, {"other": "value2"}]
        assert analyze_gate_traffic(rows, "time") is None

    # --- basic counting (no vehicle_column) ---

    def test_basic_period_counting_without_vehicle_column(self):
        rows = [
            {"time": "08:00"},
            {"time": "08:00"},
            {"time": "09:00"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.total_vehicles == 3
        assert len(result.periods) == 2

    def test_single_period(self):
        rows = [{"time": "morning"}, {"time": "morning"}]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.total_vehicles == 2
        assert len(result.periods) == 1
        # Peak and off-peak are the same when there is one period
        assert result.peak_period == "morning"
        assert result.off_peak_period == "morning"
        assert result.avg_per_period == 2.0

    def test_single_row(self):
        rows = [{"time": "noon"}]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.total_vehicles == 1
        assert result.avg_per_period == 1.0

    # --- peak / off-peak detection ---

    def test_peak_and_off_peak_detection(self):
        rows = [
            {"time": "08:00"},
            {"time": "08:00"},
            {"time": "08:00"},
            {"time": "09:00"},
            {"time": "10:00"},
            {"time": "10:00"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.peak_period == "08:00"
        assert result.off_peak_period == "09:00"

    def test_peak_with_three_periods(self):
        """Verify peak is highest, off-peak is lowest among 3 periods."""
        rows = [
            {"time": "A"},
            {"time": "A"},
            {"time": "A"},
            {"time": "A"},
            {"time": "B"},
            {"time": "B"},
            {"time": "C"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.peak_period == "A"
        assert result.off_peak_period == "C"

    # --- direction split ---

    def test_direction_split_in_out(self):
        rows = [
            {"time": "08:00", "dir": "in"},
            {"time": "08:00", "dir": "in"},
            {"time": "09:00", "dir": "out"},
        ]
        result = analyze_gate_traffic(rows, "time", direction_column="dir")
        assert result is not None
        assert result.direction_split is not None
        assert result.direction_split["in"] == 2
        assert result.direction_split["out"] == 1

    def test_direction_split_is_sorted_alphabetically(self):
        rows = [
            {"time": "08:00", "dir": "out"},
            {"time": "08:00", "dir": "in"},
        ]
        result = analyze_gate_traffic(rows, "time", direction_column="dir")
        assert result is not None
        assert result.direction_split is not None
        keys = list(result.direction_split.keys())
        assert keys == sorted(keys)

    def test_direction_column_with_all_none_values_gives_none_split(self):
        """Direction column specified but all values are None."""
        rows = [
            {"time": "08:00", "dir": None},
            {"time": "09:00", "dir": None},
        ]
        result = analyze_gate_traffic(rows, "time", direction_column="dir")
        assert result is not None
        # direction_counts is empty dict -- falsy, so dir_split should be None
        assert result.direction_split is None

    def test_no_direction_column_gives_none_split(self):
        rows = [{"time": "08:00"}, {"time": "09:00"}]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.direction_split is None

    def test_direction_values_are_lowered_and_stripped(self):
        rows = [
            {"time": "08:00", "dir": "  IN  "},
            {"time": "09:00", "dir": " Out"},
        ]
        result = analyze_gate_traffic(rows, "time", direction_column="dir")
        assert result is not None
        assert result.direction_split is not None
        assert "in" in result.direction_split
        assert "out" in result.direction_split

    def test_direction_summary_in_summary_string(self):
        rows = [
            {"time": "08:00", "dir": "in"},
            {"time": "09:00", "dir": "out"},
        ]
        result = analyze_gate_traffic(rows, "time", direction_column="dir")
        assert result is not None
        assert "Direction split" in result.summary
        assert "in=" in result.summary
        assert "out=" in result.summary

    # --- with vehicle_column ---

    def test_with_vehicle_column_counts_all_rows(self):
        rows = [
            {"time": "08:00", "vehicle": "TRK-001"},
            {"time": "08:00", "vehicle": "TRK-002"},
            {"time": "09:00", "vehicle": "TRK-003"},
        ]
        result = analyze_gate_traffic(rows, "time", vehicle_column="vehicle")
        assert result is not None
        assert result.total_vehicles == 3

    def test_skips_rows_with_none_vehicle(self):
        rows = [
            {"time": "08:00", "vehicle": "TRK-001"},
            {"time": "08:00", "vehicle": None},
            {"time": "09:00", "vehicle": "TRK-002"},
        ]
        result = analyze_gate_traffic(rows, "time", vehicle_column="vehicle")
        assert result is not None
        assert result.total_vehicles == 2

    def test_all_none_vehicles_returns_none(self):
        rows = [
            {"time": "08:00", "vehicle": None},
            {"time": "09:00", "vehicle": None},
        ]
        result = analyze_gate_traffic(rows, "time", vehicle_column="vehicle")
        assert result is None

    def test_vehicle_column_missing_from_rows_returns_none(self):
        """Rows don't even have the vehicle key -- row.get returns None."""
        rows = [
            {"time": "08:00"},
            {"time": "09:00"},
        ]
        result = analyze_gate_traffic(rows, "time", vehicle_column="vehicle")
        assert result is None

    # --- avg_per_period ---

    def test_avg_per_period_even_distribution(self):
        rows = [
            {"time": "morning"},
            {"time": "morning"},
            {"time": "afternoon"},
            {"time": "afternoon"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.avg_per_period == 2.0

    def test_avg_per_period_uneven(self):
        rows = [
            {"time": "A"},
            {"time": "A"},
            {"time": "A"},
            {"time": "B"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.avg_per_period == 2.0

    def test_avg_per_period_fractional(self):
        rows = [
            {"time": "A"},
            {"time": "A"},
            {"time": "A"},
            {"time": "B"},
            {"time": "C"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        # 5 / 3 = 1.6667 -> rounded to 1.67
        assert result.avg_per_period == pytest.approx(1.67, abs=0.01)

    # --- pct_of_total ---

    def test_pct_of_total(self):
        rows = [
            {"time": "A"},
            {"time": "A"},
            {"time": "A"},
            {"time": "B"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        period_a = [p for p in result.periods if p.period == "A"][0]
        period_b = [p for p in result.periods if p.period == "B"][0]
        assert period_a.pct_of_total == 75.0
        assert period_b.pct_of_total == 25.0

    def test_pct_of_total_single_period_is_100(self):
        rows = [{"time": "X"}, {"time": "X"}, {"time": "X"}]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.periods[0].pct_of_total == 100.0

    # --- summary ---

    def test_summary_contains_key_info(self):
        rows = [{"time": "08:00"}, {"time": "09:00"}]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert "2 periods" in result.summary
        assert "2 total vehicle passages" in result.summary
        assert "Peak" in result.summary
        assert "Off-peak" in result.summary

    def test_summary_no_direction_info_when_no_direction_column(self):
        rows = [{"time": "08:00"}, {"time": "09:00"}]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert "Direction split" not in result.summary

    # --- return type ---

    def test_return_type(self):
        rows = [{"time": "A"}]
        result = analyze_gate_traffic(rows, "time")
        assert isinstance(result, GateTrafficResult)
        assert isinstance(result.periods[0], PeriodTraffic)

    # --- time_column values are stringified ---

    def test_numeric_time_values_stringified(self):
        rows = [{"time": 8}, {"time": 8}, {"time": 9}]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.peak_period == "8"
        assert result.off_peak_period == "9"

    # --- mixed valid/invalid rows ---

    def test_some_rows_missing_time_are_skipped(self):
        rows = [
            {"time": "08:00"},
            {"time": None},
            {"time": "09:00"},
        ]
        result = analyze_gate_traffic(rows, "time")
        assert result is not None
        assert result.total_vehicles == 2


# ===================================================================
# Weighbridge Analysis Tests
# ===================================================================


class TestWeighbridgeAnalysis:
    """Tests for weighbridge_analysis."""

    # --- empty / None returns ---

    def test_empty_rows_returns_none(self):
        assert weighbridge_analysis([], "v", "g", "t") is None

    def test_all_none_returns_none(self):
        rows = [{"vehicle": None, "gross": None, "tare": None}]
        assert weighbridge_analysis(rows, "vehicle", "gross", "tare") is None

    def test_missing_vehicle_returns_none(self):
        rows = [{"vehicle": None, "gross": 5000, "tare": 2000}]
        assert weighbridge_analysis(rows, "vehicle", "gross", "tare") is None

    def test_missing_gross_returns_none(self):
        rows = [{"vehicle": "A", "gross": None, "tare": 2000}]
        assert weighbridge_analysis(rows, "vehicle", "gross", "tare") is None

    def test_missing_tare_returns_none(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": None}]
        assert weighbridge_analysis(rows, "vehicle", "gross", "tare") is None

    def test_non_numeric_gross_skipped(self):
        rows = [
            {"vehicle": "A", "gross": "not_a_number", "tare": 2000},
            {"vehicle": "B", "gross": 5000, "tare": 2000},
        ]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.total_vehicles == 1
        assert result.entries[0].vehicle == "B"

    def test_non_numeric_tare_skipped(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": "bad"},
        ]
        assert weighbridge_analysis(rows, "vehicle", "gross", "tare") is None

    # --- net weight computation ---

    def test_net_weight_is_gross_minus_tare(self):
        rows = [
            {"vehicle": "TRK-001", "gross": 5000, "tare": 2000},
            {"vehicle": "TRK-002", "gross": 6000, "tare": 2500},
        ]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].net_weight == 3000.0
        assert result.entries[1].net_weight == 3500.0

    def test_negative_net_weight(self):
        """Tare > gross results in negative net weight."""
        rows = [{"vehicle": "A", "gross": 2000, "tare": 5000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].net_weight == -3000.0

    def test_zero_net_weight(self):
        rows = [{"vehicle": "A", "gross": 3000, "tare": 3000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].net_weight == 0.0

    def test_decimal_weights(self):
        rows = [{"vehicle": "A", "gross": 5000.75, "tare": 2000.25}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].net_weight == pytest.approx(3000.5, abs=0.01)

    def test_string_numeric_values_accepted(self):
        """Weights provided as strings should be parsed via _safe_float."""
        rows = [{"vehicle": "A", "gross": "5000", "tare": "2000"}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].net_weight == 3000.0

    # --- total_net_weight and avg_net_weight ---

    def test_total_and_avg_net_weight(self):
        rows = [
            {"vehicle": "A", "gross": 10000, "tare": 4000},  # net = 6000
            {"vehicle": "B", "gross": 8000, "tare": 4000},   # net = 4000
        ]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.total_net_weight == 10000.0
        assert result.avg_net_weight == 5000.0
        assert result.total_vehicles == 2

    def test_total_net_weight_single_entry(self):
        rows = [{"vehicle": "A", "gross": 7000, "tare": 3000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.total_net_weight == 4000.0
        assert result.avg_net_weight == 4000.0

    # --- by_material totals ---

    def test_by_material_totals(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Sand"},
            {"vehicle": "B", "gross": 6000, "tare": 2000, "mat": "Sand"},
            {"vehicle": "C", "gross": 7000, "tare": 3000, "mat": "Gravel"},
        ]
        result = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        assert result is not None
        assert result.by_material is not None
        assert result.by_material["Sand"] == 7000.0   # 3000 + 4000
        assert result.by_material["Gravel"] == 4000.0

    def test_by_material_is_sorted(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Zinc"},
            {"vehicle": "B", "gross": 5000, "tare": 2000, "mat": "Aluminium"},
        ]
        result = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        assert result is not None
        assert result.by_material is not None
        keys = list(result.by_material.keys())
        assert keys == sorted(keys)

    def test_material_column_with_none_material_values(self):
        """material_column given but some rows have None material."""
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Sand"},
            {"vehicle": "B", "gross": 5000, "tare": 2000, "mat": None},
        ]
        result = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        assert result is not None
        # Both entries are valid vehicles
        assert result.total_vehicles == 2
        # Only Sand tracked in by_material
        assert result.by_material is not None
        assert "Sand" in result.by_material
        assert len(result.by_material) == 1

    def test_all_material_values_none_gives_none_by_material(self):
        """material_column given but ALL material values are None."""
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": None},
        ]
        result = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        assert result is not None
        # material_totals dict is empty (falsy) so by_material is None
        assert result.by_material is None

    def test_no_material_column_gives_none_by_material(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.by_material is None

    # --- WeighEntry dataclass ---

    def test_weigh_entry_stores_material(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Coal"},
        ]
        result = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        assert result is not None
        assert result.entries[0].material == "Coal"

    def test_weigh_entry_material_is_none_without_material_column(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].material is None

    def test_weigh_entry_gross_and_tare_stored(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].gross_weight == 5000.0
        assert result.entries[0].tare_weight == 2000.0

    def test_vehicle_stringified(self):
        """Numeric vehicle IDs are converted to strings."""
        rows = [{"vehicle": 101, "gross": 5000, "tare": 2000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.entries[0].vehicle == "101"

    # --- summary ---

    def test_summary_contains_key_info(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert "1 entries" in result.summary
        assert "net weight" in result.summary

    def test_summary_includes_material_info(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Sand"},
        ]
        result = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        assert result is not None
        assert "By material" in result.summary
        assert "Sand" in result.summary

    # --- mixed valid/invalid ---

    def test_valid_rows_among_invalid(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000},      # valid
            {"vehicle": None, "gross": 5000, "tare": 2000},      # skipped
            {"vehicle": "C", "gross": "bad", "tare": 2000},      # skipped
            {"vehicle": "D", "gross": 8000, "tare": 3000},      # valid
        ]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.total_vehicles == 2
        vehicles = [e.vehicle for e in result.entries]
        assert "A" in vehicles
        assert "D" in vehicles

    # --- return type ---

    def test_return_type(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert isinstance(result, WeighbridgeResult)
        assert isinstance(result.entries[0], WeighEntry)

    # --- rounding ---

    def test_net_weight_rounded_to_4_decimals(self):
        rows = [{"vehicle": "A", "gross": 5000.12345, "tare": 2000.11111}]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        # net = 3000.01234, rounded to 4 decimals
        assert result.entries[0].net_weight == round(3000.01234, 4)


# ===================================================================
# Material Movement Tests
# ===================================================================


class TestTrackMaterialMovement:
    """Tests for track_material_movement."""

    # --- empty / None returns ---

    def test_empty_rows_returns_none(self):
        assert track_material_movement([], "m", "q", "d") is None

    def test_all_none_returns_none(self):
        rows = [{"mat": None, "qty": None, "dir": None}]
        assert track_material_movement(rows, "mat", "qty", "dir") is None

    def test_missing_columns_returns_none(self):
        rows = [{"other": "value"}]
        assert track_material_movement(rows, "mat", "qty", "dir") is None

    # --- inward direction aliases ---

    @pytest.mark.parametrize("alias", ["in", "inward", "incoming", "receipt"])
    def test_inward_direction_aliases(self, alias):
        rows = [{"mat": "X", "qty": 10, "dir": alias}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].inward_qty == 10.0
        assert result.materials[0].outward_qty == 0.0

    @pytest.mark.parametrize("alias", ["IN", "Inward", "INCOMING", "Receipt"])
    def test_inward_direction_case_insensitive(self, alias):
        rows = [{"mat": "X", "qty": 10, "dir": alias}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].inward_qty == 10.0

    @pytest.mark.parametrize("alias", [" in ", "  inward  ", " incoming "])
    def test_inward_direction_stripped(self, alias):
        rows = [{"mat": "X", "qty": 10, "dir": alias}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].inward_qty == 10.0

    # --- outward direction aliases ---

    @pytest.mark.parametrize("alias", ["out", "outward", "outgoing", "dispatch"])
    def test_outward_direction_aliases(self, alias):
        rows = [{"mat": "X", "qty": 10, "dir": alias}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].outward_qty == 10.0
        assert result.materials[0].inward_qty == 0.0

    @pytest.mark.parametrize("alias", ["OUT", "Outward", "OUTGOING", "Dispatch"])
    def test_outward_direction_case_insensitive(self, alias):
        rows = [{"mat": "X", "qty": 10, "dir": alias}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].outward_qty == 10.0

    # --- unrecognized directions ---

    def test_unrecognized_direction_skipped(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "transfer"},
            {"mat": "A", "qty": 50, "dir": "in"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].movement_count == 1
        assert result.materials[0].inward_qty == 50.0

    def test_all_unrecognized_directions_returns_none(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "transfer"},
            {"mat": "B", "qty": 50, "dir": "internal"},
            {"mat": "C", "qty": 30, "dir": "moved"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is None

    def test_empty_string_direction_skipped(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": ""},
        ]
        # Empty string stripped lower is "" which is not in inward_keys or outward_keys
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is None

    # --- net_qty calculation ---

    def test_net_qty_positive(self):
        rows = [
            {"mat": "Steel", "qty": 100, "dir": "in"},
            {"mat": "Steel", "qty": 40, "dir": "out"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        steel = result.materials[0]
        assert steel.net_qty == 60.0

    def test_net_qty_negative_when_more_outward(self):
        rows = [
            {"mat": "X", "qty": 20, "dir": "in"},
            {"mat": "X", "qty": 80, "dir": "out"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].net_qty == -60.0
        assert result.net_movement == -60.0

    def test_net_qty_zero(self):
        rows = [
            {"mat": "X", "qty": 50, "dir": "in"},
            {"mat": "X", "qty": 50, "dir": "out"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].net_qty == 0.0
        assert result.net_movement == 0.0

    # --- basic movement ---

    def test_basic_movement_multiple_materials(self):
        rows = [
            {"mat": "Steel", "qty": 100, "dir": "in"},
            {"mat": "Steel", "qty": 40, "dir": "out"},
            {"mat": "Cement", "qty": 200, "dir": "in"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert len(result.materials) == 2
        # Materials should be sorted alphabetically
        assert result.materials[0].material == "Cement"
        assert result.materials[1].material == "Steel"

    def test_materials_sorted_alphabetically(self):
        rows = [
            {"mat": "Zinc", "qty": 10, "dir": "in"},
            {"mat": "Aluminium", "qty": 20, "dir": "in"},
            {"mat": "Copper", "qty": 30, "dir": "in"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        names = [m.material for m in result.materials]
        assert names == ["Aluminium", "Copper", "Zinc"]

    def test_movement_count(self):
        rows = [
            {"mat": "Steel", "qty": 100, "dir": "in"},
            {"mat": "Steel", "qty": 40, "dir": "out"},
            {"mat": "Steel", "qty": 60, "dir": "incoming"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].movement_count == 3

    # --- total_inward / total_outward / net_movement ---

    def test_total_inward_outward_net(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "inward"},
            {"mat": "B", "qty": 50, "dir": "outward"},
            {"mat": "A", "qty": 30, "dir": "outgoing"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.total_inward == 100.0
        assert result.total_outward == 80.0  # 50 + 30
        assert result.net_movement == 20.0

    def test_only_inward_movements(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "in"},
            {"mat": "B", "qty": 200, "dir": "receipt"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.total_inward == 300.0
        assert result.total_outward == 0.0
        assert result.net_movement == 300.0

    def test_only_outward_movements(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "dispatch"},
            {"mat": "B", "qty": 200, "dir": "out"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.total_inward == 0.0
        assert result.total_outward == 300.0
        assert result.net_movement == -300.0

    # --- time_column (reserved for future use) ---

    def test_time_column_accepted_but_not_used(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "in", "time": "08:00"},
        ]
        result = track_material_movement(
            rows, "mat", "qty", "dir", time_column="time"
        )
        assert result is not None
        assert result.materials[0].inward_qty == 100.0

    # --- string quantities ---

    def test_string_quantity_parsed(self):
        rows = [{"mat": "A", "qty": "100.5", "dir": "in"}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].inward_qty == pytest.approx(100.5)

    def test_non_numeric_quantity_skipped(self):
        rows = [
            {"mat": "A", "qty": "bad", "dir": "in"},
            {"mat": "A", "qty": 50, "dir": "in"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].inward_qty == 50.0

    # --- summary ---

    def test_summary_contains_key_info(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "in"},
            {"mat": "B", "qty": 50, "dir": "out"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert "2 materials" in result.summary
        assert "inward" in result.summary
        assert "outward" in result.summary

    # --- return type ---

    def test_return_type(self):
        rows = [{"mat": "A", "qty": 100, "dir": "in"}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert isinstance(result, MovementResult)
        assert isinstance(result.materials[0], MaterialMovement)

    # --- rounding ---

    def test_quantities_rounded_to_4_decimals(self):
        rows = [{"mat": "A", "qty": 100.123456, "dir": "in"}]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        assert result.materials[0].inward_qty == round(100.123456, 4)


# ===================================================================
# Dispatch Anomaly Detection Tests
# ===================================================================


class TestDetectDispatchAnomalies:
    """Tests for detect_dispatch_anomalies."""

    # --- empty / invalid returns ---

    def test_empty_rows_returns_empty_list(self):
        assert detect_dispatch_anomalies([], "v", "w") == []

    def test_all_none_returns_empty_list(self):
        rows = [{"vehicle": None, "weight": None}]
        assert detect_dispatch_anomalies(rows, "vehicle", "weight") == []

    def test_missing_columns_returns_empty_list(self):
        rows = [{"other": "value"}]
        assert detect_dispatch_anomalies(rows, "vehicle", "weight") == []

    def test_non_numeric_weight_skipped(self):
        rows = [{"vehicle": "A", "weight": "bad"}]
        assert detect_dispatch_anomalies(rows, "vehicle", "weight") == []

    # --- explicit bounds: overweight ---

    def test_overweight_detected_with_explicit_bounds(self):
        rows = [
            {"vehicle": "A", "weight": 5000},
            {"vehicle": "B", "weight": 9000},
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        overweight = [a for a in anomalies if a.anomaly_type == "overweight"]
        assert len(overweight) == 1
        assert overweight[0].vehicle == "B"
        assert overweight[0].weight == 9000.0

    def test_weight_exactly_at_max_is_normal(self):
        """Weight equal to expected_max is NOT an anomaly (> not >=)."""
        rows = [{"vehicle": "A", "weight": 6000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 0

    def test_weight_just_above_max_is_overweight(self):
        rows = [{"vehicle": "A", "weight": 6000.01}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "overweight"

    # --- explicit bounds: underweight ---

    def test_underweight_detected_with_explicit_bounds(self):
        rows = [
            {"vehicle": "A", "weight": 5000},
            {"vehicle": "B", "weight": 1000},
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        underweight = [a for a in anomalies if a.anomaly_type == "underweight"]
        assert len(underweight) == 1
        assert underweight[0].vehicle == "B"

    def test_weight_exactly_at_min_is_normal(self):
        """Weight equal to expected_min is NOT an anomaly (< not <=)."""
        rows = [{"vehicle": "A", "weight": 4000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 0

    def test_weight_just_below_min_is_underweight(self):
        rows = [{"vehicle": "A", "weight": 3999.99}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "underweight"

    # --- suspicious (zero/negative) ---

    def test_suspicious_zero_weight(self):
        rows = [
            {"vehicle": "A", "weight": 5000},
            {"vehicle": "B", "weight": 0},
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        suspicious = [a for a in anomalies if a.anomaly_type == "suspicious"]
        assert len(suspicious) == 1
        assert suspicious[0].vehicle == "B"
        assert suspicious[0].weight == 0.0

    def test_suspicious_negative_weight(self):
        rows = [
            {"vehicle": "A", "weight": 5000},
            {"vehicle": "B", "weight": -100},
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        suspicious = [a for a in anomalies if a.anomaly_type == "suspicious"]
        assert len(suspicious) == 1
        assert suspicious[0].vehicle == "B"
        assert suspicious[0].weight == -100.0

    def test_suspicious_takes_priority_over_underweight(self):
        """Zero weight is below expected_min, but classified as suspicious, not underweight."""
        rows = [{"vehicle": "A", "weight": 0}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=1000, expected_max=5000
        )
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "suspicious"

    def test_negative_takes_priority_over_underweight(self):
        """Negative weight is below expected_min, but classified as suspicious."""
        rows = [{"vehicle": "A", "weight": -50}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=1000, expected_max=5000
        )
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "suspicious"

    # --- no anomalies ---

    def test_no_anomalies_when_all_within_range(self):
        rows = [
            {"vehicle": "A", "weight": 5000},
            {"vehicle": "B", "weight": 5100},
            {"vehicle": "C", "weight": 4900},
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert anomalies == []

    def test_no_anomalies_single_vehicle_at_midpoint(self):
        rows = [{"vehicle": "A", "weight": 5000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert anomalies == []

    # --- auto-derived bounds (mean +/- 2*std) ---

    def test_auto_bounds_detect_outlier(self):
        """With many identical values and one extreme outlier, the outlier is caught."""
        rows = [{"vehicle": f"V{i}", "weight": 5000} for i in range(10)]
        rows.append({"vehicle": "OUTLIER", "weight": 50000})
        anomalies = detect_dispatch_anomalies(rows, "vehicle", "weight")
        outlier_found = any(a.vehicle == "OUTLIER" for a in anomalies)
        assert outlier_found

    def test_auto_bounds_no_anomaly_for_tight_cluster(self):
        """All weights are identical -- std=0 so bounds are tight, no anomalies."""
        rows = [{"vehicle": f"V{i}", "weight": 5000} for i in range(5)]
        anomalies = detect_dispatch_anomalies(rows, "vehicle", "weight")
        assert anomalies == []

    def test_auto_bounds_single_entry(self):
        """Single entry: std=0, bounds = [weight, weight], no anomaly."""
        rows = [{"vehicle": "A", "weight": 5000}]
        anomalies = detect_dispatch_anomalies(rows, "vehicle", "weight")
        assert anomalies == []

    def test_auto_bounds_two_entries_same_weight(self):
        rows = [
            {"vehicle": "A", "weight": 100},
            {"vehicle": "B", "weight": 100},
        ]
        anomalies = detect_dispatch_anomalies(rows, "vehicle", "weight")
        assert anomalies == []

    def test_auto_bounds_computed_correctly(self):
        """Manually verify the auto-computed bounds: mean +/- 2*std."""
        rows = [
            {"vehicle": "A", "weight": 100},
            {"vehicle": "B", "weight": 200},
            {"vehicle": "C", "weight": 300},
        ]
        # mean = 200, variance = ((100-200)^2 + (200-200)^2 + (300-200)^2)/3
        # = (10000 + 0 + 10000)/3 = 6666.67
        # std = sqrt(6666.67) ~ 81.65
        # min = 200 - 2*81.65 = 36.70
        # max = 200 + 2*81.65 = 363.30
        anomalies = detect_dispatch_anomalies(rows, "vehicle", "weight")
        # All values 100, 200, 300 are within [36.7, 363.3]
        assert anomalies == []

    def test_auto_bounds_flag_value_outside(self):
        """Verify that a value clearly outside mean +/- 2*std is flagged."""
        # 10 identical values of 100, plus one extreme
        rows = [{"vehicle": f"V{i}", "weight": 100} for i in range(10)]
        rows.append({"vehicle": "X", "weight": 500})
        # mean = (10*100 + 500)/11 = 1500/11 ~ 136.36
        # Population std computed from the data
        anomalies = detect_dispatch_anomalies(rows, "vehicle", "weight")
        assert any(a.vehicle == "X" for a in anomalies)

    def test_partial_auto_bounds_only_min_provided(self):
        """Only expected_min provided, expected_max auto-derived from mean + 2*std."""
        # 20 identical values at 100 to keep std very small, plus one extreme
        rows = [{"vehicle": f"V{i}", "weight": 100} for i in range(20)]
        rows.append({"vehicle": "D", "weight": 50000})
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=50
        )
        # expected_max is auto-derived; the outlier should exceed it
        outlier_found = any(a.vehicle == "D" for a in anomalies)
        assert outlier_found

    def test_partial_auto_bounds_only_max_provided(self):
        """Only expected_max provided, expected_min auto-derived from mean - 2*std."""
        # Use explicit min via auto-derive: 20 values at 5000 keep std small
        rows = [{"vehicle": f"V{i}", "weight": 5000} for i in range(20)]
        rows.append({"vehicle": "D", "weight": 1})  # very low outlier
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_max=100000
        )
        # expected_min is auto-derived; the outlier should fall below it
        low_found = any(a.vehicle == "D" for a in anomalies)
        assert low_found

    # --- expected_range stored ---

    def test_expected_range_stored_on_anomaly(self):
        rows = [{"vehicle": "A", "weight": 9000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        assert anomalies[0].expected_range == (4000.0, 6000.0)

    def test_expected_range_rounded(self):
        rows = [{"vehicle": "A", "weight": 0}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=3.141592653, expected_max=6.283185307
        )
        assert len(anomalies) == 1
        assert anomalies[0].expected_range == (round(3.141592653, 4), round(6.283185307, 4))

    # --- deviation_pct ---

    def test_deviation_pct_calculated(self):
        rows = [{"vehicle": "A", "weight": 8000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        # midpoint = 5000, deviation = |8000 - 5000| / 5000 * 100 = 60%
        assert anomalies[0].deviation_pct == 60.0

    def test_deviation_pct_for_underweight(self):
        rows = [{"vehicle": "A", "weight": 2000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        # midpoint = 5000, deviation = |2000 - 5000| / 5000 * 100 = 60%
        assert anomalies[0].deviation_pct == 60.0

    def test_deviation_pct_zero_when_midpoint_zero(self):
        """When midpoint is 0, deviation_pct should be 0.0."""
        rows = [{"vehicle": "A", "weight": -10}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=-5, expected_max=5
        )
        # midpoint = 0, weight <= 0 -> suspicious
        suspicious = [a for a in anomalies if a.anomaly_type == "suspicious"]
        assert len(suspicious) == 1
        assert suspicious[0].deviation_pct == 0.0

    # --- multiple anomalies ---

    def test_multiple_anomaly_types_in_same_batch(self):
        rows = [
            {"vehicle": "A", "weight": 5000},     # normal
            {"vehicle": "B", "weight": 9000},     # overweight
            {"vehicle": "C", "weight": 1000},     # underweight
            {"vehicle": "D", "weight": 0},        # suspicious
            {"vehicle": "E", "weight": -50},      # suspicious
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        types = {a.anomaly_type for a in anomalies}
        assert "overweight" in types
        assert "underweight" in types
        assert "suspicious" in types
        vehicles = {a.vehicle for a in anomalies}
        assert "A" not in vehicles  # normal vehicle not flagged
        assert "B" in vehicles
        assert "C" in vehicles
        assert "D" in vehicles
        assert "E" in vehicles

    # --- return type ---

    def test_return_type(self):
        rows = [{"vehicle": "A", "weight": 9000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert isinstance(anomalies, list)
        assert isinstance(anomalies[0], DispatchAnomaly)

    # --- string weight values ---

    def test_string_weight_values_parsed(self):
        rows = [{"vehicle": "A", "weight": "9000"}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        assert anomalies[0].weight == 9000.0

    # --- vehicle stringification ---

    def test_vehicle_values_stringified(self):
        rows = [{"vehicle": 101, "weight": 9000}]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        assert anomalies[0].vehicle == "101"

    # --- mixed valid/invalid ---

    def test_invalid_rows_skipped(self):
        rows = [
            {"vehicle": "A", "weight": 9000},       # overweight anomaly
            {"vehicle": None, "weight": 9000},       # skipped
            {"vehicle": "C", "weight": "bad"},       # skipped
            {"vehicle": "D", "weight": None},        # skipped
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        assert len(anomalies) == 1
        assert anomalies[0].vehicle == "A"


# ===================================================================
# Combined Report Tests
# ===================================================================


class TestFormatDispatchReport:
    """Tests for format_dispatch_report."""

    # --- no sections ---

    def test_no_data_provided(self):
        report = format_dispatch_report()
        assert "No analysis data provided." in report

    def test_all_none_explicitly(self):
        report = format_dispatch_report(traffic=None, weighbridge=None, movement=None)
        assert "No analysis data provided." in report

    # --- header always present ---

    def test_header_always_present_no_data(self):
        report = format_dispatch_report()
        assert "Dispatch & Gate Report" in report
        assert "=" * 50 in report

    def test_header_always_present_with_data(self):
        rows = [{"time": "08:00"}]
        traffic = analyze_gate_traffic(rows, "time")
        report = format_dispatch_report(traffic=traffic)
        assert "Dispatch & Gate Report" in report
        assert "=" * 50 in report

    # --- traffic section ---

    def test_traffic_section_present(self):
        rows = [
            {"time": "08:00"},
            {"time": "09:00"},
        ]
        traffic = analyze_gate_traffic(rows, "time")
        report = format_dispatch_report(traffic=traffic)
        assert "Gate Traffic" in report
        assert "-" * 48 in report
        assert "08:00" in report
        assert "09:00" in report
        assert "No analysis data provided" not in report

    def test_traffic_section_vehicle_counts(self):
        rows = [
            {"time": "morning"},
            {"time": "morning"},
            {"time": "afternoon"},
        ]
        traffic = analyze_gate_traffic(rows, "time")
        report = format_dispatch_report(traffic=traffic)
        assert "2 vehicles" in report  # morning has 2
        assert "1 vehicles" in report or "1 vehicle" in report  # afternoon has 1

    def test_traffic_section_totals_line(self):
        rows = [
            {"time": "A"},
            {"time": "B"},
        ]
        traffic = analyze_gate_traffic(rows, "time")
        report = format_dispatch_report(traffic=traffic)
        assert "Total: 2" in report
        assert "Avg/period" in report
        assert "Peak:" in report
        assert "Off-peak:" in report

    def test_traffic_section_with_direction_split(self):
        rows = [
            {"time": "A", "dir": "in"},
            {"time": "B", "dir": "out"},
        ]
        traffic = analyze_gate_traffic(rows, "time", direction_column="dir")
        report = format_dispatch_report(traffic=traffic)
        assert "Direction:" in report
        assert "in=" in report
        assert "out=" in report

    def test_traffic_section_no_direction_info_when_no_split(self):
        rows = [{"time": "A"}, {"time": "B"}]
        traffic = analyze_gate_traffic(rows, "time")
        report = format_dispatch_report(traffic=traffic)
        assert "Direction:" not in report

    # --- weighbridge section ---

    def test_weighbridge_section_present(self):
        rows = [
            {"vehicle": "TRK-001", "gross": 5000, "tare": 2000},
        ]
        wb = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        report = format_dispatch_report(weighbridge=wb)
        assert "Weighbridge Analysis" in report
        assert "TRK-001" in report
        assert "No analysis data provided" not in report

    def test_weighbridge_section_shows_gross_tare_net(self):
        rows = [
            {"vehicle": "TRK-001", "gross": 5000, "tare": 2000},
        ]
        wb = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        report = format_dispatch_report(weighbridge=wb)
        assert "gross=5000.0" in report
        assert "tare=2000.0" in report
        assert "net=3000.0" in report

    def test_weighbridge_section_totals_line(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000},
            {"vehicle": "B", "gross": 6000, "tare": 2500},
        ]
        wb = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        report = format_dispatch_report(weighbridge=wb)
        assert "Total net:" in report
        assert "Avg net:" in report
        assert "Vehicles: 2" in report

    def test_weighbridge_section_with_material(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Coal"},
        ]
        wb = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        report = format_dispatch_report(weighbridge=wb)
        assert "[Coal]" in report
        assert "By material:" in report
        assert "Coal=" in report

    def test_weighbridge_section_without_material_no_by_material_line(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        wb = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        report = format_dispatch_report(weighbridge=wb)
        assert "By material:" not in report

    # --- movement section ---

    def test_movement_section_present(self):
        rows = [
            {"mat": "Steel", "qty": 100, "dir": "in"},
        ]
        mv = track_material_movement(rows, "mat", "qty", "dir")
        report = format_dispatch_report(movement=mv)
        assert "Material Movement" in report
        assert "Steel" in report
        assert "No analysis data provided" not in report

    def test_movement_section_shows_in_out_net_moves(self):
        rows = [
            {"mat": "Steel", "qty": 100, "dir": "in"},
            {"mat": "Steel", "qty": 40, "dir": "out"},
        ]
        mv = track_material_movement(rows, "mat", "qty", "dir")
        report = format_dispatch_report(movement=mv)
        assert "in=100.0" in report
        assert "out=40.0" in report
        assert "net=60.0" in report
        assert "moves=2" in report

    def test_movement_section_totals_line(self):
        rows = [
            {"mat": "A", "qty": 100, "dir": "in"},
            {"mat": "B", "qty": 50, "dir": "out"},
        ]
        mv = track_material_movement(rows, "mat", "qty", "dir")
        report = format_dispatch_report(movement=mv)
        assert "Total inward:" in report
        assert "Total outward:" in report
        assert "Net:" in report

    # --- all sections combined ---

    def test_all_sections_combined(self):
        traffic_rows = [{"time": "08:00"}, {"time": "09:00"}]
        wb_rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        mv_rows = [{"mat": "Steel", "qty": 100, "dir": "in"}]

        traffic = analyze_gate_traffic(traffic_rows, "time")
        wb = weighbridge_analysis(wb_rows, "vehicle", "gross", "tare")
        mv = track_material_movement(mv_rows, "mat", "qty", "dir")

        report = format_dispatch_report(traffic=traffic, weighbridge=wb, movement=mv)
        assert "Dispatch & Gate Report" in report
        assert "Gate Traffic" in report
        assert "Weighbridge Analysis" in report
        assert "Material Movement" in report
        assert "No analysis data provided" not in report

    # --- partial sections ---

    def test_only_traffic_section(self):
        rows = [{"time": "08:00"}]
        traffic = analyze_gate_traffic(rows, "time")
        report = format_dispatch_report(traffic=traffic)
        assert "Gate Traffic" in report
        assert "Weighbridge" not in report
        assert "Material Movement" not in report
        assert "No analysis data provided" not in report

    def test_only_weighbridge_section(self):
        rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        wb = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        report = format_dispatch_report(weighbridge=wb)
        assert "Gate Traffic" not in report
        assert "Weighbridge Analysis" in report
        assert "Material Movement" not in report
        assert "No analysis data provided" not in report

    def test_only_movement_section(self):
        rows = [{"mat": "A", "qty": 100, "dir": "in"}]
        mv = track_material_movement(rows, "mat", "qty", "dir")
        report = format_dispatch_report(movement=mv)
        assert "Gate Traffic" not in report
        assert "Weighbridge" not in report
        assert "Material Movement" in report
        assert "No analysis data provided" not in report

    def test_traffic_and_weighbridge_no_movement(self):
        traffic_rows = [{"time": "08:00"}]
        wb_rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        traffic = analyze_gate_traffic(traffic_rows, "time")
        wb = weighbridge_analysis(wb_rows, "vehicle", "gross", "tare")
        report = format_dispatch_report(traffic=traffic, weighbridge=wb)
        assert "Gate Traffic" in report
        assert "Weighbridge Analysis" in report
        assert "Material Movement" not in report

    def test_traffic_and_movement_no_weighbridge(self):
        traffic_rows = [{"time": "08:00"}]
        mv_rows = [{"mat": "A", "qty": 100, "dir": "in"}]
        traffic = analyze_gate_traffic(traffic_rows, "time")
        mv = track_material_movement(mv_rows, "mat", "qty", "dir")
        report = format_dispatch_report(traffic=traffic, movement=mv)
        assert "Gate Traffic" in report
        assert "Material Movement" in report
        assert "Weighbridge" not in report

    def test_weighbridge_and_movement_no_traffic(self):
        wb_rows = [{"vehicle": "A", "gross": 5000, "tare": 2000}]
        mv_rows = [{"mat": "A", "qty": 100, "dir": "in"}]
        wb = weighbridge_analysis(wb_rows, "vehicle", "gross", "tare")
        mv = track_material_movement(mv_rows, "mat", "qty", "dir")
        report = format_dispatch_report(weighbridge=wb, movement=mv)
        assert "Gate Traffic" not in report
        assert "Weighbridge Analysis" in report
        assert "Material Movement" in report

    # --- report is a string ---

    def test_report_returns_string(self):
        report = format_dispatch_report()
        assert isinstance(report, str)

    # --- multi-entry formatting ---

    def test_weighbridge_multiple_entries_in_report(self):
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Sand"},
            {"vehicle": "B", "gross": 6000, "tare": 2500, "mat": "Gravel"},
        ]
        wb = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        report = format_dispatch_report(weighbridge=wb)
        assert "A:" in report
        assert "B:" in report
        assert "[Sand]" in report
        assert "[Gravel]" in report

    def test_movement_multiple_materials_in_report(self):
        rows = [
            {"mat": "Steel", "qty": 100, "dir": "in"},
            {"mat": "Cement", "qty": 200, "dir": "out"},
        ]
        mv = track_material_movement(rows, "mat", "qty", "dir")
        report = format_dispatch_report(movement=mv)
        assert "Steel" in report
        assert "Cement" in report


# ===================================================================
# Dataclass Instantiation Tests
# ===================================================================


class TestDataclasses:
    """Direct instantiation tests for dataclasses."""

    def test_period_traffic_fields(self):
        pt = PeriodTraffic(period="morning", vehicle_count=5, pct_of_total=50.0)
        assert pt.period == "morning"
        assert pt.vehicle_count == 5
        assert pt.pct_of_total == 50.0

    def test_gate_traffic_result_fields(self):
        gtr = GateTrafficResult(
            total_vehicles=10,
            periods=[],
            peak_period="AM",
            off_peak_period="PM",
            avg_per_period=5.0,
            direction_split=None,
            summary="test",
        )
        assert gtr.total_vehicles == 10
        assert gtr.peak_period == "AM"
        assert gtr.off_peak_period == "PM"
        assert gtr.avg_per_period == 5.0
        assert gtr.direction_split is None
        assert gtr.summary == "test"

    def test_weigh_entry_fields(self):
        we = WeighEntry(
            vehicle="A",
            gross_weight=5000.0,
            tare_weight=2000.0,
            net_weight=3000.0,
            material="Coal",
        )
        assert we.vehicle == "A"
        assert we.gross_weight == 5000.0
        assert we.tare_weight == 2000.0
        assert we.net_weight == 3000.0
        assert we.material == "Coal"

    def test_weighbridge_result_fields(self):
        wr = WeighbridgeResult(
            entries=[],
            total_net_weight=3000.0,
            avg_net_weight=3000.0,
            total_vehicles=1,
            by_material=None,
            summary="test",
        )
        assert wr.total_net_weight == 3000.0
        assert wr.avg_net_weight == 3000.0
        assert wr.total_vehicles == 1
        assert wr.by_material is None

    def test_material_movement_fields(self):
        mm = MaterialMovement(
            material="Steel",
            inward_qty=100.0,
            outward_qty=40.0,
            net_qty=60.0,
            movement_count=3,
        )
        assert mm.material == "Steel"
        assert mm.inward_qty == 100.0
        assert mm.outward_qty == 40.0
        assert mm.net_qty == 60.0
        assert mm.movement_count == 3

    def test_movement_result_fields(self):
        mr = MovementResult(
            materials=[],
            total_inward=100.0,
            total_outward=40.0,
            net_movement=60.0,
            summary="test",
        )
        assert mr.total_inward == 100.0
        assert mr.total_outward == 40.0
        assert mr.net_movement == 60.0

    def test_dispatch_anomaly_fields(self):
        da = DispatchAnomaly(
            vehicle="TRK-001",
            weight=9000.0,
            expected_range=(4000.0, 6000.0),
            deviation_pct=60.0,
            anomaly_type="overweight",
        )
        assert da.vehicle == "TRK-001"
        assert da.weight == 9000.0
        assert da.expected_range == (4000.0, 6000.0)
        assert da.deviation_pct == 60.0
        assert da.anomaly_type == "overweight"


# ===================================================================
# Edge Case / Integration Tests
# ===================================================================


class TestEdgeCases:
    """Edge case and integration-style tests."""

    def test_large_dataset_gate_traffic(self):
        """Gate traffic with many periods and vehicles."""
        rows = [
            {"time": f"period_{i % 10}", "vehicle": f"V{i}"}
            for i in range(100)
        ]
        result = analyze_gate_traffic(rows, "time", vehicle_column="vehicle")
        assert result is not None
        assert result.total_vehicles == 100
        assert len(result.periods) == 10
        assert result.avg_per_period == 10.0

    def test_large_dataset_weighbridge(self):
        rows = [
            {"vehicle": f"V{i}", "gross": 5000 + i * 10, "tare": 2000}
            for i in range(50)
        ]
        result = weighbridge_analysis(rows, "vehicle", "gross", "tare")
        assert result is not None
        assert result.total_vehicles == 50

    def test_format_report_end_to_end(self):
        """Full end-to-end: create all analyses and generate report."""
        gate_rows = [
            {"time": "morning", "vehicle": "A", "dir": "in"},
            {"time": "morning", "vehicle": "B", "dir": "in"},
            {"time": "afternoon", "vehicle": "C", "dir": "out"},
        ]
        wb_rows = [
            {"vehicle": "A", "gross": 10000, "tare": 4000, "mat": "Iron"},
            {"vehicle": "B", "gross": 8000, "tare": 3000, "mat": "Iron"},
            {"vehicle": "C", "gross": 6000, "tare": 2000, "mat": "Coal"},
        ]
        mv_rows = [
            {"mat": "Iron", "qty": 6000, "dir": "receipt"},
            {"mat": "Iron", "qty": 5000, "dir": "receipt"},
            {"mat": "Coal", "qty": 4000, "dir": "dispatch"},
        ]

        traffic = analyze_gate_traffic(
            gate_rows, "time", vehicle_column="vehicle", direction_column="dir"
        )
        wb = weighbridge_analysis(
            wb_rows, "vehicle", "gross", "tare", material_column="mat"
        )
        mv = track_material_movement(mv_rows, "mat", "qty", "dir")

        report = format_dispatch_report(traffic=traffic, weighbridge=wb, movement=mv)

        # Verify all sections
        assert "Dispatch & Gate Report" in report
        assert "Gate Traffic" in report
        assert "Weighbridge Analysis" in report
        assert "Material Movement" in report

        # Verify specific data present
        assert "morning" in report
        assert "afternoon" in report
        assert "Iron" in report
        assert "Coal" in report
        assert "Direction:" in report

    def test_anomaly_detection_end_to_end(self):
        """Full anomaly detection scenario."""
        rows = [
            {"vehicle": "Normal1", "weight": 5000},
            {"vehicle": "Normal2", "weight": 5100},
            {"vehicle": "Normal3", "weight": 4900},
            {"vehicle": "Heavy", "weight": 12000},
            {"vehicle": "Light", "weight": 500},
            {"vehicle": "Empty", "weight": 0},
            {"vehicle": "Negative", "weight": -100},
        ]
        anomalies = detect_dispatch_anomalies(
            rows, "vehicle", "weight", expected_min=4000, expected_max=6000
        )
        vehicles = {a.vehicle for a in anomalies}
        assert "Normal1" not in vehicles
        assert "Normal2" not in vehicles
        assert "Normal3" not in vehicles
        assert "Heavy" in vehicles
        assert "Light" in vehicles
        assert "Empty" in vehicles
        assert "Negative" in vehicles

        # Verify types
        type_map = {a.vehicle: a.anomaly_type for a in anomalies}
        assert type_map["Heavy"] == "overweight"
        assert type_map["Light"] == "underweight"
        assert type_map["Empty"] == "suspicious"
        assert type_map["Negative"] == "suspicious"

    def test_weighbridge_by_material_accumulates_correctly(self):
        """Multiple entries for the same material accumulate net weight."""
        rows = [
            {"vehicle": "A", "gross": 5000, "tare": 2000, "mat": "Sand"},  # 3000
            {"vehicle": "B", "gross": 7000, "tare": 3000, "mat": "Sand"},  # 4000
            {"vehicle": "C", "gross": 6000, "tare": 2500, "mat": "Sand"},  # 3500
        ]
        result = weighbridge_analysis(
            rows, "vehicle", "gross", "tare", material_column="mat"
        )
        assert result is not None
        assert result.by_material is not None
        assert result.by_material["Sand"] == pytest.approx(10500.0, abs=0.01)

    def test_material_movement_accumulates_same_material(self):
        """Multiple inward/outward for same material accumulate properly."""
        rows = [
            {"mat": "Steel", "qty": 100, "dir": "in"},
            {"mat": "Steel", "qty": 200, "dir": "incoming"},
            {"mat": "Steel", "qty": 50, "dir": "out"},
            {"mat": "Steel", "qty": 30, "dir": "dispatch"},
        ]
        result = track_material_movement(rows, "mat", "qty", "dir")
        assert result is not None
        steel = result.materials[0]
        assert steel.inward_qty == 300.0
        assert steel.outward_qty == 80.0
        assert steel.net_qty == 220.0
        assert steel.movement_count == 4
