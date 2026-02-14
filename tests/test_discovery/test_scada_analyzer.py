"""Tests for the scada_analyzer discovery module."""

from __future__ import annotations

import math

import pytest

from business_brain.discovery.scada_analyzer import (
    AlarmInfo,
    AlarmResult,
    EquipmentAlarms,
    ProcessStability,
    SensorAnomaly,
    SensorResult,
    SensorStats,
    _safe_float,
    analyze_alarm_frequency,
    analyze_sensor_readings,
    compute_process_stability,
    detect_sensor_anomalies,
    format_scada_report,
)


# ===================================================================
# Helpers -- sample data builders
# ===================================================================


def _sensor_rows() -> list[dict]:
    """Sample sensor rows: 3 sensors with several readings each."""
    return [
        {"sensor": "TempA", "value": 100, "unit": "C"},
        {"sensor": "TempA", "value": 102, "unit": "C"},
        {"sensor": "TempA", "value": 101, "unit": "C"},
        {"sensor": "TempA", "value": 99, "unit": "C"},
        {"sensor": "TempA", "value": 100, "unit": "C"},
        {"sensor": "PressB", "value": 50, "unit": "bar"},
        {"sensor": "PressB", "value": 52, "unit": "bar"},
        {"sensor": "PressB", "value": 48, "unit": "bar"},
        {"sensor": "PressB", "value": 51, "unit": "bar"},
        {"sensor": "FlowC", "value": 200, "unit": "L/min"},
        {"sensor": "FlowC", "value": 210, "unit": "L/min"},
        {"sensor": "FlowC", "value": 190, "unit": "L/min"},
    ]


def _anomaly_rows() -> list[dict]:
    """Rows with known spikes and drops for anomaly detection."""
    return [
        {"sensor": "S1", "value": 100},
        {"sensor": "S1", "value": 101},
        {"sensor": "S1", "value": 100},
        {"sensor": "S1", "value": 102},
        {"sensor": "S1", "value": 500},  # spike
        {"sensor": "S1", "value": 99},
        {"sensor": "S1", "value": 101},
    ]


def _flatline_rows() -> list[dict]:
    """Rows with a flatline sequence of 12 identical values."""
    rows = []
    for i in range(12):
        rows.append({"sensor": "S1", "value": 50})
    rows.append({"sensor": "S1", "value": 60})
    rows.append({"sensor": "S1", "value": 55})
    return rows


def _persistent_high_rows() -> list[dict]:
    """Rows with 4 consecutive high values."""
    rows = [
        {"sensor": "S1", "value": 100},
        {"sensor": "S1", "value": 101},
        {"sensor": "S1", "value": 100},
        {"sensor": "S1", "value": 200},  # high
        {"sensor": "S1", "value": 210},  # high
        {"sensor": "S1", "value": 205},  # high
        {"sensor": "S1", "value": 202},  # high
        {"sensor": "S1", "value": 100},
    ]
    return rows


def _stability_rows() -> list[dict]:
    """Rows for process stability calculation."""
    # Sensor with tight distribution (should be capable)
    rows = []
    for v in [100, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100.3, 99.7, 100]:
        rows.append({"sensor": "Tight", "value": v})
    # Sensor with wide distribution (should be incapable)
    for v in [100, 120, 80, 110, 90, 130, 70, 115, 85, 105]:
        rows.append({"sensor": "Wide", "value": v})
    return rows


def _alarm_rows() -> list[dict]:
    """Sample alarm rows."""
    return [
        {"alarm": "HighTemp", "severity": "critical", "equipment": "Boiler1"},
        {"alarm": "HighTemp", "severity": "critical", "equipment": "Boiler1"},
        {"alarm": "LowPress", "severity": "warning", "equipment": "Pump1"},
        {"alarm": "LowPress", "severity": "warning", "equipment": "Pump1"},
        {"alarm": "LowPress", "severity": "warning", "equipment": "Pump1"},
        {"alarm": "Vibration", "severity": "info", "equipment": "Motor1"},
        {"alarm": "HighTemp", "severity": "critical", "equipment": "Boiler2"},
        {"alarm": "Overflow", "severity": "warning", "equipment": "Tank1"},
        {"alarm": "HighTemp", "severity": "critical", "equipment": "Boiler1"},
        {"alarm": "LowPress", "severity": "warning", "equipment": "Pump1"},
    ]


def _chattering_alarm_rows() -> list[dict]:
    """Rows with chattering alarm (same alarm > 5 consecutive times)."""
    rows = []
    for _ in range(7):
        rows.append({"alarm": "FlickerAlarm"})
    rows.append({"alarm": "OtherAlarm"})
    rows.append({"alarm": "OtherAlarm"})
    return rows


# ===================================================================
# _safe_float
# ===================================================================


class TestSafeFloat:
    def test_integer(self):
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

    def test_negative(self):
        assert _safe_float(-7.5) == -7.5


# ===================================================================
# 1. analyze_sensor_readings
# ===================================================================


class TestAnalyzeSensorReadings:
    def test_basic(self):
        result = analyze_sensor_readings(_sensor_rows(), "sensor", "value")
        assert result is not None
        assert isinstance(result, SensorResult)
        assert len(result.sensors) == 3
        assert result.total_readings == 12

    def test_empty_rows(self):
        result = analyze_sensor_readings([], "sensor", "value")
        assert result is None

    def test_no_valid_data(self):
        rows = [{"sensor": "A", "value": "abc"}, {"sensor": "B", "value": None}]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is None

    def test_single_reading(self):
        rows = [{"sensor": "A", "value": 42}]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert len(result.sensors) == 1
        s = result.sensors[0]
        assert s.reading_count == 1
        assert s.min_val == 42.0
        assert s.max_val == 42.0
        assert s.mean == 42.0
        assert s.std == 0.0

    def test_stability_index_high(self):
        # All values very close to 100 -> high stability
        rows = [{"sensor": "A", "value": v} for v in [100, 100, 100, 100, 100]]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].stability_index == 1.0

    def test_stability_index_low(self):
        # Values very spread -> low stability
        rows = [{"sensor": "A", "value": v} for v in [1, 100, 1, 100, 1]]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].stability_index < 0.5

    def test_stability_index_capped_at_zero(self):
        # std > mean -> stability would be negative, capped at 0
        rows = [{"sensor": "A", "value": v} for v in [1, 100, 1, 100]]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].stability_index >= 0.0

    def test_stability_zero_mean(self):
        # Mean is 0 -> stability should be 0
        rows = [{"sensor": "A", "value": v} for v in [-5, 5, -5, 5]]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].stability_index == 0.0

    def test_stable_vs_unstable_count(self):
        result = analyze_sensor_readings(_sensor_rows(), "sensor", "value")
        assert result is not None
        assert result.stable_count + result.unstable_count == len(result.sensors)

    def test_unit_column(self):
        result = analyze_sensor_readings(
            _sensor_rows(), "sensor", "value", unit_column="unit"
        )
        assert result is not None
        temp_sensor = [s for s in result.sensors if s.sensor == "TempA"][0]
        assert temp_sensor.unit == "C"

    def test_multiple_units_same_sensor(self):
        rows = [
            {"sensor": "A", "value": 10, "unit": "C"},
            {"sensor": "A", "value": 20, "unit": "F"},
        ]
        result = analyze_sensor_readings(rows, "sensor", "value", unit_column="unit")
        assert result is not None
        assert "C" in result.sensors[0].unit
        assert "F" in result.sensors[0].unit

    def test_summary_content(self):
        result = analyze_sensor_readings(_sensor_rows(), "sensor", "value")
        assert result is not None
        assert "3 sensors" in result.summary
        assert "12 total readings" in result.summary

    def test_stats_accuracy(self):
        rows = [{"sensor": "A", "value": v} for v in [10, 20, 30]]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        s = result.sensors[0]
        assert s.min_val == 10.0
        assert s.max_val == 30.0
        assert s.mean == 20.0

    def test_missing_sensor_column(self):
        rows = [{"value": 10}, {"value": 20}]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is None

    def test_missing_value_column(self):
        rows = [{"sensor": "A"}, {"sensor": "B"}]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is None


# ===================================================================
# 2. detect_sensor_anomalies
# ===================================================================


class TestDetectSensorAnomalies:
    def test_empty_rows(self):
        assert detect_sensor_anomalies([], "sensor", "value") == []

    def test_no_anomalies_within_limits(self):
        rows = [{"sensor": "A", "value": v} for v in [100, 101, 99, 100]]
        result = detect_sensor_anomalies(rows, "sensor", "value", low_limit=0, high_limit=200)
        assert result == []

    def test_spike_detection_with_limits(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 250},  # spike
            {"sensor": "A", "value": 100},
        ]
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=50, high_limit=200
        )
        assert len(result) == 1
        assert result[0].anomaly_type == "spike"
        assert result[0].value == 250

    def test_drop_detection_with_limits(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 10},  # drop
            {"sensor": "A", "value": 100},
        ]
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=50, high_limit=200
        )
        assert len(result) == 1
        assert result[0].anomaly_type == "drop"
        assert result[0].value == 10

    def test_auto_limits_no_anomalies(self):
        # All values close to each other -> no anomalies
        rows = [{"sensor": "A", "value": v} for v in [100, 101, 99, 100, 102]]
        result = detect_sensor_anomalies(rows, "sensor", "value")
        assert result == []

    def test_flatline_detection(self):
        result = detect_sensor_anomalies(
            _flatline_rows(), "sensor", "value", low_limit=0, high_limit=200
        )
        flatline_anomalies = [a for a in result if a.anomaly_type == "flatline"]
        assert len(flatline_anomalies) == 12

    def test_flatline_not_triggered_below_10(self):
        rows = [{"sensor": "A", "value": 50} for _ in range(9)]
        rows.append({"sensor": "A", "value": 60})
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=0, high_limit=200
        )
        flatline_anomalies = [a for a in result if a.anomaly_type == "flatline"]
        assert len(flatline_anomalies) == 0

    def test_persistent_high(self):
        result = detect_sensor_anomalies(
            _persistent_high_rows(), "sensor", "value", low_limit=50, high_limit=150
        )
        persistent = [a for a in result if a.anomaly_type == "persistent_high"]
        assert len(persistent) == 4

    def test_persistent_low(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 10},   # low
            {"sensor": "A", "value": 15},   # low
            {"sensor": "A", "value": 12},   # low
            {"sensor": "A", "value": 100},
        ]
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=50, high_limit=200
        )
        persistent = [a for a in result if a.anomaly_type == "persistent_low"]
        assert len(persistent) == 3

    def test_expected_range_in_anomaly(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 300},  # spike
        ]
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=0, high_limit=200
        )
        assert len(result) == 1
        assert result[0].expected_range == (0.0, 200.0)

    def test_multiple_sensors(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 300},  # spike for A
            {"sensor": "B", "value": 50},
            {"sensor": "B", "value": -10},  # drop for B
        ]
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=0, high_limit=200
        )
        sensors_with_anomalies = {a.sensor for a in result}
        assert "A" in sensors_with_anomalies
        assert "B" in sensors_with_anomalies

    def test_index_is_correct(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 500},  # spike at row index 2
        ]
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=0, high_limit=200
        )
        assert len(result) == 1
        assert result[0].index == 2

    def test_no_valid_data(self):
        rows = [{"sensor": "A", "value": "bad"}]
        result = detect_sensor_anomalies(rows, "sensor", "value")
        assert result == []

    def test_single_value_auto_limits_skipped(self):
        # Single value -> fewer than 2 -> auto limits can't compute, skip
        rows = [{"sensor": "A", "value": 100}]
        result = detect_sensor_anomalies(rows, "sensor", "value")
        assert result == []

    def test_flatline_at_end(self):
        rows = [{"sensor": "A", "value": 60}]
        for _ in range(10):
            rows.append({"sensor": "A", "value": 50})
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=0, high_limit=200
        )
        flatline_anomalies = [a for a in result if a.anomaly_type == "flatline"]
        assert len(flatline_anomalies) == 10


# ===================================================================
# 3. compute_process_stability
# ===================================================================


class TestComputeProcessStability:
    def test_empty_rows(self):
        assert compute_process_stability([], "sensor", "value") == []

    def test_basic_result(self):
        result = compute_process_stability(_stability_rows(), "sensor", "value")
        assert len(result) == 2

    def test_sorted_by_cpk_ascending(self):
        result = compute_process_stability(_stability_rows(), "sensor", "value")
        assert result[0].cpk <= result[1].cpk

    def test_tight_process_with_auto_limits(self):
        # Auto-derived limits always give Cpk = 1.0 -> "Marginal"
        result = compute_process_stability(_stability_rows(), "sensor", "value")
        tight = [r for r in result if r.sensor == "Tight"][0]
        assert tight.rating == "Marginal"
        assert abs(tight.cpk - 1.0) < 0.01

    def test_capable_with_explicit_limits(self):
        # Provide explicit wide spec limits so the tight sensor is Capable
        rows = [
            {"sensor": "A", "value": v, "target": "yes", "usl": 101.5, "lsl": 98.5}
            for v in [100, 100.5, 99.5, 100.2, 99.8, 100.1, 99.9, 100.3, 99.7, 100]
        ]
        result = compute_process_stability(rows, "sensor", "value", target_column="target")
        assert len(result) == 1
        assert result[0].rating == "Capable"
        assert result[0].cpk >= 1.33

    def test_auto_derived_limits(self):
        # With auto-derived limits (mean +/- 3*std), Cp = 1.0 by definition
        result = compute_process_stability(_stability_rows(), "sensor", "value")
        for r in result:
            assert abs(r.cp - 1.0) < 0.01

    def test_custom_spec_limits(self):
        rows = [
            {"sensor": "A", "value": 100, "target": "yes", "usl": 110, "lsl": 90},
            {"sensor": "A", "value": 102, "target": "yes", "usl": 110, "lsl": 90},
            {"sensor": "A", "value": 98, "target": "yes", "usl": 110, "lsl": 90},
            {"sensor": "A", "value": 101, "target": "yes", "usl": 110, "lsl": 90},
            {"sensor": "A", "value": 99, "target": "yes", "usl": 110, "lsl": 90},
        ]
        result = compute_process_stability(rows, "sensor", "value", target_column="target")
        assert len(result) == 1
        assert result[0].usl == 110.0
        assert result[0].lsl == 90.0

    def test_rating_marginal(self):
        # Create data where Cpk is between 1.0 and 1.33
        # std ~ 0.67, USL=102.5, LSL=97.5 -> Cpk ~ 1.24
        rows = [
            {"sensor": "A", "value": v, "target": "y", "usl": 102.5, "lsl": 97.5}
            for v in [99, 100, 101, 99.5, 100.5, 100, 100, 99, 101, 100]
        ]
        result = compute_process_stability(rows, "sensor", "value", target_column="target")
        assert len(result) == 1
        assert result[0].rating == "Marginal"
        assert 1.0 <= result[0].cpk < 1.33

    def test_rating_incapable(self):
        rows = [
            {"sensor": "A", "value": v, "target": "y", "usl": 102, "lsl": 98}
            for v in [90, 110, 95, 105, 100, 85, 115, 92, 108, 100]
        ]
        result = compute_process_stability(rows, "sensor", "value", target_column="target")
        assert len(result) == 1
        assert result[0].rating == "Incapable"

    def test_zero_std(self):
        # All identical values
        rows = [{"sensor": "A", "value": 100} for _ in range(5)]
        result = compute_process_stability(rows, "sensor", "value")
        assert len(result) == 1
        assert result[0].cp == 999.99
        assert result[0].cpk == 999.99
        assert result[0].rating == "Capable"

    def test_single_reading_skipped(self):
        rows = [{"sensor": "A", "value": 100}]
        result = compute_process_stability(rows, "sensor", "value")
        assert result == []

    def test_no_valid_values(self):
        rows = [{"sensor": "A", "value": "bad"}, {"sensor": "A", "value": None}]
        result = compute_process_stability(rows, "sensor", "value")
        assert result == []

    def test_multiple_sensors_sorted(self):
        result = compute_process_stability(_stability_rows(), "sensor", "value")
        assert len(result) == 2
        # Wide should have lower Cpk, so it should come first
        assert result[0].sensor == "Wide" or result[0].cpk <= result[1].cpk

    def test_cp_formula(self):
        # Manually verify Cp = (USL - LSL) / (6 * sigma)
        rows = [{"sensor": "X", "value": v} for v in [10, 20, 30, 40, 50]]
        result = compute_process_stability(rows, "sensor", "value")
        assert len(result) == 1
        r = result[0]
        # Auto-derived: USL = mean + 3*std, LSL = mean - 3*std
        # Cp = (6*std) / (6*std) = 1.0
        assert abs(r.cp - 1.0) < 0.01


# ===================================================================
# 4. analyze_alarm_frequency
# ===================================================================


class TestAnalyzeAlarmFrequency:
    def test_empty_rows(self):
        result = analyze_alarm_frequency([], "alarm")
        assert result is None

    def test_no_valid_alarms(self):
        rows = [{"other": "data"}]
        result = analyze_alarm_frequency(rows, "alarm")
        assert result is None

    def test_basic(self):
        result = analyze_alarm_frequency(
            _alarm_rows(), "alarm",
            severity_column="severity",
            equipment_column="equipment",
        )
        assert result is not None
        assert result.total_alarms == 10

    def test_severity_counts(self):
        result = analyze_alarm_frequency(
            _alarm_rows(), "alarm", severity_column="severity"
        )
        assert result is not None
        assert result.by_severity["critical"] == 4
        assert result.by_severity["warning"] == 5
        assert result.by_severity["info"] == 1

    def test_top_alarms(self):
        result = analyze_alarm_frequency(_alarm_rows(), "alarm")
        assert result is not None
        # HighTemp appears 4 times, LowPress 4 times
        top_names = [a.alarm for a in result.top_alarms]
        assert "HighTemp" in top_names
        assert "LowPress" in top_names

    def test_top_alarms_max_5(self):
        rows = []
        for i in range(10):
            rows.append({"alarm": f"Alarm_{i}"})
        result = analyze_alarm_frequency(rows, "alarm")
        assert result is not None
        assert len(result.top_alarms) <= 5

    def test_equipment_breakdown(self):
        result = analyze_alarm_frequency(
            _alarm_rows(), "alarm",
            severity_column="severity",
            equipment_column="equipment",
        )
        assert result is not None
        equip_names = [e.equipment for e in result.by_equipment]
        assert "Boiler1" in equip_names

    def test_equipment_critical_count(self):
        result = analyze_alarm_frequency(
            _alarm_rows(), "alarm",
            severity_column="severity",
            equipment_column="equipment",
        )
        assert result is not None
        boiler1 = [e for e in result.by_equipment if e.equipment == "Boiler1"][0]
        assert boiler1.critical_count == 3

    def test_chattering_detection(self):
        result = analyze_alarm_frequency(_chattering_alarm_rows(), "alarm")
        assert result is not None
        assert "FlickerAlarm" in result.chattering_alarms

    def test_no_chattering_below_threshold(self):
        rows = []
        for _ in range(5):
            rows.append({"alarm": "SomeAlarm"})
        rows.append({"alarm": "Other"})
        result = analyze_alarm_frequency(rows, "alarm")
        assert result is not None
        assert result.chattering_alarms == []

    def test_pct_of_total(self):
        rows = [
            {"alarm": "A"},
            {"alarm": "A"},
            {"alarm": "B"},
            {"alarm": "B"},
        ]
        result = analyze_alarm_frequency(rows, "alarm")
        assert result is not None
        for alarm_info in result.top_alarms:
            assert alarm_info.pct_of_total == 50.0

    def test_summary_content(self):
        result = analyze_alarm_frequency(_alarm_rows(), "alarm")
        assert result is not None
        assert "10 total alarms" in result.summary

    def test_single_alarm(self):
        rows = [{"alarm": "X"}]
        result = analyze_alarm_frequency(rows, "alarm")
        assert result is not None
        assert result.total_alarms == 1
        assert result.top_alarms[0].alarm == "X"
        assert result.top_alarms[0].pct_of_total == 100.0

    def test_chattering_at_end(self):
        rows = [{"alarm": "Other"}]
        for _ in range(6):
            rows.append({"alarm": "EndChatter"})
        result = analyze_alarm_frequency(rows, "alarm")
        assert result is not None
        assert "EndChatter" in result.chattering_alarms

    def test_no_severity_column(self):
        result = analyze_alarm_frequency(_alarm_rows(), "alarm")
        assert result is not None
        assert result.by_severity == {}

    def test_no_equipment_column(self):
        result = analyze_alarm_frequency(_alarm_rows(), "alarm")
        assert result is not None
        assert result.by_equipment == []


# ===================================================================
# 5. format_scada_report
# ===================================================================


class TestFormatScadaReport:
    def test_no_data(self):
        report = format_scada_report()
        assert "No analysis data provided." in report

    def test_header_present(self):
        report = format_scada_report()
        assert "SCADA / INDUSTRIAL SENSOR REPORT" in report

    def test_sensor_section(self):
        result = analyze_sensor_readings(_sensor_rows(), "sensor", "value")
        report = format_scada_report(sensor_result=result)
        assert "SENSOR READINGS" in report
        assert "TempA" in report

    def test_anomaly_section(self):
        anomalies = detect_sensor_anomalies(
            _anomaly_rows(), "sensor", "value", low_limit=50, high_limit=150
        )
        report = format_scada_report(anomalies=anomalies)
        assert "ANOMALIES" in report

    def test_stability_section(self):
        stability = compute_process_stability(_stability_rows(), "sensor", "value")
        report = format_scada_report(stability=stability)
        assert "PROCESS STABILITY" in report

    def test_alarm_section(self):
        alarms = analyze_alarm_frequency(_alarm_rows(), "alarm", severity_column="severity")
        report = format_scada_report(alarms=alarms)
        assert "ALARM ANALYSIS" in report

    def test_combined_report(self):
        sr = analyze_sensor_readings(_sensor_rows(), "sensor", "value")
        anomalies = detect_sensor_anomalies(
            _anomaly_rows(), "sensor", "value", low_limit=50, high_limit=150
        )
        stab = compute_process_stability(_stability_rows(), "sensor", "value")
        alarms = analyze_alarm_frequency(_alarm_rows(), "alarm")
        report = format_scada_report(
            sensor_result=sr,
            anomalies=anomalies,
            stability=stab,
            alarms=alarms,
        )
        assert "SENSOR READINGS" in report
        assert "ANOMALIES" in report
        assert "PROCESS STABILITY" in report
        assert "ALARM ANALYSIS" in report

    def test_empty_anomaly_list(self):
        report = format_scada_report(anomalies=[])
        assert "ANOMALIES" in report
        assert "Total Anomalies: 0" in report

    def test_empty_stability_list(self):
        report = format_scada_report(stability=[])
        assert "PROCESS STABILITY" in report
        assert "No stability data." in report


# ===================================================================
# Edge cases and integration tests
# ===================================================================


class TestEdgeCases:
    def test_all_identical_sensor_values_stability(self):
        rows = [{"sensor": "X", "value": 50} for _ in range(20)]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].std == 0.0
        assert result.sensors[0].stability_index == 1.0

    def test_all_identical_values_anomaly_auto_limits(self):
        # All identical -> std=0 -> limits collapse -> no anomalies outside
        rows = [{"sensor": "X", "value": 50} for _ in range(5)]
        result = detect_sensor_anomalies(rows, "sensor", "value")
        # With pstdev = 0, limits are [50, 50]. All values = 50.
        # Not > 50 and not < 50 so no out-of-range anomalies.
        # But with 5 values all the same, less than 10 -> no flatline.
        assert len(result) == 0

    def test_negative_values(self):
        rows = [{"sensor": "A", "value": v} for v in [-10, -20, -15, -12]]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].mean < 0

    def test_very_large_values(self):
        rows = [{"sensor": "A", "value": v} for v in [1e9, 1e9 + 1, 1e9 - 1]]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].stability_index > 0.99

    def test_mixed_valid_invalid_rows(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": "bad"},
            {"sensor": "A", "value": None},
            {"sensor": "A", "value": 102},
        ]
        result = analyze_sensor_readings(rows, "sensor", "value")
        assert result is not None
        assert result.sensors[0].reading_count == 2

    def test_flatline_exactly_10(self):
        rows = [{"sensor": "A", "value": 50} for _ in range(10)]
        result = detect_sensor_anomalies(
            rows, "sensor", "value", low_limit=0, high_limit=200
        )
        flatline_anomalies = [a for a in result if a.anomaly_type == "flatline"]
        assert len(flatline_anomalies) == 10

    def test_process_stability_with_two_values(self):
        rows = [
            {"sensor": "A", "value": 100},
            {"sensor": "A", "value": 200},
        ]
        result = compute_process_stability(rows, "sensor", "value")
        assert len(result) == 1
        assert result[0].cp == 1.0

    def test_alarm_all_same(self):
        rows = [{"alarm": "X"} for _ in range(10)]
        result = analyze_alarm_frequency(rows, "alarm")
        assert result is not None
        assert result.total_alarms == 10
        assert result.top_alarms[0].count == 10
        assert result.top_alarms[0].pct_of_total == 100.0
        assert "X" in result.chattering_alarms

    def test_sensor_readings_with_timestamp(self):
        rows = [
            {"sensor": "A", "value": 100, "ts": "2024-01-01 00:00"},
            {"sensor": "A", "value": 101, "ts": "2024-01-01 01:00"},
            {"sensor": "A", "value": 102, "ts": "2024-01-01 02:00"},
        ]
        result = analyze_sensor_readings(
            rows, "sensor", "value", timestamp_column="ts"
        )
        assert result is not None
        assert result.sensors[0].reading_count == 3
