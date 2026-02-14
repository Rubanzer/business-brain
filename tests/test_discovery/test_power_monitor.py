"""Tests for the power_monitor discovery module."""

from __future__ import annotations

import math

import pytest

from business_brain.discovery.power_monitor import (
    DemandPeak,
    EnergyCostResult,
    EntityPF,
    EntitySEC,
    LoadProfileResult,
    PeriodLoad,
    PowerFactorResult,
    SpecificEnergyResult,
    analyze_load_profile,
    analyze_power_factor,
    compute_specific_energy,
    detect_demand_peaks,
    energy_cost_analysis,
    format_power_report,
)


# ===================================================================
# Helpers — sample data builders
# ===================================================================


def _load_rows() -> list[dict]:
    """Sample load profile rows: 6 time periods with varying demand."""
    return [
        {"time": "00:00-04:00", "power": 200},
        {"time": "04:00-08:00", "power": 450},
        {"time": "08:00-12:00", "power": 900},
        {"time": "12:00-16:00", "power": 1000},
        {"time": "16:00-20:00", "power": 850},
        {"time": "20:00-24:00", "power": 350},
    ]


def _pf_rows() -> list[dict]:
    """Sample power factor rows for 3 entities."""
    return [
        {"entity": "PlantA", "kw": 800, "kva": 820},   # PF ~0.976
        {"entity": "PlantA", "kw": 850, "kva": 870},   # PF ~0.977
        {"entity": "PlantB", "kw": 600, "kva": 750},   # PF = 0.800
        {"entity": "PlantB", "kw": 580, "kva": 700},   # PF ~0.829
        {"entity": "PlantC", "kw": 700, "kva": 760},   # PF ~0.921
        {"entity": "PlantC", "kw": 720, "kva": 780},   # PF ~0.923
    ]


def _sec_rows() -> list[dict]:
    """Sample specific energy rows for 3 entities."""
    return [
        {"entity": "LineA", "energy": 5000, "output": 1000},  # SEC = 5.0
        {"entity": "LineA", "energy": 5200, "output": 1100},  # accumulated
        {"entity": "LineB", "energy": 6000, "output": 800},   # SEC = 7.5
        {"entity": "LineB", "energy": 5800, "output": 750},   # accumulated
        {"entity": "LineC", "energy": 4500, "output": 1000},  # SEC = 4.5
        {"entity": "LineC", "energy": 4800, "output": 1200},  # accumulated
    ]


def _peak_rows() -> list[dict]:
    """Time series rows for peak demand detection."""
    return [
        {"time": "T1", "power": 100},
        {"time": "T2", "power": 200},
        {"time": "T3", "power": 800},
        {"time": "T4", "power": 950},
        {"time": "T5", "power": 1000},
        {"time": "T6", "power": 920},
        {"time": "T7", "power": 500},
        {"time": "T8", "power": 300},
        {"time": "T9", "power": 970},
        {"time": "T10", "power": 200},
    ]


def _cost_rows() -> list[dict]:
    """Energy cost rows with per-period energy values."""
    return [
        {"time": "T1", "energy": 500},
        {"time": "T2", "energy": 300},
        {"time": "T3", "energy": 800},
        {"time": "T4", "energy": 1000},
        {"time": "T5", "energy": 200},
        {"time": "T6", "energy": 150},
    ]


# ===================================================================
# 1. analyze_load_profile
# ===================================================================


class TestAnalyzeLoadProfile:
    def test_empty_rows(self):
        result = analyze_load_profile([], "time", "power")
        assert isinstance(result, LoadProfileResult)
        assert result.periods == []
        assert result.peak_demand == 0.0
        assert result.load_factor == 0.0
        assert "No load data" in result.summary

    def test_basic_load_profile(self):
        rows = _load_rows()
        result = analyze_load_profile(rows, "time", "power")
        assert result.peak_demand == 1000.0
        assert result.min_demand == 200.0
        assert result.peak_period == "12:00-16:00"
        assert result.off_peak_period == "00:00-04:00"
        assert len(result.periods) == 6

    def test_load_factor_calculation(self):
        rows = _load_rows()
        result = analyze_load_profile(rows, "time", "power")
        expected_avg = (200 + 450 + 900 + 1000 + 850 + 350) / 6
        expected_lf = expected_avg / 1000 * 100
        assert abs(result.load_factor - round(expected_lf, 2)) < 0.01

    def test_period_classification(self):
        rows = _load_rows()
        result = analyze_load_profile(rows, "time", "power")
        classifications = {p.period: p.classification for p in result.periods}
        # 1000 = peak (100%), 900 = peak (90%), 850 = peak (85%)
        assert classifications["12:00-16:00"] == "peak"
        assert classifications["08:00-12:00"] == "peak"
        assert classifications["16:00-20:00"] == "peak"
        # 450 = shoulder (45%), 350 = off_peak (35%), 200 = off_peak (20%)
        assert classifications["00:00-04:00"] == "off_peak"
        assert classifications["20:00-24:00"] == "off_peak"
        assert classifications["04:00-08:00"] == "off_peak"

    def test_single_period(self):
        rows = [{"time": "T1", "power": 500}]
        result = analyze_load_profile(rows, "time", "power")
        assert result.peak_demand == 500.0
        assert result.avg_demand == 500.0
        assert result.load_factor == 100.0
        assert len(result.periods) == 1
        assert result.periods[0].classification == "peak"

    def test_missing_values_skipped(self):
        rows = [
            {"time": "T1", "power": 100},
            {"time": "T2", "power": None},
            {"time": None, "power": 300},
            {"time": "T4", "power": "bad"},
            {"time": "T5", "power": 500},
        ]
        result = analyze_load_profile(rows, "time", "power")
        assert len(result.periods) == 2
        assert result.peak_demand == 500.0

    def test_aggregation_across_entities(self):
        rows = [
            {"time": "T1", "power": 100, "entity": "A"},
            {"time": "T1", "power": 200, "entity": "B"},
            {"time": "T2", "power": 300, "entity": "A"},
        ]
        result = analyze_load_profile(rows, "time", "power", entity_column="entity")
        # T1 = 100 + 200 = 300, T2 = 300
        assert len(result.periods) == 2
        period_map = {p.period: p.demand for p in result.periods}
        assert period_map["T1"] == 300.0
        assert period_map["T2"] == 300.0

    def test_summary_contains_key_info(self):
        result = analyze_load_profile(_load_rows(), "time", "power")
        assert "6 periods" in result.summary
        assert "Load factor" in result.summary


# ===================================================================
# 2. analyze_power_factor
# ===================================================================


class TestAnalyzePowerFactor:
    def test_empty_rows(self):
        result = analyze_power_factor([], "entity", "kw", "kva")
        assert isinstance(result, PowerFactorResult)
        assert result.entities == []
        assert result.mean_pf == 0.0
        assert "No power factor data" in result.summary

    def test_basic_power_factor(self):
        rows = _pf_rows()
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        assert len(result.entities) == 3
        # PlantA: (800+850)/(820+870) = 1650/1690 ~ 0.9763
        plant_a = next(e for e in result.entities if e.entity == "PlantA")
        assert plant_a.power_factor > 0.95
        assert plant_a.status == "excellent"

    def test_poor_power_factor_flagged(self):
        rows = _pf_rows()
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        plant_b = next(e for e in result.entities if e.entity == "PlantB")
        # PlantB: (600+580)/(750+700) = 1180/1450 ~ 0.8138
        assert plant_b.power_factor < 0.9
        assert plant_b.status == "poor"
        assert result.penalty_risk_count >= 1

    def test_good_power_factor(self):
        rows = _pf_rows()
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        plant_c = next(e for e in result.entities if e.entity == "PlantC")
        # PlantC: (700+720)/(760+780) = 1420/1540 ~ 0.922
        assert 0.9 <= plant_c.power_factor <= 0.95
        assert plant_c.status == "good"

    def test_excellent_count(self):
        rows = _pf_rows()
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        assert result.excellent_count >= 1  # PlantA

    def test_estimated_loss_pct(self):
        rows = _pf_rows()
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        # Lower PF = higher loss
        plant_a = next(e for e in result.entities if e.entity == "PlantA")
        plant_b = next(e for e in result.entities if e.entity == "PlantB")
        assert plant_b.estimated_loss_pct > plant_a.estimated_loss_pct

    def test_entities_sorted_by_pf_descending(self):
        rows = _pf_rows()
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        pfs = [e.power_factor for e in result.entities]
        assert pfs == sorted(pfs, reverse=True)

    def test_pf_clamped_to_one(self):
        # Edge case: kw > kva (shouldn't happen but test clamping)
        rows = [{"entity": "X", "kw": 1000, "kva": 900}]
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        assert result.entities[0].power_factor <= 1.0

    def test_zero_kva(self):
        rows = [{"entity": "X", "kw": 100, "kva": 0}]
        result = analyze_power_factor(rows, "entity", "kw", "kva")
        assert result.entities[0].power_factor == 0.0
        assert result.entities[0].status == "poor"


# ===================================================================
# 3. compute_specific_energy
# ===================================================================


class TestComputeSpecificEnergy:
    def test_empty_rows(self):
        result = compute_specific_energy([], "entity", "energy", "output")
        assert isinstance(result, SpecificEnergyResult)
        assert result.entities == []
        assert result.mean_sec == 0.0
        assert "No specific energy data" in result.summary

    def test_basic_sec(self):
        rows = _sec_rows()
        result = compute_specific_energy(rows, "entity", "energy", "output")
        assert len(result.entities) == 3
        # LineC: (4500+4800)/(1000+1200) = 9300/2200 ~ 4.227
        line_c = next(e for e in result.entities if e.entity == "LineC")
        assert line_c.sec == pytest.approx(9300 / 2200, abs=0.01)

    def test_best_and_worst_entity(self):
        rows = _sec_rows()
        result = compute_specific_energy(rows, "entity", "energy", "output")
        # LineC has lowest SEC, LineB has highest
        assert result.best_entity == "LineC"
        assert result.worst_entity == "LineB"

    def test_deviation_from_best(self):
        rows = _sec_rows()
        result = compute_specific_energy(rows, "entity", "energy", "output")
        best = next(e for e in result.entities if e.entity == result.best_entity)
        assert best.deviation_from_best_pct == 0.0

        worst = next(e for e in result.entities if e.entity == result.worst_entity)
        assert worst.deviation_from_best_pct > 0

    def test_potential_savings(self):
        rows = _sec_rows()
        result = compute_specific_energy(rows, "entity", "energy", "output")
        assert result.potential_savings > 0

    def test_zero_output_entity(self):
        rows = [
            {"entity": "A", "energy": 1000, "output": 0},
            {"entity": "B", "energy": 500, "output": 100},
        ]
        result = compute_specific_energy(rows, "entity", "energy", "output")
        a = next(e for e in result.entities if e.entity == "A")
        assert a.sec == float("inf")
        # Best entity should be B (the only finite one)
        assert result.best_entity == "B"

    def test_entities_sorted_by_sec_ascending(self):
        rows = _sec_rows()
        result = compute_specific_energy(rows, "entity", "energy", "output")
        secs = [e.sec for e in result.entities]
        assert secs == sorted(secs)


# ===================================================================
# 4. detect_demand_peaks
# ===================================================================


class TestDetectDemandPeaks:
    def test_empty_rows(self):
        result = detect_demand_peaks([], "time", "power")
        assert result == []

    def test_basic_peak_detection(self):
        rows = _peak_rows()
        result = detect_demand_peaks(rows, "time", "power", threshold_pct=90)
        # max = 1000, threshold = 900
        # T3=800 no, T4=950 yes, T5=1000 yes, T6=920 yes -> group of 3
        # T9=970 yes -> group of 1
        assert len(result) >= 1
        top_peak = result[0]
        assert top_peak.demand == 1000.0

    def test_consecutive_peaks_grouped(self):
        rows = _peak_rows()
        result = detect_demand_peaks(rows, "time", "power", threshold_pct=90)
        # T4, T5, T6 form a consecutive run of 3
        group = next(p for p in result if p.duration_periods == 3)
        assert group.duration_periods == 3

    def test_threshold_100_only_max(self):
        rows = _peak_rows()
        result = detect_demand_peaks(rows, "time", "power", threshold_pct=100)
        # Only the exact max (1000) should qualify
        assert len(result) == 1
        assert result[0].demand == 1000.0

    def test_threshold_0_all_periods(self):
        rows = [
            {"time": "T1", "power": 100},
            {"time": "T2", "power": 200},
            {"time": "T3", "power": 300},
        ]
        result = detect_demand_peaks(rows, "time", "power", threshold_pct=0)
        # threshold = 0, so all rows >= 0 qualify; all consecutive => 1 group
        assert len(result) == 1
        assert result[0].duration_periods == 3

    def test_sorted_by_demand_descending(self):
        rows = _peak_rows()
        result = detect_demand_peaks(rows, "time", "power", threshold_pct=90)
        demands = [p.demand for p in result]
        assert demands == sorted(demands, reverse=True)

    def test_all_zeros(self):
        rows = [
            {"time": "T1", "power": 0},
            {"time": "T2", "power": 0},
        ]
        result = detect_demand_peaks(rows, "time", "power")
        assert result == []


# ===================================================================
# 5. energy_cost_analysis
# ===================================================================


class TestEnergyCostAnalysis:
    def test_empty_rows(self):
        result = energy_cost_analysis([], "time", "energy")
        assert isinstance(result, EnergyCostResult)
        assert result.total_energy == 0.0
        assert result.total_cost == 0.0
        assert "No energy cost data" in result.summary

    def test_with_peak_offpeak_rates(self):
        rows = _cost_rows()
        result = energy_cost_analysis(
            rows, "time", "energy",
            peak_rate=0.15, offpeak_rate=0.08,
        )
        assert result.total_energy == pytest.approx(500 + 300 + 800 + 1000 + 200 + 150, abs=0.01)
        assert result.total_cost > 0
        assert result.peak_energy > 0
        assert result.offpeak_energy > 0
        assert result.peak_cost > result.offpeak_cost  # peak energy is pricier

    def test_potential_shift_savings(self):
        rows = _cost_rows()
        result = energy_cost_analysis(
            rows, "time", "energy",
            peak_rate=0.20, offpeak_rate=0.05,
        )
        # Savings = peak_energy * (peak_rate - offpeak_rate)
        assert result.potential_shift_savings > 0

    def test_with_rate_column(self):
        rows = [
            {"time": "T1", "energy": 100, "rate": 0.10},
            {"time": "T2", "energy": 200, "rate": 0.15},
        ]
        result = energy_cost_analysis(rows, "time", "energy", rate_column="rate")
        assert result.total_cost == pytest.approx(100 * 0.10 + 200 * 0.15, abs=0.01)

    def test_avg_rate(self):
        rows = [
            {"time": "T1", "energy": 100, "rate": 0.10},
            {"time": "T2", "energy": 100, "rate": 0.20},
        ]
        result = energy_cost_analysis(rows, "time", "energy", rate_column="rate")
        # total_cost = 10 + 20 = 30, total_energy = 200, avg_rate = 0.15
        assert result.avg_rate == pytest.approx(0.15, abs=0.001)

    def test_no_rates_zero_cost(self):
        rows = _cost_rows()
        result = energy_cost_analysis(rows, "time", "energy")
        assert result.total_cost == 0.0
        assert result.avg_rate == 0.0

    def test_summary_contains_total_energy(self):
        rows = _cost_rows()
        result = energy_cost_analysis(
            rows, "time", "energy", peak_rate=0.10, offpeak_rate=0.05,
        )
        assert "Total energy" in result.summary


# ===================================================================
# 6. format_power_report
# ===================================================================


class TestFormatPowerReport:
    def test_empty_report(self):
        report = format_power_report()
        assert "POWER & ENERGY MONITORING REPORT" in report
        assert "No analysis data provided." in report

    def test_with_load_only(self):
        load = analyze_load_profile(_load_rows(), "time", "power")
        report = format_power_report(load=load)
        assert "LOAD PROFILE" in report
        assert "Peak Demand" in report
        assert "Load Factor" in report

    def test_with_pf_only(self):
        pf = analyze_power_factor(_pf_rows(), "entity", "kw", "kva")
        report = format_power_report(pf=pf)
        assert "POWER FACTOR ANALYSIS" in report
        assert "Mean PF" in report

    def test_with_sec_only(self):
        sec = compute_specific_energy(_sec_rows(), "entity", "energy", "output")
        report = format_power_report(sec=sec)
        assert "SPECIFIC ENERGY CONSUMPTION" in report
        assert "Mean SEC" in report

    def test_with_cost_only(self):
        cost = energy_cost_analysis(
            _cost_rows(), "time", "energy", peak_rate=0.10, offpeak_rate=0.05,
        )
        report = format_power_report(cost=cost)
        assert "ENERGY COST ANALYSIS" in report
        assert "Total Energy" in report

    def test_combined_report(self):
        load = analyze_load_profile(_load_rows(), "time", "power")
        pf = analyze_power_factor(_pf_rows(), "entity", "kw", "kva")
        sec = compute_specific_energy(_sec_rows(), "entity", "energy", "output")
        cost = energy_cost_analysis(
            _cost_rows(), "time", "energy", peak_rate=0.10, offpeak_rate=0.05,
        )
        report = format_power_report(load=load, pf=pf, sec=sec, cost=cost)
        assert "LOAD PROFILE" in report
        assert "POWER FACTOR ANALYSIS" in report
        assert "SPECIFIC ENERGY CONSUMPTION" in report
        assert "ENERGY COST ANALYSIS" in report

    def test_report_ends_with_separator(self):
        report = format_power_report()
        assert report.strip().endswith("=" * 60)
