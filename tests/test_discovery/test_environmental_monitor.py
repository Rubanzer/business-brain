"""Tests for the environmental_monitor discovery module."""

from __future__ import annotations

import pytest

from business_brain.discovery.environmental_monitor import (
    ComplianceScore,
    EmissionsResult,
    Exceedance,
    SourcePollutantSummary,
    WasteResult,
    WasteTypeSummary,
    WaterResult,
    WaterSourceSummary,
    analyze_emissions,
    analyze_waste_generation,
    analyze_water_usage,
    compute_compliance_score,
    format_environmental_report,
    _safe_float,
)


# ===================================================================
# Helpers -- sample data builders
# ===================================================================


def _emissions_rows() -> list[dict]:
    """Sample emissions data: 2 sources, 2 pollutants, with limits."""
    return [
        {"source": "StackA", "pollutant": "SO2", "value": 40, "limit": 50, "time": "2024-01"},
        {"source": "StackA", "pollutant": "SO2", "value": 55, "limit": 50, "time": "2024-02"},
        {"source": "StackA", "pollutant": "SO2", "value": 45, "limit": 50, "time": "2024-03"},
        {"source": "StackA", "pollutant": "SO2", "value": 48, "limit": 50, "time": "2024-04"},
        {"source": "StackA", "pollutant": "NOx", "value": 30, "limit": 40, "time": "2024-01"},
        {"source": "StackA", "pollutant": "NOx", "value": 35, "limit": 40, "time": "2024-02"},
        {"source": "StackB", "pollutant": "SO2", "value": 60, "limit": 50, "time": "2024-01"},
        {"source": "StackB", "pollutant": "SO2", "value": 42, "limit": 50, "time": "2024-02"},
        {"source": "StackB", "pollutant": "PM",  "value": 10, "limit": 15, "time": "2024-01"},
        {"source": "StackB", "pollutant": "PM",  "value": 12, "limit": 15, "time": "2024-02"},
    ]


def _emissions_rows_no_limit() -> list[dict]:
    """Emissions data without limit column."""
    return [
        {"source": "StackA", "pollutant": "SO2", "value": 40},
        {"source": "StackA", "pollutant": "SO2", "value": 55},
        {"source": "StackB", "pollutant": "NOx", "value": 30},
    ]


def _waste_rows() -> list[dict]:
    """Sample waste data with disposal methods."""
    return [
        {"type": "hazardous", "quantity": 100, "disposal": "Incinerated"},
        {"type": "hazardous", "quantity": 50,  "disposal": "Treated"},
        {"type": "non-hazardous", "quantity": 300, "disposal": "Landfill"},
        {"type": "non-hazardous", "quantity": 200, "disposal": "Recycled"},
        {"type": "non-hazardous", "quantity": 150, "disposal": "Recycled"},
        {"type": "e-waste", "quantity": 20,  "disposal": "Recycled"},
        {"type": "e-waste", "quantity": 10,  "disposal": "Landfill"},
    ]


def _waste_rows_no_disposal() -> list[dict]:
    """Waste data without disposal column."""
    return [
        {"type": "hazardous", "quantity": 100},
        {"type": "non-hazardous", "quantity": 400},
    ]


def _water_rows() -> list[dict]:
    """Sample water usage data with discharge."""
    return [
        {"source": "borewell", "consumption": 500, "discharge": 100},
        {"source": "borewell", "consumption": 600, "discharge": 120},
        {"source": "municipal", "consumption": 300, "discharge": 80},
        {"source": "recycled", "consumption": 200, "discharge": 0},
        {"source": "recycled", "consumption": 150, "discharge": 0},
    ]


def _water_rows_no_discharge() -> list[dict]:
    """Water data without discharge column."""
    return [
        {"source": "borewell", "consumption": 500},
        {"source": "municipal", "consumption": 300},
        {"source": "recycled", "consumption": 100},
    ]


# ===================================================================
# _safe_float
# ===================================================================


class TestSafeFloat:
    def test_none_returns_none(self):
        assert _safe_float(None) is None

    def test_valid_int(self):
        assert _safe_float(42) == 42.0

    def test_valid_float(self):
        assert _safe_float(3.14) == 3.14

    def test_valid_string(self):
        assert _safe_float("100.5") == 100.5

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_list_returns_none(self):
        assert _safe_float([1, 2]) is None


# ===================================================================
# 1. analyze_emissions
# ===================================================================


class TestAnalyzeEmissions:
    def test_empty_rows_returns_none(self):
        result = analyze_emissions([], "source", "pollutant", "value")
        assert result is None

    def test_all_none_values_returns_none(self):
        rows = [{"source": None, "pollutant": None, "value": None}]
        result = analyze_emissions(rows, "source", "pollutant", "value")
        assert result is None

    def test_basic_grouping(self):
        rows = _emissions_rows_no_limit()
        result = analyze_emissions(rows, "source", "pollutant", "value")
        assert result is not None
        assert isinstance(result, EmissionsResult)
        # 2 groups: StackA/SO2 and StackB/NOx
        assert len(result.source_summaries) == 2

    def test_total_and_avg(self):
        rows = _emissions_rows_no_limit()
        result = analyze_emissions(rows, "source", "pollutant", "value")
        sa_so2 = next(s for s in result.source_summaries
                      if s.source == "StackA" and s.pollutant == "SO2")
        assert sa_so2.total == pytest.approx(95.0, abs=0.01)
        assert sa_so2.avg == pytest.approx(47.5, abs=0.01)
        assert sa_so2.max_value == 55.0
        assert sa_so2.count == 2

    def test_with_limits_compliance_pct(self):
        rows = _emissions_rows()
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   limit_column="limit")
        assert result is not None
        # StackA/SO2: 4 readings, 1 exceedance (55 > 50), compliance = 75%
        sa_so2 = next(s for s in result.source_summaries
                      if s.source == "StackA" and s.pollutant == "SO2")
        assert sa_so2.compliance_pct == 75.0

    def test_exceedances_detected(self):
        rows = _emissions_rows()
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   limit_column="limit")
        assert len(result.exceedances) > 0
        # StackA/SO2: 55 > 50, StackB/SO2: 60 > 50
        sources = {(e.source, e.pollutant) for e in result.exceedances}
        assert ("StackA", "SO2") in sources
        assert ("StackB", "SO2") in sources

    def test_exceedance_excess_pct(self):
        rows = _emissions_rows()
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   limit_column="limit")
        # StackB/SO2: value=60, limit=50 -> excess=20%
        exc_b = next(e for e in result.exceedances
                     if e.source == "StackB" and e.value == 60.0)
        assert exc_b.excess_pct == pytest.approx(20.0, abs=0.01)

    def test_exceedances_sorted_by_excess_pct_desc(self):
        rows = _emissions_rows()
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   limit_column="limit")
        pcts = [e.excess_pct for e in result.exceedances]
        assert pcts == sorted(pcts, reverse=True)

    def test_overall_compliance_pct(self):
        rows = _emissions_rows()
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   limit_column="limit")
        # 10 readings total, 2 exceedances -> 80% compliance
        assert result.overall_compliance_pct == 80.0

    def test_no_limit_column_compliance_none(self):
        rows = _emissions_rows_no_limit()
        result = analyze_emissions(rows, "source", "pollutant", "value")
        for s in result.source_summaries:
            assert s.compliance_pct is None
        assert result.exceedances == []

    def test_trend_detection_with_time(self):
        # Create data where second half is much higher (increasing trend)
        rows = [
            {"source": "S", "pollutant": "P", "value": 10, "time": "2024-01"},
            {"source": "S", "pollutant": "P", "value": 12, "time": "2024-02"},
            {"source": "S", "pollutant": "P", "value": 11, "time": "2024-03"},
            {"source": "S", "pollutant": "P", "value": 30, "time": "2024-04"},
            {"source": "S", "pollutant": "P", "value": 35, "time": "2024-05"},
            {"source": "S", "pollutant": "P", "value": 40, "time": "2024-06"},
        ]
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   time_column="time")
        sp = result.source_summaries[0]
        assert sp.trend == "increasing"

    def test_trend_decreasing(self):
        rows = [
            {"source": "S", "pollutant": "P", "value": 40, "time": "2024-01"},
            {"source": "S", "pollutant": "P", "value": 38, "time": "2024-02"},
            {"source": "S", "pollutant": "P", "value": 35, "time": "2024-03"},
            {"source": "S", "pollutant": "P", "value": 10, "time": "2024-04"},
            {"source": "S", "pollutant": "P", "value": 8, "time": "2024-05"},
            {"source": "S", "pollutant": "P", "value": 5, "time": "2024-06"},
        ]
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   time_column="time")
        assert result.source_summaries[0].trend == "decreasing"

    def test_trend_stable(self):
        rows = [
            {"source": "S", "pollutant": "P", "value": 20, "time": "2024-01"},
            {"source": "S", "pollutant": "P", "value": 21, "time": "2024-02"},
            {"source": "S", "pollutant": "P", "value": 19, "time": "2024-03"},
            {"source": "S", "pollutant": "P", "value": 20, "time": "2024-04"},
            {"source": "S", "pollutant": "P", "value": 21, "time": "2024-05"},
            {"source": "S", "pollutant": "P", "value": 20, "time": "2024-06"},
        ]
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   time_column="time")
        assert result.source_summaries[0].trend == "stable"

    def test_no_time_column_trend_none(self):
        rows = _emissions_rows_no_limit()
        result = analyze_emissions(rows, "source", "pollutant", "value")
        for s in result.source_summaries:
            assert s.trend is None

    def test_non_numeric_values_skipped(self):
        rows = [
            {"source": "S", "pollutant": "P", "value": "bad"},
            {"source": "S", "pollutant": "P", "value": 10},
        ]
        result = analyze_emissions(rows, "source", "pollutant", "value")
        assert result is not None
        assert result.source_summaries[0].count == 1

    def test_summary_text(self):
        rows = _emissions_rows()
        result = analyze_emissions(rows, "source", "pollutant", "value",
                                   limit_column="limit")
        assert "Emissions analysis" in result.summary
        assert "compliance" in result.summary.lower()

    def test_single_reading(self):
        rows = [{"source": "S", "pollutant": "P", "value": 42}]
        result = analyze_emissions(rows, "source", "pollutant", "value")
        assert result is not None
        assert result.source_summaries[0].total == 42.0
        assert result.source_summaries[0].avg == 42.0
        assert result.source_summaries[0].max_value == 42.0
        assert result.source_summaries[0].count == 1


# ===================================================================
# 2. analyze_waste_generation
# ===================================================================


class TestAnalyzeWasteGeneration:
    def test_empty_rows_returns_none(self):
        assert analyze_waste_generation([], "type", "quantity") is None

    def test_all_none_returns_none(self):
        rows = [{"type": None, "quantity": None}]
        assert analyze_waste_generation(rows, "type", "quantity") is None

    def test_basic_totals_by_type(self):
        rows = _waste_rows()
        result = analyze_waste_generation(rows, "type", "quantity",
                                          disposal_column="disposal")
        assert result is not None
        assert isinstance(result, WasteResult)
        types = {wt.waste_type: wt.total_quantity for wt in result.by_type}
        assert types["hazardous"] == pytest.approx(150.0, abs=0.01)
        assert types["non-hazardous"] == pytest.approx(650.0, abs=0.01)
        assert types["e-waste"] == pytest.approx(30.0, abs=0.01)

    def test_total_waste(self):
        rows = _waste_rows()
        result = analyze_waste_generation(rows, "type", "quantity",
                                          disposal_column="disposal")
        assert result.total_waste == pytest.approx(830.0, abs=0.01)

    def test_disposal_breakdown(self):
        rows = _waste_rows()
        result = analyze_waste_generation(rows, "type", "quantity",
                                          disposal_column="disposal")
        nh = next(wt for wt in result.by_type if wt.waste_type == "non-hazardous")
        assert "landfill" in nh.disposal_breakdown
        assert "recycled" in nh.disposal_breakdown
        assert nh.disposal_breakdown["landfill"] == pytest.approx(300.0, abs=0.01)
        assert nh.disposal_breakdown["recycled"] == pytest.approx(350.0, abs=0.01)

    def test_recycling_rate(self):
        rows = _waste_rows()
        result = analyze_waste_generation(rows, "type", "quantity",
                                          disposal_column="disposal")
        # Recycled: non-hazardous 350 + e-waste 20 = 370
        # Total: 830
        expected = 370 / 830 * 100
        assert result.recycling_rate == pytest.approx(expected, abs=0.1)

    def test_diversion_rate(self):
        rows = _waste_rows()
        result = analyze_waste_generation(rows, "type", "quantity",
                                          disposal_column="disposal")
        # Landfill: non-hazardous 300 + e-waste 10 = 310
        # Diversion = (830 - 310) / 830 * 100
        expected = (830 - 310) / 830 * 100
        assert result.diversion_rate == pytest.approx(expected, abs=0.1)

    def test_no_disposal_column(self):
        rows = _waste_rows_no_disposal()
        result = analyze_waste_generation(rows, "type", "quantity")
        assert result is not None
        # No disposal info -> recycling rate = 0, diversion rate = 100
        assert result.recycling_rate == 0.0
        assert result.diversion_rate == 100.0

    def test_zero_total_returns_none(self):
        rows = [{"type": "A", "quantity": 0}]
        result = analyze_waste_generation(rows, "type", "quantity")
        assert result is None

    def test_non_numeric_skipped(self):
        rows = [
            {"type": "A", "quantity": "bad"},
            {"type": "A", "quantity": 100},
        ]
        result = analyze_waste_generation(rows, "type", "quantity")
        assert result is not None
        assert result.total_waste == 100.0

    def test_summary_text(self):
        rows = _waste_rows()
        result = analyze_waste_generation(rows, "type", "quantity",
                                          disposal_column="disposal")
        assert "Waste analysis" in result.summary
        assert "Recycling rate" in result.summary
        assert "Diversion rate" in result.summary

    def test_sorted_by_type_name(self):
        rows = _waste_rows()
        result = analyze_waste_generation(rows, "type", "quantity",
                                          disposal_column="disposal")
        names = [wt.waste_type for wt in result.by_type]
        assert names == sorted(names)


# ===================================================================
# 3. analyze_water_usage
# ===================================================================


class TestAnalyzeWaterUsage:
    def test_empty_rows_returns_none(self):
        assert analyze_water_usage([], "source", "consumption") is None

    def test_all_none_returns_none(self):
        rows = [{"source": None, "consumption": None}]
        assert analyze_water_usage(rows, "source", "consumption") is None

    def test_basic_consumption_by_source(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        assert result is not None
        assert isinstance(result, WaterResult)
        src_map = {ws.source: ws.total_consumption for ws in result.by_source}
        assert src_map["borewell"] == pytest.approx(1100.0, abs=0.01)
        assert src_map["municipal"] == pytest.approx(300.0, abs=0.01)
        assert src_map["recycled"] == pytest.approx(350.0, abs=0.01)

    def test_total_consumption(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        assert result.total_consumption == pytest.approx(1750.0, abs=0.01)

    def test_discharge_totals(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        assert result.total_discharge is not None
        assert result.total_discharge == pytest.approx(300.0, abs=0.01)

    def test_water_balance(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        assert result.overall_balance is not None
        assert result.overall_balance == pytest.approx(1450.0, abs=0.01)

    def test_per_source_balance(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        borewell = next(ws for ws in result.by_source if ws.source == "borewell")
        assert borewell.water_balance is not None
        assert borewell.water_balance == pytest.approx(880.0, abs=0.01)

    def test_recycling_ratio(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        # recycled: 350, total: 1750 -> 20%
        assert result.recycling_ratio == pytest.approx(20.0, abs=0.01)

    def test_no_discharge_column(self):
        rows = _water_rows_no_discharge()
        result = analyze_water_usage(rows, "source", "consumption")
        assert result is not None
        assert result.total_discharge is None
        assert result.overall_balance is None
        for ws in result.by_source:
            assert ws.total_discharge is None
            assert ws.water_balance is None

    def test_no_recycled_source(self):
        rows = [
            {"source": "borewell", "consumption": 500},
            {"source": "municipal", "consumption": 300},
        ]
        result = analyze_water_usage(rows, "source", "consumption")
        assert result.recycling_ratio == 0.0

    def test_zero_consumption_returns_none(self):
        rows = [{"source": "borewell", "consumption": 0}]
        result = analyze_water_usage(rows, "source", "consumption")
        assert result is None

    def test_non_numeric_skipped(self):
        rows = [
            {"source": "borewell", "consumption": "bad"},
            {"source": "borewell", "consumption": 100},
        ]
        result = analyze_water_usage(rows, "source", "consumption")
        assert result is not None
        assert result.total_consumption == 100.0

    def test_summary_text(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        assert "Water usage analysis" in result.summary
        assert "Recycling ratio" in result.summary

    def test_sorted_by_source_name(self):
        rows = _water_rows()
        result = analyze_water_usage(rows, "source", "consumption",
                                     discharge_column="discharge")
        names = [ws.source for ws in result.by_source]
        assert names == sorted(names)


# ===================================================================
# 4. compute_compliance_score
# ===================================================================


class TestComputeComplianceScore:
    def test_all_none_defaults(self):
        result = compute_compliance_score()
        assert isinstance(result, ComplianceScore)
        # All default to 50 -> overall = 50*0.4 + 50*0.3 + 50*0.3 = 50
        assert result.overall_score == 50.0
        assert result.rating == "Poor"

    def test_excellent_score(self):
        emissions = _make_emissions_result(compliance_pct=98.0)
        waste = _make_waste_result(diversion_rate=95.0)
        water = _make_water_result(recycling_ratio=90.0)
        result = compute_compliance_score(emissions, waste, water)
        # 98*0.4 + 95*0.3 + 90*0.3 = 39.2 + 28.5 + 27.0 = 94.7
        assert result.overall_score == pytest.approx(94.7, abs=0.01)
        assert result.rating == "Excellent"

    def test_good_score(self):
        emissions = _make_emissions_result(compliance_pct=85.0)
        waste = _make_waste_result(diversion_rate=75.0)
        water = _make_water_result(recycling_ratio=70.0)
        result = compute_compliance_score(emissions, waste, water)
        # 85*0.4 + 75*0.3 + 70*0.3 = 34 + 22.5 + 21 = 77.5
        assert result.overall_score == pytest.approx(77.5, abs=0.01)
        assert result.rating == "Good"

    def test_fair_score(self):
        emissions = _make_emissions_result(compliance_pct=70.0)
        waste = _make_waste_result(diversion_rate=60.0)
        water = _make_water_result(recycling_ratio=50.0)
        result = compute_compliance_score(emissions, waste, water)
        # 70*0.4 + 60*0.3 + 50*0.3 = 28 + 18 + 15 = 61
        assert result.overall_score == pytest.approx(61.0, abs=0.01)
        assert result.rating == "Fair"

    def test_poor_score(self):
        emissions = _make_emissions_result(compliance_pct=40.0)
        waste = _make_waste_result(diversion_rate=30.0)
        water = _make_water_result(recycling_ratio=20.0)
        result = compute_compliance_score(emissions, waste, water)
        # 40*0.4 + 30*0.3 + 20*0.3 = 16 + 9 + 6 = 31
        assert result.overall_score == pytest.approx(31.0, abs=0.01)
        assert result.rating == "Poor"

    def test_only_emissions(self):
        emissions = _make_emissions_result(compliance_pct=100.0)
        result = compute_compliance_score(emissions=emissions)
        # 100*0.4 + 50*0.3 + 50*0.3 = 40 + 15 + 15 = 70
        assert result.overall_score == pytest.approx(70.0, abs=0.01)

    def test_only_waste(self):
        waste = _make_waste_result(diversion_rate=100.0)
        result = compute_compliance_score(waste=waste)
        # 50*0.4 + 100*0.3 + 50*0.3 = 20 + 30 + 15 = 65
        assert result.overall_score == pytest.approx(65.0, abs=0.01)

    def test_only_water(self):
        water = _make_water_result(recycling_ratio=100.0)
        result = compute_compliance_score(water=water)
        # 50*0.4 + 50*0.3 + 100*0.3 = 20 + 15 + 30 = 65
        assert result.overall_score == pytest.approx(65.0, abs=0.01)

    def test_score_clamped_to_100(self):
        # Even with all 100, should not exceed 100
        emissions = _make_emissions_result(compliance_pct=100.0)
        waste = _make_waste_result(diversion_rate=100.0)
        water = _make_water_result(recycling_ratio=100.0)
        result = compute_compliance_score(emissions, waste, water)
        assert result.overall_score <= 100.0

    def test_summary_text(self):
        result = compute_compliance_score()
        assert "Environmental compliance score" in result.summary
        assert "Emissions" in result.summary
        assert "Waste" in result.summary
        assert "Water" in result.summary

    def test_rating_boundary_90(self):
        # Exactly 90 should be Excellent
        emissions = _make_emissions_result(compliance_pct=90.0)
        waste = _make_waste_result(diversion_rate=90.0)
        water = _make_water_result(recycling_ratio=90.0)
        result = compute_compliance_score(emissions, waste, water)
        assert result.overall_score == 90.0
        assert result.rating == "Excellent"

    def test_rating_boundary_75(self):
        emissions = _make_emissions_result(compliance_pct=75.0)
        waste = _make_waste_result(diversion_rate=75.0)
        water = _make_water_result(recycling_ratio=75.0)
        result = compute_compliance_score(emissions, waste, water)
        assert result.overall_score == 75.0
        assert result.rating == "Good"

    def test_rating_boundary_60(self):
        emissions = _make_emissions_result(compliance_pct=60.0)
        waste = _make_waste_result(diversion_rate=60.0)
        water = _make_water_result(recycling_ratio=60.0)
        result = compute_compliance_score(emissions, waste, water)
        assert result.overall_score == 60.0
        assert result.rating == "Fair"


# ===================================================================
# 5. format_environmental_report
# ===================================================================


class TestFormatEnvironmentalReport:
    def test_empty_report(self):
        report = format_environmental_report()
        assert "ENVIRONMENTAL MONITORING REPORT" in report
        assert "No analysis data provided." in report

    def test_report_header_always_present(self):
        report = format_environmental_report()
        assert "=" * 60 in report

    def test_report_ends_with_separator(self):
        report = format_environmental_report()
        assert report.strip().endswith("=" * 60)

    def test_emissions_section(self):
        rows = _emissions_rows()
        em = analyze_emissions(rows, "source", "pollutant", "value",
                               limit_column="limit")
        report = format_environmental_report(emissions=em)
        assert "EMISSIONS ANALYSIS" in report
        assert "Overall Compliance" in report

    def test_waste_section(self):
        rows = _waste_rows()
        w = analyze_waste_generation(rows, "type", "quantity",
                                     disposal_column="disposal")
        report = format_environmental_report(waste=w)
        assert "WASTE GENERATION" in report
        assert "Recycling Rate" in report
        assert "Diversion Rate" in report

    def test_water_section(self):
        rows = _water_rows()
        wat = analyze_water_usage(rows, "source", "consumption",
                                  discharge_column="discharge")
        report = format_environmental_report(water=wat)
        assert "WATER USAGE" in report
        assert "Recycling Ratio" in report

    def test_compliance_score_section(self):
        score = compute_compliance_score()
        report = format_environmental_report(score=score)
        assert "COMPLIANCE SCORE" in report
        assert "Overall Score" in report
        assert "Rating" in report

    def test_combined_all_sections(self):
        em = analyze_emissions(_emissions_rows(), "source", "pollutant",
                               "value", limit_column="limit")
        w = analyze_waste_generation(_waste_rows(), "type", "quantity",
                                     disposal_column="disposal")
        wat = analyze_water_usage(_water_rows(), "source", "consumption",
                                  discharge_column="discharge")
        score = compute_compliance_score(em, w, wat)
        report = format_environmental_report(em, w, wat, score)
        assert "EMISSIONS ANALYSIS" in report
        assert "WASTE GENERATION" in report
        assert "WATER USAGE" in report
        assert "COMPLIANCE SCORE" in report

    def test_exceedances_shown_in_report(self):
        rows = _emissions_rows()
        em = analyze_emissions(rows, "source", "pollutant", "value",
                               limit_column="limit")
        report = format_environmental_report(emissions=em)
        assert "Exceedances" in report


# ===================================================================
# Helper factories for ComplianceScore tests
# ===================================================================


def _make_emissions_result(compliance_pct: float) -> EmissionsResult:
    """Create a minimal EmissionsResult with a given compliance percentage."""
    return EmissionsResult(
        source_summaries=[],
        exceedances=[],
        overall_compliance_pct=compliance_pct,
        summary="",
    )


def _make_waste_result(diversion_rate: float) -> WasteResult:
    """Create a minimal WasteResult with a given diversion rate."""
    return WasteResult(
        by_type=[],
        total_waste=0.0,
        recycling_rate=0.0,
        diversion_rate=diversion_rate,
        summary="",
    )


def _make_water_result(recycling_ratio: float) -> WaterResult:
    """Create a minimal WaterResult with a given recycling ratio."""
    return WaterResult(
        by_source=[],
        total_consumption=0.0,
        total_discharge=None,
        overall_balance=None,
        recycling_ratio=recycling_ratio,
        summary="",
    )
