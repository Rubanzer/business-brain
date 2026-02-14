"""Tests for safety and compliance analytics module."""

from business_brain.discovery.safety_compliance import (
    ComplianceResult,
    EntityCompliance,
    EntitySafety,
    IncidentResult,
    RiskItem,
    RiskMatrixResult,
    SafetyScoreResult,
    analyze_incidents,
    compliance_rate,
    compute_safety_score,
    format_safety_report,
    risk_matrix,
)


# ===================================================================
# Incident Analysis Tests
# ===================================================================


class TestAnalyzeIncidents:
    def test_basic_incident_counts(self):
        rows = [
            {"type": "slip", "severity": "low"},
            {"type": "slip", "severity": "medium"},
            {"type": "fire", "severity": "high"},
        ]
        result = analyze_incidents(rows, "type", "severity")
        assert result.total_incidents == 3
        assert result.by_type == {"slip": 2, "fire": 1}
        assert result.by_severity == {"low": 1, "medium": 1, "high": 1}

    def test_empty_rows_returns_zero(self):
        result = analyze_incidents([], "type", "severity")
        assert result.total_incidents == 0
        assert result.by_type == {}
        assert result.by_severity == {}
        assert result.trend == "stable"
        assert result.most_common_type == ""
        assert "No incident data" in result.summary

    def test_all_none_values(self):
        rows = [{"type": None, "severity": None}]
        result = analyze_incidents(rows, "type", "severity")
        assert result.total_incidents == 0

    def test_most_common_type(self):
        rows = [
            {"type": "slip", "severity": "low"},
            {"type": "slip", "severity": "low"},
            {"type": "fire", "severity": "high"},
            {"type": "chemical", "severity": "medium"},
        ]
        result = analyze_incidents(rows, "type", "severity")
        assert result.most_common_type == "slip"

    def test_location_tracking(self):
        rows = [
            {"type": "slip", "severity": "low", "loc": "PlantA"},
            {"type": "fire", "severity": "high", "loc": "PlantA"},
            {"type": "slip", "severity": "medium", "loc": "PlantB"},
        ]
        result = analyze_incidents(rows, "type", "severity", location_column="loc")
        assert result.by_location is not None
        assert result.by_location == {"PlantA": 2, "PlantB": 1}
        assert result.most_common_location == "PlantA"

    def test_no_location_column_returns_none(self):
        rows = [{"type": "slip", "severity": "low"}]
        result = analyze_incidents(rows, "type", "severity")
        assert result.by_location is None
        assert result.most_common_location is None

    def test_trend_stable_without_dates(self):
        rows = [{"type": "slip", "severity": "low"}] * 10
        result = analyze_incidents(rows, "type", "severity")
        assert result.trend == "stable"

    def test_trend_with_dates(self):
        # First half: 2 incidents, second half: 8 incidents -> increasing
        rows = [
            {"type": "slip", "severity": "low", "date": "2024-01-01"},
            {"type": "slip", "severity": "low", "date": "2024-01-02"},
            {"type": "slip", "severity": "low", "date": "2024-06-01"},
            {"type": "slip", "severity": "low", "date": "2024-06-02"},
            {"type": "slip", "severity": "low", "date": "2024-06-03"},
            {"type": "slip", "severity": "low", "date": "2024-06-04"},
        ]
        result = analyze_incidents(rows, "type", "severity", date_column="date")
        # With 6 rows, mid = 3, first half = 3, second half = 3, ratio = 1.0 -> stable
        assert result.trend == "stable"

    def test_summary_contains_key_info(self):
        rows = [
            {"type": "slip", "severity": "low"},
            {"type": "fire", "severity": "high"},
        ]
        result = analyze_incidents(rows, "type", "severity")
        assert "2 incident(s)" in result.summary
        assert "slip" in result.summary


# ===================================================================
# Safety Score Tests
# ===================================================================


class TestComputeSafetyScore:
    def test_basic_safety_score(self):
        rows = [
            {"plant": "A", "incidents": 1, "days": 365},
            {"plant": "B", "incidents": 10, "days": 365},
        ]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        assert result is not None
        assert len(result.entities) == 2
        # Plant A: rate = 1/365 ~0.00274, score = 100 - (0.00274 * 500) ~98.63
        a = [e for e in result.entities if e.entity == "A"][0]
        assert a.safety_score > 95
        assert a.grade == "A"
        # Plant B: rate = 10/365 ~0.0274, score = 100 - (0.0274 * 500) ~86.3
        b = [e for e in result.entities if e.entity == "B"][0]
        assert 80 < b.safety_score < 95

    def test_empty_returns_none(self):
        assert compute_safety_score([], "plant", "incidents", "days") is None

    def test_all_none_returns_none(self):
        rows = [{"plant": None, "incidents": None, "days": None}]
        assert compute_safety_score(rows, "plant", "incidents", "days") is None

    def test_zero_days_gives_zero_score(self):
        rows = [{"plant": "A", "incidents": 5, "days": 0}]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        assert result is not None
        assert result.entities[0].safety_score == 0.0
        assert result.entities[0].grade == "F"

    def test_safest_and_riskiest(self):
        rows = [
            {"plant": "Safe", "incidents": 0, "days": 365},
            {"plant": "Risky", "incidents": 50, "days": 365},
        ]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        assert result.safest_entity == "Safe"
        assert result.riskiest_entity == "Risky"

    def test_sorted_by_score_descending(self):
        rows = [
            {"plant": "Bad", "incidents": 20, "days": 100},
            {"plant": "Good", "incidents": 1, "days": 100},
            {"plant": "Mid", "incidents": 5, "days": 100},
        ]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        scores = [e.safety_score for e in result.entities]
        assert scores == sorted(scores, reverse=True)

    def test_grade_boundaries(self):
        # Score clamped to [0, 100]. rate = incidents / days, score = 100 - rate * 500
        # A >= 90: need rate <= 0.02 -> incidents/days <= 0.02
        # B >= 75: need rate <= 0.05
        # C >= 60: need rate <= 0.08
        # D >= 40: need rate <= 0.12
        # F < 40: rate > 0.12
        rows = [
            {"plant": "gradeA", "incidents": 1, "days": 100},   # rate=0.01, score=95
            {"plant": "gradeB", "incidents": 4, "days": 100},   # rate=0.04, score=80
            {"plant": "gradeC", "incidents": 7, "days": 100},   # rate=0.07, score=65
            {"plant": "gradeD", "incidents": 11, "days": 100},  # rate=0.11, score=45
            {"plant": "gradeF", "incidents": 20, "days": 100},  # rate=0.20, score=0
        ]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        grades = {e.entity: e.grade for e in result.entities}
        assert grades["gradeA"] == "A"
        assert grades["gradeB"] == "B"
        assert grades["gradeC"] == "C"
        assert grades["gradeD"] == "D"
        assert grades["gradeF"] == "F"

    def test_score_clamped_at_zero(self):
        # Very high incident rate should clamp to 0, not go negative
        rows = [{"plant": "Bad", "incidents": 100, "days": 10}]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        assert result.entities[0].safety_score == 0.0

    def test_score_clamped_at_hundred(self):
        rows = [{"plant": "Perfect", "incidents": 0, "days": 365}]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        assert result.entities[0].safety_score == 100.0
        assert result.entities[0].grade == "A"

    def test_aggregates_multiple_rows(self):
        rows = [
            {"plant": "A", "incidents": 2, "days": 100},
            {"plant": "A", "incidents": 3, "days": 100},
        ]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        e = result.entities[0]
        assert e.incident_count == 5
        assert e.days_tracked == 200

    def test_summary_text(self):
        rows = [
            {"plant": "A", "incidents": 1, "days": 365},
            {"plant": "B", "incidents": 5, "days": 365},
        ]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        assert "Safety scores" in result.summary
        assert "Mean score" in result.summary

    def test_skips_non_numeric(self):
        rows = [
            {"plant": "A", "incidents": "bad", "days": 100},
            {"plant": "B", "incidents": 1, "days": 100},
        ]
        result = compute_safety_score(rows, "plant", "incidents", "days")
        assert result is not None
        assert len(result.entities) == 1


# ===================================================================
# Compliance Rate Tests
# ===================================================================


class TestComplianceRate:
    def test_basic_compliance(self):
        rows = [
            {"plant": "A", "total": 100, "passed": 98},
            {"plant": "B", "total": 100, "passed": 70},
        ]
        result = compliance_rate(rows, "plant", "total", "passed")
        assert result is not None
        a = [e for e in result.entities if e.entity == "A"][0]
        assert a.compliance_pct == 98.0
        assert a.status == "compliant"
        b = [e for e in result.entities if e.entity == "B"][0]
        assert b.compliance_pct == 70.0
        assert b.status == "non_compliant"

    def test_empty_returns_none(self):
        assert compliance_rate([], "plant", "total", "passed") is None

    def test_all_none_returns_none(self):
        rows = [{"plant": None, "total": None, "passed": None}]
        assert compliance_rate(rows, "plant", "total", "passed") is None

    def test_zero_total_gives_zero_pct(self):
        rows = [{"plant": "A", "total": 0, "passed": 0}]
        result = compliance_rate(rows, "plant", "total", "passed")
        assert result is not None
        assert result.entities[0].compliance_pct == 0.0
        assert result.entities[0].status == "non_compliant"

    def test_status_at_risk(self):
        rows = [{"plant": "A", "total": 100, "passed": 85}]
        result = compliance_rate(rows, "plant", "total", "passed")
        assert result.entities[0].status == "at_risk"

    def test_fully_compliant_and_non_compliant_counts(self):
        rows = [
            {"plant": "Good", "total": 100, "passed": 99},     # compliant
            {"plant": "OK", "total": 100, "passed": 90},       # at_risk
            {"plant": "Bad", "total": 100, "passed": 60},      # non_compliant
            {"plant": "Perfect", "total": 100, "passed": 100},  # compliant
        ]
        result = compliance_rate(rows, "plant", "total", "passed")
        assert result.fully_compliant_count == 2
        assert result.non_compliant_count == 1

    def test_sorted_by_compliance_descending(self):
        rows = [
            {"plant": "Low", "total": 100, "passed": 50},
            {"plant": "High", "total": 100, "passed": 99},
            {"plant": "Mid", "total": 100, "passed": 80},
        ]
        result = compliance_rate(rows, "plant", "total", "passed")
        pcts = [e.compliance_pct for e in result.entities]
        assert pcts == sorted(pcts, reverse=True)

    def test_mean_compliance(self):
        rows = [
            {"plant": "A", "total": 100, "passed": 100},
            {"plant": "B", "total": 100, "passed": 80},
        ]
        result = compliance_rate(rows, "plant", "total", "passed")
        assert result.mean_compliance == 90.0

    def test_aggregates_multiple_rows(self):
        rows = [
            {"plant": "A", "total": 50, "passed": 48},
            {"plant": "A", "total": 50, "passed": 47},
        ]
        result = compliance_rate(rows, "plant", "total", "passed")
        e = result.entities[0]
        assert e.total_checks == 100
        assert e.passed_checks == 95
        assert e.compliance_pct == 95.0

    def test_summary_text(self):
        rows = [
            {"plant": "A", "total": 100, "passed": 98},
            {"plant": "B", "total": 100, "passed": 70},
        ]
        result = compliance_rate(rows, "plant", "total", "passed")
        assert "Compliance" in result.summary
        assert "Mean" in result.summary


# ===================================================================
# Risk Matrix Tests
# ===================================================================


class TestRiskMatrix:
    def test_basic_risk_classification(self):
        rows = [
            {"likelihood": 5, "impact": 5},   # score=25, Critical
            {"likelihood": 4, "impact": 4},   # score=16, High
            {"likelihood": 2, "impact": 3},   # score=6, Medium
            {"likelihood": 1, "impact": 2},   # score=2, Low
        ]
        result = risk_matrix(rows, "likelihood", "impact")
        assert result is not None
        assert result.critical_count == 1
        assert result.high_count == 1
        assert result.medium_count == 1
        assert result.low_count == 1

    def test_empty_returns_none(self):
        assert risk_matrix([], "likelihood", "impact") is None

    def test_all_none_returns_none(self):
        rows = [{"likelihood": None, "impact": None}]
        assert risk_matrix(rows, "likelihood", "impact") is None

    def test_with_entity_column(self):
        rows = [
            {"entity": "RiskA", "likelihood": 5, "impact": 5},
            {"entity": "RiskB", "likelihood": 1, "impact": 1},
        ]
        result = risk_matrix(rows, "likelihood", "impact", entity_column="entity")
        assert result.items[0].entity == "RiskA"
        assert result.items[1].entity == "RiskB"

    def test_without_entity_column(self):
        rows = [{"likelihood": 3, "impact": 3}]
        result = risk_matrix(rows, "likelihood", "impact")
        assert result.items[0].entity is None

    def test_sorted_by_risk_score_descending(self):
        rows = [
            {"likelihood": 1, "impact": 1},
            {"likelihood": 5, "impact": 5},
            {"likelihood": 3, "impact": 3},
        ]
        result = risk_matrix(rows, "likelihood", "impact")
        scores = [i.risk_score for i in result.items]
        assert scores == sorted(scores, reverse=True)

    def test_risk_score_calculation(self):
        rows = [{"likelihood": 3, "impact": 4}]
        result = risk_matrix(rows, "likelihood", "impact")
        assert result.items[0].risk_score == 12.0
        assert result.items[0].risk_level == "High"

    def test_boundary_critical(self):
        rows = [{"likelihood": 4, "impact": 5}]  # score=20 -> Critical
        result = risk_matrix(rows, "likelihood", "impact")
        assert result.items[0].risk_level == "Critical"

    def test_boundary_high(self):
        rows = [{"likelihood": 3, "impact": 4}]  # score=12 -> High
        result = risk_matrix(rows, "likelihood", "impact")
        assert result.items[0].risk_level == "High"

    def test_boundary_medium(self):
        rows = [{"likelihood": 1, "impact": 5}]  # score=5 -> Medium
        result = risk_matrix(rows, "likelihood", "impact")
        assert result.items[0].risk_level == "Medium"

    def test_boundary_low(self):
        rows = [{"likelihood": 2, "impact": 2}]  # score=4 -> Low
        result = risk_matrix(rows, "likelihood", "impact")
        assert result.items[0].risk_level == "Low"

    def test_summary_text(self):
        rows = [
            {"likelihood": 5, "impact": 5},
            {"likelihood": 1, "impact": 1},
        ]
        result = risk_matrix(rows, "likelihood", "impact")
        assert "Risk matrix" in result.summary
        assert "Critical" in result.summary

    def test_skips_non_numeric(self):
        rows = [
            {"likelihood": "bad", "impact": 5},
            {"likelihood": 3, "impact": 3},
        ]
        result = risk_matrix(rows, "likelihood", "impact")
        assert result is not None
        assert len(result.items) == 1


# ===================================================================
# Format Safety Report Tests
# ===================================================================


class TestFormatSafetyReport:
    def test_all_none_reports_no_data(self):
        report = format_safety_report()
        assert "No analysis data provided" in report

    def test_header_always_present(self):
        report = format_safety_report()
        assert "Safety & Compliance Report" in report
        assert "=" * 40 in report

    def test_incidents_section(self):
        rows = [
            {"type": "slip", "severity": "low"},
            {"type": "fire", "severity": "high"},
        ]
        inc = analyze_incidents(rows, "type", "severity")
        report = format_safety_report(incidents=inc)
        assert "Incident Analysis" in report
        assert "Total incidents: 2" in report
        assert "slip" in report

    def test_safety_section(self):
        rows = [{"plant": "A", "incidents": 1, "days": 365}]
        safety = compute_safety_score(rows, "plant", "incidents", "days")
        report = format_safety_report(safety=safety)
        assert "Safety Scores" in report
        assert "A" in report

    def test_compliance_section(self):
        rows = [{"plant": "A", "total": 100, "passed": 98}]
        comp = compliance_rate(rows, "plant", "total", "passed")
        report = format_safety_report(compliance=comp)
        assert "Compliance Rates" in report
        assert "98.0%" in report

    def test_risk_section(self):
        rows = [{"entity": "X", "likelihood": 5, "impact": 5}]
        risk_res = risk_matrix(rows, "likelihood", "impact", entity_column="entity")
        report = format_safety_report(risk=risk_res)
        assert "Risk Matrix" in report
        assert "Critical" in report

    def test_combined_all_sections(self):
        inc_rows = [{"type": "slip", "severity": "low"}]
        safety_rows = [{"plant": "A", "incidents": 1, "days": 365}]
        comp_rows = [{"plant": "A", "total": 100, "passed": 98}]
        risk_rows = [{"entity": "X", "likelihood": 5, "impact": 5}]

        inc = analyze_incidents(inc_rows, "type", "severity")
        safety = compute_safety_score(safety_rows, "plant", "incidents", "days")
        comp = compliance_rate(comp_rows, "plant", "total", "passed")
        risk_res = risk_matrix(risk_rows, "likelihood", "impact", entity_column="entity")

        report = format_safety_report(
            incidents=inc, safety=safety, compliance=comp, risk=risk_res
        )
        assert "Incident Analysis" in report
        assert "Safety Scores" in report
        assert "Compliance Rates" in report
        assert "Risk Matrix" in report
