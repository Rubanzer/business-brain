"""Tests for compliance tracking and regulatory monitoring module."""

from datetime import datetime

from business_brain.discovery.compliance_tracker import (
    AreaFindings,
    AuditFindingsResult,
    CategoryCompliance,
    CategoryScore,
    ComplianceScoreResult,
    ComplianceStatusResult,
    DeadlineItem,
    DeadlineResult,
    MonthlyFindingTrend,
    OwnerDeadlines,
    SeverityCount,
    audit_compliance_status,
    analyze_audit_findings,
    compute_compliance_score,
    format_compliance_report,
    track_regulatory_deadlines,
)


# ===================================================================
# 1. audit_compliance_status Tests
# ===================================================================


class TestAuditComplianceStatus:
    def test_empty_rows_returns_none(self):
        assert audit_compliance_status([], "req", "status") is None

    def test_all_none_returns_none(self):
        rows = [{"req": None, "status": None}]
        assert audit_compliance_status(rows, "req", "status") is None

    def test_basic_compliance_count(self):
        rows = [
            {"req": "R1", "status": "Compliant"},
            {"req": "R2", "status": "Non-Compliant"},
            {"req": "R3", "status": "Compliant"},
            {"req": "R4", "status": "In Progress"},
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert result is not None
        assert result.total_requirements == 4
        assert result.compliant_count == 2
        assert result.non_compliant_count == 1
        assert result.compliance_rate == 50.0

    def test_case_insensitive_status(self):
        rows = [
            {"req": "R1", "status": "compliant"},
            {"req": "R2", "status": "COMPLIANT"},
            {"req": "R3", "status": "Compliant"},
            {"req": "R4", "status": "NON-COMPLIANT"},
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert result.compliant_count == 3
        assert result.non_compliant_count == 1

    def test_non_compliant_variants(self):
        rows = [
            {"req": "R1", "status": "Non-Compliant"},
            {"req": "R2", "status": "non compliant"},
            {"req": "R3", "status": "noncompliant"},
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert result.non_compliant_count == 3

    def test_compliance_rate_all_compliant(self):
        rows = [
            {"req": "R1", "status": "Compliant"},
            {"req": "R2", "status": "Compliant"},
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert result.compliance_rate == 100.0

    def test_compliance_rate_none_compliant(self):
        rows = [
            {"req": "R1", "status": "In Progress"},
            {"req": "R2", "status": "Not Started"},
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert result.compliance_rate == 0.0

    def test_single_row(self):
        rows = [{"req": "R1", "status": "Compliant"}]
        result = audit_compliance_status(rows, "req", "status")
        assert result.total_requirements == 1
        assert result.compliant_count == 1
        assert result.compliance_rate == 100.0

    def test_category_breakdown(self):
        rows = [
            {"req": "R1", "status": "Compliant", "cat": "Security"},
            {"req": "R2", "status": "Compliant", "cat": "Security"},
            {"req": "R3", "status": "Non-Compliant", "cat": "Security"},
            {"req": "R4", "status": "Compliant", "cat": "Privacy"},
            {"req": "R5", "status": "Non-Compliant", "cat": "Privacy"},
        ]
        result = audit_compliance_status(rows, "req", "status", category_column="cat")
        assert result.by_category is not None
        assert len(result.by_category) == 2
        sec = [c for c in result.by_category if c.category == "Security"][0]
        assert sec.total == 3
        assert sec.compliant == 2
        assert sec.compliance_rate == 66.67
        priv = [c for c in result.by_category if c.category == "Privacy"][0]
        assert priv.total == 2
        assert priv.compliant == 1
        assert priv.compliance_rate == 50.0

    def test_no_category_column_returns_none(self):
        rows = [{"req": "R1", "status": "Compliant"}]
        result = audit_compliance_status(rows, "req", "status")
        assert result.by_category is None

    def test_overdue_detection(self):
        rows = [
            {"req": "R1", "status": "In Progress", "due": "2024-01-01"},
            {"req": "R2", "status": "Compliant", "due": "2024-01-15"},
            {"req": "R3", "status": "Non-Compliant", "due": "2024-06-01"},
            {"req": "R4", "status": "In Progress", "due": "2024-06-15"},
        ]
        # ref_date = max date = 2024-06-15
        # R1: due 2024-01-01 < 2024-06-15, not compliant -> overdue
        # R2: compliant -> not overdue
        # R3: due 2024-06-01 < 2024-06-15, not compliant -> overdue
        # R4: due 2024-06-15 = ref_date -> NOT overdue (not strictly <)
        result = audit_compliance_status(rows, "req", "status", due_date_column="due")
        assert result.overdue_count == 2

    def test_overdue_compliant_not_counted(self):
        rows = [
            {"req": "R1", "status": "Compliant", "due": "2024-01-01"},
            {"req": "R2", "status": "Compliant", "due": "2024-12-01"},
        ]
        result = audit_compliance_status(rows, "req", "status", due_date_column="due")
        assert result.overdue_count == 0

    def test_no_due_date_column_zero_overdue(self):
        rows = [{"req": "R1", "status": "Non-Compliant"}]
        result = audit_compliance_status(rows, "req", "status")
        assert result.overdue_count == 0

    def test_summary_contains_key_info(self):
        rows = [
            {"req": "R1", "status": "Compliant"},
            {"req": "R2", "status": "Non-Compliant"},
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert "2 requirements" in result.summary
        assert "50.0%" in result.summary

    def test_with_whitespace_in_status(self):
        rows = [
            {"req": "R1", "status": " Compliant "},
            {"req": "R2", "status": " Non-Compliant "},
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert result.compliant_count == 1
        assert result.non_compliant_count == 1

    def test_mixed_valid_and_null_status(self):
        rows = [
            {"req": "R1", "status": "Compliant"},
            {"req": "R2", "status": None},
        ]
        # R2: req is not None so it is valid, status is None -> normalised to ""
        result = audit_compliance_status(rows, "req", "status")
        assert result.total_requirements == 2
        assert result.compliant_count == 1


# ===================================================================
# 2. analyze_audit_findings Tests
# ===================================================================


class TestAnalyzeAuditFindings:
    def test_empty_rows_returns_none(self):
        assert analyze_audit_findings([], "finding", "severity") is None

    def test_all_none_returns_none(self):
        rows = [{"finding": None, "severity": None}]
        assert analyze_audit_findings(rows, "finding", "severity") is None

    def test_basic_severity_counts(self):
        rows = [
            {"finding": "F1", "severity": "Critical"},
            {"finding": "F2", "severity": "Critical"},
            {"finding": "F3", "severity": "Major"},
            {"finding": "F4", "severity": "Minor"},
            {"finding": "F5", "severity": "Observation"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert result is not None
        assert result.total_findings == 5
        sev_dict = {s.severity: s.count for s in result.by_severity}
        assert sev_dict["Critical"] == 2
        assert sev_dict["Major"] == 1
        assert sev_dict["Minor"] == 1
        assert sev_dict["Observation"] == 1

    def test_severity_pct(self):
        rows = [
            {"finding": "F1", "severity": "Critical"},
            {"finding": "F2", "severity": "Minor"},
            {"finding": "F3", "severity": "Minor"},
            {"finding": "F4", "severity": "Minor"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity")
        sev_dict = {s.severity: s.pct for s in result.by_severity}
        assert sev_dict["Critical"] == 25.0
        assert sev_dict["Minor"] == 75.0

    def test_severity_case_normalisation(self):
        rows = [
            {"finding": "F1", "severity": "critical"},
            {"finding": "F2", "severity": "CRITICAL"},
            {"finding": "F3", "severity": "Critical"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert len(result.by_severity) == 1
        assert result.by_severity[0].severity == "Critical"
        assert result.by_severity[0].count == 3

    def test_severity_ordering(self):
        rows = [
            {"finding": "F1", "severity": "Observation"},
            {"finding": "F2", "severity": "Minor"},
            {"finding": "F3", "severity": "Critical"},
            {"finding": "F4", "severity": "Major"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity")
        sev_names = [s.severity for s in result.by_severity]
        assert sev_names == ["Critical", "Major", "Minor", "Observation"]

    def test_area_analysis(self):
        rows = [
            {"finding": "F1", "severity": "Critical", "area": "Finance"},
            {"finding": "F2", "severity": "Minor", "area": "Finance"},
            {"finding": "F3", "severity": "Critical", "area": "IT"},
            {"finding": "F4", "severity": "Major", "area": "HR"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity", area_column="area")
        assert result.by_area is not None
        assert len(result.by_area) == 3
        fin = [a for a in result.by_area if a.area == "Finance"][0]
        assert fin.count == 2
        assert fin.critical_count == 1
        it = [a for a in result.by_area if a.area == "IT"][0]
        assert it.count == 1
        assert it.critical_count == 1
        hr = [a for a in result.by_area if a.area == "HR"][0]
        assert hr.count == 1
        assert hr.critical_count == 0

    def test_no_area_column_returns_none(self):
        rows = [{"finding": "F1", "severity": "Critical"}]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert result.by_area is None

    def test_status_open_closed(self):
        rows = [
            {"finding": "F1", "severity": "Critical", "st": "Open"},
            {"finding": "F2", "severity": "Minor", "st": "Closed"},
            {"finding": "F3", "severity": "Major", "st": "Closed"},
            {"finding": "F4", "severity": "Minor", "st": "In Progress"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity", status_column="st")
        assert result.open_count == 2  # Open + In Progress
        assert result.closed_count == 2
        assert result.closure_rate == 50.0

    def test_status_resolved_and_pending(self):
        rows = [
            {"finding": "F1", "severity": "Critical", "st": "Resolved"},
            {"finding": "F2", "severity": "Minor", "st": "Pending"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity", status_column="st")
        assert result.closed_count == 1
        assert result.open_count == 1
        assert result.closure_rate == 50.0

    def test_no_status_column_zeros(self):
        rows = [{"finding": "F1", "severity": "Critical"}]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert result.open_count == 0
        assert result.closed_count == 0
        assert result.closure_rate is None

    def test_closure_rate_all_closed(self):
        rows = [
            {"finding": "F1", "severity": "Critical", "st": "Closed"},
            {"finding": "F2", "severity": "Minor", "st": "Closed"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity", status_column="st")
        assert result.closure_rate == 100.0

    def test_monthly_trend(self):
        rows = [
            {"finding": "F1", "severity": "Critical", "date": "2024-01-15"},
            {"finding": "F2", "severity": "Minor", "date": "2024-01-20"},
            {"finding": "F3", "severity": "Major", "date": "2024-02-10"},
            {"finding": "F4", "severity": "Minor", "date": "2024-03-05"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity", date_column="date")
        assert result.monthly_trend is not None
        assert len(result.monthly_trend) == 3
        assert result.monthly_trend[0].month == "2024-01"
        assert result.monthly_trend[0].count == 2
        assert result.monthly_trend[1].month == "2024-02"
        assert result.monthly_trend[1].count == 1
        assert result.monthly_trend[2].month == "2024-03"
        assert result.monthly_trend[2].count == 1

    def test_no_date_column_no_trend(self):
        rows = [{"finding": "F1", "severity": "Critical"}]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert result.monthly_trend is None

    def test_single_row(self):
        rows = [{"finding": "F1", "severity": "Critical"}]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert result.total_findings == 1
        assert result.by_severity[0].count == 1
        assert result.by_severity[0].pct == 100.0

    def test_summary_contains_key_info(self):
        rows = [
            {"finding": "F1", "severity": "Critical"},
            {"finding": "F2", "severity": "Minor"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert "2 audit findings" in result.summary
        assert "Critical" in result.summary

    def test_all_optional_params(self):
        rows = [
            {"finding": "F1", "severity": "Critical", "date": "2024-01-15",
             "area": "IT", "st": "Open"},
            {"finding": "F2", "severity": "Minor", "date": "2024-02-15",
             "area": "HR", "st": "Closed"},
        ]
        result = analyze_audit_findings(
            rows, "finding", "severity",
            date_column="date", area_column="area", status_column="st"
        )
        assert result.by_area is not None
        assert result.monthly_trend is not None
        assert result.closure_rate is not None

    def test_unknown_severity_included(self):
        rows = [
            {"finding": "F1", "severity": "Critical"},
            {"finding": "F2", "severity": "Informational"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity")
        sev_dict = {s.severity: s.count for s in result.by_severity}
        assert "Critical" in sev_dict
        assert "Informational" in sev_dict

    def test_area_sorted_by_count_descending(self):
        rows = [
            {"finding": "F1", "severity": "Minor", "area": "A"},
            {"finding": "F2", "severity": "Minor", "area": "B"},
            {"finding": "F3", "severity": "Minor", "area": "B"},
            {"finding": "F4", "severity": "Minor", "area": "B"},
        ]
        result = analyze_audit_findings(rows, "finding", "severity", area_column="area")
        assert result.by_area[0].area == "B"
        assert result.by_area[1].area == "A"


# ===================================================================
# 3. compute_compliance_score Tests
# ===================================================================


class TestComputeComplianceScore:
    def test_empty_rows_returns_none(self):
        assert compute_compliance_score([], "req", "weight", "score") is None

    def test_all_none_returns_none(self):
        rows = [{"req": None, "weight": None, "score": None}]
        assert compute_compliance_score(rows, "req", "weight", "score") is None

    def test_basic_weighted_score(self):
        rows = [
            {"req": "R1", "weight": 1.0, "score": 80.0},
            {"req": "R2", "weight": 1.0, "score": 60.0},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result is not None
        # (1*80 + 1*60) / (1+1) = 70
        assert result.overall_score == 70.0

    def test_weighted_score_different_weights(self):
        rows = [
            {"req": "R1", "weight": 3.0, "score": 90.0},
            {"req": "R2", "weight": 1.0, "score": 50.0},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score")
        # (3*90 + 1*50) / (3+1) = (270 + 50) / 4 = 80
        assert result.overall_score == 80.0

    def test_rating_excellent(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 95.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Excellent"

    def test_rating_good(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 80.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Good"

    def test_rating_needs_improvement(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 65.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Needs Improvement"

    def test_rating_critical(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 40.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Critical"

    def test_rating_boundary_90(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 90.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Excellent"

    def test_rating_boundary_75(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 75.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Good"

    def test_rating_boundary_60(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 60.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Needs Improvement"

    def test_rating_boundary_59(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 59.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.rating == "Critical"

    def test_weakest_areas_bottom_3(self):
        rows = [
            {"req": "R1", "weight": 1.0, "score": 90.0},
            {"req": "R2", "weight": 1.0, "score": 30.0},
            {"req": "R3", "weight": 1.0, "score": 50.0},
            {"req": "R4", "weight": 1.0, "score": 70.0},
            {"req": "R5", "weight": 1.0, "score": 10.0},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score")
        weakest_names = [w.category for w in result.weakest_areas]
        assert len(result.weakest_areas) == 3
        assert "R5" in weakest_names
        assert "R2" in weakest_names
        assert "R3" in weakest_names

    def test_weakest_areas_fewer_than_3(self):
        rows = [
            {"req": "R1", "weight": 1.0, "score": 90.0},
            {"req": "R2", "weight": 1.0, "score": 30.0},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert len(result.weakest_areas) == 2

    def test_category_breakdown(self):
        rows = [
            {"req": "R1", "weight": 2.0, "score": 80.0, "cat": "Security"},
            {"req": "R2", "weight": 1.0, "score": 60.0, "cat": "Security"},
            {"req": "R3", "weight": 1.0, "score": 90.0, "cat": "Privacy"},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score", category_column="cat")
        assert result.by_category is not None
        # Security: (2*80 + 1*60) / (2+1) = 220/3 = 73.33
        assert abs(result.by_category["Security"] - 73.33) < 0.01
        # Privacy: (1*90) / 1 = 90
        assert result.by_category["Privacy"] == 90.0

    def test_no_category_column_returns_none(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 80.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.by_category is None

    def test_single_row(self):
        rows = [{"req": "R1", "weight": 5.0, "score": 85.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.overall_score == 85.0
        assert len(result.weakest_areas) == 1

    def test_zero_weight_returns_none(self):
        rows = [{"req": "R1", "weight": 0.0, "score": 80.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result is None

    def test_skips_non_numeric(self):
        rows = [
            {"req": "R1", "weight": "bad", "score": 80.0},
            {"req": "R2", "weight": 1.0, "score": 70.0},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result is not None
        assert result.overall_score == 70.0

    def test_summary_contains_key_info(self):
        rows = [
            {"req": "R1", "weight": 1.0, "score": 85.0},
            {"req": "R2", "weight": 1.0, "score": 55.0},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert "70.0" in result.summary
        assert "Needs Improvement" in result.summary

    def test_score_100(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 100.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.overall_score == 100.0
        assert result.rating == "Excellent"

    def test_score_0(self):
        rows = [{"req": "R1", "weight": 1.0, "score": 0.0}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result.overall_score == 0.0
        assert result.rating == "Critical"


# ===================================================================
# 4. track_regulatory_deadlines Tests
# ===================================================================


class TestTrackRegulatoryDeadlines:
    def test_empty_rows_returns_none(self):
        assert track_regulatory_deadlines([], "reg", "deadline") is None

    def test_all_none_returns_none(self):
        rows = [{"reg": None, "deadline": None}]
        assert track_regulatory_deadlines(rows, "reg", "deadline") is None

    def test_basic_classification(self):
        # ref_date will be 2024-07-01 (the max date)
        rows = [
            {"reg": "GDPR", "deadline": "2024-05-01"},        # Overdue (-61 days)
            {"reg": "SOX", "deadline": "2024-07-01"},          # Due This Week (0 days)
            {"reg": "HIPAA", "deadline": "2024-07-05"},        # Due This Week (4 days)
            {"reg": "PCI", "deadline": "2024-07-20"},          # Due This Month (19 days)
            {"reg": "ISO", "deadline": "2024-09-01"},          # Upcoming (62 days)
        ]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            reference_date=datetime(2024, 7, 1),
        )
        assert result is not None
        assert result.total_items == 5
        assert result.overdue_count == 1
        assert result.due_this_week == 2
        assert result.due_this_month == 1
        assert result.upcoming_count == 1

    def test_sorted_by_deadline_ascending(self):
        rows = [
            {"reg": "C", "deadline": "2024-09-01"},
            {"reg": "A", "deadline": "2024-01-01"},
            {"reg": "B", "deadline": "2024-05-01"},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline")
        deadlines = [i.deadline for i in result.items]
        assert deadlines == sorted(deadlines)

    def test_completed_items_filtered_out(self):
        rows = [
            {"reg": "R1", "deadline": "2024-01-01", "st": "Completed"},
            {"reg": "R2", "deadline": "2024-06-01", "st": "Pending"},
            {"reg": "R3", "deadline": "2024-12-01", "st": "In Progress"},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline", status_column="st")
        assert result.total_items == 2
        regs = [i.regulation for i in result.items]
        assert "R1" not in regs

    def test_completed_variants_filtered(self):
        rows = [
            {"reg": "R1", "deadline": "2024-06-01", "st": "Complete"},
            {"reg": "R2", "deadline": "2024-06-01", "st": "Done"},
            {"reg": "R3", "deadline": "2024-06-01", "st": "Closed"},
            {"reg": "R4", "deadline": "2024-12-01", "st": "Active"},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline", status_column="st")
        assert result.total_items == 1
        assert result.items[0].regulation == "R4"

    def test_all_completed_returns_none(self):
        rows = [
            {"reg": "R1", "deadline": "2024-01-01", "st": "Completed"},
            {"reg": "R2", "deadline": "2024-06-01", "st": "Done"},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline", status_column="st")
        assert result is None

    def test_owner_breakdown(self):
        rows = [
            {"reg": "R1", "deadline": "2024-01-01", "owner": "Alice"},
            {"reg": "R2", "deadline": "2024-06-01", "owner": "Alice"},
            {"reg": "R3", "deadline": "2024-12-01", "owner": "Bob"},
        ]
        # ref_date = max = 2024-12-01
        # R1: overdue (Alice), R2: overdue (Alice), R3: Due This Week (Bob, 0 days)
        result = track_regulatory_deadlines(rows, "reg", "deadline", owner_column="owner")
        assert result.by_owner is not None
        alice = [o for o in result.by_owner if o.owner == "Alice"][0]
        assert alice.total == 2
        assert alice.overdue == 2
        bob = [o for o in result.by_owner if o.owner == "Bob"][0]
        assert bob.total == 1
        assert bob.overdue == 0

    def test_no_owner_column_returns_none(self):
        rows = [{"reg": "R1", "deadline": "2024-06-01"}]
        result = track_regulatory_deadlines(rows, "reg", "deadline")
        assert result.by_owner is None

    def test_deadline_item_fields(self):
        rows = [
            {"reg": "GDPR", "deadline": "2024-06-01", "st": "Pending", "owner": "Alice"},
        ]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            status_column="st", owner_column="owner",
            reference_date=datetime(2024, 6, 1),
        )
        item = result.items[0]
        assert item.regulation == "GDPR"
        assert item.deadline == "2024-06-01"
        assert item.days_until == 0
        assert item.urgency == "Due This Week"
        assert item.status == "Pending"
        assert item.owner == "Alice"

    def test_overdue_boundary(self):
        # Exactly one day before reference -> overdue
        rows = [{"reg": "R1", "deadline": "2024-05-31"}]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            reference_date=datetime(2024, 6, 1),
        )
        assert result.items[0].urgency == "Overdue"
        assert result.items[0].days_until == -1

    def test_due_this_week_boundary_7_days(self):
        rows = [{"reg": "R1", "deadline": "2024-06-08"}]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            reference_date=datetime(2024, 6, 1),
        )
        assert result.items[0].urgency == "Due This Week"
        assert result.items[0].days_until == 7

    def test_due_this_month_boundary_8_days(self):
        rows = [{"reg": "R1", "deadline": "2024-06-09"}]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            reference_date=datetime(2024, 6, 1),
        )
        assert result.items[0].urgency == "Due This Month"
        assert result.items[0].days_until == 8

    def test_due_this_month_boundary_30_days(self):
        rows = [{"reg": "R1", "deadline": "2024-07-01"}]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            reference_date=datetime(2024, 6, 1),
        )
        assert result.items[0].urgency == "Due This Month"
        assert result.items[0].days_until == 30

    def test_upcoming_boundary_31_days(self):
        rows = [{"reg": "R1", "deadline": "2024-07-02"}]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            reference_date=datetime(2024, 6, 1),
        )
        assert result.items[0].urgency == "Upcoming"
        assert result.items[0].days_until == 31

    def test_single_row(self):
        rows = [{"reg": "GDPR", "deadline": "2024-06-01"}]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline",
            reference_date=datetime(2024, 6, 1),
        )
        assert result.total_items == 1

    def test_summary_contains_key_info(self):
        rows = [
            {"reg": "R1", "deadline": "2024-01-01"},
            {"reg": "R2", "deadline": "2024-12-01"},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline")
        assert "2 regulatory deadlines" in result.summary
        assert "Overdue" in result.summary

    def test_unparseable_date_skipped(self):
        rows = [
            {"reg": "R1", "deadline": "not-a-date"},
            {"reg": "R2", "deadline": "2024-06-01"},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline")
        assert result.total_items == 1

    def test_reference_date_defaults_to_max(self):
        rows = [
            {"reg": "R1", "deadline": "2024-01-01"},
            {"reg": "R2", "deadline": "2024-06-01"},
            {"reg": "R3", "deadline": "2024-12-01"},
        ]
        # Max date = 2024-12-01
        # R1: -335 days -> Overdue
        # R2: -183 days -> Overdue
        # R3: 0 days -> Due This Week
        result = track_regulatory_deadlines(rows, "reg", "deadline")
        assert result.overdue_count == 2
        assert result.due_this_week == 1


# ===================================================================
# 5. format_compliance_report Tests
# ===================================================================


class TestFormatComplianceReport:
    def test_all_none_reports_no_data(self):
        report = format_compliance_report()
        assert "No analysis data provided" in report

    def test_header_always_present(self):
        report = format_compliance_report()
        assert "Compliance Tracking Report" in report
        assert "=" * 40 in report

    def test_status_section(self):
        rows = [
            {"req": "R1", "status": "Compliant"},
            {"req": "R2", "status": "Non-Compliant"},
        ]
        status = audit_compliance_status(rows, "req", "status")
        report = format_compliance_report(status=status)
        assert "Compliance Status" in report
        assert "Total requirements: 2" in report
        assert "Compliant: 1" in report
        assert "Non-compliant: 1" in report
        assert "50.0%" in report

    def test_findings_section(self):
        rows = [
            {"finding": "F1", "severity": "Critical"},
            {"finding": "F2", "severity": "Minor"},
        ]
        findings = analyze_audit_findings(rows, "finding", "severity")
        report = format_compliance_report(findings=findings)
        assert "Audit Findings" in report
        assert "Total findings: 2" in report
        assert "Critical" in report

    def test_score_section(self):
        rows = [
            {"req": "R1", "weight": 1.0, "score": 85.0},
            {"req": "R2", "weight": 1.0, "score": 75.0},
        ]
        score = compute_compliance_score(rows, "req", "weight", "score")
        report = format_compliance_report(score=score)
        assert "Compliance Score" in report
        assert "80.0" in report
        assert "Good" in report

    def test_deadlines_section(self):
        rows = [
            {"reg": "GDPR", "deadline": "2024-06-01"},
            {"reg": "SOX", "deadline": "2024-12-01"},
        ]
        deadlines = track_regulatory_deadlines(rows, "reg", "deadline")
        report = format_compliance_report(deadlines=deadlines)
        assert "Regulatory Deadlines" in report
        assert "Total items: 2" in report
        assert "GDPR" in report

    def test_combined_all_sections(self):
        status_rows = [{"req": "R1", "status": "Compliant"}]
        finding_rows = [{"finding": "F1", "severity": "Critical"}]
        score_rows = [{"req": "R1", "weight": 1.0, "score": 85.0}]
        deadline_rows = [{"reg": "GDPR", "deadline": "2024-06-01"}]

        status = audit_compliance_status(status_rows, "req", "status")
        findings = analyze_audit_findings(finding_rows, "finding", "severity")
        score = compute_compliance_score(score_rows, "req", "weight", "score")
        deadlines = track_regulatory_deadlines(deadline_rows, "reg", "deadline")

        report = format_compliance_report(
            status=status, findings=findings, score=score, deadlines=deadlines
        )
        assert "Compliance Status" in report
        assert "Audit Findings" in report
        assert "Compliance Score" in report
        assert "Regulatory Deadlines" in report

    def test_status_with_categories(self):
        rows = [
            {"req": "R1", "status": "Compliant", "cat": "Security"},
            {"req": "R2", "status": "Non-Compliant", "cat": "Privacy"},
        ]
        status = audit_compliance_status(rows, "req", "status", category_column="cat")
        report = format_compliance_report(status=status)
        assert "By category:" in report
        assert "Security" in report
        assert "Privacy" in report

    def test_status_with_overdue(self):
        rows = [
            {"req": "R1", "status": "In Progress", "due": "2024-01-01"},
            {"req": "R2", "status": "Compliant", "due": "2024-12-01"},
        ]
        status = audit_compliance_status(rows, "req", "status", due_date_column="due")
        report = format_compliance_report(status=status)
        assert "Overdue: 1" in report

    def test_findings_with_closure_rate(self):
        rows = [
            {"finding": "F1", "severity": "Critical", "st": "Open"},
            {"finding": "F2", "severity": "Minor", "st": "Closed"},
        ]
        findings = analyze_audit_findings(rows, "finding", "severity", status_column="st")
        report = format_compliance_report(findings=findings)
        assert "closure rate" in report

    def test_deadlines_with_owners(self):
        rows = [
            {"reg": "R1", "deadline": "2024-06-01", "owner": "Alice"},
            {"reg": "R2", "deadline": "2024-12-01", "owner": "Bob"},
        ]
        deadlines = track_regulatory_deadlines(
            rows, "reg", "deadline", owner_column="owner"
        )
        report = format_compliance_report(deadlines=deadlines)
        assert "By owner:" in report
        assert "Alice" in report
        assert "Bob" in report

    def test_score_with_categories(self):
        rows = [
            {"req": "R1", "weight": 1.0, "score": 80.0, "cat": "A"},
            {"req": "R2", "weight": 1.0, "score": 90.0, "cat": "B"},
        ]
        score = compute_compliance_score(
            rows, "req", "weight", "score", category_column="cat"
        )
        report = format_compliance_report(score=score)
        assert "By category:" in report

    def test_no_data_message_only_when_all_none(self):
        # When at least one section is provided, no "No analysis" message
        rows = [{"req": "R1", "status": "Compliant"}]
        status = audit_compliance_status(rows, "req", "status")
        report = format_compliance_report(status=status)
        assert "No analysis data provided" not in report


# ===================================================================
# 6. Dataclass construction Tests
# ===================================================================


class TestDataclassConstruction:
    def test_category_compliance_fields(self):
        cc = CategoryCompliance(category="IT", total=10, compliant=8, compliance_rate=80.0)
        assert cc.category == "IT"
        assert cc.total == 10
        assert cc.compliant == 8
        assert cc.compliance_rate == 80.0

    def test_severity_count_fields(self):
        sc = SeverityCount(severity="Critical", count=5, pct=25.0)
        assert sc.severity == "Critical"
        assert sc.count == 5
        assert sc.pct == 25.0

    def test_area_findings_fields(self):
        af = AreaFindings(area="Finance", count=10, critical_count=3)
        assert af.area == "Finance"
        assert af.count == 10
        assert af.critical_count == 3

    def test_monthly_finding_trend_fields(self):
        mt = MonthlyFindingTrend(month="2024-01", count=7)
        assert mt.month == "2024-01"
        assert mt.count == 7

    def test_category_score_fields(self):
        cs = CategoryScore(category="Security", score=85.0, weight=2.0)
        assert cs.category == "Security"
        assert cs.score == 85.0
        assert cs.weight == 2.0

    def test_deadline_item_fields(self):
        di = DeadlineItem(
            regulation="GDPR", deadline="2024-06-01", days_until=-5,
            urgency="Overdue", status="Pending", owner="Alice"
        )
        assert di.regulation == "GDPR"
        assert di.days_until == -5
        assert di.urgency == "Overdue"

    def test_owner_deadlines_fields(self):
        od = OwnerDeadlines(owner="Bob", total=5, overdue=2)
        assert od.owner == "Bob"
        assert od.total == 5
        assert od.overdue == 2


# ===================================================================
# 7. Edge Cases and Integration Tests
# ===================================================================


class TestEdgeCases:
    def test_compliance_status_numeric_status(self):
        """Status column contains numbers instead of strings."""
        rows = [{"req": "R1", "status": 1}]
        result = audit_compliance_status(rows, "req", "status")
        assert result is not None
        assert result.compliant_count == 0  # "1" != "compliant"

    def test_audit_findings_numeric_severity(self):
        """Severity column contains numbers."""
        rows = [{"finding": "F1", "severity": 1}]
        result = analyze_audit_findings(rows, "finding", "severity")
        assert result is not None
        assert result.total_findings == 1

    def test_compliance_score_string_numbers(self):
        """Weight and score columns contain string-encoded numbers."""
        rows = [{"req": "R1", "weight": "2", "score": "80"}]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert result is not None
        assert result.overall_score == 80.0

    def test_deadlines_various_date_formats(self):
        rows = [
            {"reg": "R1", "deadline": "2024-06-01"},
            {"reg": "R2", "deadline": "2024/06/15"},
            {"reg": "R3", "deadline": "06/30/2024"},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline")
        assert result is not None
        assert result.total_items == 3

    def test_deadlines_with_datetime_objects(self):
        rows = [
            {"reg": "R1", "deadline": datetime(2024, 6, 1)},
            {"reg": "R2", "deadline": datetime(2024, 12, 1)},
        ]
        result = track_regulatory_deadlines(rows, "reg", "deadline")
        assert result is not None
        assert result.total_items == 2

    def test_compliance_status_large_dataset(self):
        """Test with a larger dataset to verify no performance issues."""
        rows = [
            {"req": f"R{i}", "status": "Compliant" if i % 3 == 0 else "Non-Compliant"}
            for i in range(100)
        ]
        result = audit_compliance_status(rows, "req", "status")
        assert result.total_requirements == 100
        # 0, 3, 6, ..., 99: indices divisible by 3 => 34 items
        assert result.compliant_count == 34

    def test_findings_with_all_optional_columns(self):
        """Full integration test with all optional columns."""
        rows = [
            {"finding": "F1", "severity": "Critical", "date": "2024-01-15",
             "area": "IT", "st": "Open"},
            {"finding": "F2", "severity": "Major", "date": "2024-01-20",
             "area": "IT", "st": "Closed"},
            {"finding": "F3", "severity": "Minor", "date": "2024-02-10",
             "area": "Finance", "st": "Open"},
            {"finding": "F4", "severity": "Minor", "date": "2024-02-15",
             "area": "Finance", "st": "Closed"},
            {"finding": "F5", "severity": "Observation", "date": "2024-03-01",
             "area": "HR", "st": "Open"},
        ]
        result = analyze_audit_findings(
            rows, "finding", "severity",
            date_column="date", area_column="area", status_column="st",
        )
        assert result.total_findings == 5
        assert result.open_count == 3
        assert result.closed_count == 2
        assert result.closure_rate == 40.0
        assert result.by_area is not None
        assert result.monthly_trend is not None
        assert len(result.monthly_trend) == 3

    def test_full_combined_report(self):
        """Full integration test generating a complete report."""
        status_rows = [
            {"req": "R1", "status": "Compliant", "cat": "IT", "due": "2024-01-01"},
            {"req": "R2", "status": "Non-Compliant", "cat": "HR", "due": "2024-06-01"},
            {"req": "R3", "status": "Compliant", "cat": "IT", "due": "2024-12-01"},
        ]
        finding_rows = [
            {"finding": "F1", "severity": "Critical", "date": "2024-01-15",
             "area": "IT", "st": "Closed"},
            {"finding": "F2", "severity": "Minor", "date": "2024-02-10",
             "area": "HR", "st": "Open"},
        ]
        score_rows = [
            {"req": "Security", "weight": 3, "score": 90, "cat": "Tech"},
            {"req": "Privacy", "weight": 2, "score": 70, "cat": "Legal"},
            {"req": "Ops", "weight": 1, "score": 50, "cat": "Ops"},
        ]
        deadline_rows = [
            {"reg": "GDPR", "deadline": "2024-06-01", "st": "Pending", "owner": "Alice"},
            {"reg": "SOX", "deadline": "2024-12-01", "st": "In Progress", "owner": "Bob"},
        ]

        status = audit_compliance_status(
            status_rows, "req", "status",
            category_column="cat", due_date_column="due",
        )
        findings = analyze_audit_findings(
            finding_rows, "finding", "severity",
            date_column="date", area_column="area", status_column="st",
        )
        score = compute_compliance_score(
            score_rows, "req", "weight", "score", category_column="cat",
        )
        deadlines = track_regulatory_deadlines(
            deadline_rows, "reg", "deadline",
            status_column="st", owner_column="owner",
        )

        report = format_compliance_report(
            status=status, findings=findings, score=score, deadlines=deadlines,
        )
        assert "Compliance Tracking Report" in report
        assert "Compliance Status" in report
        assert "Audit Findings" in report
        assert "Compliance Score" in report
        assert "Regulatory Deadlines" in report
        assert len(report) > 200

    def test_compliance_status_only_req_no_status(self):
        """Row has requirement but no status."""
        rows = [{"req": "R1", "status": None}]
        result = audit_compliance_status(rows, "req", "status")
        assert result is not None
        assert result.total_requirements == 1
        assert result.compliant_count == 0

    def test_deadlines_owner_sorted_by_count_desc(self):
        rows = [
            {"reg": "R1", "deadline": "2024-06-01", "owner": "Alice"},
            {"reg": "R2", "deadline": "2024-06-15", "owner": "Alice"},
            {"reg": "R3", "deadline": "2024-07-01", "owner": "Alice"},
            {"reg": "R4", "deadline": "2024-08-01", "owner": "Bob"},
        ]
        result = track_regulatory_deadlines(
            rows, "reg", "deadline", owner_column="owner"
        )
        assert result.by_owner[0].owner == "Alice"
        assert result.by_owner[0].total == 3

    def test_score_weighted_scores_list(self):
        rows = [
            {"req": "R1", "weight": 2.0, "score": 90.0},
            {"req": "R2", "weight": 1.0, "score": 60.0},
        ]
        result = compute_compliance_score(rows, "req", "weight", "score")
        assert len(result.weighted_scores) == 2
        names = [ws.category for ws in result.weighted_scores]
        assert "R1" in names
        assert "R2" in names
