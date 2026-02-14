"""Tests for project tracking, milestones, and resource allocation module."""

from datetime import datetime

from business_brain.discovery.project_tracker import (
    MilestoneResult,
    ProjectHealth,
    ProjectInfo,
    ProjectMilestones,
    ProjectStatusResult,
    ResourceInfo,
    ResourceResult,
    RoleHours,
    analyze_milestones,
    analyze_project_status,
    analyze_resource_allocation,
    compute_project_health,
    format_project_report,
    _safe_float,
    _parse_date,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(42) == 42.0

    def test_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("100.5") == 100.5

    def test_none(self):
        assert _safe_float(None) is None

    def test_non_numeric_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_negative(self):
        assert _safe_float(-25.5) == -25.5


class TestParseDate:
    def test_iso_format(self):
        result = _parse_date("2024-06-15")
        assert result == datetime(2024, 6, 15)

    def test_slash_format(self):
        result = _parse_date("2024/06/15")
        assert result == datetime(2024, 6, 15)

    def test_us_format(self):
        result = _parse_date("06/15/2024")
        assert result == datetime(2024, 6, 15)

    def test_datetime_with_time(self):
        result = _parse_date("2024-06-15 10:30:00")
        assert result == datetime(2024, 6, 15, 10, 30, 0)

    def test_datetime_object(self):
        dt = datetime(2024, 3, 1)
        assert _parse_date(dt) is dt

    def test_none(self):
        assert _parse_date(None) is None

    def test_invalid_string(self):
        assert _parse_date("not-a-date") is None

    def test_empty_string(self):
        assert _parse_date("") is None


# ---------------------------------------------------------------------------
# analyze_project_status
# ---------------------------------------------------------------------------


class TestAnalyzeProjectStatus:
    def test_basic_status(self):
        rows = [
            {"project": "Alpha", "status": "active"},
            {"project": "Beta", "status": "completed"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert len(result.projects) == 2
        assert result.completion_rate == 50.0

    def test_empty_rows(self):
        result = analyze_project_status([], "project", "status")
        assert result is None

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        result = analyze_project_status(rows, "project", "status")
        assert result is None

    def test_status_distribution(self):
        rows = [
            {"project": "A", "status": "active"},
            {"project": "B", "status": "active"},
            {"project": "C", "status": "completed"},
            {"project": "D", "status": "on-hold"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert result.status_distribution["active"] == 2
        assert result.status_distribution["completed"] == 1
        assert result.status_distribution["on-hold"] == 1

    def test_completion_rate_all_completed(self):
        rows = [
            {"project": "A", "status": "completed"},
            {"project": "B", "status": "completed"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert result.completion_rate == 100.0

    def test_completion_rate_none_completed(self):
        rows = [
            {"project": "A", "status": "active"},
            {"project": "B", "status": "delayed"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert result.completion_rate == 0.0

    def test_with_dates(self):
        rows = [
            {"project": "A", "status": "completed", "start": "2024-01-01", "end": "2024-03-01"},
            {"project": "B", "status": "active", "start": "2024-02-01", "end": "2024-04-01"},
        ]
        result = analyze_project_status(
            rows, "project", "status",
            start_column="start", end_column="end"
        )
        assert result is not None
        proj_a = [p for p in result.projects if p.project == "A"][0]
        assert proj_a.start_date == "2024-01-01"
        assert proj_a.end_date == "2024-03-01"
        assert proj_a.duration_days == 60

    def test_avg_duration_completed_only(self):
        rows = [
            {"project": "A", "status": "completed", "start": "2024-01-01", "end": "2024-01-31"},
            {"project": "B", "status": "completed", "start": "2024-01-01", "end": "2024-02-29"},
            {"project": "C", "status": "active", "start": "2024-01-01", "end": "2024-06-01"},
        ]
        result = analyze_project_status(
            rows, "project", "status",
            start_column="start", end_column="end"
        )
        assert result is not None
        # A=30 days, B=59 days; avg=(30+59)/2=44.5
        assert result.avg_duration_days == 44.5

    def test_no_dates_avg_duration_none(self):
        rows = [
            {"project": "A", "status": "completed"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert result.avg_duration_days is None

    def test_with_budget(self):
        rows = [
            {"project": "A", "status": "active", "budget": 50000},
            {"project": "B", "status": "completed", "budget": 30000},
        ]
        result = analyze_project_status(
            rows, "project", "status", budget_column="budget"
        )
        assert result is not None
        assert result.total_budget == 80000.0
        proj_a = [p for p in result.projects if p.project == "A"][0]
        assert proj_a.budget == 50000.0

    def test_no_budget_column(self):
        rows = [{"project": "A", "status": "active"}]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert result.total_budget is None

    def test_budget_aggregation_same_project(self):
        rows = [
            {"project": "A", "status": "active", "budget": 10000},
            {"project": "A", "status": "active", "budget": 5000},
        ]
        result = analyze_project_status(
            rows, "project", "status", budget_column="budget"
        )
        assert result is not None
        assert len(result.projects) == 1
        assert result.projects[0].budget == 15000.0

    def test_status_case_insensitive(self):
        rows = [
            {"project": "A", "status": "COMPLETED"},
            {"project": "B", "status": "Completed"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert result.completion_rate == 100.0
        assert result.status_distribution.get("completed") == 2

    def test_single_project(self):
        rows = [{"project": "Solo", "status": "active"}]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert len(result.projects) == 1
        assert result.projects[0].project == "Solo"

    def test_summary_contains_key_info(self):
        rows = [
            {"project": "A", "status": "active"},
            {"project": "B", "status": "completed"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert "2 projects" in result.summary
        assert "50.0%" in result.summary

    def test_earliest_start_latest_end(self):
        rows = [
            {"project": "A", "status": "active", "start": "2024-03-01", "end": "2024-05-01"},
            {"project": "A", "status": "active", "start": "2024-01-01", "end": "2024-06-01"},
        ]
        result = analyze_project_status(
            rows, "project", "status",
            start_column="start", end_column="end"
        )
        assert result is not None
        proj = result.projects[0]
        assert proj.start_date == "2024-01-01"
        assert proj.end_date == "2024-06-01"

    def test_mixed_valid_invalid_rows(self):
        rows = [
            {"project": "A", "status": "active"},
            {"project": None, "status": "active"},
            {"project": "B", "status": None},
            {"project": "C", "status": "completed"},
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert len(result.projects) == 2  # Only A and C

    def test_invalid_budget_ignored(self):
        rows = [
            {"project": "A", "status": "active", "budget": "not_a_number"},
            {"project": "B", "status": "active", "budget": 1000},
        ]
        result = analyze_project_status(
            rows, "project", "status", budget_column="budget"
        )
        assert result is not None
        assert result.total_budget == 1000.0

    def test_invalid_dates_ignored(self):
        rows = [
            {"project": "A", "status": "active", "start": "bad", "end": "bad"},
        ]
        result = analyze_project_status(
            rows, "project", "status",
            start_column="start", end_column="end"
        )
        assert result is not None
        assert result.projects[0].start_date is None
        assert result.projects[0].end_date is None
        assert result.projects[0].duration_days is None

    def test_datetime_objects_for_dates(self):
        rows = [
            {
                "project": "A",
                "status": "completed",
                "start": datetime(2024, 1, 1),
                "end": datetime(2024, 2, 1),
            },
        ]
        result = analyze_project_status(
            rows, "project", "status",
            start_column="start", end_column="end"
        )
        assert result is not None
        assert result.projects[0].duration_days == 31


# ---------------------------------------------------------------------------
# analyze_milestones
# ---------------------------------------------------------------------------


class TestAnalyzeMilestones:
    def test_basic_milestones(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01"},
            {"project": "B", "milestone": "M1", "due": "2024-03-15"},
        ]
        result = analyze_milestones(rows, "project", "milestone", "due")
        assert result is not None
        assert result.total_milestones == 3
        assert len(result.projects) == 2

    def test_empty_rows(self):
        result = analyze_milestones([], "project", "milestone", "due")
        assert result is None

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        result = analyze_milestones(rows, "project", "milestone", "due")
        assert result is None

    def test_with_completion_dates_on_time(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-02-28"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01", "done": "2024-04-01"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        pm = result.projects[0]
        assert pm.completed == 2
        assert pm.overdue == 0
        assert pm.on_time_pct == 100.0

    def test_with_completion_dates_overdue(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-03-10"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01", "done": "2024-03-30"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        pm = result.projects[0]
        assert pm.completed == 2
        assert pm.overdue == 1  # M1 is late
        assert pm.on_time_pct == 50.0

    def test_avg_delay(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-03-11"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01", "done": "2024-04-06"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        pm = result.projects[0]
        # M1: 10 days late, M2: 5 days late => avg = 7.5
        assert pm.avg_delay_days == 7.5

    def test_health_on_track(self):
        # No overdue milestones => on_track
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-02-28"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01", "done": "2024-04-01"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        assert result.health == "on_track"

    def test_health_at_risk(self):
        # ~25% overdue => at_risk
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-03-10"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01", "done": "2024-03-30"},
            {"project": "A", "milestone": "M3", "due": "2024-05-01", "done": "2024-04-30"},
            {"project": "A", "milestone": "M4", "due": "2024-06-01", "done": "2024-05-28"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        assert result.health == "at_risk"

    def test_health_critical(self):
        # All overdue => critical
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-03-10"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01", "done": "2024-04-10"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        assert result.health == "critical"

    def test_overall_on_time_pct(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-02-28"},
            {"project": "B", "milestone": "M1", "due": "2024-03-01", "done": "2024-03-10"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        assert result.overall_on_time_pct == 50.0

    def test_no_completion_dates_on_time_pct_none(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01"},
        ]
        result = analyze_milestones(rows, "project", "milestone", "due")
        assert result is not None
        assert result.overall_on_time_pct is None

    def test_upcoming_milestones(self):
        # Milestones due within 30 days from max date in data, not completed
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "done": "2024-02-28"},
            {"project": "A", "milestone": "M2", "due": "2024-03-15"},
            {"project": "A", "milestone": "M3", "due": "2024-04-15"},
        ]
        # Max date = 2024-04-15, upcoming window = 2024-04-15 + 30 days = 2024-05-15
        # M2 due 2024-03-15 < ref_date 2024-04-15 => not upcoming (past)
        # M3 due 2024-04-15 is ref_date, within window => upcoming if not completed
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done"
        )
        assert result is not None
        assert result.upcoming_count >= 1

    def test_with_status_column(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "ms_status": "completed"},
            {"project": "A", "milestone": "M2", "due": "2024-04-01", "ms_status": "pending"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            status_column="ms_status"
        )
        assert result is not None
        pm = result.projects[0]
        assert pm.completed == 1

    def test_status_column_done(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "ms_status": "done"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            status_column="ms_status"
        )
        assert result is not None
        assert result.projects[0].completed == 1

    def test_status_column_finished(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01", "ms_status": "finished"},
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            status_column="ms_status"
        )
        assert result is not None
        assert result.projects[0].completed == 1

    def test_invalid_due_dates_ignored(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "not-a-date"},
            {"project": "A", "milestone": "M2", "due": "2024-03-01"},
        ]
        result = analyze_milestones(rows, "project", "milestone", "due")
        assert result is not None
        assert result.total_milestones == 1

    def test_summary_contains_key_info(self):
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-03-01"},
        ]
        result = analyze_milestones(rows, "project", "milestone", "due")
        assert result is not None
        assert "1 milestones" in result.summary or "1 milestone" in result.summary
        assert "1 projects" in result.summary or "1 project" in result.summary

    def test_multiple_projects(self):
        rows = [
            {"project": "Alpha", "milestone": "M1", "due": "2024-03-01"},
            {"project": "Alpha", "milestone": "M2", "due": "2024-04-01"},
            {"project": "Beta", "milestone": "M1", "due": "2024-03-15"},
            {"project": "Gamma", "milestone": "M1", "due": "2024-05-01"},
        ]
        result = analyze_milestones(rows, "project", "milestone", "due")
        assert result is not None
        assert len(result.projects) == 3
        alpha = [p for p in result.projects if p.project == "Alpha"][0]
        assert alpha.total == 2


# ---------------------------------------------------------------------------
# analyze_resource_allocation
# ---------------------------------------------------------------------------


class TestAnalyzeResourceAllocation:
    def test_basic_allocation(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 80},
            {"resource": "Alice", "project": "Beta", "hours": 40},
            {"resource": "Bob", "project": "Alpha", "hours": 120},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        assert len(result.resources) == 2

    def test_empty_rows(self):
        result = analyze_resource_allocation([], "resource", "project", "hours")
        assert result is None

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is None

    def test_total_hours(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 80},
            {"resource": "Alice", "project": "Beta", "hours": 60},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        alice = [r for r in result.resources if r.resource == "Alice"][0]
        assert alice.total_hours == 140.0
        assert alice.project_count == 2
        assert alice.avg_hours_per_project == 70.0

    def test_over_allocated(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 100},
            {"resource": "Alice", "project": "Beta", "hours": 80},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        assert "Alice" in result.over_allocated
        alice = [r for r in result.resources if r.resource == "Alice"][0]
        assert alice.utilization_status == "over_allocated"

    def test_under_utilized(self):
        rows = [
            {"resource": "Bob", "project": "Alpha", "hours": 40},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        assert "Bob" in result.under_utilized
        bob = [r for r in result.resources if r.resource == "Bob"][0]
        assert bob.utilization_status == "under_utilized"

    def test_optimal_utilization(self):
        rows = [
            {"resource": "Carol", "project": "Alpha", "hours": 100},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        carol = [r for r in result.resources if r.resource == "Carol"][0]
        assert carol.utilization_status == "optimal"
        assert carol.resource not in result.over_allocated
        assert carol.resource not in result.under_utilized

    def test_boundary_160_hours(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 160},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        alice = result.resources[0]
        assert alice.utilization_status == "optimal"

    def test_boundary_80_hours(self):
        rows = [
            {"resource": "Bob", "project": "Alpha", "hours": 80},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        bob = result.resources[0]
        assert bob.utilization_status == "optimal"

    def test_with_role_column(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 100, "role": "Developer"},
            {"resource": "Bob", "project": "Alpha", "hours": 90, "role": "Developer"},
            {"resource": "Carol", "project": "Beta", "hours": 80, "role": "Designer"},
        ]
        result = analyze_resource_allocation(
            rows, "resource", "project", "hours", role_column="role"
        )
        assert result is not None
        assert result.by_role is not None
        assert len(result.by_role) == 2
        dev = [r for r in result.by_role if r.role == "Developer"][0]
        assert dev.total_hours == 190.0
        assert dev.resource_count == 2

    def test_no_role_column(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 100},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        assert result.by_role is None

    def test_invalid_hours_ignored(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": "bad"},
            {"resource": "Bob", "project": "Alpha", "hours": 100},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        assert len(result.resources) == 1
        assert result.resources[0].resource == "Bob"

    def test_string_hours(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": "120.5"},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        assert result.resources[0].total_hours == 120.5

    def test_multiple_resources_multiple_projects(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 50},
            {"resource": "Alice", "project": "Beta", "hours": 50},
            {"resource": "Alice", "project": "Gamma", "hours": 50},
            {"resource": "Bob", "project": "Alpha", "hours": 30},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        alice = [r for r in result.resources if r.resource == "Alice"][0]
        assert alice.project_count == 3
        assert alice.avg_hours_per_project == 50.0

    def test_summary_contains_key_info(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 200},
            {"resource": "Bob", "project": "Alpha", "hours": 40},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        assert "2 resources" in result.summary
        assert "Over-allocated" in result.summary
        assert "Under-utilized" in result.summary

    def test_same_project_different_rows(self):
        rows = [
            {"resource": "Alice", "project": "Alpha", "hours": 40},
            {"resource": "Alice", "project": "Alpha", "hours": 50},
        ]
        result = analyze_resource_allocation(rows, "resource", "project", "hours")
        assert result is not None
        alice = result.resources[0]
        assert alice.total_hours == 90.0
        assert alice.project_count == 1  # Same project, counted once


# ---------------------------------------------------------------------------
# compute_project_health
# ---------------------------------------------------------------------------


class TestComputeProjectHealth:
    def test_basic_health(self):
        rows = [
            {"project": "Alpha", "planned": 100, "actual": 102},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 1
        assert result[0].project == "Alpha"
        assert result[0].health == "on_track"

    def test_empty_rows(self):
        result = compute_project_health([], "project", "planned", "actual")
        assert result == []

    def test_missing_columns(self):
        rows = [{"other": "val"}]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result == []

    def test_on_track(self):
        rows = [
            {"project": "A", "planned": 1000, "actual": 1040},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].health == "on_track"
        assert result[0].variance_pct < 5

    def test_at_risk(self):
        rows = [
            {"project": "A", "planned": 1000, "actual": 1100},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].health == "at_risk"

    def test_critical(self):
        rows = [
            {"project": "A", "planned": 1000, "actual": 1500},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].health == "critical"

    def test_variance_calculation(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 120},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].variance == 20.0
        assert result[0].variance_pct == 20.0

    def test_negative_variance(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 80},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].variance == -20.0
        assert result[0].variance_pct == 20.0  # abs percentage

    def test_performance_index(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 125},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].performance_index == 0.8  # 100/125

    def test_performance_index_under_budget(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 80},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].performance_index == 1.25  # 100/80

    def test_sorted_by_variance_pct_desc(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 105},  # 5%
            {"project": "B", "planned": 100, "actual": 150},  # 50%
            {"project": "C", "planned": 100, "actual": 110},  # 10%
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].project == "B"
        assert result[1].project == "C"
        assert result[2].project == "A"

    def test_zero_planned(self):
        rows = [
            {"project": "A", "planned": 0, "actual": 100},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 1
        assert result[0].variance_pct == 100.0

    def test_zero_planned_zero_actual(self):
        rows = [
            {"project": "A", "planned": 0, "actual": 0},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 1
        assert result[0].variance_pct == 0.0
        assert result[0].performance_index == 1.0
        assert result[0].health == "on_track"

    def test_zero_actual(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 0},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 1
        assert result[0].performance_index == float("inf")

    def test_aggregation_same_project(self):
        rows = [
            {"project": "A", "planned": 50, "actual": 60},
            {"project": "A", "planned": 50, "actual": 55},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 1
        assert result[0].planned == 100.0
        assert result[0].actual == 115.0

    def test_metric_type_cost(self):
        rows = [{"project": "A", "planned": 100, "actual": 100}]
        result = compute_project_health(
            rows, "project", "planned", "actual", metric_type="cost"
        )
        assert len(result) == 1

    def test_metric_type_time(self):
        rows = [{"project": "A", "planned": 100, "actual": 100}]
        result = compute_project_health(
            rows, "project", "planned", "actual", metric_type="time"
        )
        assert len(result) == 1

    def test_string_numeric_values(self):
        rows = [{"project": "A", "planned": "200", "actual": "210"}]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 1
        assert result[0].planned == 200.0
        assert result[0].actual == 210.0

    def test_invalid_values_ignored(self):
        rows = [
            {"project": "A", "planned": "bad", "actual": 100},
            {"project": "B", "planned": 100, "actual": 100},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 1
        assert result[0].project == "B"

    def test_multiple_projects(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 100},
            {"project": "B", "planned": 200, "actual": 250},
            {"project": "C", "planned": 300, "actual": 280},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert len(result) == 3


# ---------------------------------------------------------------------------
# format_project_report
# ---------------------------------------------------------------------------


class TestFormatProjectReport:
    def test_no_data(self):
        report = format_project_report()
        assert "No analysis data provided" in report

    def test_report_header(self):
        report = format_project_report()
        assert report.startswith("Project Tracking Report")
        assert "=" * 40 in report

    def test_with_status(self):
        status = ProjectStatusResult(
            projects=[
                ProjectInfo("Alpha", "active", "2024-01-01", "2024-06-01", 50000.0, 152),
            ],
            status_distribution={"active": 1},
            completion_rate=0.0,
            avg_duration_days=None,
            total_budget=50000.0,
            summary="test",
        )
        report = format_project_report(status=status)
        assert "Project Status" in report
        assert "Alpha" in report
        assert "active" in report
        assert "50,000.00" in report
        assert "Completion rate: 0.0%" in report

    def test_with_milestones(self):
        milestones = MilestoneResult(
            projects=[
                ProjectMilestones("Alpha", 5, 3, 1, 75.0, 2.5),
            ],
            total_milestones=5,
            overall_on_time_pct=75.0,
            health="at_risk",
            upcoming_count=2,
            summary="test",
        )
        report = format_project_report(milestones=milestones)
        assert "Milestones" in report
        assert "Alpha" in report
        assert "3/5 completed" in report
        assert "1 overdue" in report
        assert "at_risk" in report
        assert "Upcoming (30 days): 2" in report

    def test_with_resources(self):
        resources = ResourceResult(
            resources=[
                ResourceInfo("Alice", 180.0, 3, 60.0, "over_allocated"),
                ResourceInfo("Bob", 40.0, 1, 40.0, "under_utilized"),
            ],
            by_role=[RoleHours("Developer", 220.0, 2)],
            over_allocated=["Alice"],
            under_utilized=["Bob"],
            summary="test",
        )
        report = format_project_report(resources=resources)
        assert "Resource Allocation" in report
        assert "Alice" in report
        assert "over_allocated" in report
        assert "Over-allocated: Alice" in report
        assert "Under-utilized: Bob" in report
        assert "By role:" in report
        assert "Developer" in report

    def test_with_health(self):
        health = [
            ProjectHealth("Alpha", 1000, 1200, 200, 20.0, 0.8333, "critical"),
        ]
        report = format_project_report(health=health)
        assert "Project Health" in report
        assert "Alpha" in report
        assert "critical" in report
        assert "1,000.00" in report
        assert "1,200.00" in report

    def test_full_report(self):
        status = ProjectStatusResult(
            projects=[ProjectInfo("A", "active", None, None, None, None)],
            status_distribution={"active": 1},
            completion_rate=0.0,
            avg_duration_days=None,
            total_budget=None,
            summary="test",
        )
        milestones = MilestoneResult(
            projects=[ProjectMilestones("A", 2, 1, 0, 100.0, -1.0)],
            total_milestones=2,
            overall_on_time_pct=100.0,
            health="on_track",
            upcoming_count=1,
            summary="test",
        )
        resources = ResourceResult(
            resources=[ResourceInfo("Alice", 100.0, 1, 100.0, "optimal")],
            by_role=None,
            over_allocated=[],
            under_utilized=[],
            summary="test",
        )
        health = [ProjectHealth("A", 100, 95, -5, 5.0, 1.0526, "at_risk")]
        report = format_project_report(status, milestones, resources, health)
        assert "Project Tracking Report" in report
        assert "Project Status" in report
        assert "Milestones" in report
        assert "Resource Allocation" in report
        assert "Project Health" in report

    def test_status_with_avg_duration(self):
        status = ProjectStatusResult(
            projects=[ProjectInfo("A", "completed", "2024-01-01", "2024-02-01", None, 31)],
            status_distribution={"completed": 1},
            completion_rate=100.0,
            avg_duration_days=31.0,
            total_budget=None,
            summary="test",
        )
        report = format_project_report(status=status)
        assert "Avg duration (completed): 31.0 days" in report

    def test_milestones_without_upcoming(self):
        milestones = MilestoneResult(
            projects=[ProjectMilestones("A", 2, 2, 0, 100.0, -1.0)],
            total_milestones=2,
            overall_on_time_pct=100.0,
            health="on_track",
            upcoming_count=0,
            summary="test",
        )
        report = format_project_report(milestones=milestones)
        assert "Upcoming" not in report

    def test_resources_without_role(self):
        resources = ResourceResult(
            resources=[ResourceInfo("Alice", 100.0, 1, 100.0, "optimal")],
            by_role=None,
            over_allocated=[],
            under_utilized=[],
            summary="test",
        )
        report = format_project_report(resources=resources)
        assert "By role:" not in report

    def test_resources_no_over_allocated(self):
        resources = ResourceResult(
            resources=[ResourceInfo("Alice", 100.0, 1, 100.0, "optimal")],
            by_role=None,
            over_allocated=[],
            under_utilized=[],
            summary="test",
        )
        report = format_project_report(resources=resources)
        assert "Over-allocated:" not in report

    def test_status_without_budget(self):
        status = ProjectStatusResult(
            projects=[ProjectInfo("A", "active", None, None, None, None)],
            status_distribution={"active": 1},
            completion_rate=0.0,
            avg_duration_days=None,
            total_budget=None,
            summary="test",
        )
        report = format_project_report(status=status)
        assert "Total budget:" not in report


# ---------------------------------------------------------------------------
# Integration / edge cases
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_project_status_with_all_options(self):
        rows = [
            {
                "project": "Alpha",
                "status": "completed",
                "start": "2024-01-01",
                "end": "2024-03-01",
                "budget": 50000,
            },
            {
                "project": "Beta",
                "status": "active",
                "start": "2024-02-01",
                "end": "2024-08-01",
                "budget": 80000,
            },
        ]
        result = analyze_project_status(
            rows, "project", "status",
            start_column="start", end_column="end", budget_column="budget"
        )
        assert result is not None
        assert result.total_budget == 130000.0
        assert result.completion_rate == 50.0
        assert result.avg_duration_days == 60.0

    def test_milestones_with_all_options(self):
        rows = [
            {
                "project": "A",
                "milestone": "M1",
                "due": "2024-03-01",
                "done": "2024-02-28",
                "ms_status": "completed",
            },
            {
                "project": "A",
                "milestone": "M2",
                "due": "2024-04-01",
                "done": "2024-04-10",
                "ms_status": "completed",
            },
        ]
        result = analyze_milestones(
            rows, "project", "milestone", "due",
            completion_date_column="done",
            status_column="ms_status"
        )
        assert result is not None
        assert result.projects[0].completed == 2
        assert result.projects[0].overdue == 1

    def test_resource_allocation_with_role(self):
        rows = [
            {"resource": "Alice", "project": "A", "hours": 80, "role": "Dev"},
            {"resource": "Alice", "project": "B", "hours": 90, "role": "Dev"},
            {"resource": "Bob", "project": "A", "hours": 60, "role": "QA"},
        ]
        result = analyze_resource_allocation(
            rows, "resource", "project", "hours", role_column="role"
        )
        assert result is not None
        assert "Alice" in result.over_allocated
        assert "Bob" in result.under_utilized
        assert result.by_role is not None

    def test_project_health_exact_boundary_5pct(self):
        # Exactly at 5% boundary
        rows = [
            {"project": "A", "planned": 100, "actual": 105},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].variance_pct == 5.0
        assert result[0].health == "at_risk"  # 5% is >= 5, so at_risk

    def test_project_health_just_under_5pct(self):
        rows = [
            {"project": "A", "planned": 1000, "actual": 1049},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].variance_pct < 5.0
        assert result[0].health == "on_track"

    def test_project_health_exact_boundary_15pct(self):
        rows = [
            {"project": "A", "planned": 100, "actual": 115},
        ]
        result = compute_project_health(rows, "project", "planned", "actual")
        assert result[0].variance_pct == 15.0
        assert result[0].health == "critical"  # 15% is >= 15, so critical

    def test_format_report_with_none_health_list(self):
        report = format_project_report(health=None)
        assert "No analysis data provided" in report

    def test_format_report_with_empty_health_list(self):
        report = format_project_report(health=[])
        assert "No analysis data provided" in report

    def test_large_dataset(self):
        rows = [
            {"project": f"Proj{i}", "status": "active" if i % 2 == 0 else "completed"}
            for i in range(100)
        ]
        result = analyze_project_status(rows, "project", "status")
        assert result is not None
        assert len(result.projects) == 100
        assert result.completion_rate == 50.0

    def test_overdue_not_completed_past_due(self):
        # A milestone not completed and past due (relative to max date)
        rows = [
            {"project": "A", "milestone": "M1", "due": "2024-01-01"},
            {"project": "A", "milestone": "M2", "due": "2024-06-01"},
        ]
        # Max date is 2024-06-01, M1 due 2024-01-01 < ref_date => overdue
        result = analyze_milestones(rows, "project", "milestone", "due")
        assert result is not None
        pm = result.projects[0]
        assert pm.overdue >= 1
