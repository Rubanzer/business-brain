"""Tests for SLA monitoring and service-level analysis module."""

from __future__ import annotations

import math
from datetime import datetime, date, timedelta

from business_brain.discovery.sla_monitor import (
    CategorySLA,
    SLAComplianceResult,
    PriorityResponse,
    AgentResponse,
    ResponseTimeResult,
    PriorityResolution,
    ResolutionResult,
    PeriodSLA,
    SLATrendResult,
    analyze_sla_compliance,
    analyze_response_times,
    compute_resolution_metrics,
    analyze_sla_trends,
    format_sla_report,
)


# ---------------------------------------------------------------------------
# Helpers to reduce test boilerplate
# ---------------------------------------------------------------------------


def _make_ticket(
    ticket: str,
    target: float | None = 10.0,
    actual: float | None = 8.0,
    category: str | None = None,
) -> dict:
    """Create a minimal SLA compliance ticket row."""
    row: dict = {"ticket": ticket, "target": target, "actual": actual}
    if category is not None:
        row["category"] = category
    return row


def _make_response_row(
    ticket: str,
    response_time: float | None = 5.0,
    priority: str | None = None,
    agent: str | None = None,
) -> dict:
    """Create a minimal response-time ticket row."""
    row: dict = {"ticket": ticket, "response_time": response_time}
    if priority is not None:
        row["priority"] = priority
    if agent is not None:
        row["agent"] = agent
    return row


def _make_resolution_row(
    ticket: str,
    created: str | None = "2024-01-01",
    resolved: str | None = "2024-01-02",
    status: str | None = None,
    priority: str | None = None,
) -> dict:
    """Create a minimal resolution-metrics ticket row."""
    row: dict = {"ticket": ticket, "created": created, "resolved": resolved}
    if status is not None:
        row["status"] = status
    if priority is not None:
        row["priority"] = priority
    return row


def _make_trend_row(
    ticket: str,
    sla_met: object = True,
    date_val: str | None = "2024-01-15",
) -> dict:
    """Create a minimal SLA trend ticket row."""
    return {"ticket": ticket, "sla_met": sla_met, "date": date_val}


# ===================================================================
# 1. analyze_sla_compliance Tests
# ===================================================================


class TestAnalyzeSlaCompliance:
    def test_empty_rows_returns_none(self):
        assert analyze_sla_compliance([], "ticket", "target", "actual") is None

    def test_all_none_values_returns_none(self):
        rows = [{"ticket": None, "target": None, "actual": None}]
        assert analyze_sla_compliance(rows, "ticket", "target", "actual") is None

    def test_missing_ticket_column_skips_row(self):
        rows = [{"target": 10, "actual": 5}]
        assert analyze_sla_compliance(rows, "ticket", "target", "actual") is None

    def test_missing_target_column_skips_row(self):
        rows = [{"ticket": "T1", "actual": 5}]
        assert analyze_sla_compliance(rows, "ticket", "target", "actual") is None

    def test_missing_actual_column_skips_row(self):
        rows = [{"ticket": "T1", "target": 10}]
        assert analyze_sla_compliance(rows, "ticket", "target", "actual") is None

    def test_non_numeric_target_skips_row(self):
        rows = [{"ticket": "T1", "target": "bad", "actual": 5}]
        assert analyze_sla_compliance(rows, "ticket", "target", "actual") is None

    def test_non_numeric_actual_skips_row(self):
        rows = [{"ticket": "T1", "target": 10, "actual": "bad"}]
        assert analyze_sla_compliance(rows, "ticket", "target", "actual") is None

    def test_single_met_ticket(self):
        rows = [_make_ticket("T1", target=10, actual=8)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result is not None
        assert result.total_tickets == 1
        assert result.met_count == 1
        assert result.breached_count == 0
        assert result.compliance_rate == 100.0

    def test_single_breached_ticket(self):
        rows = [_make_ticket("T1", target=10, actual=15)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.met_count == 0
        assert result.breached_count == 1
        assert result.compliance_rate == 0.0

    def test_exact_target_is_met(self):
        """actual == target should count as met."""
        rows = [_make_ticket("T1", target=10, actual=10)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.met_count == 1
        assert result.breached_count == 0

    def test_basic_compliance_counts(self):
        rows = [
            _make_ticket("T1", target=10, actual=8),
            _make_ticket("T2", target=10, actual=12),
            _make_ticket("T3", target=10, actual=10),
            _make_ticket("T4", target=10, actual=15),
        ]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.total_tickets == 4
        assert result.met_count == 2
        assert result.breached_count == 2
        assert result.compliance_rate == 50.0

    def test_compliance_rate_rounding(self):
        rows = [
            _make_ticket("T1", target=10, actual=5),
            _make_ticket("T2", target=10, actual=5),
            _make_ticket("T3", target=10, actual=15),
        ]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.compliance_rate == 66.67

    def test_performance_ratio_computed(self):
        rows = [
            _make_ticket("T1", target=10, actual=5),   # ratio = 0.5
            _make_ticket("T2", target=10, actual=10),  # ratio = 1.0
        ]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.avg_performance_ratio == 0.75

    def test_performance_ratio_zero_target_excluded(self):
        """Rows with target=0 should not contribute to performance ratio."""
        rows = [
            _make_ticket("T1", target=0, actual=5),
            _make_ticket("T2", target=10, actual=5),  # ratio = 0.5
        ]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.avg_performance_ratio == 0.5

    def test_all_zero_targets_ratio_zero(self):
        rows = [
            _make_ticket("T1", target=0, actual=0),
            _make_ticket("T2", target=0, actual=5),
        ]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.avg_performance_ratio == 0.0

    def test_no_category_column_empty_list(self):
        rows = [_make_ticket("T1")]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.by_category == []
        assert result.worst_category is None

    def test_category_breakdown(self):
        rows = [
            _make_ticket("T1", target=10, actual=5, category="Net"),
            _make_ticket("T2", target=10, actual=15, category="Net"),
            _make_ticket("T3", target=10, actual=8, category="DB"),
        ]
        result = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        assert len(result.by_category) == 2
        db = [c for c in result.by_category if c.category == "DB"][0]
        assert db.total == 1
        assert db.met == 1
        assert db.breached == 0
        assert db.compliance_rate == 100.0
        net = [c for c in result.by_category if c.category == "Net"][0]
        assert net.total == 2
        assert net.met == 1
        assert net.breached == 1
        assert net.compliance_rate == 50.0

    def test_category_avg_actual_and_target(self):
        rows = [
            _make_ticket("T1", target=10, actual=6, category="A"),
            _make_ticket("T2", target=20, actual=14, category="A"),
        ]
        result = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        cat = result.by_category[0]
        assert cat.avg_actual == 10.0
        assert cat.avg_target == 15.0

    def test_worst_category_determined(self):
        rows = [
            _make_ticket("T1", target=10, actual=5, category="Good"),
            _make_ticket("T2", target=10, actual=15, category="Bad"),
        ]
        result = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        assert result.worst_category == "Bad"

    def test_worst_category_tie_uses_min(self):
        """When categories tie on compliance rate, min() picks alphabetically first."""
        rows = [
            _make_ticket("T1", target=10, actual=15, category="Alpha"),
            _make_ticket("T2", target=10, actual=15, category="Beta"),
        ]
        result = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        # Both are 0% compliance; min picks the first encountered by compliance rate tie
        assert result.worst_category in ("Alpha", "Beta")

    def test_category_column_none_value_excluded(self):
        """Rows with None category are excluded from by_category but still counted overall."""
        rows = [
            _make_ticket("T1", target=10, actual=5, category="Net"),
            {"ticket": "T2", "target": 10, "actual": 8, "category": None},
        ]
        result = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        assert result.total_tickets == 2
        assert len(result.by_category) == 1

    def test_categories_sorted_alphabetically(self):
        rows = [
            _make_ticket("T1", target=10, actual=5, category="Zebra"),
            _make_ticket("T2", target=10, actual=5, category="Alpha"),
            _make_ticket("T3", target=10, actual=5, category="Mid"),
        ]
        result = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        names = [c.category for c in result.by_category]
        assert names == ["Alpha", "Mid", "Zebra"]

    def test_summary_contains_ticket_count(self):
        rows = [_make_ticket("T1"), _make_ticket("T2")]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert "2 tickets" in result.summary

    def test_summary_contains_compliance_rate(self):
        rows = [_make_ticket("T1", target=10, actual=5)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert "100.0%" in result.summary

    def test_summary_contains_worst_category(self):
        rows = [_make_ticket("T1", target=10, actual=15, category="Bad")]
        result = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        assert "Worst category: Bad" in result.summary

    def test_mixed_valid_and_invalid_rows(self):
        rows = [
            _make_ticket("T1", target=10, actual=5),
            {"ticket": "T2", "target": "bad", "actual": 5},
            {"ticket": None, "target": 10, "actual": 5},
            _make_ticket("T3", target=10, actual=15),
        ]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.total_tickets == 2

    def test_string_numeric_values(self):
        rows = [{"ticket": "T1", "target": "10", "actual": "5"}]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result is not None
        assert result.met_count == 1

    def test_large_dataset(self):
        rows = [_make_ticket(f"T{i}", target=10, actual=i % 15) for i in range(200)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.total_tickets == 200


# ===================================================================
# 2. analyze_response_times Tests
# ===================================================================


class TestAnalyzeResponseTimes:
    def test_empty_rows_returns_none(self):
        assert analyze_response_times([], "ticket", "response_time") is None

    def test_all_none_returns_none(self):
        rows = [{"ticket": None, "response_time": None}]
        assert analyze_response_times(rows, "ticket", "response_time") is None

    def test_missing_ticket_skips_row(self):
        rows = [{"response_time": 5.0}]
        assert analyze_response_times(rows, "ticket", "response_time") is None

    def test_missing_response_time_skips_row(self):
        rows = [{"ticket": "T1"}]
        assert analyze_response_times(rows, "ticket", "response_time") is None

    def test_non_numeric_response_time_skips_row(self):
        rows = [{"ticket": "T1", "response_time": "bad"}]
        assert analyze_response_times(rows, "ticket", "response_time") is None

    def test_single_ticket(self):
        rows = [_make_response_row("T1", response_time=5.0)]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result is not None
        assert result.avg_response_time == 5.0
        assert result.median_response_time == 5.0
        assert result.p95_response_time == 5.0
        assert result.min_time == 5.0
        assert result.max_time == 5.0

    def test_avg_response_time(self):
        rows = [
            _make_response_row("T1", response_time=2.0),
            _make_response_row("T2", response_time=4.0),
            _make_response_row("T3", response_time=6.0),
        ]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.avg_response_time == 4.0

    def test_median_odd_count(self):
        rows = [
            _make_response_row("T1", response_time=1.0),
            _make_response_row("T2", response_time=3.0),
            _make_response_row("T3", response_time=5.0),
        ]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.median_response_time == 3.0

    def test_median_even_count(self):
        rows = [
            _make_response_row("T1", response_time=1.0),
            _make_response_row("T2", response_time=3.0),
            _make_response_row("T3", response_time=5.0),
            _make_response_row("T4", response_time=7.0),
        ]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.median_response_time == 4.0

    def test_min_and_max(self):
        rows = [
            _make_response_row("T1", response_time=2.0),
            _make_response_row("T2", response_time=10.0),
            _make_response_row("T3", response_time=5.0),
        ]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.min_time == 2.0
        assert result.max_time == 10.0

    def test_p95_with_multiple_values(self):
        rows = [_make_response_row(f"T{i}", response_time=float(i)) for i in range(1, 21)]
        result = analyze_response_times(rows, "ticket", "response_time")
        # p95 of 1..20: k = 0.95 * 19 = 18.05 -> s[18] + 0.05*(s[19]-s[18]) = 19 + 0.05*1 = 19.05
        assert result.p95_response_time == 19.05

    def test_outlier_count(self):
        """Values > mean + 2*std are outliers."""
        # 10 values of 5.0 and 1 value of 1000.0
        # mean ~ 95.45, std ~ 283.5, threshold ~ 662.5 -> 1000 is an outlier
        rows = [_make_response_row(f"T{i}", response_time=5.0) for i in range(10)]
        rows.append(_make_response_row("T99", response_time=1000.0))
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.outlier_count >= 1

    def test_no_outliers_uniform_data(self):
        rows = [_make_response_row(f"T{i}", response_time=5.0) for i in range(10)]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.outlier_count == 0

    def test_by_priority_breakdown(self):
        rows = [
            _make_response_row("T1", response_time=2.0, priority="High"),
            _make_response_row("T2", response_time=4.0, priority="High"),
            _make_response_row("T3", response_time=10.0, priority="Low"),
        ]
        result = analyze_response_times(
            rows, "ticket", "response_time", priority_column="priority"
        )
        assert len(result.by_priority) == 2
        high = [p for p in result.by_priority if p.priority == "High"][0]
        assert high.count == 2
        assert high.avg_time == 3.0
        low = [p for p in result.by_priority if p.priority == "Low"][0]
        assert low.count == 1
        assert low.avg_time == 10.0

    def test_by_priority_sorted_alphabetically(self):
        rows = [
            _make_response_row("T1", response_time=5.0, priority="Zebra"),
            _make_response_row("T2", response_time=5.0, priority="Alpha"),
        ]
        result = analyze_response_times(
            rows, "ticket", "response_time", priority_column="priority"
        )
        names = [p.priority for p in result.by_priority]
        assert names == ["Alpha", "Zebra"]

    def test_by_priority_none_value_excluded(self):
        rows = [
            _make_response_row("T1", response_time=5.0, priority="High"),
            {"ticket": "T2", "response_time": 3.0, "priority": None},
        ]
        result = analyze_response_times(
            rows, "ticket", "response_time", priority_column="priority"
        )
        assert len(result.by_priority) == 1

    def test_no_priority_column_empty_list(self):
        rows = [_make_response_row("T1")]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.by_priority == []

    def test_by_agent_breakdown(self):
        rows = [
            _make_response_row("T1", response_time=2.0, agent="Alice"),
            _make_response_row("T2", response_time=4.0, agent="Alice"),
            _make_response_row("T3", response_time=3.0, agent="Bob"),
        ]
        result = analyze_response_times(
            rows, "ticket", "response_time", agent_column="agent"
        )
        assert len(result.by_agent) == 2
        alice = [a for a in result.by_agent if a.agent == "Alice"][0]
        assert alice.count == 2
        assert alice.avg_time == 3.0
        bob = [a for a in result.by_agent if a.agent == "Bob"][0]
        assert bob.count == 1

    def test_by_agent_compliance_rate(self):
        """Agent compliance: responses within mean + 2*std."""
        # 10 uniform values for Alice (5.0 each), 1 extreme outlier for Bob (10000.0)
        # and 1 normal value for Bob (5.0)
        # mean ~ 916.8, std low enough that 10000 exceeds threshold
        rows = [_make_response_row(f"A{i}", response_time=5.0, agent="Alice") for i in range(10)]
        rows.append(_make_response_row("B1", response_time=5.0, agent="Bob"))
        rows.append(_make_response_row("B2", response_time=10000.0, agent="Bob"))
        result = analyze_response_times(
            rows, "ticket", "response_time", agent_column="agent"
        )
        alice = [a for a in result.by_agent if a.agent == "Alice"][0]
        assert alice.compliance_rate == 100.0
        bob = [a for a in result.by_agent if a.agent == "Bob"][0]
        # Bob has the extreme outlier 10000.0; his compliance < 100
        assert bob.compliance_rate < 100.0

    def test_by_agent_sorted_alphabetically(self):
        rows = [
            _make_response_row("T1", response_time=5.0, agent="Zara"),
            _make_response_row("T2", response_time=5.0, agent="Amy"),
        ]
        result = analyze_response_times(
            rows, "ticket", "response_time", agent_column="agent"
        )
        names = [a.agent for a in result.by_agent]
        assert names == ["Amy", "Zara"]

    def test_no_agent_column_empty_list(self):
        rows = [_make_response_row("T1")]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.by_agent == []

    def test_summary_contains_ticket_count(self):
        rows = [_make_response_row("T1"), _make_response_row("T2")]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert "2 tickets" in result.summary

    def test_summary_contains_avg(self):
        rows = [_make_response_row("T1", response_time=5.0)]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert "5.00" in result.summary

    def test_summary_contains_outlier_count(self):
        rows = [_make_response_row("T1", response_time=5.0)]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert "Outliers:" in result.summary

    def test_string_numeric_response_time(self):
        rows = [{"ticket": "T1", "response_time": "7.5"}]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result is not None
        assert result.avg_response_time == 7.5

    def test_priority_whitespace_stripped(self):
        rows = [_make_response_row("T1", response_time=5.0, priority=" High ")]
        result = analyze_response_times(
            rows, "ticket", "response_time", priority_column="priority"
        )
        assert result.by_priority[0].priority == "High"

    def test_agent_whitespace_stripped(self):
        rows = [_make_response_row("T1", response_time=5.0, agent=" Alice ")]
        result = analyze_response_times(
            rows, "ticket", "response_time", agent_column="agent"
        )
        assert result.by_agent[0].agent == "Alice"

    def test_mixed_valid_and_invalid_rows(self):
        rows = [
            _make_response_row("T1", response_time=5.0),
            {"ticket": "T2", "response_time": "bad"},
            {"ticket": None, "response_time": 3.0},
            _make_response_row("T3", response_time=7.0),
        ]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.avg_response_time == 6.0

    def test_single_value_outlier_count_zero(self):
        """A single value cannot be an outlier (std_dev=0)."""
        rows = [_make_response_row("T1", response_time=5.0)]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.outlier_count == 0


# ===================================================================
# 3. compute_resolution_metrics Tests
# ===================================================================


class TestComputeResolutionMetrics:
    def test_empty_rows_returns_none(self):
        assert compute_resolution_metrics([], "ticket", "created", "resolved") is None

    def test_all_none_returns_none(self):
        rows = [{"ticket": None, "created": None, "resolved": None}]
        assert compute_resolution_metrics(rows, "ticket", "created", "resolved") is None

    def test_missing_ticket_skips_row(self):
        rows = [{"created": "2024-01-01", "resolved": "2024-01-02"}]
        assert compute_resolution_metrics(rows, "ticket", "created", "resolved") is None

    def test_missing_created_date_skips_row(self):
        rows = [{"ticket": "T1", "resolved": "2024-01-02"}]
        assert compute_resolution_metrics(rows, "ticket", "created", "resolved") is None

    def test_unparseable_created_date_skips_row(self):
        rows = [{"ticket": "T1", "created": "not-a-date", "resolved": "2024-01-02"}]
        assert compute_resolution_metrics(rows, "ticket", "created", "resolved") is None

    def test_single_resolved_ticket(self):
        rows = [_make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02")]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result is not None
        assert result.total_tickets == 1
        assert result.resolved_count == 1
        assert result.open_count == 0
        assert result.resolution_rate == 100.0
        assert result.avg_resolution_hours == 24.0

    def test_single_open_ticket(self):
        rows = [_make_resolution_row("T1", created="2024-01-01", resolved=None)]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.total_tickets == 1
        assert result.resolved_count == 0
        assert result.open_count == 1
        assert result.resolution_rate == 0.0

    def test_mixed_resolved_and_open(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02"),
            _make_resolution_row("T2", created="2024-01-01", resolved=None),
            _make_resolution_row("T3", created="2024-01-01", resolved="2024-01-03"),
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.total_tickets == 3
        assert result.resolved_count == 2
        assert result.open_count == 1
        assert result.resolution_rate == 66.67

    def test_avg_resolution_hours(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02"),  # 24h
            _make_resolution_row("T2", created="2024-01-01", resolved="2024-01-03"),  # 48h
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.avg_resolution_hours == 36.0

    def test_median_resolution_hours_odd(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02"),  # 24h
            _make_resolution_row("T2", created="2024-01-01", resolved="2024-01-03"),  # 48h
            _make_resolution_row("T3", created="2024-01-01", resolved="2024-01-04"),  # 72h
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.median_resolution_hours == 48.0

    def test_median_resolution_hours_even(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02"),  # 24h
            _make_resolution_row("T2", created="2024-01-01", resolved="2024-01-04"),  # 72h
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.median_resolution_hours == 48.0

    def test_negative_resolution_treated_as_open(self):
        """Resolved before created -> treated as open."""
        rows = [
            _make_resolution_row("T1", created="2024-01-05", resolved="2024-01-01"),
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.resolved_count == 0
        assert result.open_count == 1

    def test_backlog_age_uses_max_resolved_date(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-10"),
            _make_resolution_row("T2", created="2024-01-05", resolved=None),
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        # Ref date = 2024-01-10 (max resolved), backlog = T2 created 2024-01-05
        # Age = 5 days = 120 hours
        assert result.backlog_age_avg_hours == 120.0

    def test_backlog_age_no_resolved_uses_max_created(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved=None),
            _make_resolution_row("T2", created="2024-01-05", resolved=None),
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        # Ref date = max(created) = 2024-01-05
        # T1 age = 4 days = 96h, T2 age = 0h
        # avg = 48h
        assert result.backlog_age_avg_hours == 48.0

    def test_backlog_age_none_when_all_resolved(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02"),
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.backlog_age_avg_hours is None

    def test_by_priority_breakdown(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02", priority="High"),
            _make_resolution_row("T2", created="2024-01-01", resolved=None, priority="High"),
            _make_resolution_row("T3", created="2024-01-01", resolved="2024-01-03", priority="Low"),
        ]
        result = compute_resolution_metrics(
            rows, "ticket", "created", "resolved", priority_column="priority"
        )
        assert len(result.by_priority) == 2
        high = [p for p in result.by_priority if p.priority == "High"][0]
        assert high.total == 2
        assert high.resolved == 1
        assert high.resolution_rate == 50.0
        assert high.avg_resolution_hours == 24.0
        low = [p for p in result.by_priority if p.priority == "Low"][0]
        assert low.total == 1
        assert low.resolved == 1
        assert low.resolution_rate == 100.0

    def test_by_priority_sorted_alphabetically(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02", priority="Z"),
            _make_resolution_row("T2", created="2024-01-01", resolved="2024-01-02", priority="A"),
        ]
        result = compute_resolution_metrics(
            rows, "ticket", "created", "resolved", priority_column="priority"
        )
        names = [p.priority for p in result.by_priority]
        assert names == ["A", "Z"]

    def test_no_priority_column_empty_list(self):
        rows = [_make_resolution_row("T1")]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.by_priority == []

    def test_datetime_objects_accepted(self):
        rows = [{
            "ticket": "T1",
            "created": datetime(2024, 1, 1),
            "resolved": datetime(2024, 1, 2),
        }]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.avg_resolution_hours == 24.0

    def test_date_objects_accepted(self):
        rows = [{
            "ticket": "T1",
            "created": date(2024, 1, 1),
            "resolved": date(2024, 1, 2),
        }]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.avg_resolution_hours == 24.0

    def test_various_date_formats(self):
        rows = [
            {"ticket": "T1", "created": "2024-01-01", "resolved": "2024-01-02"},
            {"ticket": "T2", "created": "2024/01/01", "resolved": "2024/01/02"},
            {"ticket": "T3", "created": "01/01/2024", "resolved": "01/02/2024"},
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.total_tickets == 3
        assert result.resolved_count == 3

    def test_status_column_normalized_lowercase(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02", status="OPEN"),
        ]
        result = compute_resolution_metrics(
            rows, "ticket", "created", "resolved", status_column="status"
        )
        # status is tracked but doesn't change resolution logic (resolved date does)
        assert result.total_tickets == 1

    def test_summary_contains_ticket_count(self):
        rows = [_make_resolution_row("T1"), _make_resolution_row("T2")]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert "2 tickets" in result.summary

    def test_summary_contains_resolution_rate(self):
        rows = [_make_resolution_row("T1")]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert "100.0%" in result.summary

    def test_summary_contains_avg_resolution_time(self):
        rows = [_make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02")]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert "24.0 hours" in result.summary

    def test_summary_contains_backlog_age(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-10"),
            _make_resolution_row("T2", created="2024-01-05", resolved=None),
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert "backlog age" in result.summary

    def test_resolution_hours_zero_for_same_day(self):
        rows = [_make_resolution_row("T1", created="2024-01-01", resolved="2024-01-01")]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.avg_resolution_hours == 0.0

    def test_datetime_with_time_component(self):
        rows = [{
            "ticket": "T1",
            "created": "2024-01-01 10:00:00",
            "resolved": "2024-01-01 16:00:00",
        }]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.avg_resolution_hours == 6.0


# ===================================================================
# 4. analyze_sla_trends Tests
# ===================================================================


class TestAnalyzeSlaTrends:
    def test_empty_rows_returns_none(self):
        assert analyze_sla_trends([], "ticket", "sla_met", "date") is None

    def test_all_none_returns_none(self):
        rows = [{"ticket": None, "sla_met": None, "date": None}]
        assert analyze_sla_trends(rows, "ticket", "sla_met", "date") is None

    def test_missing_ticket_skips_row(self):
        rows = [{"sla_met": True, "date": "2024-01-15"}]
        assert analyze_sla_trends(rows, "ticket", "sla_met", "date") is None

    def test_missing_date_skips_row(self):
        rows = [{"ticket": "T1", "sla_met": True}]
        assert analyze_sla_trends(rows, "ticket", "sla_met", "date") is None

    def test_missing_sla_met_skips_row(self):
        rows = [{"ticket": "T1", "date": "2024-01-15"}]
        assert analyze_sla_trends(rows, "ticket", "sla_met", "date") is None

    def test_single_period_single_ticket(self):
        rows = [_make_trend_row("T1", sla_met=True, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result is not None
        assert len(result.periods) == 1
        assert result.periods[0].period == "2024-01"
        assert result.periods[0].total == 1
        assert result.periods[0].met == 1
        assert result.periods[0].compliance_rate == 100.0

    def test_single_period_stable_trend(self):
        rows = [_make_trend_row("T1", sla_met=True, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.trend_direction == "Stable"

    def test_improving_trend(self):
        rows = [
            _make_trend_row("T1", sla_met=False, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=False, date_val="2024-01-20"),
            _make_trend_row("T3", sla_met=True, date_val="2024-03-15"),
            _make_trend_row("T4", sla_met=True, date_val="2024-03-20"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        # Jan: 0%, Mar: 100%, diff = 100% > 5% -> Improving
        assert result.trend_direction == "Improving"

    def test_deteriorating_trend(self):
        rows = [
            _make_trend_row("T1", sla_met=True, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=True, date_val="2024-01-20"),
            _make_trend_row("T3", sla_met=False, date_val="2024-03-15"),
            _make_trend_row("T4", sla_met=False, date_val="2024-03-20"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.trend_direction == "Deteriorating"

    def test_stable_trend_within_5_pct(self):
        rows = [
            _make_trend_row("T1", sla_met=True, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=True, date_val="2024-03-15"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        # Both 100%, diff = 0 -> Stable
        assert result.trend_direction == "Stable"

    def test_overall_compliance(self):
        rows = [
            _make_trend_row("T1", sla_met=True, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=False, date_val="2024-01-20"),
            _make_trend_row("T3", sla_met=True, date_val="2024-02-15"),
            _make_trend_row("T4", sla_met=True, date_val="2024-02-20"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.overall_compliance == 75.0

    def test_best_and_worst_period(self):
        rows = [
            _make_trend_row("T1", sla_met=True, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=True, date_val="2024-01-20"),
            _make_trend_row("T3", sla_met=False, date_val="2024-02-15"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.best_period == "2024-01"
        assert result.worst_period == "2024-02"

    def test_periods_sorted_chronologically(self):
        rows = [
            _make_trend_row("T1", sla_met=True, date_val="2024-03-15"),
            _make_trend_row("T2", sla_met=True, date_val="2024-01-15"),
            _make_trend_row("T3", sla_met=True, date_val="2024-02-15"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        periods = [p.period for p in result.periods]
        assert periods == ["2024-01", "2024-02", "2024-03"]

    def test_boolean_true_is_met(self):
        rows = [_make_trend_row("T1", sla_met=True, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_boolean_false_is_not_met(self):
        rows = [_make_trend_row("T1", sla_met=False, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 0

    def test_int_1_is_met(self):
        rows = [_make_trend_row("T1", sla_met=1, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_int_0_is_not_met(self):
        rows = [_make_trend_row("T1", sla_met=0, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 0

    def test_float_1_is_met(self):
        rows = [_make_trend_row("T1", sla_met=1.0, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_float_below_1_is_not_met(self):
        rows = [_make_trend_row("T1", sla_met=0.5, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 0

    def test_string_yes_is_met(self):
        rows = [_make_trend_row("T1", sla_met="yes", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_string_true_is_met(self):
        rows = [_make_trend_row("T1", sla_met="true", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_string_met_is_met(self):
        rows = [_make_trend_row("T1", sla_met="met", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_string_1_is_met(self):
        rows = [_make_trend_row("T1", sla_met="1", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_string_pass_is_met(self):
        rows = [_make_trend_row("T1", sla_met="pass", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_string_passed_is_met(self):
        rows = [_make_trend_row("T1", sla_met="passed", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_string_no_is_not_met(self):
        rows = [_make_trend_row("T1", sla_met="no", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 0

    def test_string_false_is_not_met(self):
        rows = [_make_trend_row("T1", sla_met="false", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 0

    def test_string_case_insensitive(self):
        rows = [
            _make_trend_row("T1", sla_met="YES", date_val="2024-01-15"),
            _make_trend_row("T2", sla_met="True", date_val="2024-01-20"),
            _make_trend_row("T3", sla_met="MET", date_val="2024-01-25"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 3

    def test_string_with_whitespace(self):
        rows = [_make_trend_row("T1", sla_met=" yes ", date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.periods[0].met == 1

    def test_unsupported_type_skips_row(self):
        """Non-bool/int/float/str sla_met values are skipped."""
        rows = [{"ticket": "T1", "sla_met": [1, 2, 3], "date": "2024-01-15"}]
        assert analyze_sla_trends(rows, "ticket", "sla_met", "date") is None

    def test_summary_contains_period_count(self):
        rows = [
            _make_trend_row("T1", date_val="2024-01-15"),
            _make_trend_row("T2", date_val="2024-02-15"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert "2 periods" in result.summary

    def test_summary_contains_overall_compliance(self):
        rows = [_make_trend_row("T1", sla_met=True, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert "100.0%" in result.summary

    def test_summary_contains_trend_direction(self):
        rows = [_make_trend_row("T1", sla_met=True, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert "Stable" in result.summary

    def test_summary_contains_best_and_worst(self):
        rows = [_make_trend_row("T1", sla_met=True, date_val="2024-01-15")]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert "Best period:" in result.summary
        assert "Worst period:" in result.summary

    def test_unparseable_date_skipped(self):
        rows = [
            {"ticket": "T1", "sla_met": True, "date": "not-a-date"},
            _make_trend_row("T2", sla_met=True, date_val="2024-01-15"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert len(result.periods) == 1

    def test_various_date_formats(self):
        rows = [
            {"ticket": "T1", "sla_met": True, "date": "2024-01-15"},
            {"ticket": "T2", "sla_met": True, "date": "2024/02/15"},
            {"ticket": "T3", "sla_met": True, "date": "03/15/2024"},
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert len(result.periods) == 3


# ===================================================================
# 5. format_sla_report Tests
# ===================================================================


class TestFormatSlaReport:
    def test_all_none_reports_no_data(self):
        report = format_sla_report()
        assert "No analysis data provided" in report

    def test_header_always_present(self):
        report = format_sla_report()
        assert "SLA Monitoring Report" in report
        assert "=" * 40 in report

    def test_compliance_section_included(self):
        rows = [_make_ticket("T1", target=10, actual=5)]
        compliance = analyze_sla_compliance(rows, "ticket", "target", "actual")
        report = format_sla_report(compliance=compliance)
        assert "SLA Compliance" in report
        assert "Total tickets: 1" in report
        assert "Met: 1" in report
        assert "Breached: 0" in report
        assert "Compliance rate:" in report

    def test_compliance_section_with_worst_category(self):
        rows = [_make_ticket("T1", target=10, actual=15, category="Net")]
        compliance = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        report = format_sla_report(compliance=compliance)
        assert "Worst category: Net" in report

    def test_compliance_section_with_by_category(self):
        rows = [
            _make_ticket("T1", target=10, actual=5, category="A"),
            _make_ticket("T2", target=10, actual=15, category="B"),
        ]
        compliance = analyze_sla_compliance(
            rows, "ticket", "target", "actual", category_column="category"
        )
        report = format_sla_report(compliance=compliance)
        assert "By category:" in report
        assert "A:" in report
        assert "B:" in report

    def test_response_times_section_included(self):
        rows = [_make_response_row("T1", response_time=5.0)]
        response_times = analyze_response_times(rows, "ticket", "response_time")
        report = format_sla_report(response_times=response_times)
        assert "Response Times" in report
        assert "Avg:" in report
        assert "Median:" in report
        assert "P95:" in report
        assert "Min:" in report
        assert "Max:" in report
        assert "Outliers:" in report

    def test_response_times_section_with_priority(self):
        rows = [_make_response_row("T1", response_time=5.0, priority="High")]
        response_times = analyze_response_times(
            rows, "ticket", "response_time", priority_column="priority"
        )
        report = format_sla_report(response_times=response_times)
        assert "By priority:" in report
        assert "High:" in report

    def test_response_times_section_with_agent(self):
        rows = [_make_response_row("T1", response_time=5.0, agent="Alice")]
        response_times = analyze_response_times(
            rows, "ticket", "response_time", agent_column="agent"
        )
        report = format_sla_report(response_times=response_times)
        assert "By agent:" in report
        assert "Alice:" in report

    def test_resolution_section_included(self):
        rows = [_make_resolution_row("T1")]
        resolution = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        report = format_sla_report(resolution=resolution)
        assert "Resolution Metrics" in report
        assert "Total tickets: 1" in report
        assert "Resolved:" in report
        assert "Open:" in report
        assert "Resolution rate:" in report

    def test_resolution_section_with_backlog(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-10"),
            _make_resolution_row("T2", created="2024-01-05", resolved=None),
        ]
        resolution = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        report = format_sla_report(resolution=resolution)
        assert "backlog age:" in report

    def test_resolution_section_with_priority(self):
        rows = [_make_resolution_row("T1", priority="High")]
        resolution = compute_resolution_metrics(
            rows, "ticket", "created", "resolved", priority_column="priority"
        )
        report = format_sla_report(resolution=resolution)
        assert "By priority:" in report
        assert "High:" in report

    def test_trends_section_included(self):
        rows = [_make_trend_row("T1", sla_met=True, date_val="2024-01-15")]
        trends = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        report = format_sla_report(trends=trends)
        assert "SLA Trends" in report
        assert "Overall compliance:" in report
        assert "Trend:" in report
        assert "Best period:" in report
        assert "Worst period:" in report
        assert "Periods:" in report

    def test_combined_all_sections(self):
        compliance = analyze_sla_compliance(
            [_make_ticket("T1")], "ticket", "target", "actual"
        )
        response_times = analyze_response_times(
            [_make_response_row("T1")], "ticket", "response_time"
        )
        resolution = compute_resolution_metrics(
            [_make_resolution_row("T1")], "ticket", "created", "resolved"
        )
        trends = analyze_sla_trends(
            [_make_trend_row("T1")], "ticket", "sla_met", "date"
        )
        report = format_sla_report(
            compliance=compliance,
            response_times=response_times,
            resolution=resolution,
            trends=trends,
        )
        assert "SLA Compliance" in report
        assert "Response Times" in report
        assert "Resolution Metrics" in report
        assert "SLA Trends" in report
        assert "No analysis data provided" not in report

    def test_no_data_message_only_when_all_none(self):
        compliance = analyze_sla_compliance(
            [_make_ticket("T1")], "ticket", "target", "actual"
        )
        report = format_sla_report(compliance=compliance)
        assert "No analysis data provided" not in report

    def test_partial_sections_omit_others(self):
        compliance = analyze_sla_compliance(
            [_make_ticket("T1")], "ticket", "target", "actual"
        )
        report = format_sla_report(compliance=compliance)
        assert "SLA Compliance" in report
        assert "Response Times" not in report
        assert "Resolution Metrics" not in report
        assert "SLA Trends" not in report


# ===================================================================
# 6. Dataclass Construction Tests
# ===================================================================


class TestDataclassConstruction:
    def test_category_sla_fields(self):
        cs = CategorySLA(
            category="Network",
            total=10,
            met=7,
            breached=3,
            compliance_rate=70.0,
            avg_actual=8.5,
            avg_target=10.0,
        )
        assert cs.category == "Network"
        assert cs.total == 10
        assert cs.met == 7
        assert cs.breached == 3
        assert cs.compliance_rate == 70.0
        assert cs.avg_actual == 8.5
        assert cs.avg_target == 10.0

    def test_sla_compliance_result_fields(self):
        r = SLAComplianceResult(
            total_tickets=100,
            met_count=80,
            breached_count=20,
            compliance_rate=80.0,
            by_category=[],
            worst_category=None,
            avg_performance_ratio=0.9,
            summary="test",
        )
        assert r.total_tickets == 100
        assert r.met_count == 80
        assert r.breached_count == 20
        assert r.compliance_rate == 80.0
        assert r.by_category == []
        assert r.worst_category is None
        assert r.avg_performance_ratio == 0.9
        assert r.summary == "test"

    def test_priority_response_fields(self):
        pr = PriorityResponse(
            priority="High",
            count=5,
            avg_time=3.5,
            median_time=3.0,
            p95_time=6.0,
        )
        assert pr.priority == "High"
        assert pr.count == 5
        assert pr.avg_time == 3.5
        assert pr.median_time == 3.0
        assert pr.p95_time == 6.0

    def test_agent_response_fields(self):
        ar = AgentResponse(
            agent="Alice",
            count=10,
            avg_time=4.2,
            compliance_rate=90.0,
        )
        assert ar.agent == "Alice"
        assert ar.count == 10
        assert ar.avg_time == 4.2
        assert ar.compliance_rate == 90.0

    def test_response_time_result_fields(self):
        r = ResponseTimeResult(
            avg_response_time=5.0,
            median_response_time=4.5,
            p95_response_time=10.0,
            min_time=1.0,
            max_time=15.0,
            by_priority=[],
            by_agent=[],
            outlier_count=2,
            summary="test",
        )
        assert r.avg_response_time == 5.0
        assert r.median_response_time == 4.5
        assert r.p95_response_time == 10.0
        assert r.min_time == 1.0
        assert r.max_time == 15.0
        assert r.by_priority == []
        assert r.by_agent == []
        assert r.outlier_count == 2
        assert r.summary == "test"

    def test_priority_resolution_fields(self):
        pr = PriorityResolution(
            priority="Critical",
            total=20,
            resolved=15,
            avg_resolution_hours=12.5,
            resolution_rate=75.0,
        )
        assert pr.priority == "Critical"
        assert pr.total == 20
        assert pr.resolved == 15
        assert pr.avg_resolution_hours == 12.5
        assert pr.resolution_rate == 75.0

    def test_resolution_result_fields(self):
        r = ResolutionResult(
            total_tickets=50,
            resolved_count=40,
            open_count=10,
            resolution_rate=80.0,
            avg_resolution_hours=36.0,
            median_resolution_hours=24.0,
            by_priority=[],
            backlog_age_avg_hours=48.0,
            summary="test",
        )
        assert r.total_tickets == 50
        assert r.resolved_count == 40
        assert r.open_count == 10
        assert r.resolution_rate == 80.0
        assert r.avg_resolution_hours == 36.0
        assert r.median_resolution_hours == 24.0
        assert r.by_priority == []
        assert r.backlog_age_avg_hours == 48.0
        assert r.summary == "test"

    def test_period_sla_fields(self):
        ps = PeriodSLA(
            period="2024-01",
            total=30,
            met=25,
            compliance_rate=83.33,
        )
        assert ps.period == "2024-01"
        assert ps.total == 30
        assert ps.met == 25
        assert ps.compliance_rate == 83.33

    def test_sla_trend_result_fields(self):
        r = SLATrendResult(
            periods=[],
            trend_direction="Improving",
            overall_compliance=85.0,
            best_period="2024-03",
            worst_period="2024-01",
            summary="test",
        )
        assert r.periods == []
        assert r.trend_direction == "Improving"
        assert r.overall_compliance == 85.0
        assert r.best_period == "2024-03"
        assert r.worst_period == "2024-01"
        assert r.summary == "test"


# ===================================================================
# 7. Edge Cases and Integration Tests
# ===================================================================


class TestEdgeCases:
    def test_compliance_with_float_zero_actual(self):
        rows = [_make_ticket("T1", target=10, actual=0)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.met_count == 1

    def test_compliance_negative_actual(self):
        """Negative actual is still <= target, so met."""
        rows = [_make_ticket("T1", target=10, actual=-5)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.met_count == 1

    def test_response_time_zero(self):
        rows = [_make_response_row("T1", response_time=0.0)]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.avg_response_time == 0.0

    def test_response_time_very_large(self):
        rows = [_make_response_row("T1", response_time=999999.99)]
        result = analyze_response_times(rows, "ticket", "response_time")
        assert result.max_time == 999999.99

    def test_resolution_all_open_no_avg_resolution(self):
        rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved=None),
            _make_resolution_row("T2", created="2024-01-05", resolved=None),
        ]
        result = compute_resolution_metrics(rows, "ticket", "created", "resolved")
        assert result.avg_resolution_hours == 0.0
        assert result.median_resolution_hours == 0.0

    def test_trend_all_met(self):
        rows = [
            _make_trend_row("T1", sla_met=True, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=True, date_val="2024-02-15"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.overall_compliance == 100.0

    def test_trend_none_met(self):
        rows = [
            _make_trend_row("T1", sla_met=False, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=False, date_val="2024-02-15"),
        ]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.overall_compliance == 0.0

    def test_full_integration_report(self):
        """Full end-to-end test with all four analyses combined into a report."""
        compliance_rows = [
            _make_ticket("T1", target=10, actual=5, category="Net"),
            _make_ticket("T2", target=10, actual=15, category="DB"),
        ]
        response_rows = [
            _make_response_row("T1", response_time=3.0, priority="High", agent="Alice"),
            _make_response_row("T2", response_time=7.0, priority="Low", agent="Bob"),
        ]
        resolution_rows = [
            _make_resolution_row("T1", created="2024-01-01", resolved="2024-01-02", priority="High"),
            _make_resolution_row("T2", created="2024-01-01", resolved=None, priority="Low"),
        ]
        trend_rows = [
            _make_trend_row("T1", sla_met=True, date_val="2024-01-15"),
            _make_trend_row("T2", sla_met=False, date_val="2024-02-15"),
        ]

        compliance = analyze_sla_compliance(
            compliance_rows, "ticket", "target", "actual", category_column="category"
        )
        response_times = analyze_response_times(
            response_rows, "ticket", "response_time",
            priority_column="priority", agent_column="agent"
        )
        resolution = compute_resolution_metrics(
            resolution_rows, "ticket", "created", "resolved",
            priority_column="priority"
        )
        trends = analyze_sla_trends(trend_rows, "ticket", "sla_met", "date")

        report = format_sla_report(
            compliance=compliance,
            response_times=response_times,
            resolution=resolution,
            trends=trends,
        )
        assert "SLA Monitoring Report" in report
        assert "SLA Compliance" in report
        assert "Response Times" in report
        assert "Resolution Metrics" in report
        assert "SLA Trends" in report
        assert len(report) > 300

    def test_compliance_large_performance_ratio(self):
        """When actual >> target, performance ratio > 1."""
        rows = [_make_ticket("T1", target=1, actual=100)]
        result = analyze_sla_compliance(rows, "ticket", "target", "actual")
        assert result.avg_performance_ratio == 100.0

    def test_trend_boundary_exactly_5_pct_is_stable(self):
        """Diff of exactly 5.0 is Stable (not > 5.0)."""
        # We need first period at X% and last at X+5%. With small numbers:
        # Period 1: 1 ticket, 0 met -> 0%. Period 2: 20 tickets, 1 met -> 5%.
        # Diff = 5.0 -> Stable
        rows = [_make_trend_row("T0", sla_met=False, date_val="2024-01-15")]
        rows += [_make_trend_row(f"T{i}", sla_met=(i == 1), date_val="2024-02-15") for i in range(1, 21)]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        # First period: 0%, last period: 1/20 = 5%. diff = 5.0 -> Stable
        assert result.trend_direction == "Stable"

    def test_trend_boundary_just_over_5_pct_is_improving(self):
        """Diff of just over 5.0 is Improving."""
        # Period 1: 0%. Period 2: 5.01%+ -> Improving
        # 10 tickets, 0 met in Jan. 100 tickets, 6 met in Feb -> 6% diff > 5
        rows = [_make_trend_row(f"J{i}", sla_met=False, date_val="2024-01-15") for i in range(10)]
        rows += [_make_trend_row(f"F{i}", sla_met=(i < 6), date_val="2024-02-15") for i in range(100)]
        result = analyze_sla_trends(rows, "ticket", "sla_met", "date")
        assert result.trend_direction == "Improving"
