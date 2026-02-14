"""Tests for pipeline analyzer module."""

from __future__ import annotations

from datetime import datetime

from business_brain.discovery.pipeline_analyzer import (
    OwnerWinRate,
    PipelineForecastResult,
    PipelineStageResult,
    PipelineVelocityResult,
    StageForecast,
    StageConversion,
    StageMetrics,
    StageVelocity,
    WinRateResult,
    _canonical_order,
    _is_lost,
    _is_won,
    _parse_date,
    _safe_float,
    analyze_pipeline_stages,
    analyze_pipeline_velocity,
    compute_win_rate,
    forecast_pipeline,
    format_pipeline_report,
)


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def _make_deal(
    deal: str,
    stage: str,
    value: float,
    owner: str | None = None,
    date: str | None = None,
    close_date: str | None = None,
    probability: float | None = None,
) -> dict:
    """Build a single deal row dict."""
    row: dict = {"deal": deal, "stage": stage, "value": value}
    if owner is not None:
        row["owner"] = owner
    if date is not None:
        row["date"] = date
    if close_date is not None:
        row["close_date"] = close_date
    if probability is not None:
        row["probability"] = probability
    return row


def _make_deals(specs: list[tuple]) -> list[dict]:
    """Build multiple deal rows from (deal, stage, value, ...) tuples.

    Each tuple can have 3-7 elements: deal, stage, value, [owner], [date],
    [close_date], [probability].
    """
    rows = []
    for spec in specs:
        keys = ["deal", "stage", "value", "owner", "date", "close_date", "probability"]
        kw = {}
        for i, val in enumerate(spec):
            if i < len(keys):
                kw[keys[i]] = val
        rows.append(_make_deal(**kw))
    return rows


# ---------------------------------------------------------------------------
# Helper function tests
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

    def test_invalid_string(self):
        assert _safe_float("abc") is None

    def test_empty_string(self):
        assert _safe_float("") is None

    def test_negative(self):
        assert _safe_float(-7.5) == -7.5

    def test_zero(self):
        assert _safe_float(0) == 0.0

    def test_string_zero(self):
        assert _safe_float("0") == 0.0

    def test_bool_true(self):
        # bool is subclass of int, float(True) == 1.0
        assert _safe_float(True) == 1.0

    def test_list_returns_none(self):
        assert _safe_float([1, 2]) is None


class TestParseDate:
    def test_none(self):
        assert _parse_date(None) is None

    def test_datetime_passthrough(self):
        dt = datetime(2024, 6, 15)
        assert _parse_date(dt) is dt

    def test_iso_format(self):
        result = _parse_date("2024-06-15")
        assert result == datetime(2024, 6, 15)

    def test_slash_format(self):
        result = _parse_date("2024/06/15")
        assert result == datetime(2024, 6, 15)

    def test_us_format(self):
        result = _parse_date("06/15/2024")
        assert result == datetime(2024, 6, 15)

    def test_datetime_string(self):
        result = _parse_date("2024-06-15 14:30:00")
        assert result == datetime(2024, 6, 15, 14, 30, 0)

    def test_invalid_string(self):
        assert _parse_date("not-a-date") is None

    def test_empty_string(self):
        assert _parse_date("") is None

    def test_numeric_input(self):
        # Numbers should fail gracefully
        assert _parse_date(12345) is None


class TestIsWon:
    def test_won(self):
        assert _is_won("Won") is True

    def test_closed_won(self):
        assert _is_won("Closed Won") is True

    def test_closed_won_hyphen(self):
        assert _is_won("closed-won") is True

    def test_closed_won_underscore(self):
        assert _is_won("closed_won") is True

    def test_win(self):
        assert _is_won("Win") is True

    def test_closed_win(self):
        assert _is_won("Closed Win") is True

    def test_lost_not_won(self):
        assert _is_won("Lost") is False

    def test_negotiation_not_won(self):
        assert _is_won("Negotiation") is False

    def test_case_insensitive(self):
        assert _is_won("WON") is True
        assert _is_won("won") is True

    def test_empty_string(self):
        assert _is_won("") is False


class TestIsLost:
    def test_lost(self):
        assert _is_lost("Lost") is True

    def test_closed_lost(self):
        assert _is_lost("Closed Lost") is True

    def test_closed_lost_hyphen(self):
        assert _is_lost("closed-lost") is True

    def test_closed_lost_underscore(self):
        assert _is_lost("closed_lost") is True

    def test_lose(self):
        assert _is_lost("Lose") is True

    def test_closed_lose(self):
        assert _is_lost("Closed Lose") is True

    def test_won_not_lost(self):
        assert _is_lost("Won") is False

    def test_negotiation_not_lost(self):
        assert _is_lost("Negotiation") is False

    def test_case_insensitive(self):
        assert _is_lost("LOST") is True
        assert _is_lost("lost") is True

    def test_empty_string(self):
        assert _is_lost("") is False


class TestCanonicalOrder:
    def test_lead(self):
        assert _canonical_order("Lead") == 0

    def test_qualified(self):
        assert _canonical_order("Qualified") == 1

    def test_proposal(self):
        assert _canonical_order("Proposal") == 2

    def test_negotiation(self):
        assert _canonical_order("Negotiation") == 3

    def test_won(self):
        assert _canonical_order("Won") == 4

    def test_lost(self):
        assert _canonical_order("Lost") == 5

    def test_case_insensitive(self):
        assert _canonical_order("LEAD") == 0
        assert _canonical_order("proposal") == 2

    def test_leading_trailing_spaces(self):
        assert _canonical_order("  Lead  ") == 0

    def test_unknown_stage(self):
        assert _canonical_order("Discovery") is None

    def test_empty_string(self):
        # empty string is "in" every string, so it matches the first canonical
        assert _canonical_order("") is not None


# ---------------------------------------------------------------------------
# analyze_pipeline_stages
# ---------------------------------------------------------------------------


class TestAnalyzePipelineStages:
    def test_empty_rows(self):
        assert analyze_pipeline_stages([], "deal", "stage", "value") is None

    def test_all_missing_columns(self):
        rows = [{"other": "val"}]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is None

    def test_missing_deal_column(self):
        rows = [{"stage": "Lead", "value": 100}]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is None

    def test_missing_stage_column(self):
        rows = [{"deal": "D1", "value": 100}]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is None

    def test_missing_value_column(self):
        rows = [{"deal": "D1", "stage": "Lead"}]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is None

    def test_non_numeric_value(self):
        rows = [{"deal": "D1", "stage": "Lead", "value": "abc"}]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is None

    def test_none_value(self):
        rows = [{"deal": "D1", "stage": "Lead", "value": None}]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is None

    def test_single_deal(self):
        rows = [_make_deal("D1", "Lead", 1000)]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_deals == 1
        assert result.total_value == 1000.0
        assert len(result.stages) == 1
        assert result.stages[0].stage == "Lead"
        assert result.stages[0].deal_count == 1

    def test_two_deals_same_stage(self):
        rows = [
            _make_deal("D1", "Lead", 1000),
            _make_deal("D2", "Lead", 2000),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_deals == 2
        assert result.total_value == 3000.0
        assert len(result.stages) == 1
        assert result.stages[0].avg_value == 1500.0

    def test_multiple_stages_canonical_order(self):
        rows = [
            _make_deal("D1", "Lead", 500),
            _make_deal("D2", "Qualified", 800),
            _make_deal("D3", "Proposal", 1200),
            _make_deal("D4", "Won", 2000),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        stage_names = [s.stage for s in result.stages]
        assert stage_names == ["Lead", "Qualified", "Proposal", "Won"]

    def test_non_canonical_stages_sorted_by_count(self):
        rows = [
            _make_deal("D1", "Alpha", 100),
            _make_deal("D2", "Alpha", 200),
            _make_deal("D3", "Alpha", 150),
            _make_deal("D4", "Beta", 300),
            _make_deal("D5", "Beta", 250),
            _make_deal("D6", "Gamma", 400),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        stage_names = [s.stage for s in result.stages]
        # Alpha has 3 deals, Beta 2, Gamma 1 -> sorted desc by count
        assert stage_names == ["Alpha", "Beta", "Gamma"]

    def test_pct_of_deals(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Lead", 200),
            _make_deal("D3", "Won", 300),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        lead = next(s for s in result.stages if s.stage == "Lead")
        won = next(s for s in result.stages if s.stage == "Won")
        assert abs(lead.pct_of_deals - 66.67) < 0.01
        assert abs(won.pct_of_deals - 33.33) < 0.01

    def test_pct_of_value(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Won", 400),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        lead = next(s for s in result.stages if s.stage == "Lead")
        won = next(s for s in result.stages if s.stage == "Won")
        assert lead.pct_of_value == 20.0
        assert won.pct_of_value == 80.0

    def test_conversions_between_stages(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Lead", 200),
            _make_deal("D3", "Qualified", 300),
            _make_deal("D4", "Won", 500),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert len(result.conversions) == 2
        # Lead -> Qualified: 1/2 = 50%
        lead_to_qual = result.conversions[0]
        assert lead_to_qual.from_stage == "Lead"
        assert lead_to_qual.to_stage == "Qualified"
        assert lead_to_qual.conversion_rate == 50.0

    def test_single_stage_no_conversions(self):
        rows = [_make_deal("D1", "Lead", 100)]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert len(result.conversions) == 0

    def test_weighted_value(self):
        rows = [
            _make_deal("D1", "Lead", 1000),
            _make_deal("D2", "Won", 2000),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        # Lead is stage 0, Won is stage 1. n_stages=2.
        # Lead weight = 1/2=0.5, Won weight = 2/2=1.0
        # weighted = 1000*0.5 + 2000*1.0 = 500 + 2000 = 2500
        assert result.weighted_value == 2500.0

    def test_summary_contains_total_deals(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Won", 200),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert "2 deals" in result.summary

    def test_summary_contains_stage_count(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Won", 200),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert "2 stages" in result.summary

    def test_duplicate_deal_takes_last_stage(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D1", "Qualified", 100),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_deals == 1
        assert result.stages[0].stage == "Qualified"

    def test_duplicate_deal_takes_max_value(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D1", "Lead", 500),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_value == 500.0

    def test_owner_column_collected(self):
        rows = [
            _make_deal("D1", "Lead", 100, owner="Alice"),
            _make_deal("D2", "Lead", 200, owner="Bob"),
        ]
        result = analyze_pipeline_stages(
            rows, "deal", "stage", "value", owner_column="owner"
        )
        assert result is not None
        assert result.total_deals == 2

    def test_string_value_conversion(self):
        rows = [{"deal": "D1", "stage": "Lead", "value": "1000"}]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_value == 1000.0

    def test_returns_pipeline_stage_result_type(self):
        rows = [_make_deal("D1", "Lead", 100)]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert isinstance(result, PipelineStageResult)

    def test_stage_metrics_rounding(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Lead", 200),
            _make_deal("D3", "Lead", 300),
        ]
        result = analyze_pipeline_stages(rows, "deal", "stage", "value")
        assert result is not None
        assert result.stages[0].avg_value == 200.0
        assert result.stages[0].total_value == 600.0


# ---------------------------------------------------------------------------
# analyze_pipeline_velocity
# ---------------------------------------------------------------------------


class TestAnalyzePipelineVelocity:
    def test_empty_rows(self):
        assert analyze_pipeline_velocity([], "deal", "stage", "value", "date") is None

    def test_all_missing_columns(self):
        rows = [{"other": "val"}]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is None

    def test_missing_date_column(self):
        rows = [_make_deal("D1", "Lead", 100)]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is None

    def test_single_stage_per_deal_no_velocity(self):
        # Only one stage per deal means no transitions, no velocity
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D2", "Lead", 200, date="2024-01-05"),
        ]
        # Each deal has only 1 stage, so no durations and possibly
        # cycle days from date range
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        # With only 1 stage per deal and no close date, no cycle days either
        assert result is None

    def test_two_stages_single_deal(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Qualified", 100, date="2024-01-11"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        assert len(result.stage_velocities) >= 1
        # Lead -> Qualified = 10 days
        lead_vel = next(
            (sv for sv in result.stage_velocities if sv.stage == "Lead"), None
        )
        assert lead_vel is not None
        assert lead_vel.avg_days == 10.0

    def test_multiple_deals_multiple_stages(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Qualified", 100, date="2024-01-11"),
            _make_deal("D1", "Won", 100, date="2024-01-21"),
            _make_deal("D2", "Lead", 200, date="2024-02-01"),
            _make_deal("D2", "Qualified", 200, date="2024-02-06"),
            _make_deal("D2", "Won", 200, date="2024-02-16"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        # D1: Lead 10 days, Qualified 10 days. D2: Lead 5 days, Qualified 10 days.
        lead_vel = next(
            (sv for sv in result.stage_velocities if sv.stage == "Lead"), None
        )
        assert lead_vel is not None
        assert lead_vel.avg_days == 7.5  # (10+5)/2

    def test_bottleneck_stage_is_slowest(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Qualified", 100, date="2024-01-05"),
            _make_deal("D1", "Proposal", 100, date="2024-01-25"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        # Lead = 4 days, Qualified = 20 days
        assert result.bottleneck_stage == "Qualified"

    def test_cycle_days_with_close_date(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01", close_date="2024-01-31"),
            _make_deal("D1", "Won", 100, date="2024-01-15", close_date="2024-01-31"),
        ]
        result = analyze_pipeline_velocity(
            rows, "deal", "stage", "value", "date", close_date_column="close_date"
        )
        assert result is not None
        assert result.avg_cycle_days == 30.0  # Jan 1 to Jan 31
        assert result.fastest_deal_days == 30.0
        assert result.slowest_deal_days == 30.0

    def test_fastest_and_slowest_deal(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Won", 100, date="2024-01-11"),
            _make_deal("D2", "Lead", 200, date="2024-02-01"),
            _make_deal("D2", "Won", 200, date="2024-02-28"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        assert result.fastest_deal_days == 10.0
        assert result.slowest_deal_days == 27.0

    def test_summary_mentions_avg_cycle(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Won", 100, date="2024-01-11"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        assert "avg cycle" in result.summary.lower()

    def test_summary_mentions_bottleneck(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Qualified", 100, date="2024-01-15"),
            _make_deal("D1", "Won", 100, date="2024-01-20"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        assert "Bottleneck" in result.summary

    def test_non_numeric_value_skipped(self):
        rows = [
            {"deal": "D1", "stage": "Lead", "value": "abc", "date": "2024-01-01"},
            _make_deal("D2", "Lead", 100, date="2024-02-01"),
            _make_deal("D2", "Won", 100, date="2024-02-15"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        # Only D2 should be included

    def test_invalid_date_skipped(self):
        rows = [
            {"deal": "D1", "stage": "Lead", "value": 100, "date": "not-a-date"},
            _make_deal("D2", "Lead", 200, date="2024-01-01"),
            _make_deal("D2", "Won", 200, date="2024-01-20"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None

    def test_returns_velocity_result_type(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Won", 100, date="2024-01-11"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert isinstance(result, PipelineVelocityResult)

    def test_stage_velocities_sorted_desc_by_avg_days(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Qualified", 100, date="2024-01-03"),
            _make_deal("D1", "Proposal", 100, date="2024-01-20"),
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        if len(result.stage_velocities) >= 2:
            for i in range(len(result.stage_velocities) - 1):
                assert (
                    result.stage_velocities[i].avg_days
                    >= result.stage_velocities[i + 1].avg_days
                )

    def test_datetime_objects_in_date_column(self):
        rows = [
            {"deal": "D1", "stage": "Lead", "value": 100, "date": datetime(2024, 1, 1)},
            {"deal": "D1", "stage": "Won", "value": 100, "date": datetime(2024, 1, 15)},
        ]
        result = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        assert result is not None
        assert result.avg_cycle_days == 14.0


# ---------------------------------------------------------------------------
# compute_win_rate
# ---------------------------------------------------------------------------


class TestComputeWinRate:
    def test_empty_rows(self):
        assert compute_win_rate([], "deal", "stage") is None

    def test_all_missing_columns(self):
        rows = [{"other": "val"}]
        result = compute_win_rate(rows, "deal", "stage")
        assert result is None

    def test_no_won_or_lost_deals(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Qualified", 200),
        ]
        result = compute_win_rate(rows, "deal", "stage")
        assert result is None

    def test_all_won(self):
        rows = [
            _make_deal("D1", "Won", 100),
            _make_deal("D2", "Won", 200),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        assert result.overall_win_rate == 100.0
        assert result.total_won == 2
        assert result.total_lost == 0
        assert result.won_value == 300.0
        assert result.lost_value == 0.0

    def test_all_lost(self):
        rows = [
            _make_deal("D1", "Lost", 100),
            _make_deal("D2", "Lost", 200),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        assert result.overall_win_rate == 0.0
        assert result.total_won == 0
        assert result.total_lost == 2
        assert result.lost_value == 300.0

    def test_mixed_won_lost(self):
        rows = [
            _make_deal("D1", "Won", 1000),
            _make_deal("D2", "Lost", 500),
            _make_deal("D3", "Won", 2000),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        assert abs(result.overall_win_rate - 66.67) < 0.01
        assert result.total_won == 2
        assert result.total_lost == 1

    def test_closed_won_recognized(self):
        rows = [
            _make_deal("D1", "Closed Won", 1000),
            _make_deal("D2", "Closed Lost", 500),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        assert result.total_won == 1
        assert result.total_lost == 1
        assert result.overall_win_rate == 50.0

    def test_by_owner(self):
        rows = [
            _make_deal("D1", "Won", 1000, owner="Alice"),
            _make_deal("D2", "Won", 2000, owner="Alice"),
            _make_deal("D3", "Lost", 500, owner="Alice"),
            _make_deal("D4", "Won", 800, owner="Bob"),
            _make_deal("D5", "Lost", 300, owner="Bob"),
            _make_deal("D6", "Lost", 400, owner="Bob"),
        ]
        result = compute_win_rate(
            rows, "deal", "stage", value_column="value", owner_column="owner"
        )
        assert result is not None
        assert len(result.by_owner) == 2

        alice = next(o for o in result.by_owner if o.owner == "Alice")
        assert alice.won == 2
        assert alice.lost == 1
        assert abs(alice.win_rate - 66.67) < 0.01
        assert alice.won_value == 3000.0

        bob = next(o for o in result.by_owner if o.owner == "Bob")
        assert bob.won == 1
        assert bob.lost == 2
        assert abs(bob.win_rate - 33.33) < 0.01

    def test_best_performer(self):
        rows = [
            _make_deal("D1", "Won", 1000, owner="Alice"),
            _make_deal("D2", "Lost", 500, owner="Alice"),
            _make_deal("D3", "Won", 800, owner="Bob"),
        ]
        result = compute_win_rate(
            rows, "deal", "stage", value_column="value", owner_column="owner"
        )
        assert result is not None
        # Bob has 100% win rate, Alice has 50%
        assert result.best_performer == "Bob"

    def test_no_owner_column(self):
        rows = [
            _make_deal("D1", "Won", 1000),
            _make_deal("D2", "Lost", 500),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        assert len(result.by_owner) == 0
        assert result.best_performer is None

    def test_summary_contains_win_rate(self):
        rows = [
            _make_deal("D1", "Won", 1000),
            _make_deal("D2", "Lost", 500),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        assert "Win rate" in result.summary

    def test_summary_mentions_best_performer(self):
        rows = [
            _make_deal("D1", "Won", 1000, owner="Alice"),
            _make_deal("D2", "Lost", 500, owner="Bob"),
        ]
        result = compute_win_rate(
            rows, "deal", "stage", value_column="value", owner_column="owner"
        )
        assert result is not None
        assert "Alice" in result.summary

    def test_without_value_column(self):
        rows = [
            _make_deal("D1", "Won", 1000),
            _make_deal("D2", "Lost", 500),
        ]
        result = compute_win_rate(rows, "deal", "stage")
        assert result is not None
        assert result.total_won == 1
        assert result.total_lost == 1
        # Without value_column, values default to 0
        assert result.won_value == 0.0
        assert result.lost_value == 0.0

    def test_duplicate_deal_takes_last_stage(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D1", "Won", 100),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        assert result.total_won == 1

    def test_returns_win_rate_result_type(self):
        rows = [_make_deal("D1", "Won", 100)]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert isinstance(result, WinRateResult)

    def test_by_owner_sorted_alphabetically(self):
        rows = [
            _make_deal("D1", "Won", 100, owner="Charlie"),
            _make_deal("D2", "Won", 200, owner="Alice"),
            _make_deal("D3", "Lost", 300, owner="Bob"),
        ]
        result = compute_win_rate(
            rows, "deal", "stage", value_column="value", owner_column="owner"
        )
        assert result is not None
        owner_names = [o.owner for o in result.by_owner]
        assert owner_names == sorted(owner_names)

    def test_deals_with_open_stage_not_counted(self):
        rows = [
            _make_deal("D1", "Won", 1000),
            _make_deal("D2", "Lost", 500),
            _make_deal("D3", "Negotiation", 800),
        ]
        result = compute_win_rate(rows, "deal", "stage", value_column="value")
        assert result is not None
        # D3 is neither won nor lost, should not affect win rate
        assert result.total_won == 1
        assert result.total_lost == 1
        assert result.overall_win_rate == 50.0


# ---------------------------------------------------------------------------
# forecast_pipeline
# ---------------------------------------------------------------------------


class TestForecastPipeline:
    def test_empty_rows(self):
        assert forecast_pipeline([], "deal", "stage", "value") is None

    def test_all_missing_columns(self):
        rows = [{"other": "val"}]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is None

    def test_non_numeric_value(self):
        rows = [{"deal": "D1", "stage": "Lead", "value": "abc"}]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is None

    def test_single_deal(self):
        rows = [_make_deal("D1", "Lead", 1000)]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_raw == 1000.0
        assert len(result.stages) == 1

    def test_default_probability_assigned(self):
        rows = [_make_deal("D1", "Lead", 1000)]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        # Single stage gets first default probability = 0.10
        assert result.stages[0].probability == 0.10

    def test_explicit_probability(self):
        rows = [_make_deal("D1", "Lead", 1000, probability=0.5)]
        result = forecast_pipeline(
            rows, "deal", "stage", "value", probability_column="probability"
        )
        assert result is not None
        assert result.stages[0].probability == 0.5
        assert result.stages[0].weighted_value == 500.0

    def test_probability_over_1_normalized(self):
        rows = [_make_deal("D1", "Lead", 1000, probability=75)]
        result = forecast_pipeline(
            rows, "deal", "stage", "value", probability_column="probability"
        )
        assert result is not None
        assert result.stages[0].probability == 0.75
        assert result.stages[0].weighted_value == 750.0

    def test_probability_clamped_to_0_1(self):
        rows = [_make_deal("D1", "Lead", 1000, probability=150)]
        result = forecast_pipeline(
            rows, "deal", "stage", "value", probability_column="probability"
        )
        assert result is not None
        # 150 > 1 -> /100 -> 1.5 -> clamped to 1.0
        assert result.stages[0].probability == 1.0

    def test_total_raw_and_weighted(self):
        rows = [
            _make_deal("D1", "Lead", 1000, probability=0.5),
            _make_deal("D2", "Lead", 2000, probability=0.5),
        ]
        result = forecast_pipeline(
            rows, "deal", "stage", "value", probability_column="probability"
        )
        assert result is not None
        assert result.total_raw == 3000.0
        assert result.total_weighted == 1500.0

    def test_coverage_ratio(self):
        rows = [
            _make_deal("D1", "Lead", 1000, probability=0.25),
        ]
        result = forecast_pipeline(
            rows, "deal", "stage", "value", probability_column="probability"
        )
        assert result is not None
        # raw=1000, weighted=250, coverage=1000/250=4.0
        assert result.coverage_ratio == 4.0

    def test_stages_ordered_by_deal_count_desc(self):
        rows = [
            _make_deal("D1", "Alpha", 100),
            _make_deal("D2", "Alpha", 200),
            _make_deal("D3", "Alpha", 150),
            _make_deal("D4", "Beta", 300),
            _make_deal("D5", "Beta", 250),
            _make_deal("D6", "Gamma", 500),
        ]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        stage_names = [s.stage for s in result.stages]
        assert stage_names == ["Alpha", "Beta", "Gamma"]

    def test_summary_mentions_deals_count(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Lead", 200),
        ]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        assert "2 deals" in result.summary

    def test_summary_mentions_coverage_ratio(self):
        rows = [_make_deal("D1", "Lead", 100)]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        assert "Coverage ratio" in result.summary

    def test_expected_close_value_equals_weighted(self):
        rows = [_make_deal("D1", "Lead", 1000, probability=0.5)]
        result = forecast_pipeline(
            rows, "deal", "stage", "value", probability_column="probability"
        )
        assert result is not None
        assert result.expected_close_value == result.total_weighted

    def test_duplicate_deal_takes_last_stage_and_max_value(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D1", "Qualified", 500),
        ]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_raw == 500.0
        assert len(result.stages) == 1
        assert result.stages[0].stage == "Qualified"

    def test_returns_forecast_result_type(self):
        rows = [_make_deal("D1", "Lead", 100)]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert isinstance(result, PipelineForecastResult)

    def test_multiple_stages_default_probabilities(self):
        # With 3 stages (ordered by count), they get first 3 default probs
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Lead", 100),
            _make_deal("D3", "Lead", 100),
            _make_deal("D4", "Qualified", 200),
            _make_deal("D5", "Qualified", 200),
            _make_deal("D6", "Won", 300),
        ]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        # Lead=3 deals (most), Qualified=2, Won=1
        # Probs: Lead=0.10, Qualified=0.25, Won=0.50
        lead = next(s for s in result.stages if s.stage == "Lead")
        qualified = next(s for s in result.stages if s.stage == "Qualified")
        won = next(s for s in result.stages if s.stage == "Won")
        assert lead.probability == 0.10
        assert qualified.probability == 0.25
        assert won.probability == 0.50

    def test_zero_value_deals(self):
        rows = [_make_deal("D1", "Lead", 0)]
        result = forecast_pipeline(rows, "deal", "stage", "value")
        assert result is not None
        assert result.total_raw == 0.0
        assert result.total_weighted == 0.0
        assert result.coverage_ratio == 0.0

    def test_none_probability_uses_default(self):
        rows = [
            {"deal": "D1", "stage": "Lead", "value": 1000, "probability": None},
        ]
        result = forecast_pipeline(
            rows, "deal", "stage", "value", probability_column="probability"
        )
        assert result is not None
        # None probability falls back to default
        assert result.stages[0].probability == 0.10


# ---------------------------------------------------------------------------
# format_pipeline_report
# ---------------------------------------------------------------------------


class TestFormatPipelineReport:
    def test_empty_report(self):
        report = format_pipeline_report()
        assert "PIPELINE ANALYSIS REPORT" in report
        assert "No data available" in report

    def test_report_has_separator_lines(self):
        report = format_pipeline_report()
        assert "=" * 60 in report

    def test_stages_section(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Won", 200),
        ]
        stages = analyze_pipeline_stages(rows, "deal", "stage", "value")
        report = format_pipeline_report(stages=stages)
        assert "PIPELINE STAGES" in report
        assert "Total Deals" in report
        assert "Total Value" in report
        assert "Weighted Value" in report
        assert "Lead" in report
        assert "Won" in report

    def test_stages_section_conversions(self):
        rows = [
            _make_deal("D1", "Lead", 100),
            _make_deal("D2", "Won", 200),
        ]
        stages = analyze_pipeline_stages(rows, "deal", "stage", "value")
        report = format_pipeline_report(stages=stages)
        assert "Conversions" in report

    def test_velocity_section(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Won", 100, date="2024-01-11"),
        ]
        velocity = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        report = format_pipeline_report(velocity=velocity)
        assert "PIPELINE VELOCITY" in report
        assert "Avg Cycle" in report
        assert "Fastest Deal" in report
        assert "Slowest Deal" in report

    def test_velocity_section_bottleneck(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01"),
            _make_deal("D1", "Qualified", 100, date="2024-01-10"),
            _make_deal("D1", "Won", 100, date="2024-01-12"),
        ]
        velocity = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        report = format_pipeline_report(velocity=velocity)
        assert "Bottleneck" in report

    def test_win_rate_section(self):
        rows = [
            _make_deal("D1", "Won", 1000),
            _make_deal("D2", "Lost", 500),
        ]
        win_rate = compute_win_rate(rows, "deal", "stage", value_column="value")
        report = format_pipeline_report(win_rate=win_rate)
        assert "WIN RATE ANALYSIS" in report
        assert "Win Rate" in report
        assert "Won:" in report
        assert "Lost:" in report

    def test_win_rate_section_best_performer(self):
        rows = [
            _make_deal("D1", "Won", 1000, owner="Alice"),
            _make_deal("D2", "Lost", 500, owner="Bob"),
        ]
        win_rate = compute_win_rate(
            rows, "deal", "stage", value_column="value", owner_column="owner"
        )
        report = format_pipeline_report(win_rate=win_rate)
        assert "Best Performer" in report
        assert "Alice" in report

    def test_win_rate_section_by_owner(self):
        rows = [
            _make_deal("D1", "Won", 1000, owner="Alice"),
            _make_deal("D2", "Lost", 500, owner="Bob"),
        ]
        win_rate = compute_win_rate(
            rows, "deal", "stage", value_column="value", owner_column="owner"
        )
        report = format_pipeline_report(win_rate=win_rate)
        assert "By Owner" in report

    def test_forecast_section(self):
        rows = [
            _make_deal("D1", "Lead", 1000),
            _make_deal("D2", "Won", 2000),
        ]
        forecast = forecast_pipeline(rows, "deal", "stage", "value")
        report = format_pipeline_report(forecast=forecast)
        assert "PIPELINE FORECAST" in report
        assert "Raw Value" in report
        assert "Weighted Value" in report
        assert "Expected Close" in report
        assert "Coverage Ratio" in report

    def test_combined_report(self):
        rows = [
            _make_deal("D1", "Lead", 100, date="2024-01-01", owner="Alice"),
            _make_deal("D2", "Won", 200, date="2024-01-15", owner="Bob"),
            _make_deal("D3", "Lost", 150, date="2024-01-20", owner="Alice"),
        ]
        stages = analyze_pipeline_stages(
            rows, "deal", "stage", "value", owner_column="owner"
        )
        velocity = analyze_pipeline_velocity(rows, "deal", "stage", "value", "date")
        win_rate = compute_win_rate(
            rows, "deal", "stage", value_column="value", owner_column="owner"
        )
        forecast = forecast_pipeline(rows, "deal", "stage", "value")
        report = format_pipeline_report(
            stages=stages, velocity=velocity, win_rate=win_rate, forecast=forecast
        )
        assert "PIPELINE STAGES" in report
        assert "WIN RATE ANALYSIS" in report
        assert "PIPELINE FORECAST" in report

    def test_no_data_message_when_all_none(self):
        report = format_pipeline_report(
            stages=None, velocity=None, win_rate=None, forecast=None
        )
        assert "No data available" in report

    def test_report_with_only_stages(self):
        rows = [_make_deal("D1", "Lead", 100)]
        stages = analyze_pipeline_stages(rows, "deal", "stage", "value")
        report = format_pipeline_report(stages=stages)
        assert "PIPELINE STAGES" in report
        # Should not contain sections we didn't provide
        assert "WIN RATE ANALYSIS" not in report
        assert "PIPELINE FORECAST" not in report


# ---------------------------------------------------------------------------
# Dataclass field tests
# ---------------------------------------------------------------------------


class TestDataclassFields:
    def test_stage_metrics_fields(self):
        sm = StageMetrics(
            stage="Lead",
            deal_count=10,
            total_value=50000.0,
            avg_value=5000.0,
            pct_of_deals=40.0,
            pct_of_value=25.0,
        )
        assert sm.stage == "Lead"
        assert sm.deal_count == 10
        assert sm.total_value == 50000.0
        assert sm.avg_value == 5000.0
        assert sm.pct_of_deals == 40.0
        assert sm.pct_of_value == 25.0

    def test_stage_conversion_fields(self):
        sc = StageConversion(
            from_stage="Lead",
            to_stage="Qualified",
            conversion_rate=65.5,
        )
        assert sc.from_stage == "Lead"
        assert sc.to_stage == "Qualified"
        assert sc.conversion_rate == 65.5

    def test_pipeline_stage_result_fields(self):
        psr = PipelineStageResult(
            stages=[],
            conversions=[],
            total_deals=25,
            total_value=100000.0,
            weighted_value=75000.0,
            summary="test",
        )
        assert psr.total_deals == 25
        assert psr.total_value == 100000.0
        assert psr.weighted_value == 75000.0
        assert psr.summary == "test"

    def test_stage_velocity_fields(self):
        sv = StageVelocity(
            stage="Qualified",
            avg_days=12.5,
            deal_count=8,
        )
        assert sv.stage == "Qualified"
        assert sv.avg_days == 12.5
        assert sv.deal_count == 8

    def test_pipeline_velocity_result_fields(self):
        pvr = PipelineVelocityResult(
            stage_velocities=[],
            avg_cycle_days=30.0,
            fastest_deal_days=10.0,
            slowest_deal_days=60.0,
            bottleneck_stage="Negotiation",
            summary="test",
        )
        assert pvr.avg_cycle_days == 30.0
        assert pvr.fastest_deal_days == 10.0
        assert pvr.slowest_deal_days == 60.0
        assert pvr.bottleneck_stage == "Negotiation"

    def test_owner_win_rate_fields(self):
        owr = OwnerWinRate(
            owner="Alice",
            won=10,
            lost=5,
            win_rate=66.67,
            won_value=50000.0,
        )
        assert owr.owner == "Alice"
        assert owr.won == 10
        assert owr.lost == 5
        assert owr.win_rate == 66.67
        assert owr.won_value == 50000.0

    def test_win_rate_result_fields(self):
        wrr = WinRateResult(
            overall_win_rate=60.0,
            total_won=12,
            total_lost=8,
            won_value=120000.0,
            lost_value=40000.0,
            by_owner=[],
            best_performer="Alice",
            summary="test",
        )
        assert wrr.overall_win_rate == 60.0
        assert wrr.total_won == 12
        assert wrr.total_lost == 8
        assert wrr.won_value == 120000.0
        assert wrr.lost_value == 40000.0
        assert wrr.best_performer == "Alice"

    def test_win_rate_result_no_best_performer(self):
        wrr = WinRateResult(
            overall_win_rate=50.0,
            total_won=5,
            total_lost=5,
            won_value=25000.0,
            lost_value=25000.0,
            by_owner=[],
            best_performer=None,
            summary="test",
        )
        assert wrr.best_performer is None

    def test_stage_forecast_fields(self):
        sf = StageForecast(
            stage="Proposal",
            deal_count=5,
            raw_value=50000.0,
            probability=0.50,
            weighted_value=25000.0,
        )
        assert sf.stage == "Proposal"
        assert sf.deal_count == 5
        assert sf.raw_value == 50000.0
        assert sf.probability == 0.50
        assert sf.weighted_value == 25000.0

    def test_pipeline_forecast_result_fields(self):
        pfr = PipelineForecastResult(
            stages=[],
            total_raw=200000.0,
            total_weighted=100000.0,
            expected_close_value=100000.0,
            coverage_ratio=2.0,
            summary="test",
        )
        assert pfr.total_raw == 200000.0
        assert pfr.total_weighted == 100000.0
        assert pfr.expected_close_value == 100000.0
        assert pfr.coverage_ratio == 2.0
