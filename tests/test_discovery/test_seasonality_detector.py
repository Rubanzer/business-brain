"""Tests for the seasonality detector module."""

from business_brain.db.discovery_models import TableProfile
from business_brain.discovery.seasonality_detector import (
    _scan_table_seasonality,
    detect_seasonality,
)


class _Prof:
    """Lightweight profile stand-in."""
    def __init__(self, table_name, columns_dict, row_count=100, domain="general"):
        self.table_name = table_name
        self.row_count = row_count
        self.domain_hint = domain
        self.column_classification = {"columns": columns_dict, "domain_hint": domain}


class TestShiftCycleDetection:
    """Test shift pattern detection."""

    def test_shift_column_detected(self):
        prof = _Prof("production", {
            "shift": {"semantic_type": "categorical", "cardinality": 3},
            "output_tons": {"semantic_type": "numeric_metric", "cardinality": 50},
            "timestamp": {"semantic_type": "temporal", "cardinality": 30},
        }, domain="manufacturing")
        insights = _scan_table_seasonality(prof)
        shift_insights = [i for i in insights if i.evidence.get("pattern_type") == "shift_cycle"]
        assert len(shift_insights) >= 1
        assert shift_insights[0].evidence["shift_column"] == "shift"
        assert shift_insights[0].evidence["shift_count"] == 3

    def test_shift_column_with_2_shifts(self):
        prof = _Prof("factory", {
            "shift_name": {"semantic_type": "categorical", "cardinality": 2},
            "power_kva": {"semantic_type": "numeric_metric", "cardinality": 40},
            "date": {"semantic_type": "temporal", "cardinality": 20},
        })
        insights = _scan_table_seasonality(prof)
        shift_insights = [i for i in insights if i.evidence.get("pattern_type") == "shift_cycle"]
        assert len(shift_insights) >= 1

    def test_no_shift_if_cardinality_too_high(self):
        """6+ values shouldn't be treated as shifts."""
        prof = _Prof("data", {
            "shift": {"semantic_type": "categorical", "cardinality": 10},
            "value": {"semantic_type": "numeric_metric", "cardinality": 50},
            "date": {"semantic_type": "temporal", "cardinality": 20},
        })
        insights = _scan_table_seasonality(prof)
        shift_insights = [i for i in insights if i.evidence.get("pattern_type") == "shift_cycle"]
        assert len(shift_insights) == 0

    def test_no_shift_if_cardinality_1(self):
        prof = _Prof("data", {
            "shift": {"semantic_type": "categorical", "cardinality": 1},
            "value": {"semantic_type": "numeric_metric", "cardinality": 50},
            "date": {"semantic_type": "temporal", "cardinality": 20},
        })
        insights = _scan_table_seasonality(prof)
        shift_insights = [i for i in insights if i.evidence.get("pattern_type") == "shift_cycle"]
        assert len(shift_insights) == 0


class TestDayOfWeekDetection:
    """Test day-of-week pattern detection."""

    def test_enough_dates_triggers_dow(self):
        prof = _Prof("sales", {
            "date": {"semantic_type": "temporal", "cardinality": 14},
            "revenue": {"semantic_type": "numeric_currency", "cardinality": 50},
        })
        insights = _scan_table_seasonality(prof)
        dow_insights = [i for i in insights if i.evidence.get("pattern_type") == "day_of_week"]
        assert len(dow_insights) == 1
        assert dow_insights[0].evidence["temporal_column"] == "date"

    def test_too_few_dates_no_dow(self):
        prof = _Prof("tiny", {
            "date": {"semantic_type": "temporal", "cardinality": 5},
            "value": {"semantic_type": "numeric_metric", "cardinality": 10},
        })
        insights = _scan_table_seasonality(prof)
        dow_insights = [i for i in insights if i.evidence.get("pattern_type") == "day_of_week"]
        assert len(dow_insights) == 0

    def test_exactly_7_dates_triggers(self):
        prof = _Prof("weekly", {
            "timestamp": {"semantic_type": "temporal", "cardinality": 7},
            "output": {"semantic_type": "numeric_metric", "cardinality": 30},
        })
        insights = _scan_table_seasonality(prof)
        dow_insights = [i for i in insights if i.evidence.get("pattern_type") == "day_of_week"]
        assert len(dow_insights) == 1

    def test_only_one_dow_per_table(self):
        """Even with multiple temporal columns, only one DOW insight per table."""
        prof = _Prof("multi_date", {
            "start_date": {"semantic_type": "temporal", "cardinality": 30},
            "end_date": {"semantic_type": "temporal", "cardinality": 30},
            "value": {"semantic_type": "numeric_metric", "cardinality": 50},
        })
        insights = _scan_table_seasonality(prof)
        dow_insights = [i for i in insights if i.evidence.get("pattern_type") == "day_of_week"]
        assert len(dow_insights) == 1


class TestMonthlyTrendDetection:
    """Test monthly trend detection."""

    def test_enough_dates_triggers_monthly(self):
        prof = _Prof("production", {
            "date": {"semantic_type": "temporal", "cardinality": 90},
            "output": {"semantic_type": "numeric_metric", "cardinality": 50},
        })
        insights = _scan_table_seasonality(prof)
        monthly = [i for i in insights if i.evidence.get("pattern_type") == "monthly_trend"]
        assert len(monthly) == 1

    def test_exactly_30_triggers(self):
        prof = _Prof("data", {
            "date": {"semantic_type": "temporal", "cardinality": 30},
            "metric": {"semantic_type": "numeric_metric", "cardinality": 20},
        })
        insights = _scan_table_seasonality(prof)
        monthly = [i for i in insights if i.evidence.get("pattern_type") == "monthly_trend"]
        assert len(monthly) == 1

    def test_29_dates_no_monthly(self):
        prof = _Prof("data", {
            "date": {"semantic_type": "temporal", "cardinality": 29},
            "metric": {"semantic_type": "numeric_metric", "cardinality": 20},
        })
        insights = _scan_table_seasonality(prof)
        monthly = [i for i in insights if i.evidence.get("pattern_type") == "monthly_trend"]
        assert len(monthly) == 0


class TestDistributionSkew:
    """Test categorical distribution skew detection."""

    def test_highly_skewed_detected(self):
        prof = _Prof("orders", {
            "status": {
                "semantic_type": "categorical", "cardinality": 3,
                "sample_values": ["complete"] * 10 + ["pending", "cancelled"],
            },
            "date": {"semantic_type": "temporal", "cardinality": 30},
            "amount": {"semantic_type": "numeric_currency", "cardinality": 50},
        })
        insights = _scan_table_seasonality(prof)
        skew = [i for i in insights if i.evidence.get("pattern_type") == "distribution_skew"]
        assert len(skew) == 1
        assert skew[0].evidence["dominant_value"] == "complete"

    def test_balanced_not_flagged(self):
        prof = _Prof("data", {
            "category": {
                "semantic_type": "categorical", "cardinality": 3,
                "sample_values": ["A", "B", "C"] * 5,
            },
            "date": {"semantic_type": "temporal", "cardinality": 15},
            "value": {"semantic_type": "numeric_metric", "cardinality": 30},
        })
        insights = _scan_table_seasonality(prof)
        skew = [i for i in insights if i.evidence.get("pattern_type") == "distribution_skew"]
        assert len(skew) == 0

    def test_too_few_samples_not_flagged(self):
        prof = _Prof("tiny", {
            "status": {
                "semantic_type": "categorical", "cardinality": 2,
                "sample_values": ["A", "A", "B"],  # < 10 samples
            },
            "date": {"semantic_type": "temporal", "cardinality": 10},
            "val": {"semantic_type": "numeric_metric", "cardinality": 10},
        })
        insights = _scan_table_seasonality(prof)
        skew = [i for i in insights if i.evidence.get("pattern_type") == "distribution_skew"]
        assert len(skew) == 0


class TestEdgeCases:
    """Test edge cases."""

    def test_no_temporal_columns(self):
        prof = _Prof("data", {
            "name": {"semantic_type": "categorical", "cardinality": 10},
            "value": {"semantic_type": "numeric_metric", "cardinality": 50},
        })
        insights = _scan_table_seasonality(prof)
        assert len(insights) == 0

    def test_no_numeric_columns(self):
        prof = _Prof("data", {
            "date": {"semantic_type": "temporal", "cardinality": 30},
            "name": {"semantic_type": "categorical", "cardinality": 10},
        })
        insights = _scan_table_seasonality(prof)
        assert len(insights) == 0

    def test_too_few_rows(self):
        prof = _Prof("tiny", {
            "date": {"semantic_type": "temporal", "cardinality": 5},
            "value": {"semantic_type": "numeric_metric", "cardinality": 3},
        }, row_count=5)
        insights = _scan_table_seasonality(prof)
        assert len(insights) == 0

    def test_no_classification(self):
        prof = _Prof("empty", {}, row_count=100)
        prof.column_classification = None
        insights = _scan_table_seasonality(prof)
        assert len(insights) == 0

    def test_empty_profiles(self):
        insights = detect_seasonality([])
        assert insights == []

    def test_multiple_profiles(self):
        prof_a = _Prof("production", {
            "shift": {"semantic_type": "categorical", "cardinality": 3},
            "output": {"semantic_type": "numeric_metric", "cardinality": 50},
            "date": {"semantic_type": "temporal", "cardinality": 90},
        })
        prof_b = _Prof("sales", {
            "date": {"semantic_type": "temporal", "cardinality": 30},
            "revenue": {"semantic_type": "numeric_currency", "cardinality": 80},
        })
        insights = detect_seasonality([prof_a, prof_b])
        assert len(insights) > 0
        tables = {t for i in insights for t in i.source_tables}
        assert "production" in tables
        assert "sales" in tables

    def test_insight_has_valid_fields(self):
        prof = _Prof("data", {
            "date": {"semantic_type": "temporal", "cardinality": 30},
            "value": {"semantic_type": "numeric_metric", "cardinality": 50},
        })
        insights = _scan_table_seasonality(prof)
        for ins in insights:
            assert ins.id is not None
            assert ins.insight_type == "seasonality"
            assert ins.severity in ("critical", "warning", "info")
            assert ins.source_tables == ["data"]
            assert ins.evidence is not None
            assert ins.suggested_actions is not None
