"""Tests for the discovery engine orchestrator (unit tests — no DB)."""

from business_brain.db.discovery_models import Insight, TableProfile
from business_brain.discovery.anomaly_detector import detect_anomalies
from business_brain.discovery.engine import _apply_scoring


class TestApplyScoring:
    """Test the scoring formula application."""

    def _make_insight(self, severity="info", source_tables=None, impact=0):
        i = Insight()
        i.severity = severity
        i.source_tables = source_tables or ["table1"]
        i.impact_score = impact
        return i

    def test_critical_single_table(self):
        insight = self._make_insight("critical", ["t1"], 50)
        _apply_scoring(insight)
        # severity_score = 1.0 * 40 = 40
        # cross_table = 0 (single table)
        # magnitude = 50/100 * 30 = 15
        assert insight.impact_score == 55

    def test_info_cross_table(self):
        insight = self._make_insight("info", ["t1", "t2"], 60)
        _apply_scoring(insight)
        # severity_score = 0.3 * 40 = 12
        # cross_table = 30 (two tables)
        # magnitude = 60/100 * 30 = 18
        assert insight.impact_score == 60

    def test_warning_single_table(self):
        insight = self._make_insight("warning", ["t1"], 40)
        _apply_scoring(insight)
        # severity_score = 0.6 * 40 = 24
        # cross_table = 0
        # magnitude = 40/100 * 30 = 12
        assert insight.impact_score == 36

    def test_max_score_capped_at_100(self):
        insight = self._make_insight("critical", ["t1", "t2"], 100)
        _apply_scoring(insight)
        # severity_score = 40 + cross_table = 30 + magnitude = 30 = 100
        assert insight.impact_score <= 100

    def test_zero_impact(self):
        insight = self._make_insight("info", ["t1"], 0)
        _apply_scoring(insight)
        # severity_score = 0.3 * 40 = 12, cross=0, magnitude=0
        assert insight.impact_score == 12

    def test_empty_source_tables(self):
        insight = self._make_insight("warning", [], 50)
        _apply_scoring(insight)
        # severity = 0.6*40=24, cross=0 (empty list), magnitude=50/100*30=15
        assert insight.impact_score == 39

    def test_none_source_tables(self):
        insight = self._make_insight("info", None, 30)
        _apply_scoring(insight)
        # severity = 0.3*40=12, cross=0 (None -> []), magnitude=30/100*30=9
        assert insight.impact_score == 21

    def test_unknown_severity_defaults(self):
        insight = self._make_insight("unknown_sev", ["t1"], 50)
        _apply_scoring(insight)
        # Unknown severity defaults to 0.3: 0.3*40=12, cross=0, magnitude=15
        assert insight.impact_score == 27

    def test_three_tables_still_cross_bonus(self):
        insight = self._make_insight("info", ["t1", "t2", "t3"], 50)
        _apply_scoring(insight)
        # severity=12, cross=30 (>1 table), magnitude=15
        assert insight.impact_score == 57


class TestAnomalyDetection:
    """Test anomaly detection from profiles."""

    def _make_profile(self, table_name, columns_dict, row_count=100):
        p = TableProfile()
        p.table_name = table_name
        p.row_count = row_count
        p.column_classification = {
            "columns": columns_dict,
            "domain_hint": "general",
        }
        return p

    def test_null_spike_detection(self):
        profile = self._make_profile("data", {
            "value": {
                "semantic_type": "numeric_metric",
                "null_count": 30,
                "cardinality": 50,
                "sample_values": ["1", "2", "3"],
            },
        }, row_count=100)

        insights = detect_anomalies([profile])
        null_insights = [i for i in insights if "null" in i.title.lower()]
        assert len(null_insights) >= 1
        assert null_insights[0].severity in ("warning", "info")

    def test_negative_currency_detection(self):
        profile = self._make_profile("finance", {
            "amount": {
                "semantic_type": "numeric_currency",
                "null_count": 0,
                "cardinality": 50,
                "sample_values": ["-100", "200", "300"],
                "stats": {"mean": 133.33, "min": -100, "max": 300, "stdev": 164.99},
            },
        })

        insights = detect_anomalies([profile])
        currency_anomalies = [i for i in insights if "negative" in i.title.lower()]
        assert len(currency_anomalies) >= 1
        assert currency_anomalies[0].severity == "critical"

    def test_constant_column_detection(self):
        profile = self._make_profile("data", {
            "status": {
                "semantic_type": "categorical",
                "null_count": 0,
                "cardinality": 1,
                "sample_values": ["active"],
            },
        }, row_count=50)

        insights = detect_anomalies([profile])
        constant = [i for i in insights if "constant" in i.title.lower()]
        assert len(constant) >= 1

    def test_time_series_detection(self):
        profile = self._make_profile("sales", {
            "order_date": {
                "semantic_type": "temporal",
                "null_count": 0,
                "cardinality": 30,
                "sample_values": ["2024-01-01"],
            },
            "revenue": {
                "semantic_type": "numeric_currency",
                "null_count": 0,
                "cardinality": 80,
                "sample_values": ["1000", "2000"],
                "stats": {"mean": 1500, "min": 1000, "max": 2000},
            },
        })

        insights = detect_anomalies([profile])
        time_insights = [i for i in insights if i.insight_type == "trend"]
        assert len(time_insights) >= 1

    def test_no_anomalies_clean_data(self):
        profile = self._make_profile("clean", {
            "id": {
                "semantic_type": "identifier",
                "null_count": 0,
                "cardinality": 100,
                "sample_values": ["1", "2", "3"],
            },
        })

        insights = detect_anomalies([profile])
        # Should have no anomalies for a clean identifier column
        anomalies = [i for i in insights if i.insight_type == "anomaly"]
        assert len(anomalies) == 0

    def test_percentage_out_of_range(self):
        profile = self._make_profile("metrics", {
            "completion_pct": {
                "semantic_type": "numeric_percentage",
                "null_count": 0,
                "cardinality": 20,
                "sample_values": ["50", "110", "80"],
                "stats": {"mean": 80, "min": 50, "max": 110, "stdev": 24.5},
            },
        })

        insights = detect_anomalies([profile])
        pct_anomalies = [i for i in insights if "percentage" in i.title.lower() or "range" in i.title.lower()]
        assert len(pct_anomalies) >= 1

    def test_empty_profiles(self):
        insights = detect_anomalies([])
        assert insights == []
