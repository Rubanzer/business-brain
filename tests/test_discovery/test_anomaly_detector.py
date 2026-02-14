"""Tests for the anomaly detector module."""

from business_brain.db.discovery_models import Insight, TableProfile
from business_brain.discovery.anomaly_detector import (
    _manufacturing_anomalies,
    _MANUFACTURING_RANGES,
    detect_anomalies,
    _scan_table,
)


class _Prof:
    """Lightweight stand-in for TableProfile."""

    def __init__(self, table_name, columns_dict, row_count=100, domain="general"):
        self.table_name = table_name
        self.row_count = row_count
        self.domain_hint = domain
        self.column_classification = {
            "columns": columns_dict,
            "domain_hint": domain,
        }


# ---------------------------------------------------------------------------
# 1. Null spike detection
# ---------------------------------------------------------------------------


class TestNullSpike:
    def test_null_above_10_pct_triggers(self):
        prof = _Prof("t", {"col_a": {
            "semantic_type": "text",
            "null_count": 15,
            "cardinality": 50,
        }}, row_count=100)
        results = _scan_table(prof)
        null_insights = [i for i in results if "null" in i.title.lower()]
        assert len(null_insights) == 1
        assert null_insights[0].severity == "info"  # 15% < 30%

    def test_null_at_exactly_10_pct_not_triggered(self):
        """10% threshold is exclusive (> 0.10)."""
        prof = _Prof("t", {"col_a": {
            "semantic_type": "text",
            "null_count": 10,
            "cardinality": 50,
        }}, row_count=100)
        results = _scan_table(prof)
        null_insights = [i for i in results if "null" in i.title.lower()]
        assert len(null_insights) == 0

    def test_null_above_30_pct_is_warning(self):
        prof = _Prof("t", {"col_a": {
            "semantic_type": "text",
            "null_count": 35,
            "cardinality": 50,
        }}, row_count=100)
        results = _scan_table(prof)
        null_insights = [i for i in results if "null" in i.title.lower()]
        assert len(null_insights) == 1
        assert null_insights[0].severity == "warning"

    def test_null_at_exactly_30_pct_is_info(self):
        """30% boundary: pct=30.0, which is NOT > 30, so severity is info."""
        prof = _Prof("t", {"col_a": {
            "semantic_type": "text",
            "null_count": 30,
            "cardinality": 50,
        }}, row_count=100)
        results = _scan_table(prof)
        null_insights = [i for i in results if "null" in i.title.lower()]
        assert len(null_insights) == 1
        assert null_insights[0].severity == "info"

    def test_null_impact_capped_at_80(self):
        prof = _Prof("t", {"col_a": {
            "semantic_type": "text",
            "null_count": 95,
            "cardinality": 5,
        }}, row_count=100)
        results = _scan_table(prof)
        null_insights = [i for i in results if "null" in i.title.lower()]
        assert null_insights[0].impact_score == 80

    def test_null_zero_row_count_no_crash(self):
        prof = _Prof("t", {"col_a": {
            "semantic_type": "text",
            "null_count": 5,
            "cardinality": 0,
        }}, row_count=0)
        results = _scan_table(prof)
        null_insights = [i for i in results if "null" in i.title.lower()]
        assert len(null_insights) == 0


# ---------------------------------------------------------------------------
# 2. Numeric outliers
# ---------------------------------------------------------------------------


class TestNumericOutliers:
    def test_outlier_detected(self):
        prof = _Prof("t", {"val": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 10.0, "stdev": 2.0, "min": 1.0, "max": 100.0},
            "sample_values": ["10", "12", "100", "9", "11"],
        }})
        results = _scan_table(prof)
        outlier = [i for i in results if "outlier" in i.title.lower()]
        assert len(outlier) == 1
        assert "100" in str(outlier[0].evidence["outlier_samples"])

    def test_no_outlier_when_all_within_2sigma(self):
        prof = _Prof("t", {"val": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 5,
            "stats": {"mean": 10.0, "stdev": 5.0, "min": 2.0, "max": 18.0},
            "sample_values": ["10", "12", "8", "15", "5"],
        }})
        results = _scan_table(prof)
        outlier = [i for i in results if "outlier" in i.title.lower()]
        assert len(outlier) == 0

    def test_outlier_comma_separated_numbers(self):
        prof = _Prof("t", {"val": {
            "semantic_type": "numeric_currency",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1000.0, "stdev": 100.0, "min": 800.0, "max": 50000.0},
            "sample_values": ["1,000", "1,100", "50,000"],
        }})
        results = _scan_table(prof)
        outlier = [i for i in results if "outlier" in i.title.lower()]
        assert len(outlier) == 1

    def test_zero_stdev_no_outlier(self):
        prof = _Prof("t", {"val": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 1,
            "stats": {"mean": 10.0, "stdev": 0, "min": 10.0, "max": 10.0},
            "sample_values": ["10"],
        }})
        results = _scan_table(prof)
        outlier = [i for i in results if "outlier" in i.title.lower()]
        assert len(outlier) == 0

    def test_outlier_samples_capped_at_5(self):
        prof = _Prof("t", {"val": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 10.0, "stdev": 1.0, "min": 1.0, "max": 100.0},
            "sample_values": ["100", "200", "300", "400", "500", "600", "700"],
        }})
        results = _scan_table(prof)
        outlier = [i for i in results if "outlier" in i.title.lower()]
        assert len(outlier[0].evidence["outlier_samples"]) <= 5


# ---------------------------------------------------------------------------
# 3. Impossible currency values
# ---------------------------------------------------------------------------


class TestImpossibleCurrency:
    def test_negative_currency_flagged(self):
        prof = _Prof("t", {"amount": {
            "semantic_type": "numeric_currency",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 500.0, "min": -100.0, "max": 1000.0},
            "sample_values": [],
        }})
        results = _scan_table(prof)
        neg = [i for i in results if "negative" in i.title.lower()]
        assert len(neg) == 1
        assert neg[0].severity == "critical"
        assert neg[0].impact_score == 70

    def test_positive_currency_not_flagged(self):
        prof = _Prof("t", {"amount": {
            "semantic_type": "numeric_currency",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 500.0, "min": 10.0, "max": 1000.0},
            "sample_values": [],
        }})
        results = _scan_table(prof)
        neg = [i for i in results if "negative" in i.title.lower()]
        assert len(neg) == 0

    def test_zero_min_not_flagged(self):
        prof = _Prof("t", {"amount": {
            "semantic_type": "numeric_currency",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 500.0, "min": 0, "max": 1000.0},
            "sample_values": [],
        }})
        results = _scan_table(prof)
        neg = [i for i in results if "negative" in i.title.lower()]
        assert len(neg) == 0


# ---------------------------------------------------------------------------
# 4. Out-of-range percentages
# ---------------------------------------------------------------------------


class TestPercentageRange:
    def test_above_100_flagged(self):
        prof = _Prof("t", {"pct": {
            "semantic_type": "numeric_percentage",
            "null_count": 0,
            "cardinality": 20,
            "stats": {"mean": 50.0, "min": 10.0, "max": 120.0},
            "sample_values": [],
        }})
        results = _scan_table(prof)
        pct = [i for i in results if "percentage" in i.title.lower()]
        assert len(pct) == 1
        assert pct[0].severity == "critical"

    def test_below_0_flagged(self):
        prof = _Prof("t", {"pct": {
            "semantic_type": "numeric_percentage",
            "null_count": 0,
            "cardinality": 20,
            "stats": {"mean": 50.0, "min": -5.0, "max": 90.0},
            "sample_values": [],
        }})
        results = _scan_table(prof)
        pct = [i for i in results if "percentage" in i.title.lower()]
        assert len(pct) == 1

    def test_valid_0_to_100_not_flagged(self):
        prof = _Prof("t", {"pct": {
            "semantic_type": "numeric_percentage",
            "null_count": 0,
            "cardinality": 20,
            "stats": {"mean": 50.0, "min": 0.0, "max": 100.0},
            "sample_values": [],
        }})
        results = _scan_table(prof)
        pct = [i for i in results if "percentage" in i.title.lower()]
        assert len(pct) == 0


# ---------------------------------------------------------------------------
# 5. High cardinality categorical
# ---------------------------------------------------------------------------


class TestHighCardinalityCategorical:
    def test_high_cardinality_detected(self):
        prof = _Prof("t", {"category": {
            "semantic_type": "categorical",
            "null_count": 0,
            "cardinality": 60,
            "sample_values": ["a", "b"],
        }}, row_count=100)
        results = _scan_table(prof)
        hc = [i for i in results if "cardinality" in i.title.lower()]
        assert len(hc) == 1
        assert hc[0].severity == "info"

    def test_low_cardinality_not_flagged(self):
        prof = _Prof("t", {"category": {
            "semantic_type": "categorical",
            "null_count": 0,
            "cardinality": 3,
            "sample_values": ["a", "b", "c"],
        }}, row_count=100)
        results = _scan_table(prof)
        hc = [i for i in results if "cardinality" in i.title.lower()]
        assert len(hc) == 0

    def test_exactly_5_cardinality_not_flagged(self):
        """cardinality must be > 5 AND > row_count*0.5."""
        prof = _Prof("t", {"category": {
            "semantic_type": "categorical",
            "null_count": 0,
            "cardinality": 5,
            "sample_values": ["a", "b", "c", "d", "e"],
        }}, row_count=100)
        results = _scan_table(prof)
        hc = [i for i in results if "cardinality" in i.title.lower()]
        assert len(hc) == 0


# ---------------------------------------------------------------------------
# 6. Constant columns
# ---------------------------------------------------------------------------


class TestConstantColumn:
    def test_constant_column_detected(self):
        prof = _Prof("t", {"status": {
            "semantic_type": "categorical",
            "null_count": 0,
            "cardinality": 1,
            "sample_values": ["ACTIVE"],
        }}, row_count=50)
        results = _scan_table(prof)
        const = [i for i in results if "constant" in i.title.lower()]
        assert len(const) == 1
        assert const[0].impact_score == 10

    def test_constant_column_not_flagged_for_single_row(self):
        """row_count must be > 1 to flag constant column."""
        prof = _Prof("t", {"status": {
            "semantic_type": "categorical",
            "null_count": 0,
            "cardinality": 1,
            "sample_values": ["ACTIVE"],
        }}, row_count=1)
        results = _scan_table(prof)
        const = [i for i in results if "constant" in i.title.lower()]
        assert len(const) == 0

    def test_cardinality_2_not_constant(self):
        prof = _Prof("t", {"status": {
            "semantic_type": "categorical",
            "null_count": 0,
            "cardinality": 2,
            "sample_values": ["ACTIVE", "INACTIVE"],
        }}, row_count=50)
        results = _scan_table(prof)
        const = [i for i in results if "constant" in i.title.lower()]
        assert len(const) == 0


# ---------------------------------------------------------------------------
# 7. Time series detection
# ---------------------------------------------------------------------------


class TestTimeSeriesDetection:
    def test_temporal_plus_numeric_creates_trend_insight(self):
        prof = _Prof("sales", {
            "order_date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
            "revenue": {"semantic_type": "numeric_currency", "null_count": 0, "cardinality": 80},
        }, row_count=100)
        results = _scan_table(prof)
        trend = [i for i in results if i.insight_type == "trend"]
        assert len(trend) == 1
        assert "time series" in trend[0].title.lower()
        assert "query" in trend[0].evidence
        assert "chart_spec" in trend[0].evidence

    def test_no_trend_without_temporal(self):
        prof = _Prof("t", {
            "revenue": {"semantic_type": "numeric_currency", "null_count": 0, "cardinality": 80},
        }, row_count=100)
        results = _scan_table(prof)
        trend = [i for i in results if i.insight_type == "trend"]
        assert len(trend) == 0

    def test_no_trend_without_numeric(self):
        prof = _Prof("t", {
            "order_date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
            "name": {"semantic_type": "text", "null_count": 0, "cardinality": 50},
        }, row_count=100)
        results = _scan_table(prof)
        trend = [i for i in results if i.insight_type == "trend"]
        assert len(trend) == 0

    def test_trend_chart_spec_structure(self):
        prof = _Prof("production", {
            "timestamp": {"semantic_type": "temporal", "null_count": 0, "cardinality": 100},
            "output_tons": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 50},
            "power_kva": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 50},
        }, row_count=200)
        results = _scan_table(prof)
        trend = [i for i in results if i.insight_type == "trend"]
        chart = trend[0].evidence["chart_spec"]
        assert chart["type"] == "line"
        assert chart["x"] == "timestamp"
        assert len(chart["y"]) <= 2

    def test_trend_limits_to_3_numeric_cols(self):
        prof = _Prof("t", {
            "date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
            "a": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 10},
            "b": {"semantic_type": "numeric_currency", "null_count": 0, "cardinality": 10},
            "c": {"semantic_type": "numeric_percentage", "null_count": 0, "cardinality": 10},
            "d": {"semantic_type": "numeric_metric", "null_count": 0, "cardinality": 10},
        }, row_count=100)
        results = _scan_table(prof)
        trend = [i for i in results if i.insight_type == "trend"]
        # source_columns = temporal + numeric[:3]
        assert len(trend[0].source_columns) <= 4  # 1 temporal + 3 numeric


# ---------------------------------------------------------------------------
# Top-level detect_anomalies function
# ---------------------------------------------------------------------------


class TestDetectAnomalies:
    def test_processes_multiple_profiles(self):
        p1 = _Prof("t1", {"a": {
            "semantic_type": "text", "null_count": 50, "cardinality": 10,
        }}, row_count=100)
        p2 = _Prof("t2", {"b": {
            "semantic_type": "text", "null_count": 60, "cardinality": 10,
        }}, row_count=100)
        results = detect_anomalies([p1, p2])
        null_insights = [i for i in results if "null" in i.title.lower()]
        tables = {i.source_tables[0] for i in null_insights}
        assert "t1" in tables
        assert "t2" in tables

    def test_empty_profiles_returns_empty(self):
        assert detect_anomalies([]) == []

    def test_missing_classification_skipped(self):
        prof = _Prof("t", {})
        prof.column_classification = None
        results = detect_anomalies([prof])
        assert results == []

    def test_missing_columns_key_skipped(self):
        prof = _Prof("t", {})
        prof.column_classification = {"domain_hint": "sales"}
        results = detect_anomalies([prof])
        assert results == []

    def test_all_insights_have_required_fields(self):
        prof = _Prof("t", {
            "date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
            "val": {
                "semantic_type": "numeric_currency",
                "null_count": 20,
                "cardinality": 50,
                "stats": {"mean": 500.0, "stdev": 100.0, "min": -5.0, "max": 2000.0},
                "sample_values": ["500", "2000", "-5"],
            },
        }, row_count=100)
        results = detect_anomalies([prof])
        for insight in results:
            assert insight.id is not None
            assert insight.insight_type in ("anomaly", "trend")
            assert insight.severity in ("critical", "warning", "info")
            assert 0 <= insight.impact_score <= 100
            assert insight.title
            assert insight.source_tables


# ---------------------------------------------------------------------------
# 8. Manufacturing-specific anomaly detection
# ---------------------------------------------------------------------------


class TestManufacturingAnomalies:
    def test_temperature_below_range(self):
        prof = _Prof("scada", {"furnace_temp": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1500.0, "stdev": 50.0, "min": 1200.0, "max": 1600.0},
            "sample_values": [],
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "below expected range" in i.title.lower()]
        assert len(mfg) == 1
        assert mfg[0].evidence["rule"] == "Furnace Temperature"

    def test_temperature_above_range(self):
        prof = _Prof("scada", {"temperature": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1600.0, "stdev": 50.0, "min": 1500.0, "max": 1800.0},
            "sample_values": [],
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "above expected range" in i.title.lower()]
        assert len(mfg) == 1

    def test_temperature_in_range_no_alert(self):
        prof = _Prof("scada", {"furnace_temp": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1550.0, "stdev": 30.0, "min": 1450.0, "max": 1650.0},
            "sample_values": [],
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "expected range" in i.title.lower()]
        assert len(mfg) == 0

    def test_power_factor_below_range(self):
        prof = _Prof("electrical", {"power_factor": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 20,
            "stats": {"mean": 0.85, "stdev": 0.05, "min": 0.70, "max": 0.95},
            "sample_values": [],
        }}, row_count=100, domain="energy")
        results = _scan_table(prof)
        mfg = [i for i in results if "power factor" in i.title.lower()]
        assert len(mfg) == 1

    def test_kva_above_range(self):
        prof = _Prof("scada", {"kva": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1500.0, "stdev": 200.0, "min": 1000.0, "max": 2500.0},
            "sample_values": [],
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "kva" in i.title.lower()]
        assert len(mfg) == 1

    def test_non_manufacturing_domain_skipped(self):
        """Non-manufacturing domains should not trigger domain-specific rules."""
        prof = _Prof("t", {"temperature": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1500.0, "stdev": 50.0, "min": 1200.0, "max": 1800.0},
            "sample_values": [],
        }}, row_count=100, domain="sales")
        results = _scan_table(prof)
        mfg = [i for i in results if "expected range" in i.title.lower()]
        assert len(mfg) == 0

    def test_no_stats_skipped(self):
        prof = _Prof("scada", {"temperature": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "expected range" in i.title.lower()]
        assert len(mfg) == 0

    def test_both_below_and_above(self):
        """Min below range AND max above range should give 2 insights."""
        prof = _Prof("scada", {"temperature": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1500.0, "stdev": 200.0, "min": 1000.0, "max": 1800.0},
            "sample_values": [],
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "expected range" in i.title.lower()]
        assert len(mfg) == 2

    def test_manufacturing_rules_count(self):
        """We should have at least 3 manufacturing rules defined."""
        assert len(_MANUFACTURING_RANGES) >= 3

    def test_manufacturing_anomaly_severity(self):
        """Manufacturing anomalies should be warning severity."""
        prof = _Prof("scada", {"temp": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1500.0, "min": 1000.0, "max": 1600.0},
            "sample_values": [],
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "expected range" in i.title.lower()]
        for ins in mfg:
            assert ins.severity == "warning"
            assert ins.impact_score == 55

    def test_exact_boundary_not_flagged(self):
        """Values exactly at range boundaries should not be flagged."""
        prof = _Prof("scada", {"temperature": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 50,
            "stats": {"mean": 1550.0, "min": 1400.0, "max": 1700.0},
            "sample_values": [],
        }}, row_count=100, domain="manufacturing")
        results = _scan_table(prof)
        mfg = [i for i in results if "expected range" in i.title.lower()]
        assert len(mfg) == 0

    def test_energy_domain_triggers_mfg_rules(self):
        """domain='energy' should also trigger manufacturing rules."""
        prof = _Prof("meters", {"pf": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 20,
            "stats": {"mean": 0.90, "min": 0.75, "max": 0.99},
            "sample_values": [],
        }}, row_count=100, domain="energy")
        results = _scan_table(prof)
        mfg = [i for i in results if "expected range" in i.title.lower()]
        assert len(mfg) >= 1


# ---------------------------------------------------------------------------
# 9. Edge cases and multi-anomaly detection
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_non_numeric_sample_values_ignored(self):
        prof = _Prof("t", {"val": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 10,
            "stats": {"mean": 10.0, "stdev": 2.0, "min": 1.0, "max": 100.0},
            "sample_values": ["abc", "def", "100"],
        }})
        results = _scan_table(prof)
        outlier = [i for i in results if "outlier" in i.title.lower()]
        assert len(outlier) == 1

    def test_missing_stats_no_crash(self):
        prof = _Prof("t", {"val": {
            "semantic_type": "numeric_metric",
            "null_count": 0,
            "cardinality": 10,
        }})
        results = _scan_table(prof)
        # No crash, and no outlier insight since no stats
        outlier = [i for i in results if "outlier" in i.title.lower()]
        assert len(outlier) == 0

    def test_multiple_columns_multiple_anomalies(self):
        """A profile with many issues should produce multiple insights."""
        prof = _Prof("messy", {
            "amount": {
                "semantic_type": "numeric_currency",
                "null_count": 40,
                "cardinality": 50,
                "stats": {"mean": 500.0, "stdev": 100.0, "min": -10.0, "max": 2000.0},
                "sample_values": ["500", "2000", "-10"],
            },
            "status": {
                "semantic_type": "categorical",
                "null_count": 0,
                "cardinality": 1,
                "sample_values": ["active"],
            },
            "date": {
                "semantic_type": "temporal",
                "null_count": 0,
                "cardinality": 30,
            },
        }, row_count=100)
        results = _scan_table(prof)
        # Should have: null spike, outlier, negative currency, constant, trend
        types = set()
        for r in results:
            if "null" in r.title.lower():
                types.add("null")
            if "outlier" in r.title.lower():
                types.add("outlier")
            if "negative" in r.title.lower():
                types.add("negative")
            if "constant" in r.title.lower():
                types.add("constant")
            if r.insight_type == "trend":
                types.add("trend")
        assert len(types) >= 4

    def test_percentage_both_violations(self):
        """Both min < 0 and max > 100 should produce one insight."""
        prof = _Prof("t", {"pct": {
            "semantic_type": "numeric_percentage",
            "null_count": 0,
            "cardinality": 20,
            "stats": {"mean": 50.0, "min": -10.0, "max": 150.0},
            "sample_values": [],
        }})
        results = _scan_table(prof)
        pct = [i for i in results if "percentage" in i.title.lower()]
        assert len(pct) == 1  # Only one insight per column

    def test_all_insights_have_id(self):
        prof = _Prof("sales", {
            "date": {"semantic_type": "temporal", "null_count": 0, "cardinality": 30},
            "rev": {"semantic_type": "numeric_currency", "null_count": 15, "cardinality": 80,
                    "stats": {"mean": 1000, "stdev": 200, "min": -5, "max": 5000},
                    "sample_values": ["1000", "5000", "-5"]},
        }, row_count=100)
        results = _scan_table(prof)
        for ins in results:
            assert ins.id is not None
            assert len(ins.id) > 10  # UUID format
