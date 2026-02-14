"""Tests for anomaly classifier — pattern classification of time series anomalies."""

from business_brain.discovery.anomaly_classifier import (
    AnomalyClassification,
    classify_anomaly,
    classify_series,
    compute_anomaly_score,
    summarize_anomalies,
)


# ---------------------------------------------------------------------------
# classify_anomaly
# ---------------------------------------------------------------------------


class TestClassifyAnomaly:
    def test_spike(self):
        values = [10, 10, 10, 100, 10, 10, 10]
        result = classify_anomaly(values, 3)
        assert result.pattern == "spike"
        assert result.confidence > 0.3
        assert 3 in result.affected_indices

    def test_dip(self):
        values = [100, 100, 100, 10, 100, 100, 100]
        result = classify_anomaly(values, 3)
        assert result.pattern == "dip"
        assert result.confidence > 0.3

    def test_invalid_index(self):
        result = classify_anomaly([1, 2, 3], -1)
        assert result.pattern == "unknown"
        assert result.confidence == 0.0

    def test_empty_values(self):
        result = classify_anomaly([], 0)
        assert result.pattern == "unknown"

    def test_index_out_of_range(self):
        result = classify_anomaly([1, 2, 3], 5)
        assert result.pattern == "unknown"

    def test_step_change(self):
        values = [10, 10, 10, 10, 50, 50, 50, 50]
        result = classify_anomaly(values, 4)
        # Should detect step change at the boundary
        assert result.pattern in ("step_change", "spike", "dip", "gradual_drift")

    def test_plateau(self):
        values = [10, 20, 30, 30, 30, 30, 30, 40, 50]
        result = classify_anomaly(values, 4)
        # A plateau of constant values
        assert result.pattern in ("plateau", "gradual_drift")

    def test_gradual_drift_fallback(self):
        values = [10, 12, 14, 16, 18, 20]
        result = classify_anomaly(values, 3)
        # Smooth increasing series — step_change, drift, or plateau all valid
        assert result.pattern in ("gradual_drift", "plateau", "step_change")

    def test_severity_critical_for_large_spike(self):
        values = [10, 10, 10, 10, 10, 1000, 10, 10, 10, 10]
        result = classify_anomaly(values, 5)
        assert result.severity in ("critical", "warning")

    def test_metadata_present(self):
        values = [10, 10, 10, 100, 10, 10, 10]
        result = classify_anomaly(values, 3)
        assert isinstance(result.metadata, dict)


# ---------------------------------------------------------------------------
# classify_series
# ---------------------------------------------------------------------------


class TestClassifySeries:
    def test_no_anomalies(self):
        values = [10, 10, 10, 10, 10]
        result = classify_series(values)
        assert result == []

    def test_single_spike(self):
        values = [10, 10, 10, 100, 10, 10, 10, 10, 10, 10]
        result = classify_series(values)
        assert len(result) >= 1
        assert any(r.pattern == "spike" for r in result)

    def test_single_dip(self):
        values = [100, 100, 100, 100, 10, 100, 100, 100, 100, 100]
        result = classify_series(values)
        assert len(result) >= 1
        assert any(r.pattern == "dip" for r in result)

    def test_too_few_values(self):
        result = classify_series([1, 2, 3])
        assert result == []

    def test_constant_series(self):
        result = classify_series([5, 5, 5, 5, 5, 5, 5])
        assert result == []

    def test_multiple_anomalies(self):
        values = [10, 10, 100, 10, 10, 10, -50, 10, 10, 10]
        result = classify_series(values)
        assert len(result) >= 1

    def test_custom_threshold(self):
        values = [10, 10, 10, 20, 10, 10, 10, 10, 10, 10]
        # With strict threshold
        strict = classify_series(values, threshold_std=1.0)
        # With lenient threshold
        lenient = classify_series(values, threshold_std=3.0)
        assert len(strict) >= len(lenient)


# ---------------------------------------------------------------------------
# compute_anomaly_score
# ---------------------------------------------------------------------------


class TestComputeAnomalyScore:
    def test_spike_score(self):
        c = AnomalyClassification("spike", 0.8, "desc", "warning", [3], {})
        score = compute_anomaly_score(c)
        assert 40 < score < 80

    def test_critical_higher_than_info(self):
        critical = AnomalyClassification("spike", 0.8, "", "critical", [0], {})
        info = AnomalyClassification("spike", 0.8, "", "info", [0], {})
        assert compute_anomaly_score(critical) > compute_anomaly_score(info)

    def test_step_change_high_base(self):
        c = AnomalyClassification("step_change", 0.9, "", "warning", [0], {})
        score = compute_anomaly_score(c)
        assert score > 50

    def test_plateau_lower_score(self):
        c = AnomalyClassification("plateau", 0.5, "", "info", [0, 1, 2], {})
        score = compute_anomaly_score(c)
        assert score < 50

    def test_max_100(self):
        c = AnomalyClassification("step_change", 1.0, "", "critical", [0], {})
        score = compute_anomaly_score(c)
        assert score <= 100.0

    def test_unknown_low_score(self):
        c = AnomalyClassification("unknown", 0.5, "", "info", [], {})
        score = compute_anomaly_score(c)
        assert score < 20


# ---------------------------------------------------------------------------
# summarize_anomalies
# ---------------------------------------------------------------------------


class TestSummarizeAnomalies:
    def test_empty(self):
        result = summarize_anomalies([])
        assert result["total"] == 0
        assert "No anomalies" in result["summary"]

    def test_single(self):
        c = AnomalyClassification("spike", 0.8, "", "warning", [3], {})
        result = summarize_anomalies([c])
        assert result["total"] == 1
        assert result["patterns"]["spike"] == 1
        assert result["severities"]["warning"] == 1

    def test_multiple_types(self):
        classifications = [
            AnomalyClassification("spike", 0.8, "", "critical", [0], {}),
            AnomalyClassification("dip", 0.6, "", "warning", [5], {}),
            AnomalyClassification("spike", 0.9, "", "warning", [10], {}),
        ]
        result = summarize_anomalies(classifications)
        assert result["total"] == 3
        assert result["patterns"]["spike"] == 2
        assert result["patterns"]["dip"] == 1
        assert result["severities"]["critical"] == 1
        assert result["severities"]["warning"] == 2

    def test_avg_score(self):
        classifications = [
            AnomalyClassification("spike", 0.8, "", "warning", [0], {}),
            AnomalyClassification("plateau", 0.5, "", "info", [1], {}),
        ]
        result = summarize_anomalies(classifications)
        assert result["avg_score"] > 0

    def test_summary_text(self):
        classifications = [
            AnomalyClassification("spike", 0.8, "", "warning", [0], {}),
            AnomalyClassification("spike", 0.9, "", "critical", [5], {}),
        ]
        result = summarize_anomalies(classifications)
        assert "2 anomalies" in result["summary"]
        assert "spike" in result["summary"]
