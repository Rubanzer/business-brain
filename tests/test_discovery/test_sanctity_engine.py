"""Tests for the sanctity engine module."""

from business_brain.db.discovery_models import TableProfile
from business_brain.db.v3_models import MetricThreshold, SanctityIssue
from business_brain.discovery.sanctity_engine import _check_threshold


class TestCheckThreshold:
    """Test threshold checking against column stats."""

    def _make_threshold(self, **kwargs):
        t = MetricThreshold()
        t.metric_name = kwargs.get("metric_name", "test_metric")
        t.table_name = kwargs.get("table_name", "test_table")
        t.column_name = kwargs.get("column_name", "test_col")
        t.unit = kwargs.get("unit", "kWh")
        t.normal_min = kwargs.get("normal_min")
        t.normal_max = kwargs.get("normal_max")
        t.warning_min = kwargs.get("warning_min")
        t.warning_max = kwargs.get("warning_max")
        t.critical_min = kwargs.get("critical_min")
        t.critical_max = kwargs.get("critical_max")
        return t

    def test_within_normal_range(self):
        threshold = self._make_threshold(
            normal_min=300, normal_max=500,
            warning_min=200, warning_max=600,
            critical_min=100, critical_max=700,
        )
        stats = {"min": 310, "max": 490}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 0

    def test_critical_min_violation(self):
        threshold = self._make_threshold(
            critical_min=100, critical_max=700,
        )
        stats = {"min": 50, "max": 500}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert "below critical minimum" in issues[0].description

    def test_critical_max_violation(self):
        threshold = self._make_threshold(
            critical_min=100, critical_max=700,
        )
        stats = {"min": 200, "max": 800}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 1
        assert issues[0].severity == "critical"
        assert "exceeds critical maximum" in issues[0].description

    def test_warning_min_violation(self):
        threshold = self._make_threshold(
            normal_min=300, normal_max=500,
            warning_min=200, warning_max=600,
        )
        stats = {"min": 180, "max": 450}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_warning_not_double_counted_with_critical(self):
        threshold = self._make_threshold(
            warning_min=200, warning_max=600,
            critical_min=100, critical_max=700,
        )
        stats = {"min": 50, "max": 500}
        # Should only get critical, not also warning
        issues = _check_threshold("table", "col", stats, threshold)
        critical = [i for i in issues if i.severity == "critical"]
        warning = [i for i in issues if i.severity == "warning"]
        assert len(critical) == 1
        assert len(warning) == 0

    def test_no_thresholds_set(self):
        threshold = self._make_threshold()
        stats = {"min": -100, "max": 1000}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 0

    def test_both_min_and_max_critical(self):
        threshold = self._make_threshold(
            critical_min=0, critical_max=100,
        )
        stats = {"min": -10, "max": 150}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 2
        assert all(i.severity == "critical" for i in issues)

    def test_warning_max_violation(self):
        threshold = self._make_threshold(
            normal_min=300, normal_max=500,
            warning_min=200, warning_max=600,
        )
        stats = {"min": 300, "max": 650}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert "exceeds warning threshold" in issues[0].description

    def test_warning_max_not_double_counted_with_critical(self):
        """If max exceeds critical_max, warning_max should not produce a separate issue."""
        threshold = self._make_threshold(
            warning_max=600,
            critical_max=700,
        )
        stats = {"min": 300, "max": 800}
        issues = _check_threshold("table", "col", stats, threshold)
        critical = [i for i in issues if i.severity == "critical"]
        warning = [i for i in issues if i.severity == "warning"]
        assert len(critical) == 1
        assert len(warning) == 0

    def test_missing_stats_min(self):
        """Stats with no 'min' key should not crash."""
        threshold = self._make_threshold(critical_min=100)
        stats = {"max": 500}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 0

    def test_missing_stats_max(self):
        """Stats with no 'max' key should not crash."""
        threshold = self._make_threshold(critical_max=700)
        stats = {"min": 200}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 0

    def test_exact_boundary_critical_min(self):
        """Value exactly at critical_min should not trigger."""
        threshold = self._make_threshold(critical_min=100)
        stats = {"min": 100, "max": 500}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 0

    def test_exact_boundary_critical_max(self):
        """Value exactly at critical_max should not trigger."""
        threshold = self._make_threshold(critical_max=700)
        stats = {"min": 200, "max": 700}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 0

    def test_just_below_critical_min(self):
        """Value just below critical_min should trigger."""
        threshold = self._make_threshold(critical_min=100)
        stats = {"min": 99.9, "max": 500}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 1

    def test_just_above_critical_max(self):
        """Value just above critical_max should trigger."""
        threshold = self._make_threshold(critical_max=700)
        stats = {"min": 200, "max": 700.1}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 1

    def test_unit_in_description(self):
        """Unit should appear in issue description."""
        threshold = self._make_threshold(
            critical_min=100, unit="°C",
        )
        stats = {"min": 50, "max": 500}
        issues = _check_threshold("table", "col", stats, threshold)
        assert "°C" in issues[0].description

    def test_table_and_col_in_issue(self):
        """Issue should reference correct table and column names."""
        threshold = self._make_threshold(critical_min=0)
        stats = {"min": -10, "max": 100}
        issues = _check_threshold("scada_readings", "furnace_temp", stats, threshold)
        assert issues[0].table_name == "scada_readings"
        assert issues[0].column_name == "furnace_temp"

    def test_all_four_violations_at_once(self):
        """Warning + critical on both min and max sides."""
        threshold = self._make_threshold(
            normal_min=300, normal_max=500,
            warning_min=200, warning_max=600,
            critical_min=100, critical_max=700,
        )
        # Min below critical, max above critical
        stats = {"min": 50, "max": 800}
        issues = _check_threshold("table", "col", stats, threshold)
        # Should get 2 critical (min + max), but NOT warning (suppressed by critical)
        assert len(issues) == 2
        assert all(i.severity == "critical" for i in issues)

    def test_expected_range_in_warning(self):
        """Warning issues should include the normal range in expected_range."""
        threshold = self._make_threshold(
            normal_min=300, normal_max=500,
            warning_min=200, warning_max=600,
        )
        stats = {"min": 180, "max": 400}
        issues = _check_threshold("table", "col", stats, threshold)
        assert len(issues) == 1
        assert "300" in issues[0].expected_range
        assert "500" in issues[0].expected_range
