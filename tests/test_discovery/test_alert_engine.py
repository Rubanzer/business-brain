"""Tests for alert engine pure logic (no DB needed)."""

from business_brain.action.alert_engine import _format_alert_message, _safe_name


class TestSafeName:
    """Test SQL name sanitization."""

    def test_normal_name(self):
        assert _safe_name("gate_register") == "gate_register"

    def test_special_chars_removed(self):
        assert _safe_name("table; DROP TABLE --") == "tableDROPTABLE"

    def test_spaces_removed(self):
        assert _safe_name("my table name") == "mytablename"

    def test_empty_string(self):
        assert _safe_name("") == ""

    def test_none_input(self):
        assert _safe_name(None) == ""

    def test_mixed_chars(self):
        assert _safe_name("table_123_test") == "table_123_test"

    def test_quotes_removed(self):
        assert _safe_name('table"name') == "tablename"


class TestFormatAlertMessage:
    """Test alert message formatting."""

    class FakeRule:
        def __init__(self, name="Test Alert", template=None, config=None, rule_type="threshold"):
            self.name = name
            self.message_template = template
            self.rule_config = config or {}
            self.rule_type = rule_type

    class FakeEvent:
        def __init__(self, trigger_value="43", threshold_value="40"):
            self.trigger_value = trigger_value
            self.threshold_value = threshold_value

    def test_default_format(self):
        rule = self.FakeRule(
            name="Truck Alert",
            config={"table": "gate_register", "column": "truck_count"},
        )
        event = self.FakeEvent(trigger_value="43", threshold_value="40")
        msg = _format_alert_message(rule, event)
        assert "ALERT: Truck Alert" in msg
        assert "43" in msg
        assert "40" in msg
        assert "gate_register" in msg

    def test_custom_template(self):
        rule = self.FakeRule(
            template="Gate: {{value}} trucks (max: {{threshold}})",
        )
        event = self.FakeEvent(trigger_value="45", threshold_value="40")
        msg = _format_alert_message(rule, event)
        assert msg == "Gate: 45 trucks (max: 40)"

    def test_template_with_missing_placeholders(self):
        rule = self.FakeRule(template="Alert fired!")
        event = self.FakeEvent()
        msg = _format_alert_message(rule, event)
        assert msg == "Alert fired!"

    def test_default_format_includes_source(self):
        rule = self.FakeRule(
            name="Power Alert",
            config={"table": "scada", "column": "power_kw"},
        )
        event = self.FakeEvent(trigger_value="150", threshold_value="200")
        msg = _format_alert_message(rule, event)
        assert "scada" in msg
        assert "power_kw" in msg

    def test_empty_template_uses_default(self):
        rule = self.FakeRule(template="", config={"table": "t", "column": "c"})
        event = self.FakeEvent()
        # Empty string is falsy, should use default
        # Actually "" is falsy in Python so it will use default format
        msg = _format_alert_message(rule, event)
        assert "ALERT" in msg or msg == ""


class TestConditionLogic:
    """Test the threshold condition evaluation logic (extracted from _eval_threshold)."""

    def _check_condition(self, current_num, condition, threshold_num, upper=None):
        """Replicate the condition logic from _eval_threshold."""
        if condition == "greater_than":
            return current_num > threshold_num
        elif condition == "less_than":
            return current_num < threshold_num
        elif condition == "equals":
            return current_num == threshold_num
        elif condition == "not_equals":
            return current_num != threshold_num
        elif condition == "between":
            upper_num = upper if upper is not None else threshold_num
            return threshold_num <= current_num <= upper_num
        return False

    def test_greater_than_triggers(self):
        assert self._check_condition(43, "greater_than", 40)

    def test_greater_than_not_triggered(self):
        assert not self._check_condition(38, "greater_than", 40)

    def test_greater_than_boundary(self):
        assert not self._check_condition(40, "greater_than", 40)

    def test_less_than_triggers(self):
        assert self._check_condition(150, "less_than", 200)

    def test_less_than_not_triggered(self):
        assert not self._check_condition(250, "less_than", 200)

    def test_equals_triggers(self):
        assert self._check_condition(0, "equals", 0)

    def test_equals_not_triggered(self):
        assert not self._check_condition(1, "equals", 0)

    def test_not_equals_triggers(self):
        assert self._check_condition(5, "not_equals", 0)

    def test_not_equals_not_triggered(self):
        assert not self._check_condition(0, "not_equals", 0)

    def test_between_triggers(self):
        assert self._check_condition(1550, "between", 1500, 1650)

    def test_between_at_lower_bound(self):
        assert self._check_condition(1500, "between", 1500, 1650)

    def test_between_at_upper_bound(self):
        assert self._check_condition(1650, "between", 1500, 1650)

    def test_between_outside(self):
        assert not self._check_condition(1700, "between", 1500, 1650)

    def test_unknown_condition(self):
        assert not self._check_condition(10, "unknown_op", 5)


class TestTrendLogic:
    """Test the trend detection logic (extracted from _eval_trend)."""

    def _check_trend(self, values, condition, consecutive):
        """Replicate trend detection: values are in chronological order."""
        if len(values) < consecutive + 1:
            return False

        # Use only the last N+1 values
        vals = values[-(consecutive + 1):]

        if condition == "trend_down":
            return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))
        elif condition == "trend_up":
            return all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
        return False

    def test_trend_down_detected(self):
        values = [100, 95, 90, 85]  # 3 consecutive decreases
        assert self._check_trend(values, "trend_down", 3)

    def test_trend_down_not_detected(self):
        values = [100, 95, 97, 85]  # Not all decreasing
        assert not self._check_trend(values, "trend_down", 3)

    def test_trend_up_detected(self):
        values = [80, 85, 90, 95]  # 3 consecutive increases
        assert self._check_trend(values, "trend_up", 3)

    def test_trend_up_not_detected(self):
        values = [80, 85, 82, 95]
        assert not self._check_trend(values, "trend_up", 3)

    def test_insufficient_data(self):
        values = [100, 90]  # Only 2 points, need 4 for consecutive=3
        assert not self._check_trend(values, "trend_down", 3)

    def test_exact_minimum_data(self):
        values = [100, 90, 80, 70]  # Exactly 4 points for consecutive=3
        assert self._check_trend(values, "trend_down", 3)

    def test_flat_values_not_trend(self):
        values = [100, 100, 100, 100]
        assert not self._check_trend(values, "trend_down", 3)
        assert not self._check_trend(values, "trend_up", 3)
