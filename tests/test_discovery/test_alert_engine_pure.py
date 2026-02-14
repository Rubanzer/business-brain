"""Tests for alert engine pure functions."""

from business_brain.action.alert_engine import _format_alert_message, _safe_name


class _FakeRule:
    """Minimal stand-in for AlertRule."""

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "Test Alert")
        self.message_template = kwargs.get("message_template")
        self.rule_config = kwargs.get("rule_config", {})
        self.rule_type = kwargs.get("rule_type", "threshold")


class _FakeEvent:
    """Minimal stand-in for AlertEvent."""

    def __init__(self, **kwargs):
        self.trigger_value = kwargs.get("trigger_value", "42")
        self.threshold_value = kwargs.get("threshold_value", "100")


class TestSafeName:
    def test_basic(self):
        assert _safe_name("my_table") == "my_table"

    def test_strips_special(self):
        assert _safe_name("table-name.1") == "tablename1"

    def test_sql_injection_attempt(self):
        assert _safe_name('"; DROP TABLE users; --') == "DROPTABLEusers"

    def test_empty_string(self):
        assert _safe_name("") == ""

    def test_none(self):
        assert _safe_name(None) == ""

    def test_spaces(self):
        assert _safe_name("my table") == "mytable"

    def test_underscores_preserved(self):
        assert _safe_name("a_b_c") == "a_b_c"

    def test_alphanumeric_preserved(self):
        assert _safe_name("table123") == "table123"

    def test_mixed_case_preserved(self):
        assert _safe_name("MyTable") == "MyTable"


class TestFormatAlertMessage:
    def test_default_format(self):
        rule = _FakeRule(
            name="High Temperature",
            rule_config={"table": "sensors", "column": "temp"},
        )
        event = _FakeEvent(trigger_value="150", threshold_value="100")
        msg = _format_alert_message(rule, event)
        assert "ALERT: High Temperature" in msg
        assert "Current: 150" in msg
        assert "Threshold: 100" in msg
        assert "sensors" in msg
        assert "temp" in msg

    def test_custom_template(self):
        rule = _FakeRule(
            message_template="Value is {{value}}, threshold was {{threshold}}",
        )
        event = _FakeEvent(trigger_value="42", threshold_value="100")
        msg = _format_alert_message(rule, event)
        assert msg == "Value is 42, threshold was 100"

    def test_template_empty_values(self):
        rule = _FakeRule(
            message_template="V={{value}} T={{threshold}}",
        )
        event = _FakeEvent(trigger_value="", threshold_value="")
        msg = _format_alert_message(rule, event)
        assert msg == "V= T="

    def test_template_none_values(self):
        rule = _FakeRule(
            message_template="V={{value}} T={{threshold}}",
        )
        event = _FakeEvent()
        event.trigger_value = None
        event.threshold_value = None
        msg = _format_alert_message(rule, event)
        assert msg == "V= T="

    def test_default_format_missing_config(self):
        rule = _FakeRule(rule_config={})
        event = _FakeEvent()
        msg = _format_alert_message(rule, event)
        assert "unknown" in msg

    def test_default_format_has_timestamp(self):
        rule = _FakeRule(rule_config={"table": "t", "column": "c"})
        event = _FakeEvent()
        msg = _format_alert_message(rule, event)
        assert "UTC" in msg

    def test_template_with_no_placeholders(self):
        rule = _FakeRule(message_template="Static alert message")
        event = _FakeEvent()
        msg = _format_alert_message(rule, event)
        assert msg == "Static alert message"

    def test_template_takes_precedence(self):
        """When template is set, default format is not used."""
        rule = _FakeRule(
            name="Alert",
            message_template="Custom: {{value}}",
            rule_config={"table": "t", "column": "c"},
        )
        event = _FakeEvent(trigger_value="99")
        msg = _format_alert_message(rule, event)
        assert msg == "Custom: 99"
        assert "ALERT:" not in msg

    def test_default_format_includes_source(self):
        rule = _FakeRule(rule_config={"table": "production", "column": "output"})
        event = _FakeEvent()
        msg = _format_alert_message(rule, event)
        assert "production.output" in msg
