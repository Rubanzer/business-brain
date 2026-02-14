"""Tests for alert parser helper functions (no LLM needed)."""

from business_brain.action.alert_parser import build_confirmation_message


class TestBuildConfirmationMessage:
    """Test confirmation message generation."""

    def test_threshold_greater_than(self):
        rule = {
            "name": "Truck Count Alert",
            "table": "gate_register",
            "column": "truck_count",
            "condition": "greater_than",
            "threshold": 40,
            "notification_channel": "telegram",
        }
        msg = build_confirmation_message(rule)
        assert "truck_count" in msg
        assert "gate_register" in msg
        assert "exceeds 40" in msg
        assert "Telegram" in msg

    def test_threshold_less_than(self):
        rule = {
            "name": "Power Drop Alert",
            "table": "scada_readings",
            "column": "power_kw",
            "condition": "less_than",
            "threshold": 200,
            "notification_channel": "feed",
        }
        msg = build_confirmation_message(rule)
        assert "drops below 200" in msg
        assert "Feed" in msg

    def test_trend_down(self):
        rule = {
            "name": "Production Drop Alert",
            "table": "production",
            "column": "output",
            "condition": "trend_down",
            "threshold": 3,
            "notification_channel": "telegram",
        }
        msg = build_confirmation_message(rule)
        assert "decreases for 3" in msg

    def test_absence(self):
        rule = {
            "name": "SCADA Absence",
            "table": "scada_readings",
            "column": "power_kw",
            "condition": "absent",
            "threshold": 30,
            "notification_channel": "telegram",
        }
        msg = build_confirmation_message(rule)
        assert "no new data" in msg
        assert "30 minutes" in msg

    def test_between_condition(self):
        rule = {
            "name": "Temperature Range",
            "table": "furnace",
            "column": "temp",
            "condition": "between",
            "threshold": 1500,
            "threshold_upper": 1650,
            "notification_channel": "feed",
        }
        msg = build_confirmation_message(rule)
        assert "between 1500 and 1650" in msg

    def test_missing_fields(self):
        rule = {}
        msg = build_confirmation_message(rule)
        assert "unknown" in msg.lower()

    def test_equals_condition(self):
        rule = {
            "name": "Status Alert",
            "table": "machines",
            "column": "status",
            "condition": "equals",
            "threshold": 0,
            "notification_channel": "feed",
        }
        msg = build_confirmation_message(rule)
        assert "equals 0" in msg

    def test_not_equals_condition(self):
        rule = {
            "name": "Status Change",
            "table": "machines",
            "column": "status",
            "condition": "not_equals",
            "threshold": "active",
            "notification_channel": "telegram",
        }
        msg = build_confirmation_message(rule)
        assert "is not active" in msg

    def test_trend_up_condition(self):
        rule = {
            "name": "Rising Temp",
            "table": "furnace",
            "column": "temperature",
            "condition": "trend_up",
            "threshold": 5,
            "notification_channel": "feed",
        }
        msg = build_confirmation_message(rule)
        assert "increases for 5" in msg

    def test_alert_name_in_message(self):
        rule = {
            "name": "My Custom Alert",
            "table": "t",
            "column": "c",
            "condition": "greater_than",
            "threshold": 100,
        }
        msg = build_confirmation_message(rule)
        assert "My Custom Alert" in msg

    def test_unknown_condition_fallback(self):
        rule = {
            "name": "Test",
            "table": "t",
            "column": "c",
            "condition": "custom_condition",
            "threshold": 42,
        }
        msg = build_confirmation_message(rule)
        assert "custom_condition" in msg
        assert "42" in msg
