"""Tests for data validation rules engine."""

from business_brain.discovery.validation_rules import (
    ValidationRule,
    auto_generate_rules,
    create_rule,
    evaluate_rules,
    format_violation_summary,
)


# ---------------------------------------------------------------------------
# evaluate_rules
# ---------------------------------------------------------------------------


class TestEvaluateRules:
    def test_not_null_pass(self):
        rules = [create_rule("id_nn", "id", "not_null")]
        rows = [{"id": 1}, {"id": 2}]
        report = evaluate_rules(rows, rules)
        assert report.rules_passed == 1
        assert report.total_violations == 0
        assert report.pass_rate == 100.0

    def test_not_null_fail(self):
        rules = [create_rule("id_nn", "id", "not_null")]
        rows = [{"id": 1}, {"id": None}, {"id": ""}]
        report = evaluate_rules(rows, rules)
        assert report.rules_failed == 1
        assert report.total_violations == 2

    def test_positive_pass(self):
        rules = [create_rule("amt_pos", "amount", "positive")]
        rows = [{"amount": 10}, {"amount": 0}]
        report = evaluate_rules(rows, rules)
        assert report.rules_passed == 1

    def test_positive_fail(self):
        rules = [create_rule("amt_pos", "amount", "positive")]
        rows = [{"amount": 10}, {"amount": -5}]
        report = evaluate_rules(rows, rules)
        assert report.rules_failed == 1
        assert report.violations[0].value == -5

    def test_range_pass(self):
        rules = [create_rule("pct", "pct", "range", min_val=0, max_val=100)]
        rows = [{"pct": 50}, {"pct": 0}, {"pct": 100}]
        report = evaluate_rules(rows, rules)
        assert report.rules_passed == 1

    def test_range_fail_below(self):
        rules = [create_rule("pct", "pct", "range", min_val=0, max_val=100)]
        rows = [{"pct": -10}]
        report = evaluate_rules(rows, rules)
        assert report.total_violations == 1
        assert "below minimum" in report.violations[0].message

    def test_range_fail_above(self):
        rules = [create_rule("pct", "pct", "range", min_val=0, max_val=100)]
        rows = [{"pct": 150}]
        report = evaluate_rules(rows, rules)
        assert report.total_violations == 1
        assert "exceeds maximum" in report.violations[0].message

    def test_pattern_pass(self):
        rules = [create_rule("email_pat", "email", "pattern", pattern=r"^\S+@\S+\.\S+$")]
        rows = [{"email": "a@b.com"}]
        report = evaluate_rules(rows, rules)
        assert report.rules_passed == 1

    def test_pattern_fail(self):
        rules = [create_rule("email_pat", "email", "pattern", pattern=r"^\S+@\S+\.\S+$")]
        rows = [{"email": "not-an-email"}]
        report = evaluate_rules(rows, rules)
        assert report.rules_failed == 1

    def test_min_length_pass(self):
        rules = [create_rule("name_len", "name", "min_length", min_length=3)]
        rows = [{"name": "Bob"}, {"name": "Alice"}]
        report = evaluate_rules(rows, rules)
        assert report.rules_passed == 1

    def test_min_length_fail(self):
        rules = [create_rule("name_len", "name", "min_length", min_length=3)]
        rows = [{"name": "AB"}]
        report = evaluate_rules(rows, rules)
        assert report.rules_failed == 1

    def test_multiple_rules(self):
        rules = [
            create_rule("id_nn", "id", "not_null", severity="critical"),
            create_rule("amt_pos", "amount", "positive"),
        ]
        rows = [{"id": 1, "amount": 10}, {"id": None, "amount": -5}]
        report = evaluate_rules(rows, rules)
        assert report.total_rules == 2
        assert report.rules_failed == 2
        assert report.total_violations == 2

    def test_max_violations_per_rule(self):
        rules = [create_rule("nn", "val", "not_null")]
        rows = [{"val": None} for _ in range(100)]
        report = evaluate_rules(rows, rules, max_violations_per_rule=5)
        assert report.total_violations == 5

    def test_empty_rows(self):
        rules = [create_rule("nn", "id", "not_null")]
        report = evaluate_rules([], rules)
        assert report.rules_passed == 1
        assert report.total_violations == 0

    def test_empty_rules(self):
        report = evaluate_rules([{"id": 1}], [])
        assert report.total_rules == 0
        assert report.pass_rate == 100.0

    def test_unknown_rule_type(self):
        rules = [create_rule("bad", "id", "nonexistent")]
        report = evaluate_rules([{"id": 1}], rules)
        assert report.rules_failed == 1

    def test_null_value_in_positive(self):
        rules = [create_rule("pos", "val", "positive")]
        rows = [{"val": None}]
        report = evaluate_rules(rows, rules)
        assert report.rules_passed == 1  # null doesn't fail positive check

    def test_summary_text(self):
        rules = [create_rule("nn", "id", "not_null")]
        rows = [{"id": None}]
        report = evaluate_rules(rows, rules)
        assert "1 rules" in report.summary or "1 failed" in report.summary


# ---------------------------------------------------------------------------
# auto_generate_rules
# ---------------------------------------------------------------------------


class TestAutoGenerateRules:
    def test_identifier_gets_not_null_and_unique(self):
        cols = {"id": {"semantic_type": "identifier"}}
        rules = auto_generate_rules(cols)
        rule_types = {r.rule_type for r in rules}
        assert "not_null" in rule_types
        assert "unique" in rule_types

    def test_currency_gets_positive(self):
        cols = {"amount": {"semantic_type": "numeric_currency"}}
        rules = auto_generate_rules(cols)
        assert any(r.rule_type == "positive" for r in rules)

    def test_percentage_gets_range(self):
        cols = {"pct": {"semantic_type": "numeric_percentage"}}
        rules = auto_generate_rules(cols)
        range_rules = [r for r in rules if r.rule_type == "range"]
        assert len(range_rules) == 1
        assert range_rules[0].params["min_val"] == 0
        assert range_rules[0].params["max_val"] == 100

    def test_temporal_gets_not_null(self):
        cols = {"date": {"semantic_type": "temporal"}}
        rules = auto_generate_rules(cols)
        assert any(r.rule_type == "not_null" and r.column == "date" for r in rules)

    def test_empty_columns(self):
        assert auto_generate_rules({}) == []

    def test_mixed_columns(self):
        cols = {
            "id": {"semantic_type": "identifier"},
            "amount": {"semantic_type": "numeric_currency"},
            "date": {"semantic_type": "temporal"},
        }
        rules = auto_generate_rules(cols)
        assert len(rules) >= 4  # id: not_null + unique, amount: positive, date: not_null


# ---------------------------------------------------------------------------
# format_violation_summary
# ---------------------------------------------------------------------------


class TestFormatViolationSummary:
    def test_all_pass(self):
        rules = [create_rule("nn", "id", "not_null")]
        report = evaluate_rules([{"id": 1}], rules)
        text = format_violation_summary(report)
        assert "passed" in text

    def test_has_violations(self):
        rules = [create_rule("nn", "id", "not_null")]
        report = evaluate_rules([{"id": None}], rules)
        text = format_violation_summary(report)
        assert "FAIL" in text
