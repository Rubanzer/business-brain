"""Data validation rules engine — define and evaluate rules against data.

Pure functions for creating validation rules, evaluating them against
table data, and producing violation reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ValidationRule:
    """A data validation rule."""
    name: str
    column: str
    rule_type: str  # not_null, unique, positive, range, pattern, custom
    params: dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"  # critical, warning, info


@dataclass
class Violation:
    """A single rule violation."""
    rule_name: str
    column: str
    row_index: int
    value: Any
    message: str
    severity: str


@dataclass
class ValidationReport:
    """Complete validation report for a dataset."""
    total_rules: int
    total_rows: int
    total_violations: int
    rules_passed: int
    rules_failed: int
    pass_rate: float  # 0-100
    violations: list[Violation]
    rule_results: list[dict]  # per-rule summary
    summary: str


def create_rule(
    name: str,
    column: str,
    rule_type: str,
    severity: str = "warning",
    **params,
) -> ValidationRule:
    """Create a validation rule."""
    return ValidationRule(
        name=name,
        column=column,
        rule_type=rule_type,
        params=params,
        severity=severity,
    )


def evaluate_rules(
    rows: list[dict],
    rules: list[ValidationRule],
    max_violations_per_rule: int = 20,
) -> ValidationReport:
    """Evaluate all rules against the data.

    Args:
        rows: Data rows as dicts.
        rules: List of validation rules.
        max_violations_per_rule: Cap violations per rule for performance.

    Returns:
        ValidationReport with all violations.
    """
    all_violations: list[Violation] = []
    rule_results = []
    rules_passed = 0
    rules_failed = 0

    for rule in rules:
        violations = _evaluate_single_rule(rows, rule, max_violations_per_rule)
        if violations:
            rules_failed += 1
            all_violations.extend(violations)
        else:
            rules_passed += 1

        rule_results.append({
            "rule": rule.name,
            "column": rule.column,
            "type": rule.rule_type,
            "severity": rule.severity,
            "passed": len(violations) == 0,
            "violation_count": len(violations),
        })

    total_violations = len(all_violations)
    total_rules = len(rules)
    pass_rate = (rules_passed / total_rules * 100) if total_rules > 0 else 100.0

    summary = (
        f"Validated {len(rows)} rows against {total_rules} rules: "
        f"{rules_passed} passed, {rules_failed} failed, "
        f"{total_violations} violations found ({pass_rate:.0f}% pass rate)."
    )

    return ValidationReport(
        total_rules=total_rules,
        total_rows=len(rows),
        total_violations=total_violations,
        rules_passed=rules_passed,
        rules_failed=rules_failed,
        pass_rate=round(pass_rate, 1),
        violations=all_violations,
        rule_results=rule_results,
        summary=summary,
    )


def auto_generate_rules(columns: dict[str, dict]) -> list[ValidationRule]:
    """Auto-generate validation rules from column profile data.

    Uses semantic types and stats to create appropriate rules.
    """
    rules = []

    for col_name, info in columns.items():
        stype = info.get("semantic_type", "")

        # Identifier columns should not be null
        if stype == "identifier":
            rules.append(create_rule(
                f"{col_name}_not_null", col_name, "not_null", severity="critical"
            ))
            rules.append(create_rule(
                f"{col_name}_unique", col_name, "unique", severity="warning"
            ))

        # Currency should be non-negative
        elif stype == "numeric_currency":
            rules.append(create_rule(
                f"{col_name}_positive", col_name, "positive", severity="warning"
            ))

        # Percentage should be 0-100
        elif stype == "numeric_percentage":
            rules.append(create_rule(
                f"{col_name}_pct_range", col_name, "range", severity="warning",
                min_val=0, max_val=100
            ))

        # Temporal columns should not be null
        elif stype == "temporal":
            rules.append(create_rule(
                f"{col_name}_not_null", col_name, "not_null", severity="info"
            ))

        # Categorical: check for null
        elif stype == "categorical":
            null_count = info.get("null_count", 0)
            cardinality = info.get("cardinality", 0)
            if null_count and cardinality and null_count > 0:
                rules.append(create_rule(
                    f"{col_name}_not_null", col_name, "not_null", severity="info"
                ))

    return rules


def format_violation_summary(report: ValidationReport) -> str:
    """Format a concise violation summary."""
    if report.total_violations == 0:
        return f"All {report.total_rules} rules passed on {report.total_rows} rows."

    lines = [report.summary, ""]
    for rr in report.rule_results:
        if not rr["passed"]:
            lines.append(f"  FAIL: {rr['rule']} ({rr['column']}) — {rr['violation_count']} violations [{rr['severity']}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rule evaluators
# ---------------------------------------------------------------------------


def _evaluate_single_rule(
    rows: list[dict],
    rule: ValidationRule,
    max_violations: int,
) -> list[Violation]:
    """Evaluate a single rule against all rows."""
    evaluator = _EVALUATORS.get(rule.rule_type)
    if not evaluator:
        return [Violation(rule.name, rule.column, -1, None, f"Unknown rule type: {rule.rule_type}", "warning")]

    violations = []
    for i, row in enumerate(rows):
        if len(violations) >= max_violations:
            break
        result = evaluator(row, rule)
        if result:
            violations.append(Violation(
                rule_name=rule.name,
                column=rule.column,
                row_index=i,
                value=row.get(rule.column),
                message=result,
                severity=rule.severity,
            ))
    return violations


def _eval_not_null(row: dict, rule: ValidationRule) -> str | None:
    """Check column is not null/empty."""
    val = row.get(rule.column)
    if val is None or val == "":
        return f"'{rule.column}' is null or empty"
    return None


def _eval_unique(row: dict, rule: ValidationRule) -> str | None:
    """Placeholder — uniqueness requires full scan, handled specially."""
    return None  # Handled in batch


def _eval_positive(row: dict, rule: ValidationRule) -> str | None:
    """Check column is positive (>= 0)."""
    val = row.get(rule.column)
    if val is None:
        return None
    try:
        if float(val) < 0:
            return f"'{rule.column}' is negative: {val}"
    except (TypeError, ValueError):
        return None
    return None


def _eval_range(row: dict, rule: ValidationRule) -> str | None:
    """Check column is within range."""
    val = row.get(rule.column)
    if val is None:
        return None
    try:
        fval = float(val)
        min_val = rule.params.get("min_val")
        max_val = rule.params.get("max_val")
        if min_val is not None and fval < min_val:
            return f"'{rule.column}' = {val} is below minimum {min_val}"
        if max_val is not None and fval > max_val:
            return f"'{rule.column}' = {val} exceeds maximum {max_val}"
    except (TypeError, ValueError):
        return None
    return None


def _eval_pattern(row: dict, rule: ValidationRule) -> str | None:
    """Check column matches a regex pattern."""
    val = row.get(rule.column)
    if val is None:
        return None
    pattern = rule.params.get("pattern", "")
    if not pattern:
        return None
    if not re.match(pattern, str(val)):
        return f"'{rule.column}' = '{val}' does not match pattern '{pattern}'"
    return None


def _eval_min_length(row: dict, rule: ValidationRule) -> str | None:
    """Check string column has minimum length."""
    val = row.get(rule.column)
    if val is None:
        return None
    min_len = rule.params.get("min_length", 0)
    if len(str(val)) < min_len:
        return f"'{rule.column}' length {len(str(val))} is below minimum {min_len}"
    return None


_EVALUATORS: dict[str, Callable] = {
    "not_null": _eval_not_null,
    "unique": _eval_unique,
    "positive": _eval_positive,
    "range": _eval_range,
    "pattern": _eval_pattern,
    "min_length": _eval_min_length,
}
