"""What-if scenario engine — simple scenario modeling and comparison.

Pure functions for defining scenarios, computing outcomes, and
comparing alternatives. No DB or LLM dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Scenario:
    """A what-if scenario definition."""
    name: str
    parameters: dict[str, float]  # parameter_name -> value
    description: str = ""


@dataclass
class ScenarioOutcome:
    """Computed outcome for a scenario."""
    scenario_name: str
    result: float
    parameters: dict[str, float]
    breakdown: dict[str, float]  # intermediate calculation results
    interpretation: str


@dataclass
class ScenarioComparison:
    """Comparison of multiple scenario outcomes."""
    baseline_name: str
    outcomes: list[ScenarioOutcome]
    best_scenario: str
    worst_scenario: str
    range_min: float
    range_max: float
    sensitivity: dict[str, float]  # parameter -> impact score
    summary: str


def evaluate_scenario(
    scenario: Scenario,
    formula: str,
    base_values: dict[str, float] | None = None,
) -> ScenarioOutcome:
    """Evaluate a single scenario against a formula.

    Args:
        scenario: Scenario with parameter overrides.
        formula: Simple math expression using parameter names.
                 Supports: +, -, *, /, (, ), and parameter names.
        base_values: Default values for parameters not in the scenario.

    Returns:
        ScenarioOutcome with computed result.
    """
    params = dict(base_values or {})
    params.update(scenario.parameters)

    try:
        result = _safe_eval(formula, params)
    except Exception as e:
        return ScenarioOutcome(
            scenario_name=scenario.name,
            result=0.0,
            parameters=params,
            breakdown={},
            interpretation=f"Error evaluating formula: {e}",
        )

    interpretation = f"With parameters {_format_params(scenario.parameters)}, result = {result:,.2f}"

    return ScenarioOutcome(
        scenario_name=scenario.name,
        result=round(result, 4),
        parameters=params,
        breakdown={"result": result},
        interpretation=interpretation,
    )


def compare_scenarios(
    scenarios: list[Scenario],
    formula: str,
    base_values: dict[str, float] | None = None,
) -> ScenarioComparison:
    """Evaluate and compare multiple scenarios.

    Args:
        scenarios: List of scenario definitions.
        formula: Math expression to evaluate.
        base_values: Default parameter values.

    Returns:
        ScenarioComparison with all outcomes ranked.
    """
    if not scenarios:
        return ScenarioComparison(
            baseline_name="",
            outcomes=[],
            best_scenario="",
            worst_scenario="",
            range_min=0,
            range_max=0,
            sensitivity={},
            summary="No scenarios provided.",
        )

    outcomes = [evaluate_scenario(s, formula, base_values) for s in scenarios]

    # Sort by result
    sorted_outcomes = sorted(outcomes, key=lambda o: o.result, reverse=True)
    best = sorted_outcomes[0].scenario_name
    worst = sorted_outcomes[-1].scenario_name
    range_min = sorted_outcomes[-1].result
    range_max = sorted_outcomes[0].result

    # Sensitivity analysis
    sensitivity = _compute_sensitivity(scenarios, formula, base_values or {})

    baseline = scenarios[0].name
    summary = (
        f"Compared {len(scenarios)} scenarios. "
        f"Best: {best} ({range_max:,.2f}), Worst: {worst} ({range_min:,.2f}). "
        f"Range: {range_max - range_min:,.2f}."
    )

    return ScenarioComparison(
        baseline_name=baseline,
        outcomes=sorted_outcomes,
        best_scenario=best,
        worst_scenario=worst,
        range_min=round(range_min, 4),
        range_max=round(range_max, 4),
        sensitivity=sensitivity,
        summary=summary,
    )


def breakeven_analysis(
    formula: str,
    variable: str,
    base_values: dict[str, float],
    target: float = 0.0,
    search_range: tuple[float, float] = (-1000, 1000),
    steps: int = 100,
) -> dict:
    """Find the value of a variable that makes the formula equal to target.

    Simple grid search approach.
    """
    lo, hi = search_range
    step_size = (hi - lo) / steps

    best_val = lo
    best_diff = float("inf")

    for i in range(steps + 1):
        test_val = lo + i * step_size
        params = dict(base_values)
        params[variable] = test_val
        try:
            result = _safe_eval(formula, params)
            diff = abs(result - target)
            if diff < best_diff:
                best_diff = diff
                best_val = test_val
        except Exception:
            continue

    return {
        "variable": variable,
        "breakeven_value": round(best_val, 4),
        "target": target,
        "achieved_result": round(_safe_eval(formula, {**base_values, variable: best_val}), 4),
        "precision": round(best_diff, 6),
    }


def sensitivity_table(
    formula: str,
    variable: str,
    base_values: dict[str, float],
    variations: list[float] | None = None,
) -> list[dict]:
    """Generate a sensitivity table showing how results change with one variable.

    Args:
        formula: Expression to evaluate.
        variable: Variable to vary.
        base_values: Base parameter values.
        variations: List of multipliers (e.g., [0.8, 0.9, 1.0, 1.1, 1.2]).

    Returns:
        List of {variable_value, result, change_pct} dicts.
    """
    if variations is None:
        variations = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]

    base_val = base_values.get(variable, 0)
    base_result = _safe_eval(formula, base_values)
    table = []

    for mult in variations:
        test_val = base_val * mult
        params = dict(base_values)
        params[variable] = test_val
        try:
            result = _safe_eval(formula, params)
            change_pct = ((result - base_result) / abs(base_result) * 100) if base_result != 0 else 0
            table.append({
                "variable_value": round(test_val, 4),
                "multiplier": mult,
                "result": round(result, 4),
                "change_pct": round(change_pct, 2),
            })
        except Exception:
            continue

    return table


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_eval(formula: str, params: dict[str, float]) -> float:
    """Safely evaluate a math formula with parameters.

    Only allows: numbers, +, -, *, /, (, ), and known parameter names.
    """
    # Replace parameter names with values
    expr = formula
    for name, value in sorted(params.items(), key=lambda x: -len(x[0])):
        expr = expr.replace(name, str(float(value)))

    # Validate: only allow digits, operators, spaces, dots, parens
    allowed = set("0123456789.+-*/() ")
    if not all(c in allowed for c in expr):
        raise ValueError(f"Invalid characters in expression: {expr}")

    # Evaluate
    return float(eval(expr))  # safe because we've validated the character set


def _format_params(params: dict[str, float]) -> str:
    """Format parameters as readable string."""
    return ", ".join(f"{k}={v:,.2f}" for k, v in params.items())


def _compute_sensitivity(
    scenarios: list[Scenario],
    formula: str,
    base_values: dict[str, float],
) -> dict[str, float]:
    """Compute which parameters have the most impact on results."""
    if not base_values:
        return {}

    base_result = _safe_eval(formula, base_values)
    impacts: dict[str, float] = {}

    for param, value in base_values.items():
        if value == 0:
            continue
        # Test ±10% change
        params_up = dict(base_values)
        params_up[param] = value * 1.1
        params_down = dict(base_values)
        params_down[param] = value * 0.9

        try:
            result_up = _safe_eval(formula, params_up)
            result_down = _safe_eval(formula, params_down)
            impact = abs(result_up - result_down) / abs(base_result) * 100 if base_result != 0 else 0
            impacts[param] = round(impact, 2)
        except Exception:
            continue

    return dict(sorted(impacts.items(), key=lambda x: -x[1]))
