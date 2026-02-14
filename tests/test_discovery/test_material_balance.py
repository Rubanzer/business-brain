"""Tests for material balance and mass flow analysis module."""

from business_brain.discovery.material_balance import (
    ComponentMix,
    EntityBalance,
    LeakagePoint,
    MaterialBalanceResult,
    MixResult,
    compute_material_balance,
    compute_mix_analysis,
    detect_material_leakage,
    format_balance_report,
)


# ===================================================================
# compute_material_balance Tests
# ===================================================================


class TestComputeMaterialBalance:
    def test_basic_balance(self):
        rows = [
            {"plant": "P1", "input": 1000, "output": 950},
            {"plant": "P2", "input": 1000, "output": 800},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result is not None
        assert len(result.entities) == 2
        p1 = [e for e in result.entities if e.entity == "P1"][0]
        assert p1.total_input == 1000
        assert p1.total_output == 950
        assert p1.loss == 50
        assert p1.recovery_pct == 95.0
        assert p1.loss_pct == 5.0

    def test_empty_rows_returns_none(self):
        assert compute_material_balance([], "plant", "input", "output") is None

    def test_all_none_values_returns_none(self):
        rows = [{"plant": None, "input": None, "output": None}]
        assert compute_material_balance(rows, "plant", "input", "output") is None

    def test_explicit_loss_column(self):
        rows = [
            {"plant": "P1", "input": 1000, "output": 900, "loss": 80},
        ]
        result = compute_material_balance(
            rows, "plant", "input", "output", loss_column="loss"
        )
        assert result is not None
        e = result.entities[0]
        # Loss comes from loss_column, not input - output
        assert e.loss == 80
        assert e.loss_pct == 8.0

    def test_computed_loss_when_no_loss_column(self):
        rows = [{"plant": "P1", "input": 500, "output": 450}]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result.entities[0].loss == 50

    def test_best_and_worst_recovery_entity(self):
        rows = [
            {"plant": "Good", "input": 100, "output": 98},
            {"plant": "Bad", "input": 100, "output": 60},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result.best_recovery_entity == "Good"
        assert result.worst_recovery_entity == "Bad"

    def test_sorted_by_recovery_descending(self):
        rows = [
            {"plant": "Low", "input": 100, "output": 50},
            {"plant": "High", "input": 100, "output": 99},
            {"plant": "Mid", "input": 100, "output": 75},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        recoveries = [e.recovery_pct for e in result.entities]
        assert recoveries == sorted(recoveries, reverse=True)

    def test_aggregates_multiple_rows_per_entity(self):
        rows = [
            {"plant": "P1", "input": 500, "output": 480},
            {"plant": "P1", "input": 500, "output": 470},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        e = result.entities[0]
        assert e.total_input == 1000
        assert e.total_output == 950
        assert e.loss == 50
        assert e.recovery_pct == 95.0

    def test_overall_totals(self):
        rows = [
            {"plant": "P1", "input": 1000, "output": 900},
            {"plant": "P2", "input": 2000, "output": 1800},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result.total_input == 3000
        assert result.total_output == 2700
        assert result.total_loss == 300
        assert result.overall_recovery_pct == 90.0

    def test_zero_input_no_division_error(self):
        rows = [{"plant": "P1", "input": 0, "output": 0}]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result is not None
        assert result.entities[0].recovery_pct == 0.0
        assert result.entities[0].loss_pct == 0.0

    def test_summary_contains_key_info(self):
        rows = [
            {"plant": "P1", "input": 1000, "output": 950},
            {"plant": "P2", "input": 1000, "output": 800},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert "Material balance" in result.summary
        assert "P1" in result.summary or "P2" in result.summary
        assert "recovery" in result.summary.lower() or "Recovery" in result.summary

    def test_skips_non_numeric_values(self):
        rows = [
            {"plant": "P1", "input": "bad", "output": 900},
            {"plant": "P2", "input": 1000, "output": 950},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result is not None
        assert len(result.entities) == 1
        assert result.entities[0].entity == "P2"

    def test_single_entity(self):
        rows = [{"plant": "Solo", "input": 200, "output": 180}]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result.best_recovery_entity == "Solo"
        assert result.worst_recovery_entity == "Solo"


# ===================================================================
# detect_material_leakage Tests
# ===================================================================


class TestDetectMaterialLeakage:
    def test_basic_leakage(self):
        rows = [
            {"stage": "crushing", "qty": 1000},
            {"stage": "grinding", "qty": 950},
            {"stage": "flotation", "qty": 850},
        ]
        result = detect_material_leakage(rows, "stage", "qty")
        assert len(result) == 2
        # Alphabetical: crushing -> flotation -> grinding
        # With sequence: crushing -> grinding -> flotation
        # Without sequence sorted: crushing, flotation, grinding
        assert result[0].from_stage == "crushing"
        assert result[0].to_stage == "flotation"

    def test_empty_rows(self):
        assert detect_material_leakage([], "stage", "qty") == []

    def test_single_stage_returns_empty(self):
        rows = [{"stage": "crushing", "qty": 1000}]
        assert detect_material_leakage(rows, "stage", "qty") == []

    def test_with_explicit_sequence(self):
        rows = [
            {"stage": "crushing", "qty": 1000},
            {"stage": "grinding", "qty": 950},
            {"stage": "flotation", "qty": 850},
        ]
        result = detect_material_leakage(
            rows, "stage", "qty",
            sequence=["crushing", "grinding", "flotation"],
        )
        assert len(result) == 2
        assert result[0].from_stage == "crushing"
        assert result[0].to_stage == "grinding"
        assert result[0].input_qty == 1000
        assert result[0].output_qty == 950
        assert result[0].loss == 50
        assert result[0].loss_pct == 5.0

        assert result[1].from_stage == "grinding"
        assert result[1].to_stage == "flotation"

    def test_severity_critical(self):
        rows = [
            {"stage": "A", "qty": 1000},
            {"stage": "B", "qty": 800},  # 20% loss -> critical
        ]
        result = detect_material_leakage(rows, "stage", "qty")
        assert result[0].severity == "critical"
        assert result[0].loss_pct == 20.0

    def test_severity_warning(self):
        rows = [
            {"stage": "A", "qty": 1000},
            {"stage": "B", "qty": 920},  # 8% loss -> warning
        ]
        result = detect_material_leakage(rows, "stage", "qty")
        assert result[0].severity == "warning"

    def test_severity_ok(self):
        rows = [
            {"stage": "A", "qty": 1000},
            {"stage": "B", "qty": 970},  # 3% loss -> ok
        ]
        result = detect_material_leakage(rows, "stage", "qty")
        assert result[0].severity == "ok"

    def test_aggregates_multiple_rows_per_stage(self):
        rows = [
            {"stage": "A", "qty": 500},
            {"stage": "A", "qty": 500},
            {"stage": "B", "qty": 450},
            {"stage": "B", "qty": 450},
        ]
        result = detect_material_leakage(rows, "stage", "qty")
        assert result[0].input_qty == 1000
        assert result[0].output_qty == 900
        assert result[0].loss == 100

    def test_zero_input_stage(self):
        rows = [
            {"stage": "A", "qty": 0},
            {"stage": "B", "qty": 100},
        ]
        result = detect_material_leakage(rows, "stage", "qty")
        assert result[0].loss_pct == 0.0

    def test_sequence_filters_unknown_stages(self):
        rows = [
            {"stage": "crushing", "qty": 1000},
            {"stage": "grinding", "qty": 950},
            {"stage": "unknown", "qty": 500},
        ]
        result = detect_material_leakage(
            rows, "stage", "qty",
            sequence=["crushing", "grinding"],
        )
        assert len(result) == 1
        assert result[0].from_stage == "crushing"
        assert result[0].to_stage == "grinding"


# ===================================================================
# compute_mix_analysis Tests
# ===================================================================


class TestComputeMixAnalysis:
    def test_basic_mix_no_recipe(self):
        rows = [
            {"component": "iron_ore", "qty": 600},
            {"component": "coal", "qty": 300},
            {"component": "flux", "qty": 100},
        ]
        result = compute_mix_analysis(rows, "component", "qty")
        assert result is not None
        assert result.total_quantity == 1000
        assert len(result.components) == 3
        iron = [c for c in result.components if c.component == "iron_ore"][0]
        assert iron.actual_ratio == 0.6
        assert iron.target_ratio is None
        assert iron.deviation_pct is None

    def test_empty_rows_returns_none(self):
        assert compute_mix_analysis([], "component", "qty") is None

    def test_all_none_returns_none(self):
        rows = [{"component": None, "qty": None}]
        assert compute_mix_analysis(rows, "component", "qty") is None

    def test_with_recipe_deviation(self):
        rows = [
            {"component": "iron_ore", "qty": 650},
            {"component": "coal", "qty": 250},
            {"component": "flux", "qty": 100},
        ]
        recipe = {"iron_ore": 0.6, "coal": 0.3, "flux": 0.1}
        result = compute_mix_analysis(rows, "component", "qty", recipe=recipe)
        assert result is not None
        assert result.deviation_from_recipe is not None
        iron = [c for c in result.components if c.component == "iron_ore"][0]
        # actual = 0.65, target = 0.6, deviation = (0.65-0.6)/0.6*100 = 8.33%
        assert iron.target_ratio == 0.6
        assert abs(iron.deviation_pct - 8.33) < 0.1

    def test_recipe_with_missing_component(self):
        rows = [
            {"component": "iron_ore", "qty": 700},
            {"component": "coal", "qty": 300},
        ]
        recipe = {"iron_ore": 0.6, "coal": 0.3, "flux": 0.1}
        result = compute_mix_analysis(rows, "component", "qty", recipe=recipe)
        flux = [c for c in result.components if c.component == "flux"][0]
        assert flux.quantity == 0.0
        assert flux.actual_ratio == 0.0
        assert flux.deviation_pct == -100.0

    def test_no_recipe_deviation_is_none(self):
        rows = [{"component": "iron_ore", "qty": 100}]
        result = compute_mix_analysis(rows, "component", "qty")
        assert result.deviation_from_recipe is None

    def test_aggregates_multiple_rows(self):
        rows = [
            {"component": "iron_ore", "qty": 300},
            {"component": "iron_ore", "qty": 300},
            {"component": "coal", "qty": 200},
            {"component": "coal", "qty": 100},
        ]
        result = compute_mix_analysis(rows, "component", "qty")
        assert result.total_quantity == 900
        iron = [c for c in result.components if c.component == "iron_ore"][0]
        assert iron.quantity == 600

    def test_zero_total_quantity_returns_none(self):
        rows = [{"component": "iron_ore", "qty": 0}]
        result = compute_mix_analysis(rows, "component", "qty")
        assert result is None

    def test_summary_text(self):
        rows = [
            {"component": "iron_ore", "qty": 600},
            {"component": "coal", "qty": 400},
        ]
        result = compute_mix_analysis(rows, "component", "qty")
        assert "Mix analysis" in result.summary
        assert "iron_ore" in result.summary or "coal" in result.summary

    def test_recipe_perfect_match(self):
        rows = [
            {"component": "iron_ore", "qty": 60},
            {"component": "coal", "qty": 30},
            {"component": "flux", "qty": 10},
        ]
        recipe = {"iron_ore": 0.6, "coal": 0.3, "flux": 0.1}
        result = compute_mix_analysis(rows, "component", "qty", recipe=recipe)
        assert result.deviation_from_recipe == 0.0
        for c in result.components:
            assert c.deviation_pct == 0.0


# ===================================================================
# format_balance_report Tests
# ===================================================================


class TestFormatBalanceReport:
    def _make_result(self) -> MaterialBalanceResult:
        """Helper to create a MaterialBalanceResult for testing."""
        rows = [
            {"plant": "P1", "input": 1000, "output": 950},
            {"plant": "P2", "input": 1000, "output": 800},
        ]
        result = compute_material_balance(rows, "plant", "input", "output")
        assert result is not None
        return result

    def test_report_has_header(self):
        result = self._make_result()
        report = format_balance_report(result)
        assert "Material Balance Report" in report
        assert "=" * 40 in report

    def test_report_has_overall_section(self):
        result = self._make_result()
        report = format_balance_report(result)
        assert "Overall" in report
        assert "Total Input" in report
        assert "Total Output" in report
        assert "Total Loss" in report
        assert "Recovery" in report

    def test_report_has_entity_breakdown(self):
        result = self._make_result()
        report = format_balance_report(result)
        assert "Entity Breakdown" in report
        assert "P1" in report
        assert "P2" in report

    def test_report_with_leakage(self):
        result = self._make_result()
        leakage = [
            LeakagePoint(
                from_stage="crushing",
                to_stage="grinding",
                input_qty=1000,
                output_qty=950,
                loss=50,
                loss_pct=5.0,
                severity="ok",
            ),
            LeakagePoint(
                from_stage="grinding",
                to_stage="flotation",
                input_qty=950,
                output_qty=800,
                loss=150,
                loss_pct=15.79,
                severity="critical",
            ),
        ]
        report = format_balance_report(result, leakage=leakage)
        assert "Leakage Analysis" in report
        assert "crushing" in report
        assert "grinding" in report
        assert "flotation" in report
        assert "critical" in report
        assert "Critical: 1" in report
        assert "Warning: 0" in report

    def test_report_without_leakage(self):
        result = self._make_result()
        report = format_balance_report(result)
        assert "Leakage Analysis" not in report

    def test_report_with_empty_leakage_list(self):
        result = self._make_result()
        report = format_balance_report(result, leakage=[])
        assert "Leakage Analysis" not in report
