"""Tests for inventory optimizer pure functions."""

import math

from business_brain.discovery.inventory_optimizer import (
    ItemHealth,
    ItemTurnover,
    TurnoverResult,
    EOQResult,
    ReorderPoint,
    HealthResult,
    analyze_inventory_health,
    compute_eoq,
    compute_inventory_turnover,
    compute_reorder_point,
    compute_safety_stock,
    format_inventory_report,
)


# ---------------------------------------------------------------------------
# compute_inventory_turnover
# ---------------------------------------------------------------------------

class TestComputeInventoryTurnover:
    def test_basic_turnover(self):
        rows = [
            {"item": "A", "cogs": 120000, "avg_inv": 10000},
            {"item": "B", "cogs": 50000, "avg_inv": 25000},
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert len(result.items) == 2
        # A: 120000/10000 = 12, B: 50000/25000 = 2
        a_item = next(it for it in result.items if it.item == "A")
        b_item = next(it for it in result.items if it.item == "B")
        assert abs(a_item.turnover_ratio - 12.0) < 0.01
        assert abs(b_item.turnover_ratio - 2.0) < 0.01

    def test_empty_rows_returns_none(self):
        result = compute_inventory_turnover([], "item", "cogs", "avg_inv")
        assert result is None

    def test_none_values_skipped(self):
        rows = [
            {"item": "A", "cogs": 100, "avg_inv": 10},
            {"item": None, "cogs": 200, "avg_inv": 20},
            {"item": "C", "cogs": None, "avg_inv": 30},
            {"item": "D", "cogs": 400, "avg_inv": None},
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert len(result.items) == 1
        assert result.items[0].item == "A"

    def test_all_invalid_returns_none(self):
        rows = [{"item": None, "cogs": 100, "avg_inv": 10}]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is None

    def test_category_classification(self):
        rows = [
            {"item": "Fast", "cogs": 130000, "avg_inv": 10000},   # 13x -> fast
            {"item": "Normal", "cogs": 50000, "avg_inv": 10000},  # 5x -> normal
            {"item": "Slow", "cogs": 10000, "avg_inv": 10000},    # 1x -> slow
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        fast = next(it for it in result.items if it.item == "Fast")
        normal = next(it for it in result.items if it.item == "Normal")
        slow = next(it for it in result.items if it.item == "Slow")
        assert fast.category == "fast"
        assert normal.category == "normal"
        assert slow.category == "slow"

    def test_best_and_worst_item(self):
        rows = [
            {"item": "Best", "cogs": 200000, "avg_inv": 10000},  # 20x
            {"item": "Worst", "cogs": 5000, "avg_inv": 10000},   # 0.5x
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert result.best_item == "Best"
        assert result.worst_item == "Worst"

    def test_slow_and_fast_movers(self):
        rows = [
            {"item": "F1", "cogs": 150000, "avg_inv": 10000},  # 15x
            {"item": "S1", "cogs": 5000, "avg_inv": 10000},    # 0.5x
            {"item": "S2", "cogs": 15000, "avg_inv": 10000},   # 1.5x
            {"item": "N1", "cogs": 60000, "avg_inv": 10000},   # 6x
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert "F1" in result.fast_movers
        assert "S1" in result.slow_movers
        assert "S2" in result.slow_movers
        assert "N1" not in result.slow_movers
        assert "N1" not in result.fast_movers

    def test_days_of_inventory(self):
        rows = [{"item": "X", "cogs": 36500, "avg_inv": 100}]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        # turnover = 365, days = 365/365 = 1
        assert abs(result.items[0].days_of_inventory - 1.0) < 0.01

    def test_single_item(self):
        rows = [{"item": "Only", "cogs": 10000, "avg_inv": 5000}]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert len(result.items) == 1
        assert result.best_item == "Only"
        assert result.worst_item == "Only"

    def test_zero_avg_inventory(self):
        rows = [{"item": "Z", "cogs": 10000, "avg_inv": 0}]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert result.items[0].turnover_ratio == 0.0
        assert result.items[0].days_of_inventory == 0.0

    def test_summary_contains_key_info(self):
        rows = [
            {"item": "A", "cogs": 100000, "avg_inv": 10000},
            {"item": "B", "cogs": 20000, "avg_inv": 10000},
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert "2 items" in result.summary
        assert "A" in result.summary
        assert "B" in result.summary

    def test_non_numeric_values_skipped(self):
        rows = [
            {"item": "A", "cogs": "not_a_number", "avg_inv": 10},
            {"item": "B", "cogs": 200, "avg_inv": 20},
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert len(result.items) == 1
        assert result.items[0].item == "B"

    def test_mean_turnover(self):
        rows = [
            {"item": "A", "cogs": 100000, "avg_inv": 10000},  # 10x
            {"item": "B", "cogs": 40000, "avg_inv": 10000},   # 4x
        ]
        result = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        assert result is not None
        assert abs(result.mean_turnover - 7.0) < 0.01


# ---------------------------------------------------------------------------
# compute_eoq
# ---------------------------------------------------------------------------

class TestComputeEOQ:
    def test_basic_eoq(self):
        # Classic example: D=10000, S=100, H=2
        # EOQ = sqrt(2*10000*100/2) = sqrt(1000000) = 1000
        result = compute_eoq(10000, 100, 2)
        assert result is not None
        assert abs(result.eoq - 1000.0) < 0.01

    def test_annual_orders(self):
        result = compute_eoq(10000, 100, 2)
        assert result is not None
        # 10000 / 1000 = 10 orders
        assert abs(result.annual_orders - 10.0) < 0.01

    def test_order_interval(self):
        result = compute_eoq(10000, 100, 2)
        assert result is not None
        # 365 / 10 = 36.5 days
        assert abs(result.order_interval_days - 36.5) < 0.1

    def test_total_costs_equal(self):
        # At EOQ, total ordering cost = total holding cost
        result = compute_eoq(10000, 100, 2)
        assert result is not None
        assert abs(result.total_ordering_cost - result.total_holding_cost) < 0.01

    def test_total_cost(self):
        result = compute_eoq(10000, 100, 2)
        assert result is not None
        assert abs(result.total_cost - (result.total_ordering_cost + result.total_holding_cost)) < 0.01

    def test_zero_demand_returns_none(self):
        assert compute_eoq(0, 100, 2) is None

    def test_negative_demand_returns_none(self):
        assert compute_eoq(-100, 100, 2) is None

    def test_zero_ordering_cost_returns_none(self):
        assert compute_eoq(10000, 0, 2) is None

    def test_zero_holding_cost_returns_none(self):
        assert compute_eoq(10000, 100, 0) is None

    def test_small_values(self):
        result = compute_eoq(100, 10, 1)
        assert result is not None
        expected = math.sqrt(2 * 100 * 10 / 1)
        assert abs(result.eoq - expected) < 0.01


# ---------------------------------------------------------------------------
# compute_reorder_point
# ---------------------------------------------------------------------------

class TestComputeReorderPoint:
    def test_basic_rop(self):
        result = compute_reorder_point(50, 7, 100)
        assert abs(result.rop - 450.0) < 0.01  # 50*7 + 100

    def test_zero_safety_stock(self):
        result = compute_reorder_point(50, 7)
        assert abs(result.rop - 350.0) < 0.01  # 50*7

    def test_zero_demand(self):
        result = compute_reorder_point(0, 7, 100)
        assert abs(result.rop - 100.0) < 0.01

    def test_zero_lead_time(self):
        result = compute_reorder_point(50, 0, 100)
        assert abs(result.rop - 100.0) < 0.01

    def test_stored_values(self):
        result = compute_reorder_point(25, 14, 50)
        assert result.daily_demand == 25
        assert result.lead_time_days == 14
        assert result.safety_stock == 50


# ---------------------------------------------------------------------------
# compute_safety_stock
# ---------------------------------------------------------------------------

class TestComputeSafetyStock:
    def test_95_service_level(self):
        # Z=1.645, sigma=10, LT=4 -> 1.645 * 10 * sqrt(4) = 1.645 * 10 * 2 = 32.9
        ss = compute_safety_stock(10, 4, 0.95)
        assert abs(ss - 32.9) < 0.01

    def test_99_service_level(self):
        # Z=2.326, sigma=10, LT=4 -> 2.326 * 10 * 2 = 46.52
        ss = compute_safety_stock(10, 4, 0.99)
        assert abs(ss - 46.52) < 0.01

    def test_90_service_level(self):
        # Z=1.282, sigma=10, LT=4 -> 1.282 * 10 * 2 = 25.64
        ss = compute_safety_stock(10, 4, 0.90)
        assert abs(ss - 25.64) < 0.01

    def test_zero_std_returns_zero(self):
        assert compute_safety_stock(0, 4, 0.95) == 0.0

    def test_negative_std_returns_zero(self):
        assert compute_safety_stock(-5, 4, 0.95) == 0.0

    def test_zero_lead_time_returns_zero(self):
        assert compute_safety_stock(10, 0, 0.95) == 0.0

    def test_unknown_service_level_defaults_to_95(self):
        # Should fall back to Z=1.645
        ss = compute_safety_stock(10, 4, 0.85)
        expected = 1.645 * 10 * math.sqrt(4)
        assert abs(ss - expected) < 0.01

    def test_lead_time_one_day(self):
        ss = compute_safety_stock(10, 1, 0.95)
        assert abs(ss - 16.45) < 0.01  # 1.645 * 10 * 1


# ---------------------------------------------------------------------------
# analyze_inventory_health
# ---------------------------------------------------------------------------

class TestAnalyzeInventoryHealth:
    def test_basic_health(self):
        rows = [
            {"item": "A", "qty": 100, "min": 20, "max": 200, "rop": 50},
            {"item": "B", "qty": 250, "min": 20, "max": 200, "rop": 50},  # overstocked
            {"item": "C", "qty": 10, "min": 20, "max": 200, "rop": 50},   # understocked
            {"item": "D", "qty": 40, "min": 20, "max": 200, "rop": 50},   # at reorder
        ]
        result = analyze_inventory_health(rows, "item", "qty", "min", "max", "rop")
        assert result is not None
        assert result.healthy_count == 1
        assert result.overstocked_count == 1
        assert result.understocked_count == 1
        assert result.at_reorder_count == 1

    def test_empty_rows_returns_none(self):
        assert analyze_inventory_health([], "item", "qty") is None

    def test_no_thresholds_all_healthy(self):
        rows = [
            {"item": "A", "qty": 100},
            {"item": "B", "qty": 200},
        ]
        result = analyze_inventory_health(rows, "item", "qty")
        assert result is not None
        assert result.healthy_count == 2
        assert result.overstocked_count == 0
        assert result.understocked_count == 0

    def test_only_min_column(self):
        rows = [
            {"item": "A", "qty": 5, "min": 10},   # understocked
            {"item": "B", "qty": 50, "min": 10},   # healthy
        ]
        result = analyze_inventory_health(rows, "item", "qty", min_column="min")
        assert result is not None
        assert result.understocked_count == 1
        assert result.healthy_count == 1

    def test_only_max_column(self):
        rows = [
            {"item": "A", "qty": 300, "max": 200},  # overstocked
            {"item": "B", "qty": 100, "max": 200},   # healthy
        ]
        result = analyze_inventory_health(rows, "item", "qty", max_column="max")
        assert result is not None
        assert result.overstocked_count == 1
        assert result.healthy_count == 1

    def test_reorder_point_equality(self):
        # qty == reorder_point should trigger "reorder"
        rows = [{"item": "A", "qty": 50, "rop": 50}]
        result = analyze_inventory_health(rows, "item", "qty", reorder_column="rop")
        assert result is not None
        assert result.at_reorder_count == 1

    def test_none_qty_skipped(self):
        rows = [
            {"item": "A", "qty": None},
            {"item": "B", "qty": 100},
        ]
        result = analyze_inventory_health(rows, "item", "qty")
        assert result is not None
        assert len(result.items) == 1

    def test_all_none_returns_none(self):
        rows = [
            {"item": "A", "qty": None},
            {"item": None, "qty": 100},
        ]
        result = analyze_inventory_health(rows, "item", "qty")
        # Only one valid row (item=None is skipped, qty=None is skipped)
        # Actually item=None is None -> skipped, item=A qty=None -> skipped
        assert result is None

    def test_summary_content(self):
        rows = [
            {"item": "A", "qty": 100, "min": 20, "max": 200},
        ]
        result = analyze_inventory_health(rows, "item", "qty", "min", "max")
        assert result is not None
        assert "1 items" in result.summary
        assert "healthy" in result.summary.lower() or "1 healthy" in result.summary

    def test_non_numeric_qty_skipped(self):
        rows = [
            {"item": "A", "qty": "bad"},
            {"item": "B", "qty": 100},
        ]
        result = analyze_inventory_health(rows, "item", "qty")
        assert result is not None
        assert len(result.items) == 1

    def test_overstocked_takes_priority_over_reorder(self):
        # If qty > max, it's overstocked even if reorder check might fire
        rows = [{"item": "A", "qty": 300, "max": 200, "rop": 500}]
        result = analyze_inventory_health(rows, "item", "qty", max_column="max", reorder_column="rop")
        assert result is not None
        assert result.items[0].status == "overstocked"

    def test_understocked_takes_priority_over_reorder(self):
        rows = [{"item": "A", "qty": 5, "min": 10, "rop": 50}]
        result = analyze_inventory_health(rows, "item", "qty", min_column="min", reorder_column="rop")
        assert result is not None
        assert result.items[0].status == "understocked"


# ---------------------------------------------------------------------------
# format_inventory_report
# ---------------------------------------------------------------------------

class TestFormatInventoryReport:
    def test_no_data(self):
        report = format_inventory_report()
        assert "No inventory data" in report

    def test_turnover_only(self):
        rows = [
            {"item": "A", "cogs": 100000, "avg_inv": 10000},
        ]
        turnover = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        report = format_inventory_report(turnover=turnover)
        assert "Turnover Analysis" in report
        assert "A" in report
        assert "Inventory Health" not in report

    def test_health_only(self):
        rows = [
            {"item": "A", "qty": 100, "min": 10, "max": 200},
        ]
        health = analyze_inventory_health(rows, "item", "qty", "min", "max")
        report = format_inventory_report(health=health)
        assert "Inventory Health" in report
        assert "A" in report
        assert "Turnover Analysis" not in report

    def test_combined_report(self):
        t_rows = [{"item": "A", "cogs": 100000, "avg_inv": 10000}]
        h_rows = [{"item": "A", "qty": 100, "min": 10, "max": 200}]
        turnover = compute_inventory_turnover(t_rows, "item", "cogs", "avg_inv")
        health = analyze_inventory_health(h_rows, "item", "qty", "min", "max")
        report = format_inventory_report(turnover=turnover, health=health)
        assert "Turnover Analysis" in report
        assert "Inventory Health" in report
        assert "Inventory Report" in report

    def test_report_includes_slow_movers(self):
        rows = [
            {"item": "SlowPoke", "cogs": 1000, "avg_inv": 10000},  # 0.1x -> slow
        ]
        turnover = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        report = format_inventory_report(turnover=turnover)
        assert "SlowPoke" in report
        assert "Slow Movers" in report

    def test_report_includes_fast_movers(self):
        rows = [
            {"item": "Speedy", "cogs": 200000, "avg_inv": 10000},  # 20x -> fast
        ]
        turnover = compute_inventory_turnover(rows, "item", "cogs", "avg_inv")
        report = format_inventory_report(turnover=turnover)
        assert "Speedy" in report
        assert "Fast Movers" in report
