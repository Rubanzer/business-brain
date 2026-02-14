"""Tests for logistics and dispatch tracking analytics module."""

from business_brain.discovery.logistics_tracker import (
    DeliveryResult,
    DispatchResult,
    EntityDelivery,
    EntityTransit,
    PeriodDispatch,
    TransitResult,
    VehicleUtil,
    VehicleUtilResult,
    analyze_delivery_performance,
    analyze_dispatch_frequency,
    compute_transit_time,
    compute_vehicle_utilization,
    format_logistics_report,
)


# ===================================================================
# 1. analyze_delivery_performance
# ===================================================================


class TestAnalyzeDeliveryPerformance:
    def test_basic_delivery(self):
        rows = [
            {"carrier": "A", "promised": 10, "actual": 10},  # on time
            {"carrier": "A", "promised": 10, "actual": 12},  # late by 2
            {"carrier": "B", "promised": 10, "actual": 8},   # early by 2
            {"carrier": "B", "promised": 10, "actual": 10},  # on time
        ]
        result = analyze_delivery_performance(rows, "carrier", "promised", "actual")
        assert result is not None
        assert result.total_deliveries == 4
        assert result.on_time_count == 3  # A(1 on-time) + B(2 on-time: early + exact)
        assert len(result.entities) == 2

    def test_on_time_rate(self):
        rows = [
            {"carrier": "A", "promised": 10, "actual": 10},
            {"carrier": "A", "promised": 10, "actual": 10},
            {"carrier": "A", "promised": 10, "actual": 15},  # late
            {"carrier": "A", "promised": 10, "actual": 8},   # early = on time
        ]
        result = analyze_delivery_performance(rows, "carrier", "promised", "actual")
        assert result is not None
        a = result.entities[0]
        # 3 on time (2 exact + 1 early), 1 late => 75%
        assert a.on_time_rate == 75.0
        assert a.late_count == 1
        assert a.early_count == 1

    def test_avg_delay(self):
        rows = [
            {"carrier": "A", "promised": 10, "actual": 12},  # +2
            {"carrier": "A", "promised": 10, "actual": 14},  # +4
        ]
        result = analyze_delivery_performance(rows, "carrier", "promised", "actual")
        assert result is not None
        assert result.avg_delay == 3.0

    def test_best_and_worst_entity(self):
        rows = [
            {"carrier": "Good", "promised": 10, "actual": 10},
            {"carrier": "Good", "promised": 10, "actual": 9},
            {"carrier": "Bad", "promised": 10, "actual": 20},
            {"carrier": "Bad", "promised": 10, "actual": 25},
        ]
        result = analyze_delivery_performance(rows, "carrier", "promised", "actual")
        assert result is not None
        assert result.best_entity == "Good"
        assert result.worst_entity == "Bad"

    def test_empty_rows_returns_none(self):
        assert analyze_delivery_performance([], "c", "p", "a") is None

    def test_all_none_values_returns_none(self):
        rows = [{"carrier": None, "promised": None, "actual": None}]
        result = analyze_delivery_performance(rows, "carrier", "promised", "actual")
        assert result is None

    def test_single_entity(self):
        rows = [{"route": "R1", "promised": 5, "actual": 5}]
        result = analyze_delivery_performance(rows, "route", "promised", "actual")
        assert result is not None
        assert len(result.entities) == 1
        assert result.entities[0].on_time_rate == 100.0
        assert result.best_entity == "R1"
        assert result.worst_entity == "R1"

    def test_summary_is_string(self):
        rows = [{"c": "X", "p": 1, "a": 2}]
        result = analyze_delivery_performance(rows, "c", "p", "a")
        assert isinstance(result.summary, str)
        assert "X" in result.summary


# ===================================================================
# 2. compute_vehicle_utilization
# ===================================================================


class TestComputeVehicleUtilization:
    def test_basic_utilization(self):
        rows = [
            {"truck": "T1", "capacity": 100, "load": 80},
            {"truck": "T2", "capacity": 100, "load": 50},
        ]
        result = compute_vehicle_utilization(rows, "truck", "capacity", "load")
        assert result is not None
        assert len(result.vehicles) == 2

        t1 = [v for v in result.vehicles if v.vehicle == "T1"][0]
        assert t1.utilization_pct == 80.0
        assert t1.status == "optimal"

        t2 = [v for v in result.vehicles if v.vehicle == "T2"][0]
        assert t2.utilization_pct == 50.0
        assert t2.status == "underloaded"

    def test_overloaded_vehicle(self):
        rows = [{"truck": "T1", "capacity": 100, "load": 120}]
        result = compute_vehicle_utilization(rows, "truck", "capacity", "load")
        assert result is not None
        v = result.vehicles[0]
        assert v.utilization_pct == 120.0
        assert v.status == "overloaded"
        assert result.overloaded_count == 1

    def test_underloaded_count(self):
        rows = [
            {"truck": "T1", "capacity": 100, "load": 30},
            {"truck": "T2", "capacity": 100, "load": 40},
            {"truck": "T3", "capacity": 100, "load": 80},
        ]
        result = compute_vehicle_utilization(rows, "truck", "capacity", "load")
        assert result is not None
        assert result.underloaded_count == 2

    def test_mean_utilization(self):
        rows = [
            {"truck": "T1", "capacity": 100, "load": 80},
            {"truck": "T2", "capacity": 100, "load": 60},
        ]
        result = compute_vehicle_utilization(rows, "truck", "capacity", "load")
        assert result is not None
        # T1: 80%, T2: 60% => mean = 70%
        assert result.mean_utilization == 70.0

    def test_multiple_trips_per_vehicle(self):
        rows = [
            {"truck": "T1", "capacity": 100, "load": 80},
            {"truck": "T1", "capacity": 100, "load": 60},
        ]
        result = compute_vehicle_utilization(rows, "truck", "capacity", "load")
        assert result is not None
        v = result.vehicles[0]
        assert v.total_trips == 2
        assert v.avg_load == 70.0  # (80+60)/2
        assert v.utilization_pct == 70.0

    def test_empty_rows_returns_none(self):
        assert compute_vehicle_utilization([], "v", "c", "l") is None

    def test_none_values_skipped(self):
        rows = [
            {"truck": "T1", "capacity": None, "load": 80},
            {"truck": "T1", "capacity": 100, "load": 80},
        ]
        result = compute_vehicle_utilization(rows, "truck", "capacity", "load")
        assert result is not None
        assert result.vehicles[0].total_trips == 1

    def test_zero_capacity(self):
        rows = [{"truck": "T1", "capacity": 0, "load": 50}]
        result = compute_vehicle_utilization(rows, "truck", "capacity", "load")
        assert result is not None
        assert result.vehicles[0].utilization_pct == 0.0


# ===================================================================
# 3. analyze_dispatch_frequency
# ===================================================================


class TestAnalyzeDispatchFrequency:
    def test_basic_dispatch(self):
        rows = [
            {"month": "Jan", "site": "A"},
            {"month": "Jan", "site": "B"},
            {"month": "Feb", "site": "A"},
        ]
        result = analyze_dispatch_frequency(rows, "month")
        assert result is not None
        assert result.total_dispatches == 3
        assert len(result.periods) == 2

    def test_peak_and_trough(self):
        rows = [
            {"week": "W1"}, {"week": "W1"}, {"week": "W1"},
            {"week": "W2"},
            {"week": "W3"}, {"week": "W3"},
        ]
        result = analyze_dispatch_frequency(rows, "week")
        assert result is not None
        assert result.peak_period == "W1"
        assert result.trough_period == "W2"

    def test_avg_per_period(self):
        rows = [
            {"day": "Mon"}, {"day": "Mon"},
            {"day": "Tue"}, {"day": "Tue"},
            {"day": "Wed"}, {"day": "Wed"},
        ]
        result = analyze_dispatch_frequency(rows, "day")
        assert result is not None
        assert result.avg_per_period == 2.0

    def test_pct_of_total(self):
        rows = [
            {"day": "Mon"},
            {"day": "Mon"},
            {"day": "Mon"},
            {"day": "Tue"},
        ]
        result = analyze_dispatch_frequency(rows, "day")
        assert result is not None
        mon = [p for p in result.periods if p.period == "Mon"][0]
        assert mon.pct_of_total == 75.0

    def test_entity_column_filter(self):
        rows = [
            {"month": "Jan", "site": "A"},
            {"month": "Jan", "site": None},  # filtered out
            {"month": "Feb", "site": "B"},
        ]
        result = analyze_dispatch_frequency(rows, "month", entity_column="site")
        assert result is not None
        assert result.total_dispatches == 2

    def test_empty_rows_returns_none(self):
        assert analyze_dispatch_frequency([], "month") is None

    def test_all_none_time_returns_none(self):
        rows = [{"month": None}, {"month": None}]
        result = analyze_dispatch_frequency(rows, "month")
        assert result is None


# ===================================================================
# 4. compute_transit_time
# ===================================================================


class TestComputeTransitTime:
    def test_basic_transit(self):
        rows = [
            {"route": "R1", "depart": 0, "arrive": 10},
            {"route": "R1", "depart": 5, "arrive": 18},
            {"route": "R2", "depart": 0, "arrive": 5},
        ]
        result = compute_transit_time(rows, "route", "depart", "arrive")
        assert result is not None
        assert len(result.entities) == 2

        r1 = [e for e in result.entities if e.entity == "R1"][0]
        # R1 trips: 10, 13 => avg 11.5
        assert r1.avg_transit_time == 11.5
        assert r1.min_transit == 10.0
        assert r1.max_transit == 13.0
        assert r1.trip_count == 2

    def test_fastest_and_slowest(self):
        rows = [
            {"route": "Fast", "depart": 0, "arrive": 2},
            {"route": "Slow", "depart": 0, "arrive": 20},
        ]
        result = compute_transit_time(rows, "route", "depart", "arrive")
        assert result is not None
        assert result.fastest_entity == "Fast"
        assert result.slowest_entity == "Slow"

    def test_consistency_single_trip(self):
        rows = [{"route": "R1", "depart": 0, "arrive": 10}]
        result = compute_transit_time(rows, "route", "depart", "arrive")
        assert result is not None
        assert result.entities[0].consistency == 0.0

    def test_consistency_multiple_trips(self):
        rows = [
            {"route": "R1", "depart": 0, "arrive": 10},
            {"route": "R1", "depart": 0, "arrive": 10},
            {"route": "R1", "depart": 0, "arrive": 10},
        ]
        result = compute_transit_time(rows, "route", "depart", "arrive")
        assert result is not None
        # All identical => std = 0 => consistency = 0
        assert result.entities[0].consistency == 0.0

    def test_empty_rows_returns_none(self):
        assert compute_transit_time([], "r", "d", "a") is None

    def test_none_values_returns_none(self):
        rows = [{"route": None, "depart": None, "arrive": None}]
        result = compute_transit_time(rows, "route", "depart", "arrive")
        assert result is None

    def test_avg_transit_overall(self):
        rows = [
            {"route": "R1", "depart": 0, "arrive": 10},  # 10
            {"route": "R2", "depart": 0, "arrive": 20},  # 20
        ]
        result = compute_transit_time(rows, "route", "depart", "arrive")
        assert result is not None
        assert result.avg_transit == 15.0


# ===================================================================
# 5. format_logistics_report
# ===================================================================


class TestFormatLogisticsReport:
    def test_no_data_provided(self):
        report = format_logistics_report()
        assert "No analysis data provided." in report
        assert "Logistics & Dispatch Report" in report

    def test_delivery_section(self):
        rows = [
            {"c": "A", "p": 10, "a": 10},
            {"c": "A", "p": 10, "a": 12},
        ]
        delivery = analyze_delivery_performance(rows, "c", "p", "a")
        report = format_logistics_report(delivery=delivery)
        assert "Delivery Performance" in report
        assert "on-time=" in report

    def test_vehicle_section(self):
        rows = [{"truck": "T1", "cap": 100, "ld": 80}]
        vehicle = compute_vehicle_utilization(rows, "truck", "cap", "ld")
        report = format_logistics_report(vehicle=vehicle)
        assert "Vehicle Utilization" in report
        assert "optimal" in report

    def test_dispatch_section(self):
        rows = [{"month": "Jan"}, {"month": "Feb"}]
        dispatch = analyze_dispatch_frequency(rows, "month")
        report = format_logistics_report(dispatch=dispatch)
        assert "Dispatch Frequency" in report
        assert "Jan" in report

    def test_transit_section(self):
        rows = [{"r": "R1", "d": 0, "a": 10}]
        transit = compute_transit_time(rows, "r", "d", "a")
        report = format_logistics_report(transit=transit)
        assert "Transit Time" in report
        assert "R1" in report

    def test_combined_report(self):
        del_rows = [{"c": "A", "p": 10, "a": 10}]
        veh_rows = [{"t": "T1", "cap": 100, "ld": 80}]
        disp_rows = [{"month": "Jan"}]
        transit_rows = [{"r": "R1", "d": 0, "a": 10}]

        delivery = analyze_delivery_performance(del_rows, "c", "p", "a")
        vehicle = compute_vehicle_utilization(veh_rows, "t", "cap", "ld")
        dispatch = analyze_dispatch_frequency(disp_rows, "month")
        transit = compute_transit_time(transit_rows, "r", "d", "a")

        report = format_logistics_report(
            delivery=delivery,
            vehicle=vehicle,
            dispatch=dispatch,
            transit=transit,
        )
        assert "Delivery Performance" in report
        assert "Vehicle Utilization" in report
        assert "Dispatch Frequency" in report
        assert "Transit Time" in report
        assert "No analysis data provided." not in report
