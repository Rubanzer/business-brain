"""SCADA / industrial sensor data analytics for manufacturing plants.

Pure functions for sensor reading analysis, anomaly detection, process
stability (Cp/Cpk) computation, alarm frequency analysis, and combined
report formatting.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SensorStats:
    """Aggregated statistics for a single sensor."""

    sensor: str
    min_val: float
    max_val: float
    mean: float
    std: float
    reading_count: int
    stability_index: float
    unit: str | None = None


@dataclass
class SensorResult:
    """Complete sensor reading analysis result."""

    sensors: list[SensorStats]
    stable_count: int
    unstable_count: int
    total_readings: int
    summary: str


@dataclass
class SensorAnomaly:
    """A single sensor anomaly event."""

    sensor: str
    value: float
    expected_range: tuple[float, float]
    anomaly_type: str
    index: int


@dataclass
class ProcessStability:
    """Process capability metrics for a single sensor."""

    sensor: str
    mean: float
    std: float
    usl: float
    lsl: float
    cp: float
    cpk: float
    rating: str


@dataclass
class AlarmInfo:
    """Frequency info for a single alarm type."""

    alarm: str
    count: int
    pct_of_total: float


@dataclass
class EquipmentAlarms:
    """Alarm summary for a single equipment entity."""

    equipment: str
    alarm_count: int
    critical_count: int


@dataclass
class AlarmResult:
    """Complete alarm frequency analysis result."""

    total_alarms: int
    by_severity: dict[str, int]
    by_equipment: list[EquipmentAlarms]
    top_alarms: list[AlarmInfo]
    chattering_alarms: list[str]
    summary: str


# ---------------------------------------------------------------------------
# 1. analyze_sensor_readings
# ---------------------------------------------------------------------------


def analyze_sensor_readings(
    rows: list[dict],
    sensor_column: str,
    value_column: str,
    timestamp_column: str | None = None,
    unit_column: str | None = None,
) -> SensorResult | None:
    """Analyse sensor readings: per-sensor stats, stability, and health.

    Args:
        rows: Data rows as dicts.
        sensor_column: Column identifying the sensor.
        value_column: Numeric column with sensor values.
        timestamp_column: Optional column with timestamps for gap detection.
        unit_column: Optional column with measurement units.

    Returns:
        SensorResult or None if no valid data.
    """
    if not rows:
        return None

    # Accumulate values per sensor
    acc: dict[str, list[float]] = {}
    units: dict[str, set[str]] = {}
    timestamps: dict[str, list[str]] = {}

    for row in rows:
        sensor = row.get(sensor_column)
        val = _safe_float(row.get(value_column))
        if sensor is None or val is None:
            continue
        key = str(sensor)
        acc.setdefault(key, []).append(val)

        if unit_column is not None:
            u = row.get(unit_column)
            if u is not None:
                units.setdefault(key, set()).add(str(u))

        if timestamp_column is not None:
            ts = row.get(timestamp_column)
            if ts is not None:
                timestamps.setdefault(key, []).append(str(ts))

    if not acc:
        return None

    sensors: list[SensorStats] = []
    total_readings = 0

    for name in sorted(acc.keys()):
        values = acc[name]
        count = len(values)
        total_readings += count

        min_val = min(values)
        max_val = max(values)
        mean = statistics.mean(values)
        std = statistics.pstdev(values)  # population std for sensor data

        # Stability index: 1 - (std / mean), capped [0, 1]
        if mean > 0:
            raw_stability = 1.0 - (std / mean)
            stability_index = max(0.0, min(1.0, raw_stability))
        else:
            stability_index = 0.0

        unit: str | None = None
        if name in units:
            unit_set = units[name]
            unit = ", ".join(sorted(unit_set))

        sensors.append(
            SensorStats(
                sensor=name,
                min_val=round(min_val, 4),
                max_val=round(max_val, 4),
                mean=round(mean, 4),
                std=round(std, 4),
                reading_count=count,
                stability_index=round(stability_index, 4),
                unit=unit,
            )
        )

    stable_count = sum(1 for s in sensors if s.stability_index > 0.8)
    unstable_count = len(sensors) - stable_count

    summary = (
        f"Sensor analysis across {len(sensors)} sensors, "
        f"{total_readings} total readings. "
        f"{stable_count} stable (stability > 0.8), "
        f"{unstable_count} unstable."
    )

    return SensorResult(
        sensors=sensors,
        stable_count=stable_count,
        unstable_count=unstable_count,
        total_readings=total_readings,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. detect_sensor_anomalies
# ---------------------------------------------------------------------------


def detect_sensor_anomalies(
    rows: list[dict],
    sensor_column: str,
    value_column: str,
    low_limit: float | None = None,
    high_limit: float | None = None,
) -> list[SensorAnomaly]:
    """Detect anomalies in sensor readings.

    If limits are provided, readings outside [low_limit, high_limit] are
    flagged.  Otherwise, auto-compute limits as mean +/- 3*std per sensor.

    Anomaly types:
    - "spike": value > high_limit
    - "drop": value < low_limit
    - "persistent_high": 3+ consecutive above high_limit
    - "persistent_low": 3+ consecutive below low_limit
    - "flatline": 10+ consecutive identical values

    Args:
        rows: Data rows as dicts.
        sensor_column: Column identifying the sensor.
        value_column: Numeric column with sensor values.
        low_limit: Optional global lower limit.
        high_limit: Optional global upper limit.

    Returns:
        List of SensorAnomaly.
    """
    if not rows:
        return []

    # Collect per-sensor ordered values with original row indices
    sensor_data: dict[str, list[tuple[int, float]]] = {}
    for i, row in enumerate(rows):
        sensor = row.get(sensor_column)
        val = _safe_float(row.get(value_column))
        if sensor is None or val is None:
            continue
        sensor_data.setdefault(str(sensor), []).append((i, val))

    if not sensor_data:
        return []

    anomalies: list[SensorAnomaly] = []

    for sensor_name, indexed_values in sorted(sensor_data.items()):
        values = [v for _, v in indexed_values]
        indices = [i for i, _ in indexed_values]

        # Determine limits
        if low_limit is not None and high_limit is not None:
            lo = low_limit
            hi = high_limit
        else:
            if len(values) < 2:
                continue
            mean = statistics.mean(values)
            std = statistics.pstdev(values)
            lo = mean - 3 * std
            hi = mean + 3 * std

        expected_range = (round(lo, 4), round(hi, 4))

        # --- Pass 1: detect flatlines (10+ consecutive identical values) ---
        flatline_indices: set[int] = set()
        run_start = 0
        for j in range(1, len(values)):
            if values[j] != values[run_start]:
                if j - run_start >= 10:
                    for k in range(run_start, j):
                        flatline_indices.add(k)
                run_start = j
        # Check final run
        if len(values) - run_start >= 10:
            for k in range(run_start, len(values)):
                flatline_indices.add(k)

        for local_idx in sorted(flatline_indices):
            anomalies.append(
                SensorAnomaly(
                    sensor=sensor_name,
                    value=values[local_idx],
                    expected_range=expected_range,
                    anomaly_type="flatline",
                    index=indices[local_idx],
                )
            )

        # --- Pass 2: detect out-of-range anomalies ---
        out_high: list[int] = []  # local indices that are above hi
        out_low: list[int] = []   # local indices that are below lo

        for j, val in enumerate(values):
            if j in flatline_indices:
                continue
            if val > hi:
                out_high.append(j)
            elif val < lo:
                out_low.append(j)

        # Detect persistent sequences (3+ consecutive)
        persistent_high_set: set[int] = set()
        persistent_low_set: set[int] = set()

        # Check consecutive high
        _mark_persistent(out_high, persistent_high_set)
        # Check consecutive low
        _mark_persistent(out_low, persistent_low_set)

        # Emit anomalies for high
        for j in out_high:
            if j in persistent_high_set:
                atype = "persistent_high"
            else:
                atype = "spike"
            anomalies.append(
                SensorAnomaly(
                    sensor=sensor_name,
                    value=values[j],
                    expected_range=expected_range,
                    anomaly_type=atype,
                    index=indices[j],
                )
            )

        # Emit anomalies for low
        for j in out_low:
            if j in persistent_low_set:
                atype = "persistent_low"
            else:
                atype = "drop"
            anomalies.append(
                SensorAnomaly(
                    sensor=sensor_name,
                    value=values[j],
                    expected_range=expected_range,
                    anomaly_type=atype,
                    index=indices[j],
                )
            )

    return anomalies


def _mark_persistent(sorted_indices: list[int], target: set[int]) -> None:
    """Mark indices that appear in runs of 3+ consecutive values."""
    if len(sorted_indices) < 3:
        return
    run: list[int] = [sorted_indices[0]]
    for j in range(1, len(sorted_indices)):
        if sorted_indices[j] == run[-1] + 1:
            run.append(sorted_indices[j])
        else:
            if len(run) >= 3:
                target.update(run)
            run = [sorted_indices[j]]
    if len(run) >= 3:
        target.update(run)


# ---------------------------------------------------------------------------
# 3. compute_process_stability
# ---------------------------------------------------------------------------


def compute_process_stability(
    rows: list[dict],
    sensor_column: str,
    value_column: str,
    target_column: str | None = None,
) -> list[ProcessStability]:
    """Compute process capability (Cp, Cpk) per sensor.

    USL/LSL are auto-derived as mean +/- 3*std unless *target_column*
    provides them (expected format: row with ``usl`` and ``lsl`` keys
    alongside sensor name).

    Rating: Cpk >= 1.33 => "Capable", >= 1.0 => "Marginal", else "Incapable".

    Results are sorted by Cpk ascending (worst first).

    Args:
        rows: Data rows as dicts.
        sensor_column: Column identifying the sensor.
        value_column: Numeric column with measurements.
        target_column: Optional column providing spec-limit info.

    Returns:
        List of ProcessStability, sorted by Cpk ascending.
    """
    if not rows:
        return []

    # Collect values per sensor; also collect target specs if provided
    acc: dict[str, list[float]] = {}
    specs: dict[str, dict[str, float]] = {}

    for row in rows:
        sensor = row.get(sensor_column)
        val = _safe_float(row.get(value_column))
        if sensor is None or val is None:
            continue
        key = str(sensor)
        acc.setdefault(key, []).append(val)

        if target_column is not None:
            target_val = row.get(target_column)
            if target_val is not None:
                # Try to parse USL and LSL from the row
                usl_val = _safe_float(row.get("usl"))
                lsl_val = _safe_float(row.get("lsl"))
                if usl_val is not None and lsl_val is not None:
                    specs[key] = {"usl": usl_val, "lsl": lsl_val}

    if not acc:
        return []

    results: list[ProcessStability] = []

    for name in sorted(acc.keys()):
        values = acc[name]
        if len(values) < 2:
            continue

        mean = statistics.mean(values)
        std = statistics.pstdev(values)

        if name in specs:
            usl = specs[name]["usl"]
            lsl = specs[name]["lsl"]
        else:
            usl = mean + 3 * std
            lsl = mean - 3 * std

        if std > 0:
            cp = (usl - lsl) / (6 * std)
            cpk = min((usl - mean), (mean - lsl)) / (3 * std)
        else:
            # Zero variation — all values identical
            if lsl <= mean <= usl:
                cp = 999.99
                cpk = 999.99
            else:
                cp = 0.0
                cpk = 0.0

        if cpk >= 1.33:
            rating = "Capable"
        elif cpk >= 1.0:
            rating = "Marginal"
        else:
            rating = "Incapable"

        results.append(
            ProcessStability(
                sensor=name,
                mean=round(mean, 4),
                std=round(std, 4),
                usl=round(usl, 4),
                lsl=round(lsl, 4),
                cp=round(cp, 4),
                cpk=round(cpk, 4),
                rating=rating,
            )
        )

    # Sort by Cpk ascending (worst first)
    results.sort(key=lambda r: r.cpk)
    return results


# ---------------------------------------------------------------------------
# 4. analyze_alarm_frequency
# ---------------------------------------------------------------------------


def analyze_alarm_frequency(
    rows: list[dict],
    alarm_column: str,
    severity_column: str | None = None,
    timestamp_column: str | None = None,
    equipment_column: str | None = None,
) -> AlarmResult | None:
    """Analyse alarm frequency, severity distribution, and chattering.

    Args:
        rows: Data rows as dicts.
        alarm_column: Column identifying the alarm type/code.
        severity_column: Optional column with severity (critical/warning/info).
        timestamp_column: Optional column with timestamps.
        equipment_column: Optional column identifying equipment.

    Returns:
        AlarmResult or None if no valid data.
    """
    if not rows:
        return None

    alarm_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    equipment_acc: dict[str, dict[str, int]] = {}  # equipment -> {alarm_count, critical_count}
    alarm_sequence: list[str] = []  # ordered alarm names for chattering

    total = 0

    for row in rows:
        alarm = row.get(alarm_column)
        if alarm is None:
            continue
        alarm_str = str(alarm)
        total += 1
        alarm_counts[alarm_str] = alarm_counts.get(alarm_str, 0) + 1
        alarm_sequence.append(alarm_str)

        if severity_column is not None:
            sev = row.get(severity_column)
            if sev is not None:
                sev_str = str(sev).lower()
                severity_counts[sev_str] = severity_counts.get(sev_str, 0) + 1

        if equipment_column is not None:
            equip = row.get(equipment_column)
            if equip is not None:
                equip_str = str(equip)
                if equip_str not in equipment_acc:
                    equipment_acc[equip_str] = {"alarm_count": 0, "critical_count": 0}
                equipment_acc[equip_str]["alarm_count"] += 1

                if severity_column is not None:
                    sev = row.get(severity_column)
                    if sev is not None and str(sev).lower() == "critical":
                        equipment_acc[equip_str]["critical_count"] += 1

    if total == 0:
        return None

    # Top 5 most frequent alarms
    sorted_alarms = sorted(alarm_counts.items(), key=lambda x: -x[1])
    top_alarms: list[AlarmInfo] = []
    for alarm_name, count in sorted_alarms[:5]:
        pct = count / total * 100
        top_alarms.append(
            AlarmInfo(alarm=alarm_name, count=count, pct_of_total=round(pct, 2))
        )

    # Equipment summary
    by_equipment: list[EquipmentAlarms] = []
    for equip_name in sorted(equipment_acc.keys()):
        d = equipment_acc[equip_name]
        by_equipment.append(
            EquipmentAlarms(
                equipment=equip_name,
                alarm_count=d["alarm_count"],
                critical_count=d["critical_count"],
            )
        )

    # Chattering detection: same alarm > 5 times in succession
    chattering_alarms: list[str] = []
    chattering_set: set[str] = set()
    if alarm_sequence:
        run_alarm = alarm_sequence[0]
        run_len = 1
        for j in range(1, len(alarm_sequence)):
            if alarm_sequence[j] == run_alarm:
                run_len += 1
            else:
                if run_len > 5 and run_alarm not in chattering_set:
                    chattering_alarms.append(run_alarm)
                    chattering_set.add(run_alarm)
                run_alarm = alarm_sequence[j]
                run_len = 1
        # Check final run
        if run_len > 5 and run_alarm not in chattering_set:
            chattering_alarms.append(run_alarm)
            chattering_set.add(run_alarm)

    # Build summary
    parts = [f"Alarm analysis: {total} total alarms"]
    if severity_counts:
        sev_parts = ", ".join(f"{k}: {v}" for k, v in sorted(severity_counts.items()))
        parts.append(f"By severity: {sev_parts}")
    if top_alarms:
        top_str = ", ".join(f"{a.alarm} ({a.count})" for a in top_alarms[:3])
        parts.append(f"Top alarms: {top_str}")
    if chattering_alarms:
        parts.append(f"Chattering detected: {', '.join(chattering_alarms)}")
    summary = ". ".join(parts) + "."

    return AlarmResult(
        total_alarms=total,
        by_severity=dict(sorted(severity_counts.items())),
        by_equipment=by_equipment,
        top_alarms=top_alarms,
        chattering_alarms=chattering_alarms,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. format_scada_report
# ---------------------------------------------------------------------------


def format_scada_report(
    sensor_result: SensorResult | None = None,
    anomalies: list[SensorAnomaly] | None = None,
    stability: list[ProcessStability] | None = None,
    alarms: AlarmResult | None = None,
) -> str:
    """Generate a combined SCADA monitoring report.

    Args:
        sensor_result: Sensor reading analysis result.
        anomalies: List of detected anomalies.
        stability: Process stability results.
        alarms: Alarm frequency analysis result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("=" * 60)
    sections.append("SCADA / INDUSTRIAL SENSOR REPORT")
    sections.append("=" * 60)

    if sensor_result is not None:
        lines = ["", "SENSOR READINGS", "-" * 40]
        lines.append(f"  Total Readings:  {sensor_result.total_readings}")
        lines.append(f"  Sensors:         {len(sensor_result.sensors)}")
        lines.append(f"  Stable:          {sensor_result.stable_count}")
        lines.append(f"  Unstable:        {sensor_result.unstable_count}")
        if sensor_result.sensors:
            lines.append("")
            lines.append(
                f"  {'Sensor':<20} {'Min':>10} {'Max':>10} "
                f"{'Mean':>10} {'Std':>10} {'Stab':>6} {'N':>5}"
            )
            lines.append(
                f"  {'-'*20} {'-'*10} {'-'*10} "
                f"{'-'*10} {'-'*10} {'-'*6} {'-'*5}"
            )
            for s in sensor_result.sensors:
                lines.append(
                    f"  {s.sensor:<20} {s.min_val:>10.2f} {s.max_val:>10.2f} "
                    f"{s.mean:>10.2f} {s.std:>10.2f} {s.stability_index:>6.3f} "
                    f"{s.reading_count:>5}"
                )
        sections.append("\n".join(lines))

    if anomalies is not None:
        lines = ["", "ANOMALIES", "-" * 40]
        lines.append(f"  Total Anomalies: {len(anomalies)}")
        if anomalies:
            lines.append("")
            lines.append(
                f"  {'Sensor':<20} {'Value':>10} {'Type':<18} {'Index':>5}"
            )
            lines.append(
                f"  {'-'*20} {'-'*10} {'-'*18} {'-'*5}"
            )
            for a in anomalies:
                lines.append(
                    f"  {a.sensor:<20} {a.value:>10.2f} {a.anomaly_type:<18} "
                    f"{a.index:>5}"
                )
        sections.append("\n".join(lines))

    if stability is not None:
        lines = ["", "PROCESS STABILITY", "-" * 40]
        if stability:
            lines.append(
                f"  {'Sensor':<20} {'Cp':>8} {'Cpk':>8} {'Rating':<12}"
            )
            lines.append(
                f"  {'-'*20} {'-'*8} {'-'*8} {'-'*12}"
            )
            for ps in stability:
                lines.append(
                    f"  {ps.sensor:<20} {ps.cp:>8.4f} {ps.cpk:>8.4f} "
                    f"{ps.rating:<12}"
                )
        else:
            lines.append("  No stability data.")
        sections.append("\n".join(lines))

    if alarms is not None:
        lines = ["", "ALARM ANALYSIS", "-" * 40]
        lines.append(f"  Total Alarms:    {alarms.total_alarms}")
        if alarms.by_severity:
            sev = ", ".join(f"{k}: {v}" for k, v in alarms.by_severity.items())
            lines.append(f"  By Severity:     {sev}")
        if alarms.top_alarms:
            lines.append("  Top Alarms:")
            for a in alarms.top_alarms:
                lines.append(
                    f"    {a.alarm}: {a.count} ({a.pct_of_total:.1f}%)"
                )
        if alarms.chattering_alarms:
            lines.append(
                f"  Chattering:      {', '.join(alarms.chattering_alarms)}"
            )
        sections.append("\n".join(lines))

    if (
        sensor_result is None
        and anomalies is None
        and stability is None
        and alarms is None
    ):
        sections.append("")
        sections.append("No analysis data provided.")

    sections.append("")
    sections.append("=" * 60)

    return "\n".join(sections)
