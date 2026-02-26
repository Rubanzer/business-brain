"""Domain benchmark checker — compares profiled column stats against industry benchmarks.

Pure-function pass that creates Insight records when metrics fall outside
industry-standard thresholds (good/average/poor). Designed to surface
actionable findings like "Average SEC is 623 kWh/ton — rated POOR".
"""

from __future__ import annotations

import logging
import re
import uuid

from business_brain.cognitive.domain_knowledge import get_benchmarks
from business_brain.db.discovery_models import Insight, TableProfile

logger = logging.getLogger(__name__)

# Map benchmark keys to column name patterns
_BENCHMARK_COLUMN_MAP: dict[str, list[str]] = {
    "sec_kwh_per_ton": ["sec", "specific_energy", "kwh_per_ton", "sec_kwh"],
    "yield_pct": ["yield_pct", "yield_percent", "yield"],
    "power_factor": ["power_factor", "pf"],
    "tap_to_tap_min": ["tap_to_tap", "tap_to_tap_min"],
    "rejection_rate": ["rejection_rate", "rejection_pct"],
    "electrode_consumption": ["electrode_consumption", "electrode_kg"],
    "furnace_temp": ["furnace_temp", "tapping_temp", "bath_temp"],
}


def _parse_threshold(val: str) -> float | None:
    """Extract a numeric threshold from benchmark strings like '<500', '>600', '88-92%'."""
    m = re.search(r"[\d.]+", val.replace(",", "").replace("₹", ""))
    return float(m.group()) if m else None


def _rate_value(mean: float, benchmark: dict) -> tuple[str, str]:
    """Rate a value as good/average/poor against a benchmark entry.

    Returns (rating, explanation).
    """
    good_str = benchmark["good"]
    poor_str = benchmark["poor"]
    unit = benchmark.get("unit", "")

    good_val = _parse_threshold(good_str)
    poor_val = _parse_threshold(poor_str)

    if good_val is None or poor_val is None:
        return "unknown", ""

    # Determine direction: "good" with < means lower is better, > means higher is better
    if "<" in good_str:
        # Lower is better (e.g., SEC, rejection_rate)
        if mean < good_val:
            return "GOOD", f"below {good_val}{unit} target"
        elif mean > poor_val:
            return "POOR", f"above {poor_val}{unit} threshold"
        else:
            return "AVERAGE", f"between {good_val}-{poor_val}{unit}"
    else:
        # Higher is better (e.g., yield, power_factor)
        if mean > good_val:
            return "GOOD", f"above {good_val}{unit} target"
        elif mean < poor_val:
            return "POOR", f"below {poor_val}{unit} threshold"
        else:
            return "AVERAGE", f"between {poor_val}-{good_val}{unit}"


def check_benchmarks(profiles: list[TableProfile]) -> list[Insight]:
    """Check profiled metrics against industry benchmarks. Pure function."""
    benchmarks = get_benchmarks()
    if not benchmarks:
        return []

    insights: list[Insight] = []

    for profile in profiles:
        cls = profile.column_classification
        if not cls or "columns" not in cls:
            continue

        cols = cls["columns"]

        for bench_key, bench_info in benchmarks.items():
            patterns = _BENCHMARK_COLUMN_MAP.get(bench_key, [])
            if not patterns:
                continue

            for col_name, col_info in cols.items():
                col_lower = col_name.lower()
                if not any(p == col_lower or col_lower.endswith("_" + p) or col_lower.startswith(p + "_") for p in patterns):
                    continue

                stats = col_info.get("stats")
                if not stats or "mean" not in stats:
                    continue

                mean = stats["mean"]
                rating, explanation = _rate_value(mean, bench_info)
                if rating in ("unknown", "GOOD"):
                    continue

                unit = bench_info.get("unit", "")
                context = bench_info.get("context", "")
                severity = "warning" if rating == "POOR" else "info"
                impact = 60 if rating == "POOR" else 35
                metric_name = bench_key.replace("_", " ").title()

                insights.append(Insight(
                    id=str(uuid.uuid4()),
                    insight_type="anomaly",
                    severity=severity,
                    impact_score=impact,
                    title=f"{metric_name} rated {rating} in {profile.table_name}",
                    description=(
                        f"Average {col_name} is {mean:.1f}{unit} — rated {rating} "
                        f"(benchmark: good {bench_info['good']}, avg {bench_info['average']}, "
                        f"poor {bench_info['poor']}). {context}"
                    ),
                    source_tables=[profile.table_name],
                    source_columns=[col_name],
                    evidence={
                        "mean": round(mean, 2),
                        "min": stats.get("min"),
                        "max": stats.get("max"),
                        "benchmark_key": bench_key,
                        "rating": rating,
                    },
                    suggested_actions=[
                        f"Investigate why {col_name} is in the {rating.lower()} range",
                        f"Review operational practices affecting {metric_name.lower()}",
                    ],
                ))
                break  # One match per benchmark key per profile

    logger.info("Benchmark checker: %d insights from %d profiles", len(insights), len(profiles))
    return insights
