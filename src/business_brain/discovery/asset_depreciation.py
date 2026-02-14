"""Asset lifecycle and depreciation analysis.

Pure functions for computing depreciation schedules, analyzing asset age
distributions, calculating book values, and evaluating maintenance cost
ratios for capital asset management.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


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


def _parse_date(val) -> datetime | None:
    """Try to parse a value into a datetime."""
    if val is None:
        return None
    if isinstance(val, datetime):
        return val
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ):
        try:
            return datetime.strptime(str(val), fmt)
        except (ValueError, TypeError):
            continue
    return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class YearDepreciation:
    """Depreciation detail for a single year."""

    year: int
    depreciation_amount: float
    book_value: float


@dataclass
class AssetSchedule:
    """Depreciation schedule for a single asset."""

    asset: str
    cost: float
    salvage: float
    useful_life: int
    annual_depreciation: float
    schedule: list[YearDepreciation]


@dataclass
class DepreciationScheduleResult:
    """Result of depreciation schedule computation."""

    assets: list[AssetSchedule]
    total_cost: float
    total_annual_depreciation: float
    method: str
    summary: str


@dataclass
class LifecycleStage:
    """Count and percentage for a lifecycle stage."""

    stage: str
    count: int
    pct: float


@dataclass
class CategoryAge:
    """Average age breakdown by category."""

    category: str
    count: int
    avg_age: float


@dataclass
class AssetAgeResult:
    """Result of asset age analysis."""

    total_assets: int
    avg_age: float
    by_lifecycle_stage: list[LifecycleStage]
    by_category: list[CategoryAge] | None
    weighted_avg_age: float | None
    summary: str


@dataclass
class AssetBookValue:
    """Book value detail for a single asset."""

    asset: str
    original_cost: float
    salvage_value: float
    age_years: float
    book_value: float
    depreciation_pct: float


@dataclass
class BookValueResult:
    """Result of book value computation."""

    assets: list[AssetBookValue]
    total_original_cost: float
    total_book_value: float
    depreciation_pct: float
    fully_depreciated_count: int
    summary: str


@dataclass
class AssetMaintRatio:
    """Maintenance cost ratio for a single asset."""

    asset: str
    maintenance_cost: float
    asset_value: float
    ratio_pct: float
    is_replacement_candidate: bool


@dataclass
class MaintenanceCostResult:
    """Result of maintenance cost ratio analysis."""

    assets: list[AssetMaintRatio]
    avg_ratio: float
    replacement_candidates: int
    total_maintenance: float
    total_asset_value: float
    summary: str


# ---------------------------------------------------------------------------
# 1. Depreciation Schedule
# ---------------------------------------------------------------------------


def compute_depreciation_schedule(
    rows: list[dict],
    asset_column: str,
    cost_column: str,
    useful_life_column: str,
    method: str = "straight_line",
    salvage_column: str | None = None,
) -> DepreciationScheduleResult | None:
    """Compute annual depreciation schedule for each asset.

    Supports three methods:
    - straight_line: (cost - salvage) / useful_life per year
    - declining_balance: double-declining balance at rate 2/useful_life
    - sum_of_years_digits: (cost - salvage) * (remaining / sum_of_years)

    Args:
        rows: Data rows as dicts.
        asset_column: Column identifying the asset.
        cost_column: Column with original cost.
        useful_life_column: Column with useful life in years.
        method: Depreciation method.
        salvage_column: Optional column with salvage value.

    Returns:
        DepreciationScheduleResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect unique assets (first occurrence wins for cost/life/salvage)
    asset_data: dict[str, dict] = {}
    for row in rows:
        name = row.get(asset_column)
        cost = _safe_float(row.get(cost_column))
        life = _safe_float(row.get(useful_life_column))
        if name is None or cost is None or life is None:
            continue
        life_int = int(life)
        if life_int <= 0:
            continue
        salvage = 0.0
        if salvage_column is not None:
            s = _safe_float(row.get(salvage_column))
            if s is not None:
                salvage = s
        key = str(name)
        if key not in asset_data:
            asset_data[key] = {"cost": cost, "life": life_int, "salvage": salvage}

    if not asset_data:
        return None

    assets: list[AssetSchedule] = []
    total_cost = 0.0
    total_annual_dep = 0.0

    for asset_name in sorted(asset_data.keys()):
        info = asset_data[asset_name]
        cost = info["cost"]
        life = info["life"]
        salvage = info["salvage"]
        total_cost += cost

        schedule: list[YearDepreciation] = []
        book = cost
        first_year_dep = 0.0

        if method == "straight_line":
            annual = (cost - salvage) / life
            first_year_dep = annual
            for yr in range(1, life + 1):
                dep = annual
                book = max(cost - annual * yr, salvage)
                schedule.append(YearDepreciation(
                    year=yr,
                    depreciation_amount=round(dep, 4),
                    book_value=round(book, 4),
                ))

        elif method == "declining_balance":
            rate = 2.0 / life
            first_year_dep = cost * rate
            for yr in range(1, life + 1):
                dep = book * rate
                # Cannot depreciate below salvage
                if book - dep < salvage:
                    dep = book - salvage
                book = book - dep
                schedule.append(YearDepreciation(
                    year=yr,
                    depreciation_amount=round(dep, 4),
                    book_value=round(book, 4),
                ))

        elif method == "sum_of_years_digits":
            soy = life * (life + 1) / 2
            first_year_dep = (cost - salvage) * (life / soy)
            for yr in range(1, life + 1):
                remaining = life - yr + 1
                dep = (cost - salvage) * (remaining / soy)
                book = book - dep
                book = max(book, salvage)
                schedule.append(YearDepreciation(
                    year=yr,
                    depreciation_amount=round(dep, 4),
                    book_value=round(book, 4),
                ))
        else:
            # Unknown method, fall back to straight-line
            annual = (cost - salvage) / life
            first_year_dep = annual
            for yr in range(1, life + 1):
                dep = annual
                book = max(cost - annual * yr, salvage)
                schedule.append(YearDepreciation(
                    year=yr,
                    depreciation_amount=round(dep, 4),
                    book_value=round(book, 4),
                ))

        total_annual_dep += first_year_dep

        assets.append(AssetSchedule(
            asset=asset_name,
            cost=round(cost, 4),
            salvage=round(salvage, 4),
            useful_life=life,
            annual_depreciation=round(first_year_dep, 4),
            schedule=schedule,
        ))

    summary = (
        f"Depreciation schedule for {len(assets)} assets using {method}: "
        f"Total cost = {total_cost:,.2f}, "
        f"Total first-year depreciation = {total_annual_dep:,.2f}."
    )

    return DepreciationScheduleResult(
        assets=assets,
        total_cost=round(total_cost, 4),
        total_annual_depreciation=round(total_annual_dep, 4),
        method=method,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 2. Asset Age Analysis
# ---------------------------------------------------------------------------


def analyze_asset_age(
    rows: list[dict],
    asset_column: str,
    purchase_date_column: str,
    category_column: str | None = None,
    cost_column: str | None = None,
) -> AssetAgeResult | None:
    """Analyze asset age distribution.

    Computes age relative to the maximum date found in the data.

    Lifecycle stages:
    - New: 0-2 years
    - Mid-life: 2-5 years
    - Aging: 5-10 years
    - End-of-life: >10 years

    Args:
        rows: Data rows as dicts.
        asset_column: Column identifying the asset.
        purchase_date_column: Column with purchase date.
        category_column: Optional column for category grouping.
        cost_column: Optional column for cost-weighted average age.

    Returns:
        AssetAgeResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect asset data (first occurrence per asset)
    asset_info: list[dict] = []
    all_dates: list[datetime] = []

    seen: set[str] = set()
    for row in rows:
        name = row.get(asset_column)
        dt = _parse_date(row.get(purchase_date_column))
        if name is None or dt is None:
            continue
        key = str(name)
        if key in seen:
            continue
        seen.add(key)
        all_dates.append(dt)
        entry: dict = {"name": key, "date": dt}
        if category_column is not None:
            cat = row.get(category_column)
            entry["category"] = str(cat) if cat is not None else None
        if cost_column is not None:
            c = _safe_float(row.get(cost_column))
            entry["cost"] = c
        asset_info.append(entry)

    if not asset_info:
        return None

    # Reference date: max date in data
    ref_date = max(all_dates)

    # Compute ages
    ages: list[float] = []
    for item in asset_info:
        delta = ref_date - item["date"]
        age_years = delta.days / 365.25
        item["age"] = age_years
        ages.append(age_years)

    total_assets = len(asset_info)
    avg_age = sum(ages) / total_assets

    # Lifecycle stages
    stage_counts: dict[str, int] = {"New": 0, "Mid-life": 0, "Aging": 0, "End-of-life": 0}
    for age in ages:
        if age <= 2:
            stage_counts["New"] += 1
        elif age <= 5:
            stage_counts["Mid-life"] += 1
        elif age <= 10:
            stage_counts["Aging"] += 1
        else:
            stage_counts["End-of-life"] += 1

    by_lifecycle: list[LifecycleStage] = []
    for stage_name in ("New", "Mid-life", "Aging", "End-of-life"):
        cnt = stage_counts[stage_name]
        pct = cnt / total_assets * 100 if total_assets > 0 else 0.0
        by_lifecycle.append(LifecycleStage(
            stage=stage_name,
            count=cnt,
            pct=round(pct, 2),
        ))

    # By category
    by_category: list[CategoryAge] | None = None
    if category_column is not None:
        cat_ages: dict[str, list[float]] = {}
        for item in asset_info:
            cat = item.get("category")
            if cat is not None:
                cat_ages.setdefault(cat, []).append(item["age"])
        by_category = []
        for cat_name in sorted(cat_ages.keys()):
            cat_list = cat_ages[cat_name]
            by_category.append(CategoryAge(
                category=cat_name,
                count=len(cat_list),
                avg_age=round(sum(cat_list) / len(cat_list), 2),
            ))

    # Weighted average age
    weighted_avg: float | None = None
    if cost_column is not None:
        total_weight = 0.0
        weighted_sum = 0.0
        for item in asset_info:
            c = item.get("cost")
            if c is not None and c > 0:
                total_weight += c
                weighted_sum += c * item["age"]
        if total_weight > 0:
            weighted_avg = round(weighted_sum / total_weight, 2)

    summary = (
        f"Asset age analysis for {total_assets} assets: "
        f"Average age = {avg_age:.1f} years. "
        f"New: {stage_counts['New']}, Mid-life: {stage_counts['Mid-life']}, "
        f"Aging: {stage_counts['Aging']}, End-of-life: {stage_counts['End-of-life']}."
    )

    return AssetAgeResult(
        total_assets=total_assets,
        avg_age=round(avg_age, 2),
        by_lifecycle_stage=by_lifecycle,
        by_category=by_category,
        weighted_avg_age=weighted_avg,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 3. Book Values
# ---------------------------------------------------------------------------


def compute_book_values(
    rows: list[dict],
    asset_column: str,
    cost_column: str,
    purchase_date_column: str,
    useful_life_column: str,
    salvage_column: str | None = None,
) -> BookValueResult | None:
    """Compute current book value for each asset using straight-line depreciation.

    Age is computed relative to the maximum date in the data.

    book_value = cost - ((cost - salvage) / useful_life * years_elapsed)
    Minimum book value is salvage (or 0 if no salvage).

    Args:
        rows: Data rows as dicts.
        asset_column: Column identifying the asset.
        cost_column: Column with original cost.
        purchase_date_column: Column with purchase date.
        useful_life_column: Column with useful life in years.
        salvage_column: Optional column with salvage value.

    Returns:
        BookValueResult or None if no valid data.
    """
    if not rows:
        return None

    # Collect all valid dates first to determine reference date
    all_dates: list[datetime] = []
    for row in rows:
        dt = _parse_date(row.get(purchase_date_column))
        if dt is not None:
            all_dates.append(dt)

    if not all_dates:
        return None

    ref_date = max(all_dates)

    # Collect asset data (first occurrence per asset)
    seen: set[str] = set()
    asset_list: list[dict] = []
    for row in rows:
        name = row.get(asset_column)
        cost = _safe_float(row.get(cost_column))
        dt = _parse_date(row.get(purchase_date_column))
        life = _safe_float(row.get(useful_life_column))
        if name is None or cost is None or dt is None or life is None:
            continue
        life_val = life
        if life_val <= 0:
            continue
        key = str(name)
        if key in seen:
            continue
        seen.add(key)
        salvage = 0.0
        if salvage_column is not None:
            s = _safe_float(row.get(salvage_column))
            if s is not None:
                salvage = s
        age_years = (ref_date - dt).days / 365.25
        asset_list.append({
            "name": key,
            "cost": cost,
            "salvage": salvage,
            "life": life_val,
            "age": age_years,
        })

    if not asset_list:
        return None

    assets: list[AssetBookValue] = []
    total_original = 0.0
    total_book = 0.0
    fully_dep_count = 0

    for item in asset_list:
        cost = item["cost"]
        salvage = item["salvage"]
        life = item["life"]
        age = item["age"]

        annual_dep = (cost - salvage) / life
        book = cost - annual_dep * age
        book = max(book, salvage)

        dep_pct = ((cost - book) / cost * 100) if cost > 0 else 0.0

        total_original += cost
        total_book += book

        # Fully depreciated: book value equals salvage (with tolerance)
        is_fully_dep = abs(book - salvage) < 0.01
        if is_fully_dep:
            fully_dep_count += 1

        assets.append(AssetBookValue(
            asset=item["name"],
            original_cost=round(cost, 4),
            salvage_value=round(salvage, 4),
            age_years=round(age, 2),
            book_value=round(book, 4),
            depreciation_pct=round(dep_pct, 2),
        ))

    overall_dep_pct = ((total_original - total_book) / total_original * 100) if total_original > 0 else 0.0

    summary = (
        f"Book values for {len(assets)} assets: "
        f"Total original cost = {total_original:,.2f}, "
        f"Total book value = {total_book:,.2f} "
        f"({overall_dep_pct:.1f}% depreciated). "
        f"{fully_dep_count} fully depreciated."
    )

    return BookValueResult(
        assets=assets,
        total_original_cost=round(total_original, 4),
        total_book_value=round(total_book, 4),
        depreciation_pct=round(overall_dep_pct, 2),
        fully_depreciated_count=fully_dep_count,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 4. Maintenance Cost Ratio
# ---------------------------------------------------------------------------


def analyze_maintenance_cost_ratio(
    rows: list[dict],
    asset_column: str,
    maintenance_cost_column: str,
    asset_value_column: str,
    date_column: str | None = None,
) -> MaintenanceCostResult | None:
    """Compute maintenance cost as percentage of asset value.

    Assets where maintenance exceeds 50% of value are flagged as
    replacement candidates.

    Args:
        rows: Data rows as dicts.
        asset_column: Column identifying the asset.
        maintenance_cost_column: Column with maintenance cost.
        asset_value_column: Column with current asset value.
        date_column: Optional column for trend tracking over time.

    Returns:
        MaintenanceCostResult or None if no valid data.
    """
    if not rows:
        return None

    # Aggregate maintenance cost and asset value per asset
    asset_agg: dict[str, dict[str, float]] = {}
    for row in rows:
        name = row.get(asset_column)
        maint = _safe_float(row.get(maintenance_cost_column))
        value = _safe_float(row.get(asset_value_column))
        if name is None or maint is None or value is None:
            continue
        key = str(name)
        if key not in asset_agg:
            asset_agg[key] = {"maint": 0.0, "value": 0.0, "count": 0}
        asset_agg[key]["maint"] += maint
        asset_agg[key]["value"] += value
        asset_agg[key]["count"] += 1

    if not asset_agg:
        return None

    assets: list[AssetMaintRatio] = []
    total_maint = 0.0
    total_value = 0.0
    replacement_count = 0
    ratios: list[float] = []

    for asset_name in sorted(asset_agg.keys()):
        info = asset_agg[asset_name]
        maint_cost = info["maint"]
        # Use average value if multiple rows (the value represents the asset
        # value at each data point; averaging gives a representative figure)
        asset_val = info["value"] / info["count"] if info["count"] > 0 else info["value"]

        ratio = (maint_cost / asset_val * 100) if asset_val > 0 else 0.0
        is_candidate = ratio > 50.0

        if is_candidate:
            replacement_count += 1

        total_maint += maint_cost
        total_value += asset_val
        ratios.append(ratio)

        assets.append(AssetMaintRatio(
            asset=asset_name,
            maintenance_cost=round(maint_cost, 4),
            asset_value=round(asset_val, 4),
            ratio_pct=round(ratio, 2),
            is_replacement_candidate=is_candidate,
        ))

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    summary = (
        f"Maintenance cost analysis for {len(assets)} assets: "
        f"Average ratio = {avg_ratio:.1f}%. "
        f"{replacement_count} replacement candidate(s) (>50% ratio). "
        f"Total maintenance = {total_maint:,.2f}, "
        f"Total asset value = {total_value:,.2f}."
    )

    return MaintenanceCostResult(
        assets=assets,
        avg_ratio=round(avg_ratio, 2),
        replacement_candidates=replacement_count,
        total_maintenance=round(total_maint, 4),
        total_asset_value=round(total_value, 4),
        summary=summary,
    )


# ---------------------------------------------------------------------------
# 5. Format Asset Report
# ---------------------------------------------------------------------------


def format_asset_report(
    depreciation: DepreciationScheduleResult | None = None,
    age: AssetAgeResult | None = None,
    book_values: BookValueResult | None = None,
    maintenance: MaintenanceCostResult | None = None,
) -> str:
    """Generate a combined text report from available asset analyses.

    Each section is only included when the corresponding parameter is not None.

    Args:
        depreciation: Depreciation schedule result.
        age: Asset age analysis result.
        book_values: Book value result.
        maintenance: Maintenance cost ratio result.

    Returns:
        Formatted multi-section report string.
    """
    sections: list[str] = []
    sections.append("Asset Depreciation & Lifecycle Report")
    sections.append("=" * 45)

    if depreciation is not None:
        lines = ["", "Depreciation Schedule", "-" * 43]
        lines.append(f"  Method: {depreciation.method}")
        for a in depreciation.assets:
            lines.append(
                f"  {a.asset}: cost={a.cost:,.2f}, salvage={a.salvage:,.2f}, "
                f"life={a.useful_life}yr, annual_dep={a.annual_depreciation:,.2f}"
            )
            for yd in a.schedule:
                lines.append(
                    f"    Year {yd.year}: dep={yd.depreciation_amount:,.2f}, "
                    f"book={yd.book_value:,.2f}"
                )
        lines.append(f"  Total cost: {depreciation.total_cost:,.2f}")
        lines.append(f"  Total first-year depreciation: {depreciation.total_annual_depreciation:,.2f}")
        sections.append("\n".join(lines))

    if age is not None:
        lines = ["", "Asset Age Analysis", "-" * 43]
        lines.append(f"  Total assets: {age.total_assets}")
        lines.append(f"  Average age: {age.avg_age:.1f} years")
        for ls in age.by_lifecycle_stage:
            lines.append(f"  {ls.stage}: {ls.count} ({ls.pct:.1f}%)")
        if age.by_category:
            lines.append("  By category:")
            for ca in age.by_category:
                lines.append(f"    {ca.category}: {ca.count} assets, avg age {ca.avg_age:.1f}yr")
        if age.weighted_avg_age is not None:
            lines.append(f"  Weighted avg age (by cost): {age.weighted_avg_age:.1f} years")
        sections.append("\n".join(lines))

    if book_values is not None:
        lines = ["", "Book Values", "-" * 43]
        for a in book_values.assets:
            lines.append(
                f"  {a.asset}: original={a.original_cost:,.2f}, "
                f"book={a.book_value:,.2f} ({a.depreciation_pct:.1f}% depreciated)"
            )
        lines.append(f"  Total original: {book_values.total_original_cost:,.2f}")
        lines.append(f"  Total book value: {book_values.total_book_value:,.2f}")
        lines.append(f"  Overall depreciation: {book_values.depreciation_pct:.1f}%")
        lines.append(f"  Fully depreciated: {book_values.fully_depreciated_count}")
        sections.append("\n".join(lines))

    if maintenance is not None:
        lines = ["", "Maintenance Cost Analysis", "-" * 43]
        for a in maintenance.assets:
            flag = " [REPLACE]" if a.is_replacement_candidate else ""
            lines.append(
                f"  {a.asset}: maint={a.maintenance_cost:,.2f}, "
                f"value={a.asset_value:,.2f}, ratio={a.ratio_pct:.1f}%{flag}"
            )
        lines.append(f"  Average ratio: {maintenance.avg_ratio:.1f}%")
        lines.append(f"  Replacement candidates: {maintenance.replacement_candidates}")
        lines.append(f"  Total maintenance: {maintenance.total_maintenance:,.2f}")
        lines.append(f"  Total asset value: {maintenance.total_asset_value:,.2f}")
        sections.append("\n".join(lines))

    if depreciation is None and age is None and book_values is None and maintenance is None:
        sections.append("\nNo analysis data provided.")

    return "\n".join(sections)
