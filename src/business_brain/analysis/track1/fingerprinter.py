"""DataFingerprinter — enriched column classification for analysis.

Extends the existing column_classifier with:
- Role detection: MEASURE, DIMENSION, TIME_INDEX, GRAIN_KEY, FOREIGN_KEY, FREE_TEXT
- Distribution fitting for numeric columns
- Relationship readiness: which columns can join to other tables
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.analysis.tools import compute
from business_brain.db.discovery_models import DiscoveredRelationship, TableProfile
from business_brain.cognitive.column_classifier import classify_columns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for role detection
# ---------------------------------------------------------------------------

_ID_RE = re.compile(
    r"(^id$|_id$|^key$|_key$|^code$|_code$|_no$|_number$|^number$"
    r"|^invoice|^order_id|^employee_id|^customer_id|_row_id$"
    r"|^heat_no|^machine_id|^batch_id|^lot_no|^wo_id|^asset_code"
    r"|^challan|^vehicle_no|^truck_no|^serial_no|^equipment_id)",
    re.IGNORECASE,
)

_TIME_RE = re.compile(
    r"(date|_at$|_on$|timestamp|^time$|^month$|^year$|^quarter$|^week$"
    r"|^period$|^day$|^created|^updated|^modified)",
    re.IGNORECASE,
)

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9_]")


def _safe(name: str) -> str:
    return _SAFE_NAME_RE.sub("", name)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ColumnFingerprint:
    name: str
    semantic_type: str  # from column_classifier
    role: str  # MEASURE / DIMENSION / TIME_INDEX / GRAIN_KEY / FOREIGN_KEY / FREE_TEXT
    distribution: dict | None = None  # {type, params, fit_score}
    cardinality: int = 0
    null_rate: float = 0.0
    joinable_to: list[dict] = field(default_factory=list)


@dataclass
class TableFingerprint:
    table_name: str
    row_count: int
    data_hash: str
    domain_hint: str
    time_index: str | None
    measures: list[str]
    dimensions: list[str]
    columns: dict[str, ColumnFingerprint] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Role inference
# ---------------------------------------------------------------------------

_MEASURE_TYPES = {"numeric_metric", "numeric_currency", "numeric_percentage"}
_DIMENSION_TYPES = {"categorical", "boolean"}
_TIME_TYPES = {"temporal"}
_TEXT_TYPES = {"text"}
_ID_TYPES = {"identifier"}


def _infer_role(
    col_name: str,
    semantic_type: str,
    cardinality_ratio: float,
    is_foreign_key: bool,
) -> str:
    """Infer the analysis role of a column."""
    if is_foreign_key:
        return "FOREIGN_KEY"
    if semantic_type in _TIME_TYPES or _TIME_RE.search(col_name):
        return "TIME_INDEX"
    if semantic_type in _ID_TYPES or (cardinality_ratio > 0.9 and _ID_RE.search(col_name)):
        return "GRAIN_KEY"
    if semantic_type in _MEASURE_TYPES:
        return "MEASURE"
    if semantic_type in _DIMENSION_TYPES:
        return "DIMENSION"
    if semantic_type in _TEXT_TYPES:
        return "FREE_TEXT"
    return "DIMENSION"


# ---------------------------------------------------------------------------
# Core fingerprinting
# ---------------------------------------------------------------------------


async def fingerprint_table(
    session: AsyncSession,
    table_name: str,
    profile: TableProfile | None = None,
) -> TableFingerprint:
    """Build a full TableFingerprint from a table's data and metadata."""

    # 1. Load or fetch profile
    if profile is None:
        result = await session.execute(
            select(TableProfile).where(TableProfile.table_name == table_name)
        )
        profile = result.scalar_one_or_none()

    row_count = profile.row_count if profile else 0
    data_hash = profile.data_hash or "" if profile else ""
    classification = profile.column_classification if profile else None
    domain_hint = profile.domain_hint or "general" if profile else "general"

    # 2. If no cached classification, fetch sample and classify
    if not classification:
        safe_table = _safe(table_name)
        try:
            sample_result = await session.execute(
                sql_text(f'SELECT * FROM "{safe_table}" LIMIT 100')
            )
            sample_rows = [dict(r._mapping) for r in sample_result.fetchall()]
            columns = list(sample_rows[0].keys()) if sample_rows else []
            classification_result = classify_columns(columns, sample_rows)
            classification = classification_result.get("columns", {})
            domain_hint = classification_result.get("domain_hint", "general")
        except Exception:
            logger.warning("Failed to classify columns for %s", table_name, exc_info=True)
            classification = {}

    # 3. Get row count if missing
    if row_count == 0:
        safe_table = _safe(table_name)
        try:
            cnt = await session.execute(sql_text(f'SELECT COUNT(*) FROM "{safe_table}"'))
            row_count = cnt.scalar() or 0
        except Exception:
            pass

    # 4. Compute data_hash if missing
    if not data_hash:
        safe_table = _safe(table_name)
        try:
            hash_result = await session.execute(
                sql_text(f"SELECT MD5(CAST(COUNT(*) AS TEXT) || CAST(MAX(ctid) AS TEXT)) FROM \"{safe_table}\"")
            )
            data_hash = hash_result.scalar() or hashlib.md5(table_name.encode()).hexdigest()
        except Exception:
            data_hash = hashlib.md5(table_name.encode()).hexdigest()

    # 5. Load discovered relationships for cross-table join readiness
    rel_result = await session.execute(
        select(DiscoveredRelationship).where(
            (DiscoveredRelationship.table_a == table_name)
            | (DiscoveredRelationship.table_b == table_name)
        )
    )
    relationships = rel_result.scalars().all()

    # Build joinable_to lookup: column_name -> [{table, column, confidence}]
    joinable_map: dict[str, list[dict]] = {}
    for rel in relationships:
        if rel.table_a == table_name:
            joinable_map.setdefault(rel.column_a, []).append({
                "table": rel.table_b,
                "column": rel.column_b,
                "confidence": rel.confidence,
            })
        else:
            joinable_map.setdefault(rel.column_b, []).append({
                "table": rel.table_a,
                "column": rel.column_a,
                "confidence": rel.confidence,
            })

    # 6. Build per-column fingerprints
    col_fingerprints: dict[str, ColumnFingerprint] = {}
    measures: list[str] = []
    dimensions: list[str] = []
    time_index: str | None = None

    fk_columns = set(joinable_map.keys())

    for col_name, col_info in (classification or {}).items():
        if isinstance(col_info, str):
            semantic_type = col_info
            stats = {}
        else:
            semantic_type = col_info.get("type", "text")
            stats = col_info.get("stats", {})

        cardinality = stats.get("cardinality", 0)
        null_pct = stats.get("null_pct", 0.0)
        cardinality_ratio = cardinality / row_count if row_count > 0 else 0.0

        is_fk = col_name in fk_columns
        role = _infer_role(col_name, semantic_type, cardinality_ratio, is_fk)

        # Distribution fitting for measures
        distribution = None
        if role == "MEASURE" and stats.get("sample_values"):
            try:
                nums = [float(v) for v in stats["sample_values"] if v is not None]
                if len(nums) >= 20:
                    distribution = compute.detect_distribution(nums)
            except (ValueError, TypeError):
                pass

        fp = ColumnFingerprint(
            name=col_name,
            semantic_type=semantic_type,
            role=role,
            distribution=distribution,
            cardinality=int(cardinality),
            null_rate=float(null_pct) / 100.0 if null_pct > 1 else float(null_pct),
            joinable_to=joinable_map.get(col_name, []),
        )
        col_fingerprints[col_name] = fp

        if role == "MEASURE":
            measures.append(col_name)
        elif role == "DIMENSION":
            dimensions.append(col_name)
        elif role == "TIME_INDEX" and time_index is None:
            time_index = col_name

    return TableFingerprint(
        table_name=table_name,
        row_count=row_count,
        data_hash=data_hash,
        domain_hint=domain_hint,
        time_index=time_index,
        measures=measures,
        dimensions=dimensions,
        columns=col_fingerprints,
    )
