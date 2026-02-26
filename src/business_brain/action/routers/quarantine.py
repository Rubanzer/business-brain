"""Data Quarantine routes — view, approve, and reject quarantined rows."""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session
from business_brain.db.v3_models import DataQuarantine, SanctityIssue

logger = logging.getLogger(__name__)

router = APIRouter(tags=["quarantine"])


# ---------------------------------------------------------------------------
# Request Models
# ---------------------------------------------------------------------------


class ReviewRequest(BaseModel):
    reviewed_by: str
    note: Optional[str] = None


class BulkReviewRequest(BaseModel):
    ids: list[int]
    action: str  # "approve" or "reject"
    reviewed_by: str
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/quarantine")
async def list_quarantined(
    table_name: Optional[str] = Query(None),
    status: str = Query("quarantined"),
    limit: int = Query(50, le=200),
    session: AsyncSession = Depends(get_session),
) -> dict:
    """List quarantined rows, optionally filtered by table and status."""
    try:
        query = (
            select(DataQuarantine)
            .where(DataQuarantine.validation_status == status)
            .order_by(DataQuarantine.quarantined_at.desc())
            .limit(limit)
        )
        if table_name:
            query = query.where(DataQuarantine.table_name == table_name)

        result = await session.execute(query)
        rows = list(result.scalars().all())

        # Also get summary counts
        count_query = (
            select(
                DataQuarantine.validation_status,
                func.count(DataQuarantine.id).label("count"),
            )
            .group_by(DataQuarantine.validation_status)
        )
        if table_name:
            count_query = count_query.where(DataQuarantine.table_name == table_name)

        count_result = await session.execute(count_query)
        counts = {row.validation_status: row.count for row in count_result.all()}

        return {
            "rows": [
                {
                    "id": r.id,
                    "table_name": r.table_name,
                    "upload_batch_id": r.upload_batch_id,
                    "row_index": r.row_index,
                    "row_data": r.row_data,
                    "issues": r.issues,
                    "validation_status": r.validation_status,
                    "quarantined_at": r.quarantined_at.isoformat() if r.quarantined_at else None,
                    "reviewed_by": r.reviewed_by,
                    "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                    "review_note": r.review_note,
                }
                for r in rows
            ],
            "total": len(rows),
            "counts": counts,
        }
    except Exception:
        logger.exception("Failed to list quarantined rows")
        return {"rows": [], "total": 0, "counts": {}}


@router.get("/quarantine/summary")
async def quarantine_summary(
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get summary of quarantined data across all tables."""
    try:
        query = select(
            DataQuarantine.table_name,
            DataQuarantine.validation_status,
            func.count(DataQuarantine.id).label("count"),
        ).group_by(DataQuarantine.table_name, DataQuarantine.validation_status)

        result = await session.execute(query)
        raw = result.all()

        tables: dict[str, dict] = {}
        for row in raw:
            if row.table_name not in tables:
                tables[row.table_name] = {"quarantined": 0, "approved": 0, "rejected": 0}
            tables[row.table_name][row.validation_status] = row.count

        return {
            "tables": tables,
            "total_quarantined": sum(t.get("quarantined", 0) for t in tables.values()),
            "total_approved": sum(t.get("approved", 0) for t in tables.values()),
            "total_rejected": sum(t.get("rejected", 0) for t in tables.values()),
        }
    except Exception:
        logger.exception("Failed to get quarantine summary")
        return {"tables": {}, "total_quarantined": 0, "total_approved": 0, "total_rejected": 0}


@router.get("/quarantine/batch/{batch_id}")
async def get_batch(
    batch_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get all quarantined rows from a specific upload batch."""
    try:
        result = await session.execute(
            select(DataQuarantine)
            .where(DataQuarantine.upload_batch_id == batch_id)
            .order_by(DataQuarantine.row_index)
        )
        rows = list(result.scalars().all())
        if not rows:
            return {"error": "Batch not found"}

        return {
            "batch_id": batch_id,
            "table_name": rows[0].table_name,
            "total": len(rows),
            "rows": [
                {
                    "id": r.id,
                    "row_index": r.row_index,
                    "row_data": r.row_data,
                    "issues": r.issues,
                    "validation_status": r.validation_status,
                    "reviewed_by": r.reviewed_by,
                }
                for r in rows
            ],
        }
    except Exception:
        logger.exception("Failed to get quarantine batch %s", batch_id)
        return {"error": "Failed to fetch batch"}


@router.post("/quarantine/{item_id}/approve")
async def approve_quarantined(
    item_id: int,
    req: ReviewRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Approve a quarantined row — insert it into the target table."""
    try:
        result = await session.execute(
            select(DataQuarantine).where(DataQuarantine.id == item_id)
        )
        item = result.scalar_one_or_none()
        if not item:
            return {"error": "Quarantined item not found"}

        if item.validation_status != "quarantined":
            return {"error": f"Item already {item.validation_status}"}

        # Insert the row into the target table
        row_data = item.row_data
        if row_data and isinstance(row_data, dict):
            columns = list(row_data.keys())
            col_list = ", ".join(f'"{c}"' for c in columns)
            param_list = ", ".join(f":p{i}" for i in range(len(columns)))
            params = {f"p{i}": v for i, v in enumerate(row_data.values())}

            try:
                sql = f'INSERT INTO "{item.table_name}" ({col_list}) VALUES ({param_list})'
                await session.execute(text(sql), params)
            except Exception as insert_err:
                logger.warning("Insert failed for quarantine item %d: %s", item_id, insert_err)
                return {"error": f"Insert failed: {str(insert_err)}"}

        # Mark as approved
        item.validation_status = "approved"
        item.reviewed_by = req.reviewed_by
        item.reviewed_at = datetime.now(timezone.utc)
        item.review_note = req.note
        await session.commit()

        return {"status": "approved", "id": item_id, "table_name": item.table_name}

    except Exception:
        logger.exception("Failed to approve quarantined item %d", item_id)
        await session.rollback()
        return {"error": "Failed to approve item"}


@router.post("/quarantine/{item_id}/reject")
async def reject_quarantined(
    item_id: int,
    req: ReviewRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Reject a quarantined row — discard it."""
    try:
        result = await session.execute(
            select(DataQuarantine).where(DataQuarantine.id == item_id)
        )
        item = result.scalar_one_or_none()
        if not item:
            return {"error": "Quarantined item not found"}

        if item.validation_status != "quarantined":
            return {"error": f"Item already {item.validation_status}"}

        item.validation_status = "rejected"
        item.reviewed_by = req.reviewed_by
        item.reviewed_at = datetime.now(timezone.utc)
        item.review_note = req.note
        await session.commit()

        return {"status": "rejected", "id": item_id}

    except Exception:
        logger.exception("Failed to reject quarantined item %d", item_id)
        await session.rollback()
        return {"error": "Failed to reject item"}


@router.post("/quarantine/bulk-review")
async def bulk_review(
    req: BulkReviewRequest,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Approve or reject multiple quarantined rows at once."""
    if req.action not in ("approve", "reject"):
        return {"error": "Action must be 'approve' or 'reject'"}

    results = {"approved": 0, "rejected": 0, "errors": []}
    now = datetime.now(timezone.utc)

    try:
        query_result = await session.execute(
            select(DataQuarantine).where(
                DataQuarantine.id.in_(req.ids),
                DataQuarantine.validation_status == "quarantined",
            )
        )
        items = list(query_result.scalars().all())

        for item in items:
            if req.action == "approve":
                # Insert into target table
                row_data = item.row_data
                if row_data and isinstance(row_data, dict):
                    columns = list(row_data.keys())
                    col_list = ", ".join(f'"{c}"' for c in columns)
                    param_list = ", ".join(f":p{i}" for i in range(len(columns)))
                    params = {f"p{i}": v for i, v in enumerate(row_data.values())}
                    try:
                        sql = f'INSERT INTO "{item.table_name}" ({col_list}) VALUES ({param_list})'
                        await session.execute(text(sql), params)
                    except Exception as insert_err:
                        results["errors"].append(
                            {"id": item.id, "error": str(insert_err)}
                        )
                        continue

                item.validation_status = "approved"
                results["approved"] += 1
            else:
                item.validation_status = "rejected"
                results["rejected"] += 1

            item.reviewed_by = req.reviewed_by
            item.reviewed_at = now
            item.review_note = req.note

        await session.commit()
        return {"status": "completed", **results}

    except Exception:
        logger.exception("Bulk review failed")
        await session.rollback()
        return {"error": "Bulk review failed"}
