"""Focus Mode (Table Scoping) routes."""

import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session
from business_brain.memory import metadata_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["focus"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/focus")
async def get_focus(session: AsyncSession = Depends(get_session)) -> dict:
    """Get current focus scope (which tables are included/excluded)."""
    from sqlalchemy import select
    from business_brain.db.v3_models import FocusScope

    try:
        result = await session.execute(select(FocusScope))
        rows = list(result.scalars().all())
        all_entries = await metadata_store.get_all(session)
        all_table_names = [e.table_name for e in all_entries]

        focused = {r.table_name: r.is_included for r in rows}
        tables = []
        for t in all_table_names:
            tables.append({
                "table_name": t,
                "is_included": focused.get(t, True),
            })

        active = any(not v for v in focused.values()) or (len(focused) > 0 and len(focused) < len(all_table_names))
        return {"active": bool(rows) and active, "tables": tables, "total": len(all_table_names)}
    except Exception:
        logger.exception("Error fetching focus scope")
        return {"active": False, "tables": [], "total": 0}


@router.put("/focus")
async def update_focus(body: dict, session: AsyncSession = Depends(get_session)) -> dict:
    """Update focus scope — set which tables are included/excluded."""
    from sqlalchemy import select
    from business_brain.db.v3_models import FocusScope

    tables = body.get("tables", [])
    if not tables:
        raise HTTPException(status_code=400, detail="'tables' array is required")

    try:
        all_entries = await metadata_store.get_all(session)
        valid_names = {e.table_name for e in all_entries}
        for t in tables:
            if t.get("table_name") not in valid_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"Table '{t.get('table_name')}' not found in metadata"
                )

        from sqlalchemy import delete as sa_delete
        await session.execute(sa_delete(FocusScope))

        for t in tables:
            scope = FocusScope(
                table_name=t["table_name"],
                is_included=t.get("is_included", True),
            )
            session.add(scope)

        await session.commit()

        included = sum(1 for t in tables if t.get("is_included", True))
        return {"status": "updated", "total": len(tables), "included": included}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Error updating focus scope")
        await session.rollback()
        return JSONResponse({"error": "Failed to update focus scope"}, status_code=500)


@router.delete("/focus")
async def clear_focus(session: AsyncSession = Depends(get_session)) -> dict:
    """Clear focus scope — disable focus mode (analyze all tables)."""
    from business_brain.db.v3_models import FocusScope
    from sqlalchemy import delete as sa_delete

    try:
        result = await session.execute(sa_delete(FocusScope))
        await session.commit()
        return {"status": "cleared", "rows_removed": result.rowcount}
    except Exception:
        logger.exception("Error clearing focus scope")
        await session.rollback()
        return {"status": "cleared", "rows_removed": 0}
