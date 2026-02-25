"""Reports routes."""

import json
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from business_brain.db.connection import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["reports"])


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/reports")
async def list_reports(session: AsyncSession = Depends(get_session)) -> list[dict]:
    """List all deployed reports."""
    from business_brain.discovery.feed_store import get_reports

    reports = await get_reports(session)
    return [
        {
            "id": r.id,
            "name": r.name,
            "insight_id": r.insight_id,
            "query": r.query,
            "chart_spec": r.chart_spec,
            "last_result": r.last_result,
            "last_run_at": r.last_run_at.isoformat() if r.last_run_at else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "active": r.active,
        }
        for r in reports
    ]


@router.get("/reports/{report_id}")
async def get_report(
    report_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Get a single report with latest data."""
    from business_brain.discovery.feed_store import get_report as _get_report

    report = await _get_report(session, report_id)
    if not report:
        return {"error": "Report not found"}
    return {
        "id": report.id,
        "name": report.name,
        "insight_id": report.insight_id,
        "query": report.query,
        "chart_spec": report.chart_spec,
        "last_result": report.last_result,
        "last_run_at": report.last_run_at.isoformat() if report.last_run_at else None,
        "created_at": report.created_at.isoformat() if report.created_at else None,
        "active": report.active,
    }


@router.post("/reports/{report_id}/refresh")
async def refresh_report(
    report_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Re-run a report's query and update results."""
    from business_brain.discovery.feed_store import refresh_report as _refresh

    report = await _refresh(session, report_id)
    if not report:
        return {"error": "Report not found"}
    return {
        "status": "refreshed",
        "report_id": report.id,
        "last_run_at": report.last_run_at.isoformat() if report.last_run_at else None,
        "row_count": len(report.last_result) if isinstance(report.last_result, list) else 0,
    }


@router.delete("/reports/{report_id}")
async def delete_report(
    report_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Remove a deployed report."""
    from business_brain.discovery.feed_store import delete_report as _delete

    deleted = await _delete(session, report_id)
    if not deleted:
        return {"error": "Report not found"}
    return {"status": "deleted", "report_id": report_id}


@router.get("/reports/{report_id}/export")
async def export_report(
    report_id: str,
    format: str = "json",
    session: AsyncSession = Depends(get_session),
):
    """Export a single deployed report's data as CSV or JSON."""
    from fastapi.responses import Response
    from business_brain.discovery.feed_store import get_report as _get_report

    report = await _get_report(session, report_id)
    if not report:
        return {"error": "Report not found"}

    rows = report.last_result if isinstance(report.last_result, list) else []
    export_data = {
        "report_name": report.name,
        "insight_id": report.insight_id,
        "query": report.query,
        "chart_spec": report.chart_spec,
        "last_run_at": report.last_run_at.isoformat() if report.last_run_at else None,
        "data": rows,
    }

    if format == "csv" and rows:
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: str(v) if v is not None else "" for k, v in row.items()})

        safe_name = report.name.replace(" ", "_").replace("/", "_")[:50]
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}.csv"'},
        )

    return Response(
        content=json.dumps(export_data, default=str, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{report_id[:8]}.json"'},
    )
