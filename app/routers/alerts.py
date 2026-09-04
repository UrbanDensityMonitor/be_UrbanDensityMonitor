"""
Router untuk Alerts endpoints
Semua SQL ada di repository layer (Clean Architecture)
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Depends

from app.auth.jwt_handler import verify_jwt
from app.core.dependencies import get_alert_repository
from app.repositories.alert_repository import AlertRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/alerts",
    tags=["Alerts & Notifications"]
)


def _serialize_row(row: dict) -> dict:
    """Serialize record: UUID -> str, datetime -> ISO string"""
    item = dict(row)
    for key in ("id", "traffic_history_id", "stream_id"):
        if key in item and item[key] is not None:
            item[key] = str(item[key])
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    return item


@router.get("")
async def get_alerts(
    stream_id: Optional[str] = Query(None, description="Filter alert CCTV tertentu"),
    is_read: Optional[bool] = Query(None, description="Filter yang belum/sudah dibaca"),
    limit: int = Query(20, description="Maksimal data alert"),
    repo: AlertRepository = Depends(get_alert_repository),
    user_info: dict = Depends(verify_jwt)
):
    """Get daftar alert (optional filter per stream & status baca)"""
    try:
        rows = await repo.find_all(
            stream_id=stream_id,
            is_read=is_read,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Error fetching alerts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mengambil data alert")

    alert_list = [_serialize_row(row) for row in rows]

    return {"data": alert_list}


@router.patch("/{alert_id}/read")
async def mark_alert_read(
    alert_id: str,
    repo: AlertRepository = Depends(get_alert_repository),
    user_info: dict = Depends(verify_jwt)
):
    """Tandai alert sebagai sudah dibaca"""
    try:
        updated = await repo.mark_as_read(alert_id)
    except Exception as e:
        logger.error(f"Error marking alert {alert_id} as read: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal menandai alert")

    if not updated:
        raise HTTPException(status_code=404, detail="Alert tidak ditemukan")

    return {"message": f"✅ Alert {alert_id} berhasil ditandai sudah dibaca"}
