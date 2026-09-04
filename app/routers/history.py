"""
Router untuk Traffic History endpoints
Semua SQL ada di repository layer (Clean Architecture)
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Depends

from app.auth.jwt_handler import verify_jwt
from app.core.dependencies import get_traffic_history_repository
from app.repositories.traffic_history_repository import TrafficHistoryRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/history",
    tags=["Traffic History"]
)


def _serialize_row(row: dict) -> dict:
    """Serialize record: UUID -> str, datetime -> ISO string"""
    item = dict(row)
    for key in ("id", "stream_id"):
        if key in item and item[key] is not None:
            item[key] = str(item[key])
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    return item


@router.get("")
async def get_history(
    stream_id: Optional[str] = Query(None, description="Filter berdasarkan ID CCTV"),
    limit: int = Query(50, description="Jumlah maksimal data"),
    offset: int = Query(0, description="Mulai dari urutan ke berapa"),
    repo: TrafficHistoryRepository = Depends(get_traffic_history_repository),
    user_info: dict = Depends(verify_jwt)
):
    """Get riwayat deteksi kepadatan lalu lintas (optional filter per stream)"""
    try:
        if stream_id:
            rows = await repo.find_by_stream(stream_id, limit=limit, offset=offset)
        else:
            rows = await repo.find_all(limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"Error fetching history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mengambil data history")

    history_list = [_serialize_row(row) for row in rows]

    return {
        "data": history_list,
        "limit": limit,
        "offset": offset,
        "total_returned": len(history_list)
    }
