from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional
import logging

from app.db.asyncpg_client import get_db
from app.auth.jwt_handler import verify_jwt
from app.db.repositories.alert_repository import AlertRepository
from asyncpg import Pool

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/alerts",
    tags=["Alerts & Notifications"]
)

@router.get("/")
async def get_alerts(
    stream_id: Optional[str] = Query(None, description="Filter alert CCTV tertentu"),
    is_read: Optional[bool] = Query(None, description="Filter yang belum/sudah dibaca"),
    limit: int = Query(20, description="Maksimal data alert"),
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = AlertRepository(db)
    user_id = user_info.get("sub")

    try:
        if stream_id:
            owner_id = await repo.get_private_stream_owner(stream_id)
            if owner_id and owner_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="⛔ Anda tidak memiliki akses ke alerts stream ini."
                )
            alert_list = await repo.get_alerts_by_stream(stream_id, is_read, limit)
        else:
            alert_list = await repo.get_all_alerts_for_user(user_id, is_read, limit)

        return {"data": alert_list}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error di alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.patch("/{alert_id}/read")
async def mark_alert_read(
    alert_id: str, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = AlertRepository(db)
    user_id = user_info.get("sub")

    try:
        alert_row = await repo.get_alert(alert_id)
        if not alert_row:
            raise HTTPException(status_code=404, detail="❌ Alert tidak ditemukan.")

        owner_id = await repo.get_private_stream_owner(alert_row["stream_id"])
        if owner_id and owner_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="⛔ Anda tidak memiliki akses ke alert ini."
            )

        await repo.mark_as_read(alert_id)
        return {"message": f"✅ Alert {alert_id} berhasil ditandai sudah dibaca"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error di mark alert read: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
