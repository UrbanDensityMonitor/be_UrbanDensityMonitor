from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional
import logging

from app.db.asyncpg_client import get_db
from app.auth.jwt_handler import verify_jwt
from app.db.repositories.history_repository import HistoryRepository
from asyncpg import Pool

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/history",
    tags=["Traffic History"]
)

@router.get("/")
async def get_history(
    stream_id: Optional[str] = Query(None, description="Filter berdasarkan ID CCTV"),
    limit: int = Query(50, description="Jumlah maksimal data"),
    offset: int = Query(0, description="Mulai dari urutan ke berapa"),
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = HistoryRepository(db)
    user_id = user_info.get("sub")

    try:
        if stream_id:
            owner_id = await repo.get_private_stream_owner(stream_id)
            if owner_id and owner_id != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="⛔ Anda tidak memiliki akses ke history stream ini."
                )
            
            history_list = await repo.get_history_by_stream(stream_id, limit, offset)
        else:
            history_list = await repo.get_all_history_for_user(user_id, limit, offset)

        return {"data": history_list, "limit": limit, "offset": offset, "total_returned": len(history_list)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error di history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
