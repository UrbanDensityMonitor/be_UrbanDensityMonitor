from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional
from datetime import datetime

from app.db.asyncpg_client import get_db_pool
from app.auth.jwt_handler import verify_jwt

import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/history",
    tags=["Traffic History"]
)

@router.get("/")
async def get_history(
    stream_id: Optional[str] = Query(None, description="Filter berdasarkan ID CCTV"),
    limit: int = Query(50, description="Jumlah maksimal data"),
    offset: int = Query(0, description="Mulai dari urutan ke berapa"),
    user_info: dict = Depends(verify_jwt)
):
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database belum siap!")

    user_id = user_info.get("sub")

    if stream_id:
        # Cek apakah stream_id milik private_streams
        owner_row = await pool.fetchrow(
            "SELECT user_id FROM private_streams WHERE id = $1", stream_id
        )
        if owner_row:
            # Private stream → hanya owner yang boleh melihat history-nya
            if str(owner_row["user_id"]) != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="⛔ Anda tidak memiliki akses ke history stream ini."
                )

        query = """
        SELECT id, stream_id, person_count, motorcycle_count, car_count, bus_count, truck_count,
               total_vehicle_count, person_vehicle_ratio, density_status, recorded_at,
               average_speed, road_occupancy, congestion_index, stream_source
        FROM traffic_history
        WHERE stream_id = $1
        ORDER BY recorded_at DESC
        LIMIT $2 OFFSET $3
        """
        rows = await pool.fetch(query, stream_id, limit, offset)
    else:
        # Tanpa filter stream_id: tampilkan semua public + private milik user saja
        query = """
        SELECT id, stream_id, person_count, motorcycle_count, car_count, bus_count, truck_count,
               total_vehicle_count, person_vehicle_ratio, density_status, recorded_at,
               average_speed, road_occupancy, congestion_index, stream_source
        FROM traffic_history
        WHERE stream_source = 'public'
           OR (stream_source = 'private' AND stream_id IN (
               SELECT id FROM private_streams WHERE user_id = $1
           ))
        ORDER BY recorded_at DESC
        LIMIT $2 OFFSET $3
        """
        rows = await pool.fetch(query, user_id, limit, offset)

    history_list = []
    for row in rows:
        item = dict(row)
        item["id"] = str(item["id"])
        item["stream_id"] = str(item["stream_id"])
        if isinstance(item["recorded_at"], datetime):
            item["recorded_at"] = item["recorded_at"].isoformat()
        history_list.append(item)

    return {"data": history_list, "limit": limit, "offset": offset, "total_returned": len(history_list)}

