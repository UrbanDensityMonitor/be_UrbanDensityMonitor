from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional
from datetime import datetime

from app.db.asyncpg_client import get_db_pool
from app.auth.jwt_handler import verify_jwt

import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/alerts",
    tags=["Alerts & Notifications"]
)

@router.get("/")
async def get_alerts(
    stream_id: Optional[str] = Query(None, description="Filter alert CCTV tertentu"),
    is_read: Optional[bool] = Query(None, description="Filter yang belum/sudah dibaca"),
    limit: int = Query(20, description="Maksimal data alert"),
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
            # Private stream → hanya owner yang boleh melihat alerts-nya
            if str(owner_row["user_id"]) != user_id:
                raise HTTPException(
                    status_code=403,
                    detail="⛔ Anda tidak memiliki akses ke alerts stream ini."
                )

        # Sudah terverifikasi: public stream atau private milik user
        query = "SELECT id, traffic_history_id, stream_id, alert_type, alert_message, is_read, created_at FROM alerts WHERE stream_id = $1"
        params = [stream_id]
        counter = 2
    else:
        # Tanpa filter stream_id: tampilkan public alerts + private alerts milik user saja
        query = """SELECT id, traffic_history_id, stream_id, alert_type, alert_message, is_read, created_at
        FROM alerts
        WHERE (
            stream_id IN (SELECT id FROM streams)
            OR stream_id IN (SELECT id FROM private_streams WHERE user_id = $1)
        )"""
        params = [user_id]
        counter = 2

    if is_read is not None:
        query += f" AND is_read = ${counter}"
        params.append(is_read)
        counter += 1

    query += f" ORDER BY created_at DESC LIMIT ${counter}"
    params.append(limit)

    rows = await pool.fetch(query, *params)
    alert_list = []
    for row in rows:
        item = dict(row)
        item["id"] = str(item["id"])
        item["traffic_history_id"] = str(item["traffic_history_id"])
        item["stream_id"] = str(item["stream_id"])
        if isinstance(item["created_at"], datetime):
            item["created_at"] = item["created_at"].isoformat()
        alert_list.append(item)

    return {"data": alert_list}

@router.patch("/{alert_id}/read")
async def mark_alert_read(alert_id: str, user_info: dict = Depends(verify_jwt)):
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database belum siap!")

    user_id = user_info.get("sub")

    # Ambil alert beserta stream_id untuk cek ownership
    alert_row = await pool.fetchrow(
        "SELECT id, stream_id FROM alerts WHERE id = $1", alert_id
    )
    if not alert_row:
        raise HTTPException(status_code=404, detail="❌ Alert tidak ditemukan.")

    # Cek apakah stream_id alert ini milik private_streams
    owner_row = await pool.fetchrow(
        "SELECT user_id FROM private_streams WHERE id = $1", alert_row["stream_id"]
    )
    if owner_row:
        # Private stream alert → hanya owner yang boleh mark as read
        if str(owner_row["user_id"]) != user_id:
            raise HTTPException(
                status_code=403,
                detail="⛔ Anda tidak memiliki akses ke alert ini."
            )

    # Public alert atau private alert milik user → lanjut update
    query = "UPDATE alerts SET is_read = true WHERE id = $1"
    await pool.execute(query, alert_id)
    return {"message": f"✅ Alert {alert_id} berhasil ditandai sudah dibaca"}

