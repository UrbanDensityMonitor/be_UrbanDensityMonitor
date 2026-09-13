from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.db.asyncpg_client import get_db_pool
from app.auth.jwt_handler import verify_jwt

import logging

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/private-streams",
    tags=["Private CCTV Streams"]
)


class PrivateStreamCreate(BaseModel):
    camera_name: str
    location_name: str
    stream_url: str
    protocol: Optional[str] = "rtsp"
    stream_type: Optional[str] = "live"


class PrivateStreamUpdate(BaseModel):
    camera_name: Optional[str] = None
    location_name: Optional[str] = None
    stream_url: Optional[str] = None
    protocol: Optional[str] = None
    stream_type: Optional[str] = None


@router.get("/")
async def get_private_streams(user_info: dict = Depends(verify_jwt)):
    """List semua private cameras milik authenticated user."""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database belum siap!")

    user_id = user_info.get("sub")
    query = """
    SELECT id, user_id, camera_name, location_name, stream_url, protocol, stream_type, status, created_at
    FROM private_streams
    WHERE user_id = $1
    ORDER BY created_at DESC
    """
    try:
        rows = await pool.fetch(query, user_id)
        streams_data = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["user_id"] = str(item["user_id"])
            if isinstance(item.get("created_at"), datetime):
                item["created_at"] = item["created_at"].isoformat()
            streams_data.append(item)
        return {"data": streams_data, "total": len(streams_data)}
    except Exception as e:
        logger.error(f"❌ Error get private streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_private_stream(stream: PrivateStreamCreate, user_info: dict = Depends(verify_jwt)):
    """Tambah private camera baru. user_id diambil dari JWT, bukan request body."""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database belum siap!")

    user_id = user_info.get("sub")
    query = """
    INSERT INTO private_streams (user_id, camera_name, location_name, stream_url, protocol, stream_type)
    VALUES ($1, $2, $3, $4, $5, $6)
    RETURNING id
    """
    try:
        new_id = await pool.fetchval(
            query,
            user_id,
            stream.camera_name,
            stream.location_name,
            stream.stream_url,
            stream.protocol,
            stream.stream_type
        )
        return {
            "message": f"✅ Private camera '{stream.camera_name}' berhasil ditambahkan!",
            "id": str(new_id)
        }
    except Exception as e:
        error_msg = str(e)
        # Tangkap error dari DB trigger limit 5 camera
        if "maximum" in error_msg.lower() or "limit" in error_msg.lower() or "5" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="⛔ Batas maksimal 5 private camera tercapai. Hapus salah satu camera sebelum menambah yang baru."
            )
        logger.error(f"❌ Error create private stream: {e}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.put("/{stream_id}")
async def update_private_stream(stream_id: str, stream: PrivateStreamUpdate, user_info: dict = Depends(verify_jwt)):
    """Update private camera. Hanya owner yang boleh mengubah."""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database belum siap!")

    user_id = user_info.get("sub")
    query = """
    UPDATE private_streams
    SET camera_name   = COALESCE($1, camera_name),
        location_name = COALESCE($2, location_name),
        stream_url    = COALESCE($3, stream_url),
        protocol      = COALESCE($4, protocol),
        stream_type   = COALESCE($5, stream_type)
    WHERE id = $6 AND user_id = $7
    RETURNING id
    """
    try:
        updated_id = await pool.fetchval(
            query,
            stream.camera_name,
            stream.location_name,
            stream.stream_url,
            stream.protocol,
            stream.stream_type,
            stream_id,
            user_id
        )
        if not updated_id:
            raise HTTPException(
                status_code=404,
                detail="❌ Camera tidak ditemukan atau bukan milik Anda."
            )
        return {"message": f"✅ Private camera {stream_id} berhasil diperbarui!"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error update private stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{stream_id}")
async def delete_private_stream(stream_id: str, user_info: dict = Depends(verify_jwt)):
    """Hapus private camera. Hanya owner yang boleh menghapus."""
    pool = get_db_pool()
    if not pool:
        raise HTTPException(status_code=500, detail="Database belum siap!")

    user_id = user_info.get("sub")
    query = "DELETE FROM private_streams WHERE id = $1 AND user_id = $2 RETURNING id"
    try:
        deleted_id = await pool.fetchval(query, stream_id, user_id)
        if not deleted_id:
            raise HTTPException(
                status_code=404,
                detail="❌ Camera tidak ditemukan atau bukan milik Anda."
            )
        return {"message": f"🗑️ Private camera {stream_id} berhasil dihapus!"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error delete private stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))
