from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from asyncpg import Pool
import logging

from app.db.asyncpg_client import get_db
from app.auth.jwt_handler import verify_jwt
from app.db.repositories.private_stream_repository import PrivateStreamRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/private-streams",
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
async def get_private_streams(
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    """List semua private cameras milik authenticated user."""
    repo = PrivateStreamRepository(db)
    user_id = user_info.get("sub")
    
    try:
        streams_data = await repo.get_all_streams(user_id)
        return {"data": streams_data, "total": len(streams_data)}
    except Exception as e:
        logger.error(f"❌ Error get private streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/")
async def create_private_stream(
    stream: PrivateStreamCreate, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    """Tambah private camera baru. user_id diambil dari JWT."""
    repo = PrivateStreamRepository(db)
    user_id = user_info.get("sub")
    
    try:
        new_id = await repo.create_stream(
            user_id,
            stream.camera_name,
            stream.location_name,
            stream.stream_url,
            stream.protocol,
            stream.stream_type
        )
        return {
            "message": f"✅ Private camera '{stream.camera_name}' berhasil ditambahkan!",
            "id": new_id
        }
    except Exception as e:
        error_msg = str(e)
        if "maximum" in error_msg.lower() or "limit" in error_msg.lower() or "5" in error_msg:
            raise HTTPException(
                status_code=400,
                detail="⛔ Batas maksimal 5 private camera tercapai. Hapus salah satu camera sebelum menambah yang baru."
            )
        logger.error(f"❌ Error create private stream: {e}")
        raise HTTPException(status_code=500, detail=error_msg)


@router.put("/{stream_id}")
async def update_private_stream(
    stream_id: str, 
    stream: PrivateStreamUpdate, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    """Update private camera. Hanya owner yang boleh mengubah."""
    repo = PrivateStreamRepository(db)
    user_id = user_info.get("sub")
    
    try:
        updated_id = await repo.update_stream(
            stream_id,
            user_id,
            stream.camera_name,
            stream.location_name,
            stream.stream_url,
            stream.protocol,
            stream.stream_type
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
async def delete_private_stream(
    stream_id: str, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    """Hapus private camera. Hanya owner yang boleh menghapus."""
    repo = PrivateStreamRepository(db)
    user_id = user_info.get("sub")
    
    try:
        deleted_id = await repo.delete_stream(stream_id, user_id)
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
