from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

from app.db.asyncpg_client import get_db
from app.auth.jwt_handler import verify_jwt
from app.db.repositories.stream_repository import StreamRepository
from asyncpg import Pool

router = APIRouter(
    prefix="/streams",
    tags=["Streams CCTV"]
)

class StreamCreate(BaseModel):
    location_name: str
    stream_url: str
    stream_type: str

class StreamResponse(BaseModel):
    id: str
    location_name: str
    stream_url: str
    stream_type: str
    status: str

@router.get("/", response_model=dict)
async def get_streams(
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = StreamRepository(db)
    try:
        streams_data = await repo.get_all_streams()
        return {"data": streams_data, "total": len(streams_data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/")
async def add_stream(
    stream: StreamCreate, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = StreamRepository(db)
    try:
        new_id = await repo.add_stream(stream.location_name, stream.stream_url, stream.stream_type)
        return {"message": f"✅ CCTV {stream.location_name} berhasil ditambahkan!", "id": new_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{stream_id}")
async def delete_stream(
    stream_id: str, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = StreamRepository(db)
    try:
        await repo.delete_stream(stream_id)
        return {"message": f"🗑️ CCTV {stream_id} berhasil dihapus!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
