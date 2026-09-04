from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional

from app.services.stream_service import StreamService
from app.core.dependencies import get_stream_service
from app.schemas.stream import StreamCreate, StreamListResponse
from app.auth.jwt_handler import verify_jwt

router = APIRouter(
    prefix="/api/v1/streams",
    tags=["Streams CCTV"]
)


@router.get("", response_model=StreamListResponse)
async def get_streams(
    status: Optional[str] = Query(None, description="Filter by status (active/inactive)"),
    service: StreamService = Depends(get_stream_service),
    user_info: dict = Depends(verify_jwt)
):
    """
    Get all CCTV streams
    
    Query Parameters:
        - status: Optional filter by status
    
    Returns:
        List of streams with total count
    """
    try:
        streams = await service.get_all_streams(status_filter=status)
        return StreamListResponse(
            data=streams,
            total=len(streams)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{stream_id}")
async def get_stream(
    stream_id: str,
    service: StreamService = Depends(get_stream_service),
    user_info: dict = Depends(verify_jwt)
):
    """Get single stream by ID"""
    stream = await service.get_stream_by_id(stream_id)
    
    if not stream:
        raise HTTPException(status_code=404, detail="Stream tidak ditemukan")
    
    return stream


@router.post("")
async def add_stream(
    stream: StreamCreate,
    service: StreamService = Depends(get_stream_service),
    user_info: dict = Depends(verify_jwt)
):
    """Create new CCTV stream"""
    try:
        new_id = await service.create_stream(stream)
        return {
            "message": f"✅ CCTV {stream.location_name} berhasil ditambahkan!",
            "id": new_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{stream_id}")
async def delete_stream(
    stream_id: str,
    service: StreamService = Depends(get_stream_service),
    user_info: dict = Depends(verify_jwt)
):
    """Delete CCTV stream"""
    try:
        deleted = await service.delete_stream(stream_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Stream tidak ditemukan")
        
        return {"message": f"🗑️ CCTV {stream_id} berhasil dihapus!"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
