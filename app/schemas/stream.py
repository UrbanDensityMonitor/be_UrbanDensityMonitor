"""
Pydantic schemas untuk Stream-related operations
"""
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl
from uuid import UUID


class StreamCreate(BaseModel):
    """Schema untuk create new stream"""
    location_name: str = Field(..., min_length=3, max_length=100, description="Nama lokasi CCTV")
    stream_url: str = Field(..., description="URL stream CCTV (HLS/RTSP/HTTP)")
    stream_type: str = Field(..., pattern="^(HLS|RTSP|HTTP)$", description="Tipe stream")
    
    class Config:
        json_schema_extra = {
            "example": {
                "location_name": "Jl. Sudirman - Jakarta",
                "stream_url": "https://example.com/stream.m3u8",
                "stream_type": "HLS"
            }
        }


class StreamUpdate(BaseModel):
    """Schema untuk update stream"""
    location_name: Optional[str] = Field(None, min_length=3, max_length=100)
    stream_url: Optional[str] = None
    stream_type: Optional[str] = Field(None, pattern="^(HLS|RTSP|HTTP)$")
    status: Optional[str] = Field(None, pattern="^(active|inactive)$")


class StreamResponse(BaseModel):
    """Schema untuk stream response"""
    id: str
    location_name: str
    stream_url: str
    stream_type: str
    status: str
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class StreamListResponse(BaseModel):
    """Schema untuk list of streams response"""
    data: List[StreamResponse]
    total: int
