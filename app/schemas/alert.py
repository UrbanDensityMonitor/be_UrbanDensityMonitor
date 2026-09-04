"""
Pydantic schemas untuk Alert system
"""
from typing import Optional, List
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class AlertInfo(BaseModel):
    """Alert information untuk WebSocket updates"""
    triggered: bool = Field(..., description="Apakah alert triggered")
    type: str = Field(..., description="Alert type: High Density, Anomaly, cleared")
    message: str = Field(..., description="Alert message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "triggered": True,
                "type": "High Density",
                "message": "🚨 KEPADATAN TINGGI terdeteksi di Jl. Sudirman pukul 14:30 WIB"
            }
        }


class AlertCreate(BaseModel):
    """Schema untuk create alert record"""
    traffic_history_id: UUID  # UUID dari traffic_history.id
    stream_id: str
    alert_type: str
    alert_message: str


class AlertResponse(BaseModel):
    """Schema untuk alert response"""
    id: UUID
    traffic_history_id: UUID
    stream_id: str
    alert_type: str
    alert_message: str
    is_read: bool = False
    created_at: datetime

    class Config:
        from_attributes = True


class AlertListResponse(BaseModel):
    """Schema untuk list of alerts"""
    data: List[AlertResponse]
    total: int
    unread_count: int = 0


class AlertMarkReadRequest(BaseModel):
    """Schema untuk mark alert as read"""
    alert_ids: List[UUID] = Field(..., description="List of alert IDs (UUID) to mark as read")
