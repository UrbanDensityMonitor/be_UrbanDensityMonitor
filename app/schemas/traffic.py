"""
Pydantic schemas untuk Traffic analysis
"""
from typing import Optional, Dict, List
from datetime import datetime
from uuid import UUID
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from app.schemas.detection import VehicleCounts
from app.schemas.alert import AlertInfo


class DensityStatus(str, Enum):
    """Enum untuk density status"""
    LOW = "Low Density"
    MEDIUM = "Medium Density"
    HIGH = "High Density"
    ANOMALY = "Anomaly"


class TrafficUpdate(BaseModel):
    """
    WebSocket payload untuk real-time traffic updates
    Dikirim setiap frame yang diproses
    """
    type: str = Field(default="frame_update", description="Message type")
    stream_id: str
    counts: VehicleCounts
    person_vehicle_ratio: float = Field(default=0.0)
    density_status: DensityStatus
    average_speed: float = Field(default=0.0, description="km/h")
    road_occupancy: float = Field(default=0.0, description="0.0 - 1.0")
    congestion_index: float = Field(default=0.0)
    frame_base64: Optional[str] = Field(None, description="Base64 encoded frame")
    frame: Optional[str] = Field(None, description="Data URL untuk frame")
    alert: Optional[AlertInfo] = Field(None, description="Alert info jika triggered")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "frame_update",
                "stream_id": "123e4567-e89b-12d3-a456-426614174000",
                "counts": {
                    "person": 5,
                    "motorcycle": 12,
                    "car": 8,
                    "bus": 2,
                    "truck": 1
                },
                "person_vehicle_ratio": 0.22,
                "density_status": "Medium Density",
                "average_speed": 35.5,
                "road_occupancy": 0.45,
                "congestion_index": 0.67,
                "alert": None
            }
        }


class TrafficHistoryCreate(BaseModel):
    """
    Schema untuk create traffic history record.

    Semua kolom float dikonstrain ke batas kolom DB NUMERIC(5,2):
    max absolute value = 999.99.

    Validator meng-clamp nilai (bukan raise error) agar stream tetap
    berjalan meski sesekali ada outlier dari tracker/feature extractor.
    """
    stream_id: str
    person_count: int
    motorcycle_count: int
    car_count: int
    bus_count: int
    truck_count: int
    total_vehicle_count: int
    person_vehicle_ratio: float   # 0.0 – 999.99
    density_status: str
    average_speed: float          # km/h, 0.0 – 999.99
    road_occupancy: float         # idealnya 0.0 – 1.0, capped di 999.99
    congestion_index: float       # 0.0 – 999.99

    @field_validator("average_speed", "road_occupancy", "congestion_index", "person_vehicle_ratio", mode="before")
    @classmethod
    def clamp_numeric_db_range(cls, v: float) -> float:
        """
        Clamp nilai ke batas kolom NUMERIC(5,2) di Supabase (max 999.99).
        Outlier bisa muncul dari feature extractor saat kalkulasi speed/occupancy
        di frame pertama atau saat tracking object keluar area ROI.
        Menggunakan clamp (bukan raise) agar stream tetap berjalan.
        """
        _MIN, _MAX = 0.0, 999.99
        try:
            value = float(v)
        except (TypeError, ValueError):
            return _MIN
        return max(_MIN, min(_MAX, round(value, 2)))


class TrafficHistoryResponse(BaseModel):
    """Schema untuk traffic history response"""
    id: UUID
    stream_id: str
    person_count: int
    motorcycle_count: int
    car_count: int
    bus_count: int
    truck_count: int
    total_vehicle_count: int
    person_vehicle_ratio: float
    density_status: str
    average_speed: float
    road_occupancy: float
    congestion_index: float
    recorded_at: datetime

    class Config:
        from_attributes = True


class TrafficHistoryListResponse(BaseModel):
    """Schema untuk list of traffic history"""
    data: List[TrafficHistoryResponse]
    total: int
