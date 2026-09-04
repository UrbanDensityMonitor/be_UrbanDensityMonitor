"""
Pydantic schemas untuk Detection results
"""
from typing import Tuple, List
from pydantic import BaseModel, Field


class DetectionResult(BaseModel):
    """Single detection result dari YOLO"""
    bbox: Tuple[int, int, int, int] = Field(..., description="Bounding box (x1, y1, x2, y2)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    class_id: int = Field(..., description="Class ID dari model")
    class_name: str = Field(..., description="Class name (car, motorcycle, etc.)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "bbox": [100, 150, 200, 250],
                "confidence": 0.87,
                "class_id": 1,
                "class_name": "car"
            }
        }


class VehicleCounts(BaseModel):
    """Jumlah kendaraan per kategori"""
    person: int = Field(default=0, ge=0)
    motorcycle: int = Field(default=0, ge=0)
    car: int = Field(default=0, ge=0)
    bus: int = Field(default=0, ge=0)
    truck: int = Field(default=0, ge=0)
    
    @property
    def total_vehicles(self) -> int:
        """Total kendaraan (tanpa person)"""
        return self.motorcycle + self.car + self.bus + self.truck
    
    @property
    def total_with_person(self) -> int:
        """Total termasuk person"""
        return self.person + self.total_vehicles
    
    class Config:
        json_schema_extra = {
            "example": {
                "person": 5,
                "motorcycle": 12,
                "car": 8,
                "bus": 2,
                "truck": 1
            }
        }


class FrameAnalysis(BaseModel):
    """Complete analysis hasil dari 1 frame"""
    frame_number: int
    timestamp: float
    detections: List[DetectionResult]
    vehicle_counts: VehicleCounts
    average_speed: float = Field(default=0.0, description="Average speed (km/h)")
    road_occupancy: float = Field(default=0.0, ge=0.0, le=1.0, description="Road occupancy ratio")
    congestion_index: float = Field(default=0.0, ge=0.0, description="Congestion index")
