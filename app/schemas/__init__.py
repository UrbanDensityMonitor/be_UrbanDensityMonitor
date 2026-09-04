"""
Pydantic schemas untuk request/response validation
"""
from app.schemas.stream import (
    StreamCreate,
    StreamResponse,
    StreamListResponse,
    StreamUpdate
)
from app.schemas.detection import (
    DetectionResult,
    VehicleCounts,
    FrameAnalysis
)
from app.schemas.traffic import (
    TrafficUpdate,
    TrafficHistoryResponse,
    DensityStatus
)
from app.schemas.alert import (
    AlertInfo,
    AlertResponse,
    AlertListResponse
)

__all__ = [
    # Stream schemas
    "StreamCreate",
    "StreamResponse",
    "StreamListResponse",
    "StreamUpdate",
    # Detection schemas
    "DetectionResult",
    "VehicleCounts",
    "FrameAnalysis",
    # Traffic schemas
    "TrafficUpdate",
    "TrafficHistoryResponse",
    "DensityStatus",
    # Alert schemas
    "AlertInfo",
    "AlertResponse",
    "AlertListResponse",
]
