"""
Detection data container (runtime).

DetectionService (app/services/detection_service.py) menghasilkan List[Detection].
VehicleDetector versi riset ada di research/vehicle_detector.py.
"""
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Detection:
    """
    Container for a single detection result.
    """
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]
