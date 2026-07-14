"""
ROI management: loading, saving, and point-in-polygon operations.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import cv2

from ..logger import logger


class ROIManager:
    """
    Manages Region of Interest polygons for traffic analysis.
    Supports multiple ROIs, saving/loading, and spatial queries.
    """
    
    def __init__(self, roi_path: Optional[Path] = None) -> None:
        """
        Initialize ROI manager.
        
        Args:
            roi_path: Path to ROI JSON file (optional)
        """
        self.roi_path = roi_path
        self.polygons: Dict[str, Polygon] = {}
        self.active_roi: Optional[str] = None
        
        if roi_path and Path(roi_path).exists():
            self.load_roi(roi_path)
    
    def add_roi(self, 
                name: str, 
                points: List[Tuple[float, float]], 
                set_active: bool = True) -> None:
        """
        Add a new ROI polygon.
        
        Args:
            name: ROI name/identifier
            points: List of (x, y) coordinates forming polygon
            set_active: Whether to set as active ROI
            
        Raises:
            ValueError: If points don't form valid polygon
        """
        if len(points) < 3:
            raise ValueError(f"ROI requires at least 3 points, got {len(points)}")
        
        # Create Shapely polygon
        polygon = Polygon(points)
        
        if not polygon.is_valid:
            # Attempt to fix invalid polygon
            polygon = polygon.buffer(0)
            if not polygon.is_valid:
                raise ValueError("Cannot create valid polygon from given points")
        
        self.polygons[name] = polygon
        
        if set_active:
            self.active_roi = name
        
        logger.info(f"Added ROI '{name}' with {len(points)} points, area: {polygon.area:.2f}")
    
    def remove_roi(self, name: str) -> None:
        """
        Remove an ROI by name.
        
        Args:
            name: ROI identifier
        """
        if name in self.polygons:
            del self.polygons[name]
            
            if self.active_roi == name:
                self.active_roi = next(iter(self.polygons)) if self.polygons else None
            
            logger.info(f"Removed ROI '{name}'")
    
    def set_active_roi(self, name: str) -> None:
        """
        Set active ROI by name.
        
        Args:
            name: ROI identifier
            
        Raises:
            KeyError: If ROI name not found
        """
        if name not in self.polygons:
            raise KeyError(f"ROI '{name}' not found")
        
        self.active_roi = name
        logger.info(f"Active ROI set to '{name}'")
    
    def get_active_polygon(self) -> Optional[Polygon]:
        """
        Get currently active ROI polygon.
        
        Returns:
            Active polygon or None
        """
        if self.active_roi:
            return self.polygons[self.active_roi]
        return None
    
    def is_point_inside(self, 
                        point: Tuple[float, float], 
                        roi_name: Optional[str] = None) -> bool:
        """
        Check if point is inside an ROI.
        
        Args:
            point: (x, y) coordinates
            roi_name: ROI to check (default: active ROI)
            
        Returns:
            True if point is inside polygon
        """
        polygon = self.get_polygon(roi_name)
        if polygon is None:
            return True  # No ROI means everything is valid
        
        return polygon.contains(Point(point))
    
    def is_bbox_inside(self, 
                       bbox: Tuple[float, float, float, float],
                       threshold: float = 0.5,
                       roi_name: Optional[str] = None) -> bool:
        """
        Check if bounding box is partially inside ROI.
        
        Args:
            bbox: (x1, y1, x2, y2) coordinates
            threshold: Minimum overlap ratio to consider inside
            roi_name: ROI to check (default: active ROI)
            
        Returns:
            True if bbox has sufficient overlap with ROI
        """
        polygon = self.get_polygon(roi_name)
        if polygon is None:
            return True
        
        # Create box polygon from bbox
        bbox_poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
        
        # Calculate overlap
        if not polygon.intersects(bbox_poly):
            return False
        
        intersection = polygon.intersection(bbox_poly)
        overlap_ratio = intersection.area / bbox_poly.area
        
        return overlap_ratio >= threshold
    
    def filter_points(self, 
                      points: List[Tuple[float, float]], 
                      roi_name: Optional[str] = None) -> List[Tuple[float, float]]:
        """
        Filter points to those inside ROI.
        
        Args:
            points: List of (x, y) coordinates
            roi_name: ROI to check (default: active ROI)
            
        Returns:
            List of points inside ROI
        """
        return [p for p in points if self.is_point_inside(p, roi_name)]
    
    def get_polygon(self, roi_name: Optional[str] = None) -> Optional[Polygon]:
        """
        Get polygon by name.
        
        Args:
            roi_name: ROI name (default: active ROI)
            
        Returns:
            Shapely polygon or None
        """
        if roi_name is None:
            roi_name = self.active_roi
        
        return self.polygons.get(roi_name)
    
    def get_polygon_points(self, roi_name: Optional[str] = None) -> List[List[float]]:
        """
        Get polygon vertices as list of coordinates.
        
        Args:
            roi_name: ROI name (default: active ROI)
            
        Returns:
            List of [x, y] coordinates
        """
        polygon = self.get_polygon(roi_name)
        if polygon is None:
            return []
        
        # Get exterior coordinates
        coords = list(polygon.exterior.coords)
        return [[float(x), float(y)] for x, y in coords]
    
    def get_roi_area(self, roi_name: Optional[str] = None) -> float:
        """
        Calculate ROI area in pixels.
        
        Args:
            roi_name: ROI name (default: active ROI)
            
        Returns:
            Area in square pixels
        """
        polygon = self.get_polygon(roi_name)
        if polygon is None:
            return 0.0
        
        return polygon.area
    
    def draw_roi_on_frame(self, 
                          frame: np.ndarray, 
                          roi_name: Optional[str] = None,
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 2) -> np.ndarray:
        """
        Draw ROI polygon on frame.
        
        Args:
            frame: Input frame
            roi_name: ROI to draw (default: all ROIs)
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Frame with drawn ROI
        """
        frame_copy = frame.copy()
        
        polygons_to_draw = {}
        if roi_name:
            if roi_name in self.polygons:
                polygons_to_draw[roi_name] = self.polygons[roi_name]
        else:
            polygons_to_draw = self.polygons
        
        for name, polygon in polygons_to_draw.items():
            # Get points as integer array
            points = np.array(polygon.exterior.coords, dtype=np.int32)
            
            # Draw polygon
            cv2.polylines(frame_copy, [points], True, color, thickness)
            
            # Draw ROI name
            centroid = polygon.centroid
            cv2.putText(
                frame_copy, name,
                (int(centroid.x), int(centroid.y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        return frame_copy
    
    def save_roi(self, filepath: Optional[Path] = None) -> None:
        """
        Save all ROIs to JSON file.
        
        Args:
            filepath: Save path (default: self.roi_path)
        """
        if filepath is None:
            filepath = self.roi_path
        
        if filepath is None:
            raise ValueError("No filepath specified for saving ROI")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        roi_data = {
            'active_roi': self.active_roi,
            'polygons': {}
        }
        
        for name, polygon in self.polygons.items():
            coords = [[float(x), float(y)] for x, y in polygon.exterior.coords]
            roi_data['polygons'][name] = coords
        
        with open(filepath, 'w') as f:
            json.dump(roi_data, f, indent=2)
        
        logger.info(f"Saved {len(self.polygons)} ROIs to {filepath}")
    
    def load_roi(self, filepath: Path) -> None:
        """
        Load ROIs from JSON file.
        
        Args:
            filepath: Path to ROI JSON file
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"ROI file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            roi_data = json.load(f)
        
        self.polygons.clear()
        
        for name, coords in roi_data.get('polygons', {}).items():
            points = [(float(x), float(y)) for x, y in coords]
            self.add_roi(name, points, set_active=False)
        
        self.active_roi = roi_data.get('active_roi')
        
        logger.info(f"Loaded {len(self.polygons)} ROIs from {filepath}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about all ROIs.
        
        Returns:
            Dictionary with ROI statistics
        """
        stats = {
            'total_rois': len(self.polygons),
            'active_roi': self.active_roi,
            'rois': {}
        }
        
        for name, polygon in self.polygons.items():
            stats['rois'][name] = {
                'area_pixels': polygon.area,
                'perimeter': polygon.length,
                'centroid': (polygon.centroid.x, polygon.centroid.y),
                'num_vertices': len(polygon.exterior.coords) - 1
            }
        
        return stats