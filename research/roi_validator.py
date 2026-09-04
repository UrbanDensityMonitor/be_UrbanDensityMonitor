"""
ROI validation utilities for ensuring correct polygon configuration.
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from shapely.geometry import Polygon, Point
from shapely.validation import explain_validity

from ..logger import logger


class ROIValidator:
    """
    Validates ROI polygons and provides quality metrics.
    """
    
    def __init__(self, min_area: float = 1000.0, max_vertices: int = 20) -> None:
        """
        Initialize validator.
        
        Args:
            min_area: Minimum polygon area in pixels
            max_vertices: Maximum allowed vertices for performance
        """
        self.min_area = min_area
        self.max_vertices = max_vertices
    
    def validate_polygon(self, 
                         points: List[Tuple[float, float]]) -> Tuple[bool, str]:
        """
        Validate a single polygon.
        
        Args:
            points: List of (x, y) coordinates
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Check minimum vertices
        if len(points) < 3:
            return False, "Polygon must have at least 3 vertices"
        
        # Check maximum vertices
        if len(points) > self.max_vertices:
            return False, f"Polygon has {len(points)} vertices (max: {self.max_vertices})"
        
        # Create Shapely polygon
        try:
            polygon = Polygon(points)
        except Exception as e:
            return False, f"Failed to create polygon: {str(e)}"
        
        # Check if empty
        if polygon.is_empty:
            return False, "Polygon is empty"
        
        # Check validity
        if not polygon.is_valid:
            reason = explain_validity(polygon)
            return False, f"Invalid polygon: {reason}"
        
        # Check minimum area
        if polygon.area < self.min_area:
            return False, f"Polygon area too small: {polygon.area:.2f} (min: {self.min_area})"
        
        # Check for duplicate vertices
        unique_points = set(points)
        if len(unique_points) < len(points):
            logger.warning("Polygon contains duplicate vertices")
        
        # Check if polygon is too narrow
        bounds = polygon.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
        if aspect_ratio > 100:
            return False, f"Polygon too narrow (aspect ratio: {aspect_ratio:.2f})"
        
        return True, "Valid polygon"
    
    def validate_all_rois(self, 
                          rois: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Any]:
        """
        Validate all ROIs and return report.
        
        Args:
            rois: Dictionary of ROI names to point lists
            
        Returns:
            Validation report dictionary
        """
        report = {
            'total_rois': len(rois),
            'valid_rois': 0,
            'invalid_rois': 0,
            'results': {},
            'warnings': []
        }
        
        for name, points in rois.items():
            is_valid, message = self.validate_polygon(points)
            
            report['results'][name] = {
                'valid': is_valid,
                'message': message,
                'num_points': len(points)
            }
            
            if is_valid:
                report['valid_rois'] += 1
            else:
                report['invalid_rois'] += 1
                report['warnings'].append(f"ROI '{name}': {message}")
        
        return report
    
    def compute_overlap(self, 
                       roi1: List[Tuple[float, float]], 
                       roi2: List[Tuple[float, float]]) -> float:
        """
        Compute overlap ratio between two ROIs.
        
        Args:
            roi1: First polygon points
            roi2: Second polygon points
            
        Returns:
            Overlap ratio (0 to 1)
        """
        poly1 = Polygon(roi1)
        poly2 = Polygon(roi2)
        
        if not poly1.intersects(poly2):
            return 0.0
        
        intersection = poly1.intersection(poly2)
        union = poly1.union(poly2)
        
        if union.area == 0:
            return 0.0
        
        return intersection.area / union.area
    
    def check_roi_overlaps(self, 
                          rois: Dict[str, List[Tuple[float, float]]]) -> List[str]:
        """
        Check for overlapping ROIs.
        
        Args:
            rois: Dictionary of ROIs
            
        Returns:
            List of warning messages
        """
        warnings = []
        names = list(rois.keys())
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                overlap = self.compute_overlap(rois[names[i]], rois[names[j]])
                
                if overlap > 0.5:
                    warnings.append(
                        f"High overlap ({overlap:.2%}) between '{names[i]}' and '{names[j]}'"
                    )
        
        return warnings
    
    def suggest_roi_bounds(self, 
                          frame_shape: Tuple[int, int],
                          road_region: str = "center") -> List[Tuple[int, int]]:
        """
        Suggest default ROI based on frame dimensions.
        
        Args:
            frame_shape: (height, width) of frame
            road_region: Where to place ROI ("center", "bottom", "full")
            
        Returns:
            Suggested polygon points
        """
        h, w = frame_shape
        
        margin = 0.1  # 10% margin
        
        if road_region == "center":
            # Center rectangle
            points = [
                (int(w * margin), int(h * 0.3)),
                (int(w * (1 - margin)), int(h * 0.3)),
                (int(w * (1 - margin)), int(h * 0.7)),
                (int(w * margin), int(h * 0.7))
            ]
        elif road_region == "bottom":
            # Bottom trapezoid
            points = [
                (int(w * 0.2), int(h * 0.5)),
                (int(w * 0.8), int(h * 0.5)),
                (int(w * 0.9), int(h * 0.9)),
                (int(w * 0.1), int(h * 0.9))
            ]
        else:  # full
            points = [
                (int(w * margin), int(h * margin)),
                (int(w * (1 - margin)), int(h * margin)),
                (int(w * (1 - margin)), int(h * (1 - margin))),
                (int(w * margin), int(h * (1 - margin)))
            ]
        
        return points