"""
Vehicle density and road occupancy calculations.
"""

import numpy as np
from typing import List, Tuple
import cv2

from ..logger import logger
from ..tracking.track_manager import Track


class DensityCalculator:
    """
    Calculates traffic density metrics from vehicle positions.
    """
    
    def __init__(self, 
                 road_length: float = 50.0,
                 road_width: float = 7.0) -> None:
        """
        Initialize density calculator.
        
        Args:
            road_length: Length of monitored road section (meters)
            road_width: Width of monitored road section (meters)
        """
        self.road_length = road_length
        self.road_width = road_width
        self.road_area = road_length * road_width  # square meters
    
    def calculate_density(self, 
                          vehicle_count: int, 
                          roi_area_pixels: float,
                          lane_count: int = 1) -> float:
        """
        Calculate vehicle density (vehicles per km per lane).
        
        Args:
            vehicle_count: Number of vehicles
            roi_area_pixels: ROI area in pixels
            lane_count: Number of lanes
            
        Returns:
            Density in vehicles/km/lane
        """
        if self.road_length <= 0:
            return 0.0
        
        # Density per km
        density_per_km = (vehicle_count / self.road_length) * 1000
        
        # Per lane
        density_per_lane = density_per_km / lane_count if lane_count > 0 else density_per_km
        
        return density_per_lane
    
    def calculate_occupancy(self, 
                           tracks: List[Track], 
                           roi_area_pixels: float) -> float:
        """
        Calculate road occupancy percentage.
        
        Args:
            tracks: List of vehicle tracks
            roi_area_pixels: ROI area in pixels
            
        Returns:
            Occupancy percentage (0-100)
        """
        if roi_area_pixels <= 0:
            return 0.0
        
        total_vehicle_area = 0.0
        
        for track in tracks:
            if track.bboxes:
                last_bbox = track.bboxes[-1]
                vehicle_area = (last_bbox[2] - last_bbox[0]) * (last_bbox[3] - last_bbox[1])
                total_vehicle_area += vehicle_area
        
        occupancy = (total_vehicle_area / roi_area_pixels) * 100
        
        return min(occupancy, 100.0)
    
    def calculate_density_heatmap(self,
                                  positions: List[Tuple[float, float]],
                                  frame_shape: Tuple[int, int],
                                  grid_size: int = 20) -> np.ndarray:
        """
        Generate density heatmap grid.
        
        Args:
            positions: List of (x, y) vehicle positions
            frame_shape: (height, width) of frame
            grid_size: Size of grid cells in pixels
            
        Returns:
            2D numpy array with density values
        """
        height, width = frame_shape
        
        grid_height = height // grid_size + 1
        grid_width = width // grid_size + 1
        
        heatmap = np.zeros((grid_height, grid_width), dtype=np.float32)
        
        for x, y in positions:
            grid_x = int(x // grid_size)
            grid_y = int(y // grid_size)
            
            if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                heatmap[grid_y, grid_x] += 1
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    
    def get_level_of_service(self, density: float) -> str:
        """
        Determine Level of Service (LOS) based on density.
        
        Args:
            density: Vehicle density (vehicles/km/lane)
            
        Returns:
            LOS grade (A-F)
        """
        if density <= 7:
            return "A"
        elif density <= 11:
            return "B"
        elif density <= 16:
            return "C"
        elif density <= 22:
            return "D"
        elif density <= 28:
            return "E"
        else:
            return "F"
    
    def update_road_dimensions(self, length: float, width: float) -> None:
        """
        Update road dimensions.
        
        Args:
            length: Road length in meters
            width: Road width in meters
        """
        self.road_length = length
        self.road_width = width
        self.road_area = length * width
        logger.info(f"Road dimensions updated: {length}m x {width}m")