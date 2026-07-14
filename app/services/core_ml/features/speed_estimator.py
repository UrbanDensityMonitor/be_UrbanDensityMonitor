"""
Vehicle speed estimation from tracking data.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from ..logger import logger
from ..tracking.track_manager import Track


@dataclass
class SpeedMeasurement:
    """Container for speed measurement."""
    track_id: int
    speed_px_per_frame: float
    speed_mps: float  # meters per second
    speed_kmh: float  # kilometers per hour
    confidence: float


class SpeedEstimator:
    """
    Estimates vehicle speed from tracking trajectories.
    Supports pixel-to-real-world coordinate conversion.
    """
    
    def __init__(self, 
                 pixel_to_meter_ratio: float = 0.05,
                 frame_rate: float = 30.0,
                 smoothing_window: int = 5) -> None:
        """
        Initialize speed estimator.
        
        Args:
            pixel_to_meter_ratio: Conversion factor (meters per pixel)
            frame_rate: Video frame rate
            smoothing_window: Window size for speed smoothing
        """
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        self.frame_rate = frame_rate
        self.smoothing_window = smoothing_window
    
    def estimate_speed(self, track: Track) -> Optional[SpeedMeasurement]:
        """
        Estimate current speed for a track.
        
        Args:
            track: Vehicle track
            
        Returns:
            SpeedMeasurement or None if insufficient data
        """
        if track.length < 2:
            return None
        
        # Get recent positions
        recent_positions = track.positions[-self.smoothing_window:]
        
        if len(recent_positions) < 2:
            return None
        
        # Calculate displacement in pixels
        total_displacement = 0.0
        for i in range(1, len(recent_positions)):
            p1 = recent_positions[i-1]
            p2 = recent_positions[i]
            displacement = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            total_displacement += displacement
        
        # Average displacement per frame
        avg_displacement = total_displacement / (len(recent_positions) - 1)
        
        # Convert to real-world units
        speed_mps = avg_displacement * self.pixel_to_meter_ratio * self.frame_rate
        speed_kmh = speed_mps * 3.6
        
        # Confidence based on track quality
        confidence = min(track.average_confidence * track.length / 30, 1.0)
        
        return SpeedMeasurement(
            track_id=track.track_id,
            speed_px_per_frame=avg_displacement,
            speed_mps=speed_mps,
            speed_kmh=speed_kmh,
            confidence=confidence
        )
    
    def estimate_trajectory_speed(self, track: Track) -> List[float]:
        """
        Estimate speed for entire trajectory.
        
        Args:
            track: Vehicle track
            
        Returns:
            List of speed values (km/h) for each position
        """
        if track.length < 2:
            return []
        
        speeds = []
        
        for i in range(1, len(track.positions)):
            p1 = track.positions[i-1]
            p2 = track.positions[i]
            
            displacement = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            speed_kmh = displacement * self.pixel_to_meter_ratio * self.frame_rate * 3.6
            speeds.append(speed_kmh)
        
        # Add first speed (same as second)
        if speeds:
            speeds.insert(0, speeds[0])
        
        return speeds
    
    def get_average_speed(self, tracks: List[Track]) -> float:
        """
        Calculate average speed across multiple tracks.
        
        Args:
            tracks: List of vehicle tracks
            
        Returns:
            Average speed in km/h
        """
        speeds = []
        
        for track in tracks:
            speed_measurement = self.estimate_speed(track)
            if speed_measurement and speed_measurement.confidence > 0.5:
                speeds.append(speed_measurement.speed_kmh)
        
        return float(np.mean(speeds)) if speeds else 0.0
    
    def classify_speed(self, speed_kmh: float) -> str:
        """
        Classify speed into category.
        
        Args:
            speed_kmh: Speed in km/h
            
        Returns:
            Speed category string
        """
        if speed_kmh < 5:
            return "stopped"
        elif speed_kmh < 20:
            return "slow"
        elif speed_kmh < 40:
            return "moderate"
        elif speed_kmh < 60:
            return "normal"
        else:
            return "fast"
    
    def set_calibration(self, pixel_to_meter_ratio: float) -> None:
        """
        Update calibration ratio.
        
        Args:
            pixel_to_meter_ratio: New conversion factor
        """
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        logger.info(f"Speed calibration updated: {pixel_to_meter_ratio} m/px")