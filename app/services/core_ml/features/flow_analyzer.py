"""
Traffic flow and queue analysis.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass

from ..logger import logger
from ..tracking.track_manager import Track


@dataclass
class FlowMetrics:
    """Container for flow analysis metrics."""
    flow_rate: float  # vehicles per hour
    headway: float  # average time between vehicles (seconds)
    queue_length: float  # meters
    stopped_vehicles: int
    throughput: float  # vehicles per lane per hour


class FlowAnalyzer:
    """
    Analyzes traffic flow patterns and queue formation.
    """
    
    def __init__(self, 
                 queue_threshold_speed: float = 5.0,
                 jam_density: float = 150.0) -> None:
        """
        Initialize flow analyzer.
        
        Args:
            queue_threshold_speed: Speed below which vehicle is considered queued (km/h)
            jam_density: Density at which traffic is considered jammed (vehicles/km)
        """
        self.queue_threshold_speed = queue_threshold_speed
        self.jam_density = jam_density
        
        self.vehicle_count_history: deque = deque(maxlen=3600)  # Last hour of counts
        self.arrival_times: deque = deque(maxlen=100)  # Recent vehicle arrival times
    
    def calculate_flow(self, 
                       tracks: List[Track], 
                       time_window: float = 3600) -> float:
        """
        Calculate traffic flow rate (vehicles per hour).
        
        Args:
            tracks: Current vehicle tracks
            time_window: Time window in seconds for flow calculation
            
        Returns:
            Flow rate in vehicles/hour
        """
        # Store current count
        current_time = max([t.last_frame for t in tracks]) if tracks else 0
        
        # Simple flow estimation based on current count and time window
        active_vehicles = len([t for t in tracks if t.is_active])
        
        # Project to hourly rate
        if time_window > 0:
            flow_rate = active_vehicles * (3600 / time_window)
        else:
            flow_rate = 0.0
        
        return flow_rate
    
    def calculate_headway(self, 
                          tracks: List[Track],
                          frame_rate: float = 30.0) -> float:
        """
        Calculate average time headway between consecutive vehicles.
        
        Args:
            tracks: Vehicle tracks sorted by position
            frame_rate: Video frame rate
            
        Returns:
            Average headway in seconds
        """
        if len(tracks) < 2:
            return 0.0
        
        # Sort tracks by position (y-coordinate)
        sorted_tracks = sorted(tracks, key=lambda t: t.positions[-1][1] if t.positions else 0)
        
        headways = []
        for i in range(1, len(sorted_tracks)):
            pos1 = sorted_tracks[i-1].positions[-1] if sorted_tracks[i-1].positions else (0, 0)
            pos2 = sorted_tracks[i].positions[-1] if sorted_tracks[i].positions else (0, 0)
            
            # Distance between vehicles in pixels
            distance = np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            # Estimate time headway based on distance and typical speed
            avg_speed = (np.mean(sorted_tracks[i-1].velocities) if sorted_tracks[i-1].velocities else 30)
            avg_speed += (np.mean(sorted_tracks[i].velocities) if sorted_tracks[i].velocities else 30)
            avg_speed /= 2
            
            if avg_speed > 0:
                headway = distance / avg_speed * frame_rate
                headways.append(headway)
        
        return float(np.mean(headways)) if headways else 0.0
    
    def estimate_queue_length(self, 
                             stopped_vehicles: int,
                             average_speed: float,
                             vehicle_length: float = 5.0,
                             gap_between_vehicles: float = 2.0) -> float:
        """
        Estimate queue length in meters.
        
        Args:
            stopped_vehicles: Number of stopped vehicles
            average_speed: Average speed of all vehicles
            vehicle_length: Average vehicle length (meters)
            gap_between_vehicles: Average gap in queue (meters)
            
        Returns:
            Estimated queue length in meters
        """
        # Only count queue if speed is low
        if average_speed > self.queue_threshold_speed * 2:
            return 0.0
        
        # Estimate based on stopped vehicles
        vehicle_spacing = vehicle_length + gap_between_vehicles
        queue_length = stopped_vehicles * vehicle_spacing
        
        return queue_length
    
    def detect_congestion_wave(self, 
                              speed_history: List[float],
                              threshold: float = 0.3) -> bool:
        """
        Detect congestion wave propagation.
        
        Args:
            speed_history: Historical speed values
            threshold: Speed reduction threshold for wave detection
            
        Returns:
            True if congestion wave detected
        """
        if len(speed_history) < 10:
            return False
        
        # Check for rapid speed reduction
        recent_speeds = speed_history[-10:]
        older_speeds = speed_history[-20:-10]
        
        if not older_speeds:
            return False
        
        recent_avg = np.mean(recent_speeds)
        older_avg = np.mean(older_speeds)
        
        if older_avg > 0:
            reduction = (older_avg - recent_avg) / older_avg
            return reduction > threshold
        
        return False
    
    def get_flow_metrics(self, 
                        tracks: List[Track],
                        frame_rate: float = 30.0) -> FlowMetrics:
        """
        Calculate comprehensive flow metrics.
        
        Args:
            tracks: List of vehicle tracks
            frame_rate: Video frame rate
            
        Returns:
            FlowMetrics object
        """
        # Flow rate
        flow_rate = self.calculate_flow(tracks)
        
        # Headway
        headway = self.calculate_headway(tracks, frame_rate)
        
        # Queue analysis
        stopped = len([t for t in tracks 
                      if t.velocities and t.velocities[-1] < self.queue_threshold_speed])
        
        speeds = [t.velocities[-1] for t in tracks if t.velocities]
        avg_speed = np.mean(speeds) if speeds else 0.0
        
        queue_length = self.estimate_queue_length(stopped, avg_speed)
        
        # Throughput
        throughput = flow_rate  # Simplified
        
        return FlowMetrics(
            flow_rate=flow_rate,
            headway=headway,
            queue_length=queue_length,
            stopped_vehicles=stopped,
            throughput=throughput
        )