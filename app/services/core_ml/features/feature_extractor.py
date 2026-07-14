"""
Main feature extraction orchestrator combining all traffic features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..logger import logger
from ..config import FeatureExtractionConfig
from ..tracking.track_manager import Track
from ..roi.roi_manager import ROIManager
from .speed_estimator import SpeedEstimator
from .density_calculator import DensityCalculator
from .flow_analyzer import FlowAnalyzer


@dataclass
class TrafficFeatures:
    """
    Container for extracted traffic features per time window.
    """
    timestamp: float
    vehicle_count: int
    vehicle_density: float  # vehicles per km
    road_occupancy: float  # percentage
    average_speed: float  # km/h
    vehicle_flow: float  # vehicles per hour
    queue_length: float  # meters
    stopped_vehicles: int
    class_distribution: Dict[str, int] = field(default_factory=dict)
    speed_variance: float = 0.0
    congestion_index: float = 0.0  # 0-1 scale


class FeatureExtractor:
    """
    Extracts comprehensive traffic features from tracking data.
    """
    
    def __init__(self, 
                 config: FeatureExtractionConfig,
                 roi_manager: ROIManager) -> None:
        """
        Initialize feature extractor.
        
        Args:
            config: Feature extraction configuration
            roi_manager: ROI manager instance
        """
        self.config = config
        self.roi_manager = roi_manager
        
        # Sub-extractors
        self.speed_estimator = SpeedEstimator(
            pixel_to_meter_ratio=config.pixel_to_meter_ratio,
            frame_rate=30  # Will be updated during processing
        )
        self.density_calculator = DensityCalculator(
            road_length=config.road_length_meters,
            road_width=config.road_width_meters
        )
        self.flow_analyzer = FlowAnalyzer(
            queue_threshold_speed=config.queue_threshold_speed
        )
        
        self.feature_history: List[TrafficFeatures] = []
    
    def extract_frame_features(self,
                               tracks: List[Track],
                               frame_number: int,
                               timestamp: float) -> TrafficFeatures:
        """
        Extract features for a single frame.
        
        Args:
            tracks: Active tracks in current frame
            frame_number: Current frame number
            timestamp: Current timestamp in seconds
            
        Returns:
            TrafficFeatures object
        """
        # Filter tracks inside ROI
        roi_tracks = self._filter_tracks_by_roi(tracks)
        
        # Extract features
        vehicle_count = len(roi_tracks)
        
        class_distribution = defaultdict(int)
        speeds = []
        stopped_count = 0
        
        for track in roi_tracks:
            class_distribution[track.class_name] += 1
            
            if track.velocities:
                speed = track.velocities[-1]
                speeds.append(speed)
                
                if speed < self.config.queue_threshold_speed:
                    stopped_count += 1
        
        # Calculate aggregate features
        average_speed = np.mean(speeds) if speeds else 0.0
        speed_variance = np.var(speeds) if len(speeds) > 1 else 0.0
        
        # Density and occupancy
        roi_area = self.roi_manager.get_roi_area()
        vehicle_density = self.density_calculator.calculate_density(
            vehicle_count, roi_area
        )
        
        road_occupancy = self.density_calculator.calculate_occupancy(
            roi_tracks, roi_area
        )
        
        # Flow analysis
        vehicle_flow = self.flow_analyzer.calculate_flow(
            roi_tracks, time_window=3600  # vehicles per hour
        )
        
        queue_length = self.flow_analyzer.estimate_queue_length(
            stopped_count, average_speed
        )
        
        # Congestion index (0-1)
        congestion_index = self._calculate_congestion_index(
            vehicle_density, average_speed, road_occupancy
        )
        
        features = TrafficFeatures(
            timestamp=timestamp,
            vehicle_count=vehicle_count,
            vehicle_density=vehicle_density,
            road_occupancy=road_occupancy,
            average_speed=average_speed,
            vehicle_flow=vehicle_flow,
            queue_length=queue_length,
            stopped_vehicles=stopped_count,
            class_distribution=dict(class_distribution),
            speed_variance=speed_variance,
            congestion_index=congestion_index
        )
        
        self.feature_history.append(features)
        
        return features
    
    def _filter_tracks_by_roi(self, tracks: List[Track]) -> List[Track]:
        """
        Filter tracks to those within active ROI.
        
        Args:
            tracks: List of all tracks
            
        Returns:
            Filtered list of tracks inside ROI
        """
        if not self.roi_manager.get_active_polygon():
            return tracks
        
        filtered = []
        for track in tracks:
            if track.positions:
                last_position = track.positions[-1]
                if self.roi_manager.is_point_inside(last_position):
                    filtered.append(track)
        
        return filtered
    
    def _calculate_congestion_index(self,
                                    density: float,
                                    speed: float,
                                    occupancy: float) -> float:
        """
        Calculate congestion index on 0-1 scale.
        
        Args:
            density: Vehicle density (vehicles/km)
            speed: Average speed (km/h)
            occupancy: Road occupancy percentage
            
        Returns:
            Congestion index (0=free flow, 1=fully congested)
        """
        # Normalize each factor
        max_density = 200  # vehicles per km
        max_speed = 120  # km/h
        
        density_factor = min(density / max_density, 1.0)
        speed_factor = 1.0 - min(speed / max_speed, 1.0)
        occupancy_factor = occupancy / 100.0
        
        # Weighted combination
        congestion_index = (
            0.4 * density_factor +
            0.3 * speed_factor +
            0.3 * occupancy_factor
        )
        
        return min(max(congestion_index, 0.0), 1.0)
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """
        Convert feature history to DataFrame.
        
        Returns:
            DataFrame with all features
        """
        if not self.feature_history:
            return pd.DataFrame()
        
        data = []
        for features in self.feature_history:
            row = {
                'timestamp': features.timestamp,
                'vehicle_count': features.vehicle_count,
                'vehicle_density': features.vehicle_density,
                'road_occupancy': features.road_occupancy,
                'average_speed': features.average_speed,
                'vehicle_flow': features.vehicle_flow,
                'queue_length': features.queue_length,
                'stopped_vehicles': features.stopped_vehicles,
                'speed_variance': features.speed_variance,
                'congestion_index': features.congestion_index,
            }
            
            # Add class distribution
            for class_name, count in features.class_distribution.items():
                row[f'count_{class_name}'] = count
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Calculate statistics over all extracted features.
        
        Returns:
            Dictionary with feature statistics
        """
        if not self.feature_history:
            return {}
        
        stats = {
            'total_samples': len(self.feature_history),
            'metrics': {}
        }
        
        numeric_fields = [
            'vehicle_count', 'vehicle_density', 'road_occupancy',
            'average_speed', 'vehicle_flow', 'queue_length',
            'congestion_index'
        ]
        
        for field in numeric_fields:
            values = [getattr(f, field) for f in self.feature_history]
            stats['metrics'][field] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return stats
    
    def clear_history(self) -> None:
        """Clear feature history."""
        self.feature_history.clear()
        logger.debug("Feature history cleared")