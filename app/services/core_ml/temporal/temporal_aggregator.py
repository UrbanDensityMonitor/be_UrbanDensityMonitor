"""
Temporal aggregation of traffic features across multiple time windows.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from ..logger import logger
from ..config import TemporalAggregationConfig
from ..features.feature_extractor import TrafficFeatures
from .window_manager import WindowManager, TimeWindow


@dataclass
class AggregatedFeatures:
    """
    Container for temporally aggregated traffic features.
    """
    window_start: float
    window_end: float
    window_size: float
    
    # Aggregated metrics
    avg_vehicle_count: float
    max_vehicle_count: int
    avg_density: float
    max_density: float
    avg_occupancy: float
    avg_speed: float
    min_speed: float
    avg_flow: float
    max_queue_length: float
    avg_congestion_index: float
    
    # Variability metrics
    speed_variance: float
    count_variance: float
    density_variance: float
    
    # Class distribution (averaged)
    avg_class_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Derived states
    dominant_state: str = "unknown"
    stability_score: float = 0.0  # How stable traffic is (0-1)
    
    # Metadata
    sample_count: int = 0
    complete: bool = False


class TemporalAggregator:
    """
    Aggregates traffic features across multiple temporal windows.
    Supports statistical aggregation and state characterization.
    """
    
    def __init__(self, config: TemporalAggregationConfig) -> None:
        """
        Initialize temporal aggregator.
        
        Args:
            config: Temporal aggregation configuration
        """
        self.config = config
        self.window_manager = WindowManager(
            window_sizes=[float(w) for w in config.windows_seconds],
            overlap=0.5,  # 50% overlap between windows
            min_samples_per_window=3
        )
        
        self.aggregated_features: Dict[float, List[AggregatedFeatures]] = defaultdict(list)
        
        logger.info(f"Temporal aggregator initialized with windows: {config.windows_seconds}")
    
    def add_frame_features(self, 
                          features: TrafficFeatures,
                          timestamp: float) -> None:
        """
        Add frame-level features for aggregation.
        
        Args:
            features: TrafficFeatures from single frame
            timestamp: Frame timestamp in seconds
        """
        self.window_manager.add_sample(timestamp, features)
    
    def aggregate_window(self, window: TimeWindow) -> AggregatedFeatures:
        """
        Aggregate features within a single time window.
        
        Args:
            window: TimeWindow with feature samples
            
        Returns:
            AggregatedFeatures object
        """
        if not window.features:
            return self._create_empty_aggregation(window)
        
        # Extract metrics
        counts = [f.vehicle_count for f in window.features]
        densities = [f.vehicle_density for f in window.features]
        occupancies = [f.road_occupancy for f in window.features]
        speeds = [f.average_speed for f in window.features]
        flows = [f.vehicle_flow for f in window.features]
        queues = [f.queue_length for f in window.features]
        congestion = [f.congestion_index for f in window.features]
        
        # Aggregate
        avg_speed = np.mean(speeds) if speeds else 0.0
        min_speed = np.min(speeds) if speeds else 0.0
        
        # Class distribution
        class_dist = defaultdict(float)
        for f in window.features:
            for class_name, count in f.class_distribution.items():
                class_dist[class_name] += count
        for class_name in class_dist:
            class_dist[class_name] /= len(window.features)
        
        # Stability score (inverse of variability)
        speed_cv = np.std(speeds) / (avg_speed + 1e-6) if speeds else 0.0
        count_cv = np.std(counts) / (np.mean(counts) + 1e-6) if counts else 0.0
        stability = 1.0 / (1.0 + speed_cv + count_cv)
        
        # Dominant state
        dominant_state = self._classify_traffic_state(
            np.mean(densities) if densities else 0,
            avg_speed,
            np.mean(occupancies) if occupancies else 0
        )
        
        aggregated = AggregatedFeatures(
            window_start=window.start_time,
            window_end=window.end_time,
            window_size=window.duration,
            
            avg_vehicle_count=float(np.mean(counts)) if counts else 0.0,
            max_vehicle_count=int(np.max(counts)) if counts else 0,
            avg_density=float(np.mean(densities)) if densities else 0.0,
            max_density=float(np.max(densities)) if densities else 0.0,
            avg_occupancy=float(np.mean(occupancies)) if occupancies else 0.0,
            avg_speed=avg_speed,
            min_speed=min_speed,
            avg_flow=float(np.mean(flows)) if flows else 0.0,
            max_queue_length=float(np.max(queues)) if queues else 0.0,
            avg_congestion_index=float(np.mean(congestion)) if congestion else 0.0,
            
            speed_variance=float(np.var(speeds)) if len(speeds) > 1 else 0.0,
            count_variance=float(np.var(counts)) if len(counts) > 1 else 0.0,
            density_variance=float(np.var(densities)) if len(densities) > 1 else 0.0,
            
            avg_class_distribution=dict(class_dist),
            dominant_state=dominant_state,
            stability_score=float(stability),
            
            sample_count=window.sample_count,
            complete=window.is_complete
        )
        
        return aggregated
    
    def _classify_traffic_state(self, 
                                density: float, 
                                speed: float, 
                                occupancy: float) -> str:
        """
        Classify traffic state based on aggregated metrics.
        
        Args:
            density: Vehicle density
            speed: Average speed
            occupancy: Road occupancy
            
        Returns:
            Traffic state category
        """
        if speed < 5:
            return "congested"
        elif speed < 20:
            if density > 30:
                return "heavy"
            else:
                return "slow"
        elif speed < 40:
            return "moderate"
        elif density < 10:
            return "free_flow"
        else:
            return "normal"
    
    def _create_empty_aggregation(self, window: TimeWindow) -> AggregatedFeatures:
        """Create empty aggregation for windows with no data."""
        return AggregatedFeatures(
            window_start=window.start_time,
            window_end=window.end_time,
            window_size=window.duration,
            avg_vehicle_count=0.0,
            max_vehicle_count=0,
            avg_density=0.0,
            max_density=0.0,
            avg_occupancy=0.0,
            avg_speed=0.0,
            min_speed=0.0,
            avg_flow=0.0,
            max_queue_length=0.0,
            avg_congestion_index=0.0,
            speed_variance=0.0,
            count_variance=0.0,
            density_variance=0.0,
            sample_count=0,
            complete=False
        )
    
    def process_all_windows(self) -> Dict[float, List[AggregatedFeatures]]:
        """
        Process all completed windows for all window sizes.
        
        Returns:
            Dictionary mapping window sizes to aggregated features
        """
        completed = self.window_manager.get_completed_windows()
        
        for window_size, windows in completed.items():
            for window in windows:
                aggregated = self.aggregate_window(window)
                self.aggregated_features[window_size].append(aggregated)
        
        logger.info(f"Processed windows: {sum(len(v) for v in self.aggregated_features.values())} total")
        
        return self.aggregated_features
    
    def get_aggregated_dataframe(self, 
                                 window_size: Optional[float] = None) -> pd.DataFrame:
        """
        Convert aggregated features to DataFrame.
        
        Args:
            window_size: Specific window size (None for default)
            
        Returns:
            DataFrame with aggregated features
        """
        if window_size is None:
            window_size = self.config.default_window
        
        if window_size not in self.aggregated_features:
            return pd.DataFrame()
        
        data = []
        for agg in self.aggregated_features[window_size]:
            row = {
                'window_start': agg.window_start,
                'window_end': agg.window_end,
                'window_size': agg.window_size,
                'avg_vehicle_count': agg.avg_vehicle_count,
                'max_vehicle_count': agg.max_vehicle_count,
                'avg_density': agg.avg_density,
                'max_density': agg.max_density,
                'avg_occupancy': agg.avg_occupancy,
                'avg_speed': agg.avg_speed,
                'min_speed': agg.min_speed,
                'avg_flow': agg.avg_flow,
                'max_queue_length': agg.max_queue_length,
                'avg_congestion_index': agg.avg_congestion_index,
                'speed_variance': agg.speed_variance,
                'count_variance': agg.count_variance,
                'density_variance': agg.density_variance,
                'dominant_state': agg.dominant_state,
                'stability_score': agg.stability_score,
                'sample_count': agg.sample_count
            }
            
            # Add class distribution
            for class_name, avg_count in agg.avg_class_distribution.items():
                row[f'avg_{class_name}_count'] = avg_count
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_feature_matrix(self, 
                          window_size: Optional[float] = None,
                          normalize: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for clustering.
        
        Args:
            window_size: Window size to use
            normalize: Whether to normalize features
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        if window_size is None:
            window_size = self.config.default_window
        
        df = self.get_aggregated_dataframe(window_size)
        
        if df.empty:
            return np.array([]), []
        
        # Select numeric features for clustering
        feature_columns = [
            'avg_vehicle_count', 'avg_density', 'avg_occupancy',
            'avg_speed', 'avg_flow', 'avg_congestion_index',
            'speed_variance', 'density_variance'
        ]
        
        # Filter to existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        # Extract matrix
        X = df[feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Normalize if requested
        if normalize and X.size > 0:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        return X, feature_columns
    
    def get_window_statistics(self) -> Dict[str, Any]:
        """Get comprehensive window statistics."""
        return self.window_manager.get_window_statistics()
    
    def clear(self) -> None:
        """Clear all aggregated data."""
        self.aggregated_features.clear()
        self.window_manager.clear()
        logger.debug("Temporal aggregator cleared")