from collections import deque
import numpy as np
from src.features.feature_extractor import TrafficFeatures

class StreamFeatureAggregator:
    def __init__(self, window_size_seconds: float) -> None:
        self.window_size = window_size_seconds
        self.history = deque()
        
    def add_sample(self, timestamp: float, features: TrafficFeatures) -> None:
        self.history.append((timestamp, features))
        while self.history and self.history[0][0] < timestamp - self.window_size:
            self.history.popleft()
            
    def get_aggregated_features(self) -> np.ndarray:
        if not self.history:
            return np.zeros((1, 8))
            
        counts = [h[1].vehicle_count for h in self.history]
        densities = [h[1].vehicle_density for h in self.history]
        occupancies = [h[1].road_occupancy for h in self.history]
        speeds = [h[1].average_speed for h in self.history]
        flows = [h[1].vehicle_flow for h in self.history]
        congestions = [h[1].congestion_index for h in self.history]
        
        avg_vehicle_count = float(np.mean(counts))
        avg_density = float(np.mean(densities))
        avg_occupancy = float(np.mean(occupancies))
        avg_speed = float(np.mean(speeds))
        avg_flow = float(np.mean(flows))
        avg_congestion_index = float(np.mean(congestions))
        speed_variance = float(np.var(speeds)) if len(speeds) > 1 else 0.0
        density_variance = float(np.var(densities)) if len(densities) > 1 else 0.0
        
        return np.array([[
            avg_vehicle_count,
            avg_density,
            avg_occupancy,
            avg_speed,
            avg_flow,
            avg_congestion_index,
            speed_variance,
            density_variance
        ]])
