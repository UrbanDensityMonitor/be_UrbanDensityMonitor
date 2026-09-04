"""
DBSCAN clustering for traffic state discovery.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from ..logger import logger
from ..config import ClusteringConfig


@dataclass
class ClusterResult:
    """
    Container for clustering results.
    """
    labels: np.ndarray
    n_clusters: int
    n_noise: int
    cluster_sizes: Dict[int, int]
    cluster_centers: Dict[int, np.ndarray]
    silhouette_score: float
    parameters: Dict[str, Any]
    feature_names: List[str]
    cluster_statistics: Dict[int, Dict[str, float]] = field(default_factory=dict)
    state_mapping: Dict[int, str] = field(default_factory=dict)


class DBSCANClusterer:
    """
    DBSCAN-based clustering for discovering traffic states.
    """
    
    def __init__(self, config: ClusteringConfig) -> None:
        """
        Initialize DBSCAN clusterer.
        
        Args:
            config: Clustering configuration
        """
        self.config = config
        self.scaler = StandardScaler()
        self.clusterer: Optional[DBSCAN] = None
        self.result: Optional[ClusterResult] = None
        self.is_fitted = False
        
        logger.info(f"DBSCAN clusterer initialized with eps={config.eps}, min_samples={config.min_samples}")
    
    def fit(self, 
            X: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            eps: Optional[float] = None,
            min_samples: Optional[int] = None) -> ClusterResult:
        """
        Fit DBSCAN clustering on feature matrix.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            feature_names: Names of features
            eps: Epsilon parameter (default from config)
            min_samples: Min samples parameter (default from config)
            
        Returns:
            ClusterResult with clustering output
        """
        if eps is None:
            eps = self.config.eps
        if min_samples is None:
            min_samples = self.config.min_samples
        
        if X.size == 0:
            raise ValueError("Empty feature matrix")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit DBSCAN
        self.clusterer = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=self.config.metric,
            n_jobs=-1
        )
        
        labels = self.clusterer.fit_predict(X_scaled)
        
        # Analyze results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(labels).count(-1)
        
        # Calculate cluster sizes
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[int(label)] = int(np.sum(labels == label))
        
        # Calculate cluster centers
        cluster_centers = {}
        for label in unique_labels:
            if label != -1:
                mask = labels == label
                cluster_centers[int(label)] = X[mask].mean(axis=0)
        
        # Calculate silhouette score (only if more than 1 cluster and not all noise)
        silhouette = -1.0
        if n_clusters > 1 and n_noise < len(labels) * 0.5:
            from sklearn.metrics import silhouette_score
            try:
                # Use only non-noise points for silhouette
                mask = labels != -1
                if np.sum(mask) > 1:
                    silhouette = silhouette_score(X_scaled[mask], labels[mask])
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")
        
        # Create feature names if not provided
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Calculate cluster statistics
        cluster_stats = self._calculate_cluster_statistics(X, labels, feature_names)
        
        # Map clusters to traffic states
        state_mapping = self._map_clusters_to_states(cluster_centers, feature_names)
        
        self.result = ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            cluster_sizes=cluster_sizes,
            cluster_centers=cluster_centers,
            silhouette_score=silhouette,
            parameters={'eps': eps, 'min_samples': min_samples},
            feature_names=feature_names,
            cluster_statistics=cluster_stats,
            state_mapping=state_mapping
        )
        
        self.is_fitted = True
        
        logger.info(f"Clustering complete: {n_clusters} clusters found, {n_noise} noise points, silhouette={silhouette:.3f}")
        
        return self.result
    
    def _calculate_cluster_statistics(self, 
                                     X: np.ndarray, 
                                     labels: np.ndarray,
                                     feature_names: List[str]) -> Dict[int, Dict[str, float]]:
        """
        Calculate statistical properties of each cluster.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: List of feature names
            
        Returns:
            Dictionary of cluster statistics
        """
        unique_labels = set(labels)
        statistics = {}
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = labels == label
            cluster_data = X[mask]
            
            stats = {
                'size': int(np.sum(mask)),
                'density': float(np.sum(mask) / len(labels)),
            }
            
            # Feature-wise statistics
            for i, feature_name in enumerate(feature_names):
                if i < cluster_data.shape[1]:
                    stats[f'{feature_name}_mean'] = float(np.mean(cluster_data[:, i]))
                    stats[f'{feature_name}_std'] = float(np.std(cluster_data[:, i]))
            
            statistics[int(label)] = stats
        
        return statistics
    
    def _map_clusters_to_states(self, 
                               cluster_centers: Dict[int, np.ndarray],
                               feature_names: List[str]) -> Dict[int, str]:
        """
        Map cluster labels to semantic traffic states.
        
        Args:
            cluster_centers: Dictionary of cluster centers
            feature_names: Feature names for interpretation
            
        Returns:
            Mapping from cluster ID to state name
        """
        state_mapping = {}
        
        # Find key feature indices
        speed_idx = next((i for i, name in enumerate(feature_names) if 'speed' in name.lower()), 0)
        density_idx = next((i for i, name in enumerate(feature_names) if 'density' in name.lower()), 1)
        congestion_idx = next((i for i, name in enumerate(feature_names) if 'congestion' in name.lower()), 2)
        
        for label, center in cluster_centers.items():
            speed = center[speed_idx] if speed_idx < len(center) else 0
            density = center[density_idx] if density_idx < len(center) else 0
            congestion = center[congestion_idx] if congestion_idx < len(center) else 0
            
            # Classify based on center values
            if congestion > 0.7 or speed < 10:
                state = "Heavy Congestion"
            elif congestion > 0.4 or speed < 30:
                state = "Moderate Traffic"
            elif density < 5 and speed > 50:
                state = "Free Flow"
            elif speed > 40:
                state = "Normal Flow"
            else:
                state = "Mixed Traffic"
            
            state_mapping[int(label)] = state
        
        return state_mapping
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data (nearest cluster center).
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted or self.result is None:
            raise RuntimeError("Clusterer not fitted")
        
        X_scaled = self.scaler.transform(X)
        
        # For DBSCAN, we find nearest cluster center
        labels = np.full(X.shape[0], -1, dtype=int)
        
        for i, sample in enumerate(X_scaled):
            min_dist = float('inf')
            best_label = -1
            
            for label, center in self.result.cluster_centers.items():
                center_scaled = self.scaler.transform(center.reshape(1, -1))[0]
                dist = np.linalg.norm(sample - center_scaled)
                
                if dist < min_dist and dist < self.config.eps:
                    min_dist = dist
                    best_label = label
            
            labels[i] = best_label
        
        return labels
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """
        Get summary of clustering results.
        
        Returns:
            Dictionary with clustering summary
        """
        if not self.is_fitted or self.result is None:
            return {}
        
        return {
            'n_clusters': self.result.n_clusters,
            'n_noise': self.result.n_noise,
            'noise_ratio': self.result.n_noise / len(self.result.labels) if len(self.result.labels) > 0 else 0,
            'cluster_sizes': self.result.cluster_sizes,
            'silhouette_score': self.result.silhouette_score,
            'parameters': self.result.parameters,
            'state_mapping': self.result.state_mapping
        }
    
    def get_labels_dataframe(self, 
                            timestamps: Optional[List[float]] = None) -> pd.DataFrame:
        """
        Create DataFrame with cluster labels.
        
        Args:
            timestamps: Optional timestamps for each sample
            
        Returns:
            DataFrame with labels
        """
        if not self.is_fitted or self.result is None:
            return pd.DataFrame()
        
        data = {'cluster_label': self.result.labels}
        
        if timestamps is not None and len(timestamps) == len(self.result.labels):
            data['timestamp'] = timestamps
        
        # Add state names
        data['traffic_state'] = [
            self.result.state_mapping.get(label, 'Noise')
            for label in self.result.labels
        ]
        
        return pd.DataFrame(data)