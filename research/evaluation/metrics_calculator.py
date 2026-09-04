"""
Clustering and traffic state evaluation metrics calculator.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    fowlkes_mallows_score
)
from scipy.spatial.distance import cdist, pdist

from ..logger import logger


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    internal_metrics: Dict[str, float]
    external_metrics: Dict[str, float]
    stability_metrics: Dict[str, float]
    quality_score: float
    overall_rating: str


class MetricsCalculator:
    """
    Comprehensive metrics calculator for clustering evaluation.
    """
    
    def __init__(self) -> None:
        """Initialize metrics calculator."""
        self.metrics: Optional[EvaluationMetrics] = None
        
        logger.info("MetricsCalculator initialized")
    
    def calculate_all_metrics(self,
                             X: np.ndarray,
                             labels: np.ndarray,
                             ground_truth: Optional[np.ndarray] = None,
                             feature_names: Optional[List[str]] = None) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Predicted cluster labels
            ground_truth: Ground truth labels (optional)
            feature_names: Feature names
            
        Returns:
            EvaluationMetrics object
        """
        logger.info("Calculating all evaluation metrics...")
        
        # Filter noise points
        mask = labels != -1
        X_valid = X[mask]
        labels_valid = labels[mask]
        
        if len(labels_valid) < 2:
            logger.warning("Insufficient valid samples for evaluation")
            return self._empty_metrics()
        
        n_clusters = len(set(labels_valid))
        
        # Internal metrics
        internal = self._calculate_internal_metrics(X_valid, labels_valid)
        
        # External metrics (if ground truth available)
        external = {}
        if ground_truth is not None:
            gt_valid = ground_truth[mask]
            external = self._calculate_external_metrics(labels_valid, gt_valid)
        
        # Stability metrics
        stability = self._calculate_stability_metrics(X_valid, labels_valid)
        
        # Quality score
        quality_score = self._calculate_quality_score(internal, external, stability)
        
        # Overall rating
        overall_rating = self._get_rating(quality_score)
        
        self.metrics = EvaluationMetrics(
            internal_metrics=internal,
            external_metrics=external,
            stability_metrics=stability,
            quality_score=quality_score,
            overall_rating=overall_rating
        )
        
        logger.info(f"Metrics calculated: quality_score={quality_score:.3f}, "
                   f"rating={overall_rating}")
        
        return self.metrics
    
    def _calculate_internal_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate internal clustering metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of internal metrics
        """
        metrics = {}
        
        # Silhouette Score
        try:
            metrics['silhouette_score'] = float(silhouette_score(X, labels))
        except Exception:
            metrics['silhouette_score'] = -1.0
        
        # Davies-Bouldin Index
        try:
            metrics['davies_bouldin_index'] = float(davies_bouldin_score(X, labels))
        except Exception:
            metrics['davies_bouldin_index'] = float('inf')
        
        # Calinski-Harabasz Index
        try:
            metrics['calinski_harabasz_index'] = float(calinski_harabasz_score(X, labels))
        except Exception:
            metrics['calinski_harabasz_index'] = -1.0
        
        # Dunn Index
        metrics['dunn_index'] = self._calculate_dunn_index(X, labels)
        
        # C-Index
        metrics['c_index'] = self._calculate_c_index(X, labels)
        
        # Cluster cohesion
        metrics['cohesion'] = self._calculate_cohesion(X, labels)
        
        # Cluster separation
        metrics['separation'] = self._calculate_separation(X, labels)
        
        return metrics
    
    def _calculate_external_metrics(self, 
                                   labels: np.ndarray, 
                                   ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Calculate external validation metrics.
        
        Args:
            labels: Predicted labels
            ground_truth: Ground truth labels
            
        Returns:
            Dictionary of external metrics
        """
        metrics = {}
        
        try:
            metrics['adjusted_rand_index'] = float(adjusted_rand_score(ground_truth, labels))
        except Exception:
            metrics['adjusted_rand_index'] = 0.0
        
        try:
            metrics['normalized_mutual_info'] = float(normalized_mutual_info_score(ground_truth, labels))
        except Exception:
            metrics['normalized_mutual_info'] = 0.0
        
        try:
            metrics['homogeneity'] = float(homogeneity_score(ground_truth, labels))
        except Exception:
            metrics['homogeneity'] = 0.0
        
        try:
            metrics['completeness'] = float(completeness_score(ground_truth, labels))
        except Exception:
            metrics['completeness'] = 0.0
        
        try:
            metrics['v_measure'] = float(v_measure_score(ground_truth, labels))
        except Exception:
            metrics['v_measure'] = 0.0
        
        try:
            metrics['fowlkes_mallows'] = float(fowlkes_mallows_score(ground_truth, labels))
        except Exception:
            metrics['fowlkes_mallows'] = 0.0
        
        return metrics
    
    def _calculate_stability_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering stability metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of stability metrics
        """
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        
        metrics = {}
        
        # Average pairwise distance within clusters
        intra_dists = []
        for label in unique_labels:
            mask = labels == label
            cluster_points = X[mask]
            
            if len(cluster_points) > 1:
                dists = pdist(cluster_points)
                intra_dists.append(np.mean(dists))
        
        if intra_dists:
            metrics['avg_intra_cluster_distance'] = float(np.mean(intra_dists))
            metrics['std_intra_cluster_distance'] = float(np.std(intra_dists))
        else:
            metrics['avg_intra_cluster_distance'] = 0.0
            metrics['std_intra_cluster_distance'] = 0.0
        
        # Average inter-cluster distance
        if n_clusters > 1:
            centers = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
            inter_dists = pdist(centers)
            metrics['avg_inter_cluster_distance'] = float(np.mean(inter_dists))
            
            # Ratio of separation to cohesion
            if metrics['avg_intra_cluster_distance'] > 0:
                metrics['separation_cohesion_ratio'] = (
                    metrics['avg_inter_cluster_distance'] / 
                    metrics['avg_intra_cluster_distance']
                )
            else:
                metrics['separation_cohesion_ratio'] = 0.0
        else:
            metrics['avg_inter_cluster_distance'] = 0.0
            metrics['separation_cohesion_ratio'] = 0.0
        
        # Cluster size balance
        cluster_sizes = [np.sum(labels == l) for l in unique_labels]
        if cluster_sizes:
            metrics['size_entropy'] = float(
                -sum((s/len(labels)) * np.log(s/len(labels)) 
                     for s in cluster_sizes if s > 0) / np.log(len(cluster_sizes))
            ) if len(cluster_sizes) > 0 else 0.0
        
        return metrics
    
    def _calculate_dunn_index(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Dunn Index."""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0.0
        
        # Minimum inter-cluster distance
        min_inter = float('inf')
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                mask_i = labels == unique_labels[i]
                mask_j = labels == unique_labels[j]
                
                if np.any(mask_i) and np.any(mask_j):
                    dist = np.min(cdist(X[mask_i], X[mask_j]))
                    min_inter = min(min_inter, dist)
        
        # Maximum intra-cluster distance
        max_intra = 0.0
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                dist = np.max(pdist(X[mask]))
                max_intra = max(max_intra, dist)
        
        if max_intra > 0:
            return float(min_inter / max_intra)
        return 0.0
    
    def _calculate_c_index(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate C-Index."""
        unique_labels = np.unique(labels)
        
        # Within-cluster distances
        within_dists = []
        for label in unique_labels:
            mask = labels == label
            if np.sum(mask) > 1:
                dists = pdist(X[mask])
                within_dists.extend(dists.tolist())
        
        if not within_dists:
            return 0.0
        
        # All pairwise distances
        all_dists = pdist(X)
        
        # Sort and find min/max
        sorted_dists = np.sort(all_dists)
        
        n_within = len(within_dists)
        n_total = len(sorted_dists)
        
        if n_total == 0:
            return 0.0
        
        S_min = sorted_dists[:n_within].sum()
        S_max = sorted_dists[-n_within:].sum()
        S = sum(within_dists)
        
        if S_max == S_min:
            return 0.0
        
        return float((S - S_min) / (S_max - S_min))
    
    def _calculate_cohesion(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster cohesion (compactness)."""
        unique_labels = np.unique(labels)
        total_cohesion = 0.0
        
        for label in unique_labels:
            mask = labels == label
            cluster_points = X[mask]
            
            if len(cluster_points) > 1:
                center = cluster_points.mean(axis=0)
                cohesion = np.mean(cdist(cluster_points, center.reshape(1, -1)))
                total_cohesion += cohesion * len(cluster_points)
        
        return float(total_cohesion / len(labels)) if len(labels) > 0 else 0.0
    
    def _calculate_separation(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Calculate cluster separation."""
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < 2:
            return 0.0
        
        centers = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
        total_separation = np.mean(pdist(centers))
        
        return float(total_separation)
    
    def _calculate_quality_score(self,
                                internal: Dict[str, float],
                                external: Dict[str, float],
                                stability: Dict[str, float]) -> float:
        """
        Calculate composite quality score.
        
        Args:
            internal: Internal metrics
            external: External metrics
            stability: Stability metrics
            
        Returns:
            Quality score (0-1)
        """
        score = 0.0
        n_components = 0
        
        # Silhouette score contribution
        sil = internal.get('silhouette_score', -1)
        if sil >= 0:
            score += max(0, sil) * 0.3
            n_components += 0.3
        
        # Davies-Bouldin contribution (inverse)
        db = internal.get('davies_bouldin_index', float('inf'))
        if db != float('inf') and db > 0:
            score += (1.0 / (1.0 + db)) * 0.2
            n_components += 0.2
        
        # Dunn index contribution
        dunn = internal.get('dunn_index', 0)
        if dunn >= 0:
            score += min(1.0, dunn) * 0.15
            n_components += 0.15
        
        # Separation/cohesion ratio
        ratio = stability.get('separation_cohesion_ratio', 0)
        if ratio >= 0:
            score += min(1.0, ratio / 10) * 0.15
            n_components += 0.15
        
        # Size balance
        entropy = stability.get('size_entropy', 0)
        if entropy >= 0:
            score += entropy * 0.1
            n_components += 0.1
        
        # External validation bonus
        if external:
            ari = external.get('adjusted_rand_index', 0)
            nmi = external.get('normalized_mutual_info', 0)
            score += (ari + nmi) / 2 * 0.1
            n_components += 0.1
        
        # Normalize
        if n_components > 0:
            score /= n_components
        
        return float(min(max(score, 0.0), 1.0))
    
    def _get_rating(self, score: float) -> str:
        """Convert score to rating string."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Unsatisfactory"
    
    def _empty_metrics(self) -> EvaluationMetrics:
        """Create empty metrics for invalid inputs."""
        return EvaluationMetrics(
            internal_metrics={},
            external_metrics={},
            stability_metrics={},
            quality_score=0.0,
            overall_rating="Insufficient Data"
        )
    
    def export_metrics(self, output_path: Path) -> None:
        """
        Export metrics to CSV and JSON.
        
        Args:
            output_path: Output directory
        """
        if self.metrics is None:
            logger.warning("No metrics calculated yet")
            return
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Combine all metrics
        all_metrics = {}
        all_metrics.update(self.metrics.internal_metrics)
        all_metrics.update(self.metrics.external_metrics)
        all_metrics.update(self.metrics.stability_metrics)
        all_metrics['quality_score'] = self.metrics.quality_score
        all_metrics['overall_rating'] = self.metrics.overall_rating
        
        # Save as DataFrame
        metrics_df = pd.DataFrame([
            {'metric': k, 'value': v} for k, v in all_metrics.items()
        ])
        metrics_df.to_csv(output_path / 'evaluation_metrics.csv', index=False)
        
        logger.info(f"Metrics exported to: {output_path}")