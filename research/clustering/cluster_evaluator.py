"""
Clustering evaluation and validation metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from ..logger import logger
from .dbscan_clustering import ClusterResult


@dataclass
class EvaluationReport:
    """Container for clustering evaluation results."""
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_index: float
    noise_ratio: float
    cluster_separation: float
    cluster_cohesion: Dict[int, float]
    inter_cluster_distances: Dict[Tuple[int, int], float]
    stability_score: float
    summary: Dict[str, Any]


class ClusterEvaluator:
    """
    Comprehensive evaluation of clustering results.
    """
    
    def __init__(self) -> None:
        """Initialize cluster evaluator."""
        self.report: Optional[EvaluationReport] = None
    
    def evaluate(self, 
                 X: np.ndarray,
                 result: ClusterResult,
                 X_scaled: Optional[np.ndarray] = None) -> EvaluationReport:
        """
        Perform comprehensive clustering evaluation.
        
        Args:
            X: Original feature matrix
            result: Clustering result
            X_scaled: Pre-scaled feature matrix (optional)
            
        Returns:
            EvaluationReport with all metrics
        """
        if result.n_clusters < 1:
            logger.warning("Cannot evaluate: no clusters found")
            return self._create_empty_report()
        
        # Remove noise points
        mask = result.labels != -1
        X_valid = X[mask]
        labels_valid = result.labels[mask]
        
        if len(X_valid) < 2 or result.n_clusters < 2:
            logger.warning("Cannot evaluate: insufficient clustered samples")
            return self._create_empty_report()
        
        # Standard metrics
        try:
            sil_score = silhouette_score(X_valid, labels_valid)
        except Exception:
            sil_score = -1.0
        
        try:
            db_index = davies_bouldin_score(X_valid, labels_valid)
        except Exception:
            db_index = float('inf')
        
        try:
            ch_index = calinski_harabasz_score(X_valid, labels_valid)
        except Exception:
            ch_index = -1.0
        
        # Noise ratio
        noise_ratio = result.n_noise / len(result.labels) if len(result.labels) > 0 else 0
        
        # Cluster cohesion (intra-cluster distances)
        cohesion = self._calculate_cohesion(X_valid, labels_valid, result)
        
        # Cluster separation (inter-cluster distances)
        separation, inter_distances = self._calculate_separation(
            X_valid, labels_valid, result
        )
        
        # Stability score
        stability = self._estimate_stability(X_valid, labels_valid, result)
        
        # Summary
        summary = {
            'total_samples': len(result.labels),
            'clustered_samples': int(np.sum(mask)),
            'noise_samples': result.n_noise,
            'n_clusters': result.n_clusters,
            'cluster_sizes': result.cluster_sizes,
            'overall_quality': self._rate_overall_quality(
                sil_score, db_index, noise_ratio
            )
        }
        
        self.report = EvaluationReport(
            silhouette_score=sil_score,
            davies_bouldin_index=db_index,
            calinski_harabasz_index=ch_index,
            noise_ratio=noise_ratio,
            cluster_separation=separation,
            cluster_cohesion=cohesion,
            inter_cluster_distances=inter_distances,
            stability_score=stability,
            summary=summary
        )
        
        logger.info(
            f"Evaluation complete: silhouette={sil_score:.3f}, "
            f"DB={db_index:.3f}, CH={ch_index:.1f}"
        )
        
        return self.report
    
    def _calculate_cohesion(self, 
                           X: np.ndarray, 
                           labels: np.ndarray,
                           result: ClusterResult) -> Dict[int, float]:
        """Calculate intra-cluster cohesion."""
        cohesion = {}
        
        for label in result.cluster_centers.keys():
            mask = labels == label
            cluster_points = X[mask]
            
            if len(cluster_points) > 1:
                center = cluster_points.mean(axis=0)
                distances = np.linalg.norm(cluster_points - center, axis=1)
                cohesion[label] = float(np.mean(distances))
            else:
                cohesion[label] = 0.0
        
        return cohesion
    
    def _calculate_separation(self, 
                             X: np.ndarray, 
                             labels: np.ndarray,
                             result: ClusterResult) -> Tuple[float, Dict[Tuple[int, int], float]]:
        """Calculate inter-cluster separation."""
        centers = result.cluster_centers
        labels_list = list(centers.keys())
        
        inter_distances = {}
        total_distance = 0.0
        count = 0
        
        for i in range(len(labels_list)):
            for j in range(i + 1, len(labels_list)):
                label_i = labels_list[i]
                label_j = labels_list[j]
                
                center_i = centers[label_i]
                center_j = centers[label_j]
                
                distance = np.linalg.norm(center_i - center_j)
                inter_distances[(label_i, label_j)] = float(distance)
                
                total_distance += distance
                count += 1
        
        avg_separation = total_distance / count if count > 0 else 0.0
        
        return avg_separation, inter_distances
    
    def _estimate_stability(self, 
                          X: np.ndarray, 
                          labels: np.ndarray,
                          result: ClusterResult) -> float:
        """Estimate clustering stability."""
        if len(X) < 10:
            return 0.0
        
        # Simple stability estimation based on cluster size consistency
        sizes = list(result.cluster_sizes.values())
        
        if not sizes:
            return 0.0
        
        # Coefficient of variation of cluster sizes
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        cv = std_size / mean_size if mean_size > 0 else 0.0
        
        # Lower CV = more balanced clusters = potentially more stable
        stability = 1.0 / (1.0 + cv)
        
        return float(stability)
    
    def _rate_overall_quality(self, 
                             silhouette: float, 
                             db_index: float,
                             noise_ratio: float) -> str:
        """Rate overall clustering quality."""
        if silhouette > 0.7 and noise_ratio < 0.1:
            return "Excellent"
        elif silhouette > 0.5 and noise_ratio < 0.2:
            return "Good"
        elif silhouette > 0.3 and noise_ratio < 0.3:
            return "Fair"
        elif silhouette > 0.1:
            return "Poor"
        else:
            return "Unsatisfactory"
    
    def _create_empty_report(self) -> EvaluationReport:
        """Create empty evaluation report."""
        return EvaluationReport(
            silhouette_score=-1.0,
            davies_bouldin_index=float('inf'),
            calinski_harabasz_index=-1.0,
            noise_ratio=1.0,
            cluster_separation=0.0,
            cluster_cohesion={},
            inter_cluster_distances={},
            stability_score=0.0,
            summary={'overall_quality': 'No clusters found'}
        )
    
    def export_report(self) -> Dict[str, Any]:
        """
        Export evaluation report as dictionary.
        
        Returns:
            Dictionary with all evaluation metrics
        """
        if self.report is None:
            return {}
        
        return {
            'silhouette_score': self.report.silhouette_score,
            'davies_bouldin_index': self.report.davies_bouldin_index,
            'calinski_harabasz_index': self.report.calinski_harabasz_index,
            'noise_ratio': self.report.noise_ratio,
            'cluster_separation': self.report.cluster_separation,
            'stability_score': self.report.stability_score,
            'summary': self.report.summary
        }
    
    def compare_with_baseline(self, 
                             result: ClusterResult,
                             baseline_labels: np.ndarray) -> Dict[str, float]:
        """
        Compare clustering with ground truth or baseline.
        
        Args:
            result: Clustering result
            baseline_labels: Baseline/ground truth labels
            
        Returns:
            Dictionary of comparison metrics
        """
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        # Filter noise points for comparison
        mask = result.labels != -1
        
        if np.sum(mask) < 2:
            return {'ari': 0.0, 'nmi': 0.0}
        
        ari = adjusted_rand_score(
            baseline_labels[mask], 
            result.labels[mask]
        )
        
        nmi = normalized_mutual_info_score(
            baseline_labels[mask],
            result.labels[mask]
        )
        
        return {
            'adjusted_rand_index': ari,
            'normalized_mutual_info': nmi
        }