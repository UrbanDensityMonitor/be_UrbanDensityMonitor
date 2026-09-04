"""
Grid search optimization for DBSCAN parameters.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from itertools import product
from dataclasses import dataclass
import warnings

from ..logger import logger
from ..config import ClusteringConfig
from .dbscan_clustering import DBSCANClusterer, ClusterResult


@dataclass
class GridSearchResult:
    """Container for grid search results."""
    best_params: Dict[str, Any]
    best_score: float
    best_result: Optional[ClusterResult]
    all_results: List[Dict[str, Any]]
    search_space: Dict[str, List[float]]
    metric_name: str


class GridSearchOptimizer:
    """
    Grid search for optimal DBSCAN parameters.
    Evaluates based on silhouette score and cluster validity.
    """
    
    def __init__(self, config: ClusteringConfig) -> None:
        """
        Initialize grid search optimizer.
        
        Args:
            config: Clustering configuration with search ranges
        """
        self.config = config
        self.clusterer = DBSCANClusterer(config)
        self.search_results: List[Dict[str, Any]] = []
        
        logger.info(f"Grid search initialized: eps={config.eps_range}, min_samples={config.min_samples_range}")
    
    def search(self, 
               X: np.ndarray,
               feature_names: Optional[List[str]] = None,
               metric: str = 'silhouette',
               verbose: bool = True) -> GridSearchResult:
        """
        Perform grid search over DBSCAN parameters.
        
        Args:
            X: Feature matrix
            feature_names: Feature names
            metric: Evaluation metric ('silhouette', 'davies_bouldin', 'calinski_harabasz', 'combined')
            verbose: Print progress
            
        Returns:
            GridSearchResult with best parameters
        """
        if X.size == 0:
            raise ValueError("Empty feature matrix")
        
        # Create parameter grid
        param_grid = {
            'eps': self.config.eps_range,
            'min_samples': self.config.min_samples_range
        }
        
        total_combinations = len(param_grid['eps']) * len(param_grid['min_samples'])
        
        if verbose:
            logger.info(f"Starting grid search: {total_combinations} combinations")
        
        best_score = -float('inf')
        best_params = {}
        best_result = None
        
        self.search_results = []
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for eps, min_samples in product(param_grid['eps'], param_grid['min_samples']):
                try:
                    # Fit clusterer
                    result = self.clusterer.fit(
                        X, 
                        feature_names=feature_names,
                        eps=eps,
                        min_samples=min_samples
                    )
                    
                    # Calculate metrics
                    scores = self._evaluate_clustering(result, X, metric)
                    
                    search_entry = {
                        'eps': eps,
                        'min_samples': min_samples,
                        'n_clusters': result.n_clusters,
                        'n_noise': result.n_noise,
                        'scores': scores
                    }
                    
                    self.search_results.append(search_entry)
                    
                    # Update best
                    combined_score = scores.get('combined', scores.get(metric, -1))
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = {'eps': eps, 'min_samples': min_samples}
                        best_result = result
                    
                    if verbose:
                        logger.debug(
                            f"eps={eps:.2f}, min_samples={min_samples}: "
                            f"clusters={result.n_clusters}, noise={result.n_noise}, "
                            f"score={combined_score:.3f}"
                        )
                
                except Exception as e:
                    logger.warning(f"Failed for eps={eps}, min_samples={min_samples}: {e}")
                    continue
        
        if best_result is None:
            raise RuntimeError("No valid clustering found in grid search")
        
        # Update config with best parameters
        self.config.eps = best_params['eps']
        self.config.min_samples = best_params['min_samples']
        
        grid_result = GridSearchResult(
            best_params=best_params,
            best_score=best_score,
            best_result=best_result,
            all_results=self.search_results,
            search_space=param_grid,
            metric_name=metric
        )
        
        logger.info(f"Grid search complete: best params={best_params}, score={best_score:.3f}")
        
        return grid_result
    
    def _evaluate_clustering(self, 
                            result: ClusterResult, 
                            X: np.ndarray,
                            primary_metric: str) -> Dict[str, float]:
        """
        Evaluate clustering using multiple metrics.
        
        Args:
            result: Clustering result
            X: Feature matrix
            primary_metric: Primary evaluation metric
            
        Returns:
            Dictionary of metric scores
        """
        scores = {}
        
        # Only evaluate if we have clusters
        if result.n_clusters < 1:
            scores['silhouette'] = -1.0
            scores['davies_bouldin'] = float('inf')
            scores['calinski_harabasz'] = -1.0
            scores['combined'] = -1.0
            return scores
        
        # Remove noise points for evaluation
        mask = result.labels != -1
        n_valid = np.sum(mask)
        
        if n_valid < 2 or result.n_clusters < 2:
            scores['silhouette'] = -1.0
            scores['davies_bouldin'] = float('inf')
            scores['calinski_harabasz'] = -1.0
            scores['combined'] = -1.0
            return scores
        
        X_valid = X[mask]
        labels_valid = result.labels[mask]
        
        # Silhouette score
        try:
            from sklearn.metrics import silhouette_score
            scores['silhouette'] = silhouette_score(X_valid, labels_valid)
        except Exception:
            scores['silhouette'] = -1.0
        
        # Davies-Bouldin index (lower is better)
        try:
            from sklearn.metrics import davies_bouldin_score
            scores['davies_bouldin'] = davies_bouldin_score(X_valid, labels_valid)
        except Exception:
            scores['davies_bouldin'] = float('inf')
        
        # Calinski-Harabasz index (higher is better)
        try:
            from sklearn.metrics import calinski_harabasz_score
            scores['calinski_harabasz'] = calinski_harabasz_score(X_valid, labels_valid)
        except Exception:
            scores['calinski_harabasz'] = -1.0
        
        # Combined score (weighted average)
        silhouette_norm = max(0, scores['silhouette'])
        
        db_score = scores['davies_bouldin']
        db_norm = 1.0 / (1.0 + db_score) if db_score != float('inf') else 0.0
        
        ch_score = scores['calinski_harabasz']
        ch_norm = min(1.0, ch_score / 10000) if ch_score > 0 else 0.0
        
        # Penalize extreme numbers of clusters
        cluster_penalty = 1.0
        if result.n_clusters <= 1:
            cluster_penalty = 0.3
        elif result.n_clusters > 10:
            cluster_penalty = 0.7
        elif result.n_clusters > 15:
            cluster_penalty = 0.5
        
        # Penalize high noise ratio
        noise_ratio = result.n_noise / len(result.labels) if len(result.labels) > 0 else 1.0
        noise_penalty = 1.0 - min(0.5, noise_ratio)
        
        scores['combined'] = (
            0.4 * silhouette_norm +
            0.2 * db_norm +
            0.2 * ch_norm +
            0.1 * cluster_penalty +
            0.1 * noise_penalty
        )
        
        return scores
    
    def get_search_dataframe(self) -> pd.DataFrame:
        """
        Convert grid search results to DataFrame.
        
        Returns:
            DataFrame with search results
        """
        if not self.search_results:
            return pd.DataFrame()
        
        data = []
        for result in self.search_results:
            row = {
                'eps': result['eps'],
                'min_samples': result['min_samples'],
                'n_clusters': result['n_clusters'],
                'n_noise': result['n_noise'],
            }
            row.update(result['scores'])
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """
        Get best parameters found.
        
        Returns:
            Dictionary with best parameters
        """
        return {
            'eps': self.config.eps,
            'min_samples': self.config.min_samples
        }