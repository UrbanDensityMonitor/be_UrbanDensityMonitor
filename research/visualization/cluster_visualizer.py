"""
Cluster visualization: scatter plots, state distributions, and transition diagrams.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from ..logger import logger
from ..config import VisualizationConfig
from ..clustering.dbscan_clustering import ClusterResult


class ClusterVisualizer:
    """
    Visualizes clustering results for traffic state discovery.
    """
    
    def __init__(self, config: VisualizationConfig) -> None:
        """
        Initialize cluster visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_cluster_scatter(self,
                             X: np.ndarray,
                             result: ClusterResult,
                             method: str = 'pca',
                             title: str = 'Traffic State Clusters',
                             save_path: Optional[Path] = None) -> Figure:
        """
        Create 2D scatter plot of clusters using dimensionality reduction.
        
        Args:
            X: Feature matrix
            result: Clustering result
            method: Reduction method ('pca' or 'tsne')
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if len(X) < 2:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.text(0.5, 0.5, 'Insufficient Data for Clustering Projection\n(Need >= 2 samples)', 
                    ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            if save_path:
                self._save_figure(fig, save_path)
            return fig

        # Reduce dimensions to 2D
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        else:
            reducer = PCA(n_components=2)
        
        X_2d = reducer.fit_transform(X)
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Get unique labels
        unique_labels = set(result.labels)
        n_clusters = result.n_clusters
        
        # Color map
        colors = matplotlib.colormaps.get_cmap(self.config.colormap)
        
        # Plot noise points
        noise_mask = result.labels == -1
        if np.any(noise_mask):
            ax.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                      c='gray', marker='x', alpha=0.3, s=30, label='Noise')
        
        # Plot clusters
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = result.labels == label
            color = colors(label / max(n_clusters, 1))
            
            # Get state name
            state_name = result.state_mapping.get(label, f'Cluster {label}')
            
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                      c=[color], label=state_name,
                      alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(title)
        
        # Add cluster centers in 2D
        if result.cluster_centers:
            centers_2d_list = []
            for label in sorted(result.cluster_centers.keys()):
                mask = result.labels == label
                if np.any(mask):
                    centers_2d_list.append(X_2d[mask].mean(axis=0))
            if centers_2d_list:
                centers_2d = np.array(centers_2d_list)
                ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                          c='red', marker='X', s=200, edgecolors='black',
                          linewidth=2, label='Centers', zorder=5)
        
        # Remove duplicate labels in legend if multiple clusters map to the same state name
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_state_distribution(self,
                               result: ClusterResult,
                               title: str = 'Traffic State Distribution',
                               save_path: Optional[Path] = None) -> Figure:
        """
        Plot bar chart of cluster/state distribution.
        
        Args:
            result: Clustering result
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Group sizes by state name
        state_totals = {}
        for cluster_id, size in result.cluster_sizes.items():
            state_name = result.state_mapping.get(cluster_id, f'Cluster {cluster_id}')
            state_totals[state_name] = state_totals.get(state_name, 0) + size
            
        states = list(state_totals.keys())
        sizes = list(state_totals.values())
        
        if not sizes and result.n_noise == 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.text(0.5, 0.5, 'No State Data Available', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            if save_path:
                self._save_figure(fig, save_path)
            return fig

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        n_colors = max(1, len(states))
        colors = matplotlib.colormaps.get_cmap(self.config.colormap).resampled(n_colors)
        
        if states:
            bars = ax1.bar(range(len(states)), sizes, color=[colors(i) for i in range(len(states))])
            ax1.set_xlabel('Traffic State')
            ax1.set_ylabel('Number of Samples')
            ax1.set_title('Cluster Sizes')
            ax1.set_xticks(range(len(states)))
            ax1.set_xticklabels(states, rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, size in zip(bars, sizes):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(size), ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No Cluster Data', ha='center', va='center')
            ax1.set_axis_off()
        
        # Pie chart
        if result.n_noise > 0:
            sizes_with_noise = sizes + [result.n_noise]
            labels_with_noise = states + ['Noise']
            explode = [0.05] * len(sizes_with_noise)
        else:
            sizes_with_noise = sizes
            labels_with_noise = states
            explode = [0.05] * len(sizes)
        
        n_pie_colors = max(1, len(sizes_with_noise))
        pie_colors = matplotlib.colormaps.get_cmap(self.config.colormap).resampled(n_pie_colors)
        
        if sizes_with_noise:
            ax2.pie(sizes_with_noise, labels=labels_with_noise, autopct='%1.1f%%',
                   explode=explode, colors=[pie_colors(i) for i in range(len(sizes_with_noise))],
                   shadow=True, startangle=90)
        else:
            ax2.text(0.5, 0.5, 'No Data', ha='center', va='center')
            ax2.set_axis_off()
            
        ax2.set_title('State Distribution')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_timeline_states(self,
                             labels: np.ndarray,
                             state_mapping: Dict[int, str],
                             timestamps: List[float],
                             title: str = 'Traffic State Timeline',
                             save_path: Optional[Path] = None) -> Figure:
        """
        Plot timeline of traffic states as colored regions.
        
        Args:
            labels: Cluster labels
            state_mapping: Mapping from label to state name
            timestamps: Timestamps for each sample
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        if len(labels) == 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.text(0.5, 0.5, 'No Timeline Data Available', ha='center', va='center', fontsize=12)
            ax.set_axis_off()
            if save_path:
                self._save_figure(fig, save_path)
            return fig

        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Convert timestamps to datetime
        times = pd.to_datetime(timestamps, unit='s')
        
        # Get unique states
        unique_labels = sorted(set(labels))
        states = [state_mapping.get(l, f'Unknown_{l}') for l in unique_labels]
        
        # Create color map for states
        n_states = max(1, len(unique_labels))
        colors = matplotlib.colormaps.get_cmap(self.config.colormap).resampled(n_states)
        state_to_color = {state: colors(i) for i, state in enumerate(states)}
        
        # Plot each point as colored dot
        for i, (time, label) in enumerate(zip(times, labels)):
            state = state_mapping.get(label, f'Unknown_{label}')
            color = state_to_color.get(state, 'gray')
            ax.scatter(time, 0, c=[color], s=100, marker='s', alpha=0.7)
        
        # Add state transitions
        prev_label = None
        for i, (time, label) in enumerate(zip(times, labels)):
            if label != prev_label and prev_label is not None and i > 0:
                ax.axvline(x=time, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
            prev_label = label
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=state_to_color[state], label=state)
            for state in states
        ]
        ax.legend(handles=legend_elements, loc='upper right', ncol=len(states)//2 + 1)
        
        ax.set_xlabel('Time')
        ax.set_title(title)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_feature_importance(self,
                               X: np.ndarray,
                               result: ClusterResult,
                               title: str = 'Feature Distribution by Cluster',
                               save_path: Optional[Path] = None) -> Figure:
        """
        Plot feature distributions across clusters.
        
        Args:
            X: Feature matrix
            result: Clustering result
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        n_features = X.shape[1]
        feature_names = result.feature_names if result.feature_names else [f'F{i}' for i in range(n_features)]
        
        n_clusters = result.n_clusters
        if n_clusters == 0:
            return plt.figure()
        
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 3 * n_features))
        
        if isinstance(axes, np.ndarray):
            axes_list = list(axes.flat)
        else:
            axes_list = [axes]
        
        for i, (ax, feature_name) in enumerate(zip(axes_list, feature_names)):
            for label in sorted(result.cluster_centers.keys()):
                mask = result.labels == label
                if np.any(mask):
                    state = result.state_mapping.get(label, f'Cluster {label}')
                    ax.hist(X[mask, i], bins=20, alpha=0.5, label=state, density=True)
            
            ax.set_xlabel(feature_name)
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """Save figure to file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Figure saved: {save_path}")
        plt.close(fig)