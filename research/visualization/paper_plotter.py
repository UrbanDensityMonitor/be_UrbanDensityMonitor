"""
Publication-ready figure generation for research paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import seaborn as sns

from ..logger import logger
from ..config import VisualizationConfig


class PaperPlotter:
    """
    Generates publication-quality figures for research papers.
    All figures follow academic formatting standards.
    """
    
    def __init__(self, config: VisualizationConfig) -> None:
        """
        Initialize paper plotter.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self._setup_paper_style()
    
    def _setup_paper_style(self) -> None:
        """Setup matplotlib style for publication."""
        plt.style.use('default')
        
        plt.rcParams.update({
            'figure.figsize': (8, 6),
            'figure.dpi': 300,
            'font.size': 11,
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'axes.linewidth': 1.0,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.05,
            'savefig.format': 'pdf',
        })
    
    def plot_pipeline_overview(self, save_path: Path) -> Figure:
        """
        Generate pipeline overview diagram.
        
        Args:
            save_path: Save location
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Pipeline stages
        stages = [
            'Video Input',
            'YOLO Detection',
            'ByteTrack\nTracking',
            'ROI Filtering',
            'Feature\nExtraction',
            'Temporal\nAggregation',
            'DBSCAN\nClustering',
            'Traffic State\nDiscovery'
        ]
        
        n_stages = len(stages)
        positions = range(n_stages)
        
        # Draw boxes
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, n_stages))
        
        for i, (stage, color) in enumerate(zip(stages, colors)):
            rect = Rectangle((i - 0.4, 0), 0.8, 1, 
                                facecolor=color, edgecolor='black', 
                                linewidth=1.5, alpha=0.8)
            ax.add_patch(rect)
            ax.text(i, 0.5, stage, ha='center', va='center',
                   fontsize=8, fontweight='bold', color='white')
        
        # Draw arrows
        for i in range(n_stages - 1):
            ax.annotate('', xy=(i + 0.55, 0.5), xytext=(i + 0.4, 0.5),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.set_xlim(-0.5, n_stages - 0.5)
        ax.set_ylim(-0.2, 1.2)
        ax.axis('off')
        ax.set_title('Traffic State Discovery Pipeline', fontsize=14, fontweight='bold', pad=20)
        
        self._save_paper_figure(fig, save_path)
        return fig
    
    def plot_cluster_comparison(self,
                              X: np.ndarray,
                              result: Any,
                              save_path: Path) -> Figure:
        """
        Generate comprehensive cluster comparison figure.
        
        Args:
            X: Feature matrix
            result: Clustering result
            save_path: Save location
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        
        # Create grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Subplot 1: Cluster scatter
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_cluster_scatter_paper(ax1, X, result)
        
        # Subplot 2: State distribution pie
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_state_pie_paper(ax2, result)
        
        # Subplot 3: Feature importance heatmap
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_feature_heatmap_paper(ax3, X, result)
        
        fig.suptitle('Traffic State Clustering Analysis', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        self._save_paper_figure(fig, save_path)
        return fig
    
    def _plot_cluster_scatter_paper(self, ax, X, result):
        """Paper-style cluster scatter plot."""
        if X.shape[0] < 2:
            ax.text(0.5, 0.5, 'Insufficient Data for PCA\n(Need >= 2 samples)', 
                    ha='center', va='center', fontsize=10)
            ax.set_axis_off()
            return

        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        unique_labels = sorted(set(result.labels))
        colors = plt.cm.Set2(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = result.labels == label
            if label == -1:
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                         c='gray', marker='x', s=20, alpha=0.3, label='Noise')
            else:
                state = result.state_mapping.get(label, f'C{label}')
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                         c=[color], s=30, alpha=0.7, label=state,
                         edgecolors='black', linewidth=0.3)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)
    
    def _plot_state_pie_paper(self, ax, result):
        """Paper-style state distribution pie chart."""
        states = [result.state_mapping.get(k, f'C{k}') 
                 for k in sorted(result.cluster_sizes.keys())]
        sizes = [result.cluster_sizes[k] for k in sorted(result.cluster_sizes.keys())]
        
        if result.n_noise > 0:
            states.append('Noise')
            sizes.append(result.n_noise)
        
        if not sizes:
            ax.text(0.5, 0.5, 'No State Data Available', ha='center', va='center', fontsize=10)
            ax.set_axis_off()
            return

        colors = plt.cm.Set3(np.linspace(0, 1, len(states)))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=states, autopct='%1.1f%%',
            colors=colors, startangle=90,
            textprops={'fontsize': 8}
        )
        
        for autotext in autotexts:
            autotext.set_fontsize(7)
    
    def _plot_feature_heatmap_paper(self, ax, X, result):
        """Paper-style feature heatmap by cluster."""
        feature_names = result.feature_names if result.feature_names else [f'F{i}' for i in range(X.shape[1])]
        
        # Calculate mean feature values per cluster
        cluster_features = {}
        for label in sorted(result.cluster_centers.keys()):
            mask = result.labels == label
            cluster_features[f'C{label}'] = X[mask].mean(axis=0)
        
        if not cluster_features:
            ax.text(0.5, 0.5, 'No Clusters Discovered\n(All points identified as Noise)', 
                    ha='center', va='center', fontsize=10)
            ax.set_axis_off()
            return

        # Create heatmap
        data = np.array(list(cluster_features.values()))
        cluster_labels = list(cluster_features.keys())
        
        im = ax.imshow(data.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
        
        ax.set_xticks(range(len(cluster_labels)))
        ax.set_xticklabels(cluster_labels)
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        
        plt.colorbar(im, ax=ax, label='Standardized Value')
        
        # Add text annotations
        for i in range(len(cluster_labels)):
            for j in range(len(feature_names)):
                ax.text(i, j, f'{data[i, j]:.2f}', 
                       ha='center', va='center', fontsize=6)
    
    def plot_evaluation_summary(self,
                               evaluation_data: Dict[str, Any],
                               save_path: Path) -> Figure:
        """
        Generate evaluation metrics summary figure.
        
        Args:
            evaluation_data: Dictionary of evaluation metrics
            save_path: Save location
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot 1: Parameter search heatmap
        if 'grid_search' in evaluation_data:
            df = pd.DataFrame(evaluation_data['grid_search'])
            pivot = df.pivot_table(
                values='combined',
                index='min_samples',
                columns='eps'
            )
            
            im = axes[0].imshow(pivot.values, cmap='viridis', aspect='auto')
            axes[0].set_xticks(range(len(pivot.columns)))
            axes[0].set_xticklabels([f'{x:.2f}' for x in pivot.columns], rotation=45)
            axes[0].set_yticks(range(len(pivot.index)))
            axes[0].set_yticklabels(pivot.index)
            axes[0].set_xlabel('eps')
            axes[0].set_ylabel('min_samples')
            axes[0].set_title('Parameter Search Heatmap')
            plt.colorbar(im, ax=axes[0], label='Combined Score')
        
        # Plot 2: Metric comparison bar chart
        if 'metrics' in evaluation_data:
            metrics = evaluation_data['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = axes[1].bar(metric_names, metric_values, 
                             color=plt.cm.Set2(np.linspace(0, 1, len(metric_names))))
            axes[1].set_ylabel('Score')
            axes[1].set_title('Clustering Metrics')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, metric_values):
                axes[1].text(bar.get_x() + bar.get_width()/2, 
                           bar.get_height() + 0.01,
                           f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.suptitle('Clustering Evaluation Summary', fontweight='bold')
        plt.tight_layout()
        
        self._save_paper_figure(fig, save_path)
        return fig
    
    def _save_paper_figure(self, fig: Figure, save_path: Path) -> None:
        """Save paper-quality figure."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save both PDF and PNG
        fig.savefig(save_path.with_suffix('.pdf'), format='pdf', dpi=300)
        fig.savefig(save_path.with_suffix('.png'), format='png', dpi=300)
        
        logger.info(f"Paper figure saved: {save_path}")
        plt.close(fig)