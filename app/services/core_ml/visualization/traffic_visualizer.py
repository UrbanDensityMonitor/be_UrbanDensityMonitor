"""
Traffic metrics visualization: density, occupancy, speed, flow, vehicle count.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta

from ..logger import logger
from ..config import VisualizationConfig


class TrafficVisualizer:
    """
    Generates comprehensive traffic analysis visualizations.
    """
    
    def __init__(self, config: VisualizationConfig) -> None:
        """
        Initialize traffic visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        self._setup_style()
    
    def _setup_style(self) -> None:
        """Setup matplotlib style for publication quality."""
        plt.style.use('seaborn-v0_8-darkgrid')
        
        plt.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.font_size + 2,
            'axes.labelsize': self.config.font_size,
            'legend.fontsize': self.config.font_size - 2,
            'xtick.labelsize': self.config.font_size - 2,
            'ytick.labelsize': self.config.font_size - 2,
            'lines.linewidth': 1.5,
            'lines.markersize': 4,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
        })
    
    def plot_vehicle_count(self,
                          df: pd.DataFrame,
                          time_column: str = 'window_start',
                          value_column: str = 'avg_vehicle_count',
                          title: str = 'Vehicle Count Over Time',
                          save_path: Optional[Path] = None) -> Figure:
        """
        Plot vehicle count over time.
        
        Args:
            df: DataFrame with temporal data
            time_column: Column name for timestamps
            value_column: Column name for vehicle count
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()
        
        # Convert timestamps to datetime if needed
        if pd.api.types.is_numeric_dtype(df[time_column]):
            times = pd.to_datetime(df[time_column], unit='s').values
        else:
            times = df[time_column].values
        
        y_values = df[value_column].values
        
        ax.plot(times, y_values,  # type: ignore
                color='#2196F3', linewidth=2, 
                marker='o', markersize=3, alpha=0.8)
        
        ax.fill_between(times, y_values, alpha=0.2, color='#2196F3')  # type: ignore
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Vehicle Count')
        ax.set_title(title)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics annotation
        mean_val = df[value_column].mean()
        max_val = df[value_column].max()
        ax.axhline(y=float(mean_val), color='red', linestyle='--', alpha=0.5,  # type: ignore
                  label=f'Mean: {mean_val:.1f}')
        
        ax.legend()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_density(self,
                    df: pd.DataFrame,
                    time_column: str = 'window_start',
                    density_column: str = 'avg_density',
                    title: str = 'Traffic Density Over Time',
                    save_path: Optional[Path] = None) -> Figure:
        """
        Plot traffic density over time with LOS regions.
        
        Args:
            df: DataFrame with temporal data
            time_column: Column name for timestamps
            density_column: Column name for density
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()
        
        if pd.api.types.is_numeric_dtype(df[time_column]):
            times = pd.to_datetime(df[time_column], unit='s').values
        else:
            times = df[time_column].values
        
        # LOS regions
        los_boundaries = [
            (0, 7, 'A - Free Flow', '#4CAF50', 0.1),
            (7, 11, 'B - Reasonable Free', '#8BC34A', 0.1),
            (11, 16, 'C - Stable Flow', '#FFEB3B', 0.1),
            (16, 22, 'D - Approaching Unstable', '#FF9800', 0.1),
            (22, 28, 'E - Unstable Flow', '#FF5722', 0.1),
            (28, 50, 'F - Forced Flow', '#F44336', 0.1),
        ]
        
        for lower, upper, label, color, alpha in los_boundaries:
            ax.axhspan(lower, upper, alpha=alpha, color=color, label=label)
        
        ax.plot(times, df[density_column].values,  # type: ignore
                color='#1976D2', linewidth=2, marker='s', 
                markersize=2, alpha=0.9)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Density (vehicles/km/lane)')
        ax.set_title(title)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
        
        ax.legend(loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_occupancy(self,
                      df: pd.DataFrame,
                      time_column: str = 'window_start',
                      occupancy_column: str = 'avg_occupancy',
                      title: str = 'Road Occupancy Over Time',
                      save_path: Optional[Path] = None) -> Figure:
        """
        Plot road occupancy percentage over time.
        
        Args:
            df: DataFrame with temporal data
            time_column: Column name for timestamps
            occupancy_column: Column name for occupancy
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots()
        
        if pd.api.types.is_numeric_dtype(df[time_column]):
            times = pd.to_datetime(df[time_column], unit='s').values
        else:
            times = df[time_column].values
        
        # Create gradient fill based on occupancy level
        ax.plot(times, df[occupancy_column].values,  # type: ignore
                color='#9C27B0', linewidth=2, alpha=0.9)
        
        # Color regions
        ax.fill_between(times, 0, 30, alpha=0.1, color='green', label='Low')  # type: ignore
        ax.fill_between(times, 30, 60, alpha=0.1, color='orange', label='Medium')  # type: ignore
        ax.fill_between(times, 60, 100, alpha=0.1, color='red', label='High')  # type: ignore
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Occupancy (%)')
        ax.set_title(title)
        ax.set_ylim(0, 100)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_speed(self,
                  df: pd.DataFrame,
                  time_column: str = 'window_start',
                  speed_column: str = 'avg_speed',
                  title: str = 'Average Speed Over Time',
                  save_path: Optional[Path] = None) -> Figure:
        """
        Plot vehicle speed over time.
        
        Args:
            df: DataFrame with temporal data
            time_column: Column name for timestamps
            speed_column: Column name for speed
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax1 = plt.subplots()
        
        if pd.api.types.is_numeric_dtype(df[time_column]):
            times = pd.to_datetime(df[time_column], unit='s').values
        else:
            times = df[time_column].values
        
        # Speed line
        ax1.plot(times, df[speed_column].values,  # type: ignore
                color='#FF6F00', linewidth=2, marker='D', 
                markersize=2, alpha=0.8)
        
        # Speed variance as shaded area
        if 'speed_variance' in df.columns:
            std = np.sqrt(df['speed_variance'])
            ax1.fill_between(times,  # type: ignore
                           (df[speed_column] - std).values,
                           (df[speed_column] + std).values,
                           alpha=0.2, color='#FF6F00')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Speed (km/h)', color='#FF6F00')
        ax1.tick_params(axis='y', labelcolor='#FF6F00')
        
        # Secondary axis for congestion index
        if 'avg_congestion_index' in df.columns:
            ax2 = ax1.twinx()
            ax2.plot(times, (df['avg_congestion_index'] * 100).values,  # type: ignore
                    color='#F44336', linewidth=1.5, linestyle='--',
                    alpha=0.6, label='Congestion Index')
            ax2.set_ylabel('Congestion Index (%)', color='#F44336')
            ax2.tick_params(axis='y', labelcolor='#F44336')
            ax2.set_ylim(0, 100)
        
        ax1.set_title(title)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate()
        
        ax1.grid(True, alpha=0.3)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def plot_multi_feature_dashboard(self,
                                    df: pd.DataFrame,
                                    time_column: str = 'window_start',
                                    save_path: Optional[Path] = None) -> Figure:
        """
        Create multi-panel traffic dashboard.
        
        Args:
            df: DataFrame with temporal data
            time_column: Column name for timestamps
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        if pd.api.types.is_numeric_dtype(df[time_column]):
            times = pd.to_datetime(df[time_column], unit='s').values
        else:
            times = df[time_column].values
        
        # Panel 1: Vehicle Count & Flow
        ax = axes[0]
        y_count = df['avg_vehicle_count'].values if 'avg_vehicle_count' in df.columns else np.zeros(len(df))
        ax.plot(times, y_count,  # type: ignore
                color='#2196F3', linewidth=1.5, label='Vehicle Count')
        if 'avg_flow' in df.columns:
            ax2 = ax.twinx()
            ax2.plot(times, df['avg_flow'].values,  # type: ignore
                    color='#4CAF50', linewidth=1.5, linestyle='--', 
                    label='Flow Rate (veh/h)')
            ax2.set_ylabel('Flow (veh/h)', color='#4CAF50')
        ax.set_ylabel('Count', color='#2196F3')
        ax.set_title('Traffic Metrics Dashboard')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Density
        ax = axes[1]
        y_density = df['avg_density'].values if 'avg_density' in df.columns else np.zeros(len(df))
        ax.plot(times, y_density,  # type: ignore
                color='#FF9800', linewidth=1.5)
        ax.fill_between(times, y_density,  # type: ignore
                       alpha=0.3, color='#FF9800')
        ax.set_ylabel('Density (veh/km)')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Speed
        ax = axes[2]
        y_speed = df['avg_speed'].values if 'avg_speed' in df.columns else np.zeros(len(df))
        ax.plot(times, y_speed,  # type: ignore
                color='#E91E63', linewidth=1.5)
        ax.set_ylabel('Speed (km/h)')
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Congestion Index
        ax = axes[3]
        y_congestion = (np.asarray(df['avg_congestion_index']) * 100) if 'avg_congestion_index' in df.columns else np.zeros(len(df))
        ax.plot(times, y_congestion,  # type: ignore
                color='#F44336', linewidth=1.5)
        ax.fill_between(times, 0, y_congestion,  # type: ignore
                       alpha=0.3, color='#F44336')
        ax.set_ylabel('Congestion (%)')
        ax.set_xlabel('Time')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        
        fig.autofmt_xdate()
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """
        Save figure to file with proper settings.
        
        Args:
            fig: Matplotlib figure
            save_path: Path to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        fig.savefig(
            save_path,
            dpi=self.config.dpi,
            bbox_inches='tight',
            format=self.config.save_format
        )
        
        logger.info(f"Figure saved: {save_path}")
        plt.close(fig)