"""
Congestion heatmap generation for spatial traffic analysis.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from typing import List, Tuple, Optional
from scipy.ndimage import gaussian_filter

from ..logger import logger
from ..config import VisualizationConfig


class HeatmapGenerator:
    """
    Generates spatial heatmaps for vehicle positions and congestion patterns.
    """
    
    def __init__(self, config: VisualizationConfig) -> None:
        """
        Initialize heatmap generator.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
    
    def generate_density_heatmap(self,
                                positions: List[Tuple[float, float]],
                                frame_shape: Tuple[int, int],
                                sigma: float = 25.0,
                                colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Generate vehicle density heatmap.
        
        Args:
            positions: List of (x, y) vehicle positions
            frame_shape: (height, width) of output
            sigma: Gaussian smoothing sigma
            colormap: OpenCV colormap
            
        Returns:
            Heatmap image (BGR format)
        """
        height, width = frame_shape
        heatmap = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate positions
        for x, y in positions:
            xi, yi = int(x), int(y)
            if 0 <= xi < width and 0 <= yi < height:
                heatmap[yi, xi] += 1
        
        # Apply Gaussian smoothing
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        return heatmap_colored
    
    def generate_speed_heatmap(self,
                              positions: List[Tuple[float, float]],
                              speeds: List[float],
                              frame_shape: Tuple[int, int],
                              sigma: float = 20.0) -> np.ndarray:
        """
        Generate speed-based heatmap (red = slow, green = fast).
        
        Args:
            positions: List of (x, y) positions
            speeds: List of speeds in km/h
            frame_shape: (height, width)
            sigma: Smoothing sigma
            
        Returns:
            Heatmap image
        """
        height, width = frame_shape
        speed_map = np.zeros((height, width), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        # Accumulate weighted speeds
        for (x, y), speed in zip(positions, speeds):
            xi, yi = int(x), int(y)
            if 0 <= xi < width and 0 <= yi < height:
                speed_map[yi, xi] += speed
                count_map[yi, xi] += 1
        
        # Calculate average speed
        mask = count_map > 0
        avg_speed = np.zeros_like(speed_map)
        avg_speed[mask] = speed_map[mask] / count_map[mask]
        
        # Smooth
        avg_speed = gaussian_filter(avg_speed, sigma=sigma)
        
        # Create RGB heatmap: red for slow, green for fast
        heatmap_rgb = np.zeros((height, width, 3), dtype=np.uint8)
        
        if avg_speed.max() > 0:
            # Normalize speed
            speed_norm = avg_speed / max(avg_speed.max(), 1)
            
            # Red channel: inverse of speed
            heatmap_rgb[:, :, 2] = ((1 - speed_norm) * 255).astype(np.uint8)
            # Green channel: proportional to speed
            heatmap_rgb[:, :, 1] = (speed_norm * 255).astype(np.uint8)
            # Blue channel: low
            heatmap_rgb[:, :, 0] = 0
        
        return heatmap_rgb
    
    def overlay_heatmap(self,
                       frame: np.ndarray,
                       heatmap: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """
        Overlay heatmap on video frame.
        
        Args:
            frame: Original frame
            heatmap: Heatmap to overlay
            alpha: Transparency factor
            
        Returns:
            Blended image
        """
        # Resize heatmap if needed
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Blend
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        
        return overlay
    
    def create_congestion_heatmap_figure(self,
                                        positions: List[Tuple[float, float]],
                                        congestion_values: List[float],
                                        frame_shape: Tuple[int, int],
                                        title: str = 'Congestion Heatmap',
                                        save_path: Optional[Path] = None) -> Figure:
        """
        Create publication-quality congestion heatmap.
        
        Args:
            positions: Vehicle positions
            congestion_values: Congestion index per vehicle
            frame_shape: Output dimensions
            title: Figure title
            save_path: Save location
            
        Returns:
            Matplotlib figure
        """
        height, width = frame_shape
        
        # Create grid
        grid_size = 30
        grid_h = height // grid_size + 1
        grid_w = width // grid_size + 1
        
        congestion_grid = np.zeros((grid_h, grid_w))
        count_grid = np.zeros((grid_h, grid_w))
        
        for (x, y), congestion in zip(positions, congestion_values):
            gx = int(x // grid_size)
            gy = int(y // grid_size)
            
            if 0 <= gx < grid_w and 0 <= gy < grid_h:
                congestion_grid[gy, gx] += congestion
                count_grid[gy, gx] += 1
        
        # Average
        mask = count_grid > 0
        avg_congestion = np.zeros_like(congestion_grid)
        avg_congestion[mask] = congestion_grid[mask] / count_grid[mask]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        im = ax.imshow(avg_congestion, cmap='YlOrRd', 
                      interpolation='bilinear', origin='upper',
                      extent=(0.0, float(width), float(height), 0.0))
        
        plt.colorbar(im, ax=ax, label='Average Congestion Index')
        
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        ax.set_title(title)
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def _save_figure(self, fig: Figure, save_path: Path) -> None:
        """Save matplotlib figure."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
        logger.info(f"Heatmap saved: {save_path}")
        plt.close(fig)