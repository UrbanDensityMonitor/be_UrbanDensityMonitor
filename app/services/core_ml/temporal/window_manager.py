"""
Time window management for temporal aggregation of traffic features.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Generator
from dataclasses import dataclass, field
from collections import deque
import pandas as pd

from ..logger import logger


@dataclass
class TimeWindow:
    """
    Represents a sliding time window for feature aggregation.
    """
    start_time: float
    end_time: float
    duration: float
    features: List[Any] = field(default_factory=list)
    is_complete: bool = False
    
    @property
    def sample_count(self) -> int:
        """Number of samples in window."""
        return len(self.features)
    
    @property
    def midpoint_time(self) -> float:
        """Center timestamp of window."""
        return (self.start_time + self.end_time) / 2


class WindowManager:
    """
    Manages multiple sliding time windows for temporal aggregation.
    Supports overlapping and non-overlapping windows.
    """
    
    def __init__(self, 
                 window_sizes: List[float],
                 overlap: float = 0.0,
                 min_samples_per_window: int = 3) -> None:
        """
        Initialize window manager.
        
        Args:
            window_sizes: List of window durations in seconds
            overlap: Overlap ratio between windows (0-1)
            min_samples_per_window: Minimum samples required for valid window
        """
        self.window_sizes = sorted(window_sizes)
        self.overlap = overlap
        self.min_samples = min_samples_per_window
        
        # Track windows for each size
        self.windows: Dict[float, List[TimeWindow]] = {
            size: [] for size in window_sizes
        }
        
        # Track completed windows
        self.completed_windows: Dict[float, List[TimeWindow]] = {
            size: [] for size in window_sizes
        }
        
        # Feature buffer
        self.feature_buffer: deque = deque(maxlen=10000)
        self.current_time: float = 0.0
        
        logger.info(f"Window manager initialized with sizes: {window_sizes}s")
    
    def add_sample(self, 
                   timestamp: float, 
                   features: Any) -> None:
        """
        Add a feature sample to the buffer.
        
        Args:
            timestamp: Sample timestamp
            features: Feature object to store
        """
        self.feature_buffer.append({
            'timestamp': timestamp,
            'features': features
        })
        self.current_time = max(self.current_time, timestamp)
        
        # Update windows
        self._update_windows()
    
    def _update_windows(self) -> None:
        """Update window states based on current time."""
        for window_size in self.window_sizes:
            # Calculate new window boundaries
            step = window_size * (1 - self.overlap)
            
            # Find latest complete window
            last_end = 0.0
            if self.windows[window_size]:
                last_end = self.windows[window_size][-1].end_time
            
            # Create new windows if needed
            current_start = last_end + step if last_end > 0 else 0.0
            
            while current_start + window_size <= self.current_time:
                new_window = TimeWindow(
                    start_time=current_start,
                    end_time=current_start + window_size,
                    duration=window_size
                )
                
                # Populate window with relevant samples
                self._populate_window(new_window)
                
                # Mark complete if enough samples
                if new_window.sample_count >= self.min_samples:
                    new_window.is_complete = True
                    self.completed_windows[window_size].append(new_window)
                
                self.windows[window_size].append(new_window)
                current_start += step
        
        # Clean up old windows (keep last 1000)
        for window_size in self.window_sizes:
            if len(self.windows[window_size]) > 1000:
                self.windows[window_size] = self.windows[window_size][-500:]
            
            if len(self.completed_windows[window_size]) > 1000:
                self.completed_windows[window_size] = \
                    self.completed_windows[window_size][-500:]
    
    def _populate_window(self, window: TimeWindow) -> None:
        """
        Populate window with relevant samples from buffer.
        
        Args:
            window: TimeWindow to populate
        """
        window.features = []
        
        for sample in self.feature_buffer:
            if window.start_time <= sample['timestamp'] <= window.end_time:
                window.features.append(sample['features'])
    
    def get_current_window(self, window_size: float) -> Optional[TimeWindow]:
        """
        Get the most recent window of specified size.
        
        Args:
            window_size: Window duration in seconds
            
        Returns:
            Latest TimeWindow or None
        """
        if window_size not in self.windows:
            return None
        
        windows = self.windows[window_size]
        return windows[-1] if windows else None
    
    def get_completed_windows(self, 
                             window_size: Optional[float] = None) -> Dict[float, List[TimeWindow]]:
        """
        Get completed windows.
        
        Args:
            window_size: Specific window size (None for all)
            
        Returns:
            Dictionary of window sizes to completed windows
        """
        if window_size is not None:
            return {window_size: self.completed_windows.get(window_size, [])}
        
        return self.completed_windows
    
    def get_window_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about windows.
        
        Returns:
            Dictionary with window statistics
        """
        stats = {}
        
        for window_size in self.window_sizes:
            all_windows = self.windows[window_size]
            completed = self.completed_windows[window_size]
            
            sample_counts = [w.sample_count for w in all_windows if w.sample_count > 0]
            
            stats[str(window_size)] = {
                'total_windows': len(all_windows),
                'completed_windows': len(completed),
                'avg_samples_per_window': float(np.mean(sample_counts)) if sample_counts else 0.0,
                'min_samples': int(np.min(sample_counts)) if sample_counts else 0,
                'max_samples': int(np.max(sample_counts)) if sample_counts else 0,
            }
        
        return stats
    
    def get_samples_in_range(self, 
                            start_time: float, 
                            end_time: float) -> List[Any]:
        """
        Get all feature samples within a time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            
        Returns:
            List of feature objects
        """
        samples = []
        for sample in self.feature_buffer:
            if start_time <= sample['timestamp'] <= end_time:
                samples.append(sample['features'])
        return samples
    
    def clear(self) -> None:
        """Clear all windows and buffer."""
        self.feature_buffer.clear()
        for window_size in self.window_sizes:
            self.windows[window_size].clear()
            self.completed_windows[window_size].clear()
        self.current_time = 0.0
        logger.debug("Window manager cleared")
    
    def export_window_data(self, 
                          window_size: float,
                          include_incomplete: bool = False) -> pd.DataFrame:
        """
        Export window data to DataFrame.
        
        Args:
            window_size: Window duration in seconds
            include_incomplete: Whether to include incomplete windows
            
        Returns:
            DataFrame with window data
        """
        windows = self.completed_windows.get(window_size, [])
        
        if include_incomplete:
            windows = self.windows.get(window_size, [])
        
        data = []
        for window in windows:
            data.append({
                'start_time': window.start_time,
                'end_time': window.end_time,
                'midpoint_time': window.midpoint_time,
                'duration': window.duration,
                'sample_count': window.sample_count,
                'is_complete': window.is_complete
            })
        
        return pd.DataFrame(data)