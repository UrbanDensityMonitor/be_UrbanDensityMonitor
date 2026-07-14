"""
Progress tracking utilities for long-running pipeline operations.
"""

import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

from ..logger import logger


@dataclass
class ProgressInfo:
    """Container for progress information."""
    current_step: int
    total_steps: int
    percentage: float
    elapsed_time: float
    estimated_remaining: float
    steps_per_second: float
    status: str


class ProgressTracker:
    """
    Tracks and reports progress for pipeline steps.
    Provides ETA and performance metrics.
    """
    
    def __init__(self, 
                 total_steps: int,
                 description: str = "Processing",
                 update_interval: float = 1.0) -> None:
        """
        Initialize progress tracker.
        
        Args:
            total_steps: Total number of steps
            description: Description of the task
            update_interval: Minimum interval between updates (seconds)
        """
        self.total_steps = total_steps
        self.description = description
        self.update_interval = update_interval
        
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.step_times: deque = deque(maxlen=100)
        
        self.callbacks: list = []
        self.substeps: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"Progress tracker initialized: {description} ({total_steps} steps)")
    
    def update(self, 
               steps: int = 1, 
               status: Optional[str] = None,
               force_update: bool = False) -> None:
        """
        Update progress by specified steps.
        
        Args:
            steps: Number of steps completed
            status: Optional status message
            force_update: Force update regardless of interval
        """
        self.current_step = min(self.current_step + steps, self.total_steps)
        
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        # Track step timing
        if steps > 0:
            step_time = elapsed / steps
            self.step_times.append(step_time)
        
        # Update only if interval elapsed or forced
        if elapsed >= self.update_interval or force_update or self.current_step >= self.total_steps:
            self.last_update_time = current_time
            self._report_progress(status)
    
    def _report_progress(self, status: Optional[str] = None) -> None:
        """
        Report current progress.
        
        Args:
            status: Status message
        """
        info = self.get_progress_info()
        
        # Build progress message
        progress_msg = (
            f"{self.description}: {info.percentage:.1f}% "
            f"({self.current_step}/{self.total_steps}) "
            f"[{info.elapsed_time:.1f}s<{info.estimated_remaining:.1f}s, "
            f"{info.steps_per_second:.1f} steps/s]"
        )
        
        if status:
            progress_msg += f" - {status}"
        
        logger.info(progress_msg)
        
        # Execute callbacks
        for callback in self.callbacks:
            try:
                callback(info)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def get_progress_info(self) -> ProgressInfo:
        """
        Get current progress information.
        
        Returns:
            ProgressInfo object
        """
        elapsed = time.time() - self.start_time
        
        percentage = (self.current_step / self.total_steps * 100) if self.total_steps > 0 else 0
        
        # Calculate steps per second
        if len(self.step_times) > 0:
            avg_step_time = sum(self.step_times) / len(self.step_times)
            steps_per_second = 1.0 / avg_step_time if avg_step_time > 0 else 0
        else:
            steps_per_second = self.current_step / elapsed if elapsed > 0 else 0
        
        # Estimate remaining time
        remaining_steps = self.total_steps - self.current_step
        
        if steps_per_second > 0:
            estimated_remaining = remaining_steps / steps_per_second
        elif self.current_step > 0:
            time_per_step = elapsed / self.current_step
            estimated_remaining = remaining_steps * time_per_step
        else:
            estimated_remaining = 0
        
        # Determine status
        if self.current_step >= self.total_steps:
            status = "Complete"
        elif self.current_step == 0:
            status = "Starting"
        else:
            status = "Running"
        
        return ProgressInfo(
            current_step=self.current_step,
            total_steps=self.total_steps,
            percentage=percentage,
            elapsed_time=elapsed,
            estimated_remaining=estimated_remaining,
            steps_per_second=steps_per_second,
            status=status
        )
    
    def start_substep(self, name: str, total_steps: int) -> None:
        """
        Start tracking a substep.
        
        Args:
            name: Substep name
            total_steps: Total steps for substep
        """
        self.substeps[name] = {
            'current': 0,
            'total': total_steps,
            'start_time': time.time()
        }
        
        logger.debug(f"Substep started: {name} ({total_steps} steps)")
    
    def update_substep(self, name: str, steps: int = 1) -> None:
        """
        Update substep progress.
        
        Args:
            name: Substep name
            steps: Steps completed
        """
        if name in self.substeps:
            self.substeps[name]['current'] += steps
    
    def end_substep(self, name: str) -> None:
        """
        End substep tracking.
        
        Args:
            name: Substep name
        """
        if name in self.substeps:
            substep = self.substeps[name]
            elapsed = time.time() - substep['start_time']
            
            logger.debug(
                f"Substep complete: {name} ({substep['current']}/{substep['total']}) "
                f"[{elapsed:.2f}s]"
            )
    
    def add_callback(self, callback: Callable) -> None:
        """
        Add progress callback function.
        
        Args:
            callback: Function to call with ProgressInfo
        """
        self.callbacks.append(callback)
    
    def get_elapsed_time(self) -> float:
        """Get total elapsed time."""
        return time.time() - self.start_time
    
    def get_eta(self) -> float:
        """Get estimated time remaining."""
        info = self.get_progress_info()
        return info.estimated_remaining
    
    def reset(self, new_total: Optional[int] = None) -> None:
        """
        Reset progress tracker.
        
        Args:
            new_total: New total steps (optional)
        """
        if new_total is not None:
            self.total_steps = new_total
        
        self.current_step = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.step_times.clear()
        self.substeps.clear()
        
        logger.debug(f"Progress tracker reset: {self.total_steps} steps")
    
    def format_time(self, seconds: float) -> str:
        """
        Format seconds into human-readable time string.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get progress summary as dictionary.
        
        Returns:
            Progress summary dictionary
        """
        info = self.get_progress_info()
        
        return {
            'description': self.description,
            'current_step': info.current_step,
            'total_steps': info.total_steps,
            'percentage': info.percentage,
            'elapsed_time': info.elapsed_time,
            'elapsed_time_formatted': self.format_time(info.elapsed_time),
            'estimated_remaining': info.estimated_remaining,
            'estimated_remaining_formatted': self.format_time(info.estimated_remaining),
            'steps_per_second': info.steps_per_second,
            'status': info.status,
            'substeps': {
                name: {
                    'progress': f"{data['current']}/{data['total']}",
                    'percentage': (data['current'] / data['total'] * 100) if data['total'] > 0 else 0
                }
                for name, data in self.substeps.items()
            }
        }