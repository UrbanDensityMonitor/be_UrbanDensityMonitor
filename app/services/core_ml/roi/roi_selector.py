"""
Interactive ROI selection tool using OpenCV mouse callbacks.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable

from ..logger import logger
from ..utils.video_reader import VideoReader


class ROISelector:
    """
    Interactive tool for drawing polygon ROIs on video frames.
    Supports multiple ROI creation with visual feedback.
    """
    
    def __init__(self, window_name: str = "ROI Selector") -> None:
        """
        Initialize ROI selector.
        
        Args:
            window_name: OpenCV window name
        """
        self.window_name = window_name
        self.points: List[Tuple[int, int]] = []
        self.current_point: Optional[Tuple[int, int]] = None
        self.rois: List[List[Tuple[int, int]]] = []
        self.frame: Optional[np.ndarray] = None
        self.original_frame: Optional[np.ndarray] = None
        self.drawing_complete = False
        
        # Colors
        self.current_color = (0, 255, 0)  # Green for current
        self.completed_color = (0, 255, 255)  # Yellow for completed
        self.point_color = (0, 0, 255)  # Red for points
        
    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: any) -> None:
        """
        Mouse callback for drawing polygon.
        
        Args:
            event: OpenCV mouse event
            x: X coordinate
            y: Y coordinate
            flags: Event flags
            param: Additional parameters
        """
        if event == cv2.EVENT_MOUSEMOVE:
            self.current_point = (x, y)
            self._update_display()
        
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            logger.debug(f"Point added: ({x}, {y})")
            self._update_display()
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.points) > 2:
                # Complete polygon
                self.rois.append(self.points.copy())
                logger.info(f"ROI completed with {len(self.points)} points")
                self.points.clear()
                self.drawing_complete = True
                self._update_display()
    
    def _update_display(self) -> None:
        """Update the display with current drawings."""
        if self.original_frame is None:
            return
        
        self.frame = self.original_frame.copy()
        
        # Draw completed ROIs
        for roi_points in self.rois:
            if len(roi_points) > 2:
                pts = np.array(roi_points, np.int32)
                cv2.polylines(self.frame, [pts], True, self.completed_color, 2)
                
                # Fill with semi-transparent color
                overlay = self.frame.copy()
                cv2.fillPoly(overlay, [pts], self.completed_color)
                cv2.addWeighted(overlay, 0.3, self.frame, 0.7, 0, self.frame)
                
                # Label
                cv2.putText(
                    self.frame, f"ROI {len(self.rois)}",
                    roi_points[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
        
        # Draw current polygon in progress
        if len(self.points) > 1:
            pts = np.array(self.points, np.int32)
            cv2.polylines(self.frame, [pts], False, self.current_color, 2)
            
            # Draw line to current mouse position
            if self.current_point:
                cv2.line(
                    self.frame,
                    self.points[-1],
                    self.current_point,
                    self.current_color, 1
                )
        
        # Draw points
        for point in self.points:
            cv2.circle(self.frame, point, 5, self.point_color, -1)
        
        # Draw current mouse position
        if self.current_point:
            cv2.circle(self.frame, self.current_point, 3, (255, 255, 255), -1)
        
        # Instructions overlay
        self._draw_instructions()
        
        cv2.imshow(self.window_name, self.frame)
    
    def _draw_instructions(self) -> None:
        """Draw instruction text on frame."""
        instructions = [
            "LEFT CLICK: Add point",
            "RIGHT CLICK: Complete ROI",
            "ENTER: Save & Exit",
            "C: Clear current polygon",
            "R: Reset all ROIs",
            "Q: Quit without saving"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                self.frame, instruction,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA
            )
            y_offset += 25
    
    def select_roi_from_frame(self, 
                              frame: np.ndarray,
                              window_title: Optional[str] = None) -> List[List[Tuple[int, int]]]:
        """
        Open interactive ROI selector on a single frame.
        
        Args:
            frame: Input frame for ROI selection
            window_title: Custom window title (optional)
            
        Returns:
            List of ROIs, each ROI is list of (x,y) points
        """
        if window_title:
            self.window_name = window_title
        
        self.original_frame = frame.copy()
        self.frame = frame.copy()
        self.points = []
        self.rois = []
        self.drawing_complete = False
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        logger.info("ROI Selector opened. Draw polygons and press ENTER to save.")
        
        self._update_display()
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # ENTER
                logger.info(f"Saving {len(self.rois)} ROIs")
                break
            elif key == ord('c'):
                self.points.clear()
                logger.debug("Current polygon cleared")
                self._update_display()
            elif key == ord('r'):
                self.points.clear()
                self.rois.clear()
                logger.debug("All ROIs reset")
                self._update_display()
            elif key == ord('q'):
                self.rois.clear()
                logger.info("ROI selection cancelled")
                break
        
        cv2.destroyWindow(self.window_name)
        
        return self.rois
    
    def select_roi_from_video(self, 
                              video_path: Path,
                              frame_number: int = 0,
                              window_title: Optional[str] = None) -> List[List[Tuple[int, int]]]:
        """
        Open ROI selector on a specific frame from video.
        
        Args:
            video_path: Path to video file
            frame_number: Frame number to use for selection
            window_title: Custom window title (optional)
            
        Returns:
            List of ROIs
        """
        logger.info(f"Opening video for ROI selection: {video_path}")
        
        with VideoReader(video_path) as video_reader:
            frame = video_reader.read_frame(frame_number)
            
            if frame is None:
                logger.warning(f"Could not read frame {frame_number}, using frame 0")
                frame = video_reader.read_frame(0)
            
            if frame is None:
                raise ValueError("Could not read any frame from video")
            
            return self.select_roi_from_frame(frame, window_title)
    
    @staticmethod
    def validate_roi(roi_points: List[Tuple[int, int]]) -> bool:
        """
        Validate ROI polygon.
        
        Args:
            roi_points: List of polygon vertices
            
        Returns:
            True if valid polygon
        """
        if len(roi_points) < 3:
            return False
        
        # Check for self-intersection
        from shapely.geometry import Polygon
        try:
            polygon = Polygon(roi_points)
            return polygon.is_valid and not polygon.is_empty
        except Exception:
            return False