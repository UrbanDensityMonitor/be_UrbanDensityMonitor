"""
Video output rendering with tracking, ROI, and traffic state overlays.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from tqdm import tqdm

from ..logger import logger
from ..config import VisualizationConfig
from ..detection.detector import Detection
from ..tracking.track_manager import Track
from ..roi.roi_manager import ROIManager


class VideoRenderer:
    """
    Renders processed video with all annotations and overlays.
    """
    
    def __init__(self, config: VisualizationConfig) -> None:
        """
        Initialize video renderer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config
        
        # Color scheme for different vehicle classes
        self.class_colors = {
            'car': (0, 255, 0),        # Green
            'bus': (255, 0, 0),        # Blue
            'truck': (0, 0, 255),      # Red
            'motorcycle': (255, 255, 0), # Cyan
            'bicycle': (255, 0, 255)    # Magenta
        }
        
        # Traffic state colors
        self.state_colors = {
            'free_flow': (0, 255, 0),
            'normal': (0, 200, 0),
            'moderate': (0, 255, 255),
            'slow': (0, 165, 255),
            'heavy': (0, 0, 255),
            'congested': (0, 0, 200)
        }
    
    def draw_detections(self,
                       frame: np.ndarray,
                       detections: List[Detection],
                       show_confidence: bool = True) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            show_confidence: Whether to show confidence scores
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = self.class_colors.get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = det.class_name
            if det.track_id is not None:
                label = f"ID:{det.track_id} {label}"
            if show_confidence:
                label += f" {det.confidence:.2f}"
            
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated
    
    def draw_tracks(self,
                   frame: np.ndarray,
                   tracks: List[Track],
                   show_trail: bool = True,
                   trail_length: int = 20) -> np.ndarray:
        """
        Draw tracking information on frame.
        
        Args:
            frame: Input frame
            tracks: List of active tracks
            show_trail: Whether to show trajectory trails
            trail_length: Length of trail to show
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for track in tracks:
            if not track.is_active:
                continue
            
            color = self.class_colors.get(track.class_name, (255, 255, 255))
            
            # Draw current bounding box
            if track.bboxes:
                bbox = track.bboxes[-1]
                cv2.rectangle(annotated, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            color, 2)
            
            # Draw track ID and class name
            if track.positions:
                pos = track.positions[-1]
                cv2.putText(annotated, f"ID:{track.track_id} {track.class_name}",
                          (int(pos[0]) - 10, int(pos[1]) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # Draw trajectory trail
            if show_trail and len(track.positions) > 1:
                trail = track.positions[-trail_length:]
                for i in range(1, len(trail)):
                    pt1 = (int(trail[i-1][0]), int(trail[i-1][1]))
                    pt2 = (int(trail[i][0]), int(trail[i][1]))
                    # Fade color based on age
                    alpha = i / len(trail)
                    faded_color = tuple(int(c * alpha) for c in color)
                    cv2.line(annotated, pt1, pt2, faded_color, 1)
        
        return annotated
    
    def draw_roi_overlay(self,
                        frame: np.ndarray,
                        roi_manager: ROIManager) -> np.ndarray:
        """
        Draw ROI polygon overlay.
        
        Args:
            frame: Input frame
            roi_manager: ROI manager instance
            
        Returns:
            Annotated frame
        """
        return roi_manager.draw_roi_on_frame(
            frame, 
            color=(0, 255, 0), 
            thickness=3
        )
    
    def draw_info_overlay(self,
                         frame: np.ndarray,
                         info: Dict[str, Any],
                         position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw information overlay panel.
        
        Args:
            frame: Input frame
            info: Dictionary of information to display
            position: Starting position for text
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        y_offset = position[1]
        
        # Semi-transparent background
        overlay = annotated.copy()
        panel_height = len(info) * 25 + 20
        cv2.rectangle(overlay, 
                     (position[0] - 5, position[1] - 25),
                     (position[0] + 300, position[1] + panel_height),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, annotated, 0.7, 0, annotated)
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(annotated, text, (position[0], y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return annotated
    
    def draw_traffic_state(self,
                          frame: np.ndarray,
                          state: str,
                          metrics: Dict[str, float]) -> np.ndarray:
        """
        Draw traffic state banner on frame.
        
        Args:
            frame: Input frame
            state: Current traffic state
            metrics: Traffic metrics dictionary
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        height, width = frame.shape[:2]
        
        # State banner at top
        state_color = self.state_colors.get(state.lower().replace(' ', '_'), (128, 128, 128))
        
        cv2.rectangle(annotated, (0, 0), (width, 80), (0, 0, 0), -1)
        cv2.addWeighted(
            cv2.rectangle(annotated.copy(), (0, 0), (width, 80), state_color, -1),
            0.4, annotated, 0.6, 0, annotated
        )
        
        # State label
        cv2.putText(annotated, f"Traffic State: {state.upper()}",
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Metrics
        metric_text = " | ".join([f"{k}: {v:.1f}" for k, v in metrics.items()])
        cv2.putText(annotated, metric_text,
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def create_output_video(self,
                           input_video_path: Path,
                           output_path: Path,
                           frame_processor: Callable,
                           roi_manager: ROIManager,
                           total_frames: Optional[int] = None) -> None:
        """
        Create annotated output video.
        
        Args:
            input_video_path: Path to input video
            output_path: Path for output video
            frame_processor: Function that processes each frame
            roi_manager: ROI manager for overlay
            total_frames: Total frames to process
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(str(input_video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames is None:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter.fourcc(*self.config.video_codec)
        writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (width, height)
        )
        
        logger.info(f"Creating output video: {output_path}")
        
        with tqdm(total=total_frames, desc="Rendering video", unit="frame") as pbar:
            frame_count = 0
            while frame_count < total_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                processed_frame = frame_processor(frame, frame_count)
                
                # Draw ROI
                processed_frame = self.draw_roi_overlay(processed_frame, roi_manager)
                
                # Write frame
                writer.write(processed_frame)
                
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        writer.release()
        logger.info(f"Video saved: {output_path}")