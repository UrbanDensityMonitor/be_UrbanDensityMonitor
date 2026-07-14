"""
Video reading and frame extraction utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
from dataclasses import dataclass

from ..logger import logger


@dataclass
class VideoMetadata:
    """Container for video metadata."""
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    filepath: Path


class VideoReader:
    """
    Handles video file reading and frame extraction with caching.
    """
    
    def __init__(self, video_path: Union[str, Path]) -> None:
        """
        Initialize video reader.
        
        Args:
            video_path: Path to video file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video cannot be opened
        """
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        self.metadata = self._extract_metadata()
        self._frame_cache: dict = {}
        logger.info(f"Video loaded: {self.metadata}")
    
    def _extract_metadata(self) -> VideoMetadata:
        """Extract metadata from video file."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps > 0:
            duration = total_frames / fps
        else:
            duration = 0.0
            fps = 30.0  # Default fallback
        
        codec = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])
        
        return VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec_str,
            filepath=self.video_path
        )
    
    def read_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """
        Read specific frame by number.
        
        Args:
            frame_number: Frame index (0-based)
            
        Returns:
            Frame as numpy array or None if failed
        """
        if frame_number in self._frame_cache:
            return self._frame_cache[frame_number]
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        if ret:
            self._frame_cache[frame_number] = frame
            return frame
        
        logger.warning(f"Failed to read frame {frame_number}")
        return None
    
    def stream_frames(self, 
                     start_frame: int = 0, 
                     end_frame: Optional[int] = None,
                     step: int = 1) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Stream frames from video as generator.
        
        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index (None for all)
            step: Frame step size
            
        Yields:
            Tuple of (frame_number, frame_array)
        """
        if end_frame is None:
            end_frame = self.metadata.total_frames
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_number = start_frame
        while frame_number < end_frame:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            if (frame_number - start_frame) % step == 0:
                yield frame_number, frame
            
            frame_number += 1
    
    def get_frame_batch(self, 
                       frame_numbers: list[int]) -> list[Optional[np.ndarray]]:
        """
        Get multiple frames efficiently.
        
        Args:
            frame_numbers: List of frame indices
            
        Returns:
            List of frames (None for failed reads)
        """
        frames = []
        for fn in frame_numbers:
            frames.append(self.read_frame(fn))
        return frames
    
    def resize_frame(self, 
                    frame: np.ndarray, 
                    width: Optional[int] = None,
                    height: Optional[int] = None) -> np.ndarray:
        """
        Resize frame while maintaining aspect ratio.
        
        Args:
            frame: Input frame
            width: Target width (if None, maintain aspect ratio with height)
            height: Target height (if None, maintain aspect ratio with width)
            
        Returns:
            Resized frame
        """
        if width is None and height is None:
            return frame
        
        h, w = frame.shape[:2]
        
        if width is None:
            assert height is not None
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            assert width is not None
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
        
        return cv2.resize(frame, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    
    def extract_frames_to_directory(self, 
                                    output_dir: Union[str, Path],
                                    frame_interval: int = 30) -> list[Path]:
        """
        Extract frames to directory at specified interval.
        
        Args:
            output_dir: Directory to save frames
            frame_interval: Extract every Nth frame
            
        Returns:
            List of paths to saved frames
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for frame_number, frame in self.stream_frames(step=frame_interval):
            filename = f"frame_{frame_number:06d}.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_paths.append(filepath)
        
        logger.info(f"Extracted {len(saved_paths)} frames to {output_dir}")
        return saved_paths
    
    def get_video_writer(self, 
                        output_path: Union[str, Path],
                        fps: Optional[float] = None,
                        codec: str = "mp4v") -> cv2.VideoWriter:
        """
        Create VideoWriter for saving processed video.
        
        Args:
            output_path: Path for output video
            fps: Frames per second (default: same as input)
            codec: FourCC codec string
            
        Returns:
            OpenCV VideoWriter object
        """
        if fps is None:
            fps = self.metadata.fps
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter.fourcc(*codec)
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (self.metadata.width, self.metadata.height)
        )
        
        return writer
    
    def __del__(self) -> None:
        """Cleanup video capture."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
    
    def __enter__(self) -> 'VideoReader':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.__del__()