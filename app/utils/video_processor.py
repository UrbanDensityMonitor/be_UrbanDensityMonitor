"""
Video processing utilities menggunakan OpenCV
Handles video capture, frame streaming, and reconnection logic
"""
import asyncio
import time
import logging
from typing import AsyncGenerator, Optional
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processor untuk handle OpenCV video capture dengan auto-reconnect
    
    Usage:
        processor = VideoProcessor()
        async for frame in processor.stream_frames(stream_url):
            # Process frame
            pass
    """
    
    def __init__(
        self, 
        buffer_size: int = 1,
        reconnect_delay: float = 1.0,
        default_fps: float = 30.0
    ):
        """
        Args:
            buffer_size: OpenCV buffer size (1 = minimize latency)
            reconnect_delay: Delay sebelum reconnect (seconds)
            default_fps: Default FPS jika tidak bisa detect dari stream
        """
        self.buffer_size = buffer_size
        self.reconnect_delay = reconnect_delay
        self.default_fps = default_fps
    
    async def stream_frames(
        self, 
        stream_url: str,
        skip_rate: int = 2
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Async generator yang yield frames dari video stream
        
        Args:
            stream_url: URL stream CCTV (HLS/RTSP/HTTP)
            skip_rate: Process setiap N frame (2 = skip 1 frame)
        
        Yields:
            numpy.ndarray: Frame dari video
        
        Features:
            - Auto reconnect jika connection lost
            - Maintain playback speed (1.0x)
            - Skip frames untuk reduce processing load
        """
        cap = None
        frame_count = 0
        
        try:
            cap = self._create_capture(stream_url)
            video_fps = self._get_fps(cap)
            frame_duration = 1.0 / video_fps
            
            logger.info(f"🎥 Video stream started: {stream_url} @ {video_fps} FPS")
            
            while True:
                # Check if capture is still open
                if not cap or not cap.isOpened():
                    logger.warning(f"🔄 Reconnecting to {stream_url}...")
                    if cap:
                        cap.release()
                    cap = self._create_capture(stream_url)
                    await asyncio.sleep(self.reconnect_delay)
                    continue
                
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    logger.warning("⚠️ Frame kosong, reconnecting...")
                    cap.release()
                    await asyncio.sleep(self.reconnect_delay)
                    continue
                
                frame_count += 1
                
                # Skip frames based on skip_rate
                if frame_count % skip_rate == 0:
                    yield frame
                
                # Maintain playback speed (prevent too fast processing)
                elapsed = time.time() - start_time
                sleep_time = frame_duration - elapsed
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"❌ Video processing error: {e}", exc_info=True)
            raise
        finally:
            if cap:
                cap.release()
                logger.info("🛑 Video stream stopped")
    
    def _create_capture(self, url: str) -> cv2.VideoCapture:
        """
        Create OpenCV VideoCapture object
        
        Args:
            url: Stream URL
        
        Returns:
            cv2.VideoCapture object
        """
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        return cap
    
    def _get_fps(self, cap: cv2.VideoCapture) -> float:
        """
        Get FPS dari video capture dengan validation
        
        Args:
            cap: VideoCapture object
        
        Returns:
            FPS value (validated)
        """
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Validate FPS value
        # HLS streams sometimes return weird values (0 or 90000)
        if not fps or fps <= 0 or fps > 120:
            logger.warning(f"⚠️ Invalid FPS detected: {fps}, using default: {self.default_fps}")
            return self.default_fps
        
        return fps
    
    async def stream_frames_with_resize(
        self,
        stream_url: str,
        resize_to: tuple[int, int],
        skip_rate: int = 2
    ) -> AsyncGenerator[np.ndarray, None]:
        """
        Stream frames dengan auto resize
        
        Args:
            stream_url: URL stream
            resize_to: Target size (width, height)
            skip_rate: Frame skip rate
        
        Yields:
            Resized frames
        """
        async for frame in self.stream_frames(stream_url, skip_rate):
            resized = cv2.resize(frame, resize_to)
            yield resized
