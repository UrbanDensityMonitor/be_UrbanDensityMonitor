"""
Frame encoding utilities untuk convert OpenCV frames ke berbagai format
"""
import base64
import logging
from typing import Literal
import cv2
import numpy as np

logger = logging.getLogger(__name__)

FormatType = Literal['jpeg', 'png']


class FrameEncoder:
    """
    Frame encoder untuk convert OpenCV frames
    
    Features:
        - Base64 encoding
        - Data URL generation
        - Configurable compression
        - Multiple format support (JPEG, PNG)
    
    Usage:
        encoder = FrameEncoder(jpeg_quality=60)
        b64_str = encoder.encode_to_base64(frame)
        data_url = encoder.encode_to_data_url(frame)
    """
    
    def __init__(self, jpeg_quality: int = 60):
        """
        Args:
            jpeg_quality: JPEG compression quality (0-100)
                         60 = Good balance between size & quality
                         90 = High quality (larger size)
                         40 = Low quality (smaller size)
        """
        if not 0 <= jpeg_quality <= 100:
            raise ValueError("JPEG quality must be between 0 and 100")
        
        self.jpeg_quality = jpeg_quality
    
    def encode_to_base64(
        self, 
        frame: np.ndarray,
        format: FormatType = 'jpeg'
    ) -> str:
        """
        Encode frame to base64 string
        
        Args:
            frame: OpenCV frame (numpy array)
            format: 'jpeg' or 'png'
        
        Returns:
            Base64 encoded string
        
        Raises:
            ValueError: Jika encoding gagal
        """
        # Prepare encoding parameters
        if format == 'jpeg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            ext = '.jpg'
        elif format == 'png':
            # PNG is lossless, no quality parameter needed
            encode_params = []
            ext = '.png'
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Encode frame
        success, buffer = cv2.imencode(ext, frame, encode_params)
        
        if not success:
            raise ValueError(f"Failed to encode frame to {format}")
        
        # Convert to base64
        b64_bytes = base64.b64encode(buffer)
        b64_str = b64_bytes.decode('utf-8')
        
        return b64_str
    
    def encode_to_data_url(
        self, 
        frame: np.ndarray,
        format: FormatType = 'jpeg'
    ) -> str:
        """
        Encode frame to data URL untuk direct HTML usage
        
        Args:
            frame: OpenCV frame (numpy array)
            format: 'jpeg' or 'png'
        
        Returns:
            Data URL string (e.g., "data:image/jpeg;base64,...")
        
        Example:
            data_url = encoder.encode_to_data_url(frame)
            # Can be used directly in HTML:
            # <img src="{data_url}" />
        """
        b64_str = self.encode_to_base64(frame, format)
        mime_type = f'image/{format}'
        return f"data:{mime_type};base64,{b64_str}"
    
    def estimate_size(self, frame: np.ndarray, format: FormatType = 'jpeg') -> int:
        """
        Estimate encoded size dalam bytes
        
        Args:
            frame: OpenCV frame
            format: Encoding format
        
        Returns:
            Estimated size in bytes
        """
        try:
            b64_str = self.encode_to_base64(frame, format)
            # Base64 adds ~33% overhead
            return len(b64_str.encode('utf-8'))
        except Exception as e:
            logger.error(f"Failed to estimate size: {e}")
            return 0
    
    @staticmethod
    def decode_from_base64(b64_str: str) -> np.ndarray:
        """
        Decode base64 string back to OpenCV frame
        
        Args:
            b64_str: Base64 encoded image string
        
        Returns:
            OpenCV frame (numpy array)
        """
        # Decode base64
        img_bytes = base64.b64decode(b64_str)
        
        # Convert to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise ValueError("Failed to decode image from base64")
        
        return frame
