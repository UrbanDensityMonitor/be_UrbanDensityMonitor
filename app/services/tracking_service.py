"""
Tracking Service untuk object tracking & feature extraction
Encapsulates ByteTrack tracking dan feature extraction logic
"""
import logging
from typing import List
from app.services.core_ml.config import TrackingConfig, FeatureExtractionConfig
from app.services.core_ml.tracking.byte_tracker import ByteTrackTracker
from app.services.core_ml.features.feature_extractor import FeatureExtractor
from app.services.core_ml.roi.roi_manager import ROIManager
from app.services.core_ml.detection.detector import Detection
from app.services.density_service import FrameFeatures
import numpy as np
import time

logger = logging.getLogger(__name__)


class TrackingService:
    """
    Service untuk tracking objects dan extract features
    
    Features:
        - ByteTrack multi-object tracking
        - ROI (Region of Interest) management
        - Feature extraction (speed, occupancy, congestion)
    
    Usage:
        service = TrackingService()
        features = await service.update_and_extract(detections, frame, frame_number)
    """
    
    def __init__(
        self,
        tracking_config: TrackingConfig = None,
        feature_config: FeatureExtractionConfig = None
    ):
        """
        Initialize tracking service
        
        Args:
            tracking_config: Configuration untuk ByteTrack
            feature_config: Configuration untuk feature extraction
        """
        # Use default configs if not provided
        self.tracking_config = tracking_config or TrackingConfig()
        self.feature_config = feature_config or FeatureExtractionConfig()
        
        # Initialize components
        self.tracker = ByteTrackTracker(self.tracking_config)
        self.roi_manager = ROIManager()
        self.feature_extractor = FeatureExtractor(
            self.feature_config,
            self.roi_manager
        )
        
        logger.info("✅ Tracking service initialized")
    
    async def update_and_extract(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_number: int
    ) -> FrameFeatures:
        """
        Update tracker dengan detections dan extract features
        
        Args:
            detections: List of Detection objects dari YOLO
            frame: Current frame (numpy array)
            frame_number: Frame number untuk tracking
        
        Returns:
            FrameFeatures object dengan extracted features
        """
        timestamp = time.time()
        
        # Update tracker
        track_result = self.tracker.update(detections, frame, frame_number)
        
        # Extract features dari tracked objects
        features = self.feature_extractor.extract_frame_features(
            track_result.tracks,
            frame_number,
            timestamp
        )
        
        # Convert ke FrameFeatures format untuk DensityService
        frame_features = FrameFeatures(
            vehicle_count=features.vehicle_count,
            average_speed=features.average_speed,
            road_occupancy=features.road_occupancy,
            congestion_index=features.congestion_index
        )
        
        return frame_features
    
    def reset(self):
        """
        Reset tracker state
        Useful ketika ganti stream atau restart tracking
        """
        self.tracker = ByteTrackTracker(self.tracking_config)
        logger.info("🔄 Tracker reset")
    
    def get_active_tracks_count(self) -> int:
        """
        Get number of active tracks
        
        Returns:
            Number of currently tracked objects
        """
        return len(self.tracker.tracked_stracks) if hasattr(self.tracker, 'tracked_stracks') else 0
