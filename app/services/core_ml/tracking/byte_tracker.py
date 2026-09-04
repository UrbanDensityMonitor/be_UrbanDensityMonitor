"""
ByteTrack integration with custom track management.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import cv2

from .._logging import logger
from ..config import TrackingConfig
from ..detection.detector import Detection
from .track_manager import TrackManager, Track


@dataclass
class TrackResult:
    """Container for tracking result per frame."""
    frame_number: int
    tracks: List[Track]
    active_detections: List[Detection]


class ByteTrackTracker:
    """
    ByteTrack-based multi-object tracker for vehicles.
    Implements tracking using ByteTrack algorithm with custom management.
    """
    
    def __init__(self, config: TrackingConfig) -> None:
        """
        Initialize ByteTrack tracker.
        
        Args:
            config: Tracking configuration
        """
        self.config = config
        self.track_manager = TrackManager(
            max_missed_frames=config.track_buffer,
            min_track_length=5
        )
        self.frame_count = 0
        
        # ByteTrack parameters
        self.track_thresh = config.track_thresh
        self.match_thresh = config.match_thresh
        self.track_buffer = config.track_buffer
        self.frame_rate = config.frame_rate
        
        # IOU matching threshold
        # If match_thresh > 0.5, treat it as a cost threshold (Cost = 1 - IoU <= match_thresh => IoU >= 1 - match_thresh)
        self.iou_threshold = 1.0 - config.match_thresh if config.match_thresh > 0.5 else config.match_thresh
        
        logger.info(f"ByteTrack tracker initialized with thresh={config.track_thresh}")
    
    def update(self, 
               detections: List[Detection], 
               frame: Optional[np.ndarray] = None,
               frame_number: Optional[int] = None) -> TrackResult:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of current detections
            frame: Current frame (optional, for visualization)
            frame_number: Frame number (auto-increment if None)
            
        Returns:
            TrackResult with current tracks and detections
        """
        if frame_number is not None:
            self.frame_count = frame_number
        else:
            self.frame_count += 1
        
        # Split detections into high and low confidence
        high_conf_dets = [d for d in detections if d.confidence >= self.track_thresh]
        low_conf_dets = [d for d in detections if d.confidence < self.track_thresh]
        
        # Get active tracks
        active_tracks = [self.track_manager.tracks[tid] 
                        for tid in self.track_manager.active_tracks]
        
        # First association: high confidence detections to active tracks
        matched, unmatched_dets, unmatched_tracks = self._associate_detections_to_tracks(
            high_conf_dets, active_tracks
        )
        
        # Update matched tracks
        for track, detection in matched:
            velocity = self._estimate_velocity(track, detection)
            self.track_manager.update_track(
                track.track_id, 
                detection, 
                self.frame_count,
                velocity
            )
        
        # Second association: remaining tracks with low confidence detections
        if low_conf_dets and unmatched_tracks:
            matched_low, _, _ = self._associate_detections_to_tracks(
                low_conf_dets, unmatched_tracks
            )
            
            for track, detection in matched_low:
                velocity = self._estimate_velocity(track, detection)
                self.track_manager.update_track(
                    track.track_id,
                    detection,
                    self.frame_count,
                    velocity
                )
        
        # Create new tracks for unmatched high-confidence detections
        for detection in unmatched_dets:
            if detection.confidence >= self.track_thresh + 0.1:  # Higher threshold for new tracks
                self.track_manager.create_track(detection, self.frame_count)
        
        # Deactivate stale tracks
        self.track_manager.deactivate_stale_tracks(self.frame_count)
        
        # Remove very old tracks
        self.track_manager.remove_old_tracks(self.frame_count)
        
        # Assign track IDs to detections
        self._assign_track_ids_to_detections(detections)
        
        current_tracks = [self.track_manager.tracks[tid] 
                         for tid in self.track_manager.active_tracks]
        
        return TrackResult(
            frame_number=self.frame_count,
            tracks=current_tracks,
            active_detections=detections
        )
    
    def _associate_detections_to_tracks(self,
                                       detections: List[Detection],
                                       tracks: List[Track]) -> Tuple[List, List, List]:
        """
        Associate detections to tracks using IOU matching.
        
        Args:
            detections: List of detections
            tracks: List of tracks
            
        Returns:
            Tuple of (matched pairs, unmatched detections, unmatched tracks)
        """
        if not tracks or not detections:
            return [], detections, tracks
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, det in enumerate(detections):
            for j, track in enumerate(tracks):
                if track.bboxes:
                    iou_matrix[i, j] = self._compute_iou(
                        det.bbox, 
                        track.bboxes[-1]
                    )
        
        # Greedy matching sorted by IOU (highest first)
        matched_pairs = []
        matched_det_indices = set()
        matched_track_indices = set()
        
        # Get all detection-track pairs with IoU >= threshold
        pairs = []
        for i in range(len(detections)):
            for j in range(len(tracks)):
                iou = iou_matrix[i, j]
                if iou >= self.iou_threshold:
                    pairs.append((iou, i, j))
        
        # Sort pairs by IoU in descending order
        pairs.sort(key=lambda x: x[0], reverse=True)
        
        for iou, i, j in pairs:
            if i in matched_det_indices or j in matched_track_indices:
                continue
            matched_pairs.append((tracks[j], detections[i]))
            matched_det_indices.add(i)
            matched_track_indices.add(j)
        
        unmatched_dets = [d for i, d in enumerate(detections) 
                         if i not in matched_det_indices]
        unmatched_tracks = [t for j, t in enumerate(tracks) 
                           if j not in matched_track_indices]
        
        return matched_pairs, unmatched_dets, unmatched_tracks
    
    def _compute_iou(self, 
                     bbox1: Tuple[int, int, int, int],
                     bbox2: Tuple[int, int, int, int]) -> float:
        """
        Compute Intersection over Union between two bounding boxes.
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            IoU value between 0 and 1
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _estimate_velocity(self, 
                          track: Track, 
                          new_detection: Detection) -> float:
        """
        Estimate velocity of a track in pixels per frame.
        
        Args:
            track: Existing track
            new_detection: New detection
            
        Returns:
            Estimated velocity magnitude
        """
        if not track.positions:
            return 0.0
        
        last_pos = track.positions[-1]
        new_pos = new_detection.center
        
        displacement = np.sqrt(
            (new_pos[0] - last_pos[0])**2 + 
            (new_pos[1] - last_pos[1])**2
        )
        
        # Convert to speed (pixels per frame * fps for real speed)
        velocity = displacement * self.frame_rate
        
        return velocity
    
    def _assign_track_ids_to_detections(self, detections: List[Detection]) -> None:
        """
        Assign track IDs to detections based on current tracks.
        
        Args:
            detections: List of detections to update
        """
        for detection in detections:
            # Find matching track
            for track_id in self.track_manager.active_tracks:
                track = self.track_manager.tracks[track_id]
                if track.bboxes:
                    iou = self._compute_iou(detection.bbox, track.bboxes[-1])
                    if iou >= self.iou_threshold:
                        detection.track_id = track.track_id
                        break
    
    def get_all_tracks(self) -> List[Track]:
        """
        Get all tracked vehicles.
        
        Returns:
            List of all tracks
        """
        return list(self.track_manager.tracks.values())
    
    def get_valid_tracks(self) -> List[Track]:
        """
        Get tracks with sufficient length.
        
        Returns:
            List of valid tracks
        """
        return self.track_manager.get_valid_tracks()
    
    def get_statistics(self) -> dict:
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        return self.track_manager.get_track_statistics()
    
    def reset(self) -> None:
        """Reset tracker state."""
        self.track_manager.clear()
        self.frame_count = 0
        logger.info("Tracker reset")