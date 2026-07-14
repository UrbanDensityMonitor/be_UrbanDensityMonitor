"""
Track state management and history maintenance.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from filterpy.kalman import KalmanFilter

from ..logger import logger
from ..detection.detector import Detection


@dataclass
class Track:
    """
    Represents a tracked vehicle with history.
    """
    track_id: int
    class_id: int
    class_name: str
    start_frame: int
    last_frame: int
    positions: List[Tuple[float, float]] = field(default_factory=list)
    bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    velocities: List[float] = field(default_factory=list)
    is_active: bool = True
    
    @property
    def age(self) -> int:
        """Get track age in frames."""
        return self.last_frame - self.start_frame
    
    @property
    def length(self) -> int:
        """Get number of frames track was detected."""
        return len(self.positions)
    
    @property
    def average_confidence(self) -> float:
        """Get average detection confidence."""
        if not self.confidences:
            return 0.0
        return np.mean(self.confidences)
    
    @property
    def total_distance(self) -> float:
        """Calculate total distance traveled in pixels."""
        if len(self.positions) < 2:
            return 0.0
        
        distance = 0.0
        for i in range(1, len(self.positions)):
            p1 = self.positions[i-1]
            p2 = self.positions[i]
            distance += np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        return distance
    
    def add_detection(self, 
                      detection: Detection, 
                      frame_number: int,
                      velocity: Optional[float] = None) -> None:
        """
        Add a new detection to the track.
        
        Args:
            detection: Detection object
            frame_number: Current frame number
            velocity: Estimated velocity (optional)
        """
        self.bboxes.append(detection.bbox)
        self.positions.append(detection.center)
        self.confidences.append(detection.confidence)
        self.last_frame = frame_number
        
        if velocity is not None:
            self.velocities.append(velocity)
    
    def predict_position(self) -> Tuple[float, float]:
        """
        Predict next position based on velocity.
        
        Returns:
            Predicted (x, y) position
        """
        if len(self.positions) < 2:
            return self.positions[-1] if self.positions else (0, 0)
        
        # Simple linear extrapolation
        p1 = np.array(self.positions[-2])
        p2 = np.array(self.positions[-1])
        velocity = p2 - p1
        predicted = p2 + velocity
        
        return tuple(predicted.tolist())


class TrackManager:
    """
    Manages multiple tracks and their lifecycle.
    """
    
    def __init__(self, 
                 max_missed_frames: int = 30,
                 min_track_length: int = 5,
                 max_track_age: Optional[int] = None) -> None:
        """
        Initialize track manager.
        
        Args:
            max_missed_frames: Maximum frames before deactivating track
            min_track_length: Minimum frames to consider valid track
            max_track_age: Maximum track age in frames (optional)
        """
        self.tracks: Dict[int, Track] = {}
        self.max_missed_frames = max_missed_frames
        self.min_track_length = min_track_length
        self.max_track_age = max_track_age
        self.next_track_id = 1
        self.active_tracks: set = set()
        
    def create_track(self, 
                     detection: Detection, 
                     frame_number: int) -> Track:
        """
        Create a new track from detection.
        
        Args:
            detection: Initial detection
            frame_number: Current frame number
            
        Returns:
            New Track object
        """
        track = Track(
            track_id=self.next_track_id,
            class_id=detection.class_id,
            class_name=detection.class_name,
            start_frame=frame_number,
            last_frame=frame_number
        )
        track.add_detection(detection, frame_number)
        
        self.tracks[self.next_track_id] = track
        self.active_tracks.add(self.next_track_id)
        
        self.next_track_id += 1
        
        return track
    
    def update_track(self, 
                     track_id: int, 
                     detection: Detection,
                     frame_number: int,
                     velocity: Optional[float] = None) -> None:
        """
        Update existing track with new detection.
        
        Args:
            track_id: Track identifier
            detection: New detection
            frame_number: Current frame number
            velocity: Estimated velocity (optional)
        """
        if track_id in self.tracks:
            track = self.tracks[track_id]
            track.add_detection(detection, frame_number, velocity)
            self.active_tracks.add(track_id)
    
    def deactivate_stale_tracks(self, current_frame: int) -> List[Track]:
        """
        Deactivate tracks that haven't been seen.
        
        Args:
            current_frame: Current frame number
            
        Returns:
            List of deactivated tracks
        """
        deactivated = []
        
        for track_id in list(self.active_tracks):
            track = self.tracks[track_id]
            frames_since_update = current_frame - track.last_frame
            
            if frames_since_update > self.max_missed_frames:
                track.is_active = False
                self.active_tracks.discard(track_id)
                deactivated.append(track)
                logger.debug(f"Track {track_id} deactivated (last seen: {frames_since_update} frames ago)")
        
        return deactivated
    
    def remove_old_tracks(self, current_frame: int) -> None:
        """
        Remove tracks exceeding maximum age.
        
        Args:
            current_frame: Current frame number
        """
        if self.max_track_age is None:
            return
        
        to_remove = []
        for track_id, track in self.tracks.items():
            if (current_frame - track.start_frame) > self.max_track_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.tracks[track_id]
            self.active_tracks.discard(track_id)
    
    def get_valid_tracks(self) -> List[Track]:
        """
        Get tracks meeting minimum length requirement.
        
        Returns:
            List of valid tracks
        """
        return [track for track in self.tracks.values() 
                if track.length >= self.min_track_length]
    
    def get_track_statistics(self) -> Dict[str, any]:
        """
        Get statistics about current tracks.
        
        Returns:
            Dictionary with track statistics
        """
        valid_tracks = self.get_valid_tracks()
        
        if not valid_tracks:
            return {
                'total_tracks': 0,
                'active_tracks': 0,
                'valid_tracks': 0,
                'avg_track_length': 0.0,
                'avg_confidence': 0.0,
                'class_distribution': {}
            }
        
        class_dist = defaultdict(int)
        total_length = 0
        total_confidence = 0
        
        for track in valid_tracks:
            class_dist[track.class_name] += 1
            total_length += track.length
            total_confidence += track.average_confidence
        
        return {
            'total_tracks': len(self.tracks),
            'active_tracks': len(self.active_tracks),
            'valid_tracks': len(valid_tracks),
            'avg_track_length': total_length / len(valid_tracks),
            'avg_confidence': total_confidence / len(valid_tracks),
            'class_distribution': dict(class_dist)
        }
    
    def clear(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self.active_tracks.clear()
        logger.info("All tracks cleared")