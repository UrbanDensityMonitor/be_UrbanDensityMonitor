"""
Runtime ML configuration (dipakai oleh tracking & feature extraction).
Versi lengkap konfigurasi riset (ProjectConfig, DBSCAN, visualization, dll.)
ada di research/config.py.
"""
from dataclasses import dataclass


@dataclass
class TrackingConfig:
    """ByteTrack tracking configuration."""
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.2
    frame_rate: int = 30
    min_box_area: int = 100


@dataclass
class FeatureExtractionConfig:
    """Feature extraction parameters."""
    pixel_to_meter_ratio: float = 0.05  # meters per pixel
    road_length_meters: float = 50.0
    road_width_meters: float = 7.0
    queue_threshold_speed: float = 5.0  # km/h
    min_track_length: int = 5  # minimum frames to consider valid track
