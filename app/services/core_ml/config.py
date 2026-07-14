"""
Global configuration management for Traffic State Discovery.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from pathlib import Path
import yaml


@dataclass
class DetectionConfig:
    """YOLO detection configuration."""
    model_path: str = "models/yolov8x.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])  # car, motorcycle, bus, truck
    image_size: int = 1280
    device: str = "cuda:0"
    batch_size: int = 32


@dataclass
class TrackingConfig:
    """ByteTrack tracking configuration."""
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.2
    frame_rate: int = 30
    min_box_area: int = 100


@dataclass
class ROIConfig:
    """Region of Interest configuration."""
    roi_path: str = "configs/roi.json"
    default_polygon: Optional[List[Tuple[int, int]]] = None
    interactive: bool = True


@dataclass
class FeatureExtractionConfig:
    """Feature extraction parameters."""
    pixel_to_meter_ratio: float = 0.05  # meters per pixel
    road_length_meters: float = 50.0
    road_width_meters: float = 7.0
    queue_threshold_speed: float = 5.0  # km/h
    min_track_length: int = 5  # minimum frames to consider valid track


@dataclass
class TemporalAggregationConfig:
    """Temporal aggregation windows."""
    windows_seconds: List[int] = field(default_factory=lambda: [10, 30, 60, 300])
    default_window: int = 60


@dataclass
class ClusteringConfig:
    """DBSCAN clustering configuration."""
    eps_range: List[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0])
    min_samples_range: List[int] = field(default_factory=lambda: [3, 5, 10, 15, 20])
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"
    grid_search: bool = True


@dataclass
class VisualizationConfig:
    """Visualization settings."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 150
    colormap: str = "viridis"
    font_size: int = 12
    save_format: str = "png"
    video_fps: int = 30
    video_codec: str = "mp4v"
    show_track_id: bool = True
    show_speed: bool = True
    show_roi: bool = True
    heatmap_alpha: float = 0.6


@dataclass
class PathsConfig:
    """Project paths configuration."""
    root: Path = Path.cwd()
    videos: Path = Path("videos")
    models: Path = Path("models")
    weights: Path = Path("weights")
    configs: Path = Path("configs")
    outputs: Path = Path("outputs")
    logs: Path = Path("logs")
    datasets: Path = Path("datasets")
    paper: Path = Path("paper")
    
    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        # Convert any string paths to Path objects
        self.root = Path(self.root)
        self.videos = Path(self.videos)
        self.models = Path(self.models)
        self.weights = Path(self.weights)
        self.configs = Path(self.configs)
        self.outputs = Path(self.outputs)
        self.logs = Path(self.logs)
        self.datasets = Path(self.datasets)
        self.paper = Path(self.paper)
        
        for path in [self.videos, self.models, self.weights, self.configs,
                     self.outputs, self.logs, self.datasets, self.paper]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectConfig:
    """Main configuration class."""
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    features: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    temporal: TemporalAggregationConfig = field(default_factory=TemporalAggregationConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ProjectConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config from dictionary
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                current_attr = getattr(config, key)
                if hasattr(current_attr, '__dataclass_fields__') and isinstance(value, dict):
                    # Filter keys to only those expected by the dataclass constructor
                    fields = current_attr.__dataclass_fields__
                    filtered_value = {k: v for k, v in value.items() if k in fields}
                    setattr(config, key, type(current_attr)(**filtered_value))
                else:
                    setattr(config, key, value)
        
        return config
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (DetectionConfig, TrackingConfig, ROIConfig,
                                FeatureExtractionConfig, TemporalAggregationConfig,
                                ClusteringConfig, VisualizationConfig, PathsConfig)):
                config_dict[key] = value.__dict__
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)