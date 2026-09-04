"""
Model loading and management for YOLO detection.
"""

from pathlib import Path
from typing import Optional, Union
import urllib.request
from ultralytics import YOLO

from ..logger import logger
from ..config import DetectionConfig


class ModelLoader:
    """
    Handles downloading and loading YOLO models.
    """
    
    YOLO_MODELS = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt',
        'yolov8s.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt',
        'yolov8m.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt',
        'yolov8l.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt',
        'yolov8x.pt': 'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt',
    }
    
    def __init__(self, models_dir: Union[str, Path] = "models") -> None:
        """
        Initialize model loader.
        
        Args:
            models_dir: Directory for storing models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_model(self, model_name: str = "yolov8x.pt") -> Path:
        """
        Download YOLO model if not present locally.
        
        Args:
            model_name: Name of the model file
            
        Returns:
            Local path to model file
            
        Raises:
            ValueError: If model name is not recognized
        """
        if model_name not in self.YOLO_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.YOLO_MODELS.keys())}")
        
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            return model_path
        
        url = self.YOLO_MODELS[model_name]
        logger.info(f"Downloading {model_name} from {url}...")
        
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model downloaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
        
        return model_path
    
    def load_model(self, 
                   config: DetectionConfig, 
                   force_download: bool = False) -> YOLO:
        """
        Load YOLO model with configuration.
        
        Args:
            config: Detection configuration
            force_download: Force re-download of model
            
        Returns:
            Loaded YOLO model
        """
        model_path = Path(config.model_path)
        
        # Download if needed
        if force_download or not model_path.exists():
            model_path = self.download_model(model_path.name)
        
        logger.info(f"Loading model from: {model_path}")
        model = YOLO(str(model_path))
        
        # Set model parameters
        if hasattr(model, 'overrides'):
            model.overrides['conf'] = config.confidence_threshold
            model.overrides['iou'] = config.iou_threshold
            model.overrides['imgsz'] = config.image_size
            model.overrides['device'] = config.device
        
        logger.info("Model loaded successfully")
        return model
    
    def list_available_models(self) -> list[str]:
        """
        List locally available model files.
        
        Returns:
            List of model filenames
        """
        if not self.models_dir.exists():
            return []
        
        return [f.name for f in self.models_dir.glob("*.pt")]
    
    def cleanup_models(self, keep: list[str] = None) -> None:
        """
        Remove unused model files.
        
        Args:
            keep: List of model filenames to keep
        """
        if keep is None:
            keep = ['yolov8x.pt']
        
        for model_file in self.models_dir.glob("*.pt"):
            if model_file.name not in keep:
                model_file.unlink()
                logger.info(f"Removed model: {model_file.name}")