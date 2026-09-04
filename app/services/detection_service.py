"""
Detection Service untuk YOLO object detection
Handles model loading, inference, and result parsing
"""
import logging
import torch
import numpy as np
from typing import List, Optional
from pathlib import Path
from ultralytics import YOLO
from app.core.config import settings
from app.services.core_ml.detection.detector import Detection

logger = logging.getLogger(__name__)


class DetectionService:
    """
    Singleton service untuk YOLO detection
    
    Features:
        - Lazy model loading (loaded only once)
        - Auto device detection (CUDA/MPS/CPU)
        - Result parsing
        - Configurable detection parameters
    
    Usage:
        service = DetectionService()
        detections = await service.detect_objects(frame)
    """
    
    _instance = None
    _model = None
    _device = None
    
    # Vehicle class mapping untuk custom model
    VEHICLE_CLASSES = {
        0: 'bus',
        1: 'car',
        2: 'motorcycle',
        3: 'pickup',
        4: 'truck'
    }
    
    def __new__(cls):
        """Singleton pattern - only one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._initialize_model()
        return cls._instance
    
    @classmethod
    def _initialize_model(cls):
        """Initialize YOLO model dan detect optimal device"""
        logger.info("🤖 Initializing YOLO detection service...")
        
        # Detect optimal device
        cls._device = cls._get_optimal_device()
        logger.info(f"🎯 Using device: {cls._device}")
        
        # Load model
        model_path = Path(settings.yolo_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        
        logger.info(f"📦 Loading YOLO model from: {model_path}")
        cls._model = YOLO(str(model_path)).to(cls._device)
        logger.info("✅ YOLO model loaded successfully")
    
    @staticmethod
    def _get_optimal_device() -> str:
        """
        Auto-detect optimal device (GPU/CPU)
        
        Returns:
            'cuda' for NVIDIA GPU, 'mps' for Apple GPU, 'cpu' for fallback
        """
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return 'mps'
        else:
            logger.warning("⚠️ No GPU detected, using CPU (slower)")
            return 'cpu'
    
    async def detect_objects(
        self,
        frame: np.ndarray,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        image_size: Optional[int] = None
    ) -> List[Detection]:
        """
        Detect objects dalam frame
        
        Args:
            frame: OpenCV frame (numpy array)
            conf_threshold: Confidence threshold (default dari config)
            iou_threshold: IOU threshold (default dari config)
            image_size: Inference image size (default dari config)
        
        Returns:
            List of Detection objects
        """
        # Use config defaults if not provided
        conf = conf_threshold or settings.detection_confidence
        iou = iou_threshold or settings.detection_iou
        imgsz = image_size or settings.detection_image_size
        
        # Run YOLO inference
        # classes=[0,1,2,3,4] = hanya detect vehicle classes dari custom model
        results = self._model(
            frame,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self._device,
            classes=[0, 1, 2, 3, 4]  # Custom model vehicle classes
        )
        
        # Parse results
        detections = self._parse_results(results[0])
        
        return detections
    
    def _parse_results(self, result) -> List[Detection]:
        """
        Parse YOLO results menjadi Detection objects
        
        Args:
            result: YOLO result object
        
        Returns:
            List of Detection objects
        """
        detections = []
        
        if not result.boxes or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            class_name = self.VEHICLE_CLASSES.get(int(class_id), 'unknown')
            
            # Skip unknown classes
            if class_name == 'unknown':
                continue
            
            # Map 'pickup' to 'truck' for consistency
            if class_name == 'pickup':
                class_name = 'truck'
            
            detection = Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=float(conf),
                class_id=int(class_id),
                class_name=class_name
            )
            
            detections.append(detection)
        
        return detections
    
    def get_device(self) -> str:
        """Get current device being used"""
        return self._device
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            Dict with model info
        """
        return {
            "model_path": settings.yolo_model_path,
            "device": self._device,
            "confidence_threshold": settings.detection_confidence,
            "iou_threshold": settings.detection_iou,
            "image_size": settings.detection_image_size,
            "vehicle_classes": list(self.VEHICLE_CLASSES.values())
        }
