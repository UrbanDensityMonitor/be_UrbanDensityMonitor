"""
Vehicle detection using YOLO with configurable parameters.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from ultralytics import YOLO
import torch

from ..logger import logger
from ..config import DetectionConfig
from .model_loader import ModelLoader


@dataclass
class Detection:
    """
    Container for a single detection result.
    """
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @property
    def area(self) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    @property
    def width(self) -> float:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]


class VehicleDetector:
    """
    YOLO-based vehicle detector with configurable classes and thresholds.
    """
    
    # COCO class mappings for vehicles
    VEHICLE_CLASSES = {
        2: 'car',
        3: 'motorcycle',
        5: 'bus',
        7: 'truck',
        1: 'bicycle',  # Optional
    }
    
    def __init__(self, config: DetectionConfig) -> None:
        """
        Initialize vehicle detector.
        
        Args:
            config: Detection configuration
        """
        self.config = config
        self.model_loader = ModelLoader()
        self.model: Optional[YOLO] = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load YOLO model."""
        self.model = self.model_loader.load_model(self.config)
        
        # Dynamically update VEHICLE_CLASSES from model names
        if hasattr(self.model, 'names') and self.model.names:
            self.VEHICLE_CLASSES = self.model.names
            
            # If the model is a custom model (e.g. number of classes != 80 COCO classes)
            # AND the configuration uses the default COCO class list [2, 3, 5, 7],
            num_classes = len(self.model.names)
            if num_classes != 80 and self.config.classes == [2, 3, 5, 7]:
                self.config.classes = list(self.model.names.keys())
                logger.info(f"Custom model detected (classes: {num_classes}). Overriding detection classes to: {self.config.classes}")
            else:
                self.config.classes = [2, 3, 5, 7] # Force vehicle only
                
        logger.info(f"Detector initialized with classes: {[self.VEHICLE_CLASSES.get(c, 'unknown') for c in self.config.classes]}")
    
    def detect(self, 
               frame: np.ndarray,
               frame_number: Optional[int] = None) -> List[Detection]:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Input frame (BGR format)
            frame_number: Optional frame number for logging
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Run YOLO inference
        results = self.model(
            frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            classes=self.config.classes,
            imgsz=self.config.image_size,
            device=self.config.device,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is None:
                continue
            
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = box
                
                detection = Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf),
                    class_id=int(class_id),
                    class_name=self.VEHICLE_CLASSES.get(int(class_id), 'unknown')
                )
                detections.append(detection)
        
        if frame_number is not None and frame_number % 100 == 0:
            logger.debug(f"Frame {frame_number}: Detected {len(detections)} vehicles")
        
        return detections
    
    def detect_batch(self, 
                    frames: List[np.ndarray],
                    batch_size: Optional[int] = None) -> List[List[Detection]]:
        """
        Detect vehicles in multiple frames with batching.
        
        Args:
            frames: List of input frames
            batch_size: Batch size for inference (default from config)
            
        Returns:
            List of detection lists (one per frame)
        """
        if batch_size is None:
            batch_size = self.config.batch_size
        
        all_detections = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch_detections = []
            
            for frame in batch_frames:
                detections = self.detect(frame)
                batch_detections.append(detections)
            
            all_detections.extend(batch_detections)
        
        return all_detections
    
    def get_detection_summary(self, detections: List[Detection]) -> Dict[str, Any]:
        """
        Generate summary statistics for detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'total': 0,
                'classes': {},
                'avg_confidence': 0.0,
                'avg_area': 0.0
            }
        
        class_counts = {}
        confidences = []
        areas = []
        
        for det in detections:
            class_counts[det.class_name] = class_counts.get(det.class_name, 0) + 1
            confidences.append(det.confidence)
            areas.append(det.area)
        
        return {
            'total': len(detections),
            'classes': class_counts,
            'avg_confidence': np.mean(confidences),
            'avg_area': np.mean(areas),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
    
    def filter_detections_by_area(self, 
                                  detections: List[Detection],
                                  min_area: float = 100,
                                  max_area: Optional[float] = None) -> List[Detection]:
        """
        Filter detections by bounding box area.
        
        Args:
            detections: List of detections
            min_area: Minimum area threshold
            max_area: Maximum area threshold (optional)
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        for det in detections:
            if det.area < min_area:
                continue
            if max_area is not None and det.area > max_area:
                continue
            filtered.append(det)
        
        return filtered
    
    def reload_model(self, config: Optional[DetectionConfig] = None) -> None:
        """
        Reload model with new configuration.
        
        Args:
            config: New detection configuration (optional)
        """
        if config is not None:
            self.config = config
        
        self._load_model()
        logger.info("Model reloaded")