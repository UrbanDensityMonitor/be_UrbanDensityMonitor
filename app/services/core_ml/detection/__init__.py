"""
Vehicle detection modules using YOLO.
"""

from .detector import VehicleDetector
from .model_loader import ModelLoader

__all__ = [
    'VehicleDetector',
    'ModelLoader'
]