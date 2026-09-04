"""
Density Service untuk traffic density prediction
Refactored dari services/clustering.py dengan better structure
"""
import logging
import pickle
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FrameFeatures:
    """Features extracted dari frame untuk density prediction"""
    vehicle_count: int
    average_speed: float
    road_occupancy: float
    congestion_index: float


@dataclass
class DensityPrediction:
    """Result dari density prediction"""
    status: str  # "Low Density", "Medium Density", "High Density", "Anomaly"
    vehicle_count: int
    confidence: float = 1.0  # Future: bisa ditambahkan confidence score


class DensityService:
    """
    Service untuk memprediksi density status berdasarkan frame features.

    Features:
        - KMeans clustering model (load sekali, singleton)
        - Rule-based fallback jika model tidak tersedia
        - Weather-aware (rain detection)
        - Runtime-configurable thresholds (tidak mutasi global settings)

    Usage:
        service = DensityService()
        prediction = await service.predict(features)
    """

    _instance: Optional["DensityService"] = None
    _model = None

    def __new__(cls) -> "DensityService":
        """
        Singleton pattern yang aman:
        instance hanya di-assign SETELAH _load_model() berhasil,
        sehingga jika load gagal, pemanggilan berikutnya akan retry.
        """
        if cls._instance is None:
            instance = super().__new__(cls)
            cls._load_model()           # load dulu — bisa raise jika ada masalah serius
            cls._instance = instance    # assign SETELAH load selesai (sukses atau tidak)
        return cls._instance

    def __init__(self) -> None:
        """
        Runtime thresholds — diinisialisasi dari settings tapi bisa diubah
        tanpa menyentuh global settings (Pydantic model immutable).
        Dipanggil setiap kali DensityService() dipanggil tapi __new__ hanya sekali.
        """
        if not hasattr(self, "_initialized"):
            self._low_density_max: int = settings.low_density_max
            self._medium_density_max: int = settings.medium_density_max
            self._high_density_min: int = settings.high_density_min
            self._anomaly_rain_min: int = settings.anomaly_rain_min
            self._initialized = True
    
    @classmethod
    def _load_model(cls):
        """Load density clustering model"""
        model_path = Path(settings.density_model_path)
        
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    cls._model = pickle.load(f)
                logger.info(f"✅ Density clustering model loaded from: {model_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load density model: {e}", exc_info=True)
                cls._model = None
        else:
            logger.warning(f"⚠️ Density model not found: {model_path}, using rule-based only")
            cls._model = None
    
    async def predict(
        self,
        features: FrameFeatures,
        weather_condition: Optional[str] = None
    ) -> DensityPrediction:
        """
        Predict density status dari frame features
        
        Args:
            features: FrameFeatures object
            weather_condition: Optional weather info ("rain", "clear", etc.)
        
        Returns:
            DensityPrediction object
        
        Logic:
            1. Check for edge cases (0 vehicles, anomaly conditions)
            2. Apply hard thresholds (override KMeans if obvious)
            3. Use KMeans for gray zone
            4. Fallback to rule-based if no model
        """
        vehicle_count = features.vehicle_count
        is_raining = weather_condition == "rain"
        
        # ═══════════════════════════════════════════════════════════
        # Edge Case 1: No vehicles
        # ═══════════════════════════════════════════════════════════
        if vehicle_count == 0:
            return DensityPrediction(
                status="Low Density",
                vehicle_count=vehicle_count
            )

        # ═══════════════════════════════════════════════════════════
        # Edge Case 2: Anomaly during rain
        # ═══════════════════════════════════════════════════════════
        if is_raining and vehicle_count > self._anomaly_rain_min:
            return DensityPrediction(
                status="Anomaly",
                vehicle_count=vehicle_count
            )

        # ═══════════════════════════════════════════════════════════
        # Hard Override: Clear High Density
        # ═══════════════════════════════════════════════════════════
        if vehicle_count >= self._medium_density_max:
            return DensityPrediction(
                status="High Density",
                vehicle_count=vehicle_count
            )

        # ═══════════════════════════════════════════════════════════
        # Hard Override: Clear Low Density
        # ═══════════════════════════════════════════════════════════
        if vehicle_count < self._low_density_max:
            return DensityPrediction(
                status="Low Density",
                vehicle_count=vehicle_count
            )
        
        # ═══════════════════════════════════════════════════════════
        # Gray Zone: Use KMeans if available
        # (LOW_DENSITY_MAX <= count < MEDIUM_DENSITY_MAX)
        # ═══════════════════════════════════════════════════════════
        if self._model is not None:
            status = self._predict_with_kmeans(features)
        else:
            # Fallback: rule-based
            status = self._predict_rule_based(features)
        
        return DensityPrediction(
            status=status,
            vehicle_count=vehicle_count
        )
    
    def _predict_with_kmeans(self, features: FrameFeatures) -> str:
        """
        Predict menggunakan KMeans model
        
        Args:
            features: FrameFeatures
        
        Returns:
            Density status string
        """
        # Prepare input untuk model
        input_data = np.array([[
            features.vehicle_count,
            features.average_speed,
            features.road_occupancy,
            features.congestion_index
        ]])
        
        try:
            # Predict cluster
            cluster_id = self._model.predict(input_data)[0]
            
            # Map cluster ID ke status
            # Ini mapping dari trained model
            if cluster_id == 0:
                return "Low Density"
            elif cluster_id == 1:
                return "Medium Density"
            else:
                return "High Density"
                
        except Exception as e:
            logger.error(f"KMeans prediction failed: {e}, using rule-based fallback")
            return self._predict_rule_based(features)
    
    def _predict_rule_based(self, features: FrameFeatures) -> str:
        """
        Rule-based prediction (fallback saat model tidak tersedia).

        Menggunakan instance thresholds (bukan settings langsung) agar
        update runtime via update_thresholds() langsung berpengaruh.
        """
        vehicle_count = features.vehicle_count

        if vehicle_count < self._low_density_max:
            return "Low Density"
        elif vehicle_count < self._medium_density_max:
            return "Medium Density"
        else:
            return "High Density"
    
    def get_thresholds(self) -> dict:
        """Get current runtime threshold values."""
        return {
            "low_density_max": self._low_density_max,
            "medium_density_max": self._medium_density_max,
            "high_density_min": self._high_density_min,
            "anomaly_rain_min": self._anomaly_rain_min,
            "model_loaded": self._model is not None,
        }
    
    def update_thresholds(
        self,
        low_max: Optional[int] = None,
        medium_max: Optional[int] = None,
        high_min: Optional[int] = None,
    ) -> None:
        """
        Update thresholds dinamis untuk keperluan tuning/testing.

        Menggunakan instance variable (bukan mutasi global settings)
        sehingga aman dengan Pydantic v2 yang immutable by default.
        Perubahan hanya berlaku untuk runtime instance ini dan
        tidak dipersist ke .env / config file.
        """
        if low_max is not None:
            self._low_density_max = low_max
        if medium_max is not None:
            self._medium_density_max = medium_max
        if high_min is not None:
            self._high_density_min = high_min

        logger.info(f"Thresholds updated (runtime only): {self.get_thresholds()}")
