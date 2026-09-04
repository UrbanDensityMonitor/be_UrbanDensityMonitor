"""
Alert Service untuk alert generation & state management
"""
import logging
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
from app.schemas.alert import AlertInfo
from app.schemas.detection import VehicleCounts

logger = logging.getLogger(__name__)


class AlertState(str, Enum):
    """Alert state enum"""
    NORMAL = "normal"
    DENSE = "dense"


class AlertService:
    """
    Service untuk manage alert generation
    
    Features:
        - State-based alert triggering
        - Per-stream state management
        - Alert message generation
        - Prevent duplicate alerts
    
    Usage:
        service = AlertService()
        alert = await service.check_alert(stream_id, density_status, counts, stream_name)
    """
    
    def __init__(self):
        """Initialize alert service"""
        # Per-stream state tracking
        self.stream_states: Dict[str, AlertState] = {}
        logger.info("✅ Alert service initialized")
    
    async def check_alert(
        self,
        stream_id: str,
        density_status: str,
        vehicle_counts: VehicleCounts,
        stream_name: str
    ) -> Optional[AlertInfo]:
        """
        Check apakah perlu trigger alert based on state transition
        
        Args:
            stream_id: Stream UUID
            density_status: Current density status
            vehicle_counts: Vehicle counts object
            stream_name: Stream location name
        
        Returns:
            AlertInfo jika alert triggered, None otherwise
        
        Logic:
            - Alert hanya triggered saat STATUS BERUBAH
            - normal → dense: Alert KEPADATAN TINGGI
            - dense → normal: Alert SUDAH KEMBALI NORMAL
            - Tidak alert jika status sama
        """
        # Get previous state
        prev_state = self.stream_states.get(stream_id, AlertState.NORMAL)
        
        # Determine current state
        current_state = self._get_alert_state(density_status)
        
        # Check for state transition
        alert = None
        
        if current_state == AlertState.DENSE and prev_state == AlertState.NORMAL:
            # Transisi: normal → ramai
            alert = self._generate_congestion_alert(
                stream_name,
                density_status,
                vehicle_counts
            )
            logger.info(f"🔔 Alert triggered: {alert.message}")
        
        elif current_state == AlertState.NORMAL and prev_state == AlertState.DENSE:
            # Transisi: ramai → normal
            alert = self._generate_cleared_alert(
                stream_name,
                vehicle_counts
            )
            logger.info(f"🔔 Alert cleared: {alert.message}")
        
        # Update state
        self.stream_states[stream_id] = current_state
        
        return alert
    
    def _get_alert_state(self, density_status: str) -> AlertState:
        """
        Convert density status ke alert state
        
        Args:
            density_status: Density status string
        
        Returns:
            AlertState enum
        """
        if density_status in ["High Density", "Anomaly"]:
            return AlertState.DENSE
        else:
            return AlertState.NORMAL
    
    def _generate_congestion_alert(
        self,
        stream_name: str,
        density_status: str,
        vehicle_counts: VehicleCounts
    ) -> AlertInfo:
        """
        Generate alert untuk kondisi macet
        
        Args:
            stream_name: Location name
            density_status: Density status
            vehicle_counts: Vehicle counts
        
        Returns:
            AlertInfo object
        """
        # Hitung total vehicles
        total_vehicles = vehicle_counts.total_vehicles
        
        # Tentukan dominant vehicle type
        vehicle_dict = {
            "car": vehicle_counts.car,
            "motorcycle": vehicle_counts.motorcycle,
            "truck": vehicle_counts.truck,
            "bus": vehicle_counts.bus
        }
        
        dominant_vehicle = max(vehicle_dict, key=vehicle_dict.get) if any(vehicle_dict.values()) else "kendaraan"
        dominant_label = {
            "car": "Mobil",
            "motorcycle": "Motor",
            "truck": "Truk",
            "bus": "Bus"
        }.get(dominant_vehicle, dominant_vehicle)
        
        # Get current time
        current_time = datetime.now().strftime("%H:%M WIB")
        
        # Generate message
        message = (
            f"🚨 KEPADATAN TINGGI terdeteksi di {stream_name} pukul {current_time}. "
            f"Total {total_vehicles} kendaraan | Dominan: {dominant_label}."
        )
        
        return AlertInfo(
            triggered=True,
            type=density_status,
            message=message
        )
    
    def _generate_cleared_alert(
        self,
        stream_name: str,
        vehicle_counts: VehicleCounts
    ) -> AlertInfo:
        """
        Generate alert untuk kondisi sudah normal
        
        Args:
            stream_name: Location name
            vehicle_counts: Vehicle counts
        
        Returns:
            AlertInfo object
        """
        total_vehicles = vehicle_counts.total_vehicles
        current_time = datetime.now().strftime("%H:%M WIB")
        
        message = (
            f"✅ Kondisi jalan di {stream_name} sudah kembali normal pukul {current_time}. "
            f"Total kendaraan turun ke {total_vehicles}."
        )
        
        return AlertInfo(
            triggered=True,
            type="cleared",
            message=message
        )
    
    def reset_stream_state(self, stream_id: str):
        """
        Reset state untuk specific stream
        
        Args:
            stream_id: Stream UUID
        """
        if stream_id in self.stream_states:
            del self.stream_states[stream_id]
            logger.info(f"🔄 Alert state reset for stream: {stream_id}")
    
    def get_stream_state(self, stream_id: str) -> AlertState:
        """
        Get current state untuk stream
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            Current AlertState
        """
        return self.stream_states.get(stream_id, AlertState.NORMAL)
    
    def get_all_states(self) -> Dict[str, AlertState]:
        """
        Get all stream states
        
        Returns:
            Dict of stream_id -> AlertState
        """
        return self.stream_states.copy()
