"""
Unit tests — pytest
Run: pytest tests/ -v
"""
import numpy as np
import pytest

from app.services.core_ml.detection.detector import Detection
from app.utils.frame_annotator import draw_detections
from app.utils.frame_encoder import FrameEncoder
from app.schemas.detection import VehicleCounts
from app.services.density_service import FrameFeatures, DensityPrediction, DensityService
from app.services.alert_service import AlertService


# ═════════════════════════════════════════════════════════════════
# FrameAnnotator
# ═════════════════════════════════════════════════════════════════

class TestFrameAnnotator:
    def _detections(self):
        return [
            Detection(bbox=(100, 100, 200, 180), confidence=0.92, class_id=1, class_name='car'),
            Detection(bbox=(300, 200, 380, 260), confidence=0.75, class_id=2, class_name='motorcycle'),
            Detection(bbox=(10, 10, 90, 60), confidence=0.60, class_id=0, class_name='bus', track_id=7),
        ]

    def test_annotates_frame(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_detections(frame, self._detections())
        assert annotated.sum() > 0, "bounding box harus tergambar"

    def test_original_frame_not_mutated(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_detections(frame, self._detections())
        assert frame.sum() == 0, "frame asli tidak boleh berubah"

    def test_box_pixels_drawn(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_detections(frame, self._detections())
        # Garis atas box car (y=100, tengah x)
        assert annotated[100, 150].sum() > 0

    def test_empty_detections_returns_copy(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_detections(frame, [])
        assert annotated.sum() == 0


# ═════════════════════════════════════════════════════════════════
# FrameEncoder
# ═════════════════════════════════════════════════════════════════

class TestFrameEncoder:
    def test_encode_to_data_url(self):
        encoder = FrameEncoder(jpeg_quality=60)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        data_url = encoder.encode_to_data_url(frame)
        assert data_url.startswith("data:image/jpeg;base64,")

    def test_roundtrip_base64(self):
        encoder = FrameEncoder(jpeg_quality=90)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        b64 = encoder.encode_to_base64(frame)
        decoded = FrameEncoder.decode_from_base64(b64)
        assert decoded.shape == (480, 640, 3)

    def test_invalid_quality_rejected(self):
        with pytest.raises(ValueError):
            FrameEncoder(jpeg_quality=101)


# ═════════════════════════════════════════════════════════════════
# VehicleCounts
# ═════════════════════════════════════════════════════════════════

class TestVehicleCounts:
    def test_total_vehicles_excludes_person(self):
        counts = VehicleCounts(person=5, motorcycle=3, car=4, bus=1, truck=2)
        assert counts.total_vehicles == 10
        assert counts.total_with_person == 15

    def test_defaults_zero(self):
        counts = VehicleCounts()
        assert counts.total_vehicles == 0


# ═════════════════════════════════════════════════════════════════
# DensityService (rule-based, tanpa model)
# ═════════════════════════════════════════════════════════════════

class TestDensityService:
    @pytest.fixture
    def service(self, monkeypatch):
        """DensityService dengan model=None (pure rule-based) — tanpa load pkl."""
        import app.services.density_service as ds
        monkeypatch.setattr(ds.DensityService, "_load_model", classmethod(lambda cls: None))
        return ds.DensityService()

    def _features(self, count):
        return FrameFeatures(
            vehicle_count=count,
            average_speed=30.0,
            road_occupancy=0.4,
            congestion_index=0.3,
        )

    @pytest.mark.asyncio
    async def test_zero_vehicles_low(self, service):
        result = await service.predict(self._features(0))
        assert result.status == "Low Density"

    @pytest.mark.asyncio
    async def test_many_vehicles_high(self, service):
        result = await service.predict(self._features(30))
        assert result.status == "High Density"

    @pytest.mark.asyncio
    async def test_rain_anomaly(self, service):
        result = await service.predict(self._features(60), weather_condition="rain")
        assert result.status == "Anomaly"


# ═════════════════════════════════════════════════════════════════
# AlertService (state transition)
# ═════════════════════════════════════════════════════════════════

class TestAlertService:
    @pytest.fixture
    def service(self):
        return AlertService()

    def _counts(self, n=25):
        return VehicleCounts(car=n)

    @pytest.mark.asyncio
    async def test_normal_to_dense_triggers(self, service):
        alert = await service.check_alert("s1", "High Density", self._counts(), "Jl. Test")
        assert alert is not None and alert.triggered
        assert "KEPADATAN" in alert.message

    @pytest.mark.asyncio
    async def test_dense_to_dense_no_alert(self, service):
        await service.check_alert("s2", "High Density", self._counts(), "Jl. Test")
        alert = await service.check_alert("s2", "High Density", self._counts(), "Jl. Test")
        assert alert is None, "status sama tidak boleh trigger alert lagi"

    @pytest.mark.asyncio
    async def test_dense_to_normal_clears(self, service):
        await service.check_alert("s3", "High Density", self._counts(), "Jl. Test")
        alert = await service.check_alert("s3", "Low Density", VehicleCounts(car=2), "Jl. Test")
        assert alert is not None and alert.type == "cleared"


# ═════════════════════════════════════════════════════════════════
# ConnectionManager (websocket_service)
# ═════════════════════════════════════════════════════════════════

class TestConnectionManager:
    def test_register_unregister(self):
        from app.services.websocket_service import ConnectionManager
        cm = ConnectionManager()
        cm.register("s1")
        cm.register("s1")
        assert cm.get_stats() == {"s1": 2}
        cm.unregister("s1")
        assert cm.get_stats() == {"s1": 1}
        cm.unregister("s1")
        assert cm.get_stats() == {}

    def test_unregister_unknown_is_safe(self):
        from app.services.websocket_service import ConnectionManager
        cm = ConnectionManager()
        cm.unregister("unknown")
        assert cm.get_stats() == {}
