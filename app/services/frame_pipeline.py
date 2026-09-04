"""
Frame Pipeline — pure ML processing untuk satu frame CCTV.

Single Responsibility: menjalankan rantai ML per frame
(detect -> count -> track -> density -> alert -> annotate -> encode -> payload).
TIDAK menangani: WebSocket, threading, database, connection registry.
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List

import numpy as np

from app.services.detection_service import DetectionService
from app.services.tracking_service import TrackingService
from app.services.density_service import DensityService
from app.services.alert_service import AlertService
from app.services.core_ml.detection.detector import Detection
from app.utils.frame_encoder import FrameEncoder
from app.utils.frame_annotator import draw_detections
from app.schemas.detection import VehicleCounts
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """Hasil pemrosesan satu frame — dikonsumsi oleh subscriber (WS) & persistence."""
    payload: dict
    counts: VehicleCounts
    latency_ms: float
    fps: float


class FramePipeline:
    """
    Pure ML pipeline per frame. Stateful (tracker per stream), tapi tidak
    menyentuh I/O jaringan maupun database.

    Satu instance per stream (dipunyai StreamWorker).
    Dipanggil dari thread worker via run() — loop event loop milik worker.
    """

    def __init__(
        self,
        stream_id: str,
        stream_name: str,
        detection_service: DetectionService,
        density_service: DensityService,
        alert_service: AlertService,
        event_loop: asyncio.AbstractEventLoop,
    ):
        self.stream_id = stream_id
        self.stream_name = stream_name
        self.detection_service = detection_service
        self.tracking_service = TrackingService()
        self.density_service = density_service
        self.alert_service = alert_service
        self.frame_encoder = FrameEncoder(jpeg_quality=settings.jpeg_quality)
        self._frame_count = 0
        self._loop = event_loop

    def process(self, frame: np.ndarray) -> FrameResult:
        """
        Proses satu frame end-to-end (synchronous, CPU/GPU bound).

        Args:
            frame: OpenCV frame (BGR)

        Returns:
            FrameResult berisi payload siap kirim + counts + metrik performa
        """
        start_time = time.time()
        self._frame_count += 1

        # 1. Object Detection (YOLO)
        detections = self._run(
            self.detection_service.detect_objects(frame)
        )

        # 2. Count vehicles per kategori
        counts = self._count_vehicles(detections)

        # 3. Tracking & feature extraction
        features = self._run(
            self.tracking_service.update_and_extract(detections, frame, self._frame_count)
        )
        # Override: tracker butuh min 5 frame sebelum aktif
        features.vehicle_count = counts.total_vehicles

        # 4. Density prediction
        density_pred = self._run(self.density_service.predict(features))

        # 5. Alert check (state transition, per-stream state)
        alert = self._run(
            self.alert_service.check_alert(self.stream_id, density_pred.status, counts, self.stream_name)
        )

        # 6. Annotate + encode
        annotated = draw_detections(frame, detections)
        frame_data_url = self.frame_encoder.encode_to_data_url(annotated)
        frame_b64 = frame_data_url.split(",", 1)[1] if "," in frame_data_url else frame_data_url

        # 7. Build payload
        payload = self._build_payload(counts, density_pred, features, frame_b64, frame_data_url, alert)

        elapsed = time.time() - start_time
        return FrameResult(
            payload=payload,
            counts=counts,
            latency_ms=elapsed * 1000,
            fps=1.0 / elapsed if elapsed > 0 else 0.0,
        )

    def _run(self, coro):
        """
        Jalankan coroutine di event loop worker.

        Dipanggil secara synchronous dari worker thread saat loop SEDANG IDLE
        (di antara pemanggilan run_until_complete), jadi run_until_complete
        aman — TIDAK boleh pakai run_coroutine_threadsafe().result() di sini
        karena loop tidak sedang berjalan → callback tak pernah dieksekusi → deadlock.
        """
        return self._loop.run_until_complete(coro)

    @staticmethod
    def _count_vehicles(detections: List[Detection]) -> VehicleCounts:
        """Hitung kendaraan per kategori dari detections."""
        counts = VehicleCounts()
        for det in detections:
            if det.class_name == "person":
                counts.person += 1
            elif det.class_name == "motorcycle":
                counts.motorcycle += 1
            elif det.class_name == "car":
                counts.car += 1
            elif det.class_name == "bus":
                counts.bus += 1
            elif det.class_name == "truck":
                counts.truck += 1
        return counts

    def _build_payload(self, counts, density_pred, features, frame_b64, frame_data_url, alert) -> dict:
        """Susun payload WebSocket (format tetap sama — FE tidak berubah)."""
        payload = {
            "type": "frame_update",
            "stream_id": self.stream_id,
            "counts": {
                "person": counts.person,
                "motorcycle": counts.motorcycle,
                "car": counts.car,
                "bus": counts.bus,
                "truck": counts.truck,
            },
            "person_vehicle_ratio": 0.0,
            "density_status": density_pred.status,
            "average_speed": round(features.average_speed, 2),
            "road_occupancy": round(features.road_occupancy, 2),
            "congestion_index": round(features.congestion_index, 2),
            "frame_base64": frame_b64,
            "frame": frame_data_url,
        }
        if alert:
            payload["alert"] = {
                "triggered": alert.triggered,
                "type": alert.type,
                "message": alert.message,
            }
        return payload
