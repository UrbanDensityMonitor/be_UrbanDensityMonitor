"""
StreamWorker — shared inference worker per stream.

Arsitektur:
    - 1 StreamWorker per stream_id (reference counted)
    - Worker menjalankan 1 thread inferensi + 1 FramePipeline
    - N koneksi WebSocket bisa subscribe ke worker yang sama
    - DB write terjadi SEKALI per frame (di worker), bukan per subscriber
    - Worker berhenti otomatis saat subscriber terakhir pergi

Ini menggantikan model lama "1 koneksi = 1 pipeline YOLO" yang tidak
scalable (2 user buka CCTV sama = 2x inferensi).
"""
import asyncio
import logging
import threading
from typing import Callable, Dict, List, Optional

from app.services.detection_service import DetectionService
from app.services.density_service import DensityService
from app.services.alert_service import AlertService
from app.services.frame_pipeline import FramePipeline
from app.repositories.traffic_history_repository import TrafficHistoryRepository
from app.repositories.alert_repository import AlertRepository
from app.utils.video_processor import VideoProcessor
from app.schemas.traffic import TrafficHistoryCreate
from app.schemas.alert import AlertCreate
from app.core.config import settings

logger = logging.getLogger(__name__)

# Tipe callback subscriber: dipanggil dengan FrameResult per frame
Subscriber = Callable[..., None]


class StreamWorker:
    """
    Worker inference untuk SATU stream. Dibagikan antar semua koneksi
    WebSocket yang menonton stream yang sama.
    """

    def __init__(
        self,
        stream_id: str,
        stream_url: str,
        stream_name: str,
        detection_service: DetectionService,
        density_service: DensityService,
        alert_service: AlertService,
        traffic_repo: TrafficHistoryRepository,
        alert_repo: AlertRepository,
    ):
        self.stream_id = stream_id
        self.stream_url = stream_url
        self.stream_name = stream_name
        self.detection_service = detection_service
        self.density_service = density_service
        self.alert_service = alert_service
        self.traffic_repo = traffic_repo
        self.alert_repo = alert_repo

        self.video_processor = VideoProcessor(buffer_size=settings.video_buffer_size)

        # Subscriber management
        self._subscribers: List[Subscriber] = []
        self._lock = threading.Lock()

        # Worker lifecycle
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ═════════════════════════════════════════════════════════════════
    # Subscriber management
    # ═════════════════════════════════════════════════════════════════

    def add_subscriber(self, callback: Subscriber) -> None:
        """Register callback yang menerima FrameResult per frame."""
        with self._lock:
            self._subscribers.append(callback)

    def remove_subscriber(self, callback: Subscriber) -> None:
        """Unregister callback. Return True jika subscriber habis."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def has_subscribers(self) -> bool:
        with self._lock:
            return len(self._subscribers) > 0

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    # ═════════════════════════════════════════════════════════════════
    # Lifecycle
    # ═════════════════════════════════════════════════════════════════

    def start(self) -> None:
        """Mulai thread inferensi worker (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._worker_loop,
            name=f"stream-worker-{self.stream_id[:8]}",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"🚀 StreamWorker started: {self.stream_id} ({self.stream_name})")

    def stop(self) -> None:
        """Stop thread worker & tunggu selesai (maksimal 5 detik)."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info(f"🛑 StreamWorker stopped: {self.stream_id}")

    @property
    def is_running(self) -> bool:
        """Return True jika worker thread sedang aktif."""
        return self._running

    # ═════════════════════════════════════════════════════════════════
    # Worker loop (thread)
    # ═════════════════════════════════════════════════════════════════

    def _worker_loop(self) -> None:
        """
        Loop utama worker (di thread terpisah):
        capture frame -> pipeline ML -> fan-out ke subscribers -> save DB (1x).
        """
        # Dedicated event loop untuk coroutine service di thread ini
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        pipeline = FramePipeline(
            stream_id=self.stream_id,
            stream_name=self.stream_name,
            detection_service=self.detection_service,
            density_service=self.density_service,
            alert_service=self.alert_service,
            event_loop=self._loop,
        )

        video_gen = self.video_processor.stream_frames(
            self.stream_url,
            skip_rate=settings.video_skip_rate,
        )

        try:
            while self._running and self.has_subscribers():
                try:
                    frame = self._loop.run_until_complete(video_gen.__anext__())
                    result = pipeline.process(frame)

                    # Fan-out ke semua subscriber (non-blocking)
                    with self._lock:
                        subscribers = list(self._subscribers)
                    for cb in subscribers:
                        try:
                            cb(result)
                        except Exception as e:
                            logger.warning(f"Subscriber error (stream {self.stream_id}): {e}")

                    # Persist DB SEKALI per frame (bukan per subscriber)
                    self._save_to_database(result)

                except StopAsyncIteration:
                    logger.warning(f"⚠️ Video stream ended: {self.stream_id}")
                    break
                except Exception as e:
                    logger.error(f"❌ Frame processing error (stream {self.stream_id}): {e}", exc_info=True)
                    import time
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"❌ StreamWorker error ({self.stream_id}): {e}", exc_info=True)
        finally:
            try:
                self._loop.run_until_complete(video_gen.aclose())
            except Exception:
                pass
            self._loop.close()
            self._loop = None
            logger.info(f"🛑 Worker loop exited: {self.stream_id}")

    def _save_to_database(self, result) -> None:
        """
        Jadwalkan persist ke DB dari worker thread via main event loop.

        Early return jika main loop belum siap (misal saat startup race condition)
        agar thread worker tidak hang. Timeout 5 detik mencegah thread worker
        blocking selamanya jika DB sedang lambat atau bermasalah.
        """
        try:
            loop = _main_loop()
        except RuntimeError as e:
            logger.warning(f"⚠️ Skipping DB save — main loop belum siap: {e}")
            return
        try:
            future = asyncio.run_coroutine_threadsafe(self._persist(result), loop)
            future.result(timeout=5.0)  # batas waktu agar thread tidak blocking selamanya
        except TimeoutError:
            logger.error(f"⏱️ DB save timeout (>5s) untuk stream {self.stream_id} — frame dilewati")
        except Exception as e:
            logger.error(f"❌ Database save error (stream {self.stream_id}): {e}", exc_info=True)

    async def _persist(self, result) -> None:
        payload = result.payload
        counts = result.counts
        history_data = TrafficHistoryCreate(
            stream_id=self.stream_id,
            person_count=counts.person,
            motorcycle_count=counts.motorcycle,
            car_count=counts.car,
            bus_count=counts.bus,
            truck_count=counts.truck,
            total_vehicle_count=counts.total_vehicles,
            person_vehicle_ratio=payload.get("person_vehicle_ratio", 0.0),
            density_status=payload["density_status"],
            average_speed=payload["average_speed"],
            road_occupancy=payload["road_occupancy"],
            congestion_index=payload["congestion_index"],
        )
        history_id = await self.traffic_repo.create(history_data)

        # Simpan alert hanya untuk kondisi berbahaya (sesuai check constraint DB:
        # alert_type IN ('High Density', 'Anomaly')). Alert 'cleared' (jalan sudah
        # normal) hanya dikirim real-time via WebSocket, tidak dipersist.
        alert_info = payload.get("alert", {})
        if alert_info.get("triggered") and alert_info.get("type") != "cleared":
            alert_data = AlertCreate(
                traffic_history_id=history_id,
                stream_id=self.stream_id,
                alert_type=alert_info["type"],
                alert_message=alert_info["message"],
            )
            await self.alert_repo.create(alert_data)


def _main_loop() -> asyncio.AbstractEventLoop:
    """Ambil main event loop aplikasi (disimpan saat startup oleh asyncpg_client)."""
    from app.db.asyncpg_client import main_event_loop
    if main_event_loop is None:
        raise RuntimeError("Main event loop belum diinisialisasi")
    return main_event_loop


class StreamWorkerManager:
    """
    Registry StreamWorker: memastikan 1 worker per stream_id.
    Membuat worker saat subscriber pertama datang, menghentikan saat
    subscriber terakhir pergi.
    """

    def __init__(
        self,
        detection_service: DetectionService,
        density_service: DensityService,
        alert_service: AlertService,
        traffic_repo: TrafficHistoryRepository,
        alert_repo: AlertRepository,
    ):
        self.detection_service = detection_service
        self.density_service = density_service
        self.alert_service = alert_service
        self.traffic_repo = traffic_repo
        self.alert_repo = alert_repo
        self._workers: Dict[str, StreamWorker] = {}
        self._lock = threading.Lock()

    def subscribe(
        self,
        stream_id: str,
        stream_url: str,
        stream_name: str,
        callback: Subscriber,
    ) -> StreamWorker:
        """
        Subscribe callback ke worker stream. Buat worker jika belum ada.

        Returns:
            StreamWorker instance (untuk unsubscribe nanti)
        """
        with self._lock:
            worker = self._workers.get(stream_id)
            if worker is None:
                worker = StreamWorker(
                    stream_id=stream_id,
                    stream_url=stream_url,
                    stream_name=stream_name,
                    detection_service=self.detection_service,
                    density_service=self.density_service,
                    alert_service=self.alert_service,
                    traffic_repo=self.traffic_repo,
                    alert_repo=self.alert_repo,
                )
                self._workers[stream_id] = worker

            # Subscribe SEBELUM start — hindari race worker exit
            # (loop worker berhenti jika has_subscribers() == False)
            worker.add_subscriber(callback)
            if not worker.is_running:
                worker.start()
            return worker

    def unsubscribe(self, stream_id: str, callback: Subscriber) -> None:
        """Unsubscribe. Worker dihentikan jika tidak ada subscriber lagi."""
        with self._lock:
            worker = self._workers.get(stream_id)
            if worker is None:
                return
            worker.remove_subscriber(callback)
            if not worker.has_subscribers():
                worker.stop()
                del self._workers[stream_id]
                logger.info(f"🗑️ Worker dihapus (subscriber habis): {stream_id}")

    def get_active_streams(self) -> Dict[str, int]:
        """Mapping stream_id -> jumlah subscriber (untuk endpoint monitoring)."""
        with self._lock:
            return {sid: w.subscriber_count() for sid, w in self._workers.items()}

    def shutdown_all(self) -> None:
        """Hentikan semua worker (dipanggil saat aplikasi shutdown)."""
        with self._lock:
            workers = list(self._workers.values())
            self._workers.clear()
        for worker in workers:
            worker.stop()
        logger.info(f"🛑 Semua StreamWorker dihentikan ({len(workers)} worker)")
