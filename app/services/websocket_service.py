"""
WebSocket Service — thin orchestrator untuk koneksi WS live stream.

Tanggung jawab SATU-SATUNYA: menghubungkan koneksi WebSocket ke
StreamWorkerManager (shared worker per stream) dan mengelola registry
koneksi aktif (ConnectionManager).

Inference ML ada di StreamWorker/FramePipeline, persistensi DB di
worker (1x per frame), bukan di sini.
"""
import asyncio
import logging
import threading
from typing import Callable, Dict, Optional

from fastapi import WebSocket, WebSocketDisconnect

from app.services.stream_worker import StreamWorkerManager, StreamWorker

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Registry koneksi WebSocket aktif per stream (thread-safe)."""

    def __init__(self):
        self._connections: Dict[str, int] = {}  # stream_id -> jumlah koneksi
        self._lock = threading.Lock()

    def register(self, stream_id: str) -> None:
        with self._lock:
            self._connections[stream_id] = self._connections.get(stream_id, 0) + 1

    def unregister(self, stream_id: str) -> None:
        with self._lock:
            count = self._connections.get(stream_id, 0) - 1
            if count <= 0:
                self._connections.pop(stream_id, None)
            else:
                self._connections[stream_id] = count

    def get_stats(self) -> Dict[str, int]:
        """Mapping stream_id -> jumlah koneksi aktif."""
        with self._lock:
            return dict(self._connections)


class WebSocketService:
    """
    Orkestrator koneksi WebSocket live stream.

    Flow:
        1. Accept koneksi & register di ConnectionManager
        2. Subscribe ke StreamWorker (shared per stream, dibuat jika belum ada)
        3. Loop: terima FrameResult dari worker -> kirim ke client
        4. Disconnect: unsubscribe & unregister (worker berhenti jika subscriber habis)
    """

    def __init__(self, worker_manager: StreamWorkerManager):
        self.worker_manager = worker_manager
        self.connection_manager = ConnectionManager()
        logger.info("✅ WebSocket service initialized")

    async def handle_stream(
        self,
        websocket: WebSocket,
        stream_id: str,
        stream_url: str,
        stream_name: str,
        subprotocol: Optional[str] = None,
    ):
        """
        Handle satu koneksi WebSocket untuk satu stream.

        Args:
            websocket    : Koneksi WebSocket
            stream_id    : Stream UUID
            stream_url   : URL stream CCTV
            stream_name  : Nama lokasi stream
            subprotocol  : Nilai Sec-WebSocket-Protocol dari client yang harus di-echo
                           kembali sesuai RFC 6455 §4.2.2. None jika token dari query param.
        """
        # RFC 6455: jika client mengirim Sec-WebSocket-Protocol,
        # server WAJIB meng-echo salah satu nilai yang diterima, atau browser
        # akan langsung menutup koneksi setelah handshake.
        await websocket.accept(subprotocol=subprotocol)
        self.connection_manager.register(stream_id)
        logger.info(f"🔗 Client connected to stream: {stream_id}")

        # Queue: bridge dari callback sync worker-thread -> async loop WS
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=5)

        def on_frame(result):
            """Callback dari StreamWorker thread — thread-safe put ke queue."""
            def _put():
                try:
                    queue.put_nowait(result)
                except asyncio.QueueFull:
                    pass  # Client lambat: skip frame (real-time > lengkap)
            loop.call_soon_threadsafe(_put)

        # Subscribe ke shared worker (dibuat otomatis jika belum ada)
        worker: StreamWorker = self.worker_manager.subscribe(
            stream_id=stream_id,
            stream_url=stream_url,
            stream_name=stream_name,
            callback=on_frame,
        )

        try:
            while True:
                result = await queue.get()
                await websocket.send_json(result.payload)
                logger.debug(
                    f"📊 Stream {stream_id} | Latency: {result.latency_ms:.1f}ms | FPS: {result.fps:.1f}"
                )

        except WebSocketDisconnect:
            logger.info(f"👋 Client disconnected from stream: {stream_id}")
        except RuntimeError as e:
            # Send setelah close — client sudah pergi
            logger.info(f"🛑 WebSocket closed for stream {stream_id}: {e}")
        except Exception as e:
            logger.error(f"❌ WebSocket error for stream {stream_id}: {e}", exc_info=True)
        finally:
            self.worker_manager.unsubscribe(stream_id, on_frame)
            self.connection_manager.unregister(stream_id)
            logger.info(
                f"🛑 Client disconnected from stream: {stream_id} "
                f"(subscriber tersisa: {worker.subscriber_count()})"
            )

    def get_active_connections(self) -> Dict[str, int]:
        """Statistik koneksi per stream (untuk endpoint monitoring)."""
        return self.connection_manager.get_stats()
