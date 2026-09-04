"""
Application lifespan events (startup & shutdown)
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.db.asyncpg_client import init_db_pool, close_db_pool, get_db_pool
from app.core.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle & stateful singletons (app.state).

    Startup:
        - Initialize database connection pool
        - Create stateful service singletons (worker manager, WS service)

    Shutdown:
        - Stop semua StreamWorker
        - Close database connections
    """
    # ═══════════════════════════════════════════════════════════
    # STARTUP
    # ═══════════════════════════════════════════════════════════
    logger.info("🚀 Starting Urban Density Monitor API...")
    logger.info(f"📍 Environment: {settings.environment}")
    logger.info(f"🐛 Debug mode: {settings.debug}")

    # Initialize database
    logger.info("🗄️  Initializing database connection pool...")
    await init_db_pool()
    if get_db_pool() is None:
        logger.warning("⚠️  Database NOT connected — endpoints DB akan 500 (development/testing mode)")
    else:
        logger.info("✅ Database connection pool ready")

    # ═══════════════════════════════════════════════════════════
    # STATEFUL SINGLETONS (disimpan di app.state — shared antar request)
    # ═══════════════════════════════════════════════════════════
    from app.repositories.traffic_history_repository import TrafficHistoryRepository
    from app.repositories.alert_repository import AlertRepository
    from app.services.detection_service import DetectionService
    from app.services.density_service import DensityService
    from app.services.alert_service import AlertService
    from app.services.stream_worker import StreamWorkerManager
    from app.services.websocket_service import WebSocketService

    pool = get_db_pool()

    # Stateless ML services (singleton karena model dimuat sekali)
    app.state.detection_service = DetectionService()
    app.state.density_service = DensityService()
    app.state.alert_service = AlertService()

    # StreamWorkerManager: shared worker per stream (1 inferensi per stream)
    app.state.worker_manager = StreamWorkerManager(
        detection_service=app.state.detection_service,
        density_service=app.state.density_service,
        alert_service=app.state.alert_service,
        traffic_repo=TrafficHistoryRepository(pool),
        alert_repo=AlertRepository(pool),
    )

    # WebSocketService: thin orchestrator (butuh worker_manager)
    app.state.websocket_service = WebSocketService(app.state.worker_manager)

    logger.info("✅ Application startup complete!")

    # ═══════════════════════════════════════════════════════════
    # YIELD - Application Running
    # ═══════════════════════════════════════════════════════════
    yield

    # ═══════════════════════════════════════════════════════════
    # SHUTDOWN
    # ═══════════════════════════════════════════════════════════
    logger.info("🛑 Shutting down Urban Density Monitor API...")

    # Stop semua worker inference
    logger.info("🧹 Stopping stream workers...")
    app.state.worker_manager.shutdown_all()
    logger.info("✅ Stream workers stopped")

    # Close database connections
    logger.info("🗄️  Closing database connections...")
    await close_db_pool()
    logger.info("✅ Database connections closed")

    logger.info("✅ Application shutdown complete")
