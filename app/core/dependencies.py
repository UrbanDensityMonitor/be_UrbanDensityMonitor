"""
Dependency injection untuk FastAPI.
Stateful services (worker manager, WS service, ML services) dibuat SEKALI di
lifespan (app.state) — DI di sini hanya mengambilnya, tidak membuat instance baru.
"""
from typing import Annotated
from fastapi import Depends, HTTPException, Request, WebSocket
import asyncpg

from app.services.websocket_service import WebSocketService
from app.services.stream_worker import StreamWorkerManager


# ═══════════════════════════════════════════════════════════════════════════
# DATABASE DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════

async def get_pool() -> asyncpg.Pool:
    """
    Ambil database connection pool dari app state.

    Raises:
        HTTPException 503: Jika pool belum siap (misal DB connection gagal saat startup).
            503 Service Unavailable lebih tepat daripada 500 karena ini kondisi sementara.
    """
    from app.db.asyncpg_client import get_db_pool
    pool = get_db_pool()
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database belum tersedia. Coba beberapa saat lagi.",
        )
    return pool


# Type alias untuk dependency injection
DatabasePool = Annotated[asyncpg.Pool, Depends(get_pool)]


# ═══════════════════════════════════════════════════════════════════════════
# REPOSITORY DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════

def get_stream_repository(pool: DatabasePool):
    """Get StreamRepository instance"""
    from app.repositories.stream_repository import StreamRepository
    return StreamRepository(pool)


def get_traffic_history_repository(pool: DatabasePool):
    """Get TrafficHistoryRepository instance"""
    from app.repositories.traffic_history_repository import TrafficHistoryRepository
    return TrafficHistoryRepository(pool)


def get_alert_repository(pool: DatabasePool):
    """Get AlertRepository instance"""
    from app.repositories.alert_repository import AlertRepository
    return AlertRepository(pool)


def get_user_repository(pool: DatabasePool):
    """Get UserRepository instance"""
    from app.repositories.user_repository import UserRepository
    return UserRepository(pool)


def get_stream_service(
    repo = Depends(get_stream_repository)
):
    """Get StreamService instance (stateless — hanya koordinasi repo)"""
    from app.services.stream_service import StreamService
    return StreamService(repo)


# ═══════════════════════════════════════════════════════════════════════════
# SERVICE DEPENDENCIES (singleton dari app.state — dibuat di lifespan)
# ═══════════════════════════════════════════════════════════════════════════

def get_websocket_service(request: Request) -> WebSocketService:
    """Get WebSocketService singleton (dari app.state) — untuk HTTP endpoint"""
    return request.app.state.websocket_service


def get_worker_manager(request: Request) -> StreamWorkerManager:
    """Get StreamWorkerManager singleton (dari app.state)"""
    return request.app.state.worker_manager


def get_websocket_service_from_ws(websocket: WebSocket) -> WebSocketService:
    """Get WebSocketService singleton untuk endpoint WebSocket (scope WS, bukan HTTP Request)"""
    return websocket.app.state.websocket_service
