"""
WebSocket Router untuk real-time CCTV streaming
"""
import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from starlette.websockets import WebSocketState

from app.core.dependencies import (
    get_websocket_service,
    get_websocket_service_from_ws,
    get_stream_repository,
)
from app.services.websocket_service import WebSocketService
from app.repositories.stream_repository import StreamRepository
from app.auth.jwt_handler import verify_jwt, verify_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


def _extract_token(
    websocket: WebSocket,
    query_token: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Ekstrak JWT token dari beberapa sumber (prioritas menurun).

    Returns:
        Tuple (token, subprotocol_to_echo):
        - token              : JWT string, atau None jika tidak ada
        - subprotocol_to_echo: Nilai yang harus di-echo kembali di websocket.accept(),
                               sesuai RFC 6455 §4.2.2 — wajib jika client mengirim
                               Sec-WebSocket-Protocol. None jika token dari query param.

    Sumber token (urutan prioritas):
    1. Sec-WebSocket-Protocol header — aman (tidak tercatat di access log)
       Client: new WebSocket(url, [token])
       ⚠️ Server WAJIB echo subprotocol di accept() atau browser tutup koneksi!
    2. Query parameter ?token=... — deprecated (bisa bocor ke proxy log)
    """
    # 1. Dari Sec-WebSocket-Protocol header
    protocols = websocket.headers.get("sec-websocket-protocol")
    if protocols:
        for candidate in protocols.split(","):
            candidate = candidate.strip()
            # Format JWT: tepat 3 segmen dipisah titik (header.payload.signature)
            if candidate and candidate.count(".") == 2:
                # Kembalikan token DAN nilai untuk di-echo di accept()
                return candidate, candidate

    # 2. Fallback ke query param — tidak ada subprotocol untuk di-echo
    return query_token, None


@router.websocket("/ws/live/{stream_id}")
async def stream_websocket(
    websocket: WebSocket,
    stream_id: str,
    token: Optional[str] = Query(None, description="JWT token (deprecated, gunakan subprotocol)"),
    ws_service: WebSocketService = Depends(get_websocket_service_from_ws),
    stream_repo: StreamRepository = Depends(get_stream_repository)
):
    """
    WebSocket endpoint untuk real-time CCTV streaming dengan AI detection

    Parameters:
        - stream_id: UUID of the CCTV stream
        - token: JWT token (query param, deprecated) ATAU via Sec-WebSocket-Protocol header

    Returns WebSocket messages with:
        - type: "frame_update"
        - counts: {person, motorcycle, car, bus, truck}
        - density_status: "Low Density" | "Medium Density" | "High Density" | "Anomaly"
        - average_speed: float (km/h)
        - road_occupancy: float (0.0 - 1.0)
        - congestion_index: float
        - frame: Data URL untuk display (dengan bounding box)
        - alert: Alert information (jika triggered)

    Authentication:
        - Prioritas: Sec-WebSocket-Protocol header (aman dari log)
        - Fallback: query parameter (backward-compatible)

    Example Usage (JavaScript):
        ```javascript
        const token = "your-jwt-token";
        // Cara direkomendasikan (token via subprotocol, tidak bocor ke log):
        const ws = new WebSocket(`ws://localhost:8000/ws/live/${streamId}`, [token]);

        // Cara lama (masih didukung, deprecated):
        // const ws = new WebSocket(`ws://localhost:8000/ws/live/${streamId}?token=${token}`);

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log("Density:", data.density_status);
            console.log("Vehicles:", data.counts);
            if (data.alert) {
                console.log("Alert:", data.alert.message);
            }
        };
        ```
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # 1. Validate JWT Token
    # ═══════════════════════════════════════════════════════════════════════════
    extracted_token, subprotocol = _extract_token(websocket, token)

    if not extracted_token:
        logger.warning("❌ WebSocket authentication failed: token tidak diberikan")
        await websocket.close(code=1008, reason="Token tidak diberikan")
        return

    try:
        user = verify_token(extracted_token)
        logger.info(f"✅ WebSocket authentication success for user: {user.get('sub', 'unknown')}")
    except HTTPException as e:
        logger.warning(f"❌ WebSocket authentication failed: {e.detail}")
        await websocket.close(code=1008, reason=str(e.detail))
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. Get Stream Info
    # ═══════════════════════════════════════════════════════════════════════════
    stream = await stream_repo.find_by_id(stream_id)

    if not stream:
        logger.warning(f"❌ Stream not found: {stream_id}")
        await websocket.close(code=1008, reason="Stream not found")
        return

    stream_url = stream["stream_url"]
    stream_name = stream["location_name"] or f"Stream {stream_id}"

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Handle WebSocket Stream
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        await ws_service.handle_stream(
            websocket=websocket,
            stream_id=stream_id,
            stream_url=stream_url,
            stream_name=stream_name,
            subprotocol=subprotocol,  # echo ke client agar browser tidak tutup koneksi
        )
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from stream: {stream_id}")
    except Exception as e:
        # Pastikan websocket tertutup rapi jika error
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except Exception:
                pass
        logger.error(f"WebSocket error for stream {stream_id}: {e}", exc_info=True)


@router.get("/ws/active-connections")
async def get_active_connections(
    ws_service: WebSocketService = Depends(get_websocket_service),
    _user: dict = Depends(verify_jwt),
):
    """
    Get statistik koneksi WebSocket aktif per stream.

    Requires authentication — endpoint ini mengekspos info internal
    tentang stream yang sedang dimonitor.

    Returns:
        Mapping stream_id -> jumlah koneksi aktif
    """
    connections = ws_service.get_active_connections()
    return {
        "active_streams": list(connections.keys()),
        "connections_per_stream": connections,
        "total_streams": len(connections),
        "total_connections": sum(connections.values()),
    }
