import asyncio
import cv2
import base64
import time
import datetime
import logging
import threading
import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends

from app.db.asyncpg_client import get_db
from app.auth.jwt_handler import decode_jwt_token
from app.services.history_service import save_to_db
from app.services.clustering import predict_density

from app.core.ai_config import model, VEHICLE_CLASSES, device
from app.services.core_ml.config import TrackingConfig, FeatureExtractionConfig
from app.services.core_ml.tracking.byte_tracker import ByteTrackTracker
from app.services.core_ml.features.feature_extractor import FeatureExtractor
from app.services.core_ml.roi.roi_manager import ROIManager
from app.services.core_ml.detection.detector import Detection

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["WebSockets"]
)

@router.websocket("/live/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str, token: str = Query(default=None)):
    await websocket.accept()
    logger.info(f"🔗 Client terhubung ke stream: {stream_id}")
    
    # Ambil pool secara manual karena dependencies async context pada websocket bisa tricky
    try:
        pool_gen = get_db()
        pool = await pool_gen.__anext__()
    except Exception as e:
        logger.error(f"❌ Database belum nyambung! {e}")
        await websocket.close()
        return

    stream_source = "public"
    try:
        query_public = "SELECT stream_url, location_name FROM streams WHERE id = $1"
        row = await pool.fetchrow(query_public, stream_id)

        if row:
            stream_source = "public"
            stream_url = row["stream_url"]
            stream_name = row["location_name"] if row["location_name"] else f"Stream {stream_id}"
            logger.info(f"🎥 Membuka PUBLIC CCTV: {stream_url}")
        else:
            if not token:
                logger.warning(f"❌ Stream {stream_id} bukan public & tidak ada token!")
                await websocket.send_json({"error": "Authentication required for private stream"})
                await websocket.close()
                return

            try:
                jwt_payload = decode_jwt_token(token)
                ws_user_id = jwt_payload.get("sub")
            except jwt.ExpiredSignatureError:
                await websocket.send_json({"error": "Token expired"})
                await websocket.close()
                return
            except Exception as jwt_err:
                logger.error(f"❌ JWT decode error: {jwt_err}")
                await websocket.send_json({"error": "Invalid token"})
                await websocket.close()
                return

            query_private = "SELECT stream_url, location_name, camera_name, user_id FROM private_streams WHERE id = $1"
            row_private = await pool.fetchrow(query_private, stream_id)

            if not row_private:
                logger.warning(f"❌ Stream {stream_id} tidak ditemukan di public maupun private!")
                await websocket.send_json({"error": "Stream not found"})
                await websocket.close()
                return

            if str(row_private["user_id"]) != ws_user_id:
                logger.warning(f"⛔ User {ws_user_id} mencoba akses private stream milik {row_private['user_id']}")
                await websocket.send_json({"error": "Access denied: you are not the owner of this stream"})
                await websocket.close()
                return

            stream_source = "private"
            stream_url = row_private["stream_url"]
            stream_name = row_private["camera_name"] if row_private["camera_name"] else row_private["location_name"] or f"Private Stream {stream_id}"
            logger.info(f"🔒 Membuka PRIVATE CCTV: {stream_url} (owner: {ws_user_id})")

    except Exception as e:
        logger.error(f"❌ Error Database: {e}")
        await websocket.close()
        return

    q = asyncio.Queue(maxsize=5)
    loop = asyncio.get_running_loop()
    state = {"running": True}

    def put_to_queue(item):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def open_capture(url: str, timeout_ms: int = 8000) -> cv2.VideoCapture:
        """Buka VideoCapture dengan timeout agar tidak blokir thread terlalu lama."""
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms)
        cap.open(url)
        return cap

    def video_thread():
        tracker = ByteTrackTracker(TrackingConfig())
        roi_manager = ROIManager()
        feature_extractor = FeatureExtractor(FeatureExtractionConfig(), roi_manager)

        cap = open_capture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0 or video_fps > 120:
            video_fps = 30.0
        frame_duration = 1.0 / video_fps

        frame_count = 0
        skip_rate = 2
        prev_alert_state = "normal"
        consecutive_failures = 0
        MAX_FAILURES_BEFORE_RECONNECT = 5  # Toleransi frame kosong sebelum reconnect

        while state["running"]:
            start_waktu = time.time()
            if not cap.isOpened():
                logger.warning("🔄 Reconnecting ke CCTV...")
                cap = open_capture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                # Beri waktu HLS stream untuk buffer
                time.sleep(3)
                consecutive_failures = 0
                continue
            
            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES_BEFORE_RECONNECT:
                    logger.warning(f"⚠️ {consecutive_failures} frame kosong berturut-turut, reconnecting...")
                    cap.release()
                    consecutive_failures = 0
                    # Backoff: makin sering gagal, makin lama tunggu
                    time.sleep(3)
                else:
                    # Belum perlu reconnect, mungkin HLS sedang buffering
                    time.sleep(0.5)
                continue
            
            consecutive_failures = 0  # Reset counter jika frame berhasil didapat

            frame_count += 1
            if frame_count % skip_rate != 0:
                elapsed_time = time.time() - start_waktu
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            if model is None:
                continue

            results = model(frame, verbose=False, conf=0.45, iou=0.45, imgsz=480, device=device, classes=[0, 1, 2, 3, 4])
            
            detections = []
            if results and results[0].boxes:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    x1, y1, x2, y2 = box
                    class_name = VEHICLE_CLASSES.get(int(class_id), 'unknown')
                    
                    if class_name == 'unknown':
                        continue
                    if class_name == 'pickup':
                        class_name = 'truck'
                    
                    detections.append(Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name
                    ))

            track_result = tracker.update(detections, frame, frame_count)
            
            counts = {"person": 0, "motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
            for det in detections:
                if det.class_name in counts:
                    counts[det.class_name] += 1

            features = feature_extractor.extract_frame_features(track_result.tracks, frame_count, time.time())
            total_vehicle_direct = counts["motorcycle"] + counts["car"] + counts["bus"] + counts["truck"]
            features.vehicle_count = total_vehicle_direct

            status_jalan = predict_density(features)

            annotated_frame = results[0].plot()
            _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            rasio = 0.0

            payload = {
                "type": "frame_update",
                "stream_id": stream_id,
                "counts": counts,
                "person_vehicle_ratio": rasio,
                "density_status": status_jalan,
                "average_speed": round(features.average_speed, 2),
                "road_occupancy": round(features.road_occupancy, 2),
                "congestion_index": round(features.congestion_index, 2),
                "frame_base64": frame_b64,
                "frame": f"data:image/jpeg;base64,{frame_b64}"
            }

            current_alert_state = "dense" if status_jalan in ["High Density", "Anomaly"] else "normal"
            vehicle_counts_only = {k: v for k, v in counts.items() if k != "person"}
            dominant_vehicle = max(vehicle_counts_only, key=vehicle_counts_only.get) if any(vehicle_counts_only.values()) else "kendaraan"
            dominant_label = {"car": "Mobil", "motorcycle": "Motor", "truck": "Truk", "bus": "Bus"}.get(dominant_vehicle, dominant_vehicle)

            waktu_str = datetime.datetime.now().strftime("%H:%M WIB")

            if current_alert_state == "dense" and prev_alert_state == "normal":
                alert_msg = (
                    f"🚨 KEPADATAN TINGGI terdeteksi di {stream_name} pukul {waktu_str}. "
                    f"Total {total_vehicle_direct} kendaraan | Dominan: {dominant_label}."
                )
                payload["alert"] = {
                    "triggered": True,
                    "type": status_jalan,
                    "message": alert_msg
                }
                logger.info(f"🔔 Alert RAMAI dikirim: {alert_msg}")
                prev_alert_state = "dense"

            elif current_alert_state == "normal" and prev_alert_state == "dense":
                alert_msg = (
                    f"✅ Kondisi jalan di {stream_name} sudah kembali normal pukul {waktu_str}. "
                    f"Total kendaraan turun ke {total_vehicle_direct}."
                )
                payload["alert"] = {
                    "triggered": True,
                    "type": "cleared",
                    "message": alert_msg
                }
                logger.info(f"🔔 Alert SEPI dikirim: {alert_msg}")
                prev_alert_state = "normal"

            end_waktu = time.time()
            waktu_proses_detik = end_waktu - start_waktu
            latency_ms = waktu_proses_detik * 1000
            fps = 1.0 / waktu_proses_detik if waktu_proses_detik > 0 else 0.0

            loop.call_soon_threadsafe(put_to_queue, (payload, counts, latency_ms, fps, stream_source))
            
            elapsed_time = time.time() - start_waktu
            sleep_time = frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    t = threading.Thread(target=video_thread, daemon=True)
    t.start()

    try:
        while True:
            payload, counts, latency_ms, fps, src = await q.get()
            asyncio.create_task(save_to_db(payload, counts, stream_source=src))
            await websocket.send_json(payload)
            logger.info(f"📊 YOLO 8s Latency: {latency_ms:.1f} ms | Speed: {fps:.1f} FPS")

    except WebSocketDisconnect:
        logger.info(f"❌ Client terputus dari stream: {stream_id}")
    except Exception as e:
        logger.error(f"⚠️ Error: {e}")
    finally:
        state["running"] = False
        logger.info("🛑 Stream CCTV ditutup.")
