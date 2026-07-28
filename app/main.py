import asyncio
import cv2
import base64
import time
import datetime
import torch
import logging
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager
from ultralytics import YOLO
from app.routers import streams, history, alerts, users
from app.services.clustering import predict_density
from app.db.asyncpg_client import init_db_pool, close_db_pool, get_db_pool
from fastapi.middleware.cors import CORSMiddleware

async def save_to_db(payload, counts):
    try:
        pool = get_db_pool()
        if not pool: return
        query_history = """
        INSERT INTO traffic_history (
            stream_id, person_count, motorcycle_count, car_count, bus_count, truck_count, 
            total_vehicle_count, person_vehicle_ratio, density_status,
            average_speed, road_occupancy, congestion_index
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12) RETURNING id
        """
        total_v = counts["motorcycle"] + counts["car"] + counts["bus"] + counts["truck"]
        hist_id = await pool.fetchval(query_history,
            payload["stream_id"], counts["person"], counts["motorcycle"], counts["car"], counts["bus"], counts["truck"],
            total_v, payload["person_vehicle_ratio"], payload["density_status"],
            payload.get("average_speed", 0.0), payload.get("road_occupancy", 0.0), payload.get("congestion_index", 0.0)
        )
        if payload["density_status"] in ["High Density", "Anomaly"]:
            query_alert = """
            INSERT INTO alerts (traffic_history_id, stream_id, alert_type, alert_message)
            VALUES ($1, $2, $3, $4)
            """
            await pool.execute(query_alert, hist_id, payload["stream_id"], payload["density_status"], payload["alert"]["message"])
    except Exception as e:
        logger.error(f"❌ DATABASE ERROR: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_pool()
    yield
    await close_db_pool()

app = FastAPI(title="Urban Density Monitor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5050",
        "http://127.0.0.1:5050",
        "http://localhost:5500",
        "https://urbandensitymonitor.web.id",
        "https://www.urbandensitymonitor.web.id"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(streams.router)
app.include_router(history.router)
app.include_router(alerts.router)
app.include_router(users.router)

from app.services.core_ml.config import TrackingConfig, FeatureExtractionConfig
from app.services.core_ml.tracking.byte_tracker import ByteTrackTracker
from app.services.core_ml.features.feature_extractor import FeatureExtractor
from app.services.core_ml.roi.roi_manager import ROIManager
from app.services.core_ml.detection.detector import Detection

# Otomatis menyesuaikan dengan GPU laptop (CUDA untuk NVIDIA RTX, MPS untuk MacOS, atau CPU untuk fallback)
if torch.cuda.is_available():
    device = 'cuda'
    logger.info("🚀 Menggunakan GPU NVIDIA (CUDA) untuk pemrosesan.")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
    logger.info("🚀 Menggunakan GPU Apple (MPS) untuk pemrosesan.")
else:
    device = 'cpu'
    logger.info("⚠️ Menggunakan CPU (Fallback) karena GPU tidak terdeteksi.")

model = YOLO("app/models/best.pt").to(device)
VEHICLE_CLASSES = {0: 'bus', 1: 'car', 2: 'motorcycle', 3: 'pickup', 4: 'truck'}

@app.get("/")
def read_root():
    return {"message": "✅ Backend Urban Density Monitor Aktif!"}

@app.websocket("/ws/live/{stream_id}")
async def websocket_endpoint(websocket: WebSocket, stream_id: str):
    await websocket.accept()
    logger.info(f"🔗 Client terhubung ke stream: {stream_id}")
    pool = get_db_pool()
    if not pool:
        logger.error("❌ Database belum nyambung!")
        await websocket.close()
        return
    try:
        query = "SELECT stream_url, location_name FROM streams WHERE id = $1"
        row = await pool.fetchrow(query, stream_id)
        if not row:
            logger.warning(f"❌ CCTV dengan ID {stream_id} tidak ditemukan!")
            await websocket.close()
            return
        stream_url = row["stream_url"]
        stream_name = row["location_name"] if row["location_name"] else f"Stream {stream_id}"
        logger.info(f"🎥 Membuka CCTV: {stream_url}")

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

    def video_thread():
        tracker = ByteTrackTracker(TrackingConfig())
        roi_manager = ROIManager()
        feature_extractor = FeatureExtractor(FeatureExtractionConfig(), roi_manager)

        cap = cv2.VideoCapture(stream_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Dapatkan FPS asli video untuk menampilkan speed 1.0x
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        # HLS stream kadang mengembalikan nilai aneh seperti 0 atau 90000, kita batasi range yang masuk akal
        if not video_fps or video_fps <= 0 or video_fps > 120:
            video_fps = 30.0  # Default jika FPS tidak terbaca atau tidak wajar
        frame_duration = 1.0 / video_fps

        frame_count = 0
        skip_rate = 2  # Diturunkan dari 5 ke 2 agar video jauh lebih smooth (tampil ~15 FPS)

        # ── State-transition alert ──────────────────────────────────────────────
        # Alert hanya dikirim saat STATUS BERUBAH, bukan setiap frame.
        # "dense"  = High Density / Anomaly  →  kirim alert RAMAI
        # "normal" = Low / Medium            →  kirim alert SUDAH SEPI (jika sebelumnya ramai)
        prev_alert_state = "normal"  # kondisi awal dianggap normal

        while state["running"]:
            start_waktu = time.time()
            if not cap.isOpened():
                logger.warning("🔄 Reconnecting ke CCTV...")
                if cap is not None:
                    cap.release()
                cap = cv2.VideoCapture(stream_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(1)
                continue
            
            ret, frame = cap.read()
            if not ret:
                logger.warning("⚠️ Sinyal CCTV putus (Frame kosong)! Coba sambung ulang...")
                cap.release()
                time.sleep(1)
                continue

            frame_count += 1
            if frame_count % skip_rate != 0:
                # Sleep agar frame yang diskip tetap memakan waktu seperti video aslinya (1.0x speed)
                elapsed_time = time.time() - start_waktu
                sleep_time = frame_duration - elapsed_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                continue

            # Menggunakan custom model best.pt dengan class IDs sendiri (bukan COCO):
            # {0: 'bus', 1: 'car', 2: 'motorcycle', 3: 'pickup', 4: 'truck'}
            # classes=[0, 1, 2, 3, 4] membatasi deteksi hanya pada kendaraan target
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
                        continue # Abaikan objek yang bukan target kita
                    
                    # Map pickup ke truck jika kebetulan ada model yang mendeteksi pickup
                    if class_name == 'pickup':
                        class_name = 'truck'
                    
                    detections.append(Detection(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        confidence=float(conf),
                        class_id=int(class_id),
                        class_name=class_name
                    ))

            # 1. Update Tracker
            track_result = tracker.update(detections, frame, frame_count)
            
            # 2. Hitung counts langsung dari YOLO detections (bukan dari tracker)
            # Tracker butuh min 5 frame sebelum track "aktif" → vehicle_count sering nol
            counts = {"person": 0, "motorcycle": 0, "car": 0, "bus": 0, "truck": 0}
            for det in detections:
                if det.class_name in counts:
                    counts[det.class_name] += 1

            # 3. Extract Features (untuk speed, occupancy, congestion)
            features = feature_extractor.extract_frame_features(track_result.tracks, frame_count, time.time())
            
            # Override vehicle_count di features dengan hitungan langsung dari YOLO
            # agar predict_density mendapat angka yang akurat
            total_vehicle_direct = counts["motorcycle"] + counts["car"] + counts["bus"] + counts["truck"]
            features.vehicle_count = total_vehicle_direct

            # 4. Predict Density (pakai vehicle_count yang sudah di-override)
            status_jalan = predict_density(features)

            # Plot detections dengan kualitas JPEG yang dioptimasi agar ukuran Base64 mengecil
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

            # ── Tentukan state saat ini ─────────────────────────────────────────
            current_alert_state = "dense" if status_jalan in ["High Density", "Anomaly"] else "normal"

            # Hitung kendaraan dominan
            vehicle_counts_only = {k: v for k, v in counts.items() if k != "person"}
            dominant_vehicle = max(vehicle_counts_only, key=vehicle_counts_only.get) if any(vehicle_counts_only.values()) else "kendaraan"
            dominant_label = {"car": "Mobil", "motorcycle": "Motor", "truck": "Truk", "bus": "Bus"}.get(dominant_vehicle, dominant_vehicle)

            waktu_str = datetime.datetime.now().strftime("%H:%M WIB")

            if current_alert_state == "dense" and prev_alert_state == "normal":
                # Transisi: normal → ramai  →  kirim alert KEMACETAN
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
                # Transisi: ramai → normal  →  kirim notifikasi SUDAH SEPI
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

            loop.call_soon_threadsafe(put_to_queue, (payload, counts, latency_ms, fps))
            
            # Sleep untuk frame yang diproses agar sesuai dengan kecepatan aslinya (1.0x)
            elapsed_time = time.time() - start_waktu
            sleep_time = frame_duration - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        cap.release()

    t = threading.Thread(target=video_thread, daemon=True)
    t.start()

    try:
        while True:
            payload, counts, latency_ms, fps = await q.get()
            asyncio.create_task(save_to_db(payload, counts))
            await websocket.send_json(payload)
            logger.info(f"📊 YOLO 8s Latency: {latency_ms:.1f} ms | Speed: {fps:.1f} FPS")

    except WebSocketDisconnect:
        logger.info(f"❌ Client terputus dari stream: {stream_id}")
    except Exception as e:
        logger.error(f"⚠️ Error: {e}")
    finally:
        state["running"] = False
        logger.info("🛑 Stream CCTV ditutup.")
