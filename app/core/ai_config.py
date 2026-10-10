import torch
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = 'cuda'
    logger.info("🚀 Menggunakan GPU NVIDIA (CUDA) untuk pemrosesan.")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = 'mps'
    logger.info("🚀 Menggunakan GPU Apple (MPS) untuk pemrosesan.")
else:
    device = 'cpu'
    logger.info("⚠️ Menggunakan CPU (Fallback) karena GPU tidak terdeteksi.")

try:
    model = YOLO("app/models/best.pt").to(device)
    logger.info("✅ Model YOLO best.pt berhasil dimuat ke memory.")
except Exception as e:
    logger.error(f"❌ Gagal memuat model YOLO: {e}")
    model = None

VEHICLE_CLASSES = {0: 'bus', 1: 'car', 2: 'motorcycle', 3: 'pickup', 4: 'truck'}
