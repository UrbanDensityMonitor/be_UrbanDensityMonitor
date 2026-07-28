import numpy as np
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "density_cluster_model.pkl")

kmeans_model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        kmeans_model = pickle.load(f)

# ============================================================
# 🔧 THRESHOLD KONFIGURASI DENSITY (Ubah nilai di sini)
# ============================================================
# MODE NORMAL (production):
#   LOW_DENSITY_MAX    = 15   → 0–14 kendaraan  = Low Density (aman)
#   MEDIUM_DENSITY_MAX = 40   → 15–39 kendaraan = Medium Density (padat)
#   HIGH_DENSITY_MIN   = 40   → ≥40 kendaraan   = High Density (ALERT! 🚨)
#
# MODE TESTING (aktif sekarang — ubah ke nilai di atas jika ingin production):
#   LOW_DENSITY_MAX    = 1    → 0 kendaraan     = Low Density
#   MEDIUM_DENSITY_MAX = 3    → 1–2 kendaraan   = Medium Density
#   HIGH_DENSITY_MIN   = 3    → ≥3 kendaraan    = High Density (ALERT! 🚨)
# ============================================================
LOW_DENSITY_MAX    = 10   # 0–9 kendaraan   = Low Density
MEDIUM_DENSITY_MAX = 20   # 10–19 kendaraan = Medium Density
ANOMALY_RAIN_MIN   = 50   # untuk kondisi hujan

def predict_density(features, is_raining=False):
    vehicle_count = features.vehicle_count
    
    if vehicle_count == 0:
        return "Low Density"

    if is_raining and vehicle_count > ANOMALY_RAIN_MIN:
        return "Anomaly"

    # Hard override berdasarkan jumlah kendaraan — bypass KMeans jika sudah jelas
    # Ini mencegah KMeans salah klasifikasi akibat fitur speed/occupancy = 0
    if vehicle_count >= MEDIUM_DENSITY_MAX:
        return "High Density"
    if vehicle_count < LOW_DENSITY_MAX:
        return "Low Density"

    # Zona abu-abu (LOW_DENSITY_MAX <= count < MEDIUM_DENSITY_MAX): gunakan KMeans
    if kmeans_model is None:
        return "Medium Density"

    input_data = np.array([[
        vehicle_count, 
        features.average_speed, 
        features.road_occupancy, 
        features.congestion_index
    ]])
    
    cluster_id = kmeans_model.predict(input_data)[0]

    if cluster_id == 0: return "Low Density"
    elif cluster_id == 1: return "Medium Density"
    else: return "High Density"

