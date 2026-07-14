import numpy as np
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "density_cluster_model.pkl")

kmeans_model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        kmeans_model = pickle.load(f)

def predict_density(features, is_raining=False):
    vehicle_count = features.vehicle_count
    
    if vehicle_count == 0:
        return "Low Density"

    if is_raining and vehicle_count > 100: # Adjust threshold for anomaly
        return "Anomaly"

    if kmeans_model is None:
        if vehicle_count < 15: return "Low Density"
        elif vehicle_count < 40: return "Medium Density"
        else: return "High Density"

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

