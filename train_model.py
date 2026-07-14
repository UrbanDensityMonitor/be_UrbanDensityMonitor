import numpy as np
from sklearn.cluster import KMeans
import pickle
import os

print("Membuka Kelas Pelatihan untuk Manajer AI di Laptop Lokal...")

# Fitur: [vehicle_count, average_speed, road_occupancy, congestion_index]
data_normal = np.array([
    # Low Density / Free Flow
    [5, 60, 10, 0.1], [10, 55, 15, 0.2], [12, 50, 20, 0.2], 
    # Medium Density / Normal Flow
    [25, 40, 40, 0.4], [30, 35, 50, 0.5], [35, 30, 60, 0.6], 
    # High Density / Congestion
    [60, 15, 80, 0.8], [80, 10, 90, 0.9], [100, 5, 95, 1.0]
])

X_train = data_normal

print("Sedang menyusun rumus pola kemacetan (K-Means) berbasis fitur Scaffold...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train)

# Pastikan cluster terurut dari kepadatan rendah ke tinggi (berdasarkan kolom pertama = vehicle_count)
sorted_indices = np.argsort(kmeans.cluster_centers_[:, 0])
kmeans.cluster_centers_ = kmeans.cluster_centers_[sorted_indices]

# Karena cluster_centers_ sudah diurutkan dari rendah ke tinggi,
# kmeans.predict() secara otomatis akan memprediksi index (0=Low, 1=Medium, 2=High).
# Tidak perlu meng-override fungsi predict() karena bisa menyebabkan error Pickle.

os.makedirs("app/models", exist_ok=True)
with open("app/models/density_cluster_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

print("BUKU PANDUAN BARU (.pkl) VERSI LOKAL SUDAH BERHASIL DICETAK!")
