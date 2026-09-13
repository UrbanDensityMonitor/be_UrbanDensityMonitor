# Skema Machine Learning - Urban Density Monitoring

Dokumen ini menjelaskan alur kerja dan skema pengambilan keputusan (decision-making) dari sistem Machine Learning yang digunakan dalam project Urban Density Monitoring. Terdapat dua komponen utama dalam clustering data lalu lintas: **K-Means Clustering** (digunakan untuk prediksi density saat ini) dan **DBSCAN Clustering** (sebagai core ML untuk traffic state discovery).

---

## 1. Alur Prediksi Kepadatan Lalu Lintas (K-Means & Heuristic)
Fungsi utama prediksi berjalan pada file `app/services/clustering.py`. Prediksi menggabungkan aturan berbasis threshold (Heuristic) dan model K-Means (`density_cluster_model.pkl`).

### Fitur yang Digunakan:
- `vehicle_count`: Jumlah kendaraan.
- `average_speed`: Kecepatan rata-rata kendaraan.
- `road_occupancy`: Persentase keterisian jalan.
- `congestion_index`: Indeks kemacetan.
- `is_raining`: Status cuaca (hujan/tidak).

### Threshold Konfigurasi (Mode Testing):
- **LOW_DENSITY_MAX**: 10 (0 - 9 kendaraan = Low Density)
- **MEDIUM_DENSITY_MAX**: 20 (10 - 19 kendaraan = Medium Density)
- **ANOMALY_RAIN_MIN**: 50 (Batas kendaraan saat hujan)

### Logika Pengambilan Keputusan:
1. **Pengecekan Kendaraan Kosong:**
   - Jika `vehicle_count == 0` ➔ Kembalikan **"Low Density"**.
2. **Pengecekan Cuaca (Anomali Hujan):**
   - Jika sedang hujan (`is_raining == True`) dan `vehicle_count > 50` ➔ Kembalikan **"Anomaly"**.
3. **Hard Override (Bypass Model):**
   - Jika `vehicle_count >= 20` ➔ Kembalikan **"High Density"**.
   - Jika `vehicle_count < 10` ➔ Kembalikan **"Low Density"**.
4. **Prediksi Model K-Means (Zona Abu-abu: 10 - 19 Kendaraan):**
   - Jika kondisi tidak memenuhi hard override di atas, sistem akan menggunakan model KMeans.
   - Model menerima input array: `[vehicle_count, average_speed, road_occupancy, congestion_index]`.
   - **Hasil Clustering Model:**
     - `Cluster 0` ➔ **"Low Density"**
     - `Cluster 1` ➔ **"Medium Density"**
     - `Lainnya` ➔ **"High Density"**

---

## 2. Traffic State Discovery (DBSCAN)
Modul DBSCAN (berada di `app/services/core_ml/clustering/dbscan_clustering.py`) digunakan untuk pengelompokan yang lebih dinamis untuk menemukan keadaan lalu lintas tanpa jumlah cluster yang ditentukan sebelumnya.

### Fitur yang Digunakan:
Menggunakan berbagai fitur dari dataset (diharapkan mencakup *speed*, *density*, dan *congestion*). Data diskalakan menggunakan `StandardScaler` sebelum diproses.

### Proses Klasifikasi DBSCAN:
1. Algoritma DBSCAN mendeteksi core samples dan noise berdasarkan parameter `eps` (radius) dan `min_samples`.
2. Titik-titik yang masuk ke dalam noise akan diberi label `-1`.
3. Setelah cluster terbentuk, algoritma akan menghitung pusat setiap cluster (cluster centers).
4. **Pemetaan Traffic State berdasarkan Cluster Centers:**
   Pusat dari masing-masing cluster dievaluasi untuk menentukan label semantik state:
   - Jika `congestion > 0.7` atau `speed < 10` ➔ **"Heavy Congestion"**
   - Jika `congestion > 0.4` atau `speed < 30` ➔ **"Moderate Traffic"**
   - Jika `density < 5` dan `speed > 50` ➔ **"Free Flow"**
   - Jika `speed > 40` ➔ **"Normal Flow"**
   - Selain itu ➔ **"Mixed Traffic"**

---

## Kesimpulan
Sistem ini menggunakan pendekatan **Hybrid** (Aturan Heuristik + Machine Learning). Aturan heuristik berfungsi sebagai pengaman untuk kondisi pasti (seperti jalan kosong atau sangat padat), sedangkan K-Means digunakan untuk mendeteksi kepadatan pada rentang zona abu-abu secara presisi. Modul DBSCAN diimplementasikan untuk analisis mendalam dalam mendeteksi dan menemukan keadaan lalu lintas (traffic state discovery).
