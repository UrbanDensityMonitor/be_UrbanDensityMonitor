# Ringkasan Refactoring Backend (FastAPI)

File ini merangkum semua perubahan dan optimasi skala besar yang telah dilakukan pada folder `be_UrbanDensityMonitor`.

## 1. Migrasi ke Arsitektur Modular (API v1)
- **Restrukturisasi Route**: Seluruh route yang sebelumnya tersebar atau bertumpuk dipindahkan ke dalam struktur versi `app/api/v1/endpoints/`.
- **Penghancuran God Object `main.py`**: File `main.py` yang awalnya berisi logika koneksi, WebSocket, dan route (~400 baris) dibersihkan hingga kurang dari 40 baris. `main.py` kini murni berfungsi sebagai titik masuk (entry point) aplikasi.
- **Pemisahan Logika WebSocket**: Seluruh logika berat untuk WebSocket dan thread CCTV (YOLO, tracking, eksekusi pipeline) diekstrak ke dalam file khusus `app/api/v1/endpoints/websockets.py`.

## 2. Penerapan Repository Pattern (Isolasi Database)
- **Menghapus Query Raw SQL dari Controller**: Sebelumnya, query `asyncpg` SQL mentah ditulis langsung di dalam file route. Sekarang, semua logika database dipindahkan ke folder `app/db/repositories/`.
- **Repository yang Dibuat**:
  - `alert_repository.py`
  - `history_repository.py`
  - `stream_repository.py`
  - `private_stream_repository.py`
  - `user_repository.py`
- **Dependency Injection**: Route kini hanya memanggil repository melalui fungsi `Depends(get_db)`, membuat kode API sangat bersih dan mudah untuk di-test.

## 3. Peningkatan Stabilitas WebSocket & OpenCV
- **Timeout HLS Stream**: Menambahkan konfigurasi timeout pada FFmpeg OpenCV (`CAP_PROP_OPEN_TIMEOUT_MSEC` & `CAP_PROP_READ_TIMEOUT_MSEC`) sebesar 8 detik di `websockets.py` untuk mencegah thread tersangkut (blocking) hingga 30 detik saat stream mati/buffering.
- **Backoff & Retry Logic**: Memperbaiki logika reconnect CCTV agar tidak terlalu agresif (memberikan jeda 3 detik) sehingga aliran video berformat HLS (`.m3u8`) memiliki waktu untuk melakukan buffering.

## 4. Pembersihan Machine Learning Pipeline (Fase 5)
- **Refactoring `pipeline.py`**: Mengurai class raksasa `TrafficStatePipeline` (yang sebelumnya memiliki ~800 baris kode).
- **Ekstraksi Visualisasi & Reporting**: Memindahkan ratusan baris kode pembuatan grafik, plot (heatmap, matplotlib), dan tabel LaTeX ke class khusus `PipelineVisualizer` dan `PipelineReporter` di dalam file baru `app/services/core_ml/pipeline_exporters.py`.
