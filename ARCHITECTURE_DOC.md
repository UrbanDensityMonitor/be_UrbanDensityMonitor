# Arsitektur Proyek Backend (Urban Density Monitor)

Proyek ini dibangun menggunakan **FastAPI** dengan mengadopsi prinsip **Clean Architecture** dan **Modular Design**. Tujuannya adalah memisahkan tanggung jawab (separation of concerns), meningkatkan skalabilitas, kemudahan navigasi kode, dan memudahkan kolaborasi tim.

## 1. Struktur Direktori Utama

```text
be_UrbanDensityMonitor/
│
├── app/                        # Root folder untuk seluruh kode aplikasi
│   ├── api/                    # Layer presentasi (Controllers / Routers)
│   ├── auth/                   # Logika keamanan dan autentikasi
│   ├── core/                   # Konfigurasi aplikasi dan sistem
│   ├── db/                     # Layer database (Koneksi & Repositories)
│   ├── services/               # Layer logika bisnis & Machine Learning
│   ├── utils/                  # Helper / Fungsi utilitas
│   └── main.py                 # Entry point utama aplikasi FastAPI
│
├── requirements.txt            # Daftar dependensi library
└── README.md
```

## 2. Penjelasan Tiap Folder

### `app/api/`
Menangani semua rute (endpoints) dan validasi input/output.
- **`v1/endpoints/`**: Berisi file rute modular untuk masing-masing entitas (misal: `streams.py`, `alerts.py`, `users.py`, `websockets.py`). Di layer ini, kode *TIDAK BOLEH* memuat query database secara langsung atau logika bisnis yang rumit.
- **`v1/router.py`**: File ini bertugas mengumpulkan (`include_router`) semua endpoints dari folder `endpoints/` agar bisa didaftarkan secara massal di `main.py`.

### `app/db/`
Menangani segala interaksi dengan database (Supabase PostgreSQL via `asyncpg`).
- **`asyncpg_client.py`**: Menangani pembuatan koneksi pool ke database.
- **`repositories/`**: Mengimplementasikan **Repository Pattern**. Setiap file di sini (contoh: `user_repository.py`) berisi fungsi spesifik yang membungkus query SQL `asyncpg`. Ini memisahkan query SQL dari file endpoints.

### `app/services/`
Tempat untuk logika inti sistem (Business Logic) dan proses Machine Learning.
- **`core_ml/`**: Modul cerdas untuk analisis data lalu lintas dan integrasi model YOLO.
  - `tracking/`: Logika pelacakan objek (misal: `ByteTrackTracker`).
  - `clustering/`: Algoritma clustering data.
  - `analytics/`: Kelas seperti `traffic_state_analyzer.py` untuk memahami data (Offline Discovery Engine).
  - `pipeline.py` & `pipeline_exporters.py`: Orkestrator pipeline ML beserta logika untuk memproduksi laporan dan grafik (hasil prediksi/heatmap).

### `app/core/`
Menyimpan konfigurasi utama, variabel lingkungan, dan pengaturan server.
- **`config.py`**: Mendefinisikan class `Settings` dari Pydantic yang membaca file `.env` secara otomatis dan ketat (termasuk Supabase Credentials dan JWT Secrets).

### `app/auth/`
Menangani validasi JWT token (mendukung algoritma HS256 maupun ES256) untuk memastikan bahwa request atau WebSocket connection diotorisasi oleh Supabase Auth.

## 3. Aliran Data (Data Flow)

Ketika request datang dari Frontend:
1. **Endpoint/Route** (`app/api/v1/endpoints/*.py`) akan menerima request dan melakukan validasi otorisasi.
2. Endpoint memanggil **Repository** (`app/db/repositories/*.py`) untuk membaca/menulis data ke database Supabase.
3. Untuk proses yang rumit seperti WebSocket atau ML, endpoint akan memanggil **Services** (`app/services/`) yang bertugas menjalankan model YOLO atau kalkulasi kepadatan.
4. Data dikembalikan oleh endpoint dalam bentuk JSON (atau via stream/WebSocket) ke frontend.
