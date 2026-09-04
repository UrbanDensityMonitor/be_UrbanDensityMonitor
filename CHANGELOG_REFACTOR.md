# 📋 CHANGELOG — Refactor Urban Density Monitor Backend

> Rangkuman lengkap semua perubahan selama refactoring (Agustus–September 2026).
> Tujuan besar: project mudah di-maintain, scalable, dan mengikuti standar industri
> (Clean Architecture, Single Responsibility, fail-fast, no duplication).

---

## 🚀 FASE 3 — Security, Stability & Frontend Sync (4 September 2026)

### 1. Perbaikan Kritis Backend (FastAPI)
- **Singleton & State Safety**: `DensityService` diperbaiki agar inisialisasi model selesai sebelum instance di-assign, mencegah crash. Runtime threshold menggunakan instance variables (`self._low_density_max`) alih-alih memutasi `settings` global (Pydantic immutable).
- **Stream Worker DB Guard**: Penambahan timeout 5 detik pada eksekusi DB `_save_to_database` di `stream_worker.py`, dan *early return* jika event loop utama belum siap. Menambah properti publik `is_running`.
- **Lazy JWKS Client**: `jwt_handler.py` kini menggunakan nilai dari `settings` (bukan `os.getenv`), dan JWKS client dibuat secara *lazy* menggunakan `@lru_cache`.
- **Centralized DB Guard**: `dependencies.py` sekarang memiliki proteksi terpusat di `get_pool()`, mengembalikan HTTP **503 Service Unavailable** (bukan 500) jika DB terputus. Pengecekan pool redundan di setiap router telah dihapus.
- **WebSocket Subprotocol Echo**: Deteksi JWT diperbaiki (memastikan format 3 segmen). Token kini diterima melalui header `Sec-WebSocket-Protocol`. Sesuai standar RFC 6455, server meng-*echo* kembali subprotocol tersebut pada `websocket.accept(subprotocol=...)` untuk mencegah browser menutup koneksi (yang memicu banjir error `socket.send()`).
- **Pydantic Data Clamping**: Menambahkan `field_validator` di `TrafficHistoryCreate` untuk meng-*clamp* fitur float (`average_speed`, `road_occupancy`, dll) pada rentang `0.0 - 999.99`. Hal ini mengatasi crash `numeric field overflow` dari database (kolom `NUMERIC(5,2)`).
- **Endpoint Security**: Rute `/ws/active-connections` diproteksi dengan `verify_jwt` karena mengekspos data internal.

### 2. Perbaikan Frontend (Next.js)
- **Cookie Security**: `authService.ts` otomatis menambahkan flag `Secure` bila koneksi HTTPS, dan menyinkronkan durasi `max-age` cookie agar sesuai dengan `expires_at` dari Supabase.
- **Token via Subprotocol**: Memigrasikan pengiriman JWT ke WebSocket. Token tidak lagi dilempar via *query parameter* (yang rentan tercatat di log *proxy*), melainkan melalui parameter *protocols* (header `Sec-WebSocket-Protocol`), tersinkronisasi penuh dengan backend.
- **Robust Error Parsing**: `apiService.ts` disempurnakan untuk membaca properti `error.message` ter-*nested* dari backend (`{ error: { message: "..." } }`), memberikan fallback error HTTP yang lebih rapi alih-alih sekadar "API Request Failed".
- **Strict Middleware Validation**: `middleware.ts` memverifikasi struktur dasar JWT (memiliki 3 segmen dipisah titik) dan panjang minimum sebelum meloloskan request.
- **Race Condition Prevention**: Fungsi `refetch()` di `useTrafficData.ts` ditata agar operasi asinkron `connect()` dijalankan tanpa menahan status disconnect (*fire-and-forget* via `void`).

---

## 🏗️ FASE 2 — Arsitektur Scalable (4 September 2026)

### Item #1: Pecah God Class `websocket_service.py` (379 baris → 3 modul terpisah)

| Modul Baru | Tanggung Jawab |
|---|---|
| `app/services/frame_pipeline.py` | **Pure ML pipeline per frame** (detect → count → track → density → alert → annotate → payload). Tidak menyentuh WS/thread/DB. Menghasilkan `FrameResult` |
| `app/services/stream_worker.py` | **StreamWorker**: 1 worker inference per stream (thread + event loop sendiri + fan-out subscriber + DB write 1x). **StreamWorkerManager**: registry worker per stream_id |
| `app/services/websocket_service.py` (rewrite) | **Thin orchestrator** (~120 baris): hubungkan koneksi WS ke worker, plus `ConnectionManager` (registry koneksi per stream) |

Sebelum: 1 file = 7+ tanggung jawab. Sesudah: 1 file = 1 tanggung jawab (SRP).

### Item #2: Fix Bug DI — Stateful Service kini Singleton via `app.state`

- **Bug lama:** `get_websocket_service()` membuat instance baru per request → `active_connections` tidak pernah terisi → endpoint `/ws/active-connections` selalu 0
- **Fix:** semua stateful service (DetectionService, DensityService, AlertService, StreamWorkerManager, WebSocketService) dibuat **SEKALI di lifespan** dan disimpan di `app.state`; dependencies hanya mengambil
- `AlertService` tidak lagi di-hack jadi singleton via attribute function
- Endpoint `/ws/active-connections` kini akurat: jumlah koneksi per stream + total

### Item #3: StreamWorker Shared per Stream (hemat resource besar)

- **Sebelum:** 1 koneksi WS = 1 pipeline YOLO. 2 user buka CCTV sama = 2x inferensi (di CPU!)
- **Sesudah:** 1 worker per `stream_id` dengan reference counting. Koneksi WS hanya subscribe ke hasil worker
- **DB write 1x per frame per stream** (bukan per penonton) — tidak ada duplikat row hanya karena penonton > 1
- Worker otomatis start saat subscriber pertama datang, otomatis stop saat subscriber terakhir pergi
- Saat app shutdown: semua worker dihentikan bersih (`shutdown_all`)

### Item #4: Pemisahan Modul Riset dari Runtime (`research/`)

Dipindah ke folder `research/` (tetap di repo untuk referensi, tidak di-ship ke production):
`pipeline.py` (790 baris), `analytics/`, `evaluation/`, `visualization/`, `temporal/`,
`utils/`, `clustering/` (DBSCAN), `roi_selector/validator`, `model_loader`, `stream_aggregator`,
`vehicle_detector`, `logger.py`, `config.py` (versi lengkap).

Yang bertahan di `app/services/core_ml/` (dipakai runtime):
`detection/detector.py` (hanya dataclass `Detection`), `tracking/` (byte_tracker, track_manager),
`features/` (feature_extractor + 3 sub-extractor), `roi/roi_manager`, `config.py` (hanya
TrackingConfig + FeatureExtractionConfig).

- Logger core_ml dikonsolidasi → `app/services/core_ml/_logging.py` (standard logging, hapus duplikasi colorlog + file handler per-detik yang membuat folder `logs/` membengkak)
- `main_event_loop` disimpan di `asyncpg_client` saat startup — dipakai worker thread untuk DB ops

### Item #5: Standar Industri Lainnya

| Perubahan | Detail |
|---|---|
| **API versioning** | Semua endpoint pindah ke `/api/v1/...` (streams, history, alerts, users). WebSocket tetap `/ws/live/{id}` (standar umum tidak di-version). **FE harus update base URL!** |
| **`to_tuple()` dipindah** | Dari schemas ke repositories — urutan kolom DB adalah urusan repository, schema murni bentuk data |
| **Folder `tests/`** | 17 unit tests (FrameAnnotator, FrameEncoder, VehicleCounts, DensityService rule-based, AlertService state transition, ConnectionManager) — `pytest tests/ -v` |
| **`requirements.txt` pinned** | Semua dependency di-pin ke versi terverifikasi + section testing (pytest, pytest-asyncio, httpx) |
| **docker-compose** | Volume mount 5 bobot YOLO root dihapus (bobot sudah tidak ada, model di-bundle via `app/models/best.pt`) |
| **.gitignore** | Tambah `logs/`, `*.log`, `.pytest_cache/` |

---

## 🎯 FASE 1 — Bug Fix & Clean Architecture (31 Agustus 2026)

### 1. Bug Fix Kritis

| Masalah | Akar Penyebab | Perbaikan | File |
|---|---|---|---|
| **Bounding box tidak muncul** di stream CCTV | Frame hanya di-copy polos (`frame.copy()`), hasil deteksi YOLO tidak pernah digambar | Dibuat `FrameAnnotator` (`draw_detections()`) — kotak warna per kelas kendaraan + label `class conf #track_id`, dipanggil di pipeline WS | `app/utils/frame_annotator.py` (BARU), `app/services/websocket_service.py` |
| **DB error** `column "created_at" does not exist` di `traffic_history` | Tabel Supabase memakai kolom `recorded_at`, repository INSERT pakai `created_at` | Semua query repository & schema `TrafficHistoryResponse` diganti ke `recorded_at` | `app/repositories/traffic_history_repository.py`, `app/schemas/traffic.py` |
| **WS error** `Unexpected ASGI message 'websocket.send' after 'websocket.close'` | Saat client disconnect, thread video masih produksi frame & loop tetap kirim ke koneksi mati | (1) Tangkap `WebSocketDisconnect` + `RuntimeError` eksplisit; (2) kirim frame SEBELUM save DB; (3) cleanup: stop flag + `thread.join(5s)` | `app/services/websocket_service.py` |
| **NameError `pool`** di startup (lifespan) | Referensi variabel yang tidak di-import | Ganti ke `get_db_pool()` | `app/core/lifespan.py` |
| **NameError `query_token`** di endpoint WS | Nama parameter endpoint (`token`) tidak cocok dengan body (`query_token`) | Konsisten pakai `token` | `app/routers/websocket.py` |
| **Bocor resource**: `asyncio.run()` dipanggil per frame di thread | Membuat event loop baru tiap frame (mahal) | Dedicated event loop per thread, dibuat sekali & ditutup di `finally` | `app/services/websocket_service.py` |

### 2. Arsitektur & Clean Architecture

| Perubahan | Tujuan | File |
|---|---|---|
| **Repository pattern konsisten** — router `history`, `alerts`, `users` sebelumnya menulis SQL mentah inline; sekarang semua SQL pindah ke repository layer | Router hanya handling HTTP; SQL terpusat di satu layer → mudah diuji & diganti | `app/routers/history.py`, `alerts.py`, `users.py` (refactor total), `app/repositories/user_repository.py` (BARU) |
| **Dedup JWT verification** — sebelumnya 2 implementasi identik (HTTP & WS) | Single source of truth: `verify_token()` core + `verify_jwt()` wrapper FastAPI | `app/auth/jwt_handler.py` |
| **Model tetap `best.pt`** via `settings.yolo_model_path` | Konfigurasi model lewat `.env`, bukan hardcoded | `app/core/config.py` |
| **Pemisahan visualisasi dari logic** — anotasi frame jadi modul tersendiri | Single Responsibility | `app/utils/frame_annotator.py` (BARU) |
| **DB client pakai `settings.database_url`** (bukan `os.getenv` langsung) + `is_db_connected()` | Semua konfigurasi lewat satu pintu `pydantic-settings` | `app/db/asyncpg_client.py` |
| **Perbaikan import rusak** `src.*` → relative import | core_ml sebelumnya tidak bisa di-import (sisa porting project riset) | `app/services/core_ml/features/stream_aggregator.py`, `app/services/core_ml/clustering/stream_dbscan_helper.py` |

### 3. Keamanan

| Perubahan | Tujuan | File |
|---|---|---|
| **JWT WebSocket via header subprotocol** — token tidak lagi wajib di URL query param | Query param tercatat di access log server/proxy/browser history → token bocor. Cara aman: `new WebSocket(url, [token])`. Query param lama tetap didukung (backward-compatible, FE tidak perlu ubah apa pun) | `app/routers/websocket.py` |
| **CORS testing tidak lagi `["*"]`** | `allow_origins=["*"]` + `allow_credentials=True` invalid per spec & berisiko | `app/middlewares/cors.py` |
| **DB pool fail-fast** | Production: app langsung gagal start jika DB down (pesan jelas, container restart menangani). Dev/testing: WARNING + `/health` melaporkan `database: connected/disconnected` | `app/db/asyncpg_client.py`, `app/core/lifespan.py`, `app/main.py` |

### 4. Kebersihan / Maintainability

| Perubahan | Tujuan |
|---|---|---|
| Hapus `app/main.py.backup` (322 baris versi monolitik lama) | Menghilangkan kode mati yang bisa tertukar dengan yang aktif |
| Hapus `app/services/clustering.py` (sudah digantikan `density_service.py`) | Duplikasi logika density prediction |
| Hapus `app/services/yolo_worker.py` (script debug) | Bukan bagian production |
| Tambah `__init__.py` di `auth/`, `db/`, `routers/` | Konsistensi packaging Python, aman untuk tooling & Docker |
| Encoding frame pakai `FrameEncoder.encode_to_data_url()` (sebelumnya string concat manual) | Satu tanggung jawab per util |
| Dependency `get_user_repository()` didaftarkan | Router users via DI, bukan SQL manual |

### 5. Dilakukan oleh User (didukung, sejalan rekomendasi)

- Menghapus `train_model.py`, `test_ws.html`
- Menghapus 5 bobot YOLO root (`yolov8n/s/m/l.pt`, `yolo11n.pt`) — hanya `best.pt` dipakai
- ✅ Terverifikasi: semua import & model tetap jalan setelah penghapusan

---

## 📊 Sebelum vs Sesudah (Ringkas)

| Aspek | Sebelum | Sesudah |
|---|---|---|
| SQL di router | 3 router dengan SQL inline | 0 — semua di repository layer |
| Implementasi verifikasi JWT | 2 (duplikat) | 1 (`verify_token`) |
| Bounding box | Tidak pernah digambar | Digambar per frame (warna per kelas) |
| Kolom timestamp DB | `created_at` (error) | `recorded_at` (sesuai skema Supabase) |
| Error handling WS disconnect | Crash `RuntimeError` | Ditangani rapi, thread di-join |
| Event loop di thread | `asyncio.run()` per frame | 1 loop per thread |
| CORS testing | `["*"]` + credentials | localhost saja |
| DB gagal koneksi saat startup | Diam-diam jalan → 500 saat diakses | Fail-fast (prod) / terlihat jelas (dev) |
| File mati/backup | 3 file + backup monolitik | 0 |

---

## ⚠️ Sisa Pekerjaan (Roadmap Lanjutan)

Semua item roadmap Fase 1 (#1–#5) sudah SELESAI di Fase 2 di atas. Sisa pekerjaan opsional:

1. **Latency alert service** — `AlertService.check_alert` per stream kini shared antar koneksi
   (state per stream, bukan per koneksi) — ini perilaku yang benar, tapi perlu diperhatikan saat testing
2. **Redis pub/sub** untuk arsitektur multi-instance (saat backend di-scale > 1 container,
   StreamWorkerManager perlu koordinasi antar instance)
3. **CI/CD pipeline** (GitHub Actions): run pytest + build Docker di setiap push
4. **Migrasi FE** ke `/api/v1/` — WAJIB dilakukan setelah deploy BE ini (base URL API service di FE)

---

## ✅ Cara Verifikasi Cepat

```bash
# 1. Jalankan unit tests
venv/Scripts/python -m pytest tests/ -v

# 2. Jalankan server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Cek health (harus menampilkan status DB)
curl http://localhost:8000/health

# 4. Cek endpoint sudah /api/v1
curl http://localhost:8000/api/v1/streams/

# 5. Buka CCTV di FE (base URL sudah /api/v1) → bounding box muncul
# 6. Buka CCTV yang sama dari 2 tab → log BE hanya menampilkan 1 StreamWorker
```

---

## 📁 Peta Arsitektur Setelah Fase 2

```
app/
├── main.py                  # Entry point: FastAPI app, router, middleware, health
├── core/                    # config, lifespan (buat app.state singletons), dependencies (DI)
├── auth/                    # JWT verification (verify_token / verify_jwt)
├── db/                      # asyncpg pool (fail-fast, main_event_loop)
├── middlewares/             # CORS (env-aware), global error handler
├── routers/                # HTTP & WS endpoints (TANPA SQL)
│   ├── streams.py           #   /api/v1/streams → StreamService → StreamRepository
│   ├── history.py           #   /api/v1/history → TrafficHistoryRepository
│   ├── alerts.py            #   /api/v1/alerts → AlertRepository
│   ├── users.py             #   /api/v1/users → UserRepository (admin)
│   └── websocket.py         #   /ws/live/{id} → WebSocketService
├── schemas/                 # Pydantic models (murni bentuk data, tanpa logika DB)
├── services/                # Business logic
│   ├── websocket_service.py    # Thin orchestrator + ConnectionManager
│   ├── stream_worker.py        # StreamWorker (shared per stream) + StreamWorkerManager
│   ├── frame_pipeline.py      # Pure ML pipeline per frame
│   ├── detection_service.py   # YOLO wrapper (singleton, best.pt)
│   ├── tracking_service.py    # ByteTrack + feature extraction wrapper
│   ├── density_service.py     # KMeans + rule-based density prediction
│   ├── alert_service.py       # Alert state transition per stream
│   └── core_ml/               # Modul ML runtime only (riset ada di /research)
├── repositories/            # SEMUA SQL + urutan kolom insert ada di sini
├── models/                  # best.pt + density_cluster_model.pkl
└── utils/                   # frame_annotator, frame_encoder, video_processor, logger

research/                    # Modul riset (bukan runtime) — referensi ilmiah
tests/                       # Unit tests (pytest)
docker/                      # Dockerfile + compose
```

**Prinsip:** Router = HTTP only · Service = business logic · Worker/Pipeline = inference ·
Repository = SQL only · Schema = bentuk data · Config = satu pintu · Singleton stateful = app.state.

---

## 🎯 Daftar Perubahan Fase 1 (per Kategori)
