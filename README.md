# 🏙️ Urban Density Monitor - Backend

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-FF7139?style=for-the-badge&logo=yolo&logoColor=white)](https://ultralytics.com/yolov8)
[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)

Urban Density Monitor adalah sistem backend berbasis AI untuk mendeteksi, menghitung, dan menganalisis tingkat kepadatan lalu lintas dan manusia menggunakan aliran (stream) CCTV secara *real-time*.

> [!IMPORTANT]
> ### 📢 UPDATE TERBARU: Private Streams, WebSocket Auth & Multi-Tenancy
> Bagi tim yang ingin langsung memahami update terbaru tanpa membaca dokumen dari awal, silakan langsung lompat ke bagian **[👉 Rangkuman Perubahan Terbaru](#-rangkuman-perubahan-terbaru-changelog)**.

---

## ⚡ Rangkuman Perubahan Terbaru (Changelog)

Berikut adalah daftar perubahan arsitektur, fitur, dan keamanan backend yang baru saja diterapkan:

### 1. 🔒 Fitur Private CCTV Streams (`/api/private-streams`)
* Pengguna kini dapat menambahkan, melihat, memperbarui, dan menghapus URL CCTV pribadi mereka sendiri (RTSP, HLS, dsb.).
* Setiap CCTV pribadi terikat secara eksklusif ke akun pemilik (`user_id` dari Supabase Auth).
* Endpoint CRUD lengkap tersedia di `/api/private-streams/`.

### 2. 🔑 Autentikasi WebSocket Terpadu (`/ws/live/{stream_id}`)
* **Public Streams**: Tetap dapat diakses secara instan tanpa token.
* **Private Streams**: Mewajibkan parameter query token: `/ws/live/{stream_id}?token=<jwt_token>`.
* Sistem akan memverifikasi apakah token valid dan apakah `sub` (User ID) pada token sesuai dengan pemilik CCTV tersebut. Jika tidak berhak, koneksi ditolak secara otomatis.

### 3. 🛡️ Isolasi Data & Multi-Tenancy (Alerts & Traffic History)
* **History (`/api/history/`)**: Kolom `stream_source` (`public` / `private`) kini otomatis dicatat. User hanya dapat melihat riwayat dari CCTV publik dan CCTV miliknya sendiri. User lain tidak bisa melihat data riwayat dari private CCTV pengguna lain.
* **Alerts (`/api/alerts/`)**: Alert yang dihasilkan dari private CCTV diisolasi. Hanya pemilik CCTV yang dapat melihat dan menandai alert tersebut sebagai sudah dibaca (`is_read`).

### 4. 🔏 Dukungan Algoritma JWT Modern (ES256 & HS256)
* Backend sekarang mendukung decoding token Supabase baik dengan algoritma simetris (`HS256` via JWT Secret) maupun asimetris (`ES256` via Public Key).
* Handler JWT otomatis mendeteksi header algoritma token yang dikirimkan klien.

---

## ✨ Fitur Utama

- **Real-Time Video Analytics**: Memproses stream CCTV (HLS/m3u8, RTSP) menggunakan OpenCV.
- **AI Object Detection**: Mendeteksi kendaraan dan pejalan kaki menggunakan model YOLOv8.
- **Density Clustering**: Menggunakan model *Machine Learning* (K-Means) untuk mengklasifikasikan tingkat kepadatan (Low, Medium, High, Anomaly).
- **WebSocket Streaming**: Mengirimkan hasil deteksi dan metadata ke klien web secara *real-time*.
- **Private Stream Management**: Mendukung CCTV publik maupun CCTV privat milik user secara terisolasi.
- **Asynchronous Architecture**: Dibangun dengan FastAPI dan Asyncpg untuk performa I/O tinggi yang *non-blocking*.
- **Secure Authentication**: Melindungi API dengan JWT verification (mendukung HS256 & ES256) terintegrasi dengan Supabase Auth.
- **Hardware Agnostic**: Deteksi otomatis berjalan di GPU (CUDA) untuk performa tinggi atau jatuh kembali ke CPU (fallback).

## 🏗️ Arsitektur Sistem

- **Framework**: FastAPI
- **Database**: PostgreSQL (Supabase) via `asyncpg`
- **Computer Vision**: OpenCV (`cv2`)
- **Machine Learning**: Ultralytics YOLOv8 & Scikit-Learn
- **Concurrency**: `asyncio` & `threading` (Mencegah *blocking* saat pemrosesan frame video)
- **Deployment**: Dockerized

---

## 🚀 Panduan Memulai

### 1. Prasyarat Sistem

- Python 3.10+
- FFmpeg (wajib untuk membaca format video HLS/.m3u8)
- PostgreSQL / Akun Supabase

### 2. Instalasi Lokal

```bash
# Clone repositori
git clone https://github.com/isann2935/be_UrbanDensityMonitor.git
cd be_UrbanDensityMonitor

# Buat Virtual Environment
python3 -m venv venv
source venv/bin/activate  # Untuk Windows: venv\Scripts\activate

# Install dependensi
pip install -r requirements.txt
```

### 3. Konfigurasi Environment

Buat file `.env` di *root* direktori dan isi konfigurasi berikut:

```env
SUPABASE_URL="https://[PROJECT_ID].supabase.co"
SUPABASE_KEY="your-anon-or-service-role-key"
SUPABASE_DB_URL="postgresql://[USER]:[PASSWORD]@[HOST]:[PORT]/[DATABASE]"

# Autentikasi JWT (Gunakan salah satu atau keduanya):
# 1. Jika Supabase project menggunakan HS256:
SUPABASE_JWT_SECRET="your-jwt-secret-key"

# 2. Jika Supabase project menggunakan asymmetric ES256 (format PEM):
SUPABASE_JWT_PUBLIC_KEY="-----BEGIN PUBLIC KEY-----\n...\n-----END PUBLIC KEY-----"
```

### 4. Menjalankan Server

Gunakan Uvicorn untuk menjalankan server dalam mode *development*:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Akses dokumentasi API interaktif (Swagger UI) di: `http://localhost:8000/docs`

---

## 🐳 Deployment Menggunakan Docker (Disarankan)

Backend ini sudah dikonfigurasi sepenuhnya untuk berjalan di dalam container Docker, menjaga konsistensi environtment *production*.

```bash
# 1. Build Docker image
docker build -t urban-density-backend .

# 2. Jalankan container dengan file .env
docker run -d \
  --name urban-backend \
  --env-file .env \
  -p 8000:8000 \
  --restart unless-stopped \
  urban-density-backend
```

---

## 📡 API Endpoints Utama

### 1. Public Streams & WebSocket
| Method | Endpoint | Fungsi | Auth Required |
|---|---|---|---|
| `GET` | `/api/streams/` | Mendapatkan daftar stream CCTV publik aktif | Ya |
| `POST` | `/api/streams/` | Menambah URL stream CCTV publik baru | Ya |
| `WS` | `/ws/live/{stream_id}` | Buka koneksi WebSocket pemrosesan AI *(Publik: bebas; Privat: wajib `?token=...`)* | Kondisional |

### 2. Private Streams (Baru ✨)
| Method | Endpoint | Fungsi | Auth Required |
|---|---|---|---|
| `GET` | `/api/private-streams/` | Daftar CCTV privat milik user yang sedang login | Ya (Owner) |
| `POST` | `/api/private-streams/` | Tambah CCTV privat baru untuk user yang sedang login | Ya (Owner) |
| `GET` | `/api/private-streams/{id}` | Detail satu CCTV privat milik user | Ya (Owner) |
| `PATCH`| `/api/private-streams/{id}` | Update informasi CCTV privat | Ya (Owner) |
| `DELETE`| `/api/private-streams/{id}` | Hapus CCTV privat | Ya (Owner) |

### 3. Traffic History & Alerts
| Method | Endpoint | Fungsi | Auth Required |
|---|---|---|---|
| `GET` | `/api/history/` | Riwayat deteksi lalu lintas *(Auto filter public + private milik user)* | Ya |
| `GET` | `/api/alerts/` | Mendapatkan daftar peringatan *(Auto filter public + private milik user)* | Ya |
| `PATCH`| `/api/alerts/{id}/read`| Menandai alert sudah dibaca *(Hanya owner/publik yang berhak)* | Ya |

---

## 🛠️ Optimasi Performa (Catatan Teknis)

- **Manajemen Memori Model**: `YOLOv8` dan model klasifikasi `K-Means` (*density_cluster_model.pkl*) diinisialisasi hanya satu kali di tingkat modul/global, bukan di setiap frame, untuk menghemat RAM dan CPU/GPU.
- **Isolasi Thread**: Proses pengambilan *frame* CCTV melalui OpenCV sangat memakan CPU dan berpotensi memblokir *event loop* asinkron FastAPI. Oleh karena itu, logika `cap.read()` dijalankan pada ruang lingkup `threading.Thread` yang murni, terhubung ke *event loop* melalui `asyncio.Queue()`.
- **CORS Configured**: CORS diizinkan untuk mode pengembangan (`localhost:5500`, `localhost:5050`) dan mode production (`*.urbandensitymonitor.web.id`).

---
*Dibuat untuk kebutuhan operasional Urban Density Monitor.*
