import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.asyncpg_client import init_db_pool, close_db_pool
from app.api.v1.router import api_router
from app.api.v1.endpoints import websockets
from app.core.config import settings

# Inisialisasi Model AI secara terpusat (saat file di-import)
import app.core.ai_config

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db_pool()
    yield
    await close_db_pool()

app = FastAPI(title="Urban Density Monitor API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mendaftarkan semua endpoint REST API di bawah /api/v1
app.include_router(api_router, prefix="/api/v1")

# Mendaftarkan endpoint WebSocket di bawah /ws/v1
app.include_router(websockets.router, prefix="/ws/v1")

@app.get("/")
def read_root():
    return {"message": "✅ Backend Urban Density Monitor Aktif! (v1)"}
