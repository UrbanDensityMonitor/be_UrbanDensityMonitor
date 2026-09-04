"""
Database client untuk Supabase PostgreSQL (asyncpg connection pool)
"""
import asyncio
import logging
from typing import Optional

import asyncpg

from app.core.config import settings

logger = logging.getLogger(__name__)

pool = None

# Main event loop aplikasi — disimpan saat startup, dipakai thread worker
# untuk menjadwalkan operasi DB async ke loop utama
main_event_loop: Optional[asyncio.AbstractEventLoop] = None


async def init_db_pool():
    """
    Initialize database connection pool

    Behavior:
        - Production: raise exception jika gagal (fail-fast, container restart menangani)
        - Development/testing: log warning & lanjut dengan pool=None
          (endpoint /health melaporkan status DB)
    """
    global pool
    logger.info("🗄️ Menghubungkan ke Supabase (asyncpg)...")
    try:
        # Simpan main event loop — digunakan thread worker untuk schedule DB ops
        global main_event_loop
        try:
            main_event_loop = asyncio.get_running_loop()
            logger.debug("✅ Main event loop captured for thread worker DB access")
        except RuntimeError as loop_err:
            # Ini tidak boleh terjadi di async context, tapi log jika terjadi
            # agar tidak fail silently (DB write dari thread worker akan error)
            logger.error(
                f"❌ Gagal capture main event loop: {loop_err}. "
                "DB write dari thread worker tidak akan berfungsi."
            )

        pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=2,
            max_size=10,
            statement_cache_size=0
        )
        logger.info("✅ Database Supabase Terhubung!")
    except Exception as e:
        pool = None
        if settings.environment == "production":
            # Fail-fast: app tidak boleh jalan tanpa DB di production
            logger.error(f"❌ Gagal konek ke DB (production, fail-fast): {e}")
            raise
        logger.warning(f"⚠️ Gagal konek ke DB (mode {settings.environment}): {e}")


async def close_db_pool():
    """Close database connection pool"""
    global pool
    if pool:
        await pool.close()
        logger.info("🛑 Koneksi Database ditutup.")
        pool = None


def get_db_pool():
    """Get database connection pool (None jika belum terkoneksi)"""
    return pool


def is_db_connected() -> bool:
    """Check apakah database pool sudah terkoneksi"""
    return pool is not None
