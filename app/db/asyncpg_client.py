import asyncpg
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

pool = None

async def init_db_pool():
    global pool
    logger.info("🗄️ Menghubungkan ke Supabase (asyncpg)...")
    try:
        pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=2,
            max_size=10,
            statement_cache_size=0
        )
        logger.info("✅ Database Supabase Terhubung!")
    except Exception as e:
        logger.error(f"❌ Gagal konek ke DB: {e}")

async def close_db_pool():
    global pool
    if pool:
        await pool.close()
        logger.info("🛑 Koneksi Database ditutup.")

# Untuk pemanggilan manual (legacy) jika diperlukan
def get_db_pool():
    return pool

# Dependency Injection untuk FastAPI Endpoint
async def get_db():
    if not pool:
        raise Exception("Database pool is not initialized")
    yield pool
