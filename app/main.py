"""
Urban Density Monitor - Main Application
Refactored dengan Clean Architecture Pattern
"""
import logging
from datetime import datetime
from fastapi import FastAPI

# Core configurations
from app.core.config import settings
from app.core.lifespan import lifespan

# Middlewares
from app.middlewares.cors import setup_cors
from app.middlewares.error_handler import setup_error_handlers

# Routers
from app.routers import streams, history, alerts, users
from app.routers import websocket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# INITIALIZE FASTAPI
# ═══════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Urban Density Monitor API",
    description="AI-powered traffic density monitoring system",
    version="2.0.0",
    lifespan=lifespan,
    # Path harus persis — matikan redirect 307 (redirect cross-origin
    # membuang Authorization header di browser → false 401)
    redirect_slashes=False
)

# ═══════════════════════════════════════════════════════════════════════════
# SETUP MIDDLEWARES
# ═══════════════════════════════════════════════════════════════════════════

setup_cors(app)
setup_error_handlers(app)

# ═══════════════════════════════════════════════════════════════════════════
# INCLUDE ROUTERS
# ═══════════════════════════════════════════════════════════════════════════

app.include_router(streams.router)
app.include_router(history.router)
app.include_router(alerts.router)
app.include_router(users.router)
app.include_router(websocket.router)

# ═══════════════════════════════════════════════════════════════════════════
# ROOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/")
def read_root():
    """API root endpoint"""
    return {
        "message": "✅ Urban Density Monitor API",
        "version": "2.0.0",
        "status": "running",
        "environment": settings.environment,
        "docs_url": "/docs"
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint untuk monitoring

    Returns:
        Health status + database connectivity + timestamp
    """
    from app.db.asyncpg_client import get_db_pool
    return {
        "status": "healthy",
        "database": "connected" if get_db_pool() is not None else "disconnected",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.get("/info")
async def get_info():
    """
    Get application information
    
    Returns:
        Application configuration info (non-sensitive)
    """
    return {
        "app_name": "Urban Density Monitor",
        "version": "2.0.0",
        "environment": settings.environment,
        "features": {
            "yolo_model": settings.yolo_model_path,
            "density_model": settings.density_model_path,
            "detection_confidence": settings.detection_confidence,
            "video_skip_rate": settings.video_skip_rate
        },
        "thresholds": {
            "low_density_max": settings.low_density_max,
            "medium_density_max": settings.medium_density_max,
            "high_density_min": settings.high_density_min
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# APPLICATION STARTUP MESSAGE
# ═══════════════════════════════════════════════════════════════════════════

logger.info("=" * 70)
logger.info("🏙️  URBAN DENSITY MONITOR API - VERSION 2.0.0")
logger.info("=" * 70)
logger.info(f"📍 Environment: {settings.environment}")
logger.info(f"🐛 Debug Mode: {settings.debug}")
logger.info("=" * 70)
