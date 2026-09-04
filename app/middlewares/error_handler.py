"""
Global error handling middleware
"""
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from app.core.config import settings

logger = logging.getLogger(__name__)


async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global error handler untuk semua uncaught exceptions
    
    Args:
        request: Request object
        exc: Exception yang terjadi
    
    Returns:
        JSONResponse dengan error details
    """
    # Log error dengan context
    logger.error(
        f"Unhandled error: {exc}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else "unknown"
        },
        exc_info=True
    )
    
    # Return consistent error response
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "type": "InternalServerError",
                "message": "Terjadi kesalahan pada server",
                "details": str(exc) if settings.debug else None
            }
        }
    )


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handler untuk HTTPException
    
    Args:
        request: Request object
        exc: HTTPException
    
    Returns:
        JSONResponse dengan HTTP error details
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": {
                "type": "HTTPError",
                "message": exc.detail,
                "status_code": exc.status_code
            }
        }
    )


def setup_error_handlers(app: FastAPI) -> None:
    """
    Register all error handlers ke FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(Exception, global_error_handler)
    app.add_exception_handler(HTTPException, http_error_handler)
