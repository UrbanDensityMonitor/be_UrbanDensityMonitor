"""
CORS middleware configuration
"""
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings


def get_cors_origins() -> List[str]:
    """
    Get allowed CORS origins berdasarkan environment

    Returns:
        List of allowed origins
    """
    if settings.environment == "development":
        return [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:5050",
            "http://127.0.0.1:5050",
            "http://localhost:5500",
        ]
    elif settings.environment == "production":
        return [
            "https://urbandensitymonitor.web.id",
            "https://www.urbandensitymonitor.web.id",
        ]
    else:
        # Testing environment: localhost saja (wildcard + credentials invalid per spec)
        return [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://localhost:5050",
            "http://localhost:5500",
            "http://127.0.0.1:5050",
        ]


def setup_cors(app: FastAPI) -> None:
    """
    Setup CORS middleware untuk FastAPI app
    
    Args:
        app: FastAPI application instance
    
    Configuration:
        - Environment-aware origins
        - Credentials allowed
        - All methods & headers allowed
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
