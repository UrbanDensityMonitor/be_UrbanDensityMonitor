"""
Configuration management menggunakan Pydantic BaseSettings
Centralized environment variables & application settings
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    
    Cara menggunakan:
        from app.core.config import settings
        print(settings.supabase_url)
    """
    
    # Supabase Configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_anon_key: str = Field(..., description="Supabase anon key")
    supabase_service_role_key: str = Field(..., description="Supabase service role key")
    supabase_jwt_secret: str = Field(..., description="JWT secret untuk verification")
    database_url: str = Field(..., description="PostgreSQL connection string")
    
    # Application Configuration
    environment: str = Field(default="development", description="Environment: development/production/testing")
    debug: bool = Field(default=False, description="Debug mode")
    
    # ML Model Configuration
    yolo_model_path: str = Field(default="app/models/best.pt", description="Path to YOLO model")
    density_model_path: str = Field(default="app/models/density_cluster_model.pkl", description="Path to density clustering model")
    
    # Detection Configuration
    detection_confidence: float = Field(default=0.45, description="YOLO confidence threshold")
    detection_iou: float = Field(default=0.45, description="YOLO IOU threshold")
    detection_image_size: int = Field(default=480, description="YOLO inference image size")
    
    # Video Processing Configuration
    video_skip_rate: int = Field(default=2, description="Process every Nth frame")
    video_buffer_size: int = Field(default=1, description="OpenCV buffer size")
    jpeg_quality: int = Field(default=60, description="JPEG compression quality for streaming")
    
    # Density Thresholds
    low_density_max: int = Field(default=10, description="Max vehicles untuk Low Density")
    medium_density_max: int = Field(default=20, description="Max vehicles untuk Medium Density")
    high_density_min: int = Field(default=20, description="Min vehicles untuk High Density")
    anomaly_rain_min: int = Field(default=50, description="Min vehicles untuk Anomaly saat hujan")
    
    # Feature Flags
    enable_face_detection: bool = Field(default=False, description="Enable face detection feature")
    enable_telegram_alerts: bool = Field(default=False, description="Enable Telegram notifications")
    
    # Telegram Configuration (optional)
    telegram_bot_token: Optional[str] = Field(default=None, description="Telegram bot token")
    telegram_chat_id: Optional[str] = Field(default=None, description="Telegram chat ID")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
settings = Settings()
