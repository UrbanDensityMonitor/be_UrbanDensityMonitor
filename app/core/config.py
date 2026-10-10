from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_jwt_secret: str
    database_url: str
    
    # Daftar CORS Origins
    cors_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5050",
        "http://127.0.0.1:5050",
        "http://localhost:5500",
        "https://urbandensitymonitor.web.id",
        "https://www.urbandensitymonitor.web.id"
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore" # Mengabaikan variabel env tambahan yang tidak didefinisikan

settings = Settings()
