"""
Shared pytest fixtures & config.
"""
import os
import sys
from pathlib import Path

# Pastikan root project ada di sys.path agar `import app` bekerja
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment minimal untuk import app.core.config (tidak butuh DB asli)
os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
os.environ.setdefault("SUPABASE_ANON_KEY", "test-anon-key")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
os.environ.setdefault("SUPABASE_JWT_SECRET", "test-jwt-secret")
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
