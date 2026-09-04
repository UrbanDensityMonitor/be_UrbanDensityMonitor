"""
JWT verification handler terintegrasi dengan Supabase Auth.

Single source of truth untuk verifikasi token:
    - verify_jwt()    : FastAPI dependency untuk HTTP endpoints (Bearer token)
    - verify_token()  : core verification, dipakai HTTP & WebSocket

Konfigurasi diambil dari app.core.config.settings (Pydantic BaseSettings)
sehingga tidak ada duplikasi load_dotenv() dan semua env var sudah tervalidasi.
"""
import logging
from functools import lru_cache

import jwt
from jwt import PyJWKClient
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.config import settings

logger = logging.getLogger(__name__)

# Supabase config — diambil dari settings (sudah divalidasi, tidak bisa None)
_SUPABASE_URL: str = settings.supabase_url
_SECRET_KEY: str = settings.supabase_jwt_secret

security = HTTPBearer()


@lru_cache(maxsize=1)
def _get_jwks_client() -> PyJWKClient:
    """
    Lazy singleton JWKS client.

    Dibuat saat pertama kali dibutuhkan (bukan saat import) sehingga:
    - Unit test dapat berjalan tanpa SUPABASE_URL valid
    - Startup tidak crash jika Supabase JWKS endpoint sementara tidak reachable
    """
    jwks_url = f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json"
    logger.info(f"🔑 Initializing JWKS client: {jwks_url}")
    return PyJWKClient(jwks_url, cache_keys=True)


def verify_token(token: str) -> dict:
    """
    Core JWT verification — mendukung ES256 (via Supabase JWKS) dan HS256 (fallback).

    Args:
        token: JWT token string (raw, tanpa "Bearer " prefix)

    Returns:
        Decoded token payload sebagai dict

    Raises:
        HTTPException 401: Token expired, invalid signature, atau malformed
    """
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "HS256")

        if alg == "ES256":
            # Supabase asymmetric key — verifikasi via JWKS endpoint
            jwks_client = _get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["ES256"],
                options={"verify_aud": False},
            )
        else:
            # Supabase symmetric key (HS256) — verifikasi via shared secret
            payload = jwt.decode(
                token,
                _SECRET_KEY,
                algorithms=["HS256"],
                options={"verify_aud": False},
            )

        return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token kedaluwarsa. Silakan login ulang.",
        )
    except jwt.InvalidTokenError as e:
        logger.warning(f"Token verification failed: {type(e).__name__}: {e}")
        raise HTTPException(
            status_code=401,
            detail="Token tidak valid atau telah dimanipulasi.",
        )
    except Exception as e:
        logger.error(f"Unexpected JWT error: {e}", exc_info=True)
        raise HTTPException(
            status_code=401,
            detail="Autentikasi gagal. Silakan login ulang.",
        )


def verify_jwt(
    credentials: HTTPAuthorizationCredentials = Security(security),
) -> dict:
    """
    FastAPI Security dependency untuk HTTP endpoints (Bearer token dari header).

    Usage:
        @router.get("/protected")
        async def endpoint(user: dict = Depends(verify_jwt)):
            user_id = user["sub"]
    """
    return verify_token(credentials.credentials)
