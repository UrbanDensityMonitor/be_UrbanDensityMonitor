import os
import jwt
import logging
from jwt import PyJWKClient
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SECRET_KEY = os.getenv("SUPABASE_JWT_SECRET")

jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
jwks_client = PyJWKClient(jwks_url)

security = HTTPBearer()


def decode_jwt_token(token: str) -> dict:
    """
    Decode JWT token dengan dukungan HS256 dan ES256/JWKS.
    Bisa dipanggil langsung (misal dari WebSocket) tanpa FastAPI dependency.
    Raises jwt.ExpiredSignatureError atau Exception jika token tidak valid.
    """
    header = jwt.get_unverified_header(token)
    alg = header.get("alg")

    if alg == "ES256":
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256"],
            options={"verify_aud": False}
        )
    else:
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=["HS256"],
            options={"verify_aud": False}
        )

    return payload


def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        return decode_jwt_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="❌ Tiket kedaluwarsa! Silakan login ulang.")
    except Exception as e:
        logger.error(f"🚨 ALASAN SATPAM NOLAK TIKET: {str(e)}")
        raise HTTPException(status_code=401, detail="❌ Tiket palsu atau tidak valid!")
