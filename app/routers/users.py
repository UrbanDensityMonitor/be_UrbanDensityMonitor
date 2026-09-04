"""
Router untuk User Management endpoints (Admin only)
Semua SQL ada di repository layer (Clean Architecture)
"""
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from app.auth.jwt_handler import verify_jwt
from app.core.dependencies import get_user_repository
from app.repositories.user_repository import UserRepository

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/users",
    tags=["User Management (Admin)"]
)


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


def _serialize_row(row: dict) -> dict:
    """Serialize record: UUID -> str, datetime -> ISO string"""
    item = dict(row)
    if item.get("id") is not None:
        item["id"] = str(item["id"])
    for key, value in item.items():
        if isinstance(value, datetime):
            item[key] = value.isoformat()
    return item


async def require_admin(user_info: dict, repo: UserRepository):
    """Raise 403 jika user bukan admin"""
    user_id = user_info.get("sub")
    role = await repo.find_role_by_id(user_id)
    if role != "admin":
        raise HTTPException(status_code=403, detail="⛔ Akses ditolak! Fitur ini khusus untuk Admin.")


@router.get("")
async def get_all_users(
    repo: UserRepository = Depends(get_user_repository),
    user_info: dict = Depends(verify_jwt)
):
    """Get semua user (Admin only)"""
    await require_admin(user_info, repo)

    try:
        rows = await repo.find_all()
    except Exception as e:
        logger.error(f"Error fetching users: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal mengambil data user")

    user_list = [_serialize_row(row) for row in rows]

    return {"data": user_list, "total": len(user_list)}


@router.put("/{user_id}")
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    repo: UserRepository = Depends(get_user_repository),
    user_info: dict = Depends(verify_jwt)
):
    """Update data user (Admin only)"""
    await require_admin(user_info, repo)

    try:
        updated_id = await repo.update(
            user_id,
            full_name=user_data.full_name,
            role=user_data.role,
            is_active=user_data.is_active
        )
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal memperbarui user")

    if not updated_id:
        raise HTTPException(status_code=404, detail="User tidak ditemukan!")

    return {"message": f"✅ Data user {user_id} berhasil diperbarui!"}


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    repo: UserRepository = Depends(get_user_repository),
    user_info: dict = Depends(verify_jwt)
):
    """Delete user (Admin only)"""
    await require_admin(user_info, repo)

    try:
        deleted = await repo.delete(user_id)
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Gagal menghapus user")

    if not deleted:
        raise HTTPException(status_code=404, detail="User tidak ditemukan!")

    return {"message": f"🗑️ User {user_id} berhasil dihapus!"}
