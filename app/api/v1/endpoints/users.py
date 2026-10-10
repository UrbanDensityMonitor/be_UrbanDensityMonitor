from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from asyncpg import Pool

from app.db.asyncpg_client import get_db
from app.auth.jwt_handler import verify_jwt
from app.db.repositories.user_repository import UserRepository

router = APIRouter(
    prefix="/users",
    tags=["User Management (Admin)"]
)

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None

async def kuasai_akses_admin(user_info: dict, repo: UserRepository):
    user_id = user_info.get("sub")
    role = await repo.get_user_role(user_id)
    if role != "admin":
        raise HTTPException(status_code=403, detail="⛔ Akses ditolak! Fitur ini khusus untuk Admin.")

@router.get("/")
async def get_all_users(
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = UserRepository(db)
    await kuasai_akses_admin(user_info, repo)

    user_list = await repo.get_all_users()
    return {"data": user_list, "total": len(user_list)}

@router.put("/{user_id}")
async def update_user(
    user_id: str, 
    user_data: UserUpdate, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = UserRepository(db)
    await kuasai_akses_admin(user_info, repo)

    updated_id = await repo.update_user(user_id, user_data.full_name, user_data.role, user_data.is_active)
    if not updated_id:
        raise HTTPException(status_code=404, detail="User tidak ditemukan!")
    return {"message": f"✅ Data user {user_id} berhasil diperbarui!"}

@router.delete("/{user_id}")
async def delete_user(
    user_id: str, 
    user_info: dict = Depends(verify_jwt),
    db: Pool = Depends(get_db)
):
    repo = UserRepository(db)
    await kuasai_akses_admin(user_info, repo)

    sukses = await repo.delete_user(user_id)
    if not sukses:
        raise HTTPException(status_code=404, detail="User tidak ditemukan!")
    return {"message": f"🗑️ User {user_id} berhasil dihapus!"}
