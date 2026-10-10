from asyncpg import Pool
from typing import List, Dict, Optional
from datetime import datetime

class UserRepository:
    def __init__(self, db_pool: Pool):
        self.pool = db_pool

    async def get_user_role(self, user_id: str) -> Optional[str]:
        return await self.pool.fetchval("SELECT role FROM users WHERE id = $1", user_id)

    async def get_all_users(self) -> List[Dict]:
        query = "SELECT id, email, full_name, role, is_active, created_at FROM users ORDER BY created_at DESC"
        rows = await self.pool.fetch(query)
        user_list = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            if isinstance(item["created_at"], datetime):
                item["created_at"] = item["created_at"].isoformat()
            user_list.append(item)
        return user_list

    async def update_user(self, user_id: str, full_name: str, role: str, is_active: bool) -> Optional[str]:
        query = """
        UPDATE users
        SET full_name = COALESCE($1, full_name),
            role = COALESCE($2, role),
            is_active = COALESCE($3, is_active)
        WHERE id = $4 RETURNING id
        """
        updated_id = await self.pool.fetchval(query, full_name, role, is_active, user_id)
        return str(updated_id) if updated_id else None

    async def delete_user(self, user_id: str) -> bool:
        query = "DELETE FROM users WHERE id = $1"
        result = await self.pool.execute(query, user_id)
        return result != "DELETE 0"
