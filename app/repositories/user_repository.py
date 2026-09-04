"""
Repository untuk User data access (Supabase managed users table)
"""
import logging
from typing import List, Optional, Dict
import asyncpg

logger = logging.getLogger(__name__)


class UserRepository:
    """
    Data access layer untuk users table

    Responsibility:
        - Database queries untuk users
        - Role checking
        - No business logic
    """

    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    async def find_all(self) -> List[Dict]:
        """
        Get all users

        Returns:
            List of user records (tanpa sensitive data)
        """
        query = """
            SELECT id, email, full_name, role, is_active, created_at
            FROM users
            ORDER BY created_at DESC
        """
        try:
            rows = await self.pool.fetch(query)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching users: {e}", exc_info=True)
            raise

    async def find_role_by_id(self, user_id: str) -> Optional[str]:
        """
        Get user role by ID

        Args:
            user_id: User UUID

        Returns:
            Role string atau None jika user tidak ditemukan
        """
        query = "SELECT role FROM users WHERE id = $1"
        try:
            return await self.pool.fetchval(query, user_id)
        except Exception as e:
            logger.error(f"Error fetching role for user {user_id}: {e}", exc_info=True)
            raise

    async def update(
        self,
        user_id: str,
        full_name: Optional[str],
        role: Optional[str],
        is_active: Optional[bool]
    ) -> Optional[str]:
        """
        Update user (hanya field yang tidak None)

        Args:
            user_id: User UUID
            full_name: New full name (None = skip)
            role: New role (None = skip)
            is_active: New active status (None = skip)

        Returns:
            Updated user ID atau None jika tidak ditemukan
        """
        query = """
            UPDATE users
            SET full_name = COALESCE($1, full_name),
                role = COALESCE($2, role),
                is_active = COALESCE($3, is_active)
            WHERE id = $4
            RETURNING id
        """
        try:
            updated_id = await self.pool.fetchval(
                query, full_name, role, is_active, user_id
            )
            if updated_id:
                logger.info(f"Updated user: {user_id}")
            return updated_id
        except Exception as e:
            logger.error(f"Error updating user {user_id}: {e}", exc_info=True)
            raise

    async def delete(self, user_id: str) -> bool:
        """
        Delete user

        Args:
            user_id: User UUID

        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM users WHERE id = $1"
        try:
            result = await self.pool.execute(query, user_id)
            deleted = result == "DELETE 1"
            if deleted:
                logger.info(f"Deleted user: {user_id}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting user {user_id}: {e}", exc_info=True)
            raise
