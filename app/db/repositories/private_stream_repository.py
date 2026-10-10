from asyncpg import Pool
from typing import List, Dict, Optional
from datetime import datetime

class PrivateStreamRepository:
    def __init__(self, db_pool: Pool):
        self.pool = db_pool

    async def get_all_streams(self, user_id: str) -> List[Dict]:
        query = """
        SELECT id, user_id, camera_name, location_name, stream_url, protocol, stream_type, status, created_at
        FROM private_streams
        WHERE user_id = $1
        ORDER BY created_at DESC
        """
        rows = await self.pool.fetch(query, user_id)
        streams_data = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["user_id"] = str(item["user_id"])
            if isinstance(item.get("created_at"), datetime):
                item["created_at"] = item["created_at"].isoformat()
            streams_data.append(item)
        return streams_data

    async def create_stream(self, user_id: str, camera_name: str, location_name: str, stream_url: str, protocol: str, stream_type: str) -> str:
        query = """
        INSERT INTO private_streams (user_id, camera_name, location_name, stream_url, protocol, stream_type)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id
        """
        new_id = await self.pool.fetchval(query, user_id, camera_name, location_name, stream_url, protocol, stream_type)
        return str(new_id)

    async def update_stream(self, stream_id: str, user_id: str, camera_name: str, location_name: str, stream_url: str, protocol: str, stream_type: str) -> Optional[str]:
        query = """
        UPDATE private_streams
        SET camera_name   = COALESCE($1, camera_name),
            location_name = COALESCE($2, location_name),
            stream_url    = COALESCE($3, stream_url),
            protocol      = COALESCE($4, protocol),
            stream_type   = COALESCE($5, stream_type)
        WHERE id = $6 AND user_id = $7
        RETURNING id
        """
        updated_id = await self.pool.fetchval(query, camera_name, location_name, stream_url, protocol, stream_type, stream_id, user_id)
        return str(updated_id) if updated_id else None

    async def delete_stream(self, stream_id: str, user_id: str) -> Optional[str]:
        query = "DELETE FROM private_streams WHERE id = $1 AND user_id = $2 RETURNING id"
        deleted_id = await self.pool.fetchval(query, stream_id, user_id)
        return str(deleted_id) if deleted_id else None
