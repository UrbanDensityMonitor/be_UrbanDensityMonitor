from asyncpg import Pool
from typing import List, Dict

class StreamRepository:
    def __init__(self, db_pool: Pool):
        self.pool = db_pool

    async def get_all_streams(self) -> List[Dict]:
        query = "SELECT id, location_name, stream_url, stream_type, status FROM streams ORDER BY location_name ASC"
        rows = await self.pool.fetch(query)
        streams_data = [dict(row) for row in rows]
        for s in streams_data:
            s['id'] = str(s['id'])
        return streams_data

    async def add_stream(self, location_name: str, stream_url: str, stream_type: str) -> str:
        query = """
        INSERT INTO streams (location_name, stream_url, stream_type)
        VALUES ($1, $2, $3) RETURNING id
        """
        new_id = await self.pool.fetchval(query, location_name, stream_url, stream_type)
        return str(new_id)

    async def delete_stream(self, stream_id: str):
        query = "DELETE FROM streams WHERE id = $1"
        await self.pool.execute(query, stream_id)
