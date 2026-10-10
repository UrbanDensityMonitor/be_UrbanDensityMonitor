from asyncpg import Pool
from typing import List, Dict, Optional
from datetime import datetime

class HistoryRepository:
    def __init__(self, db_pool: Pool):
        self.pool = db_pool

    async def get_private_stream_owner(self, stream_id: str) -> Optional[str]:
        query = "SELECT user_id FROM private_streams WHERE id = $1"
        row = await self.pool.fetchrow(query, stream_id)
        return str(row["user_id"]) if row else None

    async def get_history_by_stream(self, stream_id: str, limit: int, offset: int) -> List[Dict]:
        query = """
        SELECT id, stream_id, person_count, motorcycle_count, car_count, bus_count, truck_count,
               total_vehicle_count, person_vehicle_ratio, density_status, recorded_at,
               average_speed, road_occupancy, congestion_index, stream_source
        FROM traffic_history
        WHERE stream_id = $1
        ORDER BY recorded_at DESC
        LIMIT $2 OFFSET $3
        """
        rows = await self.pool.fetch(query, stream_id, limit, offset)
        return self._format_rows(rows)

    async def get_all_history_for_user(self, user_id: str, limit: int, offset: int) -> List[Dict]:
        query = """
        SELECT id, stream_id, person_count, motorcycle_count, car_count, bus_count, truck_count,
               total_vehicle_count, person_vehicle_ratio, density_status, recorded_at,
               average_speed, road_occupancy, congestion_index, stream_source
        FROM traffic_history
        WHERE stream_source = 'public'
           OR (stream_source = 'private' AND stream_id IN (
               SELECT id FROM private_streams WHERE user_id = $1
           ))
        ORDER BY recorded_at DESC
        LIMIT $2 OFFSET $3
        """
        rows = await self.pool.fetch(query, user_id, limit, offset)
        return self._format_rows(rows)
        
    def _format_rows(self, rows) -> List[Dict]:
        history_list = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["stream_id"] = str(item["stream_id"])
            if isinstance(item["recorded_at"], datetime):
                item["recorded_at"] = item["recorded_at"].isoformat()
            history_list.append(item)
        return history_list
