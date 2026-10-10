from asyncpg import Pool
from typing import List, Dict, Optional
from datetime import datetime

class AlertRepository:
    def __init__(self, db_pool: Pool):
        self.pool = db_pool

    async def get_private_stream_owner(self, stream_id: str) -> Optional[str]:
        query = "SELECT user_id FROM private_streams WHERE id = $1"
        row = await self.pool.fetchrow(query, stream_id)
        return str(row["user_id"]) if row else None

    async def get_alert(self, alert_id: str) -> Optional[Dict]:
        query = "SELECT id, stream_id FROM alerts WHERE id = $1"
        row = await self.pool.fetchrow(query, alert_id)
        return dict(row) if row else None

    async def get_alerts_by_stream(self, stream_id: str, is_read: Optional[bool], limit: int) -> List[Dict]:
        query = "SELECT id, traffic_history_id, stream_id, alert_type, alert_message, is_read, created_at FROM alerts WHERE stream_id = $1"
        params = [stream_id]
        counter = 2
        
        if is_read is not None:
            query += f" AND is_read = ${counter}"
            params.append(is_read)
            counter += 1

        query += f" ORDER BY created_at DESC LIMIT ${counter}"
        params.append(limit)

        rows = await self.pool.fetch(query, *params)
        return self._format_rows(rows)

    async def get_all_alerts_for_user(self, user_id: str, is_read: Optional[bool], limit: int) -> List[Dict]:
        query = """SELECT id, traffic_history_id, stream_id, alert_type, alert_message, is_read, created_at
        FROM alerts
        WHERE (
            stream_id IN (SELECT id FROM streams)
            OR stream_id IN (SELECT id FROM private_streams WHERE user_id = $1)
        )"""
        params = [user_id]
        counter = 2

        if is_read is not None:
            query += f" AND is_read = ${counter}"
            params.append(is_read)
            counter += 1

        query += f" ORDER BY created_at DESC LIMIT ${counter}"
        params.append(limit)

        rows = await self.pool.fetch(query, *params)
        return self._format_rows(rows)

    async def mark_as_read(self, alert_id: str):
        query = "UPDATE alerts SET is_read = true WHERE id = $1"
        await self.pool.execute(query, alert_id)

    def _format_rows(self, rows) -> List[Dict]:
        alert_list = []
        for row in rows:
            item = dict(row)
            item["id"] = str(item["id"])
            item["traffic_history_id"] = str(item["traffic_history_id"])
            item["stream_id"] = str(item["stream_id"])
            if isinstance(item["created_at"], datetime):
                item["created_at"] = item["created_at"].isoformat()
            alert_list.append(item)
        return alert_list
