"""
Repository untuk Alert data access
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional
import asyncpg
from app.schemas.alert import AlertCreate

logger = logging.getLogger(__name__)


class AlertRepository:
    """
    Data access layer untuk alerts table
    
    Responsibility:
        - Database queries untuk alerts
        - CRUD operations
        - Alert filtering & marking
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    # Nilai alert_type yang diizinkan check constraint DB (alerts_alert_type_check)
    ALLOWED_ALERT_TYPES = ("High Density", "Anomaly")

    async def create(self, data: AlertCreate) -> int:
        """
        Create new alert

        Args:
            data: AlertCreate schema

        Returns:
            New alert ID

        Raises:
            ValueError: Jika alert_type tidak diizinkan constraint DB
        """
        if data.alert_type not in self.ALLOWED_ALERT_TYPES:
            raise ValueError(
                f"alert_type '{data.alert_type}' tidak diizinkan DB. "
                f"Harus salah satu dari: {self.ALLOWED_ALERT_TYPES}"
            )
        # Urutan kolom adalah urusan repository (bukan schema)
        values = (
            data.traffic_history_id,
            data.stream_id,
            data.alert_type,
            data.alert_message,
            datetime.now(),
        )
        query = """
            INSERT INTO alerts (
                traffic_history_id, stream_id, alert_type,
                alert_message, created_at
            )
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """
        try:
            alert_id = await self.pool.fetchval(query, *values)
            logger.info(f"Created alert: {alert_id} - {data.alert_type}")
            return alert_id
        except Exception as e:
            logger.error(f"Error creating alert: {e}", exc_info=True)
            raise
    
    async def find_all(
        self,
        stream_id: Optional[str] = None,
        is_read: Optional[bool] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get alerts dengan optional filters
        
        Args:
            stream_id: Filter by stream ID
            is_read: Filter by read status
            limit: Maximum records
            offset: Offset untuk pagination
        
        Returns:
            List of alert records
        """
        query = """
            SELECT *
            FROM alerts
            WHERE ($1::uuid IS NULL OR stream_id = $1)
              AND ($2::boolean IS NULL OR is_read = $2)
            ORDER BY created_at DESC
            LIMIT $3 OFFSET $4
        """
        try:
            rows = await self.pool.fetch(query, stream_id, is_read, limit, offset)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}", exc_info=True)
            raise
    
    async def find_unread(self, stream_id: Optional[str] = None) -> List[Dict]:
        """
        Get unread alerts
        
        Args:
            stream_id: Optional filter by stream
        
        Returns:
            List of unread alerts
        """
        return await self.find_all(stream_id=stream_id, is_read=False)
    
    async def find_by_id(self, alert_id: str) -> Optional[Dict]:
        """
        Get alert by ID
        
        Args:
            alert_id: Alert ID
        
        Returns:
            Alert record or None
        """
        query = "SELECT * FROM alerts WHERE id = $1"
        try:
            row = await self.pool.fetchrow(query, alert_id)
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching alert {alert_id}: {e}", exc_info=True)
            raise
    
    async def mark_as_read(self, alert_id: str) -> bool:
        """
        Mark single alert as read
        
        Args:
            alert_id: Alert ID
        
        Returns:
            True if updated, False if not found
        """
        query = "UPDATE alerts SET is_read = TRUE WHERE id = $1"
        try:
            result = await self.pool.execute(query, alert_id)
            updated = result == "UPDATE 1"
            if updated:
                logger.info(f"Marked alert as read: {alert_id}")
            return updated
        except Exception as e:
            logger.error(f"Error marking alert as read: {e}", exc_info=True)
            raise
    
    async def mark_multiple_as_read(self, alert_ids: List[str]) -> int:
        """
        Mark multiple alerts as read
        
        Args:
            alert_ids: List of alert IDs
        
        Returns:
            Number of updated alerts
        """
        if not alert_ids:
            return 0
        
        query = "UPDATE alerts SET is_read = TRUE WHERE id = ANY($1::uuid[])"
        try:
            result = await self.pool.execute(query, alert_ids)
            # Parse result like "UPDATE 5"
            updated_count = int(result.split()[-1]) if result else 0
            logger.info(f"Marked {updated_count} alerts as read")
            return updated_count
        except Exception as e:
            logger.error(f"Error marking multiple alerts as read: {e}", exc_info=True)
            raise
    
    async def mark_all_as_read(self, stream_id: Optional[str] = None) -> int:
        """
        Mark all alerts as read (optionally filtered by stream)
        
        Args:
            stream_id: Optional stream filter
        
        Returns:
            Number of updated alerts
        """
        query = """
            UPDATE alerts
            SET is_read = TRUE
            WHERE is_read = FALSE
              AND ($1::uuid IS NULL OR stream_id = $1)
        """
        try:
            result = await self.pool.execute(query, stream_id)
            updated_count = int(result.split()[-1]) if result else 0
            logger.info(f"Marked all alerts as read: {updated_count} alerts")
            return updated_count
        except Exception as e:
            logger.error(f"Error marking all alerts as read: {e}", exc_info=True)
            raise
    
    async def count_unread(self, stream_id: Optional[str] = None) -> int:
        """
        Count unread alerts
        
        Args:
            stream_id: Optional stream filter
        
        Returns:
            Number of unread alerts
        """
        query = """
            SELECT COUNT(*)
            FROM alerts
            WHERE is_read = FALSE
              AND ($1::uuid IS NULL OR stream_id = $1)
        """
        try:
            return await self.pool.fetchval(query, stream_id)
        except Exception as e:
            logger.error(f"Error counting unread alerts: {e}", exc_info=True)
            raise
    
    async def delete(self, alert_id: str) -> bool:
        """
        Delete alert
        
        Args:
            alert_id: Alert ID
        
        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM alerts WHERE id = $1"
        try:
            result = await self.pool.execute(query, alert_id)
            deleted = result == "DELETE 1"
            if deleted:
                logger.info(f"Deleted alert: {alert_id}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting alert: {e}", exc_info=True)
            raise
    
    async def delete_older_than_days(self, days: int) -> int:
        """
        Delete alerts older than specified days
        
        Args:
            days: Delete alerts older than this many days
        
        Returns:
            Number of deleted alerts
        """
        query = """
            DELETE FROM alerts
            WHERE created_at < NOW() - INTERVAL '1 day' * $1
        """
        try:
            result = await self.pool.execute(query, days)
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {deleted_count} old alerts")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting old alerts: {e}", exc_info=True)
            raise
