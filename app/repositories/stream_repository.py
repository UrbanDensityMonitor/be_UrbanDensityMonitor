"""
Repository untuk Stream data access
"""
import logging
from typing import List, Optional, Dict
import asyncpg
from app.schemas.stream import StreamCreate, StreamUpdate

logger = logging.getLogger(__name__)


class StreamRepository:
    """
    Data access layer untuk streams table
    
    Responsibility:
        - Database queries untuk streams
        - CRUD operations
        - No business logic
    
    Usage:
        repo = StreamRepository(pool)
        streams = await repo.find_all()
        stream = await repo.find_by_id(stream_id)
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def find_all(self, status: Optional[str] = None) -> List[Dict]:
        """
        Get all streams dengan optional status filter
        
        Args:
            status: Filter by status ('active' or 'inactive')
        
        Returns:
            List of stream records
        """
        query = """
            SELECT id, location_name, stream_url, stream_type, status, created_at
            FROM streams
            WHERE ($1::text IS NULL OR status = $1)
            ORDER BY location_name ASC
        """
        try:
            rows = await self.pool.fetch(query, status)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching streams: {e}", exc_info=True)
            raise
    
    async def find_by_id(self, stream_id: str) -> Optional[Dict]:
        """
        Get stream by ID
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            Stream record or None if not found
        """
        query = """
            SELECT id, location_name, stream_url, stream_type, status, created_at
            FROM streams
            WHERE id = $1
        """
        try:
            row = await self.pool.fetchrow(query, stream_id)
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching stream {stream_id}: {e}", exc_info=True)
            raise
    
    async def create(self, data: StreamCreate) -> str:
        """
        Create new stream
        
        Args:
            data: StreamCreate schema
        
        Returns:
            New stream ID
        """
        query = """
            INSERT INTO streams (location_name, stream_url, stream_type)
            VALUES ($1, $2, $3)
            RETURNING id
        """
        try:
            new_id = await self.pool.fetchval(
                query,
                data.location_name,
                data.stream_url,
                data.stream_type
            )
            logger.info(f"Created stream: {new_id} - {data.location_name}")
            return str(new_id)
        except Exception as e:
            logger.error(f"Error creating stream: {e}", exc_info=True)
            raise
    
    async def update(self, stream_id: str, data: StreamUpdate) -> bool:
        """
        Update stream
        
        Args:
            stream_id: Stream UUID
            data: StreamUpdate schema (only non-None fields updated)
        
        Returns:
            True if updated, False if not found
        """
        # Build dynamic update query
        update_fields = []
        values = []
        param_count = 1
        
        if data.location_name is not None:
            update_fields.append(f"location_name = ${param_count}")
            values.append(data.location_name)
            param_count += 1
        
        if data.stream_url is not None:
            update_fields.append(f"stream_url = ${param_count}")
            values.append(data.stream_url)
            param_count += 1
        
        if data.stream_type is not None:
            update_fields.append(f"stream_type = ${param_count}")
            values.append(data.stream_type)
            param_count += 1
        
        if data.status is not None:
            update_fields.append(f"status = ${param_count}")
            values.append(data.status)
            param_count += 1
        
        if not update_fields:
            return False  # Nothing to update
        
        query = f"""
            UPDATE streams
            SET {', '.join(update_fields)}
            WHERE id = ${param_count}
        """
        values.append(stream_id)
        
        try:
            result = await self.pool.execute(query, *values)
            updated = result == "UPDATE 1"
            if updated:
                logger.info(f"Updated stream: {stream_id}")
            return updated
        except Exception as e:
            logger.error(f"Error updating stream {stream_id}: {e}", exc_info=True)
            raise
    
    async def delete(self, stream_id: str) -> bool:
        """
        Delete stream
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            True if deleted, False if not found
        """
        query = "DELETE FROM streams WHERE id = $1"
        try:
            result = await self.pool.execute(query, stream_id)
            deleted = result == "DELETE 1"
            if deleted:
                logger.info(f"Deleted stream: {stream_id}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting stream {stream_id}: {e}", exc_info=True)
            raise
    
    async def find_active_streams(self) -> List[Dict]:
        """
        Get all active streams
        
        Returns:
            List of active streams
        """
        return await self.find_all(status="active")
    
    async def count(self) -> int:
        """
        Count total streams
        
        Returns:
            Total number of streams
        """
        query = "SELECT COUNT(*) FROM streams"
        try:
            return await self.pool.fetchval(query)
        except Exception as e:
            logger.error(f"Error counting streams: {e}", exc_info=True)
            raise
