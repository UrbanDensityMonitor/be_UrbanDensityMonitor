"""
Repository untuk Traffic History data access
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime, date
import asyncpg
from app.schemas.traffic import TrafficHistoryCreate

logger = logging.getLogger(__name__)


class TrafficHistoryRepository:
    """
    Data access layer untuk traffic_history table
    
    Responsibility:
        - Database queries untuk traffic history
        - Insert, select, analytics queries
        - No business logic
    """
    
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool
    
    async def create(self, data: TrafficHistoryCreate) -> int:
        """
        Create new traffic history record

        Args:
            data: TrafficHistoryCreate schema

        Returns:
            New record ID
        """
        # Urutan kolom adalah urusan repository (bukan schema)
        values = (
            data.stream_id,
            data.person_count,
            data.motorcycle_count,
            data.car_count,
            data.bus_count,
            data.truck_count,
            data.total_vehicle_count,
            data.person_vehicle_ratio,
            data.density_status,
            data.average_speed,
            data.road_occupancy,
            data.congestion_index,
            datetime.now(),
        )
        query = """
            INSERT INTO traffic_history (
                stream_id, person_count, motorcycle_count, car_count,
                bus_count, truck_count, total_vehicle_count,
                person_vehicle_ratio, density_status, average_speed,
                road_occupancy, congestion_index, recorded_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
            RETURNING id
        """
        try:
            record_id = await self.pool.fetchval(query, *values)
            return record_id
        except Exception as e:
            logger.error(f"Error creating traffic history: {e}", exc_info=True)
            raise
    
    async def find_all(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get all traffic history records dengan pagination

        Args:
            limit: Maximum records to return
            offset: Offset untuk pagination

        Returns:
            List of traffic history records
        """
        query = """
            SELECT *
            FROM traffic_history
            ORDER BY recorded_at DESC
            LIMIT $1 OFFSET $2
        """
        try:
            rows = await self.pool.fetch(query, limit, offset)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching all history: {e}", exc_info=True)
            raise

    async def find_by_stream(
        self,
        stream_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """
        Get traffic history by stream ID dengan pagination

        Args:
            stream_id: Stream UUID
            limit: Maximum records to return
            offset: Offset untuk pagination

        Returns:
            List of traffic history records
        """
        query = """
            SELECT *
            FROM traffic_history
            WHERE stream_id = $1
            ORDER BY recorded_at DESC
            LIMIT $2 OFFSET $3
        """
        try:
            rows = await self.pool.fetch(query, stream_id, limit, offset)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching history for stream {stream_id}: {e}", exc_info=True)
            raise
    
    async def find_by_date_range(
        self,
        stream_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """
        Get traffic history dalam date range tertentu
        
        Args:
            stream_id: Stream UUID
            start_date: Start datetime
            end_date: End datetime
        
        Returns:
            List of traffic history records
        """
        query = """
            SELECT *
            FROM traffic_history
            WHERE stream_id = $1
              AND recorded_at >= $2
              AND recorded_at <= $3
            ORDER BY recorded_at ASC
        """
        try:
            rows = await self.pool.fetch(query, stream_id, start_date, end_date)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching history by date range: {e}", exc_info=True)
            raise
    
    async def get_latest(self, stream_id: str) -> Optional[Dict]:
        """
        Get latest traffic record untuk stream
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            Latest traffic record or None
        """
        query = """
            SELECT *
            FROM traffic_history
            WHERE stream_id = $1
            ORDER BY recorded_at DESC
            LIMIT 1
        """
        try:
            row = await self.pool.fetchrow(query, stream_id)
            return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error fetching latest history: {e}", exc_info=True)
            raise
    
    async def get_hourly_average(
        self,
        stream_id: str,
        target_date: date
    ) -> List[Dict]:
        """
        Get hourly average traffic untuk specific date
        
        Args:
            stream_id: Stream UUID
            target_date: Target date
        
        Returns:
            List of hourly averages
        """
        query = """
            SELECT
                DATE_TRUNC('hour', recorded_at) as hour,
                AVG(total_vehicle_count) as avg_vehicles,
                AVG(average_speed) as avg_speed,
                AVG(road_occupancy) as avg_occupancy,
                COUNT(*) as sample_count
            FROM traffic_history
            WHERE stream_id = $1
              AND DATE(recorded_at) = $2
            GROUP BY hour
            ORDER BY hour
        """
        try:
            rows = await self.pool.fetch(query, stream_id, target_date)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching hourly average: {e}", exc_info=True)
            raise
    
    async def get_peak_hours(
        self,
        stream_id: str,
        target_date: date,
        top_n: int = 5
    ) -> List[Dict]:
        """
        Get peak hours (jam tersibuk) untuk specific date
        
        Args:
            stream_id: Stream UUID
            target_date: Target date
            top_n: Number of peak hours to return
        
        Returns:
            List of peak hour records
        """
        query = """
            SELECT
                DATE_TRUNC('hour', recorded_at) as hour,
                MAX(total_vehicle_count) as max_vehicles,
                AVG(total_vehicle_count) as avg_vehicles
            FROM traffic_history
            WHERE stream_id = $1
              AND DATE(recorded_at) = $2
            GROUP BY hour
            ORDER BY max_vehicles DESC
            LIMIT $3
        """
        try:
            rows = await self.pool.fetch(query, stream_id, target_date, top_n)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching peak hours: {e}", exc_info=True)
            raise
    
    async def delete_older_than(self, cutoff_date: datetime) -> int:
        """
        Delete records older than cutoff date
        Useful untuk cleanup old data
        
        Args:
            cutoff_date: Delete records before this date
        
        Returns:
            Number of deleted records
        """
        query = """
            DELETE FROM traffic_history
            WHERE recorded_at < $1
        """
        try:
            result = await self.pool.execute(query, cutoff_date)
            # Parse result string like "DELETE 42"
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f"Deleted {deleted_count} old traffic history records")
            return deleted_count
        except Exception as e:
            logger.error(f"Error deleting old records: {e}", exc_info=True)
            raise
    
    async def count_by_stream(self, stream_id: str) -> int:
        """
        Count total records untuk stream
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            Total number of records
        """
        query = "SELECT COUNT(*) FROM traffic_history WHERE stream_id = $1"
        try:
            return await self.pool.fetchval(query, stream_id)
        except Exception as e:
            logger.error(f"Error counting records: {e}", exc_info=True)
            raise
