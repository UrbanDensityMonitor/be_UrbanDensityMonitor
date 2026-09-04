"""
Stream Service untuk business logic terkait streams
"""
import logging
from typing import List, Optional
from app.repositories.stream_repository import StreamRepository
from app.schemas.stream import StreamCreate, StreamUpdate, StreamResponse

logger = logging.getLogger(__name__)


class StreamService:
    """
    Business logic layer untuk Stream operations
    
    Responsibility:
        - Coordinate between router dan repository
        - Add business rules & validation
        - Transform data if needed
    
    Usage:
        service = StreamService(repository)
        streams = await service.get_all_streams()
    """
    
    def __init__(self, repository: StreamRepository):
        """
        Initialize stream service
        
        Args:
            repository: StreamRepository instance
        """
        self.repo = repository
    
    async def get_all_streams(
        self,
        status_filter: Optional[str] = None
    ) -> List[StreamResponse]:
        """
        Get all streams dengan optional filter
        
        Args:
            status_filter: Filter by status ('active' or 'inactive')
        
        Returns:
            List of StreamResponse objects
        """
        streams = await self.repo.find_all(status=status_filter)
        
        # Convert to response schema
        return [
            StreamResponse(
                id=str(s['id']),
                location_name=s['location_name'],
                stream_url=s['stream_url'],
                stream_type=s['stream_type'],
                status=s['status'],
                created_at=s.get('created_at')
            )
            for s in streams
        ]
    
    async def get_stream_by_id(self, stream_id: str) -> Optional[StreamResponse]:
        """
        Get stream by ID
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            StreamResponse or None if not found
        """
        stream = await self.repo.find_by_id(stream_id)
        
        if not stream:
            return None
        
        return StreamResponse(
            id=str(stream['id']),
            location_name=stream['location_name'],
            stream_url=stream['stream_url'],
            stream_type=stream['stream_type'],
            status=stream['status'],
            created_at=stream.get('created_at')
        )
    
    async def create_stream(self, data: StreamCreate) -> str:
        """
        Create new stream
        
        Args:
            data: StreamCreate schema
        
        Returns:
            New stream ID
        
        Business Rules:
            - Bisa tambahkan validation di sini (e.g., cek duplicate URL)
        """
        # Business logic bisa ditambahkan di sini
        # Contoh: validate stream URL accessibility
        # Contoh: check duplicate location name
        
        stream_id = await self.repo.create(data)
        logger.info(f"✅ Stream created: {stream_id} - {data.location_name}")
        
        return stream_id
    
    async def update_stream(
        self,
        stream_id: str,
        data: StreamUpdate
    ) -> bool:
        """
        Update stream
        
        Args:
            stream_id: Stream UUID
            data: StreamUpdate schema
        
        Returns:
            True if updated, False if not found
        """
        updated = await self.repo.update(stream_id, data)
        
        if updated:
            logger.info(f"✅ Stream updated: {stream_id}")
        else:
            logger.warning(f"⚠️ Stream not found: {stream_id}")
        
        return updated
    
    async def delete_stream(self, stream_id: str) -> bool:
        """
        Delete stream
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            True if deleted, False if not found
        """
        deleted = await self.repo.delete(stream_id)
        
        if deleted:
            logger.info(f"🗑️ Stream deleted: {stream_id}")
        else:
            logger.warning(f"⚠️ Stream not found: {stream_id}")
        
        return deleted
    
    async def get_active_streams(self) -> List[StreamResponse]:
        """
        Get only active streams
        
        Returns:
            List of active StreamResponse objects
        """
        return await self.get_all_streams(status_filter="active")
    
    async def toggle_stream_status(self, stream_id: str) -> bool:
        """
        Toggle stream status (active ↔ inactive)
        
        Args:
            stream_id: Stream UUID
        
        Returns:
            True if toggled successfully
        """
        stream = await self.repo.find_by_id(stream_id)
        
        if not stream:
            return False
        
        new_status = "inactive" if stream['status'] == "active" else "active"
        
        update_data = StreamUpdate(status=new_status)
        return await self.repo.update(stream_id, update_data)
    
    async def count_streams(self) -> int:
        """
        Count total streams
        
        Returns:
            Total number of streams
        """
        return await self.repo.count()
