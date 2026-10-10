import logging
from app.db.asyncpg_client import get_db

logger = logging.getLogger(__name__)

async def save_to_db(payload, counts, stream_source="public"):
    try:
        # Menggunakan generator get_db()
        pool_gen = get_db()
        pool = await pool_gen.__anext__()
        
        query_history = """
        INSERT INTO traffic_history (
            stream_id, stream_source,
            person_count, motorcycle_count, car_count, bus_count, truck_count, 
            total_vehicle_count, person_vehicle_ratio, density_status,
            average_speed, road_occupancy, congestion_index
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13) RETURNING id
        """
        total_v = counts["motorcycle"] + counts["car"] + counts["bus"] + counts["truck"]
        hist_id = await pool.fetchval(query_history,
            payload["stream_id"], stream_source,
            counts["person"], counts["motorcycle"], counts["car"], counts["bus"], counts["truck"],
            total_v, payload["person_vehicle_ratio"], payload["density_status"],
            payload.get("average_speed", 0.0), payload.get("road_occupancy", 0.0), payload.get("congestion_index", 0.0)
        )
        if payload["density_status"] in ["High Density", "Anomaly"]:
            query_alert = """
            INSERT INTO alerts (traffic_history_id, stream_id, alert_type, alert_message)
            VALUES ($1, $2, $3, $4)
            """
            await pool.execute(query_alert, hist_id, payload["stream_id"], payload["density_status"], payload["alert"]["message"])
    except Exception as e:
        logger.error(f"❌ DATABASE ERROR: {e}")
