from fastapi import APIRouter
from app.api.v1.endpoints import streams, history, alerts, users, private_streams

api_router = APIRouter()

api_router.include_router(streams.router)
api_router.include_router(private_streams.router)
api_router.include_router(history.router)
api_router.include_router(alerts.router)
api_router.include_router(users.router)
