from fastapi import APIRouter
from .health import router as health_router
from .qa import router as qa_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(qa_router, tags=["rag"])