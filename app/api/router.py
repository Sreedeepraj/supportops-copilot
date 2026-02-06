from fastapi import APIRouter
from .health import router as health_router
from .qa import router as qa_router
from app.api.qa_agent import router as qa_agent_router
from app.api.qa_multi_agent import router as qa_multi_agent_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["system"])
api_router.include_router(qa_router, tags=["rag"])
api_router.include_router(qa_agent_router, tags=["agent"])
api_router.include_router(qa_multi_agent_router, tags=["multi_agent"])