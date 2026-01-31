from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.api.router import api_router
from app.core.settings import settings

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/v1")