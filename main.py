from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.api.router import api_router
from app.core.settings import settings
from app.core.middleware import RequestIDMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.include_router(api_router, prefix="/v1")