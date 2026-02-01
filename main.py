from fastapi import FastAPI
from app.core.lifespan import lifespan
from app.api.router import api_router
from app.core.settings import settings
from app.core.middleware import RequestIDMiddleware
from app.core.exception_handlers import app_error_handler, unhandled_exception_handler
from app.core.errors import AppError

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)

app.add_exception_handler(AppError, app_error_handler)
app.add_exception_handler(Exception, unhandled_exception_handler)

app.include_router(api_router, prefix="/v1")