import logging
from fastapi import Request
from fastapi.responses import JSONResponse
from app.core.errors import AppError

logger = logging.getLogger(__name__)


async def app_error_handler(request: Request, exc: AppError):
    # Business error → safe message
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "request_id": getattr(request.state, "request_id", None),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception):
    # Unexpected error → log full details, return safe message
    request_id = getattr(request.state, "request_id", None)
    logger.exception(f"Unhandled exception | request_id={request_id}")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
        },
    )