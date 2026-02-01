from fastapi import APIRouter
import logging

from app.core.errors import AppError
from app.core.retry import retry_on_transient_failure
from app.core.retryable import RetryableError

logger = logging.getLogger(__name__)

router = APIRouter()

_flaky_counter = {"count": 0}


@router.get("/health")
async def health_check():
    return {"status": "ok"}

@router.get("/error")
async def trigger_app_error():
    raise AppError("This is a controlled AppError", status_code=400)


@router.get("/crash")
async def trigger_unhandled_error():
    # Unhandled programmer error → should become 500 via catch-all
    return 1 / 0


@retry_on_transient_failure()
async def flaky_operation():
    _flaky_counter["count"] += 1
    logger.info(f"flaky_operation attempt={_flaky_counter['count']}")
    if _flaky_counter["count"] < 3:
        raise RetryableError("temporary upstream failure")
    return "success"


@router.get("/retry-test")
async def retry_test():
    # This will fail twice, then succeed on the 3rd attempt
    result = await flaky_operation()
    return {"result": result, "attempts": _flaky_counter["count"]}