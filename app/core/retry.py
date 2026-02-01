import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.core.settings import settings
from app.core.retryable import RetryableError

logger = logging.getLogger(__name__)


def retry_on_transient_failure():
    """
    Retry ONLY transient failures (RetryableError).
    Business errors (AppError) and programmer errors should not be retried.
    """
    return retry(
        retry=retry_if_exception_type(RetryableError),
        stop=stop_after_attempt(settings.RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=settings.RETRY_BASE_DELAY_SECONDS,
            min=settings.RETRY_BASE_DELAY_SECONDS,
            max=8,
        ),
        reraise=True,
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )