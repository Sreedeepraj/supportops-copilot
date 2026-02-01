class RetryableError(Exception):
    """Transient error: safe to retry."""
    pass


class RateLimitError(RetryableError):
    pass


class UpstreamTimeoutError(RetryableError):
    pass


class UpstreamUnavailableError(RetryableError):
    pass