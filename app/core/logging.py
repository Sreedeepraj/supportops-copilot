import logging
import sys

_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

def _to_level(name: str) -> int:
    return _LEVELS.get(name.upper(), logging.INFO)

def setup_logging(settings):
    root = logging.getLogger()
    root.setLevel(_to_level(settings.LOG_LEVEL))

    # Ensure there is at least one stdout handler
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)

    # Optional subtree overrides (hierarchy-based)
    if settings.LOG_APP_LEVEL:
        logging.getLogger("app").setLevel(_to_level(settings.LOG_APP_LEVEL))

    if settings.LOG_RAG_LEVEL:
        logging.getLogger("app.rag").setLevel(_to_level(settings.LOG_RAG_LEVEL))