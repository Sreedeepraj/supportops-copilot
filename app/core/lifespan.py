from contextlib import asynccontextmanager
from fastapi import FastAPI
from .settings import settings
from .logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging(settings)
    app.state.settings = settings
    yield
    # Shutdown (we’ll close db clients here later)