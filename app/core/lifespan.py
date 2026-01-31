from contextlib import asynccontextmanager
from fastapi import FastAPI
from .settings import Settings
from .logging import setup_logging




@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    app.state.settings = Settings()
    yield
    # Shutdown (we’ll close db clients here later)