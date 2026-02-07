from contextlib import asynccontextmanager
from fastapi import FastAPI
from .settings import settings
from .logging import setup_logging
from dotenv import load_dotenv
from app.memory.service import ensure_memory_ready

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_dotenv()
    setup_logging(settings)
    app.state.settings = settings
    ensure_memory_ready()
    yield
    # Shutdown (we’ll close db clients here later)