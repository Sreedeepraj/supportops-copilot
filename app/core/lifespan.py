from contextlib import asynccontextmanager
from fastapi import FastAPI
from .settings import settings
from .logging import setup_logging
from dotenv import load_dotenv

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    load_dotenv()
    setup_logging(settings)
    app.state.settings = settings
    yield
    # Shutdown (we’ll close db clients here later)