from contextlib import asynccontextmanager
from fastapi import FastAPI
from .settings import Settings



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.settings = Settings()
    yield
    # Shutdown (we’ll close db clients here later)