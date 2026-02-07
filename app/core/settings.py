from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "SupportOps Copilot"
    APP_ENV: str = "dev"  # dev | stage | prod

    # Auth
    JWT_SECRET: str = "change-me"  # override in .env
    JWT_ALG: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Logging
    LOG_LEVEL: str = "INFO"          # root default
    LOG_APP_LEVEL: Optional[str] = None
    LOG_RAG_LEVEL: Optional[str] = None
    # Timeouts / retries (we’ll use these later for OpenAI + tools)
    HTTP_TIMEOUT_SECONDS: float = 15.0
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BASE_DELAY_SECONDS: float = 0.5

    OPENAI_API_KEY: str
    MEMORY_ENABLED: bool = True
    MEMORY_DB_PATH: str = "data/memory.sqlite3"
    MEMORY_COLLECTION: str = "memories"
    MEMORY_MAX_MESSAGES: int = 8   # short-term window
    MEMORY_TOP_K: int = 4          # long-term recall   

settings = Settings()