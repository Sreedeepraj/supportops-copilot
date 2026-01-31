from pydantic_settings import BaseSettings, SettingsConfigDict


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
    LOG_APP_LEVEL: str | None = None # override for 'app'

    # Timeouts / retries (we’ll use these later for OpenAI + tools)
    HTTP_TIMEOUT_SECONDS: float = 15.0
    RETRY_MAX_ATTEMPTS: int = 3
    RETRY_BASE_DELAY_SECONDS: float = 0.5

settings = Settings()