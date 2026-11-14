from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All settings in one typed place, read from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    database_url: str = "postgresql+asyncpg://skytriage:skytriage@localhost:5432/skytriage"
    redis_url: str = "redis://localhost:6379/0"
    llm_api_key: str = ""
    llm_model: str = ""
    cascade_confidence_threshold: float = 0.85
    models_dir: str = "models"


settings = Settings()