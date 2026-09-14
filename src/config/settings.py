import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App Information
    APP_NAME: str = "AmazonHelp Customer Support RAG"
    APP_VERSION: str = "0.1.0"
    
    # Path configuration
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")
    DB_PATH: str = os.path.join(ARTIFACTS_DIR, "memory.db")
    
    # Model Providers (Adapter Configuration)
    LLM_PROVIDER: str = Field(default="mock", description="Provider for LLM (mock, openai, gemini)")
    
    # Optional Provider Keys (Loaded from .env)
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    
    # Classifier settings
    RANDOM_SEED: int = 42
    
    # Retrieval settings
    DENSE_MODEL_NAME: str = "all-MiniLM-L6-v2"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.ARTIFACTS_DIR, exist_ok=True)
