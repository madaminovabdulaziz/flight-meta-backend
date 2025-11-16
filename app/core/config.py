import os
from typing import List, Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from dotenv import load_dotenv
import secrets

load_dotenv()


class Settings(BaseSettings):
    # ============================================================
    # APPLICATION INFO
    # ============================================================
    PROJECT_NAME: str = "FlyUz Flight Aggregator"
    APP_NAME: str = "Havva AI"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "1.0.0"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = False

    # ============================================================
    # GEMINI AI CONFIGURATION (PRIMARY LLM)
    # ============================================================
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.0-flash"
    GEMINI_TIMEOUT: float = 60.0
    GEMINI_MAX_RETRIES: int = 4
    GEMINI_BASE_DELAY: float = 0.5
    GEMINI_TEMPERATURE: float = 0.2
    GEMINI_TOP_K: int = 40
    GEMINI_TOP_P: float = 0.8
    GEMINI_MAX_TOKENS: int = 2048

    # ============================================================
    # OPENAI CONFIGURATION (FOR MEMORY + EMBEDDINGS)
    # ============================================================
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_CHAT_MODEL: str = "gpt-4-turbo-preview"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-ada-002"
    OPENAI_MAX_TOKENS: int = 2000
    OPENAI_TEMPERATURE: float = 0.3

    # ============================================================
    # QDRANT VECTOR DATABASE (MEMORY SYSTEM)
    # ============================================================
    QDRANT_URL: str = Field(default="http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "user_memories"
    QDRANT_VECTOR_SIZE: int = 768  # ada-002 embeddings

    # Memory System thresholds
    MEMORY_RETRIEVAL_LIMIT: int = 10
    MEMORY_SIMILARITY_THRESHOLD: float = 0.5
    MEMORY_IMPORTANCE_HIGH_THRESHOLD: int = 3

    # ============================================================
    # CONVERSATION SYSTEM SETTINGS
    # ============================================================
    ENABLE_CONVERSATIONAL_AI: bool = True
    ENABLE_PERSONALIZATION: bool = True
    CONVERSATION_SESSION_TTL: int = 1800
    MAX_CONVERSATION_TURNS: int = 20
    MAX_CONVERSATION_HISTORY: int = 30
    CONVERSATION_CONTEXT_WINDOW: int = 6

    # ============================================================
    # EXISTING FLIGHT API INTEGRATIONS
    # ============================================================
    AMADEUS_API_KEY: str = ""
    AMADEUS_API_SECRET: str = ""
    AMADEUS_TOKEN_URL: str = "https://test.api.amadeus.com/v1/security/oauth2/token"
    AMADEUS_SEARCH_URL: str = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    DUFFEL_API_KEY: str
    DUFFEL_API_URL: str = "https://api.duffel.com"
    TRAVELPAYOUTS_API_TOKEN: str = ""
    TRAVELPAYOUTS_MARKER: str = ""

    # ============================================================
    # REDIS (SESSION, CACHING, MEMORY TEMP STORE)
    # ============================================================
    REDIS_URI: str = "redis://localhost:6379/0"
    REDIS_URL: Optional[str] = None  # alias
    REDIS_SESSION_TTL: int = 86400
    REDIS_FLIGHT_CACHE_TTL: int = 900

    @property
    def get_redis_url(self) -> str:
        return self.REDIS_URL or self.REDIS_URI

    # ============================================================
    # CELERY (unchanged)
    # ============================================================
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

    # ============================================================
    # DATABASE (MySQL)
    # ============================================================
    DATABASE_URL: Optional[str] = None
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = 3306
    DB_NAME: Optional[str] = None

    @property
    def get_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL

        if all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
            return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

        raise ValueError("DATABASE_URL or DB_* variables must be set")

    # ============================================================
    # SECURITY (JWT)
    # ============================================================
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 10080  # 7 days
    ALGORITHM: str = "HS256"

    # ============================================================
    # COMMUNICATION
    # ============================================================
    SENDGRID_API_KEY: str = ""
    FROM_EMAIL: str = "noreply@flyuz.uz"

    ESKIZ_EMAIL: str = ""
    ESKIZ_PASSWORD: str = ""

    # ============================================================
    # FRONTEND / CORS
    # ============================================================
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    FRONTEND_URL: str = "http://localhost:3000"

    # ============================================================
    # CACHE
    # ============================================================
    CACHE_TTL_SEARCH: int = 900
    CACHE_TTL_OFFERS: int = 300
    CACHE_TTL_CONTEXT: int = 1800

    # ============================================================
    # LOGGING
    # ============================================================
    LOG_LEVEL: str = "INFO"

    # ============================================================
    # TIMEZONE
    # ============================================================
    DEFAULT_TIMEZONE: str = "Asia/Tashkent"
    TIMEZONE_OFFSET_HOURS: int = 5

    # ============================================================
    # PYDANTIC CONFIG
    # ============================================================
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# ============================================================
# GLOBAL INSTANCE
# ============================================================
settings = Settings()


# ============================================================
# VALIDATION
# ============================================================
def validate_required_settings():
    missing = []

    if settings.ENABLE_CONVERSATIONAL_AI and not settings.GEMINI_API_KEY:
        missing.append("GEMINI_API_KEY")
    if not settings.DUFFEL_API_KEY:
        missing.append("DUFFEL_API_KEY")

    if missing:
        raise ValueError(f"Missing required env vars: {', '.join(missing)}")


try:
    validate_required_settings()
except ValueError as e:
    import logging
    logging.warning(f"Config warning: {e}")
