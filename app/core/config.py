# # # app/core/config.py
# # import os
# # from typing import List, Optional
# # from pydantic_settings import BaseSettings
# # import secrets
# # from dotenv import load_dotenv

# # load_dotenv()

# # class Settings(BaseSettings):
# #     PROJECT_NAME: str = "SkySearch AI - Flight Metasearch"
# #     API_V1_STR: str = "/api/v1"

# #     # OpenAI Configuration - with explicit defaults
# #     OPENAI_API_KEY: str
# #     OPENAI_MODEL: str = "gpt-4o"  # Default model
# #     OPENAI_MAX_TOKENS: int = 1000
# #     OPENAI_TEMPERATURE: float = 0.3
    
# #     # AI Feature Flags
# #     ENABLE_AI_SEARCH: bool = True
# #     AI_FALLBACK_TO_TRADITIONAL: bool = True
# #     AI_RATE_LIMIT_PER_MINUTE: int = 20

# #     # Amadeus Credentials (if using)
# #     AMADEUS_API_KEY: str = ""  # Made optional
# #     AMADEUS_API_SECRET: str = ""  # Made optional
# #     AMADEUS_TOKEN_URL: str = "https://test.api.amadeus.com/v1/security/oauth2/token"
# #     AMADEUS_SEARCH_URL: str = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    
# #     # Duffel API (Optional - for future use)
# #     DUFFEL_API_KEY: str = ""  # Made optional with default
# #     DUFFEL_API_URL: str = "https://api.duffel.com"  # Made optional with default
    
# #     # Redis (with defaults for Railway)
# #     REDIS_URI: str = "redis://localhost:6379"

# #     # Celery (with defaults for Railway)
# #     CELERY_BROKER_URL: str = "redis://localhost:6379/1"
# #     CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

# #     # TravelPayouts API (with empty defaults)
# #     TRAVELPAYOUTS_API_TOKEN: str = ""
# #     TRAVELPAYOUTS_MARKER: str = ""

# #     # Database Configuration (with defaults for Railway)
# #     DB_USER: str = "root"  # Override with Railway env var
# #     DB_PASSWORD: str = "password"  # Override with Railway env var
# #     DB_HOST: str # Override with Railway env var
# #     DB_PORT: int = 3306
# #     DB_NAME: str = "skysearch"
    
# #     @property
# #     def DATABASE_URL(self) -> str:
# #         return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

# #     # JWT Security
# #     SECRET_KEY: str = secrets.token_urlsafe(32)
# #     ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
# #     ALGORITHM: str = "HS256"  # Added default

# #     # Google OAuth (Optional)
# #     GOOGLE_CLIENT_ID: str = ""
# #     GOOGLE_CLIENT_SECRET: str = ""
# #     GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/api/v1/auth/google/callback"

# #     # Payment Integrations (Optional for MVP)
# #     STRIPE_SECRET_KEY: str = ""
# #     STRIPE_PUBLISHABLE_KEY: str = ""
# #     STRIPE_WEBHOOK_SECRET: str = ""
    
# #     CLICK_MERCHANT_ID: str = ""
# #     CLICK_SERVICE_ID: str = ""
# #     CLICK_SECRET_KEY: str = ""
    
# #     PAYME_MERCHANT_ID: str = ""
# #     PAYME_SECRET_KEY: str = ""
    
# #     # Email & SMS (Optional for MVP)
# #     SENDGRID_API_KEY: str = ""
# #     FROM_EMAIL: str = "noreply@skysearch.ai"
    
# #     ESKIZ_EMAIL: str = ""
# #     ESKIZ_PASSWORD: str = ""

# #     # CORS Origins
# #     BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
# #     # Frontend URL
# #     FRONTEND_URL: str = "http://localhost:3000"
    
# #     # Cache TTL Settings
# #     CACHE_TTL_SEARCH: int = 900  # 15 minutes - Added default
# #     CACHE_TTL_OFFERS: int = 300  # 5 minutes - Added default

# #     class Config:
# #         case_sensitive = True
# #         env_file = ".env"
# #         env_file_encoding = "utf-8"

# # settings = Settings()


# import os
# from typing import List, Optional
# from pydantic_settings import BaseSettings, SettingsConfigDict
# import secrets
# from dotenv import load_dotenv

# load_dotenv()

# class Settings(BaseSettings):
#     PROJECT_NAME: str = "FlyUz Flight Aggregator"
#     API_V1_STR: str = "/api/v1"

#     # OpenAI - Make optional with defaults
#     OPENAI_API_KEY: str  # Empty string as default
#     OPENAI_MODEL: str = "gpt-4o" 
#     OPENAI_MAX_TOKENS: int = 1000
#     OPENAI_TEMPERATURE: float = 0.3
    
#     # AI Feature Flags
#     ENABLE_AI_SEARCH: bool = True
#     AI_FALLBACK_TO_TRADITIONAL: bool = True
#     AI_RATE_LIMIT_PER_MINUTE: int = 20

#     # Amadeus Credentials - Make optional
#     AMADEUS_API_KEY: str = ""
#     AMADEUS_API_SECRET: str = ""
#     AMADEUS_TOKEN_URL: str = "https://test.api.amadeus.com/v1/security/oauth2/token"
#     AMADEUS_SEARCH_URL: str = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    
#     # Duffel API
#     DUFFEL_API_KEY: str
#     DUFFEL_API_URL: str = "https://api.duffel.com"
    
#     # Redis - Make optional with default
#     REDIS_URI: str = "redis://localhost:6379/0"

#     # Celery - Make optional with defaults
#     CELERY_BROKER_URL: str = "redis://localhost:6379/1"
#     CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

#     # Travelpayouts - Make optional
#     TRAVELPAYOUTS_API_TOKEN: str = ""
#     TRAVELPAYOUTS_MARKER: str = ""

#     # Database - Support both formats (Railway and local)
#     # Option 1: Full DATABASE_URL (for Railway)
#     DATABASE_URL: Optional[str] = None
    
#     # Option 2: Individual components (for local dev)
#     DB_USER: Optional[str] = None
#     DB_PASSWORD: Optional[str] = None
#     DB_HOST: Optional[str] = None
#     DB_PORT: Optional[int] = 3306
#     DB_NAME: Optional[str] = None
    
#     @property
#     def get_database_url(self) -> str:
#         """Get database URL - supports both Railway and local"""
#         if self.DATABASE_URL:
#             # Railway provides this
#             return self.DATABASE_URL
        
#         # Build from components for local dev
#         if all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
#             return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        
#         raise ValueError("DATABASE_URL or DB_* components must be set")

#     # JWT Security
#     SECRET_KEY: str = secrets.token_urlsafe(32)
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
#     ALGORITHM: str = "HS256"

#     # Google OAuth - Optional
#     GOOGLE_CLIENT_ID: str = ""
#     GOOGLE_CLIENT_SECRET: str = ""
#     GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/api/v1/auth/google/callback"

#     # Payment Providers - All optional
#     STRIPE_SECRET_KEY: str = ""
#     STRIPE_PUBLISHABLE_KEY: str = ""
#     STRIPE_WEBHOOK_SECRET: str = ""
    
#     CLICK_MERCHANT_ID: str = ""
#     CLICK_SERVICE_ID: str = ""
#     CLICK_SECRET_KEY: str = ""
    
#     PAYME_MERCHANT_ID: str = ""
#     PAYME_SECRET_KEY: str = ""
    
#     # Email - Optional
#     SENDGRID_API_KEY: str = ""
#     FROM_EMAIL: str = "noreply@flyuz.uz"
    
#     # SMS - Optional
#     ESKIZ_EMAIL: str = ""
#     ESKIZ_PASSWORD: str = ""

#     # CORS
#     BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
#     # Frontend
#     FRONTEND_URL: str = "http://localhost:3000"
    
#     # Cache TTL
#     CACHE_TTL_SEARCH: int = 900  # 15 minutes
#     CACHE_TTL_OFFERS: int = 300  # 5 minutes

#     # Pydantic v2 config
#     model_config = SettingsConfigDict(
#         case_sensitive=True,
#         env_file=".env",
#         env_file_encoding="utf-8",
#         extra="ignore"  # Ignore extra fields from .env
#     )

# settings = Settings()



# # app/core/config.py
# """
# Application configuration with support for:
# - Gemini AI (primary LLM)
# - Conversation system settings
# - Existing flight search integrations (Duffel, Amadeus, TravelPayouts)
# - All your production settings maintained
# """

# import os
# from typing import List, Optional, Literal
# from pydantic_settings import BaseSettings, SettingsConfigDict
# import secrets
# from dotenv import load_dotenv

# load_dotenv()


# class Settings(BaseSettings):
#     # ============================================================
#     # APPLICATION INFO
#     # ============================================================
#     PROJECT_NAME: str = "FlyUz Flight Aggregator"
#     API_V1_STR: str = "/api/v1"
#     VERSION: str = "1.0.0"
#     ENVIRONMENT: Literal["development", "staging", "production"] = "development"
#     DEBUG: bool = False
    
#     # ============================================================
#     # GEMINI AI CONFIGURATION (NEW - Primary LLM)
#     # ============================================================
#     GEMINI_API_KEY: str  # Required for conversation AI
#     GEMINI_MODEL: str = "gemini-2.0-flash"  # Fast and cost-effective
#     GEMINI_TIMEOUT: float = 60.0
#     GEMINI_MAX_RETRIES: int = 4
#     GEMINI_BASE_DELAY: float = 0.5
    
#     # Gemini generation config (from your Duffel code)
#     GEMINI_TEMPERATURE: float = 0.2
#     GEMINI_TOP_K: int = 40
#     GEMINI_TOP_P: float = 0.8
#     GEMINI_MAX_TOKENS: int = 2048
    
#     # ============================================================
#     # OPENAI CONFIGURATION (Existing - Keep for compatibility)
#     # ============================================================
#     OPENAI_API_KEY: str = ""  # Optional - keep your existing setup
#     OPENAI_MODEL: str = "gpt-4o"
#     OPENAI_MAX_TOKENS: int = 1000
#     OPENAI_TEMPERATURE: float = 0.3
    
#     # ============================================================
#     # AI FEATURE FLAGS
#     # ============================================================
#     ENABLE_AI_SEARCH: bool = True
#     AI_FALLBACK_TO_TRADITIONAL: bool = True
#     AI_RATE_LIMIT_PER_MINUTE: int = 20
    
#     # Conversation AI settings (NEW)
#     ENABLE_CONVERSATIONAL_AI: bool = True
#     ENABLE_IP_GEOLOCATION: bool = True
#     ENABLE_SMART_SUGGESTIONS: bool = True
#     ENABLE_PERSONALIZATION: bool = True
    
#     # ============================================================
#     # CONVERSATION SYSTEM SETTINGS (NEW)
#     # ============================================================
#     # Session management
#     CONVERSATION_SESSION_TTL: int = 1800  # 30 minutes
#     MAX_CONVERSATION_TURNS: int = 20
#     MAX_SESSION_SIZE: int = 1000  # In-memory store limit
    
#     # Prompt configuration
#     PROMPT_VERSION: str = "v2_few_shot"  # v1_basic, v2_few_shot, v3_cot
    
#     # Conversation memory
#     MAX_CONVERSATION_HISTORY: int = 30  # Max messages to keep
#     CONVERSATION_CONTEXT_WINDOW: int = 6  # Recent messages for AI context
    
#     # ============================================================
#     # FLIGHT API INTEGRATIONS (Your existing config - unchanged)
#     # ============================================================
#     # Amadeus
#     AMADEUS_API_KEY: str = ""
#     AMADEUS_API_SECRET: str = ""
#     AMADEUS_TOKEN_URL: str = "https://test.api.amadeus.com/v1/security/oauth2/token"
#     AMADEUS_SEARCH_URL: str = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    
#     # Duffel
#     DUFFEL_API_KEY: str
#     DUFFEL_API_URL: str = "https://api.duffel.com"
    
#     # TravelPayouts
#     TRAVELPAYOUTS_API_TOKEN: str = ""
#     TRAVELPAYOUTS_MARKER: str = ""
    
#     # ============================================================
#     # REDIS CONFIGURATION (Your existing config - unchanged)
#     # ============================================================
#     REDIS_URI: str = "redis://localhost:6379/0"
#     REDIS_URL: Optional[str] = None  # Alias for conversation system
    
#     @property
#     def get_redis_url(self) -> str:
#         """Get Redis URL for conversation system"""
#         return self.REDIS_URL or self.REDIS_URI

#     # ============================================================
#     # CELERY CONFIGURATION (Your existing config - unchanged)
#     # ============================================================
#     CELERY_BROKER_URL: str = "redis://localhost:6379/1"
#     CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

#     # ============================================================
#     # DATABASE CONFIGURATION (Your existing config - unchanged)
#     # ============================================================
#     # Option 1: Full DATABASE_URL (for Railway)
#     DATABASE_URL: Optional[str] = None
    
#     # Option 2: Individual components (for local dev)
#     DB_USER: Optional[str] = None
#     DB_PASSWORD: Optional[str] = None
#     DB_HOST: Optional[str] = None
#     DB_PORT: Optional[int] = 3306
#     DB_NAME: Optional[str] = None
    
#     @property
#     def get_database_url(self) -> str:
#         """Get database URL - supports both Railway and local"""
#         if self.DATABASE_URL:
#             return self.DATABASE_URL
        
#         if all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
#             return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        
#         raise ValueError("DATABASE_URL or DB_* components must be set")

#     # ============================================================
#     # JWT SECURITY (Your existing config - unchanged)
#     # ============================================================
#     SECRET_KEY: str = secrets.token_urlsafe(32)
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
#     ALGORITHM: str = "HS256"

#     # ============================================================
#     # OAUTH PROVIDERS (Your existing config - unchanged)
#     # ============================================================
#     GOOGLE_CLIENT_ID: str = ""
#     GOOGLE_CLIENT_SECRET: str = ""
#     GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/api/v1/auth/google/callback"

#     # ============================================================
#     # PAYMENT PROVIDERS (Your existing config - unchanged)
#     # ============================================================
#     STRIPE_SECRET_KEY: str = ""
#     STRIPE_PUBLISHABLE_KEY: str = ""
#     STRIPE_WEBHOOK_SECRET: str = ""
    
#     CLICK_MERCHANT_ID: str = ""
#     CLICK_SERVICE_ID: str = ""
#     CLICK_SECRET_KEY: str = ""
    
#     PAYME_MERCHANT_ID: str = ""
#     PAYME_SECRET_KEY: str = ""
    
#     # ============================================================
#     # COMMUNICATION (Your existing config - unchanged)
#     # ============================================================
#     # Email
#     SENDGRID_API_KEY: str = ""
#     FROM_EMAIL: str = "noreply@flyuz.uz"
    
#     # SMS
#     ESKIZ_EMAIL: str = ""
#     ESKIZ_PASSWORD: str = ""

#     # ============================================================
#     # CORS & FRONTEND (Your existing config - unchanged)
#     # ============================================================
#     BACKEND_CORS_ORIGINS: List[str] = ["*"]
#     FRONTEND_URL: str = "http://localhost:3000"
    
#     # ============================================================
#     # CACHE SETTINGS (Your existing config - unchanged)
#     # ============================================================
#     CACHE_TTL_SEARCH: int = 900  # 15 minutes
#     CACHE_TTL_OFFERS: int = 300  # 5 minutes
#     CACHE_TTL_CONTEXT: int = 1800  # 30 minutes for conversation context
    
#     # ============================================================
#     # LOGGING CONFIGURATION (NEW)
#     # ============================================================
#     LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
#     @property
#     def is_production(self) -> bool:
#         """Check if running in production"""
#         return self.ENVIRONMENT == "production"
    
#     @property
#     def is_development(self) -> bool:
#         """Check if running in development"""
#         return self.ENVIRONMENT == "development"

#     # ============================================================
#     # RATE LIMITING (NEW)
#     # ============================================================
#     RATE_LIMIT_ENABLED: bool = True
#     RATE_LIMIT_PER_MINUTE: int = 30  # For conversation endpoints
    
#     # ============================================================
#     # TIMEZONE SETTINGS (NEW - from your Duffel code)
#     # ============================================================
#     DEFAULT_TIMEZONE: str = "Asia/Tashkent"
#     TIMEZONE_OFFSET_HOURS: int = 5

#     # ============================================================
#     # PYDANTIC V2 CONFIG (Your existing config - unchanged)
#     # ============================================================
#     model_config = SettingsConfigDict(
#         case_sensitive=True,
#         env_file=".env",
#         env_file_encoding="utf-8",
#         extra="ignore"
#     )


# # ============================================================
# # GLOBAL SETTINGS INSTANCE
# # ============================================================
# settings = Settings()


# # ============================================================
# # HELPER FUNCTIONS
# # ============================================================

# def get_settings() -> Settings:
#     """
#     Dependency function for FastAPI.
#     Use: settings = Depends(get_settings)
#     """
#     return settings


# def validate_required_settings():
#     """
#     Validate that required settings are present.
#     Call this at startup to fail fast if misconfigured.
#     """
#     required = []
    
#     # Check Gemini (primary AI)
#     if settings.ENABLE_CONVERSATIONAL_AI and not settings.GEMINI_API_KEY:
#         required.append("GEMINI_API_KEY")
    
#     # Check Duffel (primary flight API)
#     if not settings.DUFFEL_API_KEY:
#         required.append("DUFFEL_API_KEY")
    
#     if required:
#         missing = ", ".join(required)
#         raise ValueError(
#             f"Missing required environment variables: {missing}\n"
#             f"Please set them in your .env file"
#         )


# # ============================================================
# # STARTUP VALIDATION
# # ============================================================

# try:
#     validate_required_settings()
# except ValueError as e:
#     import logging
#     logging.warning(f"Configuration warning: {e}")



# app/core/config.py

"""
Unified configuration for Havva AI
Includes:
- Your full existing FlyUz configuration
- Claude-style Memory System configuration
- Qdrant + OpenAI embeddings
- Backward compatibility with Gemini + Duffel/Amadeus
"""

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
