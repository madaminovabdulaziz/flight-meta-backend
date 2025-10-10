# app/core/config.py - FIXED for Pydantic v2
import os
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
import secrets
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "FlyUz Flight Aggregator"
    API_V1_STR: str = "/api/v1"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_MAX_TOKENS: int = 1000
    OPENAI_TEMPERATURE: float = 0.3
    
    # AI Feature Flags
    ENABLE_AI_SEARCH: bool = True
    AI_FALLBACK_TO_TRADITIONAL: bool = True
    AI_RATE_LIMIT_PER_MINUTE: int = 20

    # Amadeus Credentials
    AMADEUS_API_KEY: str
    AMADEUS_API_SECRET: str
    AMADEUS_TOKEN_URL: str = "https://test.api.amadeus.com/v1/security/oauth2/token"
    AMADEUS_SEARCH_URL: str = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    
    # Duffel API
    DUFFEL_API_KEY: str = ""
    DUFFEL_API_URL: str = "https://api.duffel.com"
    
    # Redis
    REDIS_URI: str

    # Celery
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # Travelpayouts
    TRAVELPAYOUTS_API_TOKEN: str
    TRAVELPAYOUTS_MARKER: str

    # Database - Support both formats (Railway and local)
    # Option 1: Full DATABASE_URL (for Railway)
    DATABASE_URL: Optional[str] = None
    
    # Option 2: Individual components (for local dev)
    DB_USER: Optional[str] = None
    DB_PASSWORD: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = 3306
    DB_NAME: Optional[str] = None
    
    @property
    def get_database_url(self) -> str:
        """Get database URL - supports both Railway and local"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        
        # Build from components
        if all([self.DB_USER, self.DB_PASSWORD, self.DB_HOST, self.DB_NAME]):
            return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        
        raise ValueError("DATABASE_URL or DB_* components must be set")

    # JWT Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    ALGORITHM: str = "HS256"

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/api/v1/auth/google/callback"

    # Payment Providers
    STRIPE_SECRET_KEY: str = ""
    STRIPE_PUBLISHABLE_KEY: str = ""
    STRIPE_WEBHOOK_SECRET: str = ""
    
    CLICK_MERCHANT_ID: str = ""
    CLICK_SERVICE_ID: str = ""
    CLICK_SECRET_KEY: str = ""
    
    PAYME_MERCHANT_ID: str = ""
    PAYME_SECRET_KEY: str = ""
    
    # Email
    SENDGRID_API_KEY: str = ""
    FROM_EMAIL: str = "noreply@flyuz.uz"
    
    # SMS
    ESKIZ_EMAIL: str = ""
    ESKIZ_PASSWORD: str = ""

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Frontend
    FRONTEND_URL: str = "http://localhost:3000"
    
    # Cache TTL
    CACHE_TTL_SEARCH: int = 900  # 15 minutes
    CACHE_TTL_OFFERS: int = 300  # 5 minutes

    # Pydantic v2 config - REPLACES the old Config class
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields from .env
    )

settings = Settings()


# # app/core/config.py
# import os
# from typing import List
# from pydantic import Field
# from pydantic_settings import BaseSettings, SettingsConfigDict
# import secrets
# from dotenv import load_dotenv

# load_dotenv()

# class Settings(BaseSettings):
#     PROJECT_NAME: str = "FlyUz Flight Aggregator"
#     API_V1_STR: str = "/api/v1"

#     OPENAI_API_KEY: str
#     OPENAI_MODEL: str = "gpt-4o"  # Best model for function calling
#     OPENAI_MAX_TOKENS: int = 1000
#     OPENAI_TEMPERATURE: float = 0.3  # Lower = more consistent extraction
    
#     # AI Feature Flags
#     ENABLE_AI_SEARCH: bool = True
#     AI_FALLBACK_TO_TRADITIONAL: bool = True
#     AI_RATE_LIMIT_PER_MINUTE: int = 20


#     # --- Amadeus Credentials (Existing) ---
#     AMADEUS_API_KEY: str
#     AMADEUS_API_SECRET: str
#     AMADEUS_TOKEN_URL: str = "https://test.api.amadeus.com/v1/security/oauth2/token"
#     AMADEUS_SEARCH_URL: str = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    
#     # --- NEW: Duffel API (Alternative Provider) ---
#     DUFFEL_API_KEY: str = ""
#     DUFFEL_API_URL: str = "https://api.duffel.com"
    
#     # --- Redis ---
#     REDIS_URI: str
#     REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")

#     CELERY_BROKER_URL: str
#     CELERY_RESULT_BACKEND: str

#     # --- NEW: Travelpayouts free token ---

#     TRAVELPAYOUTS_API_TOKEN: str
#     TRAVELPAYOUTS_MARKER: str

#     # --- Database (Existing - MySQL) ---
#     DB_USER: str 
#     DB_PASSWORD: str 
#     DB_HOST: str
#     DB_PORT: int
#     DB_NAME: str
    
#     @property
#     def DATABASE_URL(self) -> str:
#         return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

#     # --- JWT Security (Existing) ---
#     SECRET_KEY: str = secrets.token_urlsafe(32)
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
#     ALGORITHM: str

#     # --- Google OAuth (Existing) ---
#     GOOGLE_CLIENT_ID: str = ""
#     GOOGLE_CLIENT_SECRET: str = ""
#     GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/api/v1/auth/google/callback"

#     # --- NEW: Stripe Payment (International) ---
#     STRIPE_SECRET_KEY: str = ""
#     STRIPE_PUBLISHABLE_KEY: str = ""
#     STRIPE_WEBHOOK_SECRET: str = ""
    
#     # --- NEW: Click Payment (Uzbekistan) ---
#     CLICK_MERCHANT_ID: str = ""
#     CLICK_SERVICE_ID: str = ""
#     CLICK_SECRET_KEY: str = ""
    
#     # --- NEW: Payme Payment (Uzbekistan) ---
#     PAYME_MERCHANT_ID: str = ""
#     PAYME_SECRET_KEY: str = ""
    
#     # --- NEW: SendGrid Email ---
#     SENDGRID_API_KEY: str = ""
#     FROM_EMAIL: str = "noreply@flyuz.uz"
    
#     # --- NEW: Eskiz SMS (Uzbekistan) ---
#     ESKIZ_EMAIL: str = ""
#     ESKIZ_PASSWORD: str = ""
    
#     # --- NEW: Celery (Background Tasks) ---
#     CELERY_BROKER_URL: str = "redis://localhost:6379/1"
#     CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"

#     # --- CORS Origins ---
#     BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
#     # --- NEW: Frontend URL ---
#     FRONTEND_URL: str = "http://localhost:3000"
    
#     # --- NEW: Cache TTL Settings ---
#     CACHE_TTL_SEARCH: int  # 15 minutes
#     CACHE_TTL_OFFERS: int # 5 minutes

#     class Config:
#         case_sensitive = True
#         env_file = ".env"
#         env_file_encoding = "utf-8"

#     model_config = SettingsConfigDict(
#         env_file=".env",
#         case_sensitive=True
#     )

# settings = Settings()






#### --- older version --- # recover if neccessary 
# import os
# from typing import List
# from pydantic_settings import BaseSettings
# import secrets
# from dotenv import load_dotenv

# # This line loads the .env file
# load_dotenv()

# class Settings(BaseSettings):
#     PROJECT_NAME: str = "FlyUz Flight Aggregator"
#     API_V1_STR: str = "/api/v1"

#     # --- Amadeus Credentials (Loaded from .env) ---
#     AMADEUS_API_KEY: str
#     AMADEUS_API_SECRET: str
    
#     AMADEUS_TOKEN_URL: str = "https://test.api.amadeus.com/v1/security/oauth2/token"
#     AMADEUS_SEARCH_URL: str = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    
#     # --- Redis ---
#     REDIS_URI: str = "redis://localhost"

#     # --- Database ---
#     DB_USER: str = "root"
#     DB_PASSWORD: str = "root"
#     DB_HOST: str = "localhost"
#     DB_PORT: int = 3306
#     DB_NAME: str = "flyuz"
    
#     # This is a property that builds the URL from the parts above
#     @property
#     def DATABASE_URL(self) -> str:
#         return f"mysql+aiomysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

#     # --- JWT Security ---
#     SECRET_KEY: str = secrets.token_urlsafe(32)
#     ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7 # 7 days

#     ALGORITHM: str = "HS256"

#     # --- Google OAuth (Loaded from .env) ---
#     GOOGLE_CLIENT_ID: str
#     GOOGLE_CLIENT_SECRET: str
#     GOOGLE_REDIRECT_URI: str = "http://127.0.0.1:8000/api/v1/auth/google/callback"

#     # --- CORS Origins ---
#     BACKEND_CORS_ORIGINS: List[str] = ["*"]

#     class Config:
#         case_sensitive = True
#         # Tell pydantic to look for a .env file
#         env_file = ".env"
#         env_file_encoding = "utf-8"

# settings = Settings()
