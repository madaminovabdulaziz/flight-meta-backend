from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from app.core.config import settings
from app.db.database import engine, Base
from app.api.v1.endpoints import flights, auth, clicks, airports, hot_routes, ai_search, popular_destinations, nearest_airport

async def create_db_and_tables():
    async with engine.begin() as conn:
        # This will create tables based on your models
        await conn.run_sync(Base.metadata.create_all)

# app/main.py - Update FastAPI initialization

app = FastAPI(
    title="SkySearch AI - Flight Metasearch Engine",
    description="""
    🚀 **Intelligent Flight Search API**
    
    SkySearch AI is a next-generation flight metasearch platform that combines 
    traditional search with AI-powered features.
    
    ## Features
    
    * 🔍 **Smart Search**: Traditional and AI-powered flight search
    * 🔥 **Hot Routes**: Discover popular destinations with best prices
    * ✈️ **Airport Data**: Comprehensive airport information
    * 📊 **Analytics**: Click tracking and conversion monitoring
    
    ## Authentication
    
    Most endpoints are public. User-specific features require Bearer token authentication.
    
    ## Rate Limits
    
    * Public endpoints: 100 requests/minute per IP
    * Authenticated: 1000 requests/minute per user
    
    ## Support
    
    Contact: support@skysearch.ai
    """,
    version="1.0.0",
    contact={
        "name": "SkySearch AI Team",
        "email": "support@skysearch.ai",
    },
    license_info={
        "name": "Proprietary",
    },
    # Custom OpenAPI URL (useful for versioning)
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json"
)


# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Include your API routers here
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(nearest_airport.router, prefix=f"{settings.API_V1_STR}/nearest-airport", tags=["nearest airport"])
#app.include_router(flights.router, prefix=f"{settings.API_V1_STR}/flights", tags=["flights"])
#app.include_router(ai_search.router, prefix=f"{settings.API_V1_STR}/ai-search", tags=["ai-search"])
#app.include_router(clicks.router, prefix=f"{settings.API_V1_STR}/clicks", tags=["clicks"])
app.include_router(hot_routes.router, prefix=f"{settings.API_V1_STR}/hot-routes", tags=["hot-routes"])
app.include_router(popular_destinations.router, prefix=f"{settings.API_V1_STR}/popular-routes", tags=["popular-routes"])
#app.include_router(airports.router, prefix=f"{settings.API_V1_STR}/airports", tags=["airports"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the FlyUz Flight Aggregator API"}

