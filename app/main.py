

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio

from app.core.config import settings
from app.db.database import engine, Base
from app.api.v1.endpoints import (
    auth,
    nearest_airport,
    email_collection,
    duffel_new,
    airports,
    duffel_ai,
    duffel_flexible,
    chat,
)

# --- QDRANT INITIALIZATION ---
from app.integrations.qdrant_client import qdrant_service


# ============================================================
# DATABASE INITIALIZATION
# ============================================================
async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


# ============================================================
# FASTAPI APP SETUP
# ============================================================
app = FastAPI(
    title="SkySearch AI - Flight Metasearch Engine",
    description="""
    🚀 **Intelligent Flight Search API**

    SkySearch AI is a next-generation flight metasearch platform
    that combines traditional search with AI-powered features.

    ## Features
    * 🔍 Smart & AI-powered flight search
    * 🔥 Hot Routes discovery
    * ✈️ Airport information
    * 📊 Analytics & monitoring
    """,
    version="1.0.0",
    contact={"name": "SkySearch AI Team", "email": "support@skysearch.ai"},
    license_info={"name": "Proprietary"},
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/v1/openapi.json",
)

# ============================================================
# CORS CONFIG
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# STARTUP EVENT (CRITICAL FOR QDRANT)
# ============================================================
async def startup_event():
    await qdrant_service.connect()

    print("🚀 Application startup complete — Qdrant and DB initialized.")


# ============================================================
# API ROUTERS
# ============================================================
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(nearest_airport.router, prefix=f"{settings.API_V1_STR}/nearest-airport", tags=["nearest airport"])
app.include_router(airports.router, prefix=f"{settings.API_V1_STR}/airports", tags=["airports"])
app.include_router(duffel_new.router, prefix=f"{settings.API_V1_STR}/duffel-search", tags=["Duffel Flights"])
#app.include_router(duffel_ai.router)
app.include_router(duffel_flexible.router)
app.include_router(chat.router)
app.include_router(email_collection.router, prefix=f"{settings.API_V1_STR}/email-post", tags=["post email"])


# ============================================================
# ROOT ENDPOINT
# ============================================================
@app.get("/")
def read_root():
    return {"message": "Welcome to the FlyUz Flight Aggregator API"}
