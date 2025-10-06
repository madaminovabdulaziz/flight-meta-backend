from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from app.core.config import settings
from app.db.database import engine, Base
from app.api.v1.endpoints import flights, auth, clicks, airports, hot_routes

async def create_db_and_tables():
    async with engine.begin() as conn:
        # This will create tables based on your models
        await conn.run_sync(Base.metadata.create_all)

app = FastAPI(title=settings.PROJECT_NAME)


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
app.include_router(flights.router, prefix=f"{settings.API_V1_STR}/flights", tags=["flights"])
app.include_router(clicks.router, prefix=f"{settings.API_V1_STR}/clicks", tags=["clicks"])
app.include_router(hot_routes.router, prefix=f"{settings.API_V1_STR}/hot-routes", tags=["hot-routes"])
app.include_router(airports.router, prefix=f"{settings.API_V1_STR}/airports", tags=["airports"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the FlyUz Flight Aggregator API"}

