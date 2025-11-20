# app/db/database.py - FIXED & VERIFIED
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # 1. Get the URL from settings (Railway provides 'mysql://...')
    database_url = str(settings.get_database_url)

    # 2. 🛠️ CRITICAL FIX: Force Async Driver
    # We must swap 'mysql://' -> 'mysql+aiomysql://'
    if database_url and database_url.startswith("mysql://"):
        database_url = database_url.replace("mysql://", "mysql+aiomysql://")
    
    # 3. Create the engine with the CORRECTED url
    engine = create_async_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )
    logger.info("✅ Async database engine created successfully.")

except Exception as e:
    logger.error(f"❌ Failed to create async engine: {e}")
    raise

# Use expire_on_commit=False to keep objects usable after commit
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

Base = declarative_base()

async def get_db() -> AsyncSession:
    """
    Dependency that provides an async database session.
    Ensures session is closed after request is handled.
    """
    async with AsyncSessionLocal() as session:
        try:
            # logger.debug("New async database session created.")
            yield session
        finally:
            await session.close()
            # logger.debug("Async database session closed.")

def get_engine():
    """Helper to expose engine (useful for migrations)."""
    return engine
