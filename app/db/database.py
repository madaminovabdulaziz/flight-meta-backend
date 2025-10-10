# app/db/database.py - FIXED
import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Use the property method to get DATABASE_URL
    engine = create_async_engine(
        settings.get_database_url,  # Changed from settings.DATABASE_URL
        pool_pre_ping=True,
        echo=False  # Set to True for SQL debugging
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
            logger.debug("New async database session created.")
            yield session
        finally:
            await session.close()
            logger.debug("Async database session closed.")

def get_engine():
    """Helper to expose engine (useful for migrations)."""
    return engine