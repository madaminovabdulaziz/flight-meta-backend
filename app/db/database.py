import logging
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import settings

logging.basicConfig(level=logging.INFO)

try:
    engine = create_async_engine(settings.DATABASE_URL, pool_pre_ping=True)
    logging.info("Async engine created successfully.")
except Exception as e:
    logging.error(f"Failed to create async engine: {e}")
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
            logging.debug("New async database session created.")
            yield session
        finally:
            await session.close()
            logging.debug("Async database session closed.")

def get_engine():
    """Helper to expose engine (useful for migrations)."""
    return engine
