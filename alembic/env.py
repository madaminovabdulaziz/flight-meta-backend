from logging.config import fileConfig
import os
from sqlalchemy import create_engine, pool
from alembic import context

from app.models.models import Base
from app.core.config import settings

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData for 'autogenerate' support
target_metadata = Base.metadata


def get_url():
    """
    Get the DB URL.
    PRIORITY 1: Environment Variable (Railway)
    PRIORITY 2: App Settings
    PRIORITY 3: Alembic Config (.ini)
    """
    # 1. Check Environment Variable FIRST (Critical for Railway)
    url = os.environ.get("DATABASE_URL")
    
    # 2. Fallback to Settings
    if not url:
        try:
            url = settings.DATABASE_URL
        except:
            pass

    # 3. Fallback to Alembic Config (only if nothing else is found)
    if not url:
        url = config.get_main_option("sqlalchemy.url")

    # 4. CRITICAL FIX: Force Sync Driver (pymysql) for Migrations
    # Railway provides 'mysql+aiomysql', but Alembic needs 'mysql+pymysql'
    if url:
        if "mysql+aiomysql" in url:
            url = url.replace("mysql+aiomysql", "mysql+pymysql")
        elif url.startswith("mysql://"):
            url = url.replace("mysql://", "mysql+pymysql://")
            
    return url

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (Synchronous)."""
    
    # 1. Get the synchronous URL
    url = get_url()
    
    # 2. Create a standard synchronous engine
    connectable = create_engine(url, poolclass=pool.NullPool)

    # 3. Run migrations synchronously
    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
