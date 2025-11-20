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
    # 1. Try fetching from Alembic config
    url = config.get_main_option("sqlalchemy.url")
    
    # 2. Fallback to env var (Check specific Railway vars too)
    if not url:
        url = os.environ.get("DATABASE_URL") or os.environ.get("MYSQL_URL")

    if not url:
        raise ValueError("❌ DATABASE_URL is not set in environment variables!")

    # 3. Ensure it is synchronous (pymysql)
    # Railway often gives 'mysql://', we need 'mysql+pymysql://'
    if url.startswith("mysql://"):
        url = url.replace("mysql://", "mysql+pymysql://")
    elif "mysql+aiomysql" in url:
        url = url.replace("mysql+aiomysql", "mysql+pymysql")
        
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
