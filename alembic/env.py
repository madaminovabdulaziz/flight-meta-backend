from logging.config import fileConfig
import os
import sys
from sqlalchemy import create_engine, pool
from alembic import context

from app.models.models import Base
from app.core.config import settings

# Alembic Config object
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def get_url():
    """
    Robust URL fetcher with Debugging and Sanitization.
    """
    # 1. Try Environment Variable (Railway)
    url = os.environ.get("DATABASE_URL")
    source = "Environment Variable"

    # 2. Fallback to App Settings (Pydantic)
    if not url:
        try:
            # Convert Pydantic URL object to string if needed
            url = str(settings.DATABASE_URL)
            source = "App Settings"
        except Exception:
            pass

    # 3. Fallback to Alembic Config
    if not url:
        url = config.get_main_option("sqlalchemy.url")
        source = "Alembic Config"

    # ---------------------------------------------------------
    # DEBUGGING BLOCK: Print what we found (Mask Password)
    # ---------------------------------------------------------
    if not url:
        print("❌ [Alembic] CRITICAL: No DATABASE_URL found in Env, Settings, or Config!")
        print("   -> Please verify 'DATABASE_URL' in Railway Variables.")
        return "" # Will cause crash, but we logged why

    # Mask password for safe logging
    safe_url = url
    if "@" in safe_url:
        try:
            prefix, suffix = safe_url.split("@", 1)
            # Handle driver://user:pass part
            if ":" in prefix and "//" in prefix:
                driver_part, auth = prefix.split("://", 1)
                if ":" in auth:
                    user, _ = auth.split(":", 1)
                    safe_url = f"{driver_part}://{user}:****@{suffix}"
        except:
            pass # Skip masking if format is weird

    print(f"🔍 [Alembic] Found URL from {source}: {safe_url}")

    # 4. SANITIZATION: Remove whitespace (Common cause of Parse Error)
    url = url.strip()
    
    # Remove quotes if they were accidentally included in Railway value
    url = url.strip('"').strip("'")

    # 5. DRIVER SWAP: Force Sync Driver (pymysql)
    if "mysql+aiomysql" in url:
        url = url.replace("mysql+aiomysql", "mysql+pymysql")
    elif url.startswith("mysql://"):
        url = url.replace("mysql://", "mysql+pymysql://")
        
    return url


def run_migrations_offline() -> None:
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
    url = get_url()
    
    # Fail fast if URL is bad
    if not url:
        print("❌ [Alembic] Aborting: URL is empty.")
        sys.exit(1)

    try:
        connectable = create_engine(url, poolclass=pool.NullPool)

        with connectable.connect() as connection:
            context.configure(
                connection=connection, 
                target_metadata=target_metadata
            )

            with context.begin_transaction():
                context.run_migrations()
                
    except Exception as e:
        print(f"❌ [Alembic] Connection failed. Error: {e}")
        # Re-raise to fail the deployment
        raise 


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
