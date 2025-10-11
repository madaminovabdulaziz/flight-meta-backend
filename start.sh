#!/bin/bash
set -e

echo "🚀 Starting SkySearch AI..."

# Run migrations
echo "📦 Running database migrations..."
alembic upgrade head

# Start the app
echo "🌐 Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info