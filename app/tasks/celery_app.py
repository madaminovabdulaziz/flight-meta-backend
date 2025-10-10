from celery import Celery
from celery.schedules import crontab
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "skysearch_tasks",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        'app.tasks.cache_tasks',
        'app.tasks.cleanup_tasks',
        'app.tasks.analytics_tasks',
    ]
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes max per task
    task_soft_time_limit=240,  # Soft limit at 4 minutes
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks (prevent memory leaks)
    task_acks_late=True,  # Acknowledge task after completion (safer)
    task_reject_on_worker_lost=True,
    result_expires=3600,  # Keep results for 1 hour
)

# Celery Beat Schedule - Periodic Tasks
celery_app.conf.beat_schedule = {
    # Refresh hot routes cache every 30 minutes
    'refresh-hot-routes': {
        'task': 'app.tasks.cache_tasks.refresh_hot_routes_cache',
        'schedule': 1800.0,  # 30 minutes
        'options': {'expires': 1500}  # Expire if not executed in 25 min
    },
    
    # Warm popular search cache every hour
    'warm-popular-searches': {
        'task': 'app.tasks.cache_tasks.warm_popular_searches',
        'schedule': 3600.0,  # 1 hour
    },
    
    # Clean up expired saved searches daily at 3 AM
    'cleanup-expired-searches': {
        'task': 'app.tasks.cleanup_tasks.cleanup_expired_searches',
        'schedule': crontab(hour=3, minute=0),
    },
    
    # Clean up old click records monthly (keep last 90 days)
    'cleanup-old-clicks': {
        'task': 'app.tasks.cleanup_tasks.cleanup_old_clicks',
        'schedule': crontab(day_of_month=1, hour=4, minute=0),
    },
    
    # Check price alerts every 6 hours
    'check-price-alerts': {
        'task': 'app.tasks.analytics_tasks.check_price_alerts',
        'schedule': 21600.0,  # 6 hours
    },
    
    # Generate daily analytics report at 6 AM
    'generate-daily-analytics': {
        'task': 'app.tasks.analytics_tasks.generate_daily_analytics',
        'schedule': crontab(hour=6, minute=0),
    },
}

logger.info("✅ Celery configured successfully")