from celery import shared_task
from datetime import date, timedelta
import asyncio
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)


@shared_task(name='app.tasks.cache_tasks.refresh_hot_routes_cache', bind=True)
def refresh_hot_routes_cache(self):
    """
    Refresh cache for hot routes from Travelpayouts.
    This keeps the /hot-routes endpoint fast.
    """
    from services.amadeus_service import redis_client
    import httpx
    import json
    from app.core.config import settings
    
    logger.info("🔄 Starting hot routes cache refresh...")
    
    # Popular origins to refresh
    origins = ["TAS", "IST", "DXB", "LON", "NYC"]
    
    refreshed_count = 0
    failed_count = 0
    
    for origin in origins:
        try:
            # Fetch from Travelpayouts
            params = {
                "origin": origin,
                "currency": "USD",
                "limit": 10,
                "token": settings.TRAVELPAYOUTS_API_TOKEN
            }
            
            response = httpx.get(
                "https://api.travelpayouts.com/v2/prices/latest",
                params=params,
                timeout=10.0
            )
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("success"):
                # Store in cache
                cache_key = f"hot_routes:{origin}:10:travelpayouts"
                routes = data.get("data", [])
                
                # Run async Redis operation
                asyncio.run(
                    redis_client.set(
                        cache_key,
                        json.dumps(routes),
                        ex=3600  # 1 hour TTL
                    )
                )
                
                refreshed_count += 1
                logger.info(f"✅ Refreshed cache for {origin}: {len(routes)} routes")
            else:
                logger.warning(f"⚠️ No data returned for {origin}")
                failed_count += 1
                
        except Exception as e:
            logger.error(f"❌ Failed to refresh {origin}: {str(e)}")
            failed_count += 1
    
    result = {
        "refreshed": refreshed_count,
        "failed": failed_count,
        "total": len(origins)
    }
    
    logger.info(f"✅ Hot routes refresh completed: {result}")
    return result


@shared_task(name='app.tasks.cache_tasks.warm_popular_searches')
def warm_popular_searches():
    """
    Pre-warm cache for popular flight searches.
    Analyzes click data to find popular routes and dates.
    """
    from sqlalchemy import create_engine, func, select
    from sqlalchemy.orm import Session
    from app.models.models import Click
    from datetime import datetime
    
    logger.info("🔥 Starting cache warming for popular searches...")
    
    # Use sync engine for Celery
    engine = create_engine(
        settings.DATABASE_URL.replace("+aiomysql", "+pymysql"),
        pool_pre_ping=True
    )
    
    with Session(engine) as session:
        # Find top 10 most clicked routes in last 7 days
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        popular_routes = session.execute(
            select(
                Click.origin,
                Click.destination,
                func.count(Click.id).label('click_count')
            )
            .where(Click.clicked_at >= week_ago)
            .group_by(Click.origin, Click.destination)
            .order_by(func.count(Click.id).desc())
            .limit(10)
        ).all()
        
        if not popular_routes:
            logger.info("ℹ️ No popular routes found to warm cache")
            return {"warmed": 0}
        
        warmed_count = 0
        
        # Warm cache for each popular route
        for route in popular_routes:
            origin = route.origin
            destination = route.destination
            
            try:
                # Pre-fetch for next 7, 14, 21, 30 days
                for days_ahead in [7, 14, 21, 30]:
                    departure_date = (date.today() + timedelta(days=days_ahead)).isoformat()
                    
                    # Trigger search (will cache result)
                    from services.amadeus_service import search_flights_amadeus
                    asyncio.run(
                        search_flights_amadeus(
                            origin=origin,
                            destination=destination,
                            departure_date=departure_date,
                            max_results=10
                        )
                    )
                    
                    warmed_count += 1
                    logger.info(f"🔥 Warmed cache: {origin}-{destination} on {departure_date}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to warm {origin}-{destination}: {str(e)}")
    
    result = {"warmed": warmed_count}
    logger.info(f"✅ Cache warming completed: {result}")
    return result
