
from fastapi import APIRouter, Request, HTTPException
from services.ip_geolocation import IPGeolocationService
import logging

logger = logging.getLogger(__name__)





router = APIRouter()
geo_service = IPGeolocationService()





@router.get("/nearest-airport")
async def get_nearest_airport(request: Request):
    """
    Get nearest major airport based on user's IP address
    
    Returns:
    - airport: {iata, city, name, country}
    - detected_location: User's detected city/country
    - distance_km: Distance to nearest airport
    
    Usage:
    Frontend calls this on page load to personalize "Popular routes from {city}"
    """
    try:
        # Extract client IP (handle proxies)
        client_ip = request.client.host
        
        # Check for forwarded IP (behind proxy/load balancer)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        # Real client IP from CloudFlare (if using CF)
        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            client_ip = cf_ip
        
        # Get nearest airport
        result = await geo_service.get_nearest_airport_from_ip(client_ip)
        
        return {
            "success": True,
            **result
        }
    
    except Exception as e:
        logger.error(f"Error getting nearest airport: {e}")
        # Return default (Tashkent) on error
        return {
            "success": True,
            "airport": {
                "iata": "TAS",
                "city": "Tashkent",
                "name": "Tashkent International Airport",
                "country": "UZ"
            },
            "detected_location": {
                "city": "Tashkent",
                "country": "UZ"
            },
            "distance_km": 0,
            "message": "Showing popular routes from Tashkent"
        }


@router.get("/test-location/{ip}")
async def test_location(ip: str):
    """
    Test endpoint to check IP geolocation (for development)
    Example: /api/v1/test-location/8.8.8.8
    """
    result = await geo_service.get_nearest_airport_from_ip(ip)
    return result