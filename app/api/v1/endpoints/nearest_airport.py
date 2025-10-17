from fastapi import APIRouter, Request, HTTPException, Query
from services.ip_geolocation import IPGeolocationService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()
geo_service = IPGeolocationService()

# === Translations ===
MESSAGES = {
    "en": {
        "main": "Showing popular routes from {city}",
        "secondary": "Check out the most searched destinations"
    },
    "ru": {
        "main": "Популярные маршруты из города {city}",
        "secondary": "Посмотрите самые популярные направления"
    },
    "uz": {
        "main": "{city} shahridan mashhur yo‘nalishlar",
        "secondary": "Eng ko‘p qidirilgan yo‘nalishlarni ko‘ring"
    }
}


@router.get("/nearest-airport")
async def get_nearest_airport(
    request: Request,
    lang: str = Query("en", regex="^(en|ru|uz)$")
):
    """
    Get nearest major airport based on user's IP address

    Query params:
    - lang: "en", "ru", or "uz"

    Returns localized response with two messages.
    """
    try:
        # Extract client IP
        client_ip = request.client.host
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()

        cf_ip = request.headers.get("CF-Connecting-IP")
        if cf_ip:
            client_ip = cf_ip

        # Get nearest airport
        result = await geo_service.get_nearest_airport_from_ip(client_ip)
        city_name = result["airport"]["city"]

        # Localized messages
        main_msg = MESSAGES[lang]["main"].format(city=city_name)
        secondary_msg = MESSAGES[lang]["secondary"]

        return {
            "success": True,
            **result,
            "message": main_msg,
            "secondary_message": secondary_msg
        }

    except Exception as e:
        logger.error(f"Error getting nearest airport: {e}")

        main_msg = MESSAGES[lang]["main"].format(city="Tashkent")
        secondary_msg = MESSAGES[lang]["secondary"]

        # Default (Tashkent)
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
            "message": main_msg,
            "secondary_message": secondary_msg
        }


@router.get("/test-location/{ip}")
async def test_location(
    ip: str,
    lang: str = Query("en", regex="^(en|ru|uz)$")
):
    """
    Test endpoint to check IP geolocation (for development)
    Example: /api/v1/test-location/8.8.8.8?lang=ru
    """
    result = await geo_service.get_nearest_airport_from_ip(ip)
    city_name = result["airport"]["city"]
    main_msg = MESSAGES[lang]["main"].format(city=city_name)
    secondary_msg = MESSAGES[lang]["secondary"]

    return {**result, "message": main_msg, "secondary_message": secondary_msg}
