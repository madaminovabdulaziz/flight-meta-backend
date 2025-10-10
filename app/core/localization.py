# app/core/localization.py
"""
Smart Currency & Localization System

Design Philosophy:
1. Auto-detect user preferences (IP → country → currency/language)
2. Allow explicit override via query params
3. Cache conversion rates
4. Support multi-currency display
5. i18n ready for translations
"""

from typing import Any, Optional, Dict, Tuple
from datetime import datetime, timedelta
import httpx
import json
from fastapi import Request, Query
from app.core.config import settings
from services.amadeus_service import redis_client
import logging

logger = logging.getLogger(__name__)

# Supported currencies with their symbols and names
SUPPORTED_CURRENCIES = {
    "USD": {"symbol": "$", "name": "US Dollar", "locale": "en_US"},
    "UZS": {"symbol": "so'm", "name": "Uzbek Som", "locale": "uz_UZ"},
    "EUR": {"symbol": "€", "name": "Euro", "locale": "en_EU"},
    "GBP": {"symbol": "£", "name": "British Pound", "locale": "en_GB"},
    "RUB": {"symbol": "₽", "name": "Russian Ruble", "locale": "ru_RU"},
    "TRY": {"symbol": "₺", "name": "Turkish Lira", "locale": "tr_TR"},
    "AED": {"symbol": "د.إ", "name": "UAE Dirham", "locale": "ar_AE"},
    "KRW": {"symbol": "₩", "name": "South Korean Won", "locale": "ko_KR"},
    "INR": {"symbol": "₹", "name": "Indian Rupee", "locale": "en_IN"},
    "CNY": {"symbol": "¥", "name": "Chinese Yuan", "locale": "zh_CN"},
}

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "flag": "🇺🇸"},
    "uz": {"name": "O'zbek", "flag": "🇺🇿"},
    "ru": {"name": "Русский", "flag": "🇷🇺"},
    "tr": {"name": "Türkçe", "flag": "🇹🇷"},
}

# Country to currency mapping (for auto-detection)
COUNTRY_TO_CURRENCY = {
    "UZ": "UZS",  # Uzbekistan
    "US": "USD",  # United States
    "GB": "GBP",  # United Kingdom
    "EU": "EUR",  # European Union
    "RU": "RUB",  # Russia
    "TR": "TRY",  # Turkey
    "AE": "AED",  # UAE
    "KR": "KRW",  # South Korea
    "IN": "INR",  # India
    "CN": "CNY",  # China
}

# Country to language mapping (for auto-detection)
COUNTRY_TO_LANGUAGE = {
    "UZ": "uz",
    "US": "en",
    "GB": "en",
    "RU": "ru",
    "TR": "tr",
    "AE": "en",
}


class LocalizationService:
    """Handles currency conversion and localization"""
    
    EXCHANGE_RATE_API = "https://api.exchangerate-api.com/v4/latest/USD"
    CACHE_TTL_RATES = 3600 * 6  # 6 hours
    
    @staticmethod
    async def get_exchange_rates() -> Dict[str, float]:
        """
        Fetch current exchange rates from API.
        Cached for 6 hours since rates don't change that frequently.
        """
        cache_key = "exchange_rates:latest"
        
        # Check cache
        try:
            cached = await redis_client.get(cache_key)
            if cached:
                rates = json.loads(cached.decode('utf-8'))
                logger.debug("✅ Exchange rates from cache")
                return rates
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
        
        # Fetch from API
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(LocalizationService.EXCHANGE_RATE_API)
                response.raise_for_status()
                data = response.json()
                rates = data.get("rates", {})
                
                # Cache the rates
                await redis_client.set(
                    cache_key,
                    json.dumps(rates),
                    ex=LocalizationService.CACHE_TTL_RATES
                )
                
                logger.info(f"💱 Fetched fresh exchange rates for {len(rates)} currencies")
                return rates
                
        except Exception as e:
            logger.error(f"Failed to fetch exchange rates: {e}")
            # Return fallback rates if API fails
            return LocalizationService.get_fallback_rates()
    
    @staticmethod
    def get_fallback_rates() -> Dict[str, float]:
        """
        Fallback exchange rates (approximate, updated monthly).
        Used when API is unavailable.
        """
        return {
            "USD": 1.0,
            "UZS": 12750.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "RUB": 92.50,
            "TRY": 34.20,
            "AED": 3.67,
            "KRW": 1340.0,
            "INR": 83.50,
            "CNY": 7.24,
        }
    
    @staticmethod
    async def convert_price(
        amount: float,
        from_currency: str = "USD",
        to_currency: str = "USD"
    ) -> float:
        """Convert price from one currency to another"""
        if from_currency == to_currency:
            return amount
        
        rates = await LocalizationService.get_exchange_rates()
        
        # Convert to USD first (our base currency)
        if from_currency != "USD":
            amount_usd = amount / rates.get(from_currency, 1.0)
        else:
            amount_usd = amount
        
        # Convert from USD to target currency
        if to_currency != "USD":
            result = amount_usd * rates.get(to_currency, 1.0)
        else:
            result = amount_usd
        
        return round(result, 2)
    
    @staticmethod
    async def convert_price_dict(
        price_dict: Dict,
        to_currency: str,
        price_fields: list = ["price", "cheapest_price", "average_price", "most_expensive_price"]
    ) -> Dict:
        """Convert all price fields in a dictionary to target currency"""
        converted = price_dict.copy()
        
        from_currency = converted.get("currency", "USD")
        
        if from_currency == to_currency:
            return converted
        
        for field in price_fields:
            if field in converted and converted[field] is not None:
                converted[field] = await LocalizationService.convert_price(
                    converted[field],
                    from_currency,
                    to_currency
                )
        
        converted["currency"] = to_currency
        converted["original_currency"] = from_currency
        
        return converted
    
    @staticmethod
    def format_price(amount: float, currency: str) -> str:
        """Format price with currency symbol"""
        currency_info = SUPPORTED_CURRENCIES.get(currency, {"symbol": "$"})
        symbol = currency_info["symbol"]
        
        # For some currencies, symbol goes after number
        if currency in ["TRY", "RUB", "UZS"]:
            return f"{amount:,.0f} {symbol}"
        else:
            return f"{symbol}{amount:,.2f}"
    
    @staticmethod
    async def detect_user_location(request: Request) -> Tuple[str, str]:
        """
        Detect user's country from IP address.
        Returns (country_code, currency_code)
        
        Uses CloudFlare headers or IP lookup service.
        """
        # Try CloudFlare headers first (if using CloudFlare)
        cf_country = request.headers.get("CF-IPCountry")
        if cf_country:
            currency = COUNTRY_TO_CURRENCY.get(cf_country, "USD")
            logger.debug(f"🌍 Detected location: {cf_country} → {currency}")
            return cf_country, currency
        
        # Try to get IP from headers
        ip = (
            request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
            or request.headers.get("X-Real-IP")
            or request.client.host if request.client else None
        )
        
        if not ip or ip in ["127.0.0.1", "localhost"]:
            # Default to Uzbekistan for local development
            logger.debug("🏠 Local IP detected, defaulting to UZ/UZS")
            return "UZ", "UZS"
        
        # Use IP geolocation API (free tier)
        try:
            cache_key = f"geoip:{ip}"
            cached = await redis_client.get(cache_key)
            
            if cached:
                country = cached.decode('utf-8')
            else:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"https://ipapi.co/{ip}/country/")
                    country = response.text.strip()
                    
                    # Cache for 24 hours
                    await redis_client.set(cache_key, country, ex=86400)
            
            currency = COUNTRY_TO_CURRENCY.get(country, "USD")
            logger.info(f"🌍 IP {ip} → {country} → {currency}")
            return country, currency
            
        except Exception as e:
            logger.warning(f"Failed to detect location: {e}")
            return "UZ", "UZS"  # Default fallback


async def get_user_preferences(
    request: Request,
    currency: Optional[str] = None,
    language: Optional[str] = None,
    current_user: Optional[Any] = None  # User object if authenticated
) -> Dict[str, str]:
    """
    Smart preference detection with explicit override support.
    
    Priority:
    1. Explicit query parameters (?currency=UZS&language=uz)
    2. Authenticated user's saved preferences (from database)
    3. Auto-detected from IP/location
    4. System defaults (UZS for Uzbekistan, USD otherwise)
    
    Returns: {
        "currency": "UZS",
        "language": "uz",
        "country": "UZ",
        "currency_symbol": "so'm",
        "currency_name": "Uzbek Som"
    }
    """
    # Detect location first
    country, detected_currency = await LocalizationService.detect_user_location(request)
    detected_language = COUNTRY_TO_LANGUAGE.get(country, "en")
    
    # Priority 1: Explicit query parameters
    if currency:
        final_currency = currency.upper()
    # Priority 2: Authenticated user's preference
    elif current_user and hasattr(current_user, 'preferred_currency') and current_user.preferred_currency:
        final_currency = current_user.preferred_currency.upper()
        logger.info(f"👤 Using authenticated user's currency: {final_currency}")
    # Priority 3: Auto-detected
    else:
        final_currency = detected_currency
    
    # Same logic for language
    if language:
        final_language = language.lower()
    elif current_user and hasattr(current_user, 'preferred_language') and current_user.preferred_language:
        final_language = current_user.preferred_language.lower()
        logger.info(f"👤 Using authenticated user's language: {final_language}")
    else:
        final_language = detected_language
    
    # Validate currency is supported
    if final_currency not in SUPPORTED_CURRENCIES:
        logger.warning(f"Unsupported currency {final_currency}, falling back to USD")
        final_currency = "USD"
    
    # Validate language is supported
    if final_language not in SUPPORTED_LANGUAGES:
        logger.warning(f"Unsupported language {final_language}, falling back to en")
        final_language = "en"
    
    return {
        "currency": final_currency,
        "language": final_language,
        "country": country,
        "currency_symbol": SUPPORTED_CURRENCIES[final_currency]["symbol"],
        "currency_name": SUPPORTED_CURRENCIES[final_currency]["name"],
    }

