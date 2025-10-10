# services/travelpayouts_service.py
"""
TravelPayouts API Integration Service (Refactored)

Features:
- Latest prices for routes
- Cheapest tickets from an origin
- Monthly price calendar (flexible dates)
- Popular routes (proxy via cheapest)
- Affiliate deeplink builder

Design:
- Uses BaseAPIService for shared HTTP + retries
- Uses @redis_cache for centralized caching
- Maps API payloads to domain models via TravelPayoutsMapper
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from app.core.config import settings
from schemas.flight_segment import Flight

from services.base_api_service import BaseAPIService
from services.cache_decorator import redis_cache
from services.exceptions import (
    TravelPayoutsAPIError,
    TravelPayoutsTimeoutError,
)
from app.mappers.travelpayouts_mapper import TravelPayoutsMapper

logger = logging.getLogger(__name__)

TRAVELPAYOUTS_BASE_URL = "https://api.travelpayouts.com"
HTTP_TIMEOUT_S = 15.0

# Cache TTLs
TTL_LATEST_PRICES = 1800  # 30 minutes
TTL_CALENDAR = 3600       # 1 hour
TTL_CHEAP_ROUTES = 7200   # 2 hours


class TravelPayoutsService(BaseAPIService):
    """TravelPayouts API client for flight and price data."""

    def __init__(self) -> None:
        headers = {"X-Access-Token": settings.TRAVELPAYOUTS_API_TOKEN}
        super().__init__(TRAVELPAYOUTS_BASE_URL, headers=headers, timeout_s=HTTP_TIMEOUT_S)

        self.api_token: str = settings.TRAVELPAYOUTS_API_TOKEN
        self.marker: str = settings.TRAVELPAYOUTS_MARKER

        if not self.api_token:
            logger.warning("TravelPayouts API token not configured!")

    # ---------- Public API ----------

    @redis_cache(ttl=TTL_LATEST_PRICES)
    async def search_flights(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        currency: str = "USD",
        limit: int = 30,
    ) -> List[Flight]:
        """
        High-level search using TravelPayouts 'latest prices' endpoint.

        Notes:
        - TravelPayouts 'latest' returns cached prices from the last ~48h,
          so it's fast but not guaranteed real-time.
        - For basic metasearch use; Amadeus/other GDS can complement for accuracy.
        """
        flights = await self.get_latest_prices(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            currency=currency,
            limit=limit,
        )
        return flights

    @redis_cache(ttl=TTL_LATEST_PRICES)
    async def get_latest_prices(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: Optional[str] = None,
        return_date: Optional[str] = None,
        currency: str = "USD",
        limit: int = 30,
    ) -> List[Flight]:
        """
        Get latest prices (cached by TravelPayouts; great for popular routes).

        API: GET /v2/prices/latest
        Docs: https://support.travelpayouts.com/hc/en-us/articles/203956163
        """
        params = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "currency": currency.upper(),
            "limit": limit,
            "token": self.api_token,
        }
        if departure_date:
            params["beginning_of_period"] = departure_date
        if return_date:
            # Indicates round-trip interest; TravelPayouts still returns one-way-like rows
            params["one_way"] = "false"

        try:
            data = await self._get("/v2/prices/latest", params=params)
        except httpx.TimeoutException as e:
            raise TravelPayoutsTimeoutError("TravelPayouts latest prices timed out") from e
        except httpx.HTTPError as e:
            raise TravelPayoutsAPIError(f"TravelPayouts HTTP error: {e}") from e

        if not data.get("success", False):
            logger.warning("TravelPayouts returned success=false for latest prices")
            return []

        items = data.get("data", []) or []
        flights: List[Flight] = []
        for item in items:
            f = TravelPayoutsMapper.to_flight(item)
            if f:
                flights.append(f)

        logger.info(
            "TravelPayouts latest prices: %d flights for %s→%s",
            len(flights),
            origin,
            destination,
        )
        return flights

    @redis_cache(ttl=TTL_CHEAP_ROUTES)
    async def get_cheapest_tickets(
        self,
        *,
        origin: str,
        destination: Optional[str] = None,
        currency: str = "USD",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Cheapest tickets from an origin (optionally to a specific destination).

        Great for "Cheapest from <origin>" discovery features.

        API: GET /v1/prices/cheap
        """
        params = {
            "origin": origin.upper(),
            "currency": currency.upper(),
            "token": self.api_token,
        }
        if destination:
            params["destination"] = destination.upper()

        try:
            data = await self._get("/v1/prices/cheap", params=params)
        except httpx.TimeoutException as e:
            raise TravelPayoutsTimeoutError("TravelPayouts cheapest timed out") from e
        except httpx.HTTPError as e:
            raise TravelPayoutsAPIError(f"TravelPayouts HTTP error: {e}") from e

        if not data.get("success"):
            return []

        raw = data.get("data", {}) or {}
        results: List[Dict[str, Any]] = []
        for dest_code, prices in raw.items():
            if isinstance(prices, dict):
                for price_data in prices.values():
                    if isinstance(price_data, dict):
                        results.append(
                            {
                                "origin": origin.upper(),
                                "destination": dest_code,
                                "price": float(price_data.get("price", 0)),
                                "currency": currency.upper(),
                                "airline": price_data.get("airline"),
                                "flight_number": price_data.get("flight_number"),
                                "departure_at": price_data.get("departure_at"),
                                "return_at": price_data.get("return_at"),
                                "expires_at": price_data.get("expires_at"),
                            }
                        )
        results.sort(key=lambda x: x["price"])  # cheapest first
        return results[:limit]

    @redis_cache(ttl=TTL_CALENDAR)
    async def get_price_calendar(
        self,
        *,
        origin: str,
        destination: str,
        departure_month: Optional[str] = None,  # YYYY-MM
        return_month: Optional[str] = None,
        currency: str = "USD",
    ) -> Dict[str, Any]:
        """
        Monthly calendar of cheapest prices (perfect for flexible dates).

        API: GET /v1/prices/calendar
        """
        if not departure_month:
            departure_month = datetime.now().strftime("%Y-%m")

        params = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "depart_date": departure_month,
            "currency": currency.upper(),
            "token": self.api_token,
        }
        if return_month:
            params["return_date"] = return_month

        try:
            data = await self._get("/v1/prices/calendar", params=params)
        except httpx.TimeoutException as e:
            raise TravelPayoutsTimeoutError("TravelPayouts calendar timed out") from e
        except httpx.HTTPError as e:
            raise TravelPayoutsAPIError(f"TravelPayouts HTTP error: {e}") from e

        if not data.get("success"):
            return {"prices": {}, "currency": currency}

        calendar = {
            "origin": origin.upper(),
            "destination": destination.upper(),
            "month": departure_month,
            "currency": currency.upper(),
            "prices": data.get("data", {}),
            "updated_at": datetime.utcnow().isoformat(),
        }
        return calendar

    @redis_cache(ttl=TTL_CHEAP_ROUTES)
    async def get_popular_routes(
        self,
        *,
        origin: str,
        currency: str = "USD",
    ) -> List[Dict[str, Any]]:
        """
        Proxy "popular" routes using cheapest tickets across all destinations.
        """
        return await self.get_cheapest_tickets(
            origin=origin,
            destination=None,
            currency=currency,
            limit=20,
        )

    # ---------- Utilities ----------

    def build_deeplink(
        self,
        *,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        cabin_class: str = "economy",
    ) -> str:
        """
        Build TravelPayouts affiliate deeplink with URL-encoded params.

        Format: https://www.travelpayouts.com/flights?marker=YOUR_MARKER&...
        """
        base_url = "https://www.travelpayouts.com/flights"
        params = {
            "marker": self.marker,
            "origin": origin.upper(),
            "destination": destination.upper(),
            "depart_date": departure_date,
            "adults": adults,
            "children": children,
            "infants": infants,
            "trip_class": cabin_class.lower(),
        }
        if return_date:
            params["return_date"] = return_date
        return f"{base_url}?{urlencode(params)}"


_tp_service: Optional[TravelPayoutsService] = None

def get_travelpayouts_service() -> TravelPayoutsService:
    """Get or create TravelPayouts service singleton"""
    global _tp_service
    if _tp_service is None:
        _tp_service = TravelPayoutsService()
    return _tp_service
