# services/flight_search.py
"""
Simplified Flight Search Service - TravelPayouts Only

No Amadeus, no aggregation complexity. Pure metasearch with TravelPayouts.
When ready to become OTA, add other providers.

This is your MVP search engine!
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from schemas.flight_segment import Flight
from services.travelpayouts_service import get_travelpayouts_service
from services.cache_decorator import redis_cache

logger = logging.getLogger(__name__)

# Cache for 15 minutes (TravelPayouts data updates every 30min anyway)
SEARCH_CACHE_TTL = 900


class FlightSearchService:
    """
    Simple flight search using only TravelPayouts.
    
    Strategies:
    - search_flights: Main search endpoint
    - get_cheapest: Find absolute cheapest from origin
    - get_calendar: Flexible dates view
    - get_hot_routes: Popular destinations
    """
    
    def __init__(self):
        self.tp_service = get_travelpayouts_service()
    
    @redis_cache(ttl=SEARCH_CACHE_TTL)
    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        children: int = 0,
        cabin_class: str = "ECONOMY",
        non_stop: bool = False,
        max_price: Optional[float] = None,
        currency: str = "USD",
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Main flight search - returns structured response with metadata.
        
        Args:
            origin: 3-letter IATA code
            destination: 3-letter IATA code
            departure_date: YYYY-MM-DD
            return_date: YYYY-MM-DD (optional for one-way)
            adults: Number of passengers
            max_price: Filter by max price
            currency: USD, EUR, RUB, etc.
            limit: Max results to return
        
        Returns:
            {
                "flights": List[Flight],
                "total_results": int,
                "metadata": {
                    "route": "TAS→DXB",
                    "price_range": {...},
                    "cheapest": {...},
                    "recommendations": [...]
                }
            }
        """
        logger.info(f"Searching: {origin}→{destination}, {departure_date}, {adults} pax")
        
        try:
            # Search with TravelPayouts
            flights = await self.tp_service.search_flights(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                return_date=return_date,
                adults=adults,
                currency=currency,
                limit=limit
            )
            
            # Filter by max price if specified
            if max_price:
                flights = [f for f in flights if f.price <= max_price]
                logger.info(f"Filtered to {len(flights)} flights under ${max_price}")
            
            # Filter by non-stop if requested
            if non_stop:
                flights = [f for f in flights if f.outbound.stops == 0]
                logger.info(f"Filtered to {len(flights)} non-stop flights")
            
            # Sort by price (cheapest first)
            flights.sort(key=lambda f: f.price)
            
            # Generate metadata
            metadata = self._generate_metadata(
                flights=flights,
                origin=origin,
                destination=destination,
                departure_date=departure_date
            )
            
            return {
                "flights": flights[:limit],
                "total_results": len(flights),
                "metadata": metadata,
                "provider": "TravelPayouts"
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return {
                "flights": [],
                "total_results": 0,
                "error": str(e),
                "metadata": {
                    "recommendations": [
                        "Try different dates",
                        "Check if route exists",
                        "Try nearby airports"
                    ]
                }
            }
    
    async def get_cheapest_from_origin(
        self,
        origin: str,
        currency: str = "USD",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find cheapest flights from an origin to ANY destination.
        
        Perfect for:
        - "Cheapest flights from Tashkent"
        - Destination inspiration
        - Hot deals section
        """
        return await self.tp_service.get_cheapest_tickets(
            origin=origin,
            currency=currency,
            limit=limit
        )
    
    async def get_price_calendar(
        self,
        origin: str,
        destination: str,
        month: Optional[str] = None,
        currency: str = "USD"
    ) -> Dict[str, Any]:
        """
        Get price calendar for flexible date search.
        
        Shows cheapest price for each day of the month.
        Perfect for "I'm flexible with dates" users.
        """
        if not month:
            month = datetime.now().strftime("%Y-%m")
        
        return await self.tp_service.get_price_calendar(
            origin=origin,
            destination=destination,
            departure_month=month,
            currency=currency
        )
    
    async def get_hot_routes(
        self,
        origin: str = "TAS",
        currency: str = "USD",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get popular/trending routes from origin.
        
        Uses cheapest tickets as proxy for popularity.
        """
        return await self.tp_service.get_popular_routes(
            origin=origin,
            currency=currency
        )[:limit]
    
    def build_booking_link(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        children: int = 0,
        cabin_class: str = "economy"
    ) -> str:
        """
        Generate TravelPayouts affiliate deeplink.
        
        THIS IS HOW YOU MAKE MONEY! 💰
        User clicks this link → books → you earn 3-5% commission
        """
        return self.tp_service.build_deeplink(
            origin=origin,
            destination=destination,
            departure_date=departure_date,
            return_date=return_date,
            adults=adults,
            children=children,
            cabin_class=cabin_class
        )
    
    def _generate_metadata(
        self,
        flights: List[Flight],
        origin: str,
        destination: str,
        departure_date: str
    ) -> Dict[str, Any]:
        """Generate helpful metadata about search results"""
        
        if not flights:
            return {
                "route": f"{origin}→{destination}",
                "recommendations": [
                    "❌ No flights found",
                    "💡 Try different dates (±3 days)",
                    "🔍 Check nearby airports",
                    "📅 Use calendar view for flexible dates"
                ]
            }
        
        prices = [f.price for f in flights]
        
        # Find cheapest and fastest
        cheapest = min(flights, key=lambda f: f.price)
        direct_flights = [f for f in flights if f.outbound.stops == 0]
        
        metadata = {
            "route": f"{origin}→{destination}",
            "departure_date": departure_date,
            "price_range": {
                "min": min(prices),
                "max": max(prices),
                "avg": sum(prices) / len(prices)
            },
            "cheapest_flight": {
                "price": cheapest.price,
                "currency": cheapest.currency,
                "airline": cheapest.outbound.segments[0].airline_code if cheapest.outbound.segments else None,
                "stops": cheapest.outbound.stops
            },
            "has_direct_flights": len(direct_flights) > 0,
            "direct_flights_count": len(direct_flights),
            "recommendations": self._generate_recommendations(flights)
        }
        
        return metadata
    
    def _generate_recommendations(self, flights: List[Flight]) -> List[str]:
        """Smart recommendations based on results"""
        recs = []
        
        if not flights:
            return ["No flights found - try different dates"]
        
        prices = [f.price for f in flights]
        price_range = max(prices) - min(prices)
        
        # Price variance
        if price_range > 200:
            recs.append(
                f"💰 Prices vary by ${price_range:.0f} - book early for best deals"
            )
        
        # Direct flights
        direct_count = len([f for f in flights if f.outbound.stops == 0])
        if direct_count == 0:
            recs.append("✈️ No direct flights - all options have layovers")
        elif direct_count < len(flights) * 0.3:
            cheapest_direct = min([f.price for f in flights if f.outbound.stops == 0], default=999)
            cheapest_stop = min([f.price for f in flights if f.outbound.stops > 0], default=999)
            savings = cheapest_direct - cheapest_stop
            if savings > 50:
                recs.append(f"💡 Save ${savings:.0f} by choosing a 1-stop flight")
        
        # Best time to book
        avg_price = sum(prices) / len(prices)
        cheapest_price = min(prices)
        if cheapest_price < avg_price * 0.8:
            recs.append(
                f"🔥 Great deal! ${cheapest_price:.0f} is 20% below average"
            )
        
        # Flexible dates
        recs.append("📅 Check calendar view - flying ±2 days might save you money")
        
        return recs[:4]  # Max 4 recommendations


# ==================== SINGLETON ====================

_search_service: Optional[FlightSearchService] = None

def get_flight_search() -> FlightSearchService:
    """Get or create search service singleton"""
    global _search_service
    if _search_service is None:
        _search_service = FlightSearchService()
    return _search_service
