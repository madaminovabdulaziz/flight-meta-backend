"""
Production Flight Search Service
Generates realistic mock flight data with ALL fields required for 6-factor ranking.

This mock service simulates real flight API responses with:
- Complete airline information (ratings, on-time performance)
- Detailed layover information
- Airport convenience metrics
- Booking metadata
- All fields needed for ranking algorithm

Production notes:
- Replace with Amadeus/Kiwi/Duffel when ready
- Interface remains identical - nodes don't need changes
- Response format matches real API normalization
"""

from __future__ import annotations
import random
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# REALISTIC AIRLINE DATABASE
# ============================================================================

AIRLINES = {
    "TK": {
        "name": "Turkish Airlines",
        "iata": "TK",
        "rating": 4.4,
        "on_time_score": 0.82,
        "is_low_cost": False,
        "alliance": "Star Alliance",
        "typical_price_multiplier": 1.0,
    },
    "QR": {
        "name": "Qatar Airways",
        "iata": "QR",
        "rating": 4.7,
        "on_time_score": 0.88,
        "is_low_cost": False,
        "alliance": "Oneworld",
        "typical_price_multiplier": 1.15,
    },
    "EK": {
        "name": "Emirates",
        "iata": "EK",
        "rating": 4.6,
        "on_time_score": 0.85,
        "is_low_cost": False,
        "alliance": None,
        "typical_price_multiplier": 1.20,
    },
    "BA": {
        "name": "British Airways",
        "iata": "BA",
        "rating": 4.0,
        "on_time_score": 0.78,
        "is_low_cost": False,
        "alliance": "Oneworld",
        "typical_price_multiplier": 1.10,
    },
    "LH": {
        "name": "Lufthansa",
        "iata": "LH",
        "rating": 4.2,
        "on_time_score": 0.80,
        "is_low_cost": False,
        "alliance": "Star Alliance",
        "typical_price_multiplier": 1.05,
    },
    "PC": {
        "name": "Pegasus Airlines",
        "iata": "PC",
        "rating": 3.5,
        "on_time_score": 0.72,
        "is_low_cost": True,
        "alliance": None,
        "typical_price_multiplier": 0.70,
    },
    "W6": {
        "name": "Wizz Air",
        "iata": "W6",
        "rating": 3.2,
        "on_time_score": 0.68,
        "is_low_cost": True,
        "alliance": None,
        "typical_price_multiplier": 0.65,
    },
    "FR": {
        "name": "Ryanair",
        "iata": "FR",
        "rating": 3.0,
        "on_time_score": 0.70,
        "is_low_cost": True,
        "alliance": None,
        "typical_price_multiplier": 0.60,
    },
    "AF": {
        "name": "Air France",
        "iata": "AF",
        "rating": 4.1,
        "on_time_score": 0.79,
        "is_low_cost": False,
        "alliance": "SkyTeam",
        "typical_price_multiplier": 1.08,
    },
    "KL": {
        "name": "KLM",
        "iata": "KL",
        "rating": 4.3,
        "on_time_score": 0.83,
        "is_low_cost": False,
        "alliance": "SkyTeam",
        "typical_price_multiplier": 1.07,
    },
}


# ============================================================================
# REALISTIC AIRPORT DATABASE
# ============================================================================

AIRPORTS = {
    # London airports
    "LHR": {
        "name": "London Heathrow",
        "city": "London",
        "country": "United Kingdom",
        "distance_to_city_km": 25,
        "is_main": True,
    },
    "LGW": {
        "name": "London Gatwick",
        "city": "London",
        "country": "United Kingdom",
        "distance_to_city_km": 45,
        "is_main": False,
    },
    "STN": {
        "name": "London Stansted",
        "city": "London",
        "country": "United Kingdom",
        "distance_to_city_km": 55,
        "is_main": False,
    },
    "LTN": {
        "name": "London Luton",
        "city": "London",
        "country": "United Kingdom",
        "distance_to_city_km": 50,
        "is_main": False,
    },
    
    # Istanbul airports
    "IST": {
        "name": "Istanbul Airport",
        "city": "Istanbul",
        "country": "Turkey",
        "distance_to_city_km": 35,
        "is_main": True,
    },
    "SAW": {
        "name": "Sabiha Gökçen Airport",
        "city": "Istanbul",
        "country": "Turkey",
        "distance_to_city_km": 45,
        "is_main": False,
    },
    
    # Major hub airports (for layovers)
    "DXB": {
        "name": "Dubai International",
        "city": "Dubai",
        "country": "UAE",
        "distance_to_city_km": 10,
        "is_main": True,
    },
    "DOH": {
        "name": "Hamad International",
        "city": "Doha",
        "country": "Qatar",
        "distance_to_city_km": 12,
        "is_main": True,
    },
    "FRA": {
        "name": "Frankfurt Airport",
        "city": "Frankfurt",
        "country": "Germany",
        "distance_to_city_km": 12,
        "is_main": True,
    },
    "MUC": {
        "name": "Munich Airport",
        "city": "Munich",
        "country": "Germany",
        "distance_to_city_km": 35,
        "is_main": True,
    },
    "CDG": {
        "name": "Paris Charles de Gaulle",
        "city": "Paris",
        "country": "France",
        "distance_to_city_km": 25,
        "is_main": True,
    },
    "AMS": {
        "name": "Amsterdam Schiphol",
        "city": "Amsterdam",
        "country": "Netherlands",
        "distance_to_city_km": 15,
        "is_main": True,
    },
}


# ============================================================================
# LAYOVER SCENARIOS
# ============================================================================

LAYOVER_HUB_ROUTES = {
    ("LHR", "IST"): ["DXB", "DOH", "FRA", "MUC"],
    ("LGW", "IST"): ["DXB", "DOH", "AMS"],
    ("STN", "IST"): ["FRA", "MUC"],
    ("LTN", "IST"): ["AMS", "FRA"],
}


# ============================================================================
# FLIGHT SERVICE
# ============================================================================

class FlightService:
    """
    Production-grade mock flight service.
    Generates realistic flight options with complete metadata for ranking.
    """
    
    @staticmethod
    def _generate_flight_id() -> str:
        """Generate unique flight ID."""
        return f"FL{random.randint(10000, 99999)}"
    
    @staticmethod
    def _calculate_base_price(
        origin: str,
        destination: str,
        departure_date: date,
        travel_class: Optional[str],
        airline_data: Dict[str, Any],
    ) -> float:
        """
        Calculate realistic base price considering:
        - Route distance (approximation)
        - Seasonality
        - Airline tier
        - Travel class
        """
        # Base price by route (simplified)
        base_prices = {
            ("LHR", "IST"): 280,
            ("LGW", "IST"): 250,
            ("STN", "IST"): 200,
            ("LTN", "IST"): 180,
        }
        
        route_key = (origin, destination)
        base = base_prices.get(route_key, 300)
        
        # Apply airline multiplier
        base *= airline_data["typical_price_multiplier"]
        
        # Seasonality (simplified)
        days_until_departure = (departure_date - date.today()).days
        if days_until_departure < 7:
            base *= 1.5  # Last-minute premium
        elif days_until_departure < 14:
            base *= 1.3
        elif days_until_departure > 90:
            base *= 0.85  # Early bird discount
        
        # Travel class multiplier
        class_multipliers = {
            "Economy": 1.0,
            "Premium Economy": 1.8,
            "Business": 3.5,
            "First": 6.0,
        }
        base *= class_multipliers.get(travel_class or "Economy", 1.0)
        
        # Add some randomness (±15%)
        variance = random.uniform(0.85, 1.15)
        
        return round(base * variance, 2)
    
    @staticmethod
    def _generate_direct_flight(
        origin: str,
        destination: str,
        departure_date: date,
        airline_code: str,
        airline_data: Dict[str, Any],
        travel_class: Optional[str],
    ) -> Dict[str, Any]:
        """Generate a realistic direct flight."""
        
        # Departure time (realistic distribution)
        hour_choices = [6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        dep_hour = random.choice(hour_choices)
        dep_minute = random.choice([0, 15, 30, 45])
        
        dep_time = datetime.combine(
            departure_date,
            datetime.min.time().replace(hour=dep_hour, minute=dep_minute)
        )
        
        # Flight duration (London to Istanbul: ~3h45m direct)
        base_duration = 225  # minutes
        variance = random.randint(-15, 15)
        duration_minutes = base_duration + variance
        
        arr_time = dep_time + timedelta(minutes=duration_minutes)
        
        # Price calculation
        price = FlightService._calculate_base_price(
            origin, destination, departure_date, travel_class, airline_data
        )
        
        # Get airport data
        origin_airport = AIRPORTS.get(origin, {"distance_to_city_km": 30})
        dest_airport = AIRPORTS.get(destination, {"distance_to_city_km": 30})
        
        return {
            "id": FlightService._generate_flight_id(),
            "airline": airline_code,
            "airline_name": airline_data["name"],
            "price": price,
            "currency": "GBP",
            "origin": origin,
            "destination": destination,
            "departure_time": dep_time.isoformat(),
            "arrival_time": arr_time.isoformat(),
            "duration_minutes": duration_minutes,
            "stops": 0,
            "layovers": [],
            "fare_class": travel_class or "Economy",
            "refundable": random.choice([True, False]),
            "baggage_included": not airline_data["is_low_cost"],
            "airline_rating": airline_data["rating"],
            "on_time_score": airline_data["on_time_score"],
            "departure_airport_distance_to_city_km": origin_airport["distance_to_city_km"],
            "arrival_airport_distance_to_city_km": dest_airport["distance_to_city_km"],
            "booking_link": f"https://booking.example.com/flight/{FlightService._generate_flight_id()}",
            "is_mock": True,
        }
    
    @staticmethod
    def _generate_connecting_flight(
        origin: str,
        destination: str,
        departure_date: date,
        airline_code: str,
        airline_data: Dict[str, Any],
        travel_class: Optional[str],
        num_stops: int = 1,
    ) -> Dict[str, Any]:
        """Generate a realistic connecting flight with layover(s)."""
        
        # Select layover hub(s)
        route_key = (origin, destination)
        available_hubs = LAYOVER_HUB_ROUTES.get(route_key, ["FRA", "AMS"])
        
        if num_stops == 1:
            layover_hubs = [random.choice(available_hubs)]
        else:
            layover_hubs = random.sample(available_hubs, min(num_stops, len(available_hubs)))
        
        # First leg departure
        dep_hour = random.choice([6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17])
        dep_time = datetime.combine(
            departure_date,
            datetime.min.time().replace(hour=dep_hour, minute=random.choice([0, 30]))
        )
        
        # Build layover details
        layovers = []
        current_time = dep_time
        total_duration = 0
        
        for hub in layover_hubs:
            # Leg flight time
            leg_duration = random.randint(90, 180)
            current_time += timedelta(minutes=leg_duration)
            total_duration += leg_duration
            
            # Layover duration
            layover_duration_minutes = random.choice([60, 90, 120, 150, 180, 240, 300, 480])
            
            # Check if overnight
            is_overnight = False
            arrival_hour = current_time.hour
            departure_after_layover = current_time + timedelta(minutes=layover_duration_minutes)
            if arrival_hour >= 22 or departure_after_layover.hour <= 5:
                is_overnight = True
            
            # Risky short connection?
            min_connection_minutes = layover_duration_minutes
            
            layovers.append({
                "airport": hub,
                "airport_name": AIRPORTS.get(hub, {}).get("name", hub),
                "duration_minutes": layover_duration_minutes,
                "overnight": is_overnight,
                "min_connection_minutes": min_connection_minutes,
            })
            
            current_time += timedelta(minutes=layover_duration_minutes)
            total_duration += layover_duration_minutes
        
        # Final leg
        final_leg_duration = random.randint(90, 180)
        total_duration += final_leg_duration
        arr_time = current_time + timedelta(minutes=final_leg_duration)
        
        # Price (connecting flights cheaper)
        base_price = FlightService._calculate_base_price(
            origin, destination, departure_date, travel_class, airline_data
        )
        # Discount for connections
        discount = 0.80 if num_stops == 1 else 0.65
        price = round(base_price * discount, 2)
        
        # Get airport data
        origin_airport = AIRPORTS.get(origin, {"distance_to_city_km": 30})
        dest_airport = AIRPORTS.get(destination, {"distance_to_city_km": 30})
        
        return {
            "id": FlightService._generate_flight_id(),
            "airline": airline_code,
            "airline_name": airline_data["name"],
            "price": price,
            "currency": "GBP",
            "origin": origin,
            "destination": destination,
            "departure_time": dep_time.isoformat(),
            "arrival_time": arr_time.isoformat(),
            "duration_minutes": total_duration,
            "stops": num_stops,
            "layovers": layovers,
            "fare_class": travel_class or "Economy",
            "refundable": random.choice([True, False]),
            "baggage_included": not airline_data["is_low_cost"],
            "airline_rating": airline_data["rating"],
            "on_time_score": airline_data["on_time_score"],
            "departure_airport_distance_to_city_km": origin_airport["distance_to_city_km"],
            "arrival_airport_distance_to_city_km": dest_airport["distance_to_city_km"],
            "booking_link": f"https://booking.example.com/flight/{FlightService._generate_flight_id()}",
            "is_mock": True,
        }
    
    @staticmethod
    async def search_flights(
        origin: str,
        destination: str,
        departure_date: date,
        return_date: Optional[date] = None,
        passengers: int = 1,
        travel_class: Optional[str] = None,
        budget: Optional[float] = None,
        flexibility: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate realistic flight search results.
        
        Returns:
            List of normalized flight dictionaries with all fields required for 6-factor ranking.
        """
        
        logger.info(
            f"[FlightService] Searching: {origin} → {destination}, "
            f"dep={departure_date}, class={travel_class}, budget={budget}"
        )
        
        flights = []
        
        # Generate mix of airlines and flight types
        # 40% direct, 40% 1-stop, 20% 2-stop
        flight_distribution = [
            ("direct", 0.40),
            ("1-stop", 0.40),
            ("2-stop", 0.20),
        ]
        
        num_flights = 20
        
        for _ in range(num_flights):
            # Select random airline
            airline_code = random.choice(list(AIRLINES.keys()))
            airline_data = AIRLINES[airline_code]
            
            # Determine flight type
            flight_type = random.choices(
                [t[0] for t in flight_distribution],
                weights=[t[1] for t in flight_distribution],
                k=1
            )[0]
            
            if flight_type == "direct":
                flight = FlightService._generate_direct_flight(
                    origin, destination, departure_date,
                    airline_code, airline_data, travel_class
                )
            elif flight_type == "1-stop":
                flight = FlightService._generate_connecting_flight(
                    origin, destination, departure_date,
                    airline_code, airline_data, travel_class, num_stops=1
                )
            else:  # 2-stop
                flight = FlightService._generate_connecting_flight(
                    origin, destination, departure_date,
                    airline_code, airline_data, travel_class, num_stops=2
                )
            
            # Apply budget filter if specified
            if budget and flight["price"] > budget * 1.2:
                # Skip flights way over budget, but allow some flexibility
                continue
            
            flights.append(flight)
        
        logger.info(f"[FlightService] Generated {len(flights)} realistic flights")
        
        return flights