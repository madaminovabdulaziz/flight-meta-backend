from locust import HttpUser, task, between
from random import choice, randint
from datetime import datetime, timedelta

# Common IATA airport codes
IATA_CODES = ["TAS", "IST", "LHR", "DXB", "FRA", "CDG", "JFK", "LAX", "DOH", "DEL", "AMS"]

# Cabin classes
CABIN_CLASSES = ["ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"]

# Trip types
TRIP_TYPES = ["one-way", "round-trip"]


class FlightSearchUser(HttpUser):
    wait_time = between(1, 3)  # seconds between tasks

    @task
    def search_flights(self):
        # Random origin and destination (not the same)
        origin, destination = choice(IATA_CODES), choice(IATA_CODES)
        while destination == origin:
            destination = choice(IATA_CODES)

        # Random trip type
        trip_type = choice(TRIP_TYPES)

        # Random future dates
        today = datetime.now()
        dep_date = today + timedelta(days=randint(5, 60))  # 5–60 days ahead

        params = {
            "origin": origin,
            "destination": destination,
            "departure_date": dep_date.strftime("%Y-%m-%d"),
            "trip_type": trip_type,
            "adults": randint(1, 3),
            "children": randint(0, 2),
            "infants": randint(0, 1),
            "cabin_class": choice(CABIN_CLASSES),
            "non_stop": choice([True, False]),
            "max_results": randint(5, 20),
            "currency": choice(["USD", "EUR", "GBP", "AED"]),
        }

        # Add return_date only if round-trip
        if trip_type == "round-trip":
            return_date = dep_date + timedelta(days=randint(3, 14))
            params["return_date"] = return_date.strftime("%Y-%m-%d")

        # Perform the GET request
        with self.client.get("/api/v1/flights/search", params=params, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with {response.status_code}: {response.text}")
