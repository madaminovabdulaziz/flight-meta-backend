# app/services/airport_service.py

import csv
import logging
from typing import List, Optional
import httpx
import json
from datetime import timedelta

# Import YOUR existing Redis client
from services.amadeus_service import redis_client

logger = logging.getLogger(__name__)

OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
CACHE_KEY = "airports:all"
CACHE_TTL = int(timedelta(hours=24).total_seconds())  # 86400 seconds

class AirportService:
    """Airport autocomplete service using OpenFlights data with Redis caching."""
    
    @staticmethod
    async def _fetch_airports_from_source() -> List[dict]:
        """
        Fetch and parse airport data from OpenFlights.
        """
        logger.info(f"Fetching airports from {OPENFLIGHTS_URL}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(OPENFLIGHTS_URL)
            response.raise_for_status()
            
            airports = []
            csv_reader = csv.reader(response.text.strip().split('\n'))
            
            for row in csv_reader:
                if len(row) < 14:
                    continue
                
                # Extract fields
                iata_code = row[4].strip()
                icao_code = row[5].strip()
                name = row[1].strip()
                city = row[2].strip()
                country = row[3].strip()
                
                # Skip if no IATA code
                if not iata_code or iata_code == '\\N' or len(iata_code) != 3:
                    continue
                
                # Parse coordinates
                try:
                    latitude = float(row[6]) if row[6] != '\\N' else None
                    longitude = float(row[7]) if row[7] != '\\N' else None
                except (ValueError, IndexError):
                    latitude = None
                    longitude = None
                
                # Get timezone
                timezone = row[11].strip() if len(row) > 11 and row[11] != '\\N' else 'UTC'
                
                # Get country code
                country_code = AirportService._get_country_code(country)
                
                airports.append({
                    'iata_code': iata_code.upper(),
                    'icao_code': icao_code.upper() if icao_code != '\\N' and len(icao_code) == 4 else None,
                    'name': name,
                    'city': city,
                    'country': country,
                    'country_code': country_code,
                    'timezone': timezone,
                    'latitude': latitude,
                    'longitude': longitude,
                })
            
            logger.info(f"Fetched {len(airports)} valid airports")
            return airports
    
    @staticmethod
    def _get_country_code(country_name: str) -> str:
        """Basic country code mapping."""
        mapping = {
            'United States': 'US', 'United Kingdom': 'GB', 'Canada': 'CA',
            'Australia': 'AU', 'Germany': 'DE', 'France': 'FR', 'Spain': 'ES',
            'Italy': 'IT', 'Japan': 'JP', 'China': 'CN', 'Russia': 'RU',
            'Brazil': 'BR', 'India': 'IN', 'Mexico': 'MX', 'Turkey': 'TR',
            'Netherlands': 'NL', 'Switzerland': 'CH', 'Austria': 'AT',
            'Belgium': 'BE', 'Sweden': 'SE', 'Norway': 'NO', 'Denmark': 'DK',
            'Finland': 'FI', 'Poland': 'PL', 'Greece': 'GR', 'Portugal': 'PT',
            'Ireland': 'IE', 'New Zealand': 'NZ', 'Singapore': 'SG',
            'South Korea': 'KR', 'Thailand': 'TH', 'Malaysia': 'MY',
            'Indonesia': 'ID', 'Philippines': 'PH', 'Vietnam': 'VN',
            'United Arab Emirates': 'AE', 'Saudi Arabia': 'SA',
            'South Africa': 'ZA', 'Egypt': 'EG', 'Israel': 'IL',
            'Argentina': 'AR', 'Chile': 'CL', 'Colombia': 'CO', 'Peru': 'PE',
        }
        return mapping.get(country_name, 'XX')
    
    @staticmethod
    async def _get_all_airports() -> List[dict]:
        """
        Get all airports from cache or fetch from source.
        Compatible with YOUR Redis setup (decode_responses=False).
        """
        try:
            # Try to get from cache (bytes format)
            cached_bytes = await redis_client.get(CACHE_KEY)
            
            if cached_bytes:
                logger.debug("Airport data retrieved from cache")
                # Decode bytes to string, then parse JSON
                return json.loads(cached_bytes.decode('utf-8'))
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
        
        # Cache miss - fetch from source
        logger.info("Cache miss - fetching airports from source")
        airports = await AirportService._fetch_airports_from_source()
        
        try:
            # Store in cache (as bytes, compatible with your Redis setup)
            await redis_client.set(
                CACHE_KEY,
                json.dumps(airports).encode('utf-8'),
                ex=CACHE_TTL
            )
            logger.info(f"Cached {len(airports)} airports for {CACHE_TTL}s")
        except Exception as e:
            logger.warning(f"Failed to cache airports: {e}")
        
        return airports
    
    @staticmethod
    async def search_airports(
        query: str,
        limit: int = 10
    ) -> List['AirportResponse']:
        """
        Search airports by query string.
        """
        from schemas.airports import AirportResponse
        
        if len(query) < 2:
            return []
        
        # Get all airports (from cache or source)
        all_airports = await AirportService._get_all_airports()
        
        # Normalize query
        query_lower = query.lower()
        query_upper = query.upper()
        
        # Search and rank results
        results = []
        for airport in all_airports:
            score = 0
            
            # Exact IATA match (highest priority)
            if airport['iata_code'] == query_upper:
                score = 100
            # IATA starts with query
            elif airport['iata_code'].startswith(query_upper):
                score = 90
            # City exact match
            elif airport['city'].lower() == query_lower:
                score = 80
            # City starts with query
            elif airport['city'].lower().startswith(query_lower):
                score = 70
            # Airport name starts with query
            elif airport['name'].lower().startswith(query_lower):
                score = 60
            # City contains query
            elif query_lower in airport['city'].lower():
                score = 50
            # Airport name contains query
            elif query_lower in airport['name'].lower():
                score = 40
            # Country starts with query
            elif airport['country'].lower().startswith(query_lower):
                score = 30
            # Country contains query
            elif query_lower in airport['country'].lower():
                score = 20
            
            if score > 0:
                results.append({
                    'airport': airport,
                    'score': score
                })
        
        # Sort by score (descending) and take top results
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:limit]
        
        # Convert to response schema
        return [
            AirportResponse(**item['airport'])
            for item in top_results
        ]
    
    @staticmethod
    async def get_airport_by_iata(iata_code: str) -> Optional['AirportResponse']:
        """Get specific airport by IATA code."""
        from schemas.airports import AirportResponse
        
        all_airports = await AirportService._get_all_airports()
        
        for airport in all_airports:
            if airport['iata_code'] == iata_code.upper():
                return AirportResponse(**airport)
        
        return None
    
    @staticmethod
    async def refresh_cache():
        """Manually refresh the airport cache."""
        logger.info("Manually refreshing airport cache")
        try:
            await redis_client.delete(CACHE_KEY)
            await AirportService._get_all_airports()
            logger.info("Airport cache refreshed successfully")
        except Exception as e:
            logger.error(f"Failed to refresh airport cache: {e}")
            raise