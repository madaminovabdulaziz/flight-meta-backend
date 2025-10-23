# app/services/airport_service_multilang.py

import csv
import logging
from typing import List, Optional, Dict
import httpx
import json
from datetime import timedelta
import unicodedata

# Import YOUR existing Redis client
from services.amadeus_service import redis_client

logger = logging.getLogger(__name__)

OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
CACHE_KEY = "airports:all"
CACHE_TTL = int(timedelta(hours=24).total_seconds())

# ============================================================================
# TRANSLITERATION MAPS
# ============================================================================

# Russian/Cyrillic to Latin
CYRILLIC_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e',
    'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'kh', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'shch',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'E',
    'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    
    # Uzbek specific
    'ў': 'o', 'қ': 'q', 'ғ': 'g', 'ҳ': 'h',
    'Ў': 'O', 'Қ': 'Q', 'Ғ': 'G', 'Ҳ': 'H',
}

# Common city name translations (Russian → English)
CITY_TRANSLATIONS = {
    # Russian cities
    'москва': 'moscow',
    'санкт-петербург': 'saint petersburg',
    'петербург': 'petersburg',
    'новосибирск': 'novosibirsk',
    'екатеринбург': 'yekaterinburg',
    'казань': 'kazan',
    'нижний новгород': 'nizhny novgorod',
    'сочи': 'sochi',
    'владивосток': 'vladivostok',
    
    # Central Asia
    'ташкент': 'tashkent',
    'алматы': 'almaty',
    'астана': 'astana',
    'нур-султан': 'nur-sultan',
    'бишкек': 'bishkek',
    'душанбе': 'dushanbe',
    'ашхабад': 'ashgabat',
    'самарканд': 'samarkand',
    'бухара': 'bukhara',
    
    # Europe
    'лондон': 'london',
    'париж': 'paris',
    'берлин': 'berlin',
    'рим': 'rome',
    'мадрид': 'madrid',
    'барселона': 'barcelona',
    'амстердам': 'amsterdam',
    'вена': 'vienna',
    'прага': 'prague',
    'варшава': 'warsaw',
    'будапешт': 'budapest',
    'стамбул': 'istanbul',
    'афины': 'athens',
    'копенгаген': 'copenhagen',
    'стокгольм': 'stockholm',
    'хельсинки': 'helsinki',
    'осло': 'oslo',
    
    # Asia
    'токио': 'tokyo',
    'пекин': 'beijing',
    'шанхай': 'shanghai',
    'сеул': 'seoul',
    'бангкок': 'bangkok',
    'сингапур': 'singapore',
    'дубай': 'dubai',
    'дели': 'delhi',
    'мумбаи': 'mumbai',
    
    # Americas
    'нью-йорк': 'new york',
    'лос-анджелес': 'los angeles',
    'чикаго': 'chicago',
    'майами': 'miami',
    'торонто': 'toronto',
    'ванкувер': 'vancouver',
    
    # Others
    'каир': 'cairo',
    'йоханнесбург': 'johannesburg',
    'сидней': 'sydney',
    'мельбурн': 'melbourne',
}

# IATA code mappings for common Cyrillic inputs
CYRILLIC_IATA_MAP = {
    # Russian
    'мск': 'MOW', 'москва': 'MOW',
    'спб': 'LED', 'санкт-петербург': 'LED',
    'сво': 'SVO', 'шереметьево': 'SVO',
    'дме': 'DME', 'домодедово': 'DME',
    'внк': 'VKO', 'внуково': 'VKO',
    
    # Central Asia
    'таш': 'TAS', 'ташкент': 'TAS',
    'алм': 'ALA', 'алматы': 'ALA',
    'нур': 'NQZ', 'астана': 'NQZ',
    'биш': 'FRU', 'бишкек': 'FRU',
    'душ': 'DYU', 'душанбе': 'DYU',
    'сам': 'SKD', 'самарканд': 'SKD',
    'бух': 'BHK', 'бухара': 'BHK',
    
    # Europe
    'лон': 'LON', 'лондон': 'LON',
    'пар': 'PAR', 'париж': 'PAR',
    'бер': 'BER', 'берлин': 'BER',
    'рим': 'ROM', 'рома': 'ROM',
    'мад': 'MAD', 'мадрид': 'MAD',
    'бар': 'BCN', 'барселона': 'BCN',
    'ams': 'AMS', 'амстердам': 'AMS',
    'вен': 'VIE', 'вена': 'VIE',
    'пра': 'PRG', 'прага': 'PRG',
    'ист': 'IST', 'стамбул': 'IST',
}


class AirportService:
    """
    Enhanced airport autocomplete service with multi-language support.
    
    Supports:
    - Latin input (English)
    - Cyrillic input (Russian, Uzbek, etc.)
    - Mixed input
    - Transliteration
    """
    
    @staticmethod
    def transliterate_cyrillic(text: str) -> str:
        """
        Convert Cyrillic text to Latin.
        
        Examples:
        - "Ташкент" → "Tashkent"
        - "Москва" → "Moskva"
        - "Лондон" → "London"
        """
        result = []
        i = 0
        while i < len(text):
            # Check for two-character combinations
            if i < len(text) - 1:
                two_char = text[i:i+2]
                if two_char.lower() in ['ж', 'х', 'ц', 'ч', 'ш', 'щ', 'ю', 'я']:
                    result.append(CYRILLIC_TO_LATIN.get(text[i], text[i]))
                    i += 1
                    continue
            
            # Single character
            char = text[i]
            result.append(CYRILLIC_TO_LATIN.get(char, char))
            i += 1
        
        return ''.join(result)
    
    @staticmethod
    def normalize_query(query: str) -> List[str]:
        """
        Generate multiple normalized versions of query for better matching.
        
        Returns list of normalized strings to search for.
        
        Examples:
        - "Ташкент" → ["ташкент", "tashkent", "TAS"]
        - "Лондон" → ["лондон", "london", "LON"]
        - "москва" → ["москва", "moscow", "MOW"]
        """
        normalized = []
        query_lower = query.lower().strip()
        
        # Original query
        normalized.append(query_lower)
        
        # Check if it's a direct IATA code mapping (Cyrillic)
        if query_lower in CYRILLIC_IATA_MAP:
            iata = CYRILLIC_IATA_MAP[query_lower]
            normalized.append(iata.lower())
        
        # Check if it's a known city translation
        if query_lower in CITY_TRANSLATIONS:
            english_city = CITY_TRANSLATIONS[query_lower]
            normalized.append(english_city)
        
        # Transliterate if contains Cyrillic
        if any(char in CYRILLIC_TO_LATIN for char in query):
            transliterated = AirportService.transliterate_cyrillic(query)
            normalized.append(transliterated.lower())
            
            # Check if transliterated version is in city translations
            if transliterated.lower() in CITY_TRANSLATIONS:
                normalized.append(CITY_TRANSLATIONS[transliterated.lower()])
        
        # Remove duplicates while preserving order
        seen = set()
        result = []
        for item in normalized:
            if item not in seen:
                seen.add(item)
                result.append(item)
        
        logger.debug(f"Query '{query}' normalized to: {result}")
        return result
    
    @staticmethod
    async def _fetch_airports_from_source() -> List[dict]:
        """Fetch and parse airport data from OpenFlights."""
        logger.info(f"Fetching airports from {OPENFLIGHTS_URL}")
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(OPENFLIGHTS_URL)
            response.raise_for_status()
            
            airports = []
            csv_reader = csv.reader(response.text.strip().split('\n'))
            
            for row in csv_reader:
                if len(row) < 14:
                    continue
                
                iata_code = row[4].strip()
                icao_code = row[5].strip()
                name = row[1].strip()
                city = row[2].strip()
                country = row[3].strip()
                
                if not iata_code or iata_code == '\\N' or len(iata_code) != 3:
                    continue
                
                try:
                    latitude = float(row[6]) if row[6] != '\\N' else None
                    longitude = float(row[7]) if row[7] != '\\N' else None
                except (ValueError, IndexError):
                    latitude = None
                    longitude = None
                
                timezone = row[11].strip() if len(row) > 11 and row[11] != '\\N' else 'UTC'
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
            'Uzbekistan': 'UZ', 'Kazakhstan': 'KZ', 'Kyrgyzstan': 'KG',
            'Tajikistan': 'TJ', 'Turkmenistan': 'TM',
        }
        return mapping.get(country_name, 'XX')
    
    @staticmethod
    async def _get_all_airports() -> List[dict]:
        """Get all airports from cache or fetch from source."""
        try:
            cached_bytes = await redis_client.get(CACHE_KEY)
            if cached_bytes:
                logger.debug("Airport data retrieved from cache")
                return json.loads(cached_bytes.decode('utf-8'))
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
        
        logger.info("Cache miss - fetching airports from source")
        airports = await AirportService._fetch_airports_from_source()
        
        try:
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
        Search airports with multi-language support.
        
        Examples:
        - "TAS" → Tashkent
        - "Ташкент" → Tashkent
        - "таш" → Tashkent
        - "Лондон" → London airports (LHR, LGW, etc.)
        - "лон" → London airports
        - "moscow" → Moscow airports
        - "москва" → Moscow airports
        """
        from schemas.airports import AirportResponse
        
        if len(query) < 2:
            return []
        
        all_airports = await AirportService._get_all_airports()
        
        # Generate normalized query variations
        query_variants = AirportService.normalize_query(query)
        query_upper = query.upper()
        
        # Search and rank results
        results = []
        for airport in all_airports:
            score = 0
            
            # Check against each query variant
            for variant in query_variants:
                variant_lower = variant.lower()
                variant_upper = variant.upper()
                
                # IATA code exact match (highest priority)
                if airport['iata_code'] == variant_upper:
                    score = max(score, 100)
                # IATA starts with variant
                elif airport['iata_code'].startswith(variant_upper):
                    score = max(score, 90)
                # City exact match
                elif airport['city'].lower() == variant_lower:
                    score = max(score, 85)
                # City starts with variant
                elif airport['city'].lower().startswith(variant_lower):
                    score = max(score, 75)
                # Airport name starts with variant
                elif airport['name'].lower().startswith(variant_lower):
                    score = max(score, 65)
                # City contains variant
                elif variant_lower in airport['city'].lower():
                    score = max(score, 55)
                # Airport name contains variant
                elif variant_lower in airport['name'].lower():
                    score = max(score, 45)
                # Country starts with variant
                elif airport['country'].lower().startswith(variant_lower):
                    score = max(score, 35)
                # Country contains variant
                elif variant_lower in airport['country'].lower():
                    score = max(score, 25)
            
            # Also check original query directly (in case transliteration missed something)
            query_lower = query.lower()
            if query_lower in airport['city'].lower():
                score = max(score, 50)
            if query_lower in airport['name'].lower():
                score = max(score, 40)
            
            if score > 0:
                results.append({
                    'airport': airport,
                    'score': score
                })
        
        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:limit]
        
        logger.debug(f"Query '{query}' returned {len(top_results)} results")
        
        return [
            AirportResponse(**item['airport'])
            for item in top_results
        ]
    
    @staticmethod
    async def get_airport_by_iata(iata_code: str) -> Optional['AirportResponse']:
        """Get specific airport by IATA code."""
        from schemas.airports import AirportResponse
        
        all_airports = await AirportService._get_all_airports()
        
        # Check if it's a Cyrillic input that maps to IATA
        iata_lower = iata_code.lower()
        if iata_lower in CYRILLIC_IATA_MAP:
            iata_code = CYRILLIC_IATA_MAP[iata_lower]
        
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