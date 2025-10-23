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
# AIRPORT POPULARITY RANKINGS
# ============================================================================

# Major international hubs (popularity score: 100)
MAJOR_HUBS = {
    'LHR', 'JFK', 'LAX', 'ORD', 'DXB', 'CDG', 'AMS', 'FRA', 'IST', 'SIN',
    'HKG', 'ICN', 'NRT', 'SYD', 'YYZ', 'DEL', 'BKK', 'KUL', 'DFW', 'SFO',
    'ATL', 'PEK', 'PVG', 'CAN', 'MAD', 'BCN', 'FCO', 'MUC', 'ZRH', 'VIE',
    'LGW', 'MAN', 'BER', 'ARN', 'CPH', 'OSL', 'HEL', 'WAW', 'PRG', 'BUD',
    'ATH', 'LIS', 'DUB', 'EDI', 'GLA', 'BRU', 'AMS', 'GVA', 'NCE', 'MXP',
    'MEX', 'GRU', 'EZE', 'SCL', 'BOG', 'LIM', 'GIG', 'PTY', 'UIO', 'MVD'
}

# Regional hubs and large airports (popularity score: 80)
REGIONAL_HUBS = {
    'SVO', 'DME', 'LED', 'TAS', 'ALA', 'NQZ', 'FRU', 'SKD', 'BHK', 'DYU',
    'TBS', 'EVN', 'GYD', 'BAK', 'MCX', 'OVB', 'KZN', 'VVO', 'ROV', 'UFA',
    'KRR', 'AER', 'VKO', 'SVX', 'KJA', 'IKT', 'OMS', 'TJM', 'UUS', 'PKC',
    'MIA', 'IAH', 'SEA', 'LAS', 'PHX', 'DEN', 'BOS', 'MCO', 'EWR', 'CLT',
    'MSP', 'DTW', 'PHL', 'LGA', 'BWI', 'IAD', 'DCA', 'SAN', 'PDX', 'AUS',
    'YVR', 'YUL', 'YYC', 'YEG', 'YOW', 'YHZ', 'YWG', 'YYJ', 'YXE', 'YQR',
    'MEL', 'BNE', 'PER', 'ADL', 'AKL', 'CHC', 'WLG', 'CBR', 'OOL', 'CNS',
    'BOM', 'BLR', 'HYD', 'CCU', 'MAA', 'AMD', 'COK', 'GOI', 'PNQ', 'JAI',
    'CMB', 'DAC', 'KTM', 'RGN', 'HAN', 'SGN', 'MNL', 'CGK', 'SUB', 'DPS',
    'KUL', 'PEN', 'JHB', 'BKI', 'KCH', 'MFM', 'CTS', 'FUK', 'KIX', 'NGO',
    'HND', 'OKA', 'TPE', 'KHH', 'ICN', 'GMP', 'CJU', 'PUS', 'TAO', 'XIY',
    'CTU', 'CKG', 'WUH', 'SZX', 'XMN', 'CSX', 'KWL', 'NKG', 'HGH', 'SIA',
    'JNB', 'CPT', 'DUR', 'CAI', 'HRG', 'SSH', 'TLV', 'AMM', 'DOH', 'AUH',
    'MCT', 'KWI', 'BAH', 'RUH', 'JED', 'DMM', 'ADD', 'NBO', 'DAR', 'EBB',
    'ACC', 'LOS', 'ABV', 'DKR', 'ABJ', 'CMN', 'RAK', 'TUN', 'ALG', 'TIP'
}

# Capital city airports (popularity score: 60)
CAPITAL_AIRPORTS = {
    'MOW', 'ROM', 'PAR', 'BER', 'MAD', 'LIS', 'DUB', 'BRU', 'AMS', 'VIE',
    'BUD', 'WAW', 'PRG', 'BEG', 'ZAG', 'LJU', 'SKP', 'TIA', 'SOF', 'OTP',
    'KIV', 'RIX', 'TLL', 'VNO', 'HEL', 'CPH', 'OSL', 'STO', 'REK', 'ATH'
}

# City codes that represent multiple airports (score boost)
MULTI_AIRPORT_CITIES = {
    'LON': ['LHR', 'LGW', 'STN', 'LTN', 'LCY', 'SEN'],  # London
    'NYC': ['JFK', 'LGA', 'EWR'],  # New York
    'PAR': ['CDG', 'ORY', 'BVA'],  # Paris
    'BER': ['BER', 'TXL', 'SXF'],  # Berlin
    'MOW': ['SVO', 'DME', 'VKO'],  # Moscow
    'TYO': ['NRT', 'HND'],  # Tokyo
    'CHI': ['ORD', 'MDW'],  # Chicago
    'MIL': ['MXP', 'LIN', 'BGY'],  # Milan
    'ROM': ['FCO', 'CIA'],  # Rome
    'LON': ['LHR', 'LGW', 'STN', 'LTN', 'LCY'],  # London
    'SAO': ['GRU', 'CGH', 'VCP'],  # São Paulo
    'RIO': ['GIG', 'SDU'],  # Rio
    'BA': ['EZE', 'AEP'],  # Buenos Aires
    'OSA': ['KIX', 'ITM'],  # Osaka
    'BJS': ['PEK', 'PKX'],  # Beijing
    'SHA': ['PVG', 'SHA'],  # Shanghai
    'SEL': ['ICN', 'GMP'],  # Seoul
    'STO': ['ARN', 'BMA', 'NYO', 'VST'],  # Stockholm
}

# Popular tourist destinations (score: 70)
TOURIST_AIRPORTS = {
    'DXB', 'BKK', 'SIN', 'HKG', 'BCN', 'ROM', 'VCE', 'FCO', 'ATH', 'IST',
    'DPS', 'HKT', 'CNX', 'USM', 'KBV', 'HUI', 'CEB', 'MLE', 'HER', 'RHO',
    'CFU', 'PMI', 'IBZ', 'AGP', 'ALC', 'FAO', 'FNC', 'TFS', 'LPA', 'ACE',
    'SSH', 'HRG', 'MBJ', 'CUN', 'PUJ', 'SJO', 'PTY', 'LIR', 'GDL', 'PVR',
    'SJD', 'CZM', 'ZIH', 'HMO', 'TIJ', 'REP', 'HAN', 'SGN', 'DAD', 'NHA',
    'CNS', 'OOL', 'BNE', 'SYD', 'MEL', 'AKL', 'CHC', 'ZQN', 'NAN', 'PPT'
}

# ============================================================================
# CLOSED/INACTIVE AIRPORTS BLACKLIST
# ============================================================================

# Permanently closed airports that should NOT appear in search results
CLOSED_AIRPORTS = {
    'ISL',  # Istanbul Atatürk (closed April 2019, replaced by IST) ⭐
    'THF',  # Berlin Tempelhof (closed 2008)
    'TXL',  # Berlin Tegel (closed November 2020, replaced by BER)
    'SXF',  # Berlin Schönefeld (closed October 2020, merged into BER)
    'HKG_KAI',  # Hong Kong Kai Tak (closed 1998)
    'MHG',  # Mannheim City Airport (closed 1995)
    'OSL_FBU',  # Oslo Fornebu (closed 1998, replaced by OSL Gardermoen)
    'MUC_RIE',  # Munich-Riem (closed 1992, replaced by MUC)
    'DEN_STA',  # Denver Stapleton (closed 1995, replaced by DEN)
    'ATH_HEL',  # Athens Hellenikon (closed 2001, replaced by ATH)
}

# Note: OpenFlights may still have these. We filter them out during search.

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
                
                # Filter out closed airports
                if iata_code.upper() in CLOSED_AIRPORTS:
                    logger.debug(f"Skipping closed airport: {iata_code} - {name}")
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
    def get_airport_popularity_score(iata_code: str) -> int:
        """
        Get popularity score for an airport.
        
        Returns:
            100 - Major international hub
            80  - Regional hub / large airport
            70  - Tourist destination
            60  - Capital city airport
            40  - Medium airport
            20  - Small airport
        """
        if iata_code in MAJOR_HUBS:
            return 100
        elif iata_code in REGIONAL_HUBS:
            return 80
        elif iata_code in TOURIST_AIRPORTS:
            return 70
        elif iata_code in CAPITAL_AIRPORTS:
            return 60
        else:
            # Default scores based on IATA code patterns
            # (Not perfect, but reasonable heuristic)
            return 40  # Medium airport by default
    
    @staticmethod
    async def search_airports(
        query: str,
        limit: int = 10,
        user_country_code: Optional[str] = None
    ) -> List['AirportResponse']:
        """
        Search airports with multi-language support, popularity ranking, and location awareness.
        
        Ranking factors:
        1. Match quality (exact IATA, city prefix, etc.)
        2. Airport popularity (major hubs ranked higher)
        3. User's country (airports in user's country get bonus) ⭐ NEW
        4. Passenger traffic (implicit via popularity lists)
        
        Examples:
        - "L" → LHR, LGW, LAX, LIS (major airports starting with L)
        - "Lo" → London airports (LHR, LGW, STN...), then Los Angeles
        - "Lon" → All London airports first
        - "TAS" → Tashkent
        - "Ташкент" → Tashkent
        - "таш" → Tashkent
        - "Лондон" → London airports (LHR, LGW, etc.)
        
        Location-aware:
        - User in UK + "L" → UK airports (LHR, LGW) ranked higher
        - User in US + "L" → US airports (LAX, LAS) ranked higher
        """
        from schemas.airports import AirportResponse
        
        if len(query) < 1:
            return []
        
        all_airports = await AirportService._get_all_airports()
        
        # Generate normalized query variations
        query_variants = AirportService.normalize_query(query)
        query_upper = query.upper()
        
        logger.debug(f"Search: query='{query}', user_country={user_country_code}, variants={query_variants}")
        
        # Search and rank results
        results = []
        for airport in all_airports:
            score = 0
            
            # Get popularity bonus
            popularity_score = AirportService.get_airport_popularity_score(
                airport['iata_code']
            )
            
            # ⭐ NEW: Location-based bonus
            location_bonus = 0
            if user_country_code and airport.get('country_code') == user_country_code:
                # Boost airports in user's country
                location_bonus = 150
                logger.debug(f"Location bonus applied for {airport['iata_code']} (user country: {user_country_code})")
            
            # Check against each query variant
            for variant in query_variants:
                variant_lower = variant.lower()
                variant_upper = variant.upper()
                
                # IATA code exact match (highest priority)
                if airport['iata_code'] == variant_upper:
                    score = max(score, 1000 + popularity_score + location_bonus)
                # IATA starts with variant (very high priority for short queries)
                elif airport['iata_code'].startswith(variant_upper):
                    score = max(score, 900 + popularity_score + location_bonus)
                # City exact match
                elif airport['city'].lower() == variant_lower:
                    score = max(score, 850 + popularity_score + location_bonus)
                # City starts with variant (important for partial matches)
                elif airport['city'].lower().startswith(variant_lower):
                    # Boost score more for longer matches
                    match_quality = len(variant_lower) / len(airport['city'])
                    score = max(score, 750 + popularity_score + location_bonus + int(match_quality * 50))
                # Airport name starts with variant
                elif airport['name'].lower().startswith(variant_lower):
                    score = max(score, 650 + popularity_score + location_bonus)
                # City contains variant
                elif variant_lower in airport['city'].lower():
                    score = max(score, 550 + popularity_score + location_bonus)
                # Airport name contains variant
                elif variant_lower in airport['name'].lower():
                    score = max(score, 450 + popularity_score + location_bonus)
                # Country starts with variant
                elif airport['country'].lower().startswith(variant_lower):
                    score = max(score, 350 + popularity_score + location_bonus)
                # Country contains variant
                elif variant_lower in airport['country'].lower():
                    score = max(score, 250 + popularity_score + location_bonus)
            
            # Also check original query directly
            query_lower = query.lower()
            if query_lower in airport['city'].lower():
                score = max(score, 500 + popularity_score + location_bonus)
            if query_lower in airport['name'].lower():
                score = max(score, 400 + popularity_score + location_bonus)
            
            # Special handling for single-character queries
            # Only return major airports to avoid overwhelming results
            if len(query) == 1 and score > 0:
                if popularity_score < 60:  # Skip small airports for single char
                    continue
            
            if score > 0:
                results.append({
                    'airport': airport,
                    'score': score,
                    'popularity': popularity_score,
                    'in_user_country': location_bonus > 0
                })
        
        # Sort by score (descending), then by popularity
        results.sort(key=lambda x: (x['score'], x['popularity']), reverse=True)
        top_results = results[:limit]
        
        logger.debug(
            f"Query '{query}' returned {len(top_results)} results "
            f"(filtered from {len(results)}, user_country={user_country_code})"
        )
        
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