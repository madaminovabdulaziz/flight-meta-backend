# scripts/seed_airports.py
"""
Seed the airports database with data from OurAirports.
OurAirports is a free, community-maintained airport database.

Data source: https://ourairports.com/data/
License: Public Domain (free to use)

Usage:
    python scripts/seed_airports.py
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import csv
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.database import AsyncSessionLocal, engine, Base
from app.models.models import Airport
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OurAirports CSV download URLs (updated regularly)
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
COUNTRIES_URL = "https://davidmegginson.github.io/ourairports-data/countries.csv"

# Alternative: OpenFlights
# OPENFLIGHTS_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"


async def download_csv(url: str) -> str:
    """Download CSV file from URL."""
    logger.info(f"Downloading data from {url}...")
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.text


async def load_countries() -> dict:
    """Load country data to get full country names."""
    csv_data = await download_csv(COUNTRIES_URL)
    countries = {}
    
    reader = csv.DictReader(csv_data.splitlines())
    for row in reader:
        countries[row['code']] = {
            'name': row['name'],
            'iso_code': row['code']
        }
    
    logger.info(f"Loaded {len(countries)} countries")
    return countries


async def seed_airports():
    """Seed airports table with OurAirports data."""
    
    # Create tables if they don't exist
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Load country data
    countries = await load_countries()
    
    # Download airport data
    csv_data = await download_csv(AIRPORTS_URL)
    
    # Parse CSV
    reader = csv.DictReader(csv_data.splitlines())
    
    # Use AsyncSessionLocal to create a proper session
    async with AsyncSessionLocal() as session:
        total_count = 0
        added_count = 0
        skipped_count = 0
        updated_count = 0
        
        for row in reader:
            total_count += 1
            
            # Filter: Only include airports with IATA codes
            iata_code = row.get('iata_code', '').strip()
            if not iata_code or len(iata_code) != 3:
                skipped_count += 1
                continue
            
            # Filter: Only medium and large airports (commercial)
            airport_type = row.get('type', '')
            if airport_type not in ['large_airport', 'medium_airport']:
                skipped_count += 1
                continue
            
            # Get country info
            country_code = row.get('iso_country', '').strip()
            country_name = countries.get(country_code, {}).get('name', country_code)
            
            # Determine if airport is active (closed airports marked as 'closed')
            is_active = row.get('scheduled_service', 'yes') == 'yes'
            
            # Check if airport already exists
            result = await session.execute(
                select(Airport).where(Airport.iata_code == iata_code)
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                # Update existing airport
                existing.icao_code = row.get('ident', '')[:4] or None
                existing.name = row.get('name', '').strip()
                existing.city = row.get('municipality', '').strip() or row.get('name', '').split()[0]
                existing.country = country_name
                existing.country_code = country_code
                existing.timezone = row.get('timezone', 'UTC').strip() or 'UTC'
                existing.latitude = float(row['latitude_deg']) if row.get('latitude_deg') else None
                existing.longitude = float(row['longitude_deg']) if row.get('longitude_deg') else None
                existing.is_active = is_active
                updated_count += 1
            else:
                # Create new airport
                airport = Airport(
                    iata_code=iata_code,
                    icao_code=row.get('ident', '')[:4] or None,
                    name=row.get('name', '').strip(),
                    city=row.get('municipality', '').strip() or row.get('name', '').split()[0],
                    country=country_name,
                    country_code=country_code,
                    timezone=row.get('timezone', 'UTC').strip() or 'UTC',
                    latitude=float(row['latitude_deg']) if row.get('latitude_deg') else None,
                    longitude=float(row['longitude_deg']) if row.get('longitude_deg') else None,
                    is_active=is_active
                )
                session.add(airport)
                added_count += 1
            
            # Commit in batches of 100
            if (added_count + updated_count) % 100 == 0:
                await session.commit()
                logger.info(f"Progress: {added_count} added, {updated_count} updated, {skipped_count} skipped")
        
        # Final commit
        await session.commit()
        
        logger.info("=" * 60)
        logger.info(f"✅ Airport seeding completed!")
        logger.info(f"   Total rows processed: {total_count}")
        logger.info(f"   ✅ Added: {added_count}")
        logger.info(f"   🔄 Updated: {updated_count}")
        logger.info(f"   ⏭️  Skipped: {skipped_count}")
        logger.info("=" * 60)


async def verify_seeded_data():
    """Verify that data was seeded correctly."""
    async with AsyncSessionLocal() as session:
        # Count total airports
        result = await session.execute(select(Airport))
        airports = result.scalars().all()
        
        logger.info(f"\n📊 Verification Results:")
        logger.info(f"   Total airports in database: {len(airports)}")
        
        # Sample some airports
        sample_codes = ['JFK', 'LHR', 'IST', 'DXB', 'SIN', 'TAS', 'CDG', 'NRT']
        logger.info(f"\n🔍 Sample airports:")
        
        for code in sample_codes:
            result = await session.execute(
                select(Airport).where(Airport.iata_code == code)
            )
            airport = result.scalar_one_or_none()
            if airport:
                logger.info(f"   ✅ {code}: {airport.name}, {airport.city}, {airport.country}")
            else:
                logger.info(f"   ❌ {code}: Not found")
        
        # Count by country
        result = await session.execute(
            select(Airport.country_code).distinct()
        )
        countries = result.scalars().all()
        logger.info(f"\n🌍 Countries represented: {len(countries)}")


if __name__ == "__main__":
    print("🚀 Starting airport database seeding...")
    print("📥 Data source: OurAirports (https://ourairports.com/data/)")
    print("📜 License: Public Domain\n")
    
    asyncio.run(seed_airports())
    asyncio.run(verify_seeded_data())
    
    print("\n✅ Done! Your airport database is ready to use.")


# Alternative: OpenFlights format parser
"""
If you prefer OpenFlights data format:

async def seed_from_openflights():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    
    # OpenFlights format (no header):
    # 0: Airport ID, 1: Name, 2: City, 3: Country, 4: IATA, 5: ICAO,
    # 6: Latitude, 7: Longitude, 8: Altitude, 9: Timezone offset,
    # 10: DST, 11: Tz database timezone, 12: Type, 13: Source
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        lines = response.text.split('\n')
    
    async with async_session() as session:
        for line in lines:
            if not line.strip():
                continue
            
            parts = line.split(',')
            iata = parts[4].strip('"')
            
            if not iata or iata == '\\N' or len(iata) != 3:
                continue
            
            airport = Airport(
                iata_code=iata,
                icao_code=parts[5].strip('"') if parts[5] != '\\N' else None,
                name=parts[1].strip('"'),
                city=parts[2].strip('"'),
                country=parts[3].strip('"'),
                country_code='',  # Need to map country name to code
                timezone=parts[11].strip('"') if len(parts) > 11 else 'UTC',
                latitude=float(parts[6]) if parts[6] != '\\N' else None,
                longitude=float(parts[7]) if parts[7] != '\\N' else None,
                is_active=True
            )
            session.add(airport)
        
        await session.commit()
"""