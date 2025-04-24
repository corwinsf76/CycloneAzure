import os
import logging
from typing import Dict, Any, Optional
import aiohttp
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

import config
from database.db_utils import async_bulk_insert

# Setup logging
log = logging.getLogger(__name__)

# Load API key
API_KEY = os.getenv("SANTIMENT_API_KEY")
BASE_URL = "https://api.santiment.net/graphql"

if not API_KEY:
    log.error("Santiment API key is not set. Please add it to your .env file.")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_social_volume(slug: str, from_date: str, to_date: str, interval: str = "1d") -> Optional[Dict[str, Any]]:
    """Fetch social volume data from Santiment API."""
    query = {
        "query": f"""
        {{
            socialVolume(
                slug: "{slug}",
                from: "{from_date}",
                to: "{to_date}",
                interval: "{interval}"
            ) {{
                datetime
                value
            }}
        }}
        """
    }
    headers = {"Authorization": f"Apikey {API_KEY}"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(BASE_URL, json=query, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    social_volume = data.get('data', {}).get('socialVolume', [])
                    
                    # Process and store data
                    if social_volume:
                        processed_data = [{
                            'slug': slug,
                            'datetime': item['datetime'],
                            'value': item['value']
                        } for item in social_volume]
                        
                        await async_bulk_insert(processed_data, 'santiment_social_volume')
                        return {'slug': slug, 'data': social_volume}
                    
                return None
                
    except Exception as e:
        log.error(f"Failed to fetch social volume data for {slug}: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def fetch_active_addresses(slug: str, from_date: str, to_date: str, interval: str = "1d") -> Optional[Dict[str, Any]]:
    """Fetch active addresses data from Santiment API."""
    query = {
        "query": f"""
        {{
            dailyActiveAddresses(
                slug: "{slug}",
                from: "{from_date}",
                to: "{to_date}",
                interval: "{interval}"
            ) {{
                datetime
                activeAddresses
            }}
        }}
        """
    }
    headers = {"Authorization": f"Apikey {API_KEY}"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(BASE_URL, json=query, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    active_addresses = data.get('data', {}).get('dailyActiveAddresses', [])
                    
                    # Process and store data
                    if active_addresses:
                        processed_data = [{
                            'slug': slug,
                            'datetime': item['datetime'],
                            'active_addresses': item['activeAddresses']
                        } for item in active_addresses]
                        
                        await async_bulk_insert(processed_data, 'santiment_active_addresses')
                        return {'slug': slug, 'data': active_addresses}
                    
                return None
                
    except Exception as e:
        log.error(f"Failed to fetch active addresses data for {slug}: {e}")
        return None