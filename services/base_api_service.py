#services/base_api_service.py



from __future__ import annotations
import asyncio
import logging
from typing import Any, Dict, Optional


import httpx




class BaseAPIService:
    DEFAULT_TIMEOUT_S = 15.0
    RETRY_STATUSES = {429, 500, 502, 503, 504}


    def __init__(self, base_url: str, headers: Optional[Dict[str, str]] = None, timeout_s: float = None):
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {}
        self.timeout_s = timeout_s or self.DEFAULT_TIMEOUT_S
        self._client: Optional[httpx.AsyncClient] = None
        self.logger = logging.getLogger(self.__class__.__name__)


    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_s, headers=self.headers)
        return self._client


    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        retries: int = 3,
        backoff_base: float = 0.5,
        ) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        json = json or {}


        for attempt in range(retries):
            try:
                client = await self._get_client()
                resp = await client.request(method, url, params=params, json=json)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status in self.RETRY_STATUSES and attempt < retries - 1:
                    await asyncio.sleep((2 ** attempt) * backoff_base)
                    continue
                self.logger.error("HTTP error %s on %s %s: %s", status, method, url, e)
                raise
            except (httpx.TimeoutException, httpx.TransportError) as e:
                if attempt < retries - 1:
                    await asyncio.sleep((2 ** attempt) * backoff_base)
                    continue
                self.logger.error("Transport error on %s %s: %s", method, url, e)
                raise


    async def _get(self, endpoint: str, *, params: Optional[Dict[str, Any]] = None, **kw) -> Dict[str, Any]:
        return await self._request("GET", endpoint, params=params, **kw)