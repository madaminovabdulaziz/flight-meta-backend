# services/llm_service.py

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from google.genai import Client
from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiLLMService:
    """
    Thin wrapper around Gemini for JSON-style responses.
    Uses the same Client as your embedding service, but configured for text.
    """

    def __init__(self) -> None:
        self.client = Client(api_key=settings.GEMINI_API_KEY)
        # You can expose this via settings if you like
        self.model = getattr(settings, "GEMINI_TEXT_MODEL", "gemini-2.5-flash")

    async def _retry(self, func, attempts: int = 3, delay: float = 0.5):
        for i in range(attempts):
            try:
                return await func()
            except Exception as e:
                if i == attempts - 1:
                    logger.error(f"[GeminiLLM] Failed after retries: {e}", exc_info=True)
                    raise
                wait = delay * (2 ** i)
                logger.warning(f"[GeminiLLM] Retry {i+1}/{attempts}, waiting {wait:.1f}s")
                await asyncio.sleep(wait)

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Call Gemini with instructions to return strict JSON.
        We still defensively parse to handle minor deviations.
        """

        full_prompt = (
            system_prompt.strip()
            + "\n\n"
            + "USER MESSAGE:\n"
            + user_prompt.strip()
        )

        def sync_call():
            # Using google-genai client
            resp = self.client.models.generate_content(
                model=self.model,
                contents=[full_prompt],
            )
            # Newer google-genai has .text shortcut
            return resp.text

        raw_text = await self._retry(lambda: asyncio.to_thread(sync_call))

        # Try to extract JSON
        raw_text = raw_text.strip()
        # Sometimes models wrap JSON in ```json ... ```
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.lower().startswith("json"):
                raw_text = raw_text[4:].strip()

        try:
            return json.loads(raw_text)
        except Exception as e:
            logger.error(f"[GeminiLLM] Failed to parse JSON: {e}. Raw: {raw_text}", exc_info=True)
            # Fail safe: return empty dict so callers can fall back
            return {}


llm_service = GeminiLLMService()


async def generate_json_response(
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    """
    Convenience helper for nodes.
    """
    return await llm_service.generate_json(system_prompt, user_prompt)
