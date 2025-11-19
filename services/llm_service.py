# services/llm_service.py

import asyncio
import json
import logging
import re
from typing import Any, Dict

from openai import OpenAI
from app.core.config import settings

logger = logging.getLogger(__name__)


class OpenAILLMService:
    """
    Wrapper for OpenAI Responses API with guaranteed JSON output.
    """

    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

        # Safe default model (everyone has access)
        self.model = getattr(settings, "OPENAI_TEXT_MODEL", "gpt-4o")

    async def _retry(self, func, attempts: int = 3, delay: float = 0.5):
        """Retry wrapper for async calls."""
        for attempt in range(attempts):
            try:
                return await func()
            except Exception as e:
                if attempt == attempts - 1:
                    logger.error(f"[OpenAI-LLM] Failed after retries: {e}", exc_info=True)
                    raise

                sleep = delay * (2 ** attempt)
                logger.warning(f"[OpenAI-LLM] Retry {attempt+1}/{attempts}, waiting {sleep:.1f}s")
                await asyncio.sleep(sleep)

    async def _call_openai(self, messages) -> str:
        """
        Use Responses API (new OpenAI format). JSON enforced via instructions, not response_format.
        """

        # Responses API is synchronous -> run in thread
        def sync_call():
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                max_output_tokens=500
            )
            return response.output_text  # always plain text

        return await asyncio.to_thread(sync_call)

    def _clean_json_response(self, raw_text: str) -> str:
        """Clean markdown code fences etc."""
        raw_text = raw_text.strip()

        # ```json ... ``` removal
        if raw_text.startswith("```"):
            lines = raw_text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            raw_text = "\n".join(lines).strip()

        # Extract {...} or [...]
        match = re.search(r"(\{.*\}|\[.*\])", raw_text, re.DOTALL)
        if match:
            return match.group(1)

        return raw_text

    async def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Generate JSON strictly. Responses API → output_text.
        """

        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt.strip()
                    + "\n\nSTRICT INSTRUCTION: Respond ONLY with valid JSON. No text outside JSON."
                )
            },
            {"role": "user", "content": user_prompt.strip()},
        ]

        raw_text = await self._retry(lambda: self._call_openai(messages))

        cleaned = self._clean_json_response(raw_text)

        try:
            return json.loads(cleaned)
        except Exception as e:
            logger.error(
                f"[OpenAI-LLM] JSON decode failed: {e}\nRaw: {cleaned[:300]}"
            )
            return {}


# Singleton instance
llm_service = OpenAILLMService()


async def generate_json_response(
    system_prompt: str,
    user_prompt: str,
) -> Dict[str, Any]:
    return await llm_service.generate_json(system_prompt, user_prompt)
