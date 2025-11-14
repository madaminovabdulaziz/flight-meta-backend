# app/conversation/litellm_adapter.py
"""
LiteLLM Adapter for Multi-Provider Support
Provides unified interface to OpenAI, Anthropic, Google, and more.

Benefits:
- Automatic failover between providers
- Unified API for all LLM providers
- Built-in retry logic and rate limiting
- Cost tracking across providers
- Easy to swap providers without code changes

Supported Providers:
- OpenAI (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
- Anthropic (Claude Sonnet, Claude Opus)
- Google (Gemini Pro, Gemini Flash)
- Groq (Llama-3, Mixtral) - Future
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# LITELLM ADAPTER (LIGHTWEIGHT IMPLEMENTATION)
# ============================================================

async def call_litellm(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    **kwargs
) -> str:
    """
    Call LLM using LiteLLM unified interface.

    This is a simplified implementation. For production, install litellm:
    pip install litellm

    Then use:
    from litellm import acompletion
    response = await acompletion(model=model, messages=messages, ...)

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4", "gemini-pro")
        messages: List of message dicts with 'role' and 'content'
        temperature: Sampling temperature
        max_tokens: Maximum response tokens
        **kwargs: Additional provider-specific parameters

    Returns:
        Response text
    """
    logger.info(f"LiteLLM call: model={model}, messages={len(messages)}")

    try:
        # Try to use litellm if installed
        try:
            import litellm
            from litellm import acompletion

            # Set API keys from settings
            litellm.openai_key = getattr(settings, "OPENAI_API_KEY", None)
            litellm.anthropic_key = getattr(settings, "ANTHROPIC_API_KEY", None)

            # Call LiteLLM
            response = await acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Extract text from response
            content = response['choices'][0]['message']['content']

            logger.info(f"✅ LiteLLM response received ({len(content)} chars)")

            return content

        except ImportError:
            logger.warning(
                "litellm not installed, falling back to direct API calls. "
                "Install with: pip install litellm"
            )

            # Fallback: Direct API calls
            if model.startswith("gpt-"):
                return await call_openai_direct(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            elif model.startswith("claude-"):
                return await call_anthropic_direct(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
            else:
                raise ValueError(f"Unsupported model: {model}")

    except Exception as e:
        logger.error(f"LiteLLM call failed: {e}", exc_info=True)
        raise


# ============================================================
# DIRECT API CALLS (FALLBACK)
# ============================================================

async def call_openai_direct(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> str:
    """
    Direct OpenAI API call (fallback when litellm not installed).

    Args:
        model: OpenAI model name
        messages: Messages list
        temperature: Temperature
        max_tokens: Max tokens

    Returns:
        Response text
    """
    api_key = getattr(settings, "OPENAI_API_KEY", None)

    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        logger.info(f"✅ OpenAI direct call succeeded")

        return content


async def call_anthropic_direct(
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2048
) -> str:
    """
    Direct Anthropic API call (fallback when litellm not installed).

    Args:
        model: Anthropic model name
        messages: Messages list
        temperature: Temperature
        max_tokens: Max tokens

    Returns:
        Response text
    """
    api_key = getattr(settings, "ANTHROPIC_API_KEY", None)

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured")

    url = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json"
    }

    # Convert messages to Anthropic format
    system_message = None
    user_messages = []

    for msg in messages:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            user_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    payload = {
        "model": model,
        "messages": user_messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    if system_message:
        payload["system"] = system_message

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()

        data = response.json()
        content = data["content"][0]["text"]

        logger.info(f"✅ Anthropic direct call succeeded")

        return content


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "call_litellm",
    "call_openai_direct",
    "call_anthropic_direct"
]
