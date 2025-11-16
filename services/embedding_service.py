import hashlib
import logging
import asyncio
from typing import List, Dict

from google.genai import Client
from google.genai.types import EmbedContentConfig

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddingService:
    def __init__(self):
        self.client = Client(api_key=settings.GEMINI_API_KEY)
        self.model = "text-embedding-004"
        self.cache: Dict[str, list[float]] = {}
        self.max_batch_size = 100

    # -------------------------
    # Utility
    # -------------------------
    def _hash(self, text: str) -> str:
        """Cache key."""
        return hashlib.md5(text.encode()).hexdigest()

    async def _retry(self, func, attempts=4, delay=0.5):
        """Retry wrapper for sync Gemini calls."""
        for i in range(attempts):
            try:
                return await func()
            except Exception as e:
                if i == attempts - 1:
                    logger.error(f"Embedding call failed after retries: {e}")
                    raise

                wait = delay * (2 ** i)
                logger.warning(f"[Gemini Retry] Attempt {i+1}/{attempts}, waiting {wait:.1f}s")
                await asyncio.sleep(wait)

    # -------------------------
    # Single embedding
    # -------------------------
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        key = self._hash(text)

        # Use cache
        if key in self.cache:
            return self.cache[key]

        def sync_call():
            return self.client.models.embed_content(
                model=self.model,
                contents=[text],  # must be list
                config=EmbedContentConfig(
                    output_dimensionality=settings.QDRANT_VECTOR_SIZE
                )
            )

        # Call sync Gemini API inside thread pool
        response = await self._retry(lambda: asyncio.to_thread(sync_call))

        embedding = response.embeddings[0].values
        self.cache[key] = embedding
        return embedding

    # -------------------------
    # Batch embeddings
    # -------------------------
    async def embed_batch(self, texts: List[str]) -> List[list[float]]:
        """Batch embed multiple documents."""
        if not texts:
            return []

        results = [None] * len(texts)
        to_process = []
        index_map = {}

        # First pass: get cached embeddings
        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self.cache:
                results[i] = self.cache[key]
            else:
                index_map[len(to_process)] = i
                to_process.append(text)

        if not to_process:
            return results

        # Process in batches
        for start in range(0, len(to_process), self.max_batch_size):
            batch = to_process[start:start + self.max_batch_size]

            def sync_call():
                return self.client.models.embed_content(
                    model=self.model,
                    contents=batch,
                    config=EmbedContentConfig(
                        output_dimensionality=settings.QDRANT_VECTOR_SIZE
                    )
                )

            response = await self._retry(lambda: asyncio.to_thread(sync_call))

            for idx, emb_info in enumerate(response.embeddings):
                original_index = index_map[start + idx]
                emb = emb_info.values

                key = self._hash(texts[original_index])
                self.cache[key] = emb
                results[original_index] = emb

        return results


# Global instance
embedding_service = GeminiEmbeddingService()


# Helper functions for easy import
async def generate_embedding(text: str) -> list[float]:
    return await embedding_service.embed_text(text)


async def generate_embeddings_batch(texts: List[str]) -> List[list[float]]:
    return await embedding_service.embed_batch(texts)
