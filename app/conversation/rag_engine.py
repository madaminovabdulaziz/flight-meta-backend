# app/conversation/rag_engine.py
"""
RAG Engine - Production Layer 3
Retrieval-Augmented Generation for personalized travel intelligence.

What RAG Solves:
- LLMs can't know: Current visa rules, airline policies, airport info
- LLMs hallucinate: "All US citizens can enter Turkey visa-free" (wrong!)
- RAG provides FACTS: Retrieved from curated knowledge base

Use Cases:
1. Visa Requirements: "I have a US passport" → Retrieves visa rules for US citizens
2. Airline Policies: "What's the baggage allowance?" → Retrieves airline-specific rules
3. Airport Info: "How early should I arrive?" → Retrieves airport guidelines
4. Travel Tips: "Best time to visit Tokyo" → Retrieves seasonal recommendations

Architecture:
- Vector Database: Stores travel knowledge as embeddings
- Retrieval: Finds relevant docs using semantic search
- Generation: LLM uses retrieved context to answer
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

import numpy as np

from app.conversation.semantic_cache import EmbeddingProvider
from app.infrastructure.cache import RedisCache
from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# KNOWLEDGE DOCUMENT MODELS
# ============================================================

class DocumentType(str):
    """Types of travel knowledge documents"""
    VISA_RULES = "visa_rules"
    AIRLINE_POLICY = "airline_policy"
    AIRPORT_INFO = "airport_info"
    DESTINATION_GUIDE = "destination_guide"
    BAGGAGE_RULES = "baggage_rules"
    COVID_REQUIREMENTS = "covid_requirements"
    TRAVEL_TIPS = "travel_tips"


class KnowledgeDocument(BaseModel):
    """
    Structured knowledge document.
    Each document represents a discrete fact or policy.
    """
    doc_id: str
    doc_type: DocumentType
    title: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    source: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)


class RetrievalResult(BaseModel):
    """Result from knowledge retrieval"""
    document: KnowledgeDocument
    relevance_score: float  # 0-1 similarity score
    rank: int  # Position in results


# ============================================================
# VECTOR DATABASE (SIMPLE REDIS-BASED IMPLEMENTATION)
# ============================================================

class VectorDatabase:
    """
    Simple vector database using Redis.
    For production scale (>100k docs), migrate to Pinecone/Qdrant/Weaviate.
    """

    KNOWLEDGE_PREFIX = "knowledge:doc:"
    INDEX_PREFIX = "knowledge:index:"

    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        embedding_provider: Optional[EmbeddingProvider] = None
    ):
        """
        Initialize vector database.

        Args:
            redis_cache: Redis cache instance
            embedding_provider: Embedding provider
        """
        self.redis = redis_cache or RedisCache()
        self.embedder = embedding_provider or EmbeddingProvider()

        logger.info("✓ VectorDatabase initialized")

    async def add_document(
        self,
        document: KnowledgeDocument
    ) -> bool:
        """
        Add document to vector database.

        Args:
            document: Knowledge document

        Returns:
            True if added successfully
        """
        try:
            # Generate embedding if not present
            if not document.embedding:
                embedding = self.embedder.embed_text(document.content)
                document.embedding = embedding.tolist()

            # Store document
            doc_key = f"{self.KNOWLEDGE_PREFIX}{document.doc_id}"
            doc_json = document.model_dump_json()

            success = await self.redis.set(doc_key, doc_json, ttl=None)  # No TTL

            if success:
                # Add to searchable index
                await self._add_to_index(document)

                logger.debug(
                    f"✓ Added document: {document.doc_id} "
                    f"({document.doc_type}, {len(document.content)} chars)"
                )
                return True
            else:
                logger.warning(f"Failed to add document: {document.doc_id}")
                return False

        except Exception as e:
            logger.error(f"Error adding document: {e}", exc_info=True)
            return False

    async def search(
        self,
        query: str,
        doc_type: Optional[DocumentType] = None,
        top_k: int = 5,
        min_relevance: float = 0.7
    ) -> List[RetrievalResult]:
        """
        Search for relevant documents using semantic similarity.

        Args:
            query: Search query
            doc_type: Optional document type filter
            top_k: Number of results to return
            min_relevance: Minimum relevance score

        Returns:
            List of retrieval results
        """
        logger.debug(f"Searching knowledge base: '{query[:50]}...'")

        # Step 1: Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Step 2: Get all documents (brute force for MVP)
        all_docs = await self._get_all_documents(doc_type_filter=doc_type)

        if not all_docs:
            logger.warning("No documents in knowledge base")
            return []

        # Step 3: Compute similarities
        results: List[Tuple[KnowledgeDocument, float]] = []

        for doc in all_docs:
            if not doc.embedding:
                continue

            doc_embedding = np.array(doc.embedding)
            similarity = self.embedder.cosine_similarity(
                query_embedding,
                doc_embedding
            )

            if similarity >= min_relevance:
                results.append((doc, similarity))

        # Step 4: Sort by relevance and return top K
        results.sort(key=lambda x: x[1], reverse=True)
        top_results = results[:top_k]

        # Step 5: Format as RetrievalResult
        formatted_results = [
            RetrievalResult(
                document=doc,
                relevance_score=score,
                rank=i + 1
            )
            for i, (doc, score) in enumerate(top_results)
        ]

        logger.info(
            f"✓ Retrieved {len(formatted_results)} relevant docs "
            f"(query: '{query[:50]}...')"
        )

        return formatted_results

    async def get_document(self, doc_id: str) -> Optional[KnowledgeDocument]:
        """
        Get document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document or None
        """
        doc_key = f"{self.KNOWLEDGE_PREFIX}{doc_id}"

        try:
            doc_json = await self.redis.get(doc_key)
            if doc_json:
                doc_dict = json.loads(doc_json)
                return KnowledgeDocument(**doc_dict)

        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")

        return None

    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete document by ID.

        Args:
            doc_id: Document ID

        Returns:
            True if deleted
        """
        doc_key = f"{self.KNOWLEDGE_PREFIX}{doc_id}"

        try:
            success = await self.redis.delete(doc_key)
            if success:
                logger.info(f"Deleted document: {doc_id}")
            return success

        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

    async def count_documents(self, doc_type: Optional[DocumentType] = None) -> int:
        """
        Count documents in database.

        Args:
            doc_type: Optional document type filter

        Returns:
            Number of documents
        """
        all_docs = await self._get_all_documents(doc_type_filter=doc_type)
        return len(all_docs)

    async def _get_all_documents(
        self,
        doc_type_filter: Optional[DocumentType] = None
    ) -> List[KnowledgeDocument]:
        """
        Get all documents from database.

        Args:
            doc_type_filter: Optional type filter

        Returns:
            List of documents
        """
        try:
            keys = await self.redis.keys(f"{self.KNOWLEDGE_PREFIX}*")

            docs = []
            for key in keys:
                doc_json = await self.redis.get(key)
                if doc_json:
                    try:
                        doc_dict = json.loads(doc_json)
                        doc = KnowledgeDocument(**doc_dict)

                        # Apply filter
                        if doc_type_filter and doc.doc_type != doc_type_filter:
                            continue

                        docs.append(doc)

                    except Exception as e:
                        logger.warning(f"Error parsing document: {e}")
                        continue

            return docs

        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            return []

    async def _add_to_index(self, document: KnowledgeDocument) -> bool:
        """Add document to searchable index"""
        # Simplified index for MVP
        index_key = f"{self.INDEX_PREFIX}{document.doc_type}:{document.doc_id}"

        try:
            await self.redis.set(index_key, document.doc_id, ttl=None)
            return True
        except Exception as e:
            logger.warning(f"Failed to add to index: {e}")
            return False


# ============================================================
# RAG ENGINE
# ============================================================

class RAGEngine:
    """
    Main RAG engine that combines retrieval + generation.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        llm_gateway: Optional[Any] = None
    ):
        """
        Initialize RAG engine.

        Args:
            vector_db: Vector database instance
            llm_gateway: LLM gateway for generation
        """
        self.vector_db = vector_db or VectorDatabase()
        self.llm_gateway = llm_gateway

        logger.info("✓ RAGEngine initialized")

    async def query(
        self,
        query: str,
        doc_type: Optional[DocumentType] = None,
        top_k: int = 3,
        generate_answer: bool = True
    ) -> Dict[str, Any]:
        """
        Query knowledge base and optionally generate answer.

        Args:
            query: User query
            doc_type: Optional document type filter
            top_k: Number of relevant docs to retrieve
            generate_answer: Whether to generate LLM answer

        Returns:
            Dict with retrieved docs and optional generated answer
        """
        logger.info(f"RAG query: '{query[:50]}...'")

        # Step 1: Retrieve relevant documents
        results = await self.vector_db.search(
            query=query,
            doc_type=doc_type,
            top_k=top_k,
            min_relevance=0.7
        )

        if not results:
            logger.warning("No relevant documents found")
            return {
                "query": query,
                "retrieved_docs": [],
                "generated_answer": None,
                "has_results": False
            }

        # Step 2: Format retrieved context
        context_parts = []
        for result in results:
            context_parts.append(
                f"[{result.document.title}]\n"
                f"{result.document.content}\n"
                f"(Relevance: {result.relevance_score:.2f})"
            )

        retrieved_context = "\n\n".join(context_parts)

        response_data = {
            "query": query,
            "retrieved_docs": [
                {
                    "title": r.document.title,
                    "content": r.document.content,
                    "type": r.document.doc_type,
                    "relevance": r.relevance_score,
                    "metadata": r.document.metadata
                }
                for r in results
            ],
            "context": retrieved_context,
            "has_results": True
        }

        # Step 3: Generate answer using LLM (if enabled)
        if generate_answer and self.llm_gateway:
            answer = await self._generate_answer(query, retrieved_context)
            response_data["generated_answer"] = answer
        else:
            response_data["generated_answer"] = None

        logger.info(
            f"✓ RAG query complete: {len(results)} docs retrieved, "
            f"answer_generated={generate_answer}"
        )

        return response_data

    async def _generate_answer(
        self,
        query: str,
        context: str
    ) -> str:
        """
        Generate answer using retrieved context + LLM.

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Generated answer
        """
        prompt = f"""You are a travel expert assistant. Answer the user's question using ONLY the information provided in the context below. If the context doesn't contain the answer, say "I don't have that information in my knowledge base."

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER (use only the context above):"""

        try:
            if self.llm_gateway:
                from app.conversation.llm_gateway import get_llm_gateway
                gateway = get_llm_gateway()

                result = await gateway.call(
                    prompt=prompt,
                    task_type="generation",
                    temperature=0.3,
                    max_tokens=512
                )

                return result["response"]
            else:
                logger.warning("LLM gateway not available")
                return context

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            # Fallback: return raw context
            return f"Based on available information:\n\n{context}"

    async def ingest_knowledge(
        self,
        documents: List[KnowledgeDocument]
    ) -> Dict[str, int]:
        """
        Ingest batch of knowledge documents.

        Args:
            documents: List of documents to add

        Returns:
            Stats dict
        """
        logger.info(f"Ingesting {len(documents)} knowledge documents...")

        added = 0
        failed = 0

        for doc in documents:
            success = await self.vector_db.add_document(doc)
            if success:
                added += 1
            else:
                failed += 1

        logger.info(
            f"✓ Knowledge ingestion complete: "
            f"{added} added, {failed} failed"
        )

        return {
            "total": len(documents),
            "added": added,
            "failed": failed
        }


# ============================================================
# KNOWLEDGE BASE SEED DATA
# ============================================================

async def seed_knowledge_base(rag_engine: RAGEngine) -> Dict[str, int]:
    """
    Seed knowledge base with initial travel data.

    Args:
        rag_engine: RAG engine instance

    Returns:
        Stats dict
    """
    logger.info("Seeding knowledge base with travel data...")

    seed_docs = [
        # Visa Rules
        KnowledgeDocument(
            doc_id="visa_us_turkey",
            doc_type=DocumentType.VISA_RULES,
            title="Turkey Visa Requirements for US Citizens",
            content=(
                "US passport holders can enter Turkey visa-free for tourism or business "
                "for up to 90 days within a 180-day period. Your passport must be valid "
                "for at least 6 months beyond your planned departure date from Turkey. "
                "You do NOT need an e-Visa as of 2024."
            ),
            metadata={"country": "Turkey", "nationality": "US", "visa_type": "visa-free"},
            tags=["visa", "turkey", "us", "visa-free"]
        ),

        KnowledgeDocument(
            doc_id="visa_us_uae",
            doc_type=DocumentType.VISA_RULES,
            title="UAE Visa Requirements for US Citizens",
            content=(
                "US passport holders receive a free 30-day visa on arrival when entering "
                "the UAE (Dubai, Abu Dhabi). No advance visa application needed. "
                "Can be extended once for an additional 30 days for a fee. "
                "Passport must be valid for at least 6 months."
            ),
            metadata={"country": "UAE", "nationality": "US", "visa_type": "visa-on-arrival"},
            tags=["visa", "uae", "dubai", "us", "visa-on-arrival"]
        ),

        # Airline Policies
        KnowledgeDocument(
            doc_id="baggage_emirates_economy",
            doc_type=DocumentType.BAGGAGE_RULES,
            title="Emirates Airlines Economy Class Baggage Allowance",
            content=(
                "Economy Class passengers are allowed:\n"
                "- Carry-on: 1 piece up to 7kg (15 lbs)\n"
                "- Checked: 2 pieces, each up to 23kg (50 lbs)\n"
                "- Total checked baggage: 46kg (101 lbs)\n"
                "Excess baggage fees apply for overweight or additional bags."
            ),
            metadata={"airline": "Emirates", "cabin_class": "economy"},
            tags=["baggage", "emirates", "economy", "allowance"]
        ),

        KnowledgeDocument(
            doc_id="baggage_turkish_airlines_economy",
            doc_type=DocumentType.BAGGAGE_RULES,
            title="Turkish Airlines Economy Class Baggage Allowance",
            content=(
                "Economy Class passengers are allowed:\n"
                "- Carry-on: 1 piece up to 8kg (17 lbs)\n"
                "- Checked: 1 piece up to 20kg (44 lbs) on domestic flights\n"
                "- Checked: 1 piece up to 23kg (50 lbs) on international flights\n"
                "Miles&Smiles members get additional baggage benefits."
            ),
            metadata={"airline": "Turkish Airlines", "cabin_class": "economy"},
            tags=["baggage", "turkish-airlines", "economy"]
        ),

        # Destination Guides
        KnowledgeDocument(
            doc_id="guide_istanbul_best_time",
            doc_type=DocumentType.DESTINATION_GUIDE,
            title="Best Time to Visit Istanbul",
            content=(
                "Best times to visit Istanbul:\n"
                "- Spring (April-May): Pleasant weather, fewer crowds, 15-20°C\n"
                "- Fall (September-October): Warm, beautiful colors, 18-25°C\n"
                "- Avoid: July-August (very hot, crowded, 30+°C)\n"
                "- Winter (December-February): Cold but cheap, 5-10°C"
            ),
            metadata={"city": "Istanbul", "country": "Turkey"},
            tags=["istanbul", "turkey", "best-time", "weather", "seasons"]
        ),

        KnowledgeDocument(
            doc_id="guide_dubai_best_time",
            doc_type=DocumentType.DESTINATION_GUIDE,
            title="Best Time to Visit Dubai",
            content=(
                "Best times to visit Dubai:\n"
                "- Winter (November-March): Perfect weather, 20-30°C, peak season\n"
                "- Spring/Fall (October, April): Warm but manageable, 25-35°C\n"
                "- Avoid: Summer (June-August): Extremely hot, 40-50°C, indoor activities only\n"
                "Dubai Shopping Festival (January) and Dubai Summer Surprises (July-August) offer deals."
            ),
            metadata={"city": "Dubai", "country": "UAE"},
            tags=["dubai", "uae", "best-time", "weather"]
        ),

        # Airport Info
        KnowledgeDocument(
            doc_id="airport_ist_arrival_time",
            doc_type=DocumentType.AIRPORT_INFO,
            title="Istanbul Airport - How Early to Arrive",
            content=(
                "Istanbul Airport (IST) recommendations:\n"
                "- International flights: Arrive 3 hours before departure\n"
                "- Domestic flights: Arrive 2 hours before departure\n"
                "- Peak times (mornings 6-9 AM, evenings 5-8 PM): Add 30 minutes\n"
                "Security can take 30-60 minutes during busy periods."
            ),
            metadata={"airport": "IST", "city": "Istanbul"},
            tags=["airport", "istanbul", "ist", "arrival-time", "security"]
        ),

        KnowledgeDocument(
            doc_id="airport_dxb_arrival_time",
            doc_type=DocumentType.AIRPORT_INFO,
            title="Dubai International Airport - How Early to Arrive",
            content=(
                "Dubai International Airport (DXB) recommendations:\n"
                "- International flights: Arrive 3 hours before departure\n"
                "- Terminal 3 (Emirates): Arrive 2.5-3 hours (can be very busy)\n"
                "- Other terminals: Arrive 2.5 hours minimum\n"
                "Immigration and security are generally efficient but can have queues during peak hours."
            ),
            metadata={"airport": "DXB", "city": "Dubai"},
            tags=["airport", "dubai", "dxb", "emirates", "arrival-time"]
        ),

        # Travel Tips
        KnowledgeDocument(
            doc_id="tip_save_money_flights",
            doc_type=DocumentType.TRAVEL_TIPS,
            title="How to Save Money on Flights",
            content=(
                "Top money-saving tips:\n"
                "1. Book 6-8 weeks in advance for international flights\n"
                "2. Fly on Tuesdays, Wednesdays, or Saturdays (cheapest days)\n"
                "3. Use flexible dates search to find cheapest days (+/- 3 days)\n"
                "4. Consider nearby airports (e.g., London has 6 airports)\n"
                "5. Book one-way tickets separately if round-trip is expensive\n"
                "6. Clear cookies or search in incognito mode to avoid price increases\n"
                "7. Sign up for price alerts and fare error notifications"
            ),
            metadata={"category": "money-saving"},
            tags=["travel-tips", "save-money", "booking", "cheap-flights"]
        ),

        KnowledgeDocument(
            doc_id="tip_layover_optimization",
            doc_type=DocumentType.TRAVEL_TIPS,
            title="How to Choose the Best Layover",
            content=(
                "Layover optimization tips:\n"
                "1. Sweet spot: 2-4 hours for international connections\n"
                "2. Too short (<1.5 hours): Risk missing connection\n"
                "3. Too long (>6 hours): Wasted time unless you can explore city\n"
                "4. Consider transit visa requirements for long layovers\n"
                "5. Major hubs with efficient transfers: Singapore (SIN), Dubai (DXB), Munich (MUC)\n"
                "6. If you miss a connection due to airline delay, they must rebook you for free"
            ),
            metadata={"category": "layovers"},
            tags=["travel-tips", "layover", "connections", "transit"]
        ),
    ]

    stats = await rag_engine.ingest_knowledge(seed_docs)

    logger.info(
        f"✓ Knowledge base seeded: {stats['added']} documents added"
    )

    return stats


# ============================================================
# FACTORY FUNCTION
# ============================================================

_rag_engine_instance: Optional[RAGEngine] = None

async def get_rag_engine() -> RAGEngine:
    """
    Get singleton RAG engine instance.

    Returns:
        RAGEngine instance
    """
    global _rag_engine_instance

    if _rag_engine_instance is None:
        _rag_engine_instance = RAGEngine()

        # Seed knowledge base on first initialization
        doc_count = await _rag_engine_instance.vector_db.count_documents()
        if doc_count == 0:
            logger.info("Knowledge base empty, seeding...")
            await seed_knowledge_base(_rag_engine_instance)

    return _rag_engine_instance


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "RAGEngine",
    "VectorDatabase",
    "KnowledgeDocument",
    "DocumentType",
    "RetrievalResult",
    "get_rag_engine",
    "seed_knowledge_base"
]
