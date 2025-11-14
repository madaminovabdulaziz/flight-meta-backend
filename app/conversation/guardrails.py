# app/conversation/guardrails.py
"""
Guardrails & PII Protection - Production Layer 4
Ensures GDPR/CCPA compliance and prevents sensitive data leakage to LLMs.

CRITICAL REQUIREMENTS:
1. PII Detection: Identify names, emails, phone numbers, passport numbers
2. PII Anonymization: Replace PII with tokens before sending to LLM
3. PII De-anonymization: Restore PII in responses to user
4. Content Moderation: Block harmful/inappropriate content
5. Rate Limiting: Prevent abuse
6. Audit Logging: Track PII handling for compliance

EU/US Markets: Cannot send PII to OpenAI/Google without anonymization!

Example Flow:
User: "Book flight for John Smith, passport ABC123456"
→ Anonymized: "Book flight for <PERSON_1>, passport <PASSPORT_1>"
→ LLM processes anonymous version
→ Response: "Flight booked for <PERSON_1>"
→ De-anonymized: "Flight booked for John Smith"
"""

import re
import hashlib
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)


# ============================================================
# PII ENTITY TYPES
# ============================================================

class PIIType(str, Enum):
    """Types of Personally Identifiable Information"""
    PERSON_NAME = "PERSON_NAME"
    EMAIL = "EMAIL"
    PHONE_NUMBER = "PHONE_NUMBER"
    PASSPORT_NUMBER = "PASSPORT_NUMBER"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"               # Social Security Number
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    ADDRESS = "ADDRESS"
    IP_ADDRESS = "IP_ADDRESS"
    USER_ID = "USER_ID"


class PIIEntity(BaseModel):
    """Detected PII entity"""
    entity_type: PIIType
    original_value: str
    anonymized_value: str
    start_pos: int
    end_pos: int
    confidence: float = 1.0


# ============================================================
# PII DETECTION (REGEX-BASED + NER FALLBACK)
# ============================================================

class PIIDetector:
    """
    Detects PII using regex patterns + optional NER.
    For production, consider using Microsoft Presidio or AWS Comprehend.
    """

    # Regex patterns for common PII
    PATTERNS = {
        PIIType.EMAIL: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        PIIType.PHONE_NUMBER: r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        PIIType.PASSPORT_NUMBER: r'\b[A-Z]{1,2}\d{6,9}\b',  # Common format
        PIIType.CREDIT_CARD: r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        PIIType.SSN: r'\b\d{3}-\d{2}-\d{4}\b',
        PIIType.IP_ADDRESS: r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }

    # Common name patterns (simple heuristic)
    NAME_PATTERNS = [
        r'\b(Mr|Mrs|Ms|Dr|Prof)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
        r'\bfor\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
        r'\bname\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
    ]

    def __init__(self, use_ner: bool = False):
        """
        Initialize PII detector.

        Args:
            use_ner: Use NER for name detection (requires spacy)
        """
        self.use_ner = use_ner
        self._ner_model = None

        if use_ner:
            self._load_ner_model()

        logger.info(f"✓ PIIDetector initialized (NER={use_ner})")

    def _load_ner_model(self):
        """Load spaCy NER model for person name detection"""
        try:
            import spacy
            self._ner_model = spacy.load("en_core_web_sm")
            logger.info("✓ Loaded spaCy NER model")
        except ImportError:
            logger.warning("spaCy not installed, NER disabled")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")

    def detect(self, text: str) -> List[PIIEntity]:
        """
        Detect PII entities in text.

        Args:
            text: Input text

        Returns:
            List of detected PII entities
        """
        entities: List[PIIEntity] = []

        # 1. Detect using regex patterns
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.finditer(pattern, text)

            for match in matches:
                entities.append(PIIEntity(
                    entity_type=pii_type,
                    original_value=match.group(),
                    anonymized_value="",  # Will be assigned later
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95
                ))

        # 2. Detect names using patterns
        for pattern in self.NAME_PATTERNS:
            matches = re.finditer(pattern, text)

            for match in matches:
                # Extract the name part (usually in capturing group)
                name = match.group(1) if match.lastindex else match.group()

                # Filter out common false positives
                if self._is_likely_name(name):
                    entities.append(PIIEntity(
                        entity_type=PIIType.PERSON_NAME,
                        original_value=name,
                        anonymized_value="",
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.8
                    ))

        # 3. Use NER for additional name detection
        if self.use_ner and self._ner_model:
            ner_entities = self._detect_names_ner(text)
            entities.extend(ner_entities)

        # Remove duplicates (keep highest confidence)
        entities = self._deduplicate_entities(entities)

        logger.debug(f"Detected {len(entities)} PII entities in text")

        return entities

    def _detect_names_ner(self, text: str) -> List[PIIEntity]:
        """Detect person names using NER"""
        if not self._ner_model:
            return []

        doc = self._ner_model(text)
        entities = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                entities.append(PIIEntity(
                    entity_type=PIIType.PERSON_NAME,
                    original_value=ent.text,
                    anonymized_value="",
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    confidence=0.9
                ))

        return entities

    @staticmethod
    def _is_likely_name(text: str) -> bool:
        """
        Heuristic to filter out false positive names.

        Args:
            text: Candidate name

        Returns:
            True if likely a real name
        """
        # Filter out common false positives
        false_positives = {
            "this", "that", "what", "when", "where", "which",
            "flight", "ticket", "airport", "hotel", "trip"
        }

        if text.lower() in false_positives:
            return False

        # Must be at least 2 characters
        if len(text) < 2:
            return False

        # Must start with capital letter
        if not text[0].isupper():
            return False

        return True

    @staticmethod
    def _deduplicate_entities(entities: List[PIIEntity]) -> List[PIIEntity]:
        """
        Remove duplicate entities, keeping highest confidence.

        Args:
            entities: List of entities

        Returns:
            Deduplicated list
        """
        # Group by position range
        position_map: Dict[Tuple[int, int], PIIEntity] = {}

        for entity in entities:
            key = (entity.start_pos, entity.end_pos)

            if key not in position_map:
                position_map[key] = entity
            else:
                # Keep entity with higher confidence
                if entity.confidence > position_map[key].confidence:
                    position_map[key] = entity

        return list(position_map.values())


# ============================================================
# PII ANONYMIZATION
# ============================================================

class PIIAnonymizer:
    """
    Anonymizes PII by replacing with tokens.
    Maintains mapping for de-anonymization.
    """

    def __init__(self):
        """Initialize PII anonymizer"""
        self.entity_counter: Dict[PIIType, int] = {}
        logger.info("✓ PIIAnonymizer initialized")

    def anonymize(
        self,
        text: str,
        entities: List[PIIEntity]
    ) -> Tuple[str, Dict[str, PIIEntity]]:
        """
        Anonymize text by replacing PII with tokens.

        Args:
            text: Original text
            entities: Detected PII entities

        Returns:
            Tuple of (anonymized_text, entity_mapping)
        """
        if not entities:
            return text, {}

        # Sort entities by position (reverse order for replacement)
        sorted_entities = sorted(entities, key=lambda e: e.start_pos, reverse=True)

        anonymized_text = text
        entity_mapping: Dict[str, PIIEntity] = {}

        # Replace each entity with token
        for entity in sorted_entities:
            # Generate token
            token = self._generate_token(entity.entity_type)
            entity.anonymized_value = token

            # Replace in text
            anonymized_text = (
                anonymized_text[:entity.start_pos] +
                token +
                anonymized_text[entity.end_pos:]
            )

            # Store mapping
            entity_mapping[token] = entity

        logger.info(
            f"Anonymized {len(entities)} PII entities: {list(entity_mapping.keys())}"
        )

        return anonymized_text, entity_mapping

    def de_anonymize(
        self,
        text: str,
        entity_mapping: Dict[str, PIIEntity]
    ) -> str:
        """
        Restore original PII values from anonymized text.

        Args:
            text: Anonymized text
            entity_mapping: Mapping from tokens to original values

        Returns:
            De-anonymized text
        """
        if not entity_mapping:
            return text

        de_anonymized_text = text

        # Replace tokens with original values
        for token, entity in entity_mapping.items():
            de_anonymized_text = de_anonymized_text.replace(
                token,
                entity.original_value
            )

        logger.debug(f"De-anonymized {len(entity_mapping)} entities")

        return de_anonymized_text

    def _generate_token(self, entity_type: PIIType) -> str:
        """
        Generate anonymization token.

        Args:
            entity_type: Type of PII

        Returns:
            Token string
        """
        # Increment counter for this entity type
        if entity_type not in self.entity_counter:
            self.entity_counter[entity_type] = 0

        self.entity_counter[entity_type] += 1
        count = self.entity_counter[entity_type]

        # Generate token like "<PERSON_1>", "<EMAIL_2>", etc.
        return f"<{entity_type.value}_{count}>"


# ============================================================
# CONTENT MODERATOR
# ============================================================

class ContentModerator:
    """
    Blocks harmful, inappropriate, or off-topic content.
    """

    # Harmful content patterns
    HARMFUL_PATTERNS = [
        r'\b(hack|exploit|bypass|cheat)\b.*\b(system|security|payment)\b',
        r'\b(steal|fraud|scam|phishing)\b',
        r'\b(bomb|weapon|terrorist)\b',
    ]

    # Off-topic patterns (not travel-related)
    OFF_TOPIC_PATTERNS = [
        r'\b(recipe|cooking|food preparation)\b',
        r'\b(programming|code|software)\b.*\b(bug|feature|install)\b',
        r'\b(politics|election|government)\b.*\b(policy|law|vote)\b',
    ]

    @classmethod
    def check(cls, text: str) -> Dict[str, Any]:
        """
        Check content for policy violations.

        Args:
            text: Input text

        Returns:
            Dict with is_safe, violations, and reason
        """
        text_lower = text.lower()

        violations = []

        # Check for harmful content
        for pattern in cls.HARMFUL_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append("harmful_content")
                break

        # Check for off-topic content (optional warning)
        for pattern in cls.OFF_TOPIC_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                violations.append("off_topic")
                break

        is_safe = len(violations) == 0 or "off_topic" in violations

        result = {
            "is_safe": is_safe,
            "violations": violations,
            "reason": None
        }

        if not is_safe:
            result["reason"] = (
                "Your message contains potentially harmful content. "
                "Please rephrase and try again."
            )
            logger.warning(f"Content moderation blocked: {violations}")

        return result


# ============================================================
# GUARDRAILS MANAGER (MAIN INTERFACE)
# ============================================================

class GuardrailsManager:
    """
    Main guardrails interface combining PII protection and content moderation.
    """

    def __init__(
        self,
        enable_pii_protection: bool = True,
        enable_content_moderation: bool = True,
        enable_audit_logging: bool = True
    ):
        """
        Initialize guardrails manager.

        Args:
            enable_pii_protection: Enable PII anonymization
            enable_content_moderation: Enable content filtering
            enable_audit_logging: Log PII handling for compliance
        """
        self.enable_pii_protection = enable_pii_protection
        self.enable_content_moderation = enable_content_moderation
        self.enable_audit_logging = enable_audit_logging

        self.pii_detector = PIIDetector()
        self.pii_anonymizer = PIIAnonymizer()

        # Audit log
        self.audit_log: List[Dict[str, Any]] = []

        logger.info(
            f"✓ GuardrailsManager initialized: "
            f"pii={enable_pii_protection}, "
            f"moderation={enable_content_moderation}, "
            f"audit={enable_audit_logging}"
        )

    def process_input(
        self,
        text: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user input through guardrails.

        Args:
            text: User input text
            user_id: Optional user ID for audit

        Returns:
            Dict with processed_text, entity_mapping, and safety status
        """
        logger.debug(f"Processing input through guardrails: '{text[:50]}...'")

        # Step 1: Content moderation
        if self.enable_content_moderation:
            moderation_result = ContentModerator.check(text)

            if not moderation_result["is_safe"]:
                logger.warning(f"Content moderation blocked input: {moderation_result}")

                return {
                    "is_safe": False,
                    "processed_text": text,
                    "entity_mapping": {},
                    "violations": moderation_result["violations"],
                    "reason": moderation_result["reason"]
                }

        # Step 2: PII detection
        entities = []
        entity_mapping = {}
        processed_text = text

        if self.enable_pii_protection:
            entities = self.pii_detector.detect(text)

            if entities:
                logger.info(f"Detected {len(entities)} PII entities")

                # Anonymize PII
                processed_text, entity_mapping = self.pii_anonymizer.anonymize(
                    text=text,
                    entities=entities
                )

                logger.info(f"Anonymized text: '{processed_text[:50]}...'")

        # Step 3: Audit logging
        if self.enable_audit_logging and entities:
            self._log_pii_handling(
                user_id=user_id,
                original_text=text,
                anonymized_text=processed_text,
                entities=entities
            )

        return {
            "is_safe": True,
            "processed_text": processed_text,
            "entity_mapping": {k: v.dict() for k, v in entity_mapping.items()},
            "detected_entities": [e.entity_type.value for e in entities],
            "violations": []
        }

    def process_output(
        self,
        text: str,
        entity_mapping: Dict[str, PIIEntity]
    ) -> str:
        """
        Process LLM output to restore PII.

        Args:
            text: LLM output text
            entity_mapping: Entity mapping from input processing

        Returns:
            De-anonymized text
        """
        if not self.enable_pii_protection or not entity_mapping:
            return text

        # Convert dict back to PIIEntity objects if needed
        if entity_mapping and isinstance(list(entity_mapping.values())[0], dict):
            entity_mapping = {
                k: PIIEntity(**v) for k, v in entity_mapping.items()
            }

        de_anonymized_text = self.pii_anonymizer.de_anonymize(
            text=text,
            entity_mapping=entity_mapping
        )

        logger.debug("De-anonymized LLM output")

        return de_anonymized_text

    def _log_pii_handling(
        self,
        user_id: Optional[str],
        original_text: str,
        anonymized_text: str,
        entities: List[PIIEntity]
    ):
        """
        Log PII handling for compliance audit.

        Args:
            user_id: User ID
            original_text: Original text with PII
            anonymized_text: Anonymized text
            entities: Detected entities
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "original_text_hash": hashlib.sha256(original_text.encode()).hexdigest(),
            "anonymized_text": anonymized_text,
            "entity_types": [e.entity_type.value for e in entities],
            "entity_count": len(entities)
        }

        self.audit_log.append(audit_entry)

        # Keep only last 1000 entries in memory
        if len(self.audit_log) > 1000:
            self.audit_log = self.audit_log[-1000:]

        logger.debug(
            f"Audit log entry created: {len(entities)} PII entities for user {user_id}"
        )

    def get_audit_log(
        self,
        user_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve audit log entries.

        Args:
            user_id: Filter by user ID
            limit: Maximum entries to return

        Returns:
            List of audit entries
        """
        if user_id:
            entries = [e for e in self.audit_log if e.get("user_id") == user_id]
        else:
            entries = self.audit_log

        return entries[-limit:]


# ============================================================
# FACTORY FUNCTION
# ============================================================

_guardrails_instance: Optional[GuardrailsManager] = None

def get_guardrails_manager() -> GuardrailsManager:
    """
    Get singleton guardrails manager.

    Returns:
        GuardrailsManager instance
    """
    global _guardrails_instance

    if _guardrails_instance is None:
        _guardrails_instance = GuardrailsManager(
            enable_pii_protection=getattr(settings, "ENABLE_PII_PROTECTION", True),
            enable_content_moderation=getattr(settings, "ENABLE_CONTENT_MODERATION", True),
            enable_audit_logging=getattr(settings, "ENABLE_AUDIT_LOGGING", True)
        )

    return _guardrails_instance


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "GuardrailsManager",
    "PIIDetector",
    "PIIAnonymizer",
    "ContentModerator",
    "PIIType",
    "PIIEntity",
    "get_guardrails_manager"
]
