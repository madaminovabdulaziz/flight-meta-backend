"""
PreferenceService

Layer 2 of the memory system: structured long-term preferences stored in MySQL.

Responsibilities:
- Read user preferences from the user_preferences table
- Upsert (create/update) preferences in a normalized way
- Return a clean, domain-level dict usable by LangGraph nodes & ranking engine

This sits between:
- DB model: UserPreference (key, value_json)
- High-level logic: ranking, parameter extraction, question generation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, List, Callable

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.models.models import UserPreference  # adjust if your model lives elsewhere

logger = logging.getLogger(__name__)


class PreferenceService:
    """
    Facade for working with structured user preferences (Layer 2).

    Design principles:
    - Async, production-ready
    - Thin abstraction over SQLAlchemy
    - Returns a clean, strongly-shaped dict of preferences
    - Safe to call from LangGraph nodes and FastAPI handlers
    """

    # Canonical preference keys we support (but we do not strictly enforce)
    SUPPORTED_KEYS = {
        "preferred_airports",
        "preferred_airlines",
        "budget_range",
        "prefers_direct",
        "hates_overnight_layovers",
        "preferred_time_of_day",
        "seat_preference",
        "usual_trip_type",
    }

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession] = AsyncSessionLocal,
    ) -> None:
        """
        Args:
            session_factory: A callable that returns an AsyncSession.
                             Defaults to AsyncSessionLocal from your DB layer.
        """
        self._session_factory = session_factory

    # ---------------------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------------------

    async def get_user_preferences(self, user_id: int) -> Dict[str, Any]:
        """
        Load all preferences for a user and return as a normalized dict.

        Example output:
        {
            "preferred_airports": ["LGW", "LHR"],
            "preferred_airlines": ["TK", "QR"],
            "budget_range": [200, 500],
            "prefers_direct": True,
            "hates_overnight_layovers": True,
            "preferred_time_of_day": "morning",
        }
        """
        async with self._session_factory() as session:
            try:
                stmt = select(UserPreference).where(UserPreference.user_id == user_id)
                result = await session.execute(stmt)
                rows: List[UserPreference] = result.scalars().all()

                raw_map: Dict[str, Any] = {}
                for row in rows:
                    # Last write wins if there are duplicates
                    raw_map[row.key] = row.value_json

                # You can normalize / enforce shapes here if needed
                normalized = self._normalize_preferences(raw_map)

                logger.debug(
                    "Loaded %d preferences for user %s",
                    len(normalized),
                    user_id,
                )
                return normalized

            except Exception as e:
                logger.error(
                    "Failed to load preferences for user %s: %s",
                    user_id,
                    e,
                    exc_info=True,
                )
                # Fail safe: no preferences loaded; caller should handle default behavior
                return {}

    async def upsert_preference(
        self,
        user_id: int,
        key: str,
        value: Any,
    ) -> None:
        """
        Create or update a single preference for a user.

        This is the core write primitive. Higher-level logic (LLM, heuristics)
        decide what to write; this only persists it.

        Example:
            await service.upsert_preference(
                user_id=1,
                key="preferred_airports",
                value=["LGW", "LHR"],
            )
        """
        async with self._session_factory() as session:
            try:
                # First try to update existing
                stmt = (
                    select(UserPreference)
                    .where(
                        UserPreference.user_id == user_id,
                        UserPreference.key == key,
                    )
                    .limit(1)
                )
                result = await session.execute(stmt)
                existing: Optional[UserPreference] = result.scalars().first()

                if existing:
                    existing.value_json = value
                    logger.info(
                        "Updated preference '%s' for user %s: %s",
                        key,
                        user_id,
                        value,
                    )
                else:
                    # Insert new
                    pref = UserPreference(
                        user_id=user_id,
                        key=key,
                        value_json=value,
                    )
                    session.add(pref)
                    logger.info(
                        "Created preference '%s' for user %s: %s",
                        key,
                        user_id,
                        value,
                    )

                await session.commit()

            except Exception as e:
                await session.rollback()
                logger.error(
                    "Failed to upsert preference '%s' for user %s: %s",
                    key,
                    user_id,
                    e,
                    exc_info=True,
                )
                raise

    async def bulk_update_preferences(
        self,
        user_id: int,
        updates: Dict[str, Any],
    ) -> None:
        """
        Convenience helper to upsert multiple preferences in one go.

        Example:
            await bulk_update_preferences(1, {
                "preferred_airports": ["LGW", "LHR"],
                "prefers_direct": True,
            })
        """
        if not updates:
            return

        async with self._session_factory() as session:
            try:
                # Load existing preferences
                stmt = select(UserPreference).where(UserPreference.user_id == user_id)
                result = await session.execute(stmt)
                rows: List[UserPreference] = result.scalars().all()

                existing_map: Dict[str, UserPreference] = {
                    row.key: row for row in rows
                }

                # Apply updates
                for key, value in updates.items():
                    if key in existing_map:
                        existing_map[key].value_json = value
                    else:
                        session.add(
                            UserPreference(
                                user_id=user_id,
                                key=key,
                                value_json=value,
                            )
                        )

                await session.commit()

                logger.info(
                    "Bulk updated %d preferences for user %s",
                    len(updates),
                    user_id,
                )

            except Exception as e:
                await session.rollback()
                logger.error(
                    "Failed bulk update preferences for user %s: %s",
                    user_id,
                    e,
                    exc_info=True,
                )
                raise

    # ---------------------------------------------------------------------
    # INTERNAL HELPERS
    # ---------------------------------------------------------------------

    def _normalize_preferences(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize preferences into well-shaped structures where possible.

        This is where you can enforce types and defaults that the rest of
        your system expects.
        """
        normalized: Dict[str, Any] = {}

        # Direct copy for now; below we add a few gentle normalizations
        normalized.update(raw)

        # Example: ensure budget_range is always a 2-element list if present
        if "budget_range" in normalized:
            br = normalized["budget_range"]
            if isinstance(br, (tuple, list)) and len(br) == 2:
                normalized["budget_range"] = [float(br[0]), float(br[1])]
            elif isinstance(br, (int, float)):
                # If someone stored a single number, interpret as [0, value]
                normalized["budget_range"] = [0.0, float(br)]
            else:
                # Invalid shape, drop or log
                logger.warning("Invalid budget_range format: %r", br)
                normalized.pop("budget_range", None)

        # Example: ensure preferred_airports is always a list
        if "preferred_airports" in normalized:
            pa = normalized["preferred_airports"]
            if isinstance(pa, str):
                normalized["preferred_airports"] = [pa]
            elif isinstance(pa, (list, tuple)):
                normalized["preferred_airports"] = list(pa)
            else:
                logger.warning("Invalid preferred_airports format: %r", pa)
                normalized.pop("preferred_airports", None)

        # Example: ensure preferred_airlines is always a list
        if "preferred_airlines" in normalized:
            pl = normalized["preferred_airlines"]
            if isinstance(pl, str):
                normalized["preferred_airlines"] = [pl]
            elif isinstance(pl, (list, tuple)):
                normalized["preferred_airlines"] = list(pl)
            else:
                logger.warning("Invalid preferred_airlines format: %r", pl)
                normalized.pop("preferred_airlines", None)

        # Booleans: prefers_direct & hates_overnight_layovers
        for bool_key in ("prefers_direct", "hates_overnight_layovers"):
            if bool_key in normalized:
                val = normalized[bool_key]
                if isinstance(val, str):
                    normalized[bool_key] = val.lower() in ("1", "true", "yes", "y")
                else:
                    normalized[bool_key] = bool(val)

        return normalized
