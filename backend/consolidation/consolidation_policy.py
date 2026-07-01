"""
Consolidation Policy — Decides when and what to consolidate.

Governs the consolidation triggers and candidate selection.
"""

from __future__ import annotations

import logging

from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from config.settings import ConsolidationConfig

logger = logging.getLogger(__name__)


class ConsolidationPolicy:
    """
    Policy for deciding when to trigger consolidation and which
    memories are candidates.

    Attributes:
        config: Consolidation configuration.
    """

    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()

    def should_consolidate(self, step_count: int) -> bool:
        """Check if consolidation should run at the given step."""
        return (
            step_count > 0
            and step_count % self.config.consolidation_interval_steps == 0
        )

    def select_candidates(
        self, memories: list[MemoryRecord]
    ) -> list[MemoryRecord]:
        """
        Select memories eligible for consolidation.

        Criteria:
        - State is ACTIVE or STALE (not ARCHIVED or DELETED)
        - Not already consolidated (no consolidation_group)
        - Utility is in the moderate range (0.1 to 0.6)
          High-utility memories are preserved as-is.
          Very low-utility memories will be pruned anyway.

        Args:
            memories: All available memories.

        Returns:
            Filtered list of consolidation candidates.
        """
        candidates = [
            m for m in memories
            if (
                m.state in (MemoryState.ACTIVE, MemoryState.STALE)
                and m.consolidation_group is None
                and 0.1 <= m.utility_score <= 0.6
            )
        ]

        logger.debug(
            "Consolidation candidates: %d out of %d memories",
            len(candidates),
            len(memories),
        )

        return candidates
