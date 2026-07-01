"""
Pruner — Handles permanent removal and archival of memories.

The Pruner is the final stage of the forgetting pipeline:
1. UtilityDecay reduces utility scores over time
2. LifecycleController transitions states (ACTIVE → STALE → ARCHIVED)
3. Pruner permanently removes or archives memories below threshold

Design Decision:
- Archive-before-delete is configurable. When enabled, memories
  are moved to archived state before permanent deletion, allowing
  recovery if needed (useful during experiments).
- In production experiments, we track pruned memories for the paper's
  analysis of memory efficiency.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.memory_engine.event_bus import EventBus, EventType
from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from backend.vector_database.base_adapter import VectorStoreAdapter
from config.settings import ForgettingEngineConfig

logger = logging.getLogger(__name__)


class Pruner:
    """
    Permanently removes or archives memories below the utility threshold.

    Attributes:
        config: Forgetting engine configuration.
        vector_store: Vector database for deletion operations.
        event_bus: Event bus for pruning notifications.
        pruned_count: Total memories pruned since initialization.
    """

    def __init__(
        self,
        config: ForgettingEngineConfig,
        vector_store: VectorStoreAdapter,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize the pruner.

        Args:
            config: Forgetting engine configuration.
            vector_store: Vector database adapter.
            event_bus: Event bus for notifications.
        """
        self.config = config
        self.vector_store = vector_store
        self.event_bus = event_bus or EventBus()
        self.pruned_count: int = 0

    def prune(self, memories: list[MemoryRecord]) -> dict[str, Any]:
        """
        Prune memories that are below the utility threshold.

        Args:
            memories: Candidate memories to evaluate for pruning.

        Returns:
            Dict with:
                - pruned_ids: List of deleted memory IDs
                - archived_ids: List of archived memory IDs
                - pruned_count: Total pruned in this call
        """
        result: dict[str, Any] = {
            "pruned_ids": [],
            "archived_ids": [],
            "pruned_count": 0,
        }

        for memory in memories:
            if memory.state == MemoryState.DELETED:
                continue

            if memory.utility_score <= self.config.prune_threshold:
                if self.config.archive_before_delete and memory.state != MemoryState.ARCHIVED:
                    # Archive first
                    try:
                        memory.transition_to(MemoryState.ARCHIVED)
                        result["archived_ids"].append(memory.id)
                        self.vector_store.update(
                            ids=[memory.id],
                            metadatas=[memory.to_storage_dict()],
                        )
                    except ValueError:
                        # Already in a state that can't archive, force delete
                        self._delete_memory(memory, result)
                else:
                    self._delete_memory(memory, result)

        logger.info(
            "Pruning complete: %d pruned, %d archived",
            result["pruned_count"],
            len(result["archived_ids"]),
        )

        return result

    def _delete_memory(
        self,
        memory: MemoryRecord,
        result: dict[str, Any],
    ) -> None:
        """Permanently delete a memory from the vector store."""
        try:
            memory.transition_to(MemoryState.DELETED)
        except ValueError:
            memory.state = MemoryState.DELETED  # Force if transition is invalid

        self.vector_store.delete(ids=[memory.id])
        result["pruned_ids"].append(memory.id)
        result["pruned_count"] += 1
        self.pruned_count += 1

        self.event_bus.publish(
            EventType.MEMORY_PRUNED,
            {
                "memory_id": memory.id,
                "final_utility": memory.utility_score,
            },
        )

        logger.debug(
            "Pruned memory %s (utility=%.4f)",
            memory.id[:8],
            memory.utility_score,
        )
