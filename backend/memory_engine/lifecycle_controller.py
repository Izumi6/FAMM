"""
LifecycleController — Manages memory state transitions in FAMM.

Responsible for determining when a memory should transition between
lifecycle states based on its utility score, age, and access patterns.

State Machine:
    ACTIVE → STALE → ARCHIVED → DELETED
         ↑          ↑
         └──────────┘  (reactivation on access)

Design Decisions:
- State transitions are deterministic and rule-based (not learned),
  so they are fully reproducible and easy to reason about in the paper.
- Reactivation (STALE/ARCHIVED → ACTIVE) is possible when a memory
  is retrieved, implementing the "reinforcement" concept from the
  research hypothesis.
- The controller does NOT directly modify the vector store — it only
  updates MemoryRecord states. The Memory Manager handles persistence.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from backend.memory_engine.event_bus import EventBus, EventType
from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from config.settings import MemoryEngineConfig

logger = logging.getLogger(__name__)


class LifecycleController:
    """
    Controls memory lifecycle state transitions.

    Evaluates each memory's current state against configurable
    thresholds and triggers appropriate state transitions.

    Attributes:
        config: Memory engine configuration with thresholds.
        event_bus: Event bus for publishing state change events.
    """

    def __init__(
        self,
        config: MemoryEngineConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize the lifecycle controller.

        Args:
            config: Memory engine configuration. Uses defaults if None.
            event_bus: Event bus for state change notifications.
        """
        self.config = config or MemoryEngineConfig()
        self.event_bus = event_bus

    def evaluate(self, memory: MemoryRecord) -> MemoryState | None:
        """
        Evaluate a memory and determine if a state transition is needed.

        This does NOT apply the transition — it only returns the
        recommended new state, or None if no transition is needed.

        Args:
            memory: The memory record to evaluate.

        Returns:
            New state if a transition is recommended, None otherwise.
        """
        if memory.state == MemoryState.DELETED:
            return None  # Terminal state; no transitions possible

        age_days = memory.age_seconds() / 86400  # Convert seconds to days
        utility = memory.utility_score

        # Rule 1: Very low utility → mark for deletion (via pruner)
        if utility <= 0.01 and memory.state != MemoryState.DELETED:
            return MemoryState.DELETED

        # Rule 2: Low utility + old → archive
        if (
            utility < 0.1
            and age_days > self.config.archive_threshold_days
            and memory.state in (MemoryState.ACTIVE, MemoryState.STALE)
        ):
            return MemoryState.ARCHIVED

        # Rule 3: Moderate utility decay → stale
        if (
            utility < 0.3
            and age_days > self.config.stale_threshold_days
            and memory.state == MemoryState.ACTIVE
        ):
            return MemoryState.STALE

        # Rule 4: Stale memory with recovered utility → reactivate
        if memory.state == MemoryState.STALE and utility >= 0.3:
            return MemoryState.ACTIVE

        return None  # No transition needed

    def apply_transition(
        self,
        memory: MemoryRecord,
        new_state: MemoryState,
    ) -> None:
        """
        Apply a state transition to a memory record.

        Validates the transition and publishes an event.

        Args:
            memory: The memory record to transition.
            new_state: Target state.

        Raises:
            ValueError: If the transition is invalid.
        """
        old_state = memory.state
        memory.transition_to(new_state)

        logger.info(
            "Memory %s transitioned: %s → %s (utility=%.3f)",
            memory.id[:8],
            old_state.value,
            new_state.value,
            memory.utility_score,
        )

        if self.event_bus:
            self.event_bus.publish(
                EventType.MEMORY_STATE_CHANGED,
                {
                    "memory_id": memory.id,
                    "old_state": old_state.value,
                    "new_state": new_state.value,
                    "utility_score": memory.utility_score,
                },
            )

    def evaluate_and_apply(self, memory: MemoryRecord) -> bool:
        """
        Evaluate a memory and apply the transition if needed.

        Convenience method combining evaluate() and apply_transition().

        Args:
            memory: The memory record to process.

        Returns:
            True if a transition was applied, False otherwise.
        """
        new_state = self.evaluate(memory)
        if new_state is not None:
            self.apply_transition(memory, new_state)
            return True
        return False

    def reactivate_on_access(self, memory: MemoryRecord) -> None:
        """
        Reactivate a non-active memory when it is accessed.

        When the Goal-Aware Retriever returns a STALE or ARCHIVED
        memory, this method transitions it back to ACTIVE, implementing
        the reinforcement mechanism.

        Args:
            memory: The accessed memory record.
        """
        if memory.state in (MemoryState.STALE, MemoryState.ARCHIVED):
            self.apply_transition(memory, MemoryState.ACTIVE)
            logger.info(
                "Memory %s reactivated on access (utility=%.3f)",
                memory.id[:8],
                memory.utility_score,
            )
