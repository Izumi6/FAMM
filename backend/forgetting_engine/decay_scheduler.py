"""
Decay Scheduler — Orchestrates periodic decay cycles.

Manages the timing and execution of memory decay operations,
coordinating between the UtilityDecay module, the LifecycleController,
and the Pruner.

The scheduler runs a decay cycle every N agent interaction steps
(configurable via forgetting_engine.decay_interval_steps).
"""

from __future__ import annotations

import logging
from typing import Any

from backend.forgetting_engine.utility_decay import UtilityDecay
from backend.memory_engine.event_bus import EventBus, EventType
from backend.memory_engine.lifecycle_controller import LifecycleController
from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from config.settings import ForgettingEngineConfig, MemoryEngineConfig

logger = logging.getLogger(__name__)


class DecayScheduler:
    """
    Orchestrates periodic decay cycles across all active memories.

    Attributes:
        config: Forgetting engine configuration.
        utility_decay: Adaptive decay calculator.
        lifecycle: Memory lifecycle state controller.
        event_bus: Event bus for decay notifications.
        cycles_completed: Total number of decay cycles run.
    """

    def __init__(
        self,
        config: ForgettingEngineConfig | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize the decay scheduler.

        Args:
            config: Forgetting engine configuration.
            event_bus: Event bus for notifications.
        """
        self.config = config or ForgettingEngineConfig()
        self.event_bus = event_bus or EventBus()

        self.utility_decay = UtilityDecay(self.config)
        self.lifecycle = LifecycleController(
            config=MemoryEngineConfig(),
            event_bus=self.event_bus,
        )

        self.cycles_completed: int = 0

    def run_cycle(self, memories: list[MemoryRecord]) -> dict[str, Any]:
        """
        Execute one decay cycle on a batch of memories.

        For each memory:
        1. Apply utility-conditioned decay
        2. Check if pruning threshold is reached
        3. Evaluate lifecycle transitions

        Args:
            memories: List of active memories to process.

        Returns:
            Dict with cycle statistics:
                - decayed: Number of memories that had decay applied
                - pruned: Number of memories below prune threshold
                - transitioned: Number of lifecycle state changes
                - total_decay: Sum of all decay amounts applied
        """
        stats: dict[str, Any] = {
            "decayed": 0,
            "pruned": 0,
            "transitioned": 0,
            "total_decay": 0.0,
            "pruned_ids": [],
        }

        for memory in memories:
            if memory.state == MemoryState.DELETED:
                continue

            # Apply adaptive decay
            decay_amount = self.utility_decay.apply_decay(memory)
            stats["decayed"] += 1
            stats["total_decay"] += decay_amount

            self.event_bus.publish(
                EventType.MEMORY_DECAYED,
                {
                    "memory_id": memory.id,
                    "decay_amount": decay_amount,
                    "new_utility": memory.utility_score,
                },
            )

            # Check for pruning
            if self.utility_decay.should_prune(memory):
                stats["pruned"] += 1
                stats["pruned_ids"].append(memory.id)
            else:
                # Evaluate lifecycle transition
                if self.lifecycle.evaluate_and_apply(memory):
                    stats["transitioned"] += 1

        self.cycles_completed += 1

        self.event_bus.publish(EventType.DECAY_CYCLE_TRIGGERED, stats)

        logger.info(
            "Decay cycle #%d complete: %d decayed, %d pruned, %d transitioned",
            self.cycles_completed,
            stats["decayed"],
            stats["pruned"],
            stats["transitioned"],
        )

        return stats

    def should_run(self, step_count: int) -> bool:
        """
        Check if a decay cycle should run at the given step.

        Args:
            step_count: Current agent interaction step.

        Returns:
            True if the step aligns with the configured interval.
        """
        return step_count > 0 and step_count % self.config.decay_interval_steps == 0
