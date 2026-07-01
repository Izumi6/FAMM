"""
Utility Decay — Adaptive, utility-conditioned memory decay for FAMM.

This module implements FAMM's core forgetting mechanism, which replaces
the uniform Ebbinghaus decay used by MemoryBank.

Key Innovation:
    Decay rate is INVERSELY PROPORTIONAL to utility score.
    High-utility memories decay slower; low-utility memories decay faster.

    Formula: effective_decay = base_rate × (1 - utility_score)^exponent

    With default parameters (base_rate=0.05, exponent=2.0):
    - Utility 0.9: effective_decay = 0.05 × (0.1)² = 0.0005 (very slow)
    - Utility 0.5: effective_decay = 0.05 × (0.5)² = 0.0125 (moderate)
    - Utility 0.1: effective_decay = 0.05 × (0.9)² = 0.0405 (fast)

This means a memory with utility 0.9 decays ~80x slower than one with
utility 0.1, which is the core claim of our research hypothesis.

Comparison to MemoryBank (Ebbinghaus):
    MemoryBank: R(t) = e^{-t/S} where S is uniform for all memories
    FAMM:       utility(t+1) = utility(t) - base × (1 - utility(t))^exp

    The key difference is that FAMM's decay is CONTENT-AWARE (via utility)
    while Ebbinghaus is purely time-based.
"""

from __future__ import annotations

import logging
import math

from backend.memory_engine.memory_record import MemoryRecord
from config.settings import ForgettingEngineConfig

logger = logging.getLogger(__name__)


class UtilityDecay:
    """
    Computes adaptive decay for memories based on their utility score.

    Attributes:
        config: Forgetting engine configuration.
    """

    def __init__(self, config: ForgettingEngineConfig | None = None) -> None:
        """
        Initialize the utility decay calculator.

        Args:
            config: Forgetting engine configuration. Uses defaults if None.
        """
        self.config = config or ForgettingEngineConfig()

    def compute_effective_decay(self, memory: MemoryRecord) -> float:
        """
        Compute the effective decay amount for a memory.

        The decay is inversely proportional to utility:
            effective = base_rate × (1 - utility)^exponent

        Args:
            memory: The memory record to compute decay for.

        Returns:
            Decay amount (float ≥ 0). To be subtracted from utility_score.
        """
        base = self.config.base_decay_rate
        exponent = self.config.utility_exponent
        utility = memory.utility_score

        # Core formula: higher utility → lower decay
        effective = base * ((1.0 - utility) ** exponent)

        return max(0.0, effective)

    def apply_decay(self, memory: MemoryRecord) -> float:
        """
        Apply utility-conditioned decay to a memory.

        Computes the effective decay and subtracts it from the memory's
        utility score. Also updates the memory's stored decay_rate.

        Args:
            memory: The memory record to decay.

        Returns:
            The amount of utility that was subtracted.
        """
        effective_decay = self.compute_effective_decay(memory)
        old_utility = memory.utility_score

        memory.apply_decay(effective_decay)

        # Update the stored decay rate for transparency
        memory.decay_rate = effective_decay

        logger.debug(
            "Decay applied to %s: utility %.4f → %.4f (decay=%.6f)",
            memory.id[:8],
            old_utility,
            memory.utility_score,
            effective_decay,
        )

        return effective_decay

    def should_prune(self, memory: MemoryRecord) -> bool:
        """
        Determine if a memory should be pruned (permanently forgotten).

        Args:
            memory: The memory record to check.

        Returns:
            True if utility is below the prune threshold.
        """
        return memory.utility_score <= self.config.prune_threshold

    def project_decay_timeline(
        self,
        initial_utility: float,
        steps: int = 100,
    ) -> list[float]:
        """
        Project how a memory's utility will evolve over future decay steps.

        Useful for paper visualizations showing the decay curve shape.

        Args:
            initial_utility: Starting utility score.
            steps: Number of decay steps to project.

        Returns:
            List of projected utility scores over time.
        """
        utilities = [initial_utility]
        current = initial_utility

        for _ in range(steps):
            decay = self.config.base_decay_rate * ((1.0 - current) ** self.config.utility_exponent)
            current = max(0.0, current - decay)
            utilities.append(round(current, 6))

        return utilities
