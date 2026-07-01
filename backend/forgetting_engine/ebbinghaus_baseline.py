"""
Ebbinghaus Baseline — Uniform temporal decay for comparison.

Implements the standard Ebbinghaus forgetting curve as used by MemoryBank.
This serves as the BASELINE forgetting strategy against which FAMM's
utility-conditioned decay is compared.

Ebbinghaus Formula:
    R(t) = e^{-t/S}

Where:
    R(t) = retention (0 to 1) at time t
    t = time elapsed since creation (in hours)
    S = stability constant (higher = slower decay)

Key Difference from FAMM:
    - Ebbinghaus: ALL memories decay at the same rate (time-only)
    - FAMM: Decay rate depends on predicted future utility (content-aware)

    This is the null hypothesis: "uniform time-based decay is sufficient."
    If FAMM's utility-conditioned decay outperforms Ebbinghaus, our
    hypothesis is supported.
"""

from __future__ import annotations

import logging
import math

from backend.memory_engine.memory_record import MemoryRecord

logger = logging.getLogger(__name__)

# Default stability constant (in hours).
# Higher = slower decay. 24 means ~37% retention after 24 hours.
DEFAULT_STABILITY = 24.0
DEFAULT_PRUNE_THRESHOLD = 0.05


class EbbinghausBaseline:
    """
    Standard Ebbinghaus forgetting curve (uniform temporal decay).

    Used as the baseline forgetting strategy in experiments.
    All memories decay at the same rate regardless of content or utility.

    Attributes:
        stability: The stability constant S in the Ebbinghaus formula.
        prune_threshold: Utility below which memory is pruned.
    """

    def __init__(
        self,
        stability: float = DEFAULT_STABILITY,
        prune_threshold: float = DEFAULT_PRUNE_THRESHOLD,
    ) -> None:
        """
        Initialize the Ebbinghaus baseline.

        Args:
            stability: Stability constant S (hours). Higher = slower decay.
            prune_threshold: Retention below which memory is pruned.
        """
        self.stability = stability
        self.prune_threshold = prune_threshold

    def compute_retention(self, memory: MemoryRecord) -> float:
        """
        Compute current retention using the Ebbinghaus curve.

        R(t) = e^{-t/S}

        Args:
            memory: The memory record.

        Returns:
            Retention score in [0.0, 1.0].
        """
        age_hours = memory.age_seconds() / 3600
        retention = math.exp(-age_hours / self.stability)
        return max(0.0, min(1.0, retention))

    def apply_decay(self, memory: MemoryRecord) -> float:
        """
        Apply Ebbinghaus decay by setting utility to retention.

        Unlike FAMM's additive decay, Ebbinghaus directly computes
        the current retention from the age, overriding the utility score.

        Args:
            memory: The memory record to decay.

        Returns:
            The new retention/utility score.
        """
        retention = self.compute_retention(memory)
        old_utility = memory.utility_score

        memory.utility_score = retention
        memory.decay_rate = -math.log(retention + 1e-10) / max(memory.age_seconds() / 3600, 0.01)

        logger.debug(
            "Ebbinghaus decay for %s: utility %.4f → %.4f (age=%.1fh)",
            memory.id[:8],
            old_utility,
            retention,
            memory.age_seconds() / 3600,
        )

        return retention

    def should_prune(self, memory: MemoryRecord) -> bool:
        """
        Check if memory should be pruned based on retention.

        Args:
            memory: The memory record.

        Returns:
            True if retention is below threshold.
        """
        return self.compute_retention(memory) <= self.prune_threshold

    def project_decay_timeline(
        self,
        steps: int = 100,
        hours_per_step: float = 1.0,
    ) -> list[float]:
        """
        Project the Ebbinghaus retention curve over time.

        Useful for paper visualizations comparing against FAMM.

        Args:
            steps: Number of time steps.
            hours_per_step: Hours between each step.

        Returns:
            List of retention scores over time.
        """
        retentions = []
        for step in range(steps + 1):
            t = step * hours_per_step
            retention = math.exp(-t / self.stability)
            retentions.append(round(max(0.0, retention), 6))
        return retentions
