"""
Multi-Signal Ranker — Combines multiple retrieval signals for ranking.

This is the core ranking algorithm for Goal-Aware Retrieval.
Unlike traditional RAG that ranks by semantic similarity alone,
FAMM's ranker combines four signals:

1. Semantic similarity (from vector search)
2. Utility score (from Future Utility Predictor)
3. Goal alignment (from Goal Encoder)
4. Recency (time-based decay)

The final ranking score is a weighted combination:
    score = w1*similarity + w2*utility + w3*alignment + w4*recency

Research Rationale:
- This multi-signal approach is what makes FAMM's retrieval "goal-aware."
- By incorporating utility score (future prediction) and goal alignment
  (current goals), retrieval becomes context-sensitive rather than
  just query-sensitive.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from backend.memory_engine.memory_record import MemoryRecord
from config.settings import RankingWeightsConfig

logger = logging.getLogger(__name__)


@dataclass
class RankedResult:
    """
    A memory with its computed ranking scores.

    Used by the retriever to return ranked results with
    full score breakdown for analysis and paper figures.
    """

    memory: MemoryRecord
    final_score: float = 0.0
    semantic_similarity: float = 0.0
    utility_component: float = 0.0
    goal_alignment: float = 0.0
    recency_component: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)


class MultiSignalRanker:
    """
    Ranks candidate memories using a weighted multi-signal formula.

    Combines semantic similarity, utility score, goal alignment,
    and recency into a single ranking score.

    Attributes:
        weights: Configurable weights for each signal.
    """

    def __init__(self, weights: RankingWeightsConfig | None = None) -> None:
        """
        Initialize the ranker with configurable weights.

        Args:
            weights: Weight configuration. Uses defaults if None.
        """
        self.weights = weights or RankingWeightsConfig()

        logger.debug(
            "MultiSignalRanker initialized: sim=%.2f, util=%.2f, goal=%.2f, rec=%.2f",
            self.weights.semantic_similarity,
            self.weights.utility_score,
            self.weights.goal_alignment,
            self.weights.recency,
        )

    def rank(
        self,
        candidates: list[MemoryRecord],
        similarities: list[float],
        goal_alignments: list[float],
        top_k: int = 10,
    ) -> list[RankedResult]:
        """
        Rank candidate memories using multi-signal scoring.

        Args:
            candidates: List of candidate MemoryRecords.
            similarities: Semantic similarity scores (one per candidate).
            goal_alignments: Goal alignment scores (one per candidate).
            top_k: Number of top results to return.

        Returns:
            List of RankedResult, sorted by final_score descending.

        Raises:
            ValueError: If input lists have mismatched lengths.
        """
        if len(candidates) != len(similarities) or len(candidates) != len(goal_alignments):
            raise ValueError(
                f"Length mismatch: candidates={len(candidates)}, "
                f"similarities={len(similarities)}, alignments={len(goal_alignments)}"
            )

        if not candidates:
            return []

        results = []

        for i, memory in enumerate(candidates):
            # Normalize similarity (vector DB distances are often cosine distances)
            sim_score = max(0.0, min(1.0, similarities[i]))

            # Utility comes directly from the memory record
            utility = memory.utility_score

            # Goal alignment from the GoalEncoder
            alignment = max(0.0, min(1.0, goal_alignments[i]))

            # Recency score: exponential decay based on last access
            recency = self._compute_recency(memory)

            # Weighted combination
            final_score = (
                self.weights.semantic_similarity * sim_score
                + self.weights.utility_score * utility
                + self.weights.goal_alignment * alignment
                + self.weights.recency * recency
            )

            result = RankedResult(
                memory=memory,
                final_score=round(final_score, 4),
                semantic_similarity=round(sim_score, 4),
                utility_component=round(utility, 4),
                goal_alignment=round(alignment, 4),
                recency_component=round(recency, 4),
                score_breakdown={
                    "weighted_similarity": round(self.weights.semantic_similarity * sim_score, 4),
                    "weighted_utility": round(self.weights.utility_score * utility, 4),
                    "weighted_alignment": round(self.weights.goal_alignment * alignment, 4),
                    "weighted_recency": round(self.weights.recency * recency, 4),
                },
            )
            results.append(result)

        # Sort by final score descending
        results.sort(key=lambda r: r.final_score, reverse=True)

        return results[:top_k]

    @staticmethod
    def _compute_recency(memory: MemoryRecord) -> float:
        """
        Compute recency score from last access time.

        Uses exponential decay: more recent access → higher score.

        Formula: score = exp(-hours_since_access / 48)

        Args:
            memory: Memory record.

        Returns:
            Recency score in [0.0, 1.0].
        """
        hours_since = memory.seconds_since_access() / 3600
        score = float(np.exp(-hours_since / 48.0))
        return max(0.0, min(1.0, score))
