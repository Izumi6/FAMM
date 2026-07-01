"""
Goal Encoder — Encodes agent goals into vector space for alignment scoring.

Converts natural language goal descriptions into dense embeddings
that can be compared against memory embeddings to determine
goal-memory alignment.

Design Decisions:
- Goals are encoded using the same embedding model as memories,
  ensuring consistent semantic space.
- Multiple goals can be encoded simultaneously for efficiency.
- Goal embeddings are cached to avoid redundant encoding when
  goals don't change between retrieval calls.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.vector_database.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class GoalEncoder:
    """
    Encodes agent goals into the same embedding space as memories.

    Provides cached encoding and alignment scoring for the
    Goal-Aware Retriever.

    Attributes:
        embedding_service: Shared embedding model.
        _cache: LRU-style cache of goal text → embedding.
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """
        Initialize the goal encoder.

        Args:
            embedding_service: Embedding service (shared with MemoryManager).
        """
        self.embedding_service = embedding_service
        self._cache: dict[str, list[float]] = {}

    def encode_goals(self, goals: list[str]) -> list[list[float]]:
        """
        Encode a list of goal descriptions into embeddings.

        Uses caching: if a goal was encoded before, returns cached result.

        Args:
            goals: List of natural language goal descriptions.

        Returns:
            List of dense embedding vectors, one per goal.
        """
        if not goals:
            return []

        # Split into cached and uncached
        uncached_goals = [g for g in goals if g not in self._cache]
        if uncached_goals:
            new_embeddings = self.embedding_service.encode_batch(uncached_goals)
            for goal, embedding in zip(uncached_goals, new_embeddings):
                self._cache[goal] = embedding

        return [self._cache[g] for g in goals]

    def compute_alignment(
        self,
        memory_embedding: list[float],
        goal_embeddings: list[list[float]],
        aggregation: str = "max",
    ) -> float:
        """
        Compute alignment between a memory and a set of goals.

        The alignment score represents how well-suited a memory is
        for the agent's current goal stack.

        Args:
            memory_embedding: Dense vector for the memory.
            goal_embeddings: Dense vectors for each active goal.
            aggregation: How to combine per-goal scores.
                - "max": Return the highest alignment (default).
                - "mean": Return the average alignment.
                - "weighted_mean": Weight by goal position (first = highest).

        Returns:
            Alignment score in [0.0, 1.0].
        """
        if not goal_embeddings or not memory_embedding:
            return 0.0

        similarities = self.embedding_service.compute_batch_similarity(
            query_embedding=memory_embedding,
            candidate_embeddings=goal_embeddings,
        )

        # Clamp similarities to [0, 1]
        similarities = [max(0.0, min(1.0, s)) for s in similarities]

        if aggregation == "max":
            return max(similarities)
        elif aggregation == "mean":
            return sum(similarities) / len(similarities)
        elif aggregation == "weighted_mean":
            # Weight by position: first goal gets highest weight
            n = len(similarities)
            weights = [(n - i) / sum(range(1, n + 1)) for i in range(n)]
            return sum(w * s for w, s in zip(weights, similarities))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def clear_cache(self) -> None:
        """Clear the goal embedding cache."""
        self._cache.clear()
