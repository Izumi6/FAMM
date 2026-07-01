"""
Feature Extractor — Extracts predictive features for future utility scoring.

This module computes a feature vector for each memory at write-time
that the scoring model uses to predict the memory's future utility.

The features capture signals that correlate with future relevance:
1. Goal similarity: How semantically close is the memory to active goals?
2. Recency signal: Normalized time since creation (newer = higher)
3. Access frequency: How often has similar content been accessed?
4. Source type encoding: Reflections tend to be more durable than conversations
5. Entity overlap: Shared entities between memory and goals

Research Rationale:
- Existing systems score memories ONLY at retrieval time (reactive).
- FAMM scores at WRITE time (proactive), predicting future value.
- This enables the Forgetting Engine to protect high-utility memories
  from decay, which is our core hypothesis.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from backend.memory_engine.memory_record import MemoryRecord, SourceType
from backend.vector_database.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)

# Source type priors: higher = more likely to be useful in the future
# Based on the cognitive science principle that reflections and observations
# tend to be more durable and reusable than conversational turns.
SOURCE_TYPE_PRIORS: dict[SourceType, float] = {
    SourceType.REFLECTION: 0.85,
    SourceType.OBSERVATION: 0.70,
    SourceType.SYSTEM: 0.65,
    SourceType.CONVERSATION: 0.50,
    SourceType.CONSOLIDATED: 0.80,
}


class FeatureExtractor:
    """
    Extracts a feature vector from a MemoryRecord for utility prediction.

    The feature vector contains signals that are predictive of whether
    a memory will be useful for future agent tasks.

    Attributes:
        embedding_service: For computing semantic similarity.
        feature_names: Ordered list of feature names (for interpretability).

    Example:
        >>> extractor = FeatureExtractor(embedding_service)
        >>> features = extractor.extract(memory, goal_context=["complete the report"])
        >>> len(features)
        5
    """

    FEATURE_NAMES = [
        "goal_similarity",
        "recency_score",
        "access_frequency_score",
        "source_type_prior",
        "entity_overlap_score",
    ]

    def __init__(self, embedding_service: EmbeddingService) -> None:
        """
        Initialize the feature extractor.

        Args:
            embedding_service: Service for computing embeddings and similarity.
        """
        self.embedding_service = embedding_service

    def extract(
        self,
        memory: MemoryRecord,
        goal_context: list[str] | None = None,
    ) -> list[float]:
        """
        Extract a feature vector from a memory record.

        Args:
            memory: The memory to extract features from.
            goal_context: List of active goal descriptions (natural language).

        Returns:
            Feature vector as a list of floats in [0.0, 1.0].
        """
        features = [
            self._goal_similarity(memory, goal_context or []),
            self._recency_score(memory),
            self._access_frequency_score(memory),
            self._source_type_prior(memory),
            self._entity_overlap_score(memory, goal_context or []),
        ]

        return features

    def extract_dict(
        self,
        memory: MemoryRecord,
        goal_context: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Extract features as a named dictionary (for logging/debugging).

        Args:
            memory: The memory to extract features from.
            goal_context: Active goal descriptions.

        Returns:
            Dict mapping feature names to values.
        """
        values = self.extract(memory, goal_context)
        return dict(zip(self.FEATURE_NAMES, values))

    # ─────────────────────────────────────────────
    # Individual feature computations
    # ─────────────────────────────────────────────

    def _goal_similarity(
        self,
        memory: MemoryRecord,
        goal_context: list[str],
    ) -> float:
        """
        Compute maximum semantic similarity between memory and active goals.

        Returns the highest cosine similarity between the memory's
        embedding and any goal embedding. If no goals are provided,
        returns a neutral score of 0.5.

        Args:
            memory: Memory record (must have embedding set).
            goal_context: List of goal description strings.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        if not goal_context or not memory.embedding:
            return 0.5  # Neutral when no goals available

        goal_embeddings = self.embedding_service.encode_batch(goal_context)
        similarities = self.embedding_service.compute_batch_similarity(
            query_embedding=memory.embedding,
            candidate_embeddings=goal_embeddings,
        )

        # Return max similarity (most relevant goal)
        max_sim = max(similarities)
        # Clamp to [0, 1] (cosine sim can be slightly negative)
        return float(max(0.0, min(1.0, max_sim)))

    @staticmethod
    def _recency_score(memory: MemoryRecord) -> float:
        """
        Compute a recency score based on memory age.

        Uses an exponential decay function so that very recent
        memories score high but the score drops off rapidly.

        Formula: score = exp(-age_hours / 24)
        - Age 0 hours → 1.0
        - Age 24 hours → 0.37
        - Age 72 hours → 0.05

        Args:
            memory: Memory record.

        Returns:
            Recency score in [0.0, 1.0].
        """
        age_hours = memory.age_seconds() / 3600
        score = float(np.exp(-age_hours / 24.0))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _access_frequency_score(memory: MemoryRecord) -> float:
        """
        Score based on how frequently the memory has been accessed.

        Uses a logarithmic scale to prevent very high access counts
        from dominating.

        Formula: score = log(1 + access_count) / log(1 + max_expected)

        Args:
            memory: Memory record.

        Returns:
            Frequency score in [0.0, 1.0].
        """
        max_expected_accesses = 50  # Normalization constant
        score = float(np.log1p(memory.access_count) / np.log1p(max_expected_accesses))
        return max(0.0, min(1.0, score))

    @staticmethod
    def _source_type_prior(memory: MemoryRecord) -> float:
        """
        Return a prior probability based on memory source type.

        Different source types have different base probabilities of
        being useful in the future (based on cognitive science):
        - Reflections are intentionally created insights → high prior
        - Observations are factual records → moderate-high prior
        - Conversations may be transient → moderate prior

        Args:
            memory: Memory record.

        Returns:
            Source type prior in [0.0, 1.0].
        """
        return SOURCE_TYPE_PRIORS.get(memory.source_type, 0.5)

    @staticmethod
    def _entity_overlap_score(
        memory: MemoryRecord,
        goal_context: list[str],
    ) -> float:
        """
        Compute entity overlap between memory content and goals.

        Uses simple keyword extraction (capitalized words, quoted strings)
        as a lightweight proxy for entity recognition. This avoids the
        overhead of a full NER model while still capturing important signals.

        Args:
            memory: Memory record.
            goal_context: Goal description strings.

        Returns:
            Overlap score in [0.0, 1.0].
        """
        if not goal_context:
            return 0.0

        memory_entities = _extract_entities(memory.content)
        if not memory_entities:
            return 0.0

        goal_text = " ".join(goal_context)
        goal_entities = _extract_entities(goal_text)

        if not goal_entities:
            return 0.0

        overlap = memory_entities & goal_entities
        # Jaccard-like coefficient
        union = memory_entities | goal_entities
        return len(overlap) / len(union) if union else 0.0


def _extract_entities(text: str) -> set[str]:
    """
    Extract entity-like tokens from text.

    Uses lightweight heuristics:
    - Capitalized words (proper nouns)
    - Quoted strings
    - Technical terms (words with underscores, camelCase)

    This is intentionally simple — a research contribution in future
    utility prediction, not in entity recognition.

    Args:
        text: Input text.

    Returns:
        Set of normalized entity strings.
    """
    entities: set[str] = set()

    # Capitalized words (likely proper nouns or important terms)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update(w.lower() for w in capitalized)

    # Quoted strings
    quoted = re.findall(r'"([^"]+)"', text) + re.findall(r"'([^']+)'", text)
    entities.update(q.lower() for q in quoted)

    # Technical terms with underscores
    technical = re.findall(r'\b\w+_\w+\b', text)
    entities.update(t.lower() for t in technical)

    return entities
