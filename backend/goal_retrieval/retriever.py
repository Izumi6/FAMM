"""
Goal-Aware Retriever — Retrieves memories conditioned on agent goals.

This is FAMM's second key innovation (after the Future Utility Predictor).
Traditional RAG retrieves by query similarity alone. The Goal-Aware
Retriever adds goal context as a first-class signal in retrieval:

1. Fetch candidates via vector similarity (standard RAG)
2. Compute goal alignment for each candidate (novel)
3. Rank using multi-signal scorer (similarity + utility + alignment + recency)
4. Return top-k with full scoring breakdown

The retriever operates independently of the MemoryManager and can be
swapped out for a baseline similarity-only retriever in ablation studies.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.goal_retrieval.goal_encoder import GoalEncoder
from backend.goal_retrieval.multi_signal_ranker import MultiSignalRanker, RankedResult
from backend.memory_engine.event_bus import EventBus, EventType
from backend.memory_engine.lifecycle_controller import LifecycleController
from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from backend.vector_database.base_adapter import VectorStoreAdapter
from backend.vector_database.embedding_service import EmbeddingService
from config.settings import GoalRetrievalConfig, MemoryEngineConfig

logger = logging.getLogger(__name__)


class GoalAwareRetriever:
    """
    Retrieves memories conditioned on the agent's active goals.

    Extends traditional semantic retrieval with goal alignment scoring
    and multi-signal ranking.

    Attributes:
        config: Retrieval configuration.
        vector_store: Vector database adapter.
        embedding_service: Embedding model.
        goal_encoder: Encodes goals into embedding space.
        ranker: Multi-signal ranking algorithm.
        lifecycle: For reactivating stale memories on access.
        event_bus: For publishing retrieval events.
    """

    def __init__(
        self,
        config: GoalRetrievalConfig,
        vector_store: VectorStoreAdapter,
        embedding_service: EmbeddingService,
        event_bus: EventBus | None = None,
    ) -> None:
        """
        Initialize the Goal-Aware Retriever.

        Args:
            config: Retrieval configuration.
            vector_store: Vector database adapter.
            embedding_service: Embedding encoding service.
            event_bus: Optional event bus for retrieval events.
        """
        self.config = config
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.goal_encoder = GoalEncoder(embedding_service)
        self.ranker = MultiSignalRanker(config.ranking_weights)
        self.lifecycle = LifecycleController(
            config=MemoryEngineConfig(),
            event_bus=event_bus,
        )
        self.event_bus = event_bus or EventBus()

    def retrieve(
        self,
        query: str,
        goals: list[str] | None = None,
        top_k: int | None = None,
        include_stale: bool = False,
    ) -> list[RankedResult]:
        """
        Retrieve and rank memories using goal-aware multi-signal scoring.

        This is the primary retrieval interface for FAMM.

        Args:
            query: Natural language query.
            goals: Active agent goal descriptions.
            top_k: Number of results (default from config).
            include_stale: Whether to include STALE memories.

        Returns:
            List of RankedResult objects, sorted by score descending.
        """
        k_results = top_k or self.config.top_k_results
        k_candidates = self.config.top_k_candidates

        # Step 1: Encode query
        query_embedding = self.embedding_service.encode(query)

        # Step 2: Fetch candidates via vector similarity
        where_filter: dict[str, Any] | None = None
        if not include_stale:
            where_filter = {"state": MemoryState.ACTIVE.value}

        raw_candidates = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k_candidates,
            where=where_filter,
        )

        if not raw_candidates:
            return []

        # Step 3: Reconstruct MemoryRecords and extract similarities
        candidates: list[MemoryRecord] = []
        similarities: list[float] = []

        for raw in raw_candidates:
            record = self._raw_to_record(raw)
            if record is not None:
                candidates.append(record)
                # Convert distance to similarity
                # ChromaDB cosine distance: lower = more similar
                # Similarity = 1 - distance (for cosine)
                distance = raw.get("distance", 0.0)
                similarity = max(0.0, 1.0 - distance)
                similarities.append(similarity)

        if not candidates:
            return []

        # Step 4: Compute goal alignment for each candidate
        goals = goals or []
        if goals:
            goal_embeddings = self.goal_encoder.encode_goals(goals)
            goal_alignments = [
                self.goal_encoder.compute_alignment(
                    memory_embedding=c.embedding,
                    goal_embeddings=goal_embeddings,
                )
                for c in candidates
            ]
        else:
            # No goals: neutral alignment for all
            goal_alignments = [0.5] * len(candidates)

        # Step 5: Multi-signal ranking
        ranked_results = self.ranker.rank(
            candidates=candidates,
            similarities=similarities,
            goal_alignments=goal_alignments,
            top_k=k_results,
        )

        # Step 6: Record access and reactivate
        for result in ranked_results:
            result.memory.record_access()
            self.lifecycle.reactivate_on_access(result.memory)

            # Update store with new access info
            self.vector_store.update(
                ids=[result.memory.id],
                metadatas=[result.memory.to_storage_dict()],
            )

            self.event_bus.publish(
                EventType.MEMORY_ACCESSED,
                {
                    "memory_id": result.memory.id,
                    "query": query[:100],
                    "final_score": result.final_score,
                },
            )

        logger.info(
            "Goal-aware retrieval: query='%s...', goals=%d, candidates=%d, returned=%d",
            query[:40],
            len(goals),
            len(candidates),
            len(ranked_results),
        )

        return ranked_results

    def retrieve_simple(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RankedResult]:
        """
        Retrieve using similarity only (no goal awareness).

        Used as a baseline comparison in ablation studies.

        Args:
            query: Natural language query.
            top_k: Number of results.

        Returns:
            List of RankedResult (ranked by similarity only).
        """
        return self.retrieve(query=query, goals=None, top_k=top_k)

    @staticmethod
    def _raw_to_record(raw: dict[str, Any]) -> MemoryRecord | None:
        """Convert a raw vector store result to a MemoryRecord."""
        try:
            metadata = raw.get("metadata", {})
            embedding = raw.get("embedding", [])
            return MemoryRecord.from_storage_dict(metadata, embedding=embedding)
        except Exception:
            logger.exception("Failed to reconstruct MemoryRecord")
            return None
