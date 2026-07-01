"""
Baseline Memory Systems for Experimental Comparison.

This module implements simplified versions of existing memory systems
that FAMM is compared against in the evaluation section of the paper.

Each baseline implements the same interface (store, retrieve, step)
so they can be swapped into the evaluation harness seamlessly.

Baselines:
1. NaiveFIFO: First-in, first-out eviction (simplest possible)
2. SimilarityOnly: Standard RAG (no utility, no goals, no decay)
3. ImportanceScoring: Importance + recency (like MemGPT)
4. EbbinghausDecay: Uniform temporal decay (like MemoryBank)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

import numpy as np

from backend.memory_engine.memory_record import MemoryRecord, MemoryState, SourceType
from backend.vector_database.base_adapter import VectorStoreAdapter
from backend.vector_database.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class BaselineMemorySystem(ABC):
    """
    Abstract interface for all baseline memory systems.

    All baselines must implement store(), retrieve(), and step()
    with the same signatures so the evaluation harness can swap them.
    """

    @abstractmethod
    def store(self, content: str, **kwargs: Any) -> MemoryRecord:
        """Store a new memory."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryRecord]:
        """Retrieve memories by query."""
        ...

    @abstractmethod
    def step(self) -> dict[str, int]:
        """Advance one interaction step (trigger maintenance)."""
        ...

    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """Return system statistics."""
        ...


class NaiveFIFOBaseline(BaselineMemorySystem):
    """
    First-in, first-out eviction baseline.

    The simplest possible memory system:
    - Stores memories in order
    - Retrieves by semantic similarity
    - Evicts the oldest memory when capacity is reached
    - No utility scoring, no decay, no goals

    This represents the floor performance — any research system
    should significantly outperform this.
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedding_service: EmbeddingService,
        max_memories: int = 1000,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.max_memories = max_memories
        self.step_count = 0
        self._insertion_order: list[str] = []  # Track insertion order for FIFO
        self._cache: dict[str, MemoryRecord] = {}

    def store(self, content: str, **kwargs: Any) -> MemoryRecord:
        record = MemoryRecord(
            content=content,
            source_type=kwargs.get("source_type", SourceType.CONVERSATION),
            utility_score=0.5,  # Fixed score — no prediction
        )
        record.embedding = self.embedding_service.encode(content)

        # Evict oldest if at capacity
        if len(self._insertion_order) >= self.max_memories:
            oldest_id = self._insertion_order.pop(0)
            self.vector_store.delete(ids=[oldest_id])
            self._cache.pop(oldest_id, None)

        self.vector_store.add(
            ids=[record.id],
            embeddings=[record.embedding],
            documents=[record.content],
            metadatas=[record.to_storage_dict()],
        )
        self._insertion_order.append(record.id)
        self._cache[record.id] = record

        return record

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryRecord]:
        query_embedding = self.embedding_service.encode(query)
        results = self.vector_store.query(query_embedding=query_embedding, top_k=top_k)

        records = []
        for r in results:
            try:
                record = MemoryRecord.from_storage_dict(
                    r.get("metadata", {}),
                    embedding=r.get("embedding"),
                )
                record.record_access()
                records.append(record)
            except Exception:
                continue

        return records

    def step(self) -> dict[str, int]:
        self.step_count += 1
        return {"decayed": 0, "pruned": 0, "transitioned": 0}

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_records": self.vector_store.count(),
            "step_count": self.step_count,
            "system": "NaiveFIFO",
        }


class SimilarityOnlyBaseline(BaselineMemorySystem):
    """
    Standard RAG baseline — retrieval by semantic similarity only.

    - Stores all memories (no eviction)
    - Retrieves by cosine similarity (standard vector search)
    - No utility scoring, no goals, no decay
    - No reranking

    This represents what most basic RAG systems do.
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedding_service: EmbeddingService,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.step_count = 0
        self._cache: dict[str, MemoryRecord] = {}

    def store(self, content: str, **kwargs: Any) -> MemoryRecord:
        record = MemoryRecord(
            content=content,
            source_type=kwargs.get("source_type", SourceType.CONVERSATION),
            utility_score=0.5,
        )
        record.embedding = self.embedding_service.encode(content)

        self.vector_store.add(
            ids=[record.id],
            embeddings=[record.embedding],
            documents=[record.content],
            metadatas=[record.to_storage_dict()],
        )
        self._cache[record.id] = record
        return record

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryRecord]:
        query_embedding = self.embedding_service.encode(query)
        results = self.vector_store.query(query_embedding=query_embedding, top_k=top_k)

        records = []
        for r in results:
            try:
                record = MemoryRecord.from_storage_dict(
                    r.get("metadata", {}),
                    embedding=r.get("embedding"),
                )
                records.append(record)
            except Exception:
                continue

        return records

    def step(self) -> dict[str, int]:
        self.step_count += 1
        return {"decayed": 0, "pruned": 0, "transitioned": 0}

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_records": self.vector_store.count(),
            "step_count": self.step_count,
            "system": "SimilarityOnly",
        }


class ImportanceScoringBaseline(BaselineMemorySystem):
    """
    Importance + recency scoring baseline (MemGPT-style).

    - Assigns importance at write time (simple LLM-free heuristic)
    - Retrieves by: α × similarity + (1-α) × importance
    - No goal awareness, no adaptive decay
    - Evicts lowest-importance memories at capacity

    This approximates MemGPT's memory management without the LLM calls.
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedding_service: EmbeddingService,
        max_memories: int = 1000,
        importance_weight: float = 0.3,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.max_memories = max_memories
        self.importance_weight = importance_weight
        self.step_count = 0
        self._cache: dict[str, MemoryRecord] = {}

    def store(self, content: str, **kwargs: Any) -> MemoryRecord:
        # Simple importance heuristic (length + question marks + entity count)
        importance = self._compute_importance(content)

        record = MemoryRecord(
            content=content,
            source_type=kwargs.get("source_type", SourceType.CONVERSATION),
            utility_score=importance,
        )
        record.embedding = self.embedding_service.encode(content)

        # Evict lowest-importance if at capacity
        if self.vector_store.count() >= self.max_memories:
            self._evict_lowest_importance()

        self.vector_store.add(
            ids=[record.id],
            embeddings=[record.embedding],
            documents=[record.content],
            metadatas=[record.to_storage_dict()],
        )
        self._cache[record.id] = record
        return record

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryRecord]:
        query_embedding = self.embedding_service.encode(query)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Fetch extra for reranking
        )

        # Rerank by importance-weighted score
        scored = []
        for r in results:
            try:
                record = MemoryRecord.from_storage_dict(
                    r.get("metadata", {}),
                    embedding=r.get("embedding"),
                )
                distance = r.get("distance", 0.0)
                similarity = max(0.0, 1.0 - distance)
                combined = (
                    (1 - self.importance_weight) * similarity
                    + self.importance_weight * record.utility_score
                )
                scored.append((combined, record))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in scored[:top_k]]

    def step(self) -> dict[str, int]:
        self.step_count += 1
        return {"decayed": 0, "pruned": 0, "transitioned": 0}

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_records": self.vector_store.count(),
            "step_count": self.step_count,
            "system": "ImportanceScoring",
        }

    @staticmethod
    def _compute_importance(content: str) -> float:
        """Simple importance heuristic (no LLM)."""
        score = 0.3  # Base importance

        # Longer content tends to be more informative
        word_count = len(content.split())
        score += min(0.2, word_count / 100)

        # Questions are important (they indicate user intent)
        if "?" in content:
            score += 0.15

        # Capitalized words (potential entities/proper nouns)
        import re
        entities = re.findall(r'\b[A-Z][a-z]+\b', content)
        score += min(0.15, len(entities) * 0.03)

        # Imperative phrases (commands, preferences)
        imperatives = ["must", "should", "need", "want", "prefer", "important"]
        if any(w in content.lower() for w in imperatives):
            score += 0.1

        return min(1.0, score)

    def _evict_lowest_importance(self) -> None:
        """Remove the memory with the lowest importance score."""
        if not self._cache:
            return

        lowest = min(self._cache.values(), key=lambda m: m.utility_score)
        self.vector_store.delete(ids=[lowest.id])
        self._cache.pop(lowest.id, None)


class EbbinghausDecayBaseline(BaselineMemorySystem):
    """
    Uniform temporal decay baseline (MemoryBank-style).

    - Applies the Ebbinghaus forgetting curve to all memories uniformly
    - Retrieves by similarity (no utility or goal weighting)
    - Prunes memories that fall below retention threshold
    - No adaptive decay, no goal awareness

    This is the primary comparison target: FAMM's utility-conditioned
    decay should outperform this uniform time-based decay.
    """

    def __init__(
        self,
        vector_store: VectorStoreAdapter,
        embedding_service: EmbeddingService,
        stability: float = 24.0,
        prune_threshold: float = 0.05,
        decay_interval: int = 10,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.stability = stability
        self.prune_threshold = prune_threshold
        self.decay_interval = decay_interval
        self.step_count = 0
        self._cache: dict[str, MemoryRecord] = {}

    def store(self, content: str, **kwargs: Any) -> MemoryRecord:
        record = MemoryRecord(
            content=content,
            source_type=kwargs.get("source_type", SourceType.CONVERSATION),
            utility_score=1.0,  # Start at max retention
        )
        record.embedding = self.embedding_service.encode(content)

        self.vector_store.add(
            ids=[record.id],
            embeddings=[record.embedding],
            documents=[record.content],
            metadatas=[record.to_storage_dict()],
        )
        self._cache[record.id] = record
        return record

    def retrieve(self, query: str, top_k: int = 10) -> list[MemoryRecord]:
        query_embedding = self.embedding_service.encode(query)
        results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where={"state": MemoryState.ACTIVE.value},
        )

        records = []
        for r in results:
            try:
                record = MemoryRecord.from_storage_dict(
                    r.get("metadata", {}),
                    embedding=r.get("embedding"),
                )
                record.record_access()
                records.append(record)
            except Exception:
                continue

        return records

    def step(self) -> dict[str, int]:
        self.step_count += 1
        stats = {"decayed": 0, "pruned": 0, "transitioned": 0}

        if self.step_count % self.decay_interval == 0:
            stats = self._run_ebbinghaus_decay()

        return stats

    def _run_ebbinghaus_decay(self) -> dict[str, int]:
        """Apply uniform Ebbinghaus decay to all memories."""
        import math

        stats = {"decayed": 0, "pruned": 0, "transitioned": 0}
        to_delete: list[str] = []

        for record in list(self._cache.values()):
            if record.state == MemoryState.DELETED:
                continue

            # Ebbinghaus: R(t) = e^{-t/S}
            age_hours = record.age_seconds() / 3600
            retention = math.exp(-age_hours / self.stability)
            record.utility_score = max(0.0, retention)
            stats["decayed"] += 1

            if retention <= self.prune_threshold:
                to_delete.append(record.id)
                stats["pruned"] += 1

        # Delete pruned memories
        if to_delete:
            self.vector_store.delete(ids=to_delete)
            for mid in to_delete:
                self._cache.pop(mid, None)

        return stats

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_records": self.vector_store.count(),
            "cached_records": len(self._cache),
            "step_count": self.step_count,
            "system": "EbbinghausDecay",
        }
