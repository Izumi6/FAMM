"""
MemoryManager — Central orchestrator for all FAMM memory operations.

The Memory Manager is the primary interface that an LLM agent uses to
interact with the FAMM framework. It coordinates:

1. Write Path:  content → embedding → utility scoring → storage
2. Read Path:   query → vector search → ranking → retrieval
3. Maintenance: decay cycles, lifecycle transitions, consolidation triggers

It delegates to specialized modules (Future Utility Predictor, Goal-Aware
Retriever, Forgetting Engine, Consolidator) via composition, not inheritance.
This is critical for ablation studies — any module can be disabled by
passing None.

Design Decisions:
- The manager owns the interaction step counter, which drives
  periodic maintenance operations (decay, consolidation).
- All operations emit events via the EventBus for observability
  and module decoupling.
- The manager does NOT own the LLM — it is a pure memory system.
  The agent (or evaluation harness) calls the manager's API.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.memory_engine.event_bus import EventBus, EventType
from backend.memory_engine.lifecycle_controller import LifecycleController
from backend.memory_engine.memory_record import MemoryRecord, MemoryState, SourceType
from backend.vector_database.base_adapter import VectorStoreAdapter
from backend.vector_database.embedding_service import EmbeddingService
from config.settings import FAMMConfig

logger = logging.getLogger(__name__)


class MemoryManager:
    """
    Central controller for all memory lifecycle operations in FAMM.

    Provides the public API for:
    - store(): Create and persist a new memory
    - retrieve(): Query memories (delegates to retriever if available)
    - get_by_id(): Direct lookup by memory ID
    - reinforce(): Boost a memory's utility after successful use
    - step(): Advance the interaction counter and trigger maintenance
    - get_stats(): Return memory store statistics

    Attributes:
        config: Full FAMM configuration.
        vector_store: Vector database adapter (ChromaDB or FAISS).
        embedding_service: Text-to-vector encoding service.
        event_bus: Pub/sub event bus.
        lifecycle: Memory lifecycle state controller.
        utility_predictor: Optional Future Utility Predictor module.
        step_count: Number of agent interaction steps processed.
    """

    def __init__(
        self,
        config: FAMMConfig,
        vector_store: VectorStoreAdapter,
        embedding_service: EmbeddingService,
        event_bus: EventBus | None = None,
        utility_predictor: Any = None,
    ) -> None:
        """
        Initialize the Memory Manager.

        Args:
            config: Full FAMM configuration.
            vector_store: Concrete vector store adapter.
            embedding_service: Embedding encoding service.
            event_bus: Optional event bus for inter-module communication.
            utility_predictor: Optional FutureUtilityPredictor instance.
                               If None, memories are stored with default utility.
        """
        self.config = config
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.event_bus = event_bus or EventBus()
        self.lifecycle = LifecycleController(
            config=config.memory_engine,
            event_bus=self.event_bus,
        )
        self.utility_predictor = utility_predictor

        self.step_count: int = 0

        # In-memory index for fast ID lookups without hitting vector DB
        self._memory_cache: dict[str, MemoryRecord] = {}

        logger.info(
            "MemoryManager initialized. Backend: %s, Utility predictor: %s",
            type(vector_store).__name__,
            "enabled" if utility_predictor else "disabled (default scoring)",
        )

    # ─────────────────────────────────────────────
    # Write Path
    # ─────────────────────────────────────────────

    def store(
        self,
        content: str,
        source_type: SourceType = SourceType.CONVERSATION,
        goal_tags: list[str] | None = None,
        goal_context: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """
        Create and persist a new memory.

        This is the primary write path:
        1. Create MemoryRecord
        2. Generate embedding via EmbeddingService
        3. Score utility via FutureUtilityPredictor (if available)
        4. Persist to vector store

        Args:
            content: Raw text content of the memory.
            source_type: How this memory was created.
            goal_tags: Goal identifiers to associate with this memory.
            goal_context: Current active goals (for utility prediction).
            metadata: Additional key-value metadata.

        Returns:
            The created and persisted MemoryRecord.
        """
        # 1. Create record
        record = MemoryRecord(
            content=content,
            source_type=source_type,
            goal_tags=goal_tags or [],
            metadata=metadata or {},
        )

        # 2. Generate embedding
        record.embedding = self.embedding_service.encode(content)

        # 3. Score utility (if predictor is available)
        if self.utility_predictor is not None:
            utility_score = self.utility_predictor.predict(
                memory=record,
                goal_context=goal_context or [],
            )
            record.utility_score = utility_score
            # Set adaptive decay rate: inversely proportional to utility
            exponent = self.config.forgetting_engine.utility_exponent
            base_rate = self.config.forgetting_engine.base_decay_rate
            record.decay_rate = base_rate * ((1.0 - record.utility_score) ** exponent)
        else:
            # Default scoring without predictor
            record.utility_score = 0.5
            record.decay_rate = self.config.forgetting_engine.base_decay_rate

        # 4. Persist to vector store
        storage_metadata = record.to_storage_dict()
        self.vector_store.add(
            ids=[record.id],
            embeddings=[record.embedding],
            documents=[record.content],
            metadatas=[storage_metadata],
        )

        # 5. Cache and emit event
        self._memory_cache[record.id] = record
        self.event_bus.publish(
            EventType.MEMORY_CREATED,
            {"memory_id": record.id, "utility_score": record.utility_score},
        )

        logger.info(
            "Stored memory %s (utility=%.3f, source=%s, goals=%s)",
            record.id[:8],
            record.utility_score,
            source_type.value,
            goal_tags or [],
        )

        return record

    # ─────────────────────────────────────────────
    # Read Path
    # ─────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        include_stale: bool = False,
    ) -> list[MemoryRecord]:
        """
        Retrieve memories by semantic similarity.

        This is the basic retrieval path. For goal-aware retrieval,
        use the GoalAwareRetriever module directly.

        Args:
            query: Natural language query string.
            top_k: Maximum results (default from config).
            include_stale: Whether to include STALE state memories.

        Returns:
            List of MemoryRecord instances, ranked by similarity.
        """
        k = top_k or self.config.goal_retrieval.top_k_results

        # Encode query
        query_embedding = self.embedding_service.encode(query)

        # Build filter
        where_filter: dict[str, Any] | None = None
        if not include_stale:
            where_filter = {"state": MemoryState.ACTIVE.value}

        # Query vector store
        raw_results = self.vector_store.query(
            query_embedding=query_embedding,
            top_k=k,
            where=where_filter,
        )

        # Reconstruct MemoryRecords
        records = []
        for result in raw_results:
            record = self._result_to_record(result)
            if record:
                # Record the access
                record.record_access()
                self._update_record_in_store(record)

                # Reactivate if stale/archived
                self.lifecycle.reactivate_on_access(record)

                records.append(record)

                self.event_bus.publish(
                    EventType.MEMORY_ACCESSED,
                    {"memory_id": record.id, "query": query[:100]},
                )

        logger.debug(
            "Retrieved %d memories for query: '%s...'",
            len(records),
            query[:50],
        )

        return records

    def get_by_id(self, memory_id: str) -> MemoryRecord | None:
        """
        Retrieve a specific memory by its ID.

        Checks the in-memory cache first, then falls back to vector store.

        Args:
            memory_id: UUID of the memory to retrieve.

        Returns:
            MemoryRecord if found, None otherwise.
        """
        # Check cache first
        if memory_id in self._memory_cache:
            return self._memory_cache[memory_id]

        # Fall back to vector store
        results = self.vector_store.get(ids=[memory_id])
        if results:
            record = self._result_to_record(results[0])
            if record:
                self._memory_cache[record.id] = record
            return record

        return None

    # ─────────────────────────────────────────────
    # Reinforcement
    # ─────────────────────────────────────────────

    def reinforce(self, memory_id: str, boost: float = 0.1) -> bool:
        """
        Reinforce a memory after it was successfully used.

        Called when the agent uses a retrieved memory in its response
        and the outcome was positive. Increases utility score to slow
        future decay.

        Args:
            memory_id: ID of the memory to reinforce.
            boost: Amount to increase utility (default 0.1).

        Returns:
            True if the memory was found and reinforced, False otherwise.
        """
        record = self.get_by_id(memory_id)
        if record is None:
            logger.warning("Cannot reinforce unknown memory: %s", memory_id[:8])
            return False

        old_utility = record.utility_score
        record.reinforce(boost=boost)

        # Update adaptive decay rate
        exponent = self.config.forgetting_engine.utility_exponent
        base_rate = self.config.forgetting_engine.base_decay_rate
        record.decay_rate = base_rate * ((1.0 - record.utility_score) ** exponent)

        self._update_record_in_store(record)

        self.event_bus.publish(
            EventType.MEMORY_REINFORCED,
            {
                "memory_id": memory_id,
                "old_utility": old_utility,
                "new_utility": record.utility_score,
            },
        )

        logger.info(
            "Reinforced memory %s: utility %.3f → %.3f",
            memory_id[:8],
            old_utility,
            record.utility_score,
        )

        return True

    # ─────────────────────────────────────────────
    # Maintenance (triggered by step())
    # ─────────────────────────────────────────────

    def step(self) -> dict[str, int]:
        """
        Advance the interaction counter and trigger maintenance.

        Should be called once per agent interaction step.
        Triggers decay and consolidation cycles based on configured intervals.

        Returns:
            Dict with counts of maintenance actions performed.
        """
        self.step_count += 1
        actions: dict[str, int] = {"decayed": 0, "pruned": 0, "transitioned": 0}

        # Check if decay cycle should run
        decay_interval = self.config.forgetting_engine.decay_interval_steps
        if self.step_count % decay_interval == 0:
            decay_result = self._run_decay_cycle()
            actions.update(decay_result)

        return actions

    def _run_decay_cycle(self) -> dict[str, int]:
        """
        Execute one decay cycle across all active memories.

        Applies adaptive decay to each memory's utility score,
        then evaluates lifecycle transitions.

        Returns:
            Counts of decayed, pruned, and transitioned memories.
        """
        result = {"decayed": 0, "pruned": 0, "transitioned": 0}

        all_records = self._get_all_active_records()

        for record in all_records:
            if record.state == MemoryState.DELETED:
                continue

            # Apply utility-conditioned decay
            exponent = self.config.forgetting_engine.utility_exponent
            base_rate = self.config.forgetting_engine.base_decay_rate
            effective_decay = base_rate * ((1.0 - record.utility_score) ** exponent)

            record.apply_decay(effective_decay)
            result["decayed"] += 1

            # Check for pruning
            if record.utility_score <= self.config.forgetting_engine.prune_threshold:
                record.transition_to(MemoryState.DELETED)
                self.vector_store.delete(ids=[record.id])
                self._memory_cache.pop(record.id, None)
                result["pruned"] += 1

                self.event_bus.publish(
                    EventType.MEMORY_PRUNED,
                    {"memory_id": record.id},
                )
            else:
                # Evaluate lifecycle transition
                if self.lifecycle.evaluate_and_apply(record):
                    result["transitioned"] += 1

                self._update_record_in_store(record)

        self.event_bus.publish(EventType.DECAY_CYCLE_TRIGGERED, result)

        logger.info(
            "Decay cycle complete (step %d): %d decayed, %d pruned, %d transitioned",
            self.step_count,
            result["decayed"],
            result["pruned"],
            result["transitioned"],
        )

        return result

    # ─────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """
        Return current memory store statistics.

        Useful for monitoring and experiment logging.

        Returns:
            Dict with counts, averages, and distributions.
        """
        total = self.vector_store.count()
        cached = len(self._memory_cache)

        # Compute utility distribution from cache
        utilities = [r.utility_score for r in self._memory_cache.values()]
        avg_utility = sum(utilities) / len(utilities) if utilities else 0.0

        state_counts: dict[str, int] = {}
        for record in self._memory_cache.values():
            state = record.state.value
            state_counts[state] = state_counts.get(state, 0) + 1

        return {
            "total_records": total,
            "cached_records": cached,
            "step_count": self.step_count,
            "average_utility": round(avg_utility, 4),
            "state_distribution": state_counts,
            "events_published": self.event_bus.total_events_published,
        }

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _result_to_record(self, result: dict[str, Any]) -> MemoryRecord | None:
        """Convert a vector store result dict to a MemoryRecord."""
        try:
            metadata = result.get("metadata", {})
            embedding = result.get("embedding", [])
            return MemoryRecord.from_storage_dict(metadata, embedding=embedding)
        except Exception:
            logger.exception("Failed to reconstruct MemoryRecord from result")
            return None

    def _update_record_in_store(self, record: MemoryRecord) -> None:
        """Persist updated metadata for a record to the vector store."""
        self.vector_store.update(
            ids=[record.id],
            metadatas=[record.to_storage_dict()],
        )
        self._memory_cache[record.id] = record

    def _get_all_active_records(self) -> list[MemoryRecord]:
        """
        Retrieve all non-deleted records from the store.

        For decay cycles, we need to iterate over all memories.
        This uses the cache where available and falls back to
        querying the vector store.
        """
        # For small stores, we can query with a zero vector to get all
        # For production, this would be paginated
        if self._memory_cache:
            return list(self._memory_cache.values())

        # Fallback: query with zero vector to get all records
        zero_vec = [0.0] * self.embedding_service.dimension
        results = self.vector_store.query(
            query_embedding=zero_vec,
            top_k=self.config.memory_engine.max_memories,
        )

        records = []
        for result in results:
            record = self._result_to_record(result)
            if record and record.state != MemoryState.DELETED:
                self._memory_cache[record.id] = record
                records.append(record)

        return records
