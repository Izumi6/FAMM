"""
Consolidator — Orchestrates the full memory consolidation pipeline.

Pipeline:
1. Policy selects candidates (moderate utility, not yet consolidated)
2. ClusterEngine groups candidates by semantic + goal similarity
3. Summarizer creates consolidated records from clusters
4. Old member records are archived/deleted
5. Consolidated records are stored via MemoryManager
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from backend.consolidation.cluster_engine import ClusterEngine
from backend.consolidation.consolidation_policy import ConsolidationPolicy
from backend.consolidation.summarizer import Summarizer
from backend.memory_engine.event_bus import EventBus, EventType
from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from backend.vector_database.base_adapter import VectorStoreAdapter
from backend.vector_database.embedding_service import EmbeddingService
from config.settings import ConsolidationConfig

logger = logging.getLogger(__name__)


class Consolidator:
    """
    Orchestrates the full consolidation pipeline.

    Attributes:
        config: Consolidation configuration.
        policy: Decides when/what to consolidate.
        cluster_engine: Groups related memories.
        summarizer: Creates consolidated records.
        vector_store: For storing consolidated and removing old records.
        event_bus: For consolidation events.
        consolidations_performed: Total consolidations since init.
    """

    def __init__(
        self,
        config: ConsolidationConfig,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreAdapter,
        event_bus: EventBus | None = None,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.event_bus = event_bus or EventBus()

        self.policy = ConsolidationPolicy(config)
        self.cluster_engine = ClusterEngine(
            embedding_service=embedding_service,
            similarity_threshold=config.cluster_similarity_threshold,
            min_cluster_size=config.min_cluster_size,
        )
        self.summarizer = Summarizer(config)

        self.consolidations_performed: int = 0

    def run(self, memories: list[MemoryRecord]) -> dict[str, Any]:
        """
        Execute one consolidation cycle.

        Args:
            memories: All available memories.

        Returns:
            Dict with consolidation statistics:
                - candidates: Number of eligible memories
                - clusters_found: Number of clusters formed
                - consolidated: Number of new consolidated records
                - removed: Number of old records archived
        """
        stats: dict[str, Any] = {
            "candidates": 0,
            "clusters_found": 0,
            "consolidated": 0,
            "removed": 0,
            "new_record_ids": [],
            "removed_record_ids": [],
        }

        # Step 1: Select candidates
        candidates = self.policy.select_candidates(memories)
        stats["candidates"] = len(candidates)

        if len(candidates) < self.config.min_cluster_size:
            logger.debug("Not enough candidates for consolidation (%d)", len(candidates))
            return stats

        # Step 2: Cluster candidates
        clusters = self.cluster_engine.cluster(candidates)
        stats["clusters_found"] = len(clusters)

        if not clusters:
            logger.debug("No clusters found above similarity threshold")
            return stats

        # Step 3: Process each cluster
        for cluster in clusters:
            group_id = str(uuid.uuid4())

            # Create consolidated record
            consolidated = self.summarizer.summarize_cluster(cluster, group_id)

            # Store consolidated record
            self.vector_store.add(
                ids=[consolidated.id],
                embeddings=[consolidated.embedding],
                documents=[consolidated.content],
                metadatas=[consolidated.to_storage_dict()],
            )
            stats["new_record_ids"].append(consolidated.id)
            stats["consolidated"] += 1

            # Archive/remove old member records
            for member in cluster:
                member.consolidation_group = group_id
                try:
                    member.transition_to(MemoryState.ARCHIVED)
                    self.vector_store.update(
                        ids=[member.id],
                        metadatas=[member.to_storage_dict()],
                    )
                except ValueError:
                    # If can't archive, delete
                    self.vector_store.delete(ids=[member.id])

                stats["removed"] += 1
                stats["removed_record_ids"].append(member.id)

            self.event_bus.publish(
                EventType.MEMORY_CONSOLIDATED,
                {
                    "group_id": group_id,
                    "cluster_size": len(cluster),
                    "consolidated_id": consolidated.id,
                },
            )

        self.consolidations_performed += 1

        logger.info(
            "Consolidation complete: %d candidates → %d clusters → %d consolidated, %d archived",
            stats["candidates"],
            stats["clusters_found"],
            stats["consolidated"],
            stats["removed"],
        )

        return stats
