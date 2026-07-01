"""
Cluster Engine — Groups related memories for consolidation.

Uses semantic similarity clustering to identify groups of memories
that can be merged into consolidated records, reducing memory
footprint while preserving information.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np

from backend.memory_engine.memory_record import MemoryRecord
from backend.vector_database.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class ClusterEngine:
    """
    Groups memories by semantic similarity + goal overlap for consolidation.

    Uses a simple agglomerative approach: pairwise similarity above
    a threshold puts memories in the same cluster.

    Attributes:
        embedding_service: For computing inter-memory similarity.
        similarity_threshold: Minimum similarity for same-cluster membership.
        min_cluster_size: Minimum memories in a cluster for consolidation.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        similarity_threshold: float = 0.75,
        min_cluster_size: int = 3,
    ) -> None:
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size

    def cluster(self, memories: list[MemoryRecord]) -> list[list[MemoryRecord]]:
        """
        Group memories into clusters based on semantic similarity.

        Uses Union-Find for efficient cluster formation.

        Args:
            memories: List of memories to cluster.

        Returns:
            List of clusters, where each cluster is a list of MemoryRecords.
            Only clusters with ≥ min_cluster_size members are returned.
        """
        if len(memories) < self.min_cluster_size:
            return []

        n = len(memories)
        embeddings = [m.embedding for m in memories]

        # Filter out memories without embeddings
        valid_indices = [i for i, e in enumerate(embeddings) if e]
        if len(valid_indices) < self.min_cluster_size:
            return []

        valid_memories = [memories[i] for i in valid_indices]
        valid_embeddings = [embeddings[i] for i in valid_indices]

        # Union-Find
        parent = list(range(len(valid_memories)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Compute pairwise similarities and merge similar memories
        emb_matrix = np.array(valid_embeddings, dtype=np.float32)

        for i, j in combinations(range(len(valid_memories)), 2):
            sim = float(np.dot(emb_matrix[i], emb_matrix[j]))

            # Also consider goal tag overlap
            goal_overlap = self._goal_overlap(valid_memories[i], valid_memories[j])

            # Combined score: 70% semantic, 30% goal overlap
            combined = 0.7 * sim + 0.3 * goal_overlap

            if combined >= self.similarity_threshold:
                union(i, j)

        # Group by cluster root
        clusters: dict[int, list[MemoryRecord]] = {}
        for i, memory in enumerate(valid_memories):
            root = find(i)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(memory)

        # Filter by minimum size
        result = [c for c in clusters.values() if len(c) >= self.min_cluster_size]

        logger.info(
            "Clustering: %d memories → %d clusters (threshold=%.2f)",
            len(valid_memories),
            len(result),
            self.similarity_threshold,
        )

        return result

    @staticmethod
    def _goal_overlap(a: MemoryRecord, b: MemoryRecord) -> float:
        """Compute Jaccard similarity between goal tag sets."""
        tags_a = set(a.goal_tags)
        tags_b = set(b.goal_tags)
        if not tags_a and not tags_b:
            return 0.0
        union = tags_a | tags_b
        if not union:
            return 0.0
        return len(tags_a & tags_b) / len(union)
