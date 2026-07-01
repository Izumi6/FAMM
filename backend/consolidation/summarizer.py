"""
Summarizer — Summarizes memory clusters into consolidated records.

Supports two modes:
1. Extractive: Selects the most representative memory from the cluster
   (no LLM required, fully deterministic).
2. LLM: Uses an LLM to generate a summary (higher quality, requires Ollama).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.memory_engine.memory_record import MemoryRecord, SourceType
from config.settings import ConsolidationConfig

logger = logging.getLogger(__name__)


class Summarizer:
    """
    Creates consolidated memory records from clusters.

    Attributes:
        config: Consolidation configuration.
    """

    def __init__(self, config: ConsolidationConfig | None = None) -> None:
        self.config = config or ConsolidationConfig()

    def summarize_cluster(
        self,
        cluster: list[MemoryRecord],
        group_id: str,
    ) -> MemoryRecord:
        """
        Summarize a cluster of memories into a single consolidated record.

        Args:
            cluster: List of related memories to consolidate.
            group_id: Unique identifier for this consolidation group.

        Returns:
            A new consolidated MemoryRecord.
        """
        if self.config.summarization_mode == "llm":
            return self._summarize_llm(cluster, group_id)
        else:
            return self._summarize_extractive(cluster, group_id)

    def _summarize_extractive(
        self,
        cluster: list[MemoryRecord],
        group_id: str,
    ) -> MemoryRecord:
        """
        Extractive summarization: select the most central memory.

        The most central memory is the one whose embedding is closest
        to the centroid of the cluster.

        Also merges goal tags and computes the consolidated utility
        as the max of member utilities (preserving the highest value).
        """
        # Compute centroid
        embeddings = [m.embedding for m in cluster if m.embedding]
        if not embeddings:
            # Fallback: use highest-utility memory
            best = max(cluster, key=lambda m: m.utility_score)
            return self._create_consolidated(best.content, cluster, group_id, best.embedding)

        centroid = np.mean(embeddings, axis=0).tolist()

        # Find memory closest to centroid
        best_idx = 0
        best_sim = -1.0
        for i, emb in enumerate(embeddings):
            sim = float(np.dot(emb, centroid))
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        # Build consolidated content with context from other members
        central = cluster[best_idx]
        other_contents = [m.content for m in cluster if m.id != central.id]

        if other_contents:
            summary_content = (
                f"{central.content} "
                f"[Consolidated from {len(cluster)} related memories: "
                f"{'; '.join(c[:60] for c in other_contents[:3])}]"
            )
        else:
            summary_content = central.content

        return self._create_consolidated(summary_content, cluster, group_id, centroid)

    def _summarize_llm(
        self,
        cluster: list[MemoryRecord],
        group_id: str,
    ) -> MemoryRecord:
        """
        LLM-based summarization using Ollama.

        Falls back to extractive if LLM is unavailable.
        """
        try:
            import ollama

            contents = [m.content for m in cluster]
            prompt = (
                "Summarize the following related pieces of information into a single "
                "concise memory. Preserve all key facts, names, and relationships.\n\n"
                + "\n".join(f"- {c}" for c in contents)
            )

            response = ollama.chat(
                model=self.config.llm.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            summary_content = response["message"]["content"].strip()

            # Use centroid embedding
            embeddings = [m.embedding for m in cluster if m.embedding]
            centroid = np.mean(embeddings, axis=0).tolist() if embeddings else []

            return self._create_consolidated(summary_content, cluster, group_id, centroid)

        except Exception:
            logger.warning("LLM summarization failed, falling back to extractive")
            return self._summarize_extractive(cluster, group_id)

    def _create_consolidated(
        self,
        content: str,
        cluster: list[MemoryRecord],
        group_id: str,
        embedding: list[float],
    ) -> MemoryRecord:
        """Create a consolidated MemoryRecord from cluster metadata."""
        # Merge goal tags from all members
        all_goals: set[str] = set()
        for m in cluster:
            all_goals.update(m.goal_tags)

        # Utility: max of members (preserve highest-value signal)
        max_utility = max(m.utility_score for m in cluster)

        return MemoryRecord(
            content=content,
            embedding=embedding,
            source_type=SourceType.CONSOLIDATED,
            utility_score=max_utility,
            goal_tags=sorted(all_goals),
            consolidation_group=group_id,
            metadata={
                "consolidated_from": [m.id for m in cluster],
                "cluster_size": len(cluster),
            },
        )
