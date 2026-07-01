"""
Unit tests for Multi-Signal Ranker and Consolidation Policy.

Tests cover:
- Multi-signal ranking with weighted combination
- Score breakdown correctness
- Consolidation candidate selection
"""

import pytest

from backend.consolidation.consolidation_policy import ConsolidationPolicy
from backend.goal_retrieval.multi_signal_ranker import MultiSignalRanker, RankedResult
from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from config.settings import ConsolidationConfig, RankingWeightsConfig


class TestMultiSignalRanker:
    """Test the multi-signal ranking algorithm."""

    def test_ranking_order(self) -> None:
        """Memories with higher combined scores should rank higher."""
        ranker = MultiSignalRanker()

        candidates = [
            MemoryRecord(content="Low relevance", utility_score=0.2),
            MemoryRecord(content="High relevance", utility_score=0.9),
            MemoryRecord(content="Medium relevance", utility_score=0.5),
        ]

        # Simulate: high similarity → candidate 1, low → others
        similarities = [0.3, 0.9, 0.5]
        alignments = [0.2, 0.8, 0.4]

        results = ranker.rank(candidates, similarities, alignments, top_k=3)

        assert len(results) == 3
        # Highest combined score should be first
        assert results[0].memory.content == "High relevance"

    def test_score_breakdown_present(self) -> None:
        """Each result should have a score breakdown."""
        ranker = MultiSignalRanker()

        candidates = [MemoryRecord(content="Test", utility_score=0.5)]
        results = ranker.rank(candidates, [0.7], [0.6], top_k=1)

        assert len(results) == 1
        assert "weighted_similarity" in results[0].score_breakdown
        assert "weighted_utility" in results[0].score_breakdown
        assert "weighted_alignment" in results[0].score_breakdown
        assert "weighted_recency" in results[0].score_breakdown

    def test_top_k_limits_results(self) -> None:
        """Should return at most top_k results."""
        ranker = MultiSignalRanker()

        candidates = [MemoryRecord(content=f"Mem {i}", utility_score=0.5) for i in range(10)]
        similarities = [0.5] * 10
        alignments = [0.5] * 10

        results = ranker.rank(candidates, similarities, alignments, top_k=3)
        assert len(results) == 3

    def test_empty_candidates(self) -> None:
        """Empty candidate list should return empty results."""
        ranker = MultiSignalRanker()
        results = ranker.rank([], [], [], top_k=5)
        assert results == []

    def test_mismatched_lengths_raises(self) -> None:
        """Mismatched input lengths should raise ValueError."""
        ranker = MultiSignalRanker()
        candidates = [MemoryRecord(content="Test")]

        with pytest.raises(ValueError, match="Length mismatch"):
            ranker.rank(candidates, [0.5, 0.3], [0.5], top_k=1)


class TestConsolidationPolicy:
    """Test consolidation candidate selection."""

    def test_selects_moderate_utility(self) -> None:
        """Only moderate-utility memories should be candidates."""
        policy = ConsolidationPolicy()

        memories = [
            MemoryRecord(content="High utility", utility_score=0.9),
            MemoryRecord(content="Moderate utility", utility_score=0.4),
            MemoryRecord(content="Low utility", utility_score=0.05),
            MemoryRecord(content="Also moderate", utility_score=0.3),
        ]

        candidates = policy.select_candidates(memories)

        # Only the moderate ones (0.1 ≤ utility ≤ 0.6) should be selected
        assert len(candidates) == 2
        contents = [c.content for c in candidates]
        assert "Moderate utility" in contents
        assert "Also moderate" in contents

    def test_excludes_already_consolidated(self) -> None:
        """Already-consolidated memories should not be candidates."""
        policy = ConsolidationPolicy()

        m1 = MemoryRecord(content="Unconsolidated", utility_score=0.4)
        m2 = MemoryRecord(content="Already grouped", utility_score=0.4)
        m2.consolidation_group = "group_123"

        candidates = policy.select_candidates([m1, m2])
        assert len(candidates) == 1
        assert candidates[0].content == "Unconsolidated"

    def test_excludes_deleted_memories(self) -> None:
        """Deleted memories should not be candidates."""
        policy = ConsolidationPolicy()

        m1 = MemoryRecord(content="Active", utility_score=0.4)
        m2 = MemoryRecord(content="Deleted", utility_score=0.4)
        m2.state = MemoryState.DELETED

        candidates = policy.select_candidates([m1, m2])
        assert len(candidates) == 1

    def test_should_consolidate_at_interval(self) -> None:
        """Should return True at the configured interval."""
        config = ConsolidationConfig(consolidation_interval_steps=50)
        policy = ConsolidationPolicy(config)

        assert policy.should_consolidate(0) is False
        assert policy.should_consolidate(49) is False
        assert policy.should_consolidate(50) is True
        assert policy.should_consolidate(100) is True
