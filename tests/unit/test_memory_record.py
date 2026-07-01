"""
Unit tests for MemoryRecord — Core data structure validation.

Tests cover:
- Construction with defaults and custom values
- Field validation (Pydantic constraints)
- State transitions (valid and invalid)
- Access recording and reinforcement
- Decay application
- Serialization roundtrip (to_storage_dict / from_storage_dict)
"""

from datetime import datetime, timezone

import pytest

from backend.memory_engine.memory_record import MemoryRecord, MemoryState, SourceType


class TestMemoryRecordConstruction:
    """Test MemoryRecord creation and default values."""

    def test_create_with_defaults(self) -> None:
        """Memory created with minimal args should have valid defaults."""
        record = MemoryRecord(content="Test memory content")

        assert record.content == "Test memory content"
        assert record.embedding == []
        assert record.access_count == 0
        assert record.source_type == SourceType.CONVERSATION
        assert record.utility_score == 0.5
        assert record.goal_tags == []
        assert record.decay_rate == 0.05
        assert record.state == MemoryState.ACTIVE
        assert record.consolidation_group is None
        assert record.metadata == {}
        assert len(record.id) == 36  # UUID4 format

    def test_create_with_custom_values(self) -> None:
        """Memory created with explicit args should preserve them."""
        record = MemoryRecord(
            content="Custom memory",
            source_type=SourceType.REFLECTION,
            utility_score=0.8,
            goal_tags=["goal_a", "goal_b"],
            decay_rate=0.02,
        )

        assert record.source_type == SourceType.REFLECTION
        assert record.utility_score == 0.8
        assert record.goal_tags == ["goal_a", "goal_b"]
        assert record.decay_rate == 0.02

    def test_unique_ids(self) -> None:
        """Each MemoryRecord should get a unique ID."""
        records = [MemoryRecord(content=f"Memory {i}") for i in range(100)]
        ids = [r.id for r in records]
        assert len(set(ids)) == 100

    def test_timestamp_is_utc(self) -> None:
        """Timestamps should be in UTC."""
        record = MemoryRecord(content="Test")
        assert record.created_at.tzinfo is not None


class TestMemoryRecordValidation:
    """Test Pydantic field validation."""

    def test_empty_content_rejected(self) -> None:
        """Content must be non-empty."""
        with pytest.raises(Exception):
            MemoryRecord(content="")

    def test_utility_score_bounds(self) -> None:
        """Utility score must be in [0.0, 1.0]."""
        with pytest.raises(Exception):
            MemoryRecord(content="test", utility_score=1.5)

        with pytest.raises(Exception):
            MemoryRecord(content="test", utility_score=-0.1)

    def test_valid_utility_boundaries(self) -> None:
        """Utility score at boundaries should work."""
        record_zero = MemoryRecord(content="test", utility_score=0.0)
        record_one = MemoryRecord(content="test", utility_score=1.0)
        assert record_zero.utility_score == 0.0
        assert record_one.utility_score == 1.0

    def test_access_count_non_negative(self) -> None:
        """Access count must be >= 0."""
        with pytest.raises(Exception):
            MemoryRecord(content="test", access_count=-1)


class TestMemoryRecordMutations:
    """Test state mutation methods."""

    def test_record_access(self) -> None:
        """record_access() should update timestamp and increment count."""
        record = MemoryRecord(content="Test memory")
        old_timestamp = record.last_accessed_at

        # Small delay to ensure timestamp changes
        record.record_access()

        assert record.access_count == 1
        assert record.last_accessed_at >= old_timestamp

    def test_reinforce_increases_utility(self) -> None:
        """reinforce() should increase utility score."""
        record = MemoryRecord(content="Test", utility_score=0.5)
        record.reinforce(boost=0.2)

        assert record.utility_score == pytest.approx(0.7)
        assert record.access_count == 1  # reinforce calls record_access

    def test_reinforce_clamps_at_one(self) -> None:
        """reinforce() should not exceed 1.0."""
        record = MemoryRecord(content="Test", utility_score=0.95)
        record.reinforce(boost=0.2)

        assert record.utility_score == 1.0

    def test_apply_decay_decreases_utility(self) -> None:
        """apply_decay() should decrease utility score."""
        record = MemoryRecord(content="Test", utility_score=0.5)
        record.apply_decay(0.1)

        assert record.utility_score == pytest.approx(0.4)

    def test_apply_decay_clamps_at_zero(self) -> None:
        """apply_decay() should not go below 0.0."""
        record = MemoryRecord(content="Test", utility_score=0.03)
        record.apply_decay(0.1)

        assert record.utility_score == 0.0


class TestMemoryRecordStateTransitions:
    """Test lifecycle state transitions."""

    def test_active_to_stale(self) -> None:
        """ACTIVE → STALE should be valid."""
        record = MemoryRecord(content="Test")
        record.transition_to(MemoryState.STALE)
        assert record.state == MemoryState.STALE

    def test_active_to_archived(self) -> None:
        """ACTIVE → ARCHIVED should be valid."""
        record = MemoryRecord(content="Test")
        record.transition_to(MemoryState.ARCHIVED)
        assert record.state == MemoryState.ARCHIVED

    def test_active_to_deleted(self) -> None:
        """ACTIVE → DELETED should be valid."""
        record = MemoryRecord(content="Test")
        record.transition_to(MemoryState.DELETED)
        assert record.state == MemoryState.DELETED

    def test_stale_to_active_reactivation(self) -> None:
        """STALE → ACTIVE should be valid (reactivation on access)."""
        record = MemoryRecord(content="Test")
        record.transition_to(MemoryState.STALE)
        record.transition_to(MemoryState.ACTIVE)
        assert record.state == MemoryState.ACTIVE

    def test_archived_to_active_reactivation(self) -> None:
        """ARCHIVED → ACTIVE should be valid (reactivation on access)."""
        record = MemoryRecord(content="Test")
        record.transition_to(MemoryState.ARCHIVED)
        record.transition_to(MemoryState.ACTIVE)
        assert record.state == MemoryState.ACTIVE

    def test_deleted_is_terminal(self) -> None:
        """DELETED → anything should be invalid (terminal state)."""
        record = MemoryRecord(content="Test")
        record.transition_to(MemoryState.DELETED)

        with pytest.raises(ValueError, match="Invalid state transition"):
            record.transition_to(MemoryState.ACTIVE)

    def test_deleted_to_stale_invalid(self) -> None:
        """DELETED → STALE should be invalid."""
        record = MemoryRecord(content="Test")
        record.transition_to(MemoryState.DELETED)

        with pytest.raises(ValueError):
            record.transition_to(MemoryState.STALE)


class TestMemoryRecordSerialization:
    """Test serialization roundtrip for vector DB storage."""

    def test_to_storage_dict(self) -> None:
        """to_storage_dict() should produce a flat dictionary."""
        record = MemoryRecord(
            content="Test memory",
            source_type=SourceType.REFLECTION,
            utility_score=0.8,
            goal_tags=["goal_a", "goal_b"],
        )
        data = record.to_storage_dict()

        assert data["content"] == "Test memory"
        assert data["source_type"] == "reflection"
        assert data["utility_score"] == 0.8
        assert data["goal_tags"] == "goal_a,goal_b"
        assert data["state"] == "active"

    def test_serialization_roundtrip(self) -> None:
        """from_storage_dict(to_storage_dict()) should preserve data."""
        original = MemoryRecord(
            content="Roundtrip test",
            source_type=SourceType.OBSERVATION,
            utility_score=0.65,
            goal_tags=["planning", "research"],
            decay_rate=0.03,
        )
        embedding = [0.1, 0.2, 0.3]
        original.embedding = embedding

        data = original.to_storage_dict()
        restored = MemoryRecord.from_storage_dict(data, embedding=embedding)

        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.source_type == original.source_type
        assert restored.utility_score == pytest.approx(original.utility_score)
        assert restored.goal_tags == original.goal_tags
        assert restored.decay_rate == pytest.approx(original.decay_rate)
        assert restored.state == original.state
        assert restored.embedding == embedding

    def test_empty_goal_tags_roundtrip(self) -> None:
        """Empty goal_tags should survive serialization."""
        original = MemoryRecord(content="No goals", goal_tags=[])
        data = original.to_storage_dict()
        restored = MemoryRecord.from_storage_dict(data)

        assert restored.goal_tags == []

    def test_repr(self) -> None:
        """__repr__ should return a readable string."""
        record = MemoryRecord(content="A short memory", utility_score=0.75)
        repr_str = repr(record)

        assert "utility=0.75" in repr_str
        assert "state=active" in repr_str
        assert "A short memory" in repr_str
