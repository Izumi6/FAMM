"""
Unit tests for LifecycleController.

Tests cover:
- State evaluation logic (utility thresholds + age)
- Valid and invalid transitions
- Reactivation on access
- Event bus integration
"""

from datetime import datetime, timedelta, timezone

import pytest

from backend.memory_engine.event_bus import EventBus, EventType
from backend.memory_engine.lifecycle_controller import LifecycleController
from backend.memory_engine.memory_record import MemoryRecord, MemoryState
from config.settings import MemoryEngineConfig


class TestLifecycleEvaluation:
    """Test the evaluate() method's decision logic."""

    def setup_method(self) -> None:
        """Create a controller with default config."""
        self.controller = LifecycleController(config=MemoryEngineConfig())

    def test_healthy_memory_no_transition(self) -> None:
        """Active memory with good utility should not transition."""
        record = MemoryRecord(content="Healthy memory", utility_score=0.7)
        result = self.controller.evaluate(record)
        assert result is None

    def test_very_low_utility_triggers_deletion(self) -> None:
        """Memory with near-zero utility should be marked for deletion."""
        record = MemoryRecord(content="Dying memory", utility_score=0.005)
        result = self.controller.evaluate(record)
        assert result == MemoryState.DELETED

    def test_low_utility_old_memory_goes_stale(self) -> None:
        """Active memory with low utility and past stale threshold → STALE."""
        record = MemoryRecord(
            content="Aging memory",
            utility_score=0.2,
        )
        # Simulate aging by backdating creation
        record.created_at = datetime.now(timezone.utc) - timedelta(days=35)

        result = self.controller.evaluate(record)
        assert result == MemoryState.STALE

    def test_very_low_utility_very_old_goes_archived(self) -> None:
        """Low utility + past archive threshold → ARCHIVED."""
        record = MemoryRecord(
            content="Ancient memory",
            utility_score=0.05,
        )
        record.created_at = datetime.now(timezone.utc) - timedelta(days=100)

        result = self.controller.evaluate(record)
        assert result == MemoryState.ARCHIVED

    def test_stale_memory_with_recovered_utility_reactivates(self) -> None:
        """STALE memory with utility ≥ 0.3 should be recommended for ACTIVE."""
        record = MemoryRecord(
            content="Recovering memory",
            utility_score=0.4,
        )
        record.state = MemoryState.STALE

        result = self.controller.evaluate(record)
        assert result == MemoryState.ACTIVE

    def test_deleted_memory_no_transition(self) -> None:
        """Deleted memory should never transition."""
        record = MemoryRecord(content="Dead memory", utility_score=0.0)
        record.state = MemoryState.DELETED

        result = self.controller.evaluate(record)
        assert result is None


class TestLifecycleApplication:
    """Test apply_transition() and evaluate_and_apply()."""

    def test_apply_transition_updates_state(self) -> None:
        """apply_transition() should change the memory's state."""
        controller = LifecycleController()
        record = MemoryRecord(content="Test")

        controller.apply_transition(record, MemoryState.STALE)
        assert record.state == MemoryState.STALE

    def test_apply_transition_publishes_event(self) -> None:
        """apply_transition() should publish a state change event."""
        bus = EventBus()
        events_received: list[dict] = []

        def handler(event_type: EventType, data: dict) -> None:
            events_received.append(data)

        bus.subscribe(EventType.MEMORY_STATE_CHANGED, handler)
        controller = LifecycleController(event_bus=bus)
        record = MemoryRecord(content="Test")

        controller.apply_transition(record, MemoryState.STALE)

        assert len(events_received) == 1
        assert events_received[0]["old_state"] == "active"
        assert events_received[0]["new_state"] == "stale"

    def test_evaluate_and_apply_returns_true_on_transition(self) -> None:
        """evaluate_and_apply() should return True if a transition occurs."""
        controller = LifecycleController()
        record = MemoryRecord(content="Low utility", utility_score=0.005)

        result = controller.evaluate_and_apply(record)
        assert result is True
        assert record.state == MemoryState.DELETED

    def test_evaluate_and_apply_returns_false_when_no_transition(self) -> None:
        """evaluate_and_apply() should return False if no transition needed."""
        controller = LifecycleController()
        record = MemoryRecord(content="Healthy", utility_score=0.8)

        result = controller.evaluate_and_apply(record)
        assert result is False
        assert record.state == MemoryState.ACTIVE


class TestReactivation:
    """Test reactivation on access."""

    def test_reactivate_stale_memory(self) -> None:
        """Stale memory should be reactivated on access."""
        controller = LifecycleController()
        record = MemoryRecord(content="Stale memory")
        record.state = MemoryState.STALE

        controller.reactivate_on_access(record)
        assert record.state == MemoryState.ACTIVE

    def test_reactivate_archived_memory(self) -> None:
        """Archived memory should be reactivated on access."""
        controller = LifecycleController()
        record = MemoryRecord(content="Archived memory")
        record.transition_to(MemoryState.ARCHIVED)

        controller.reactivate_on_access(record)
        assert record.state == MemoryState.ACTIVE

    def test_no_reactivation_for_active_memory(self) -> None:
        """Active memory should not be affected by reactivate_on_access."""
        controller = LifecycleController()
        record = MemoryRecord(content="Active memory")

        controller.reactivate_on_access(record)
        assert record.state == MemoryState.ACTIVE
