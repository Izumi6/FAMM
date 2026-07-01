"""
Unit tests for EventBus — Decoupled communication validation.

Tests cover:
- Subscribing and publishing events
- Multiple handlers for same event
- Unsubscribing handlers
- Handler exceptions don't break the bus
- Event counting
"""

import pytest

from backend.memory_engine.event_bus import EventBus, EventType


class TestEventBusSubscription:
    """Test subscribe/unsubscribe mechanics."""

    def test_subscribe_and_publish(self) -> None:
        """A subscribed handler should receive published events."""
        bus = EventBus()
        received: list[dict] = []

        def handler(event_type: EventType, data: dict) -> None:
            received.append(data)

        bus.subscribe(EventType.MEMORY_CREATED, handler)
        bus.publish(EventType.MEMORY_CREATED, {"id": "test123"})

        assert len(received) == 1
        assert received[0]["id"] == "test123"

    def test_multiple_handlers(self) -> None:
        """Multiple handlers for same event should all be called."""
        bus = EventBus()
        results: list[str] = []

        bus.subscribe(EventType.MEMORY_CREATED, lambda e, d: results.append("handler_a"))
        bus.subscribe(EventType.MEMORY_CREATED, lambda e, d: results.append("handler_b"))

        bus.publish(EventType.MEMORY_CREATED)

        assert results == ["handler_a", "handler_b"]

    def test_unsubscribe(self) -> None:
        """Unsubscribed handler should not receive events."""
        bus = EventBus()
        count = {"calls": 0}

        def handler(event_type: EventType, data: dict) -> None:
            count["calls"] += 1

        bus.subscribe(EventType.MEMORY_ACCESSED, handler)
        bus.publish(EventType.MEMORY_ACCESSED)
        assert count["calls"] == 1

        bus.unsubscribe(EventType.MEMORY_ACCESSED, handler)
        bus.publish(EventType.MEMORY_ACCESSED)
        assert count["calls"] == 1  # No additional call

    def test_unsubscribe_unknown_handler_raises(self) -> None:
        """Unsubscribing an unknown handler should raise ValueError."""
        bus = EventBus()

        def handler(event_type: EventType, data: dict) -> None:
            pass

        with pytest.raises(ValueError):
            bus.unsubscribe(EventType.MEMORY_CREATED, handler)


class TestEventBusResilience:
    """Test error handling and edge cases."""

    def test_handler_exception_does_not_break_bus(self) -> None:
        """A failing handler should not prevent other handlers from running."""
        bus = EventBus()
        results: list[str] = []

        def failing_handler(event_type: EventType, data: dict) -> None:
            raise RuntimeError("Intentional test failure")

        def working_handler(event_type: EventType, data: dict) -> None:
            results.append("success")

        bus.subscribe(EventType.MEMORY_CREATED, failing_handler)
        bus.subscribe(EventType.MEMORY_CREATED, working_handler)

        # Should not raise
        bus.publish(EventType.MEMORY_CREATED)

        assert results == ["success"]

    def test_publish_with_no_handlers(self) -> None:
        """Publishing to an event with no handlers should not raise."""
        bus = EventBus()
        bus.publish(EventType.MEMORY_CREATED, {"id": "test"})
        # Should complete without error

    def test_event_count(self) -> None:
        """total_events_published should track correctly."""
        bus = EventBus()

        bus.publish(EventType.MEMORY_CREATED)
        bus.publish(EventType.MEMORY_ACCESSED)
        bus.publish(EventType.MEMORY_DECAYED)

        assert bus.total_events_published == 3

    def test_clear(self) -> None:
        """clear() should remove all handlers."""
        bus = EventBus()
        bus.subscribe(EventType.MEMORY_CREATED, lambda e, d: None)
        bus.subscribe(EventType.MEMORY_ACCESSED, lambda e, d: None)

        bus.clear()

        assert bus.handler_count == {}

    def test_handler_count(self) -> None:
        """handler_count should report correct counts per event type."""
        bus = EventBus()
        bus.subscribe(EventType.MEMORY_CREATED, lambda e, d: None)
        bus.subscribe(EventType.MEMORY_CREATED, lambda e, d: None)
        bus.subscribe(EventType.MEMORY_ACCESSED, lambda e, d: None)

        counts = bus.handler_count
        assert counts["memory_created"] == 2
        assert counts["memory_accessed"] == 1
