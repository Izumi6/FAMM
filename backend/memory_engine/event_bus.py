"""
Event Bus — Decoupled inter-module communication for FAMM.

The event bus enables modules to communicate without direct dependencies:
- Memory Engine publishes MEMORY_CREATED, MEMORY_ACCESSED events
- Forgetting Engine subscribes to MEMORY_CREATED to initialize decay tracking
- Future Utility Predictor subscribes to MEMORY_ACCESSED to update training data
- Consolidator subscribes to periodic CONSOLIDATION_TRIGGERED events

This follows the Observer pattern and keeps modules loosely coupled,
which is essential for ablation studies where we need to swap or
disable individual modules.

Design Decision:
- Synchronous event dispatch (not async) for simplicity and determinism.
- Events carry the full MemoryRecord as payload to avoid additional lookups.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """
    Types of events in the FAMM memory lifecycle.

    Each event corresponds to a significant memory operation
    that other modules may need to react to.
    """

    # Memory lifecycle events
    MEMORY_CREATED = "memory_created"
    MEMORY_ACCESSED = "memory_accessed"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_REINFORCED = "memory_reinforced"
    MEMORY_DECAYED = "memory_decayed"
    MEMORY_STATE_CHANGED = "memory_state_changed"
    MEMORY_PRUNED = "memory_pruned"
    MEMORY_CONSOLIDATED = "memory_consolidated"

    # System events
    DECAY_CYCLE_TRIGGERED = "decay_cycle_triggered"
    CONSOLIDATION_TRIGGERED = "consolidation_triggered"
    GOAL_UPDATED = "goal_updated"


# Type alias for event handler callbacks
EventHandler = Callable[[EventType, dict[str, Any]], None]


class EventBus:
    """
    Simple synchronous publish-subscribe event bus.

    Enables decoupled communication between FAMM modules.
    All handlers for a given event type are called synchronously
    in the order they were registered.

    Thread Safety:
        This implementation is NOT thread-safe. For multi-threaded
        scenarios, add locking around _handlers access.

    Example:
        >>> bus = EventBus()
        >>> bus.subscribe(EventType.MEMORY_CREATED, lambda e, d: print(f"Created: {d}"))
        >>> bus.publish(EventType.MEMORY_CREATED, {"memory_id": "abc123"})
        Created: {'memory_id': 'abc123'}
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[EventHandler]] = defaultdict(list)
        self._event_count: int = 0

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Register a handler for a specific event type.

        Args:
            event_type: The event type to listen for.
            handler: Callable that takes (EventType, data_dict) as arguments.
        """
        self._handlers[event_type].append(handler)
        logger.debug(
            "Handler %s subscribed to %s", handler.__name__, event_type.value
        )

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """
        Remove a handler for a specific event type.

        Args:
            event_type: The event type to unsubscribe from.
            handler: The handler to remove.

        Raises:
            ValueError: If the handler was not registered.
        """
        try:
            self._handlers[event_type].remove(handler)
            logger.debug(
                "Handler %s unsubscribed from %s", handler.__name__, event_type.value
            )
        except ValueError:
            raise ValueError(
                f"Handler {handler.__name__} is not subscribed to {event_type.value}"
            )

    def publish(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """
        Publish an event to all registered handlers.

        Handlers are called synchronously in registration order.
        If a handler raises an exception, it is logged but does not
        prevent subsequent handlers from executing.

        Args:
            event_type: The event to publish.
            data: Optional payload dictionary.
        """
        payload = data or {}
        self._event_count += 1

        handlers = self._handlers.get(event_type, [])
        if not handlers:
            logger.debug("No handlers for event %s", event_type.value)
            return

        logger.debug(
            "Publishing %s to %d handler(s)", event_type.value, len(handlers)
        )

        for handler in handlers:
            try:
                handler(event_type, payload)
            except Exception:
                logger.exception(
                    "Handler %s failed for event %s",
                    handler.__name__,
                    event_type.value,
                )

    def clear(self) -> None:
        """Remove all handlers for all event types."""
        self._handlers.clear()
        logger.debug("Event bus cleared")

    @property
    def total_events_published(self) -> int:
        """Return the total number of events published since creation."""
        return self._event_count

    @property
    def handler_count(self) -> dict[str, int]:
        """Return count of handlers per event type."""
        return {
            event_type.value: len(handlers)
            for event_type, handlers in self._handlers.items()
            if handlers
        }
