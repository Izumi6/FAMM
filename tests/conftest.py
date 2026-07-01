"""
FAMM Test Configuration (conftest.py)

Shared fixtures for the test suite.
"""

import pytest

from backend.memory_engine.event_bus import EventBus
from backend.memory_engine.memory_record import MemoryRecord, SourceType
from config.settings import FAMMConfig


@pytest.fixture
def default_config() -> FAMMConfig:
    """Return a default FAMM configuration for testing."""
    return FAMMConfig.default()


@pytest.fixture
def event_bus() -> EventBus:
    """Return a fresh event bus for testing."""
    return EventBus()


@pytest.fixture
def sample_memory() -> MemoryRecord:
    """Return a sample memory record for testing."""
    return MemoryRecord(
        content="The user is working on a research paper about memory management.",
        source_type=SourceType.CONVERSATION,
        utility_score=0.7,
        goal_tags=["research", "writing"],
    )


@pytest.fixture
def sample_memories() -> list[MemoryRecord]:
    """Return a batch of sample memory records with varying properties."""
    return [
        MemoryRecord(
            content="The user prefers Python for data analysis.",
            source_type=SourceType.CONVERSATION,
            utility_score=0.8,
            goal_tags=["preferences"],
        ),
        MemoryRecord(
            content="The agent completed the data preprocessing step.",
            source_type=SourceType.OBSERVATION,
            utility_score=0.6,
            goal_tags=["data_pipeline"],
        ),
        MemoryRecord(
            content="Error rates increase when memory exceeds 5000 records.",
            source_type=SourceType.REFLECTION,
            utility_score=0.9,
            goal_tags=["optimization", "research"],
        ),
        MemoryRecord(
            content="The meeting is scheduled for next Tuesday.",
            source_type=SourceType.CONVERSATION,
            utility_score=0.3,
            goal_tags=["scheduling"],
        ),
        MemoryRecord(
            content="ChromaDB supports metadata filtering with where clauses.",
            source_type=SourceType.SYSTEM,
            utility_score=0.5,
            goal_tags=["implementation"],
        ),
    ]
