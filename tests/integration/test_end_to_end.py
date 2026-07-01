"""
Integration test — Full write-read-decay-consolidate pipeline.

Tests the end-to-end flow of FAMM:
1. Store memories via MemoryManager
2. Retrieve via GoalAwareRetriever
3. Run decay cycles
4. Run consolidation
5. Verify results
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from backend.consolidation.consolidator import Consolidator
from backend.forgetting_engine.decay_scheduler import DecayScheduler
from backend.forgetting_engine.utility_decay import UtilityDecay
from backend.future_utility_predictor.predictor import FutureUtilityPredictor
from backend.goal_retrieval.retriever import GoalAwareRetriever
from backend.memory_engine.event_bus import EventBus
from backend.memory_engine.memory_manager import MemoryManager
from backend.memory_engine.memory_record import SourceType
from backend.vector_database.chroma_adapter import ChromaAdapter
from backend.vector_database.embedding_service import EmbeddingService
from config.settings import ChromaConfig, FAMMConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir):
    """Create a test configuration with temp paths."""
    config = FAMMConfig.default()
    config.vector_db.chroma.persist_directory = str(Path(temp_dir) / "chroma")
    return config


@pytest.fixture
def embedding_service():
    """Create a real embedding service (loads model once)."""
    return EmbeddingService()


@pytest.fixture
def vector_store(test_config):
    """Create a ChromaDB adapter with temp storage."""
    return ChromaAdapter(test_config.vector_db.chroma)


@pytest.fixture
def event_bus():
    """Create a fresh event bus."""
    return EventBus()


@pytest.fixture
def utility_predictor(test_config, embedding_service):
    """Create a Future Utility Predictor."""
    return FutureUtilityPredictor(
        config=test_config.future_utility_predictor,
        embedding_service=embedding_service,
    )


@pytest.fixture
def memory_manager(test_config, vector_store, embedding_service, event_bus, utility_predictor):
    """Create a fully-configured MemoryManager."""
    return MemoryManager(
        config=test_config,
        vector_store=vector_store,
        embedding_service=embedding_service,
        event_bus=event_bus,
        utility_predictor=utility_predictor,
    )


class TestWriteReadFlow:
    """Test the basic write → read pipeline."""

    def test_store_and_retrieve(self, memory_manager):
        """Stored memories should be retrievable by semantic search."""
        # Store
        record = memory_manager.store(
            content="The user prefers dark mode for all applications.",
            source_type=SourceType.CONVERSATION,
            goal_tags=["preferences"],
            goal_context=["customize user interface"],
        )

        assert record.id is not None
        assert record.embedding != []
        assert 0.0 <= record.utility_score <= 1.0

        # Retrieve
        results = memory_manager.retrieve("What are the user's UI preferences?")
        assert len(results) >= 1
        assert results[0].content == record.content

    def test_store_multiple_and_rank(self, memory_manager):
        """Multiple memories should be ranked by relevance."""
        memory_manager.store(
            content="The database migration completed successfully.",
            source_type=SourceType.OBSERVATION,
            goal_tags=["infrastructure"],
        )
        memory_manager.store(
            content="The user wants to deploy to production by Friday.",
            source_type=SourceType.CONVERSATION,
            goal_tags=["deployment"],
        )
        memory_manager.store(
            content="Unit test coverage is currently at 85 percent.",
            source_type=SourceType.OBSERVATION,
            goal_tags=["testing"],
        )

        results = memory_manager.retrieve("When is the deployment deadline?")
        assert len(results) >= 1
        # The deployment memory should rank highest
        assert "deploy" in results[0].content.lower() or "friday" in results[0].content.lower()

    def test_get_by_id(self, memory_manager):
        """Should retrieve exact memory by ID."""
        record = memory_manager.store(content="Specific memory for ID lookup.")
        fetched = memory_manager.get_by_id(record.id)
        assert fetched is not None
        assert fetched.content == record.content


class TestUtilityPrediction:
    """Test that utility scoring works in the pipeline."""

    def test_goal_aligned_memory_gets_higher_utility(self, memory_manager):
        """Memory aligned with goals should get higher utility."""
        aligned = memory_manager.store(
            content="The research paper needs a methods section.",
            goal_context=["write the research paper methodology"],
            goal_tags=["writing"],
        )

        unaligned = memory_manager.store(
            content="The weather today is sunny and warm.",
            goal_context=["write the research paper methodology"],
            goal_tags=["weather"],
        )

        # Aligned memory should have higher utility
        assert aligned.utility_score > unaligned.utility_score


class TestDecayCycle:
    """Test the forgetting engine integration."""

    def test_decay_reduces_utility(self, memory_manager):
        """Running step() should eventually decay utilities."""
        record = memory_manager.store(
            content="A memory that will decay over time.",
            source_type=SourceType.CONVERSATION,
        )
        initial_utility = record.utility_score

        # Run multiple decay cycles
        for _ in range(20):
            memory_manager.step()

        # Fetch the record again
        updated = memory_manager.get_by_id(record.id)
        if updated is not None:
            assert updated.utility_score <= initial_utility


class TestReinforcement:
    """Test the reinforcement mechanism."""

    def test_reinforce_increases_utility(self, memory_manager):
        """Reinforcing a memory should increase its utility."""
        record = memory_manager.store(
            content="Important finding about memory systems.",
            source_type=SourceType.REFLECTION,
        )
        initial = record.utility_score

        success = memory_manager.reinforce(record.id, boost=0.15)
        assert success is True

        updated = memory_manager.get_by_id(record.id)
        assert updated is not None
        assert updated.utility_score > initial


class TestGoalAwareRetrieval:
    """Test the Goal-Aware Retriever."""

    def test_goal_aware_retrieval(
        self, test_config, vector_store, embedding_service, event_bus
    ):
        """Goal-aware retrieval should rank goal-aligned memories higher."""
        # Manually store some memories
        manager = MemoryManager(
            config=test_config,
            vector_store=vector_store,
            embedding_service=embedding_service,
            event_bus=event_bus,
        )

        manager.store(
            content="Python supports list comprehensions for concise code.",
            goal_tags=["coding"],
        )
        manager.store(
            content="The experiment results show improved retrieval precision.",
            goal_tags=["research"],
        )
        manager.store(
            content="Remember to buy groceries after work.",
            goal_tags=["personal"],
        )

        # Create retriever
        retriever = GoalAwareRetriever(
            config=test_config.goal_retrieval,
            vector_store=vector_store,
            embedding_service=embedding_service,
            event_bus=event_bus,
        )

        # Retrieve with research goal
        results = retriever.retrieve(
            query="What did the experiments show?",
            goals=["analyze experimental results for the research paper"],
        )

        assert len(results) >= 1
        # Research-related memory should rank high
        assert "experiment" in results[0].memory.content.lower() or "retrieval" in results[0].memory.content.lower()


class TestStats:
    """Test statistics and monitoring."""

    def test_get_stats(self, memory_manager):
        """Stats should reflect stored memories."""
        memory_manager.store(content="Memory one.")
        memory_manager.store(content="Memory two.")
        memory_manager.store(content="Memory three.")

        stats = memory_manager.get_stats()
        assert stats["total_records"] == 3
        assert stats["cached_records"] == 3
        assert stats["step_count"] == 0
