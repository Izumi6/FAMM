#!/usr/bin/env python3
"""
run_experiment.py — Execute the FAMM comparative experiment.

Runs FAMM and all baselines on the synthetic research agent scenario,
collects metrics, and saves results to experiments/results/.

Usage:
    python run_experiment.py
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path

from backend.consolidation.consolidator import Consolidator
from backend.forgetting_engine.decay_scheduler import DecayScheduler
from backend.future_utility_predictor.predictor import FutureUtilityPredictor
from backend.goal_retrieval.retriever import GoalAwareRetriever
from backend.memory_engine.event_bus import EventBus
from backend.memory_engine.memory_manager import MemoryManager
from backend.memory_engine.memory_record import MemoryState, SourceType
from backend.vector_database.chroma_adapter import ChromaAdapter
from backend.vector_database.embedding_service import EmbeddingService
from baselines.baseline_systems import (
    EbbinghausDecayBaseline,
    ImportanceScoringBaseline,
    NaiveFIFOBaseline,
    SimilarityOnlyBaseline,
)
from config.settings import ChromaConfig, FAMMConfig
from evaluation.experiment_runner import ExperimentRunner, create_synthetic_scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("experiment")


class FAMMSystemWrapper:
    """Wraps FAMM MemoryManager to match baseline interface."""

    def __init__(self, config: FAMMConfig, persist_dir: str) -> None:
        config.vector_db.chroma.persist_directory = persist_dir
        self.embedding_service = EmbeddingService(config.embedding)
        self.event_bus = EventBus()
        self.vector_store = ChromaAdapter(config.vector_db.chroma)
        self.utility_predictor = FutureUtilityPredictor(
            config=config.future_utility_predictor,
            embedding_service=self.embedding_service,
        )
        self.manager = MemoryManager(
            config=config,
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            event_bus=self.event_bus,
            utility_predictor=self.utility_predictor,
        )
        self.retriever = GoalAwareRetriever(
            config=config.goal_retrieval,
            vector_store=self.vector_store,
            embedding_service=self.embedding_service,
            event_bus=self.event_bus,
        )
        self.goals: list[str] = []

    def store(self, content: str, **kwargs) -> object:
        source_type_str = kwargs.get("source_type", "conversation")
        source_map = {
            "conversation": SourceType.CONVERSATION,
            "observation": SourceType.OBSERVATION,
            "reflection": SourceType.REFLECTION,
            "system": SourceType.SYSTEM,
            "consolidated": SourceType.CONSOLIDATED,
        }
        source_type = source_map.get(source_type_str, SourceType.CONVERSATION)
        goal_tags = kwargs.get("goal_tags", [])

        return self.manager.store(
            content=content,
            source_type=source_type,
            goal_tags=goal_tags,
            goal_context=self.goals,
        )

    def retrieve(self, query: str, top_k: int = 10) -> list:
        results = self.retriever.retrieve(
            query=query,
            goals=self.goals,
            top_k=top_k,
        )
        # Return the memory records (unwrap RankedResult)
        return [r.memory for r in results]

    def step(self) -> dict:
        return self.manager.step()

    def get_stats(self) -> dict:
        stats = self.manager.get_stats()
        stats["system"] = "FAMM"
        return stats


def main() -> None:
    """Run the full comparative experiment."""
    logger.info("=" * 60)
    logger.info("FAMM Comparative Experiment")
    logger.info("=" * 60)

    # Create scenario
    scenario = create_synthetic_scenario()
    logger.info("Scenario: %s", scenario.name)
    logger.info("Memories: %d, Queries: %d, Steps: %d",
                len(scenario.memories), len(scenario.queries), scenario.num_steps)

    # Shared embedding service (load model once)
    embedding_service = EmbeddingService()
    logger.info("Embedding model loaded: %s (dim=%d)",
                embedding_service.config.model_name, embedding_service.dimension)

    # Create temp directories for each system's vector store
    temp_dirs: list[str] = []

    def make_temp_dir(name: str) -> str:
        d = tempfile.mkdtemp(prefix=f"famm_exp_{name}_")
        temp_dirs.append(d)
        return d

    # Initialize systems
    logger.info("Initializing memory systems...")

    # 1. FAMM (full system)
    famm_config = FAMMConfig.default()
    famm_system = FAMMSystemWrapper(famm_config, make_temp_dir("famm"))
    famm_system.goals = scenario.goals

    # 2. Baselines
    fifo_store = ChromaAdapter(ChromaConfig(
        persist_directory=make_temp_dir("fifo"),
        collection_name="fifo_memories",
    ))
    fifo = NaiveFIFOBaseline(fifo_store, embedding_service)

    sim_store = ChromaAdapter(ChromaConfig(
        persist_directory=make_temp_dir("sim"),
        collection_name="sim_memories",
    ))
    sim_only = SimilarityOnlyBaseline(sim_store, embedding_service)

    imp_store = ChromaAdapter(ChromaConfig(
        persist_directory=make_temp_dir("imp"),
        collection_name="imp_memories",
    ))
    importance = ImportanceScoringBaseline(imp_store, embedding_service)

    ebb_store = ChromaAdapter(ChromaConfig(
        persist_directory=make_temp_dir("ebb"),
        collection_name="ebb_memories",
    ))
    ebbinghaus = EbbinghausDecayBaseline(ebb_store, embedding_service)

    systems = {
        "FAMM": famm_system,
        "NaiveFIFO": fifo,
        "SimilarityOnly": sim_only,
        "ImportanceScoring": importance,
        "EbbinghausDecay": ebbinghaus,
    }

    # Run experiment
    runner = ExperimentRunner(results_dir="./experiments/results")
    results = runner.run_comparative(scenario, systems)

    # Save results
    output_path = runner.save_results(results)

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)

    header = f"{'System':<20} {'P@10':>8} {'R@10':>8} {'F1@10':>8} {'NDCG@10':>8} {'Time(s)':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for r in results:
        logger.info(
            f"{r.system_name:<20} "
            f"{r.metrics['mean_precision@10']:>8.4f} "
            f"{r.metrics['mean_recall@10']:>8.4f} "
            f"{r.metrics['mean_f1@10']:>8.4f} "
            f"{r.metrics['mean_ndcg@10']:>8.4f} "
            f"{r.total_time_seconds:>8.2f}"
        )

    logger.info("\nResults saved to: %s", output_path)

    # Cleanup temp dirs
    for d in temp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    logger.info("Experiment complete.")


if __name__ == "__main__":
    main()
