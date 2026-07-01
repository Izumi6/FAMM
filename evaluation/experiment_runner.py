"""
Experiment Runner — Orchestrates FAMM vs Baseline experiments.

This module drives the complete experiment pipeline:
1. Initialize FAMM and all baseline systems
2. Feed the same interaction sequence to each system
3. Issue retrieval queries and record metrics
4. Run maintenance cycles (decay, consolidation)
5. Aggregate and save results

The runner supports:
- Synthetic experiment scenarios (controlled, reproducible)
- Real dataset experiments (LoCoMo, MemoryAgentBench)
- Ablation configurations (disable individual FAMM modules)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentScenario:
    """
    Defines an experiment scenario with memories and queries.

    Attributes:
        name: Human-readable name.
        description: What this scenario tests.
        memories: List of (content, source_type, goal_tags) tuples.
        queries: List of (query, relevant_indices) tuples where
                 relevant_indices maps to positions in memories list.
        goals: Active goal descriptions during the scenario.
        num_steps: Total interaction steps to simulate.
    """

    name: str
    description: str
    memories: list[tuple[str, str, list[str]]]  # (content, source_type, goal_tags)
    queries: list[tuple[str, list[int]]]  # (query, indices_of_relevant_memories)
    goals: list[str] = field(default_factory=list)
    num_steps: int = 100


@dataclass
class ExperimentResult:
    """Results from running one system on one scenario."""

    system_name: str
    scenario_name: str
    metrics: dict[str, float]
    per_query_metrics: list[dict[str, float]]
    total_time_seconds: float
    memory_stats: dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class ExperimentRunner:
    """
    Orchestrates comparative experiments between FAMM and baselines.

    Usage:
        runner = ExperimentRunner(results_dir="./experiments/results")
        scenario = create_scenario(...)  # See synthetic_scenarios.py
        results = runner.run_scenario(scenario, systems_dict)
        runner.save_results(results)
    """

    def __init__(self, results_dir: str = "./experiments/results") -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_system_on_scenario(
        self,
        system: Any,
        scenario: ExperimentScenario,
        system_name: str,
    ) -> ExperimentResult:
        """
        Run a single memory system on a scenario and collect metrics.

        Args:
            system: Memory system (FAMM or baseline) with store/retrieve/step.
            scenario: Experiment scenario definition.
            system_name: Name for logging and results.

        Returns:
            ExperimentResult with all metrics.
        """
        start_time = time.time()

        # Phase 1: Store all memories
        stored_ids: list[str] = []
        for content, source_type, goal_tags in scenario.memories:
            record = system.store(
                content=content,
                source_type=source_type,
                goal_tags=goal_tags,
            )
            stored_ids.append(record.id)

        # Phase 2: Run interaction steps
        for step in range(scenario.num_steps):
            system.step()

        # Phase 3: Execute queries and collect metrics
        per_query_metrics = []
        all_precisions = []
        all_recalls = []
        all_f1s = []
        all_ndcgs = []

        for query, relevant_indices in scenario.queries:
            # Get relevant IDs
            relevant_ids = {stored_ids[i] for i in relevant_indices if i < len(stored_ids)}

            # Retrieve
            results = system.retrieve(query, top_k=10)
            retrieved_ids = [r.id if hasattr(r, "id") else r.memory.id for r in results]

            # Compute metrics
            stats = system.get_stats()
            metrics = compute_all_metrics(
                retrieved_ids=retrieved_ids,
                relevant_ids=relevant_ids,
                active_count=stats.get("total_records", 0),
                total_stored=len(stored_ids),
                utilities=[r.utility_score if hasattr(r, "utility_score") else 0.5 for r in results],
                k=10,
            )

            per_query_metrics.append({"query": query[:50], **metrics})
            all_precisions.append(metrics["precision@10"])
            all_recalls.append(metrics["recall@10"])
            all_f1s.append(metrics["f1@10"])
            all_ndcgs.append(metrics["ndcg@10"])

        # Aggregate metrics
        elapsed = time.time() - start_time
        final_stats = system.get_stats()

        aggregated = {
            "mean_precision@10": round(float(np.mean(all_precisions)), 4) if all_precisions else 0.0,
            "mean_recall@10": round(float(np.mean(all_recalls)), 4) if all_recalls else 0.0,
            "mean_f1@10": round(float(np.mean(all_f1s)), 4) if all_f1s else 0.0,
            "mean_ndcg@10": round(float(np.mean(all_ndcgs)), 4) if all_ndcgs else 0.0,
            "total_records_after": final_stats.get("total_records", 0),
            "total_time_seconds": round(elapsed, 2),
        }

        logger.info(
            "System '%s' on scenario '%s': P@10=%.3f, R@10=%.3f, F1=%.3f, NDCG=%.3f (%.1fs)",
            system_name,
            scenario.name,
            aggregated["mean_precision@10"],
            aggregated["mean_recall@10"],
            aggregated["mean_f1@10"],
            aggregated["mean_ndcg@10"],
            elapsed,
        )

        return ExperimentResult(
            system_name=system_name,
            scenario_name=scenario.name,
            metrics=aggregated,
            per_query_metrics=per_query_metrics,
            total_time_seconds=round(elapsed, 2),
            memory_stats=final_stats,
        )

    def run_comparative(
        self,
        scenario: ExperimentScenario,
        systems: dict[str, Any],
    ) -> list[ExperimentResult]:
        """
        Run all systems on the same scenario for comparison.

        Args:
            scenario: The experiment scenario.
            systems: Dict mapping system_name → system instance.

        Returns:
            List of ExperimentResult, one per system.
        """
        results = []

        for name, system in systems.items():
            logger.info("Running system '%s' on scenario '%s'", name, scenario.name)
            result = self.run_system_on_scenario(system, scenario, name)
            results.append(result)

        return results

    def save_results(
        self,
        results: list[ExperimentResult],
        filename: str | None = None,
    ) -> Path:
        """
        Save experiment results to JSON.

        Args:
            results: List of experiment results.
            filename: Output filename (auto-generated if None).

        Returns:
            Path to the saved results file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            scenario = results[0].scenario_name if results else "unknown"
            filename = f"{scenario}_{timestamp}.json"

        output_path = self.results_dir / filename

        data = {
            "experiment_timestamp": datetime.now(timezone.utc).isoformat(),
            "num_systems": len(results),
            "scenario": results[0].scenario_name if results else "unknown",
            "results": [
                {
                    "system_name": r.system_name,
                    "metrics": r.metrics,
                    "per_query_metrics": r.per_query_metrics,
                    "total_time_seconds": r.total_time_seconds,
                    "memory_stats": r.memory_stats,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info("Results saved to %s", output_path)
        return output_path


def create_synthetic_scenario() -> ExperimentScenario:
    """
    Create the primary synthetic evaluation scenario.

    This scenario tests the core hypothesis:
    - Memories aligned with goals should be retrieved more accurately
    - High-utility memories should survive decay better
    - Goal-aware retrieval should outperform similarity-only

    Returns:
        ExperimentScenario ready for the runner.
    """
    memories = [
        # Goal-aligned memories (should rank high)
        ("The research paper requires an analysis of memory decay patterns in LLM agents.",
         "reflection", ["research", "writing"]),
        ("Experimental results show FAMM achieves 15% higher precision than baselines.",
         "observation", ["research", "evaluation"]),
        ("The methodology section should explain the utility-conditioned decay formula.",
         "reflection", ["writing", "methodology"]),

        # Moderately relevant
        ("Python's scikit-learn library supports MLPRegressor for small neural networks.",
         "system", ["implementation"]),
        ("ChromaDB uses HNSW indexing for approximate nearest neighbor search.",
         "system", ["implementation"]),
        ("The agent successfully completed the data preprocessing pipeline.",
         "observation", ["data_pipeline"]),

        # Irrelevant (noise — should be forgotten over time)
        ("The weather forecast predicts rain for the upcoming weekend.",
         "conversation", ["personal"]),
        ("The user mentioned they enjoy coffee in the morning.",
         "conversation", ["personal"]),
        ("A delivery package is expected to arrive on Tuesday.",
         "conversation", ["personal"]),
        ("The office temperature should be set to 72 degrees.",
         "conversation", ["environment"]),

        # More goal-aligned
        ("MemGPT uses a tiered memory architecture with main and archival storage.",
         "reflection", ["research", "literature"]),
        ("The evaluation should include ablation studies disabling each FAMM module.",
         "reflection", ["evaluation", "methodology"]),
    ]

    queries = [
        # Research queries (memories 0, 1, 2 are relevant)
        ("What are the key findings about memory decay in our research?", [0, 1, 2]),

        # Writing queries (memories 0, 2, 11 are relevant)
        ("What should the methodology section include?", [0, 2, 11]),

        # Implementation queries (memories 3, 4 are relevant)
        ("What tools are we using for the implementation?", [3, 4]),

        # Literature queries (memories 10 are relevant)
        ("How does MemGPT manage memory?", [10]),

        # Evaluation queries (memories 1, 11 are relevant)
        ("What experiments should we run for evaluation?", [1, 11]),
    ]

    return ExperimentScenario(
        name="synthetic_research_agent",
        description=(
            "Simulates a research assistant agent with mixed memories "
            "about research, personal life, and implementation. Tests "
            "whether goal-aware retrieval prioritizes research-relevant "
            "memories and whether adaptive decay forgets irrelevant ones."
        ),
        memories=memories,
        queries=queries,
        goals=[
            "Complete the research paper on memory management for LLM agents",
            "Run experiments comparing FAMM against baseline systems",
            "Write the methodology and evaluation sections",
        ],
        num_steps=50,
    )
