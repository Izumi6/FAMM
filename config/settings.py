"""
FAMM Configuration Models

Type-safe configuration using Pydantic models.
Loads from YAML files and validates all parameters at startup.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


# ─────────────────────────────────────────────
# Sub-configurations
# ─────────────────────────────────────────────


class EmbeddingConfig(BaseModel):
    """Configuration for the embedding service."""

    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=1)
    normalize: bool = True


class ChromaConfig(BaseModel):
    """Configuration for ChromaDB backend."""

    persist_directory: str = "./chroma_data"
    collection_name: str = "famm_memories"


class FAISSConfig(BaseModel):
    """Configuration for FAISS backend."""

    index_type: str = "FlatIP"
    persist_directory: str = "./faiss_data"


class VectorDBConfig(BaseModel):
    """Configuration for the vector database layer."""

    backend: Literal["chroma", "faiss"] = "chroma"
    chroma: ChromaConfig = ChromaConfig()
    faiss: FAISSConfig = FAISSConfig()


class MemoryEngineConfig(BaseModel):
    """Configuration for the central memory engine."""

    max_memories: int = Field(default=10000, ge=100)
    stale_threshold_days: int = Field(default=30, ge=1)
    archive_threshold_days: int = Field(default=90, ge=1)


class HeuristicScorerConfig(BaseModel):
    """Weights for heuristic-based future utility scoring."""

    weight_goal_similarity: float = Field(default=0.35, ge=0.0, le=1.0)
    weight_recency: float = Field(default=0.20, ge=0.0, le=1.0)
    weight_access_frequency: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_source_type: float = Field(default=0.15, ge=0.0, le=1.0)
    weight_entity_overlap: float = Field(default=0.15, ge=0.0, le=1.0)

    @field_validator("weight_entity_overlap")
    @classmethod
    def weights_must_sum_to_one(cls, v: float, info: object) -> float:
        """Validate that all weights sum to approximately 1.0."""
        data = info.data if hasattr(info, "data") else {}
        total = (
            data.get("weight_goal_similarity", 0.35)
            + data.get("weight_recency", 0.20)
            + data.get("weight_access_frequency", 0.15)
            + data.get("weight_source_type", 0.15)
            + v
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Heuristic weights must sum to 1.0, got {total:.3f}")
        return v


class LearnedScorerConfig(BaseModel):
    """Configuration for the learned utility scoring model."""

    model_path: str = "./models/utility_scorer.pkl"
    retrain_interval_steps: int = Field(default=500, ge=10)


class FutureUtilityPredictorConfig(BaseModel):
    """Configuration for the Future Utility Predictor module."""

    mode: Literal["heuristic", "learned"] = "heuristic"
    heuristic: HeuristicScorerConfig = HeuristicScorerConfig()
    learned: LearnedScorerConfig = LearnedScorerConfig()


class RankingWeightsConfig(BaseModel):
    """Weights for multi-signal ranking in goal-aware retrieval."""

    semantic_similarity: float = Field(default=0.30, ge=0.0, le=1.0)
    utility_score: float = Field(default=0.25, ge=0.0, le=1.0)
    goal_alignment: float = Field(default=0.30, ge=0.0, le=1.0)
    recency: float = Field(default=0.15, ge=0.0, le=1.0)


class GoalRetrievalConfig(BaseModel):
    """Configuration for the Goal-Aware Retrieval module."""

    top_k_candidates: int = Field(default=50, ge=1)
    top_k_results: int = Field(default=10, ge=1)
    ranking_weights: RankingWeightsConfig = RankingWeightsConfig()
    use_llm_reranker: bool = False


class ForgettingEngineConfig(BaseModel):
    """Configuration for the Forgetting Engine."""

    decay_interval_steps: int = Field(default=10, ge=1)
    base_decay_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    prune_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    utility_exponent: float = Field(default=2.0, ge=0.1)
    archive_before_delete: bool = True


class LLMConfig(BaseModel):
    """Configuration for LLM-based operations (consolidation, reranking)."""

    model_name: str = "gemma2:2b"
    base_url: str = "http://localhost:11434"


class ConsolidationConfig(BaseModel):
    """Configuration for Adaptive Memory Consolidation."""

    consolidation_interval_steps: int = Field(default=50, ge=1)
    min_cluster_size: int = Field(default=3, ge=2)
    cluster_similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    summarization_mode: Literal["extractive", "llm"] = "extractive"
    llm: LLMConfig = LLMConfig()


class ExperimentConfig(BaseModel):
    """Configuration for experiments and evaluation."""

    seed: int = 42
    results_dir: str = "./experiments/results"
    logs_dir: str = "./experiments/logs"
    figures_dir: str = "./experiments/figures"


# ─────────────────────────────────────────────
# Root Configuration
# ─────────────────────────────────────────────


class FAMMConfig(BaseSettings):
    """
    Root configuration for the FAMM framework.

    Loads from YAML file and provides type-safe access
    to all module configurations.
    """

    embedding: EmbeddingConfig = EmbeddingConfig()
    vector_db: VectorDBConfig = VectorDBConfig()
    memory_engine: MemoryEngineConfig = MemoryEngineConfig()
    future_utility_predictor: FutureUtilityPredictorConfig = FutureUtilityPredictorConfig()
    goal_retrieval: GoalRetrievalConfig = GoalRetrievalConfig()
    forgetting_engine: ForgettingEngineConfig = ForgettingEngineConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    experiment: ExperimentConfig = ExperimentConfig()

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "FAMMConfig":
        """
        Load configuration from a YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Validated FAMMConfig instance.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If validation fails.
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        return cls(**raw_config)

    @classmethod
    def default(cls) -> "FAMMConfig":
        """Create a configuration with all default values."""
        return cls()
