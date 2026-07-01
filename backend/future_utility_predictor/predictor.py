"""
Future Utility Predictor — Main interface for predicting memory value.

This is FAMM's core novel contribution. It answers the question:
"How useful will this memory be for FUTURE agent tasks?"

Unlike existing systems that score memories reactively (at retrieval time
based on the current query), the Future Utility Predictor scores
proactively (at write time based on predicted future relevance).

The predictor supports two modes:
1. Heuristic: Weighted linear combination of features (default, no training)
2. Learned: MLP trained on retrospective labels (after sufficient data)

The mode is configured via `future_utility_predictor.mode` in the YAML config.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.future_utility_predictor.baseline_heuristic import BaselineHeuristicScorer
from backend.future_utility_predictor.feature_extractor import FeatureExtractor
from backend.future_utility_predictor.scoring_model import LearnedScoringModel
from backend.memory_engine.memory_record import MemoryRecord
from backend.vector_database.embedding_service import EmbeddingService
from config.settings import FutureUtilityPredictorConfig

logger = logging.getLogger(__name__)


class FutureUtilityPredictor:
    """
    Predicts the future utility of a memory at write-time.

    Extracts features from the memory and active goal context,
    then scores using either a heuristic or learned model.

    Attributes:
        config: Predictor configuration.
        feature_extractor: Computes predictive feature vectors.
        heuristic_scorer: Rule-based scoring (always available).
        learned_scorer: MLP-based scoring (used in "learned" mode).
        mode: Current scoring mode ("heuristic" or "learned").

    Example:
        >>> predictor = FutureUtilityPredictor(config, embedding_service)
        >>> memory = MemoryRecord(content="User prefers dark mode")
        >>> score = predictor.predict(memory, goal_context=["customize UI"])
        >>> 0.0 <= score <= 1.0
        True
    """

    def __init__(
        self,
        config: FutureUtilityPredictorConfig,
        embedding_service: EmbeddingService,
    ) -> None:
        """
        Initialize the Future Utility Predictor.

        Args:
            config: Predictor configuration.
            embedding_service: For computing semantic similarity features.
        """
        self.config = config
        self.mode = config.mode

        # Always initialize both scorers
        self.feature_extractor = FeatureExtractor(embedding_service)
        self.heuristic_scorer = BaselineHeuristicScorer(config.heuristic)
        self.learned_scorer = LearnedScoringModel(config.learned)

        logger.info("FutureUtilityPredictor initialized in '%s' mode", self.mode)

    def predict(
        self,
        memory: MemoryRecord,
        goal_context: list[str] | None = None,
    ) -> float:
        """
        Predict the future utility of a memory.

        This is the primary interface called by the MemoryManager
        at write time.

        Args:
            memory: The memory record to score (must have embedding set).
            goal_context: List of active goal descriptions.

        Returns:
            Utility score in [0.0, 1.0].
        """
        # Extract features
        features = self.feature_extractor.extract(memory, goal_context)

        # Score based on mode
        if self.mode == "learned" and self.learned_scorer.is_trained:
            score = self.learned_scorer.score(features)
        else:
            score = self.heuristic_scorer.score(features)

        logger.debug(
            "Predicted utility for memory %s: %.4f (mode=%s, features=%s)",
            memory.id[:8],
            score,
            self.mode,
            [f"{f:.3f}" for f in features],
        )

        return score

    def predict_with_explanation(
        self,
        memory: MemoryRecord,
        goal_context: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Predict utility with full feature breakdown.

        Useful for debugging, paper figures, and interpretability analysis.

        Args:
            memory: The memory record to score.
            goal_context: Active goal descriptions.

        Returns:
            Dict with score, features, breakdown, and metadata.
        """
        features = self.feature_extractor.extract(memory, goal_context)
        feature_dict = self.feature_extractor.extract_dict(memory, goal_context)

        # Get score and breakdown
        if self.mode == "learned" and self.learned_scorer.is_trained:
            score = self.learned_scorer.score(features)
            scoring_method = "learned"
            importances = self.learned_scorer.get_feature_importances()
        else:
            score = self.heuristic_scorer.score(features)
            scoring_method = "heuristic"
            importances = None

        breakdown = self.heuristic_scorer.score_with_breakdown(features)

        return {
            "utility_score": score,
            "scoring_method": scoring_method,
            "features": feature_dict,
            "heuristic_breakdown": breakdown,
            "learned_importances": importances,
            "goal_context": goal_context or [],
        }

    def record_retrospective_label(
        self,
        memory: MemoryRecord,
        was_useful: bool,
        goal_context: list[str] | None = None,
    ) -> None:
        """
        Record a retrospective training label for the learned scorer.

        Called after a task completes to label whether a memory
        was actually useful (retrieved and used) or not.

        Args:
            memory: The memory record.
            was_useful: True if memory was retrieved and used successfully.
            goal_context: Goal context at the time of storage.
        """
        features = self.feature_extractor.extract(memory, goal_context)
        label = 1.0 if was_useful else 0.0

        self.learned_scorer.add_training_example(features, label)

        logger.debug(
            "Recorded retrospective label for memory %s: useful=%s",
            memory.id[:8],
            was_useful,
        )

    def train_learned_model(self) -> dict[str, Any]:
        """
        Trigger training of the learned scoring model.

        Returns:
            Training metrics from the learned scorer.
        """
        return self.learned_scorer.train()

    def switch_mode(self, mode: str) -> None:
        """
        Switch between heuristic and learned scoring modes.

        Args:
            mode: "heuristic" or "learned".

        Raises:
            ValueError: If mode is invalid.
        """
        if mode not in ("heuristic", "learned"):
            raise ValueError(f"Invalid mode: {mode}. Must be 'heuristic' or 'learned'.")

        if mode == "learned" and not self.learned_scorer.is_trained:
            logger.warning(
                "Switching to 'learned' mode but model is not trained. "
                "Will fall back to heuristic until training data is available."
            )

        self.mode = mode
        logger.info("Scoring mode switched to '%s'", mode)
