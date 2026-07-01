"""
Baseline Heuristic Scorer — Rule-based future utility prediction.

This scorer implements a weighted linear combination of features
as the baseline scoring strategy. It serves two purposes:

1. Production: Used when no training data is available yet for
   the learned model.
2. Ablation: Used as a comparison point to measure the learned
   model's improvement over hand-crafted rules.

The weights are configurable via the FAMM configuration system
and reflect our prior beliefs about which features matter most
for predicting future utility.

Default weight rationale:
- goal_similarity (0.35): Most predictive — memories aligned with
  future goals are most likely to be retrieved.
- recency (0.20): Recent memories are more likely to be relevant
  in the near term.
- access_frequency (0.15): Frequently accessed content has proven value.
- source_type (0.15): Reflections/observations have higher base utility.
- entity_overlap (0.15): Shared entities indicate topical relevance.
"""

from __future__ import annotations

import logging

from config.settings import HeuristicScorerConfig

logger = logging.getLogger(__name__)


class BaselineHeuristicScorer:
    """
    Weighted linear combination scorer for future utility prediction.

    Computes: score = Σ(weight_i * feature_i) for all features.

    This is deterministic, interpretable, and requires no training data.

    Attributes:
        weights: Ordered list of feature weights matching FeatureExtractor output.
        config: Heuristic scorer configuration.
    """

    def __init__(self, config: HeuristicScorerConfig | None = None) -> None:
        """
        Initialize with configurable weights.

        Args:
            config: Heuristic scorer config with feature weights.
                    Uses defaults if None.
        """
        self.config = config or HeuristicScorerConfig()

        # Weights ordered to match FeatureExtractor.FEATURE_NAMES
        self.weights = [
            self.config.weight_goal_similarity,
            self.config.weight_recency,
            self.config.weight_access_frequency,
            self.config.weight_source_type,
            self.config.weight_entity_overlap,
        ]

        logger.debug("Heuristic scorer initialized with weights: %s", self.weights)

    def score(self, features: list[float]) -> float:
        """
        Compute utility score from features using weighted sum.

        Args:
            features: Feature vector from FeatureExtractor.extract().
                      Must have same length as self.weights.

        Returns:
            Utility score clamped to [0.0, 1.0].

        Raises:
            ValueError: If features length doesn't match weights length.
        """
        if len(features) != len(self.weights):
            raise ValueError(
                f"Feature count mismatch: expected {len(self.weights)}, got {len(features)}"
            )

        raw_score = sum(w * f for w, f in zip(self.weights, features))

        # Clamp to valid range
        clamped = max(0.0, min(1.0, raw_score))

        return round(clamped, 4)

    def score_with_breakdown(
        self, features: list[float]
    ) -> dict[str, float]:
        """
        Compute utility score with per-feature contribution breakdown.

        Useful for debugging and paper figures showing which features
        contribute most to the prediction.

        Args:
            features: Feature vector from FeatureExtractor.

        Returns:
            Dict with total score and per-feature weighted contributions.
        """
        from backend.future_utility_predictor.feature_extractor import FeatureExtractor

        if len(features) != len(self.weights):
            raise ValueError(
                f"Feature count mismatch: expected {len(self.weights)}, got {len(features)}"
            )

        contributions = {}
        for name, weight, feature_val in zip(
            FeatureExtractor.FEATURE_NAMES, self.weights, features
        ):
            contributions[name] = round(weight * feature_val, 4)

        total = sum(contributions.values())
        contributions["total"] = round(max(0.0, min(1.0, total)), 4)

        return contributions
