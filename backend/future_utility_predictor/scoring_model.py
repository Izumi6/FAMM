"""
Learned Scoring Model — Trainable future utility predictor.

This module implements a small MLP (multi-layer perceptron) that learns
to predict future utility from features, trained on retrospective data.

Training Data Source (Retrospective Labeling):
- After each agent task completes, we label memories that were actually
  retrieved AND used as positive examples (utility ≈ 1.0).
- Memories that existed but were NOT retrieved are negative examples
  (utility ≈ 0.0).
- This provides a supervised signal for future utility prediction.

Why not use the LLM itself for scoring?
- Latency: An LLM call at every memory write is too expensive.
- Reproducibility: A small sklearn model produces deterministic scores.
- Trainability: We can iterate on the model with retrospective data
  without retraining the LLM.

Architecture: 2-layer MLP with ReLU activation.
- Input: 5 features from FeatureExtractor
- Hidden: 16 units (intentionally small for interpretability)
- Output: 1 unit (sigmoid → utility score in [0, 1])
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from config.settings import LearnedScorerConfig

logger = logging.getLogger(__name__)


class LearnedScoringModel:
    """
    Trainable MLP for future utility prediction.

    Learns from retrospective labels: after tasks complete, memories
    that were actually used are labeled as high-utility training examples.

    Attributes:
        config: Learned scorer configuration.
        model: sklearn MLPRegressor (lazy-initialized).
        scaler: Feature scaler for normalization.
        is_trained: Whether the model has been trained on data.
        training_buffer: Buffer of (features, label) pairs for training.
    """

    def __init__(self, config: LearnedScorerConfig | None = None) -> None:
        """
        Initialize the learned scoring model.

        Args:
            config: Learned scorer configuration. Uses defaults if None.
        """
        self.config = config or LearnedScorerConfig()

        self.model: MLPRegressor | None = None
        self.scaler: StandardScaler = StandardScaler()
        self.is_trained: bool = False

        # Training data buffer
        self.training_buffer: list[tuple[list[float], float]] = []

        # Try to load pre-trained model
        self._try_load()

    def score(self, features: list[float]) -> float:
        """
        Predict utility score from features.

        If the model is not yet trained, falls back to the mean
        of features as a reasonable default.

        Args:
            features: Feature vector from FeatureExtractor.

        Returns:
            Predicted utility score in [0.0, 1.0].
        """
        if not self.is_trained or self.model is None:
            # Fallback: simple mean of features
            return float(np.clip(np.mean(features), 0.0, 1.0))

        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]

        return float(np.clip(prediction, 0.0, 1.0))

    def add_training_example(
        self,
        features: list[float],
        label: float,
    ) -> None:
        """
        Add a retrospective training example to the buffer.

        Args:
            features: Feature vector at write-time.
            label: Retrospective utility label (1.0 = used, 0.0 = unused).
        """
        self.training_buffer.append((features, float(np.clip(label, 0.0, 1.0))))

        # Auto-retrain if buffer is large enough
        if len(self.training_buffer) >= self.config.retrain_interval_steps:
            self.train()

    def train(self) -> dict[str, Any]:
        """
        Train the MLP on accumulated retrospective data.

        Returns:
            Training metrics (loss, sample count, etc.).
        """
        if len(self.training_buffer) < 10:
            logger.warning(
                "Insufficient training data (%d samples). Need at least 10.",
                len(self.training_buffer),
            )
            return {"status": "insufficient_data", "samples": len(self.training_buffer)}

        X = np.array([f for f, _ in self.training_buffer])
        y = np.array([l for _, l in self.training_buffer])

        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train MLP
        self.model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=20,
        )

        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Compute training loss
        y_pred = self.model.predict(X_scaled)
        mse = float(np.mean((y - y_pred) ** 2))

        # Save model
        self._save()

        metrics = {
            "status": "trained",
            "samples": len(self.training_buffer),
            "mse": round(mse, 6),
            "iterations": self.model.n_iter_,
        }

        logger.info(
            "Learned scorer trained: %d samples, MSE=%.6f, iterations=%d",
            metrics["samples"],
            metrics["mse"],
            metrics["iterations"],
        )

        # Clear buffer after training
        self.training_buffer.clear()

        return metrics

    def _save(self) -> None:
        """Persist the trained model and scaler to disk."""
        if self.model is None:
            return

        model_path = Path(self.config.model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "scaler": self.scaler,
            "is_trained": self.is_trained,
        }

        with open(model_path, "wb") as f:
            pickle.dump(payload, f)

        logger.debug("Learned scorer saved to %s", model_path)

    def _try_load(self) -> None:
        """Attempt to load a pre-trained model from disk."""
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            return

        try:
            with open(model_path, "rb") as f:
                payload = pickle.load(f)

            self.model = payload["model"]
            self.scaler = payload["scaler"]
            self.is_trained = payload["is_trained"]

            logger.info("Loaded pre-trained learned scorer from %s", model_path)
        except Exception:
            logger.warning("Failed to load learned scorer, starting fresh")
            self.model = None
            self.is_trained = False

    def get_feature_importances(self) -> dict[str, float] | None:
        """
        Estimate feature importances from the trained MLP.

        Uses the absolute values of the first layer weights as a
        proxy for feature importance (common heuristic for MLPs).

        Returns:
            Dict mapping feature names to importance scores, or None
            if model is not trained.
        """
        from backend.future_utility_predictor.feature_extractor import FeatureExtractor

        if not self.is_trained or self.model is None:
            return None

        # Sum of absolute weights from input to first hidden layer
        first_layer_weights = np.abs(self.model.coefs_[0])
        importances = first_layer_weights.sum(axis=1)
        importances = importances / importances.sum()  # Normalize

        return dict(zip(FeatureExtractor.FEATURE_NAMES, importances.tolist()))
