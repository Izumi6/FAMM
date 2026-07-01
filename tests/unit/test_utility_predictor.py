"""
Unit tests for Future Utility Predictor components.

Tests cover:
- Feature extraction (all 5 features)
- Heuristic scoring (weighted linear combination)
- Learned scoring model (buffer, training, prediction)
- Predictor interface (mode switching, explanation)
"""

import pytest
import numpy as np

from backend.future_utility_predictor.baseline_heuristic import BaselineHeuristicScorer
from backend.future_utility_predictor.feature_extractor import (
    FeatureExtractor,
    _extract_entities,
)
from backend.future_utility_predictor.scoring_model import LearnedScoringModel
from backend.memory_engine.memory_record import MemoryRecord, SourceType
from config.settings import HeuristicScorerConfig, LearnedScorerConfig


class TestFeatureExtraction:
    """Test individual feature computations."""

    def test_recency_score_new_memory(self) -> None:
        """A freshly created memory should have a high recency score."""
        record = MemoryRecord(content="Just created")
        score = FeatureExtractor._recency_score(record)
        assert score > 0.95  # Very recent → close to 1.0

    def test_access_frequency_zero(self) -> None:
        """Memory with no accesses should have zero frequency score."""
        record = MemoryRecord(content="Never accessed", access_count=0)
        score = FeatureExtractor._access_frequency_score(record)
        assert score == 0.0

    def test_access_frequency_increases_with_count(self) -> None:
        """Frequency score should increase with access count."""
        record_low = MemoryRecord(content="Low", access_count=2)
        record_high = MemoryRecord(content="High", access_count=20)

        score_low = FeatureExtractor._access_frequency_score(record_low)
        score_high = FeatureExtractor._access_frequency_score(record_high)

        assert score_high > score_low

    def test_source_type_priors(self) -> None:
        """Reflections should have higher prior than conversations."""
        reflection = MemoryRecord(content="Insight", source_type=SourceType.REFLECTION)
        conversation = MemoryRecord(content="Chat", source_type=SourceType.CONVERSATION)

        prior_r = FeatureExtractor._source_type_prior(reflection)
        prior_c = FeatureExtractor._source_type_prior(conversation)

        assert prior_r > prior_c

    def test_entity_overlap_no_goals(self) -> None:
        """Entity overlap with empty goals should be 0.0."""
        record = MemoryRecord(content="Some content about Python")
        score = FeatureExtractor._entity_overlap_score(record, [])
        assert score == 0.0

    def test_entity_extraction(self) -> None:
        """Entity extraction should capture capitalized words."""
        entities = _extract_entities("Python is used by Google and Microsoft")
        assert "python" in entities
        assert "google" in entities
        assert "microsoft" in entities


class TestHeuristicScorer:
    """Test the baseline heuristic scorer."""

    def test_default_weights_sum_to_one(self) -> None:
        """Default weights should sum to 1.0."""
        scorer = BaselineHeuristicScorer()
        assert abs(sum(scorer.weights) - 1.0) < 0.01

    def test_score_in_valid_range(self) -> None:
        """Score should always be in [0.0, 1.0]."""
        scorer = BaselineHeuristicScorer()
        features = [0.8, 0.5, 0.3, 0.7, 0.6]
        score = scorer.score(features)
        assert 0.0 <= score <= 1.0

    def test_score_with_all_ones(self) -> None:
        """All features at 1.0 should give max score."""
        scorer = BaselineHeuristicScorer()
        features = [1.0, 1.0, 1.0, 1.0, 1.0]
        score = scorer.score(features)
        assert score == pytest.approx(1.0)

    def test_score_with_all_zeros(self) -> None:
        """All features at 0.0 should give zero score."""
        scorer = BaselineHeuristicScorer()
        features = [0.0, 0.0, 0.0, 0.0, 0.0]
        score = scorer.score(features)
        assert score == 0.0

    def test_mismatched_features_raises(self) -> None:
        """Wrong number of features should raise ValueError."""
        scorer = BaselineHeuristicScorer()
        with pytest.raises(ValueError, match="Feature count mismatch"):
            scorer.score([0.5, 0.5])  # Only 2 instead of 5

    def test_score_with_breakdown(self) -> None:
        """Breakdown should have per-feature contributions."""
        scorer = BaselineHeuristicScorer()
        features = [0.8, 0.5, 0.3, 0.7, 0.6]
        breakdown = scorer.score_with_breakdown(features)

        assert "total" in breakdown
        assert "goal_similarity" in breakdown
        assert breakdown["total"] == pytest.approx(
            sum(v for k, v in breakdown.items() if k != "total"),
            abs=0.01,
        )


class TestLearnedScoringModel:
    """Test the trainable MLP scoring model."""

    def test_untrained_model_fallback(self, tmp_path) -> None:
        """Untrained model should use mean of features as fallback."""
        config = LearnedScorerConfig(model_path=str(tmp_path / "test_model.pkl"))
        model = LearnedScoringModel(config)

        features = [0.8, 0.6, 0.4, 0.2, 0.0]
        score = model.score(features)

        # Mean of [0.8, 0.6, 0.4, 0.2, 0.0] = 0.4
        assert score == pytest.approx(0.4)

    def test_add_training_example(self, tmp_path) -> None:
        """Training examples should accumulate in the buffer."""
        config = LearnedScorerConfig(
            model_path=str(tmp_path / "test_model.pkl"),
            retrain_interval_steps=1000,  # High to prevent auto-retrain
        )
        model = LearnedScoringModel(config)

        model.add_training_example([0.5, 0.5, 0.5, 0.5, 0.5], 1.0)
        model.add_training_example([0.1, 0.1, 0.1, 0.1, 0.1], 0.0)

        assert len(model.training_buffer) == 2

    def test_training_with_insufficient_data(self, tmp_path) -> None:
        """Training with < 10 samples should return insufficient_data."""
        config = LearnedScorerConfig(model_path=str(tmp_path / "test_model.pkl"))
        model = LearnedScoringModel(config)

        for i in range(5):
            model.add_training_example([0.5] * 5, float(i % 2))

        result = model.train()
        assert result["status"] == "insufficient_data"

    def test_training_with_sufficient_data(self, tmp_path) -> None:
        """Training with >= 10 samples should produce a trained model."""
        config = LearnedScorerConfig(model_path=str(tmp_path / "test_model.pkl"))
        model = LearnedScoringModel(config)

        # Generate synthetic training data
        rng = np.random.RandomState(42)
        for _ in range(50):
            features = rng.rand(5).tolist()
            label = float(np.mean(features) > 0.5)
            model.add_training_example(features, label)

        result = model.train()
        assert result["status"] == "trained"
        assert model.is_trained is True

        # Trained model should produce valid scores
        score = model.score([0.9, 0.9, 0.9, 0.9, 0.9])
        assert 0.0 <= score <= 1.0


class TestUtilityDecay:
    """Test the adaptive utility decay mechanism."""

    def test_high_utility_decays_slowly(self) -> None:
        """Memories with high utility should have very low decay."""
        from backend.forgetting_engine.utility_decay import UtilityDecay
        from config.settings import ForgettingEngineConfig

        decay = UtilityDecay(ForgettingEngineConfig())

        high_util = MemoryRecord(content="Important", utility_score=0.9)
        low_util = MemoryRecord(content="Unimportant", utility_score=0.1)

        decay_high = decay.compute_effective_decay(high_util)
        decay_low = decay.compute_effective_decay(low_util)

        # High utility → much slower decay
        assert decay_high < decay_low
        assert decay_high < 0.01  # Should be very small

    def test_decay_timeline_monotonically_decreasing(self) -> None:
        """Projected decay timeline should be monotonically decreasing."""
        from backend.forgetting_engine.utility_decay import UtilityDecay

        decay = UtilityDecay()
        timeline = decay.project_decay_timeline(initial_utility=0.8, steps=50)

        for i in range(1, len(timeline)):
            assert timeline[i] <= timeline[i - 1]

    def test_ebbinghaus_vs_famm_decay(self) -> None:
        """FAMM decay should be slower for high-utility memories than Ebbinghaus."""
        from backend.forgetting_engine.ebbinghaus_baseline import EbbinghausBaseline
        from backend.forgetting_engine.utility_decay import UtilityDecay

        famm = UtilityDecay()
        ebbinghaus = EbbinghausBaseline()

        # Project both curves
        famm_curve = famm.project_decay_timeline(initial_utility=0.9, steps=50)
        ebb_curve = ebbinghaus.project_decay_timeline(steps=50, hours_per_step=1.0)

        # After 50 steps, FAMM with high utility should retain more
        assert famm_curve[-1] > ebb_curve[-1]
