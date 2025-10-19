"""Tests for StatisticalValidator (Task 6.4).

This test suite validates the statistical validation pipeline orchestrator.
Tests cover initialization, validation workflow, orchestration, and integration
scenarios.

Test Categories:
- Initialization (successful load, missing file)
- Basic validation (known baseline + score → expected z-score and confidence)
- Edge cases (score = mean, very high/low scores)
- Orchestration verification (calls both components)
- Helper methods (get_baseline_info, classify)
- Integration scenarios (realistic baseline + document scores)

Tier: 1 (MVP)
Task: 6.4
"""

import pytest
import pickle
import tempfile
from pathlib import Path

from specHO.validator.pipeline import StatisticalValidator


@pytest.fixture
def test_baseline_file(tmp_path):
    """Create a temporary baseline file for testing."""
    baseline_stats = {
        'human_mean': 0.15,
        'human_std': 0.10,
        'n_documents': 100
    }

    baseline_path = tmp_path / "test_baseline.pkl"
    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline_stats, f)

    return str(baseline_path)


class TestStatisticalValidatorInitialization:
    """Tests for StatisticalValidator initialization."""

    def test_successful_initialization(self, test_baseline_file):
        """Test that validator initializes successfully with valid baseline."""
        validator = StatisticalValidator(test_baseline_file)

        assert validator.human_mean == pytest.approx(0.15)
        assert validator.human_std == pytest.approx(0.10)
        assert validator.n_documents == 100

    def test_missing_baseline_file_raises_error(self):
        """Test that missing baseline file raises FileNotFoundError."""
        nonexistent_path = "data/baseline/nonexistent_baseline.pkl"

        with pytest.raises(FileNotFoundError, match="Baseline statistics file not found"):
            StatisticalValidator(nonexistent_path)

    def test_error_message_includes_path(self):
        """Test that error message includes the invalid path."""
        nonexistent_path = "data/baseline/missing.pkl"

        with pytest.raises(FileNotFoundError, match="missing.pkl"):
            StatisticalValidator(nonexistent_path)

    def test_components_initialized(self, test_baseline_file):
        """Test that all component calculators are initialized."""
        validator = StatisticalValidator(test_baseline_file)

        assert validator.z_score_calculator is not None
        assert validator.confidence_converter is not None


class TestStatisticalValidatorBasicValidation:
    """Tests for basic validation functionality."""

    def test_validate_score_above_mean(self, test_baseline_file):
        """Test validation of score above human mean."""
        validator = StatisticalValidator(test_baseline_file)

        # Score 3 std devs above mean: (0.45 - 0.15) / 0.10 = 3.0
        z_score, confidence = validator.validate(0.45)

        assert z_score == pytest.approx(3.0, abs=1e-10)
        assert confidence > 0.998  # Z=3 is ~99.87th percentile

    def test_validate_score_below_mean(self, test_baseline_file):
        """Test validation of score below human mean."""
        validator = StatisticalValidator(test_baseline_file)

        # Score 1 std dev below mean: (0.05 - 0.15) / 0.10 = -1.0
        z_score, confidence = validator.validate(0.05)

        assert z_score == pytest.approx(-1.0, abs=1e-10)
        assert confidence == pytest.approx(0.1587, abs=1e-2)

    def test_validate_score_at_mean(self, test_baseline_file):
        """Test validation of score exactly at mean."""
        validator = StatisticalValidator(test_baseline_file)

        # Score at mean: (0.15 - 0.15) / 0.10 = 0.0
        z_score, confidence = validator.validate(0.15)

        assert z_score == pytest.approx(0.0, abs=1e-10)
        assert confidence == pytest.approx(0.5, abs=1e-10)

    def test_validate_returns_tuple(self, test_baseline_file):
        """Test that validate returns tuple of (z_score, confidence)."""
        validator = StatisticalValidator(test_baseline_file)

        result = validator.validate(0.25)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)  # z_score
        assert isinstance(result[1], float)  # confidence

    def test_validate_multiple_scores(self, test_baseline_file):
        """Test validating multiple different scores."""
        validator = StatisticalValidator(test_baseline_file)

        scores = [0.05, 0.15, 0.25, 0.35, 0.45]
        results = [validator.validate(score) for score in scores]

        # Z-scores should be monotonically increasing
        z_scores = [z for z, _ in results]
        assert z_scores == sorted(z_scores)

        # Confidences should be monotonically increasing
        confidences = [conf for _, conf in results]
        assert confidences == sorted(confidences)


class TestStatisticalValidatorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_high_score(self, test_baseline_file):
        """Test validation of very high score."""
        validator = StatisticalValidator(test_baseline_file)

        # Score far above mean
        z_score, confidence = validator.validate(0.85)

        assert z_score > 5.0
        assert confidence > 0.99999

    def test_very_low_score(self, test_baseline_file):
        """Test validation of very low score."""
        validator = StatisticalValidator(test_baseline_file)

        # Score far below mean
        z_score, confidence = validator.validate(0.0)

        assert z_score < -1.0
        assert confidence < 0.2

    def test_score_at_zero(self, test_baseline_file):
        """Test validation with minimum score (0.0)."""
        validator = StatisticalValidator(test_baseline_file)

        z_score, confidence = validator.validate(0.0)

        # (0.0 - 0.15) / 0.10 = -1.5
        assert z_score == pytest.approx(-1.5, abs=1e-10)

    def test_score_at_one(self, test_baseline_file):
        """Test validation with maximum score (1.0)."""
        validator = StatisticalValidator(test_baseline_file)

        z_score, confidence = validator.validate(1.0)

        # (1.0 - 0.15) / 0.10 = 8.5
        assert z_score == pytest.approx(8.5, abs=1e-10)


class TestStatisticalValidatorOrchestration:
    """Tests verifying proper component orchestration."""

    def test_calls_z_score_calculator(self, test_baseline_file, mocker):
        """Test that validate() calls ZScoreCalculator."""
        validator = StatisticalValidator(test_baseline_file)

        # Spy on the calculate_z_score method
        spy = mocker.spy(validator.z_score_calculator, 'calculate_z_score')

        validator.validate(0.25)

        # Verify it was called with correct arguments
        spy.assert_called_once()
        args = spy.call_args[1]  # keyword arguments
        assert args['document_score'] == 0.25
        assert args['human_mean'] == 0.15
        assert args['human_std'] == 0.10

    def test_calls_confidence_converter(self, test_baseline_file, mocker):
        """Test that validate() calls ConfidenceConverter."""
        validator = StatisticalValidator(test_baseline_file)

        # Spy on the convert_to_confidence method
        spy = mocker.spy(validator.confidence_converter, 'convert_to_confidence')

        validator.validate(0.25)

        # Verify it was called once
        spy.assert_called_once()

    def test_sequential_processing(self, test_baseline_file):
        """Test that components are called in correct sequence."""
        validator = StatisticalValidator(test_baseline_file)

        # Manual calculation to verify orchestration
        z_score, confidence = validator.validate(0.25)

        # Verify z-score is calculated correctly
        expected_z = (0.25 - 0.15) / 0.10
        assert z_score == pytest.approx(expected_z, abs=1e-10)

        # Verify confidence matches z-score conversion
        from scipy.stats import norm
        expected_conf = norm.cdf(expected_z)
        assert confidence == pytest.approx(expected_conf, abs=1e-10)


class TestStatisticalValidatorHelperMethods:
    """Tests for helper methods."""

    def test_get_baseline_info(self, test_baseline_file):
        """Test get_baseline_info returns correct information."""
        validator = StatisticalValidator(test_baseline_file)

        info = validator.get_baseline_info()

        assert info['human_mean'] == pytest.approx(0.15)
        assert info['human_std'] == pytest.approx(0.10)
        assert info['n_documents'] == 100

    def test_classify_watermarked(self, test_baseline_file):
        """Test classify() returns WATERMARKED for high scores."""
        validator = StatisticalValidator(test_baseline_file)

        # Score with z=3 (confidence > 0.99)
        label = validator.classify(0.45)

        assert label == "WATERMARKED"

    def test_classify_human(self, test_baseline_file):
        """Test classify() returns HUMAN for low scores."""
        validator = StatisticalValidator(test_baseline_file)

        # Score with z=-3 (confidence < 0.01)
        label = validator.classify(-0.15)

        assert label == "HUMAN"

    def test_classify_uncertain(self, test_baseline_file):
        """Test classify() returns UNCERTAIN for medium scores."""
        validator = StatisticalValidator(test_baseline_file)

        # Score near mean (confidence ~ 0.5-0.8)
        label = validator.classify(0.20)

        assert label == "UNCERTAIN"

    def test_classify_custom_threshold(self, test_baseline_file):
        """Test classify() with custom threshold."""
        validator = StatisticalValidator(test_baseline_file)

        # Score with confidence ~ 0.975 (z=2)
        # Should be WATERMARKED with threshold=0.95
        label_95 = validator.classify(0.35, threshold=0.95)
        assert label_95 == "WATERMARKED"

        # Should be UNCERTAIN with threshold=0.99
        label_99 = validator.classify(0.35, threshold=0.99)
        assert label_99 == "UNCERTAIN"


class TestStatisticalValidatorIntegration:
    """Integration tests with realistic scenarios."""

    def test_realistic_watermark_detection(self, test_baseline_file):
        """Test realistic watermark detection scenario."""
        validator = StatisticalValidator(test_baseline_file)

        # Scenario: AI-generated text with strong echo patterns
        watermarked_score = 0.42

        z_score, confidence = validator.validate(watermarked_score)

        # Should indicate watermark
        assert z_score > 2.0  # >95th percentile
        assert confidence > 0.95
        assert validator.classify(watermarked_score) == "WATERMARKED"

    def test_realistic_human_text(self, test_baseline_file):
        """Test realistic human text scenario."""
        validator = StatisticalValidator(test_baseline_file)

        # Scenario: Human-written text with normal echo patterns
        human_score = 0.12

        z_score, confidence = validator.validate(human_score)

        # Should be near or below mean
        assert z_score < 0.5
        assert confidence < 0.7

    def test_uncertain_boundary_case(self, test_baseline_file):
        """Test boundary case in uncertain region."""
        validator = StatisticalValidator(test_baseline_file)

        # Score right at 95th percentile threshold
        # Z ~ 1.645 → confidence ~ 0.95
        boundary_score = 0.15 + (1.645 * 0.10)  # mean + (z * std)

        z_score, confidence = validator.validate(boundary_score)

        assert z_score == pytest.approx(1.645, abs=0.01)
        assert confidence == pytest.approx(0.95, abs=0.01)

    def test_batch_validation(self, test_baseline_file):
        """Test validating a batch of documents."""
        validator = StatisticalValidator(test_baseline_file)

        # Batch of document scores
        scores = [0.08, 0.12, 0.15, 0.18, 0.25, 0.35, 0.45]

        results = []
        for score in scores:
            z, conf = validator.validate(score)
            label = validator.classify(score)
            results.append({'score': score, 'z': z, 'conf': conf, 'label': label})

        # Verify monotonic properties
        z_scores = [r['z'] for r in results]
        assert z_scores == sorted(z_scores)

        confidences = [r['conf'] for r in results]
        assert confidences == sorted(confidences)

        # Verify classification logic
        for r in results:
            if r['conf'] > 0.95:
                assert r['label'] == "WATERMARKED"
            elif r['conf'] < 0.05:
                assert r['label'] == "HUMAN"


class TestStatisticalValidatorStatelessBehavior:
    """Tests verifying stateless validation."""

    def test_multiple_validations_independent(self, test_baseline_file):
        """Test that multiple validations don't affect each other."""
        validator = StatisticalValidator(test_baseline_file)

        # First validation
        z1, conf1 = validator.validate(0.45)

        # Second validation with different score
        z2, conf2 = validator.validate(0.25)

        # Third validation same as first
        z3, conf3 = validator.validate(0.45)

        # First and third should be identical
        assert z1 == z3
        assert conf1 == conf3

        # Second should be different
        assert z1 != z2
        assert conf1 != conf2

    def test_reusable_validator(self, test_baseline_file):
        """Test that validator can be reused multiple times."""
        validator = StatisticalValidator(test_baseline_file)

        # Validate many scores
        for score in [0.1, 0.2, 0.3, 0.4, 0.5]:
            z, conf = validator.validate(score)

            # Each validation should produce valid results
            assert isinstance(z, float)
            assert isinstance(conf, float)
            assert 0.0 <= conf <= 1.0
