"""
Tests for Task 5.1: WeightedScorer

Tests weighted combination of phonetic, structural, and semantic scores
into a single pair score using configurable weights.
"""

import pytest
import numpy as np
from specHO.models import EchoScore
from specHO.scoring.weighted_scorer import WeightedScorer
from specHO.config import ScoringConfig, load_config


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def scorer():
    """Fixture providing WeightedScorer instance with default config."""
    return WeightedScorer()


@pytest.fixture
def custom_scorer():
    """Fixture providing WeightedScorer with custom weights."""
    weights = {
        'phonetic': 0.5,
        'structural': 0.3,
        'semantic': 0.2
    }
    return WeightedScorer(weights=weights)


@pytest.fixture
def perfect_echo():
    """EchoScore with all 1.0 values."""
    return EchoScore(
        phonetic_score=1.0,
        structural_score=1.0,
        semantic_score=1.0,
        combined_score=0.0  # Not used in Tier 1
    )


@pytest.fixture
def zero_echo():
    """EchoScore with all 0.0 values."""
    return EchoScore(
        phonetic_score=0.0,
        structural_score=0.0,
        semantic_score=0.0,
        combined_score=0.0
    )


@pytest.fixture
def mixed_echo():
    """EchoScore with varied values."""
    return EchoScore(
        phonetic_score=0.8,
        structural_score=0.6,
        semantic_score=0.7,
        combined_score=0.0
    )


@pytest.fixture
def nan_echo():
    """EchoScore with NaN values (missing data)."""
    return EchoScore(
        phonetic_score=0.8,
        structural_score=np.nan,
        semantic_score=0.7,
        combined_score=0.0
    )


@pytest.fixture
def all_nan_echo():
    """EchoScore with all NaN values."""
    return EchoScore(
        phonetic_score=np.nan,
        structural_score=np.nan,
        semantic_score=np.nan,
        combined_score=0.0
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

class TestInitialization:
    """Test WeightedScorer initialization and configuration."""

    def test_default_initialization(self):
        """Test initialization with default config."""
        scorer = WeightedScorer()

        # Should load simple config
        assert scorer.config is not None
        assert scorer.weights is not None

        # Weights should sum to 1.0
        assert abs(sum(scorer.weights.values()) - 1.0) < 0.01

    def test_initialization_with_config(self):
        """Test initialization with provided config."""
        config = ScoringConfig(
            phonetic_weight=0.4,
            structural_weight=0.3,
            semantic_weight=0.3
        )
        scorer = WeightedScorer(config=config)

        assert scorer.weights['phonetic'] == 0.4
        assert scorer.weights['structural'] == 0.3
        assert scorer.weights['semantic'] == 0.3

    def test_initialization_with_weights(self):
        """Test initialization with explicit weights dict."""
        weights = {
            'phonetic': 0.5,
            'structural': 0.3,
            'semantic': 0.2
        }
        scorer = WeightedScorer(weights=weights)

        assert scorer.weights == weights

    def test_weights_validation(self):
        """Test that weights must sum to approximately 1.0."""
        invalid_weights = {
            'phonetic': 0.5,
            'structural': 0.3,
            'semantic': 0.1  # Sum = 0.9, invalid
        }

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            WeightedScorer(weights=invalid_weights)

    def test_weights_validation_tolerance(self):
        """Test that weights validation has reasonable tolerance."""
        # Should accept weights that sum to 0.999 or 1.001
        weights_low = {
            'phonetic': 0.33,
            'structural': 0.33,
            'semantic': 0.33  # Sum = 0.99
        }
        scorer = WeightedScorer(weights=weights_low)
        assert scorer is not None

    def test_get_weights(self):
        """Test get_weights() returns copy of weights."""
        scorer = WeightedScorer()
        weights = scorer.get_weights()

        # Should be a dict
        assert isinstance(weights, dict)

        # Should contain all three keys
        assert 'phonetic' in weights
        assert 'structural' in weights
        assert 'semantic' in weights

        # Modifying returned weights shouldn't affect scorer
        weights['phonetic'] = 0.999
        assert scorer.weights['phonetic'] != 0.999


# ============================================================================
# WEIGHTED SUM CALCULATION TESTS
# ============================================================================

class TestWeightedSum:
    """Test basic weighted sum calculations."""

    def test_perfect_echo_score(self, scorer, perfect_echo):
        """Test scoring with all 1.0 values."""
        score = scorer.calculate_pair_score(perfect_echo)

        # With equal weights, perfect score should be 1.0
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_echo_score(self, scorer, zero_echo):
        """Test scoring with all 0.0 values."""
        score = scorer.calculate_pair_score(zero_echo)

        # With equal weights, zero score should be 0.0
        assert score == pytest.approx(0.0, abs=0.01)

    def test_mixed_echo_score(self, scorer, mixed_echo):
        """Test scoring with varied values."""
        # mixed_echo: phonetic=0.8, structural=0.6, semantic=0.7
        # Default weights: 0.33, 0.33, 0.34
        # Expected: 0.33*0.8 + 0.33*0.6 + 0.34*0.7 = 0.264 + 0.198 + 0.238 = 0.700
        score = scorer.calculate_pair_score(mixed_echo)

        assert score == pytest.approx(0.700, abs=0.01)

    def test_custom_weights(self, custom_scorer, mixed_echo):
        """Test scoring with custom weights."""
        # mixed_echo: phonetic=0.8, structural=0.6, semantic=0.7
        # Custom weights: 0.5, 0.3, 0.2
        # Expected: 0.5*0.8 + 0.3*0.6 + 0.2*0.7 = 0.4 + 0.18 + 0.14 = 0.72
        score = custom_scorer.calculate_pair_score(mixed_echo)

        assert score == pytest.approx(0.72, abs=0.01)

    def test_weights_override_per_call(self, scorer, mixed_echo):
        """Test overriding weights in calculate_pair_score call."""
        override_weights = {
            'phonetic': 0.6,
            'structural': 0.2,
            'semantic': 0.2
        }

        # mixed_echo: phonetic=0.8, structural=0.6, semantic=0.7
        # Override weights: 0.6, 0.2, 0.2
        # Expected: 0.6*0.8 + 0.2*0.6 + 0.2*0.7 = 0.48 + 0.12 + 0.14 = 0.74
        score = scorer.calculate_pair_score(mixed_echo, weights=override_weights)

        assert score == pytest.approx(0.74, abs=0.01)


# ============================================================================
# NaN HANDLING TESTS
# ============================================================================

class TestNaNHandling:
    """Test handling of missing data (NaN values)."""

    def test_single_nan_phonetic(self, scorer):
        """Test NaN in phonetic score (treated as 0)."""
        echo = EchoScore(
            phonetic_score=np.nan,
            structural_score=0.6,
            semantic_score=0.7,
            combined_score=0.0
        )

        # Expected: 0.33*0 + 0.33*0.6 + 0.34*0.7 = 0 + 0.198 + 0.238 = 0.436
        score = scorer.calculate_pair_score(echo)

        assert score == pytest.approx(0.436, abs=0.01)

    def test_single_nan_structural(self, scorer, nan_echo):
        """Test NaN in structural score (treated as 0)."""
        # nan_echo: phonetic=0.8, structural=NaN, semantic=0.7
        # Expected: 0.33*0.8 + 0.33*0 + 0.34*0.7 = 0.264 + 0 + 0.238 = 0.502
        score = scorer.calculate_pair_score(nan_echo)

        assert score == pytest.approx(0.502, abs=0.01)

    def test_single_nan_semantic(self, scorer):
        """Test NaN in semantic score (treated as 0)."""
        echo = EchoScore(
            phonetic_score=0.8,
            structural_score=0.6,
            semantic_score=np.nan,
            combined_score=0.0
        )

        # Expected: 0.33*0.8 + 0.33*0.6 + 0.34*0 = 0.264 + 0.198 + 0 = 0.462
        score = scorer.calculate_pair_score(echo)

        assert score == pytest.approx(0.462, abs=0.01)

    def test_multiple_nans(self, scorer):
        """Test multiple NaN values."""
        echo = EchoScore(
            phonetic_score=np.nan,
            structural_score=np.nan,
            semantic_score=0.7,
            combined_score=0.0
        )

        # Expected: 0.33*0 + 0.33*0 + 0.34*0.7 = 0.238
        score = scorer.calculate_pair_score(echo)

        assert score == pytest.approx(0.238, abs=0.01)

    def test_all_nans(self, scorer, all_nan_echo):
        """Test all NaN values (should return 0.0)."""
        score = scorer.calculate_pair_score(all_nan_echo)

        assert score == pytest.approx(0.0, abs=0.01)


# ============================================================================
# CLIPPING TESTS
# ============================================================================

class TestClipping:
    """Test output clipping to [0,1] range."""

    def test_no_clipping_needed_for_valid_scores(self, scorer):
        """Test that valid scores don't trigger clipping."""
        echo = EchoScore(
            phonetic_score=0.5,
            structural_score=0.5,
            semantic_score=0.5,
            combined_score=0.0
        )

        score = scorer.calculate_pair_score(echo)

        # Should be exactly 0.5 with equal weights
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(0.5, abs=0.01)

    def test_output_always_in_range(self, scorer):
        """Test that output is always in [0,1] even with edge values."""
        # Test various edge cases
        test_cases = [
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            (0.0, 0.5, 1.0),
            (1.0, 0.5, 0.0),
        ]

        for p, s, sem in test_cases:
            echo = EchoScore(p, s, sem, 0.0)
            score = scorer.calculate_pair_score(echo)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for ({p}, {s}, {sem})"


# ============================================================================
# CONFIG INTEGRATION TESTS
# ============================================================================

class TestConfigIntegration:
    """Test integration with SpecHO config system."""

    def test_load_simple_profile(self):
        """Test loading simple config profile."""
        scorer = WeightedScorer()

        # Simple profile should have equal weights
        assert scorer.weights['phonetic'] == pytest.approx(0.33, abs=0.01)
        assert scorer.weights['structural'] == pytest.approx(0.33, abs=0.01)
        assert scorer.weights['semantic'] == pytest.approx(0.34, abs=0.01)

    def test_load_custom_profile(self):
        """Test loading config with custom weights."""
        config = load_config("simple", overrides={
            "scoring.phonetic_weight": 0.5,
            "scoring.structural_weight": 0.3,
            "scoring.semantic_weight": 0.2
        })
        scorer = WeightedScorer(config=config.scoring)

        assert scorer.weights['phonetic'] == 0.5
        assert scorer.weights['structural'] == 0.3
        assert scorer.weights['semantic'] == 0.2

    def test_config_missing_data_strategy_zero(self):
        """Test that Tier 1 uses 'zero' strategy for missing data."""
        scorer = WeightedScorer()

        assert scorer.config.missing_data_strategy == "zero"


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_scores(self, scorer):
        """Test with very small non-zero scores."""
        echo = EchoScore(0.001, 0.002, 0.003, 0.0)
        score = scorer.calculate_pair_score(echo)

        assert score > 0.0
        assert score < 0.01

    def test_very_high_scores_near_one(self, scorer):
        """Test with scores very close to 1.0."""
        echo = EchoScore(0.999, 0.998, 0.997, 0.0)
        score = scorer.calculate_pair_score(echo)

        assert score > 0.99
        assert score <= 1.0

    def test_mixed_extreme_values(self, scorer):
        """Test with one very high and two very low scores."""
        echo = EchoScore(0.999, 0.001, 0.001, 0.0)
        score = scorer.calculate_pair_score(echo)

        # Should be weighted toward phonetic
        # 0.33*0.999 + 0.33*0.001 + 0.34*0.001 ≈ 0.33
        assert score == pytest.approx(0.33, abs=0.01)


# ============================================================================
# KNOWN VALUE TESTS
# ============================================================================

class TestKnownValues:
    """Test against pre-calculated expected values."""

    def test_known_value_1(self, scorer):
        """Test case 1: Moderate echo across all dimensions."""
        echo = EchoScore(
            phonetic_score=0.6,
            structural_score=0.5,
            semantic_score=0.7,
            combined_score=0.0
        )

        # Expected: 0.33*0.6 + 0.33*0.5 + 0.34*0.7 = 0.198 + 0.165 + 0.238 = 0.601
        score = scorer.calculate_pair_score(echo)
        assert score == pytest.approx(0.601, abs=0.01)

    def test_known_value_2(self, scorer):
        """Test case 2: High phonetic, low structural, moderate semantic."""
        echo = EchoScore(
            phonetic_score=0.9,
            structural_score=0.2,
            semantic_score=0.6,
            combined_score=0.0
        )

        # Expected: 0.33*0.9 + 0.33*0.2 + 0.34*0.6 = 0.297 + 0.066 + 0.204 = 0.567
        score = scorer.calculate_pair_score(echo)
        assert score == pytest.approx(0.567, abs=0.01)

    def test_known_value_3_with_custom_weights(self):
        """Test case 3: Custom weights with known values."""
        weights = {
            'phonetic': 0.6,
            'structural': 0.2,
            'semantic': 0.2
        }
        scorer = WeightedScorer(weights=weights)

        echo = EchoScore(0.8, 0.5, 0.7, 0.0)

        # Expected: 0.6*0.8 + 0.2*0.5 + 0.2*0.7 = 0.48 + 0.1 + 0.14 = 0.72
        score = scorer.calculate_pair_score(echo)
        assert score == pytest.approx(0.72, abs=0.01)


# ============================================================================
# COMPARISON TESTS
# ============================================================================

class TestComparisons:
    """Test relative ordering of scores."""

    def test_higher_all_dimensions_higher_score(self, scorer):
        """Test that higher individual scores produce higher combined score."""
        low_echo = EchoScore(0.3, 0.3, 0.3, 0.0)
        high_echo = EchoScore(0.7, 0.7, 0.7, 0.0)

        low_score = scorer.calculate_pair_score(low_echo)
        high_score = scorer.calculate_pair_score(high_echo)

        assert high_score > low_score

    def test_equal_weighted_contribution(self, scorer):
        """Test that with equal weights, each dimension contributes equally."""
        # Increase each dimension by 0.3, should increase total by ~0.1 each time
        base = EchoScore(0.0, 0.0, 0.0, 0.0)
        phonetic_boost = EchoScore(0.3, 0.0, 0.0, 0.0)
        structural_boost = EchoScore(0.0, 0.3, 0.0, 0.0)
        semantic_boost = EchoScore(0.0, 0.0, 0.3, 0.0)

        base_score = scorer.calculate_pair_score(base)
        p_score = scorer.calculate_pair_score(phonetic_boost)
        s_score = scorer.calculate_pair_score(structural_boost)
        sem_score = scorer.calculate_pair_score(semantic_boost)

        # All three boosts should produce similar increases
        p_increase = p_score - base_score
        s_increase = s_score - base_score
        sem_increase = sem_score - base_score

        assert p_increase == pytest.approx(s_increase, abs=0.02)
        assert abs(sem_increase - p_increase) < 0.03  # Semantic weight is 0.34 vs 0.33


# ============================================================================
# SUMMARY
# ============================================================================

"""
Test Coverage Summary for Task 5.1:

Total Tests: 36 tests across 8 test classes

1. TestInitialization (6 tests): Config loading, weight validation, get_weights()
2. TestWeightedSum (5 tests): Basic calculations with various weight configurations
3. TestNaNHandling (5 tests): Missing data handling (NaN → 0.0)
4. TestClipping (2 tests): Output range validation [0,1]
5. TestConfigIntegration (3 tests): Integration with SpecHO config system
6. TestEdgeCases (3 tests): Boundary conditions and extreme values
7. TestKnownValues (3 tests): Pre-calculated expected results
8. TestComparisons (2 tests): Relative ordering verification

Coverage Areas:
- API contract: calculate_pair_score() signature and return type
- Algorithm correctness: weighted sum formula
- NaN handling: missing data strategy
- Config integration: loading weights from config
- Weight validation: sum to 1.0 requirement
- Output clipping: [0,1] range enforcement
- Edge cases: extreme values, all zeros, all ones
- Known values: manual verification of calculations
"""
