"""Tests for ZScoreCalculator (Task 6.2).

This test suite validates the z-score calculation functionality for statistical
validation. Tests cover basic calculations, edge cases, input validation, and
numerical precision.

Test Categories:
- Basic calculation (positive, negative, zero z-scores)
- Edge cases (score = mean, very high/low scores)
- Input validation (zero std, negative std)
- Numerical precision (float accuracy)

Tier: 1 (MVP)
Task: 6.2
"""

import pytest
from specHO.validator.z_score import ZScoreCalculator


class TestZScoreCalculatorBasicCalculation:
    """Tests for basic z-score calculations."""

    def test_positive_z_score(self):
        """Test calculation when document score is above mean."""
        calc = ZScoreCalculator()

        # Score 3 std devs above mean
        z = calc.calculate_z_score(
            document_score=0.45,
            human_mean=0.15,
            human_std=0.10
        )

        assert z == pytest.approx(3.0, abs=1e-10)

    def test_negative_z_score(self):
        """Test calculation when document score is below mean."""
        calc = ZScoreCalculator()

        # Score 0.5 std devs below mean
        z = calc.calculate_z_score(
            document_score=0.10,
            human_mean=0.15,
            human_std=0.10
        )

        assert z == pytest.approx(-0.5, abs=1e-10)

    def test_zero_z_score(self):
        """Test calculation when document score equals mean."""
        calc = ZScoreCalculator()

        # Score exactly at mean
        z = calc.calculate_z_score(
            document_score=0.15,
            human_mean=0.15,
            human_std=0.10
        )

        assert z == pytest.approx(0.0, abs=1e-10)

    def test_large_positive_z_score(self):
        """Test calculation for very high scores."""
        calc = ZScoreCalculator()

        # Score 5 std devs above mean
        z = calc.calculate_z_score(
            document_score=0.65,
            human_mean=0.15,
            human_std=0.10
        )

        assert z == pytest.approx(5.0, abs=1e-10)

    def test_large_negative_z_score(self):
        """Test calculation for very low scores."""
        calc = ZScoreCalculator()

        # Score 3 std devs below mean
        z = calc.calculate_z_score(
            document_score=0.01,
            human_mean=0.10,
            human_std=0.03
        )

        assert z == pytest.approx(-3.0, abs=1e-10)


class TestZScoreCalculatorEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_score_equals_zero(self):
        """Test with document score of zero."""
        calc = ZScoreCalculator()

        z = calc.calculate_z_score(
            document_score=0.0,
            human_mean=0.15,
            human_std=0.10
        )

        assert z == pytest.approx(-1.5, abs=1e-10)

    def test_score_equals_one(self):
        """Test with maximum document score (1.0)."""
        calc = ZScoreCalculator()

        z = calc.calculate_z_score(
            document_score=1.0,
            human_mean=0.15,
            human_std=0.10
        )

        assert z == pytest.approx(8.5, abs=1e-10)

    def test_very_small_std(self):
        """Test with very small (but valid) standard deviation."""
        calc = ZScoreCalculator()

        # Small std amplifies z-score
        z = calc.calculate_z_score(
            document_score=0.20,
            human_mean=0.15,
            human_std=0.01
        )

        assert z == pytest.approx(5.0, abs=1e-10)

    def test_very_large_std(self):
        """Test with very large standard deviation."""
        calc = ZScoreCalculator()

        # Large std dampens z-score
        z = calc.calculate_z_score(
            document_score=0.50,
            human_mean=0.15,
            human_std=0.50
        )

        assert z == pytest.approx(0.7, abs=1e-10)

    def test_realistic_baseline_values(self):
        """Test with realistic baseline statistics."""
        calc = ZScoreCalculator()

        # Typical human baseline: mean=0.18, std=0.08
        # Watermarked score: 0.42
        z = calc.calculate_z_score(
            document_score=0.42,
            human_mean=0.18,
            human_std=0.08
        )

        assert z == pytest.approx(3.0, abs=1e-10)


class TestZScoreCalculatorInputValidation:
    """Tests for input validation and error handling."""

    def test_zero_std_raises_error(self):
        """Test that zero standard deviation raises ValueError."""
        calc = ZScoreCalculator()

        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            calc.calculate_z_score(
                document_score=0.45,
                human_mean=0.15,
                human_std=0.0
            )

    def test_negative_std_raises_error(self):
        """Test that negative standard deviation raises ValueError."""
        calc = ZScoreCalculator()

        with pytest.raises(ValueError, match="Standard deviation must be positive"):
            calc.calculate_z_score(
                document_score=0.45,
                human_mean=0.15,
                human_std=-0.10
            )

    def test_error_message_includes_invalid_value(self):
        """Test that error message includes the invalid std value."""
        calc = ZScoreCalculator()

        with pytest.raises(ValueError, match="got -0.05"):
            calc.calculate_z_score(
                document_score=0.45,
                human_mean=0.15,
                human_std=-0.05
            )


class TestZScoreCalculatorNumericalPrecision:
    """Tests for numerical precision and float accuracy."""

    def test_float_precision(self):
        """Test that z-score calculation maintains float precision."""
        calc = ZScoreCalculator()

        # Use non-round numbers
        z = calc.calculate_z_score(
            document_score=0.33333,
            human_mean=0.12345,
            human_std=0.06789
        )

        expected = (0.33333 - 0.12345) / 0.06789
        assert z == pytest.approx(expected, abs=1e-10)

    def test_multiple_decimal_places(self):
        """Test calculation with many decimal places."""
        calc = ZScoreCalculator()

        z = calc.calculate_z_score(
            document_score=0.456789,
            human_mean=0.123456,
            human_std=0.098765
        )

        expected = (0.456789 - 0.123456) / 0.098765
        assert z == pytest.approx(expected, abs=1e-10)

    def test_symmetry(self):
        """Test that z-scores are symmetric around the mean."""
        calc = ZScoreCalculator()

        mean = 0.15
        std = 0.10

        # Score 0.10 above mean
        z_above = calc.calculate_z_score(0.25, mean, std)

        # Score 0.10 below mean
        z_below = calc.calculate_z_score(0.05, mean, std)

        # Should have equal magnitude, opposite sign
        assert z_above == pytest.approx(-z_below, abs=1e-10)
        assert z_above > 0
        assert z_below < 0


class TestZScoreCalculatorStatelessBehavior:
    """Tests verifying stateless operation."""

    def test_multiple_calls_independent(self):
        """Test that multiple calls don't affect each other."""
        calc = ZScoreCalculator()

        # First call
        z1 = calc.calculate_z_score(0.45, 0.15, 0.10)

        # Second call with different values
        z2 = calc.calculate_z_score(0.25, 0.15, 0.10)

        # Third call same as first
        z3 = calc.calculate_z_score(0.45, 0.15, 0.10)

        # First and third should be identical
        assert z1 == z3
        # Second should be different
        assert z1 != z2

    def test_reusable_instance(self):
        """Test that same instance can be reused multiple times."""
        calc = ZScoreCalculator()

        results = []
        for score in [0.10, 0.20, 0.30, 0.40, 0.50]:
            z = calc.calculate_z_score(score, 0.15, 0.10)
            results.append(z)

        # All results should be different
        assert len(results) == len(set(results))

        # Results should be monotonically increasing
        assert results == sorted(results)
