"""Tests for ConfidenceConverter (Task 6.3).

This test suite validates the confidence conversion functionality for statistical
validation. Tests cover basic conversions, symmetry, extreme values, known statistical
values, and output range validation.

Test Categories:
- Basic conversion (z=0, z=1, z=2, z=3)
- Symmetry (z vs -z should sum to 1.0)
- Extreme values (z=10, z=-10)
- Known values (z=1.96 → 0.975 for 95% confidence)
- Output range validation (always in [0, 1])
- Percentile conversion

Tier: 1 (MVP)
Task: 6.3
"""

import pytest
from specHO.validator.confidence import ConfidenceConverter


class TestConfidenceConverterBasicConversion:
    """Tests for basic confidence conversions."""

    def test_z_score_zero(self):
        """Test that z=0 gives confidence=0.5 (50th percentile)."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(0.0)

        assert conf == pytest.approx(0.5, abs=1e-10)

    def test_z_score_one(self):
        """Test that z=1 gives confidence≈0.841 (84.1st percentile)."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(1.0)

        # Z=1 corresponds to 84.134th percentile
        assert conf == pytest.approx(0.8413, abs=1e-3)

    def test_z_score_two(self):
        """Test that z=2 gives confidence≈0.977 (97.7th percentile)."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(2.0)

        # Z=2 corresponds to 97.724th percentile
        assert conf == pytest.approx(0.9772, abs=1e-3)

    def test_z_score_three(self):
        """Test that z=3 gives confidence≈0.999 (99.9th percentile)."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(3.0)

        # Z=3 corresponds to 99.865th percentile
        assert conf == pytest.approx(0.9987, abs=1e-3)

    def test_negative_z_score(self):
        """Test that negative z-scores give low confidence."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(-1.0)

        # Z=-1 corresponds to 15.866th percentile
        assert conf == pytest.approx(0.1587, abs=1e-3)


class TestConfidenceConverterSymmetry:
    """Tests for symmetry properties of the normal distribution."""

    def test_symmetry_around_zero(self):
        """Test that norm.cdf(z) + norm.cdf(-z) = 1.0."""
        converter = ConfidenceConverter()

        for z in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            conf_pos = converter.convert_to_confidence(z)
            conf_neg = converter.convert_to_confidence(-z)

            assert conf_pos + conf_neg == pytest.approx(1.0, abs=1e-10)

    def test_symmetry_with_specific_values(self):
        """Test symmetry with specific z-scores."""
        converter = ConfidenceConverter()

        # Z=1 and Z=-1 should be symmetric
        conf_1 = converter.convert_to_confidence(1.0)
        conf_neg1 = converter.convert_to_confidence(-1.0)

        assert conf_1 == pytest.approx(1.0 - conf_neg1, abs=1e-10)

    def test_zero_is_symmetric_point(self):
        """Test that z=0 is the point of symmetry (0.5)."""
        converter = ConfidenceConverter()

        conf_0 = converter.convert_to_confidence(0.0)

        assert conf_0 == pytest.approx(0.5, abs=1e-10)


class TestConfidenceConverterExtremeValues:
    """Tests for extreme z-score values."""

    def test_very_high_z_score(self):
        """Test that very high z-scores approach 1.0."""
        converter = ConfidenceConverter()

        conf_10 = converter.convert_to_confidence(10.0)

        # Z=10 should be extremely close to 1.0 (or exactly 1.0 due to float precision)
        assert conf_10 >= 0.999999
        assert conf_10 <= 1.0

    def test_very_low_z_score(self):
        """Test that very low z-scores approach 0.0."""
        converter = ConfidenceConverter()

        conf_neg10 = converter.convert_to_confidence(-10.0)

        # Z=-10 should be extremely close to 0.0
        assert conf_neg10 < 0.000001
        assert conf_neg10 > 0.0

    def test_extreme_positive(self):
        """Test with z=5 (far above mean)."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(5.0)

        # Z=5 is >99.9999% confidence
        assert conf > 0.99999

    def test_extreme_negative(self):
        """Test with z=-5 (far below mean)."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(-5.0)

        # Z=-5 is <0.0001% confidence
        assert conf < 0.00001


class TestConfidenceConverterKnownValues:
    """Tests for statistically significant known z-score values."""

    def test_95th_percentile(self):
        """Test z=1.645 corresponds to 95th percentile."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(1.645)

        assert conf == pytest.approx(0.95, abs=1e-2)

    def test_97_5th_percentile(self):
        """Test z=1.96 corresponds to 97.5th percentile (2-sided 95% CI)."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(1.96)

        assert conf == pytest.approx(0.975, abs=1e-3)

    def test_99th_percentile(self):
        """Test z=2.326 corresponds to 99th percentile."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(2.326)

        assert conf == pytest.approx(0.99, abs=1e-2)

    def test_99_9th_percentile(self):
        """Test z=3.09 corresponds to 99.9th percentile."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(3.09)

        assert conf == pytest.approx(0.999, abs=1e-3)


class TestConfidenceConverterOutputRange:
    """Tests verifying output is always in valid [0, 1] range."""

    def test_output_range_for_typical_values(self):
        """Test that typical z-scores give confidence in [0, 1]."""
        converter = ConfidenceConverter()

        for z in [-3, -2, -1, 0, 1, 2, 3]:
            conf = converter.convert_to_confidence(float(z))

            assert 0.0 <= conf <= 1.0

    def test_output_range_for_extreme_values(self):
        """Test that extreme z-scores still give valid confidence."""
        converter = ConfidenceConverter()

        for z in [-100, -10, -5, 5, 10, 100]:
            conf = converter.convert_to_confidence(float(z))

            assert 0.0 <= conf <= 1.0

    def test_output_is_float_type(self):
        """Test that output is Python float, not numpy.float64."""
        converter = ConfidenceConverter()

        conf = converter.convert_to_confidence(1.5)

        assert isinstance(conf, float)
        assert type(conf).__name__ == 'float'


class TestConfidenceConverterPercentileConversion:
    """Tests for z_score_to_percentile() method."""

    def test_percentile_at_mean(self):
        """Test that z=0 gives 50th percentile."""
        converter = ConfidenceConverter()

        percentile = converter.z_score_to_percentile(0.0)

        assert percentile == pytest.approx(50.0, abs=1e-10)

    def test_percentile_one_std_above(self):
        """Test that z=1 gives ~84th percentile."""
        converter = ConfidenceConverter()

        percentile = converter.z_score_to_percentile(1.0)

        assert percentile == pytest.approx(84.13, abs=0.5)

    def test_percentile_two_std_above(self):
        """Test that z=2 gives ~97.7th percentile."""
        converter = ConfidenceConverter()

        percentile = converter.z_score_to_percentile(2.0)

        assert percentile == pytest.approx(97.72, abs=0.5)

    def test_percentile_95th(self):
        """Test 95th percentile corresponds to z≈1.645."""
        converter = ConfidenceConverter()

        percentile = converter.z_score_to_percentile(1.645)

        assert percentile == pytest.approx(95.0, abs=1.0)

    def test_percentile_range(self):
        """Test that percentile is always in [0, 100] range."""
        converter = ConfidenceConverter()

        for z in [-5, -2, 0, 2, 5]:
            percentile = converter.z_score_to_percentile(float(z))

            assert 0.0 <= percentile <= 100.0


class TestConfidenceConverterStatelessBehavior:
    """Tests verifying stateless operation."""

    def test_multiple_calls_independent(self):
        """Test that multiple calls don't affect each other."""
        converter = ConfidenceConverter()

        # First call
        conf1 = converter.convert_to_confidence(2.0)

        # Second call with different value
        conf2 = converter.convert_to_confidence(1.0)

        # Third call same as first
        conf3 = converter.convert_to_confidence(2.0)

        # First and third should be identical
        assert conf1 == conf3

        # Second should be different
        assert conf1 != conf2

    def test_reusable_instance(self):
        """Test that same instance can be reused multiple times."""
        converter = ConfidenceConverter()

        results = []
        for z in [-2, -1, 0, 1, 2]:
            conf = converter.convert_to_confidence(float(z))
            results.append(conf)

        # All results should be different
        assert len(results) == len(set(results))

        # Results should be monotonically increasing
        assert results == sorted(results)


class TestConfidenceConverterIntegrationWithZScore:
    """Tests simulating integration with ZScoreCalculator."""

    def test_realistic_watermark_detection_scenario(self):
        """Test realistic watermark detection scenario."""
        converter = ConfidenceConverter()

        # Scenario: Document score = 0.45, human baseline = 0.15±0.10
        # Z-score = (0.45 - 0.15) / 0.10 = 3.0
        z_score = 3.0

        conf = converter.convert_to_confidence(z_score)

        # Z=3 is >99.7th percentile, very likely watermarked
        assert conf > 0.997

    def test_realistic_human_text_scenario(self):
        """Test realistic human text scenario."""
        converter = ConfidenceConverter()

        # Scenario: Document score = 0.10, human baseline = 0.15±0.10
        # Z-score = (0.10 - 0.15) / 0.10 = -0.5
        z_score = -0.5

        conf = converter.convert_to_confidence(z_score)

        # Z=-0.5 is ~30th percentile, likely human
        assert 0.25 < conf < 0.35

    def test_uncertain_region(self):
        """Test scores in uncertain region (z between -2 and 2)."""
        converter = ConfidenceConverter()

        # Uncertain region: -2 ≤ z ≤ 2
        for z in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
            conf = converter.convert_to_confidence(z)

            # Should be between 2.5th and 97.5th percentile
            assert 0.025 < conf < 0.975
