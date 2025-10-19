"""Confidence converter for statistical validation.

This module implements the ConfidenceConverter class which converts z-scores
to confidence levels using the cumulative distribution function (CDF) of the
standard normal distribution.

The confidence level represents P(score ≤ document_score | human distribution),
providing a probabilistic interpretation of how likely a score came from the
non-watermarked (human) distribution.

Tier 1 uses scipy.stats.norm.cdf for conversion with no caching.

Tier: 1 (MVP)
Task: 6.3
Dependencies: scipy.stats (for normal CDF)
"""

from scipy.stats import norm


class ConfidenceConverter:
    """Convert z-scores to confidence levels using normal CDF.

    The ConfidenceConverter transforms z-scores into confidence levels in the
    range [0, 1] using the cumulative distribution function (CDF) of the
    standard normal distribution.

    Mathematical foundation:
        confidence = Φ(z) = ∫[-∞, z] φ(t) dt

    Where:
        Φ(z) = CDF of standard normal distribution
        φ(t) = PDF of standard normal distribution

    Tier 1 Implementation:
    - Convert z-score to confidence using scipy.stats.norm.cdf
    - Returns probability in [0, 1] range
    - No caching (recalculate each time)
    - Stateless operation

    Confidence interpretation:
    - confidence < 0.05: Likely human (below 5th percentile)
    - 0.05 ≤ confidence ≤ 0.95: Uncertain region
    - confidence > 0.95: Likely watermarked (above 95th percentile)
    - confidence > 0.99: Very likely watermarked (above 99th percentile)

    Common z-score to confidence mappings:
    - z = -3 → confidence = 0.001 (0.1st percentile)
    - z = -2 → confidence = 0.023 (2.3rd percentile)
    - z = -1 → confidence = 0.159 (15.9th percentile)
    - z = 0  → confidence = 0.500 (50th percentile, exactly at mean)
    - z = 1  → confidence = 0.841 (84.1st percentile)
    - z = 2  → confidence = 0.977 (97.7th percentile)
    - z = 3  → confidence = 0.999 (99.9th percentile)

    Examples:
        >>> converter = ConfidenceConverter()

        # Z-score of 0 (at mean) → 50% confidence
        >>> conf = converter.convert_to_confidence(0.0)
        >>> print(f"{conf:.3f}")
        0.500

        # Z-score of 2 (95th percentile) → ~97.7% confidence
        >>> conf = converter.convert_to_confidence(2.0)
        >>> print(f"{conf:.3f}")
        0.977

        # Z-score of 3 (very high) → ~99.9% confidence
        >>> conf = converter.convert_to_confidence(3.0)
        >>> print(f"{conf:.3f}")
        0.999

        # Negative z-score (below mean) → low confidence
        >>> conf = converter.convert_to_confidence(-2.0)
        >>> print(f"{conf:.3f}")
        0.023
    """

    def convert_to_confidence(self, z_score: float) -> float:
        """Convert z-score to confidence level using standard normal CDF.

        Uses scipy.stats.norm.cdf to calculate the area under the standard
        normal curve from -∞ to z. This represents the probability that a
        random value from the standard normal distribution is ≤ z.

        In the watermark detection context:
        - High confidence (>0.95): Score is unusually high for human text
        - Low confidence (<0.05): Score is unusually low for human text
        - Medium confidence (0.05-0.95): Score is within normal human range

        Args:
            z_score: Z-score from ZScoreCalculator (can be any real number)

        Returns:
            Confidence level in [0, 1]:
            - 0.0: Extremely unlikely (z → -∞, far below human mean)
            - 0.5: Exactly at human mean (z = 0)
            - 1.0: Extremely likely (z → +∞, far above human mean)

            Interpretation for watermark detection:
            - conf > 0.99: Very likely watermarked (>99th percentile)
            - conf > 0.95: Likely watermarked (>95th percentile)
            - 0.05 ≤ conf ≤ 0.95: Uncertain (normal human range)
            - conf < 0.05: Likely human (below 5th percentile)

        Examples:
            >>> converter = ConfidenceConverter()

            # At mean (z=0)
            >>> conf = converter.convert_to_confidence(0.0)
            >>> print(f"{conf:.3f}")
            0.500

            # 1 std dev above mean (z=1)
            >>> conf = converter.convert_to_confidence(1.0)
            >>> print(f"{conf:.3f}")
            0.841

            # 2 std devs above mean (z=2, 95% confidence)
            >>> conf = converter.convert_to_confidence(2.0)
            >>> print(f"{conf:.3f}")
            0.977

            # 3 std devs above mean (z=3, 99.7% confidence)
            >>> conf = converter.convert_to_confidence(3.0)
            >>> print(f"{conf:.3f}")
            0.999

            # 1 std dev below mean (z=-1)
            >>> conf = converter.convert_to_confidence(-1.0)
            >>> print(f"{conf:.3f}")
            0.159

            # Very high z-score
            >>> conf = converter.convert_to_confidence(5.0)
            >>> print(f"{conf:.6f}")
            0.999999

            # Very low z-score
            >>> conf = converter.convert_to_confidence(-5.0)
            >>> print(f"{conf:.6f}")
            0.000001

        Notes:
            - Tier 1: No caching, no error recovery
            - The result is always in [0, 1] range (guaranteed by CDF properties)
            - Symmetry: norm.cdf(-z) = 1 - norm.cdf(z)
            - For threshold-based classification, common cutoffs are 0.95, 0.975, 0.99
        """
        # Calculate confidence using standard normal CDF
        confidence = norm.cdf(z_score)

        # Convert to Python float (from numpy.float64)
        return float(confidence)

    def z_score_to_percentile(self, z_score: float) -> float:
        """Convert z-score to percentile (0-100 scale).

        This is a convenience method that returns the percentile rather than
        the probability. It's simply confidence * 100.

        Args:
            z_score: Z-score from ZScoreCalculator

        Returns:
            Percentile in [0, 100]:
            - 0: Lowest possible (z → -∞)
            - 50: At mean (z = 0)
            - 100: Highest possible (z → +∞)

        Examples:
            >>> converter = ConfidenceConverter()

            # At mean (50th percentile)
            >>> percentile = converter.z_score_to_percentile(0.0)
            >>> print(f"{percentile:.1f}")
            50.0

            # 95th percentile
            >>> percentile = converter.z_score_to_percentile(1.645)
            >>> print(f"{percentile:.1f}")
            95.0

            # 97.5th percentile (z=1.96)
            >>> percentile = converter.z_score_to_percentile(1.96)
            >>> print(f"{percentile:.1f}")
            97.5

            # 99th percentile
            >>> percentile = converter.z_score_to_percentile(2.326)
            >>> print(f"{percentile:.1f}")
            99.0

        Notes:
            - This is identical to convert_to_confidence() * 100
            - Provided for convenience and clarity
            - Tier 1: No caching, recalculate each time
        """
        confidence = self.convert_to_confidence(z_score)
        return confidence * 100.0
