"""Z-score calculator for statistical validation.

This module implements the ZScoreCalculator class which converts document scores
to z-scores by normalizing against baseline human/natural text statistics.

The z-score represents "how many standard deviations above/below the human mean"
a document score is, enabling standardized comparison across different baselines
without arbitrary thresholds.

Tier 1 is a pure mathematical calculation with input validation only.

Tier: 1 (MVP)
Task: 6.2
Dependencies: None (stateless math operations)
"""


class ZScoreCalculator:
    """Calculate z-scores for statistical validation.

    The ZScoreCalculator converts raw document scores into z-scores using the
    formula: z = (x - μ) / σ

    This standardization enables:
    - Comparison across different baseline distributions
    - Threshold-free classification (use percentiles instead)
    - Probabilistic confidence estimation via normal CDF

    Tier 1 Implementation:
    - Simple z-score calculation
    - Input validation (positive standard deviation)
    - Stateless operation (no caching, no state)
    - Returns float (can be negative, zero, or positive)

    Z-score interpretation:
    - z < 0: Below human average (likely human text)
    - z = 0: Exactly at human average
    - z > 0: Above human average (potentially watermarked)
    - z > 2: >97.5th percentile (likely watermarked)
    - z > 3: >99.7th percentile (very likely watermarked)

    Examples:
        >>> calc = ZScoreCalculator()

        # Document score well above human baseline
        >>> z = calc.calculate_z_score(
        ...     document_score=0.45,
        ...     human_mean=0.15,
        ...     human_std=0.10
        ... )
        >>> print(f"{z:.2f}")
        3.00

        # Document score below human baseline
        >>> z = calc.calculate_z_score(
        ...     document_score=0.10,
        ...     human_mean=0.15,
        ...     human_std=0.10
        ... )
        >>> print(f"{z:.2f}")
        -0.50

        # Document score exactly at human mean
        >>> z = calc.calculate_z_score(
        ...     document_score=0.15,
        ...     human_mean=0.15,
        ...     human_std=0.10
        ... )
        >>> print(f"{z:.2f}")
        0.00
    """

    def calculate_z_score(
        self,
        document_score: float,
        human_mean: float,
        human_std: float
    ) -> float:
        """Calculate z-score using standard formula: (x - μ) / σ

        Converts a document score to standardized units (standard deviations
        from the human baseline mean). This enables threshold-free comparison
        and probabilistic confidence estimation.

        Args:
            document_score: Score from ScoringModule, range [0, 1]
            human_mean: Baseline mean from BaselineCorpusProcessor
            human_std: Baseline standard deviation from BaselineCorpusProcessor

        Returns:
            Z-score (float, unbounded):
            - Negative: Below human average (score < mean)
            - Zero: Exactly at human average (score = mean)
            - Positive: Above human average (score > mean)

            Typical values:
            - z < -2: Very likely human (below 2.5th percentile)
            - -2 ≤ z ≤ 2: Uncertain region (2.5th - 97.5th percentile)
            - z > 2: Likely watermarked (above 97.5th percentile)
            - z > 3: Very likely watermarked (above 99.7th percentile)

        Raises:
            ValueError: If human_std is zero or negative (would cause division
                       by zero or invalid statistics)

        Examples:
            >>> calc = ZScoreCalculator()

            # High score (3 std devs above mean)
            >>> z = calc.calculate_z_score(0.45, 0.15, 0.10)
            >>> print(f"{z:.1f}")
            3.0

            # Low score (0.5 std devs below mean)
            >>> z = calc.calculate_z_score(0.10, 0.15, 0.10)
            >>> print(f"{z:.1f}")
            -0.5

            # Score at mean (z = 0)
            >>> z = calc.calculate_z_score(0.15, 0.15, 0.10)
            >>> print(f"{z:.1f}")
            0.0

            # Very high score (5 std devs above mean)
            >>> z = calc.calculate_z_score(0.65, 0.15, 0.10)
            >>> print(f"{z:.1f}")
            5.0

        Notes:
            - Tier 1: No caching, no error recovery, simple validation
            - The z-score is unbounded (can be any real number)
            - For confidence levels, pass z-score to ConfidenceConverter
        """
        # Validate inputs
        if human_std <= 0:
            raise ValueError(
                f"Standard deviation must be positive, got {human_std}. "
                f"Cannot calculate z-score with zero or negative std."
            )

        # Calculate z-score: (x - μ) / σ
        z_score = (document_score - human_mean) / human_std

        return z_score
