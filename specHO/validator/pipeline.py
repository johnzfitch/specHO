"""Statistical validation pipeline orchestrator.

This module implements the StatisticalValidator class which orchestrates the complete
statistical validation workflow from document score to confidence level.

The pipeline chains:
1. Baseline loading (BaselineCorpusProcessor)
2. Z-score calculation (ZScoreCalculator)
3. Confidence conversion (ConfidenceConverter)

Tier 1 is a simple orchestrator with baseline loading and sequential component calls.

Tier: 1 (MVP)
Task: 6.4
Dependencies: Task 6.1 (BaselineCorpusProcessor), Task 6.2 (ZScoreCalculator),
              Task 6.3 (ConfidenceConverter)
"""

from pathlib import Path
from typing import Tuple, Dict

from .baseline_builder import BaselineCorpusProcessor
from .z_score import ZScoreCalculator
from .confidence import ConfidenceConverter


class StatisticalValidator:
    """Orchestrates statistical validation pipeline.

    The StatisticalValidator provides a unified interface for converting document
    scores into confidence levels by comparing against baseline distributions. It
    chains three components:
    1. BaselineCorpusProcessor: Load baseline statistics (mean, std)
    2. ZScoreCalculator: Convert score → z-score
    3. ConfidenceConverter: Convert z-score → confidence

    Tier 1 Implementation:
    - Load baseline statistics from pickle file at initialization
    - Orchestrate ZScoreCalculator and ConfidenceConverter in sequence
    - Return both z-score and confidence for transparency
    - Simple error handling (missing baseline file)

    This orchestrator pattern keeps the validation pipeline modular and testable
    while providing a clean API for downstream consumers (SpecHODetector).

    Workflow:
        document_score (0-1 range)
            ↓
        ZScoreCalculator: z = (score - μ) / σ
            ↓
        z-score (standard deviations from mean)
            ↓
        ConfidenceConverter: conf = Φ(z)
            ↓
        confidence (0-1 probability)

    Examples:
        >>> # Initialize with baseline
        >>> validator = StatisticalValidator("data/baseline/human_stats.pkl")

        >>> # Validate a document score
        >>> z_score, confidence = validator.validate(0.45)
        >>> print(f"Z-score: {z_score:.2f}, Confidence: {confidence:.3f}")
        Z-score: 3.00, Confidence: 0.999

        >>> # Likely watermarked (high confidence)
        >>> if confidence > 0.95:
        ...     print("Likely watermarked")

        >>> # Validate human-like score
        >>> z_score, confidence = validator.validate(0.15)
        >>> print(f"Z-score: {z_score:.2f}, Confidence: {confidence:.3f}")
        Z-score: 0.00, Confidence: 0.500
    """

    def __init__(self, baseline_path: str = "data/baseline/human_stats.pkl"):
        """Initialize validator with baseline statistics.

        Loads pre-computed baseline statistics (mean and standard deviation)
        from a pickle file created by BaselineCorpusProcessor. These statistics
        represent the expected echo score distribution for human/natural text.

        Args:
            baseline_path: Path to baseline pickle file containing human_mean,
                          human_std, and n_documents keys.

        Raises:
            FileNotFoundError: If baseline file doesn't exist at the specified path.
                              Run BaselineCorpusProcessor.process_corpus() first to
                              create the baseline.

        Examples:
            >>> # Initialize with default path
            >>> validator = StatisticalValidator()

            >>> # Initialize with custom baseline
            >>> validator = StatisticalValidator("data/baseline/news_corpus.pkl")

        Notes:
            - Tier 1: Single baseline file, no caching, no version checking
            - The baseline file must contain keys: 'human_mean', 'human_std', 'n_documents'
            - Baseline statistics are loaded once at initialization for efficiency
        """
        # Validate baseline file exists
        baseline_file = Path(baseline_path)
        if not baseline_file.exists():
            raise FileNotFoundError(
                f"Baseline statistics file not found: {baseline_path}\n"
                f"Please run BaselineCorpusProcessor.process_corpus() first to create the baseline."
            )

        # Load baseline statistics
        processor = BaselineCorpusProcessor()
        self.baseline_stats: Dict[str, float] = processor.load_baseline(baseline_path)

        # Extract statistics for easy access
        self.human_mean = self.baseline_stats['human_mean']
        self.human_std = self.baseline_stats['human_std']
        self.n_documents = self.baseline_stats.get('n_documents', 0)

        # Initialize component calculators
        self.z_score_calculator = ZScoreCalculator()
        self.confidence_converter = ConfidenceConverter()

    def validate(self, document_score: float) -> Tuple[float, float]:
        """Validate document score against baseline distribution.

        Converts a document echo score into standardized z-score and confidence
        level by comparing against the baseline human/natural text distribution.

        The validation process:
        1. Calculate z-score: how many standard deviations from human mean
        2. Convert z-score to confidence: probability of being this extreme
        3. Return both for transparency and flexibility

        Args:
            document_score: Score from ScoringModule, in [0, 1] range.
                           Higher scores indicate more echoing patterns.

        Returns:
            Tuple of (z_score, confidence):
            - z_score (float): Standard deviations from human mean
              - Negative: Below human average (likely human)
              - Zero: Exactly at human average
              - Positive: Above human average (potentially watermarked)
              - > 2: Likely watermarked (>97.5th percentile)
              - > 3: Very likely watermarked (>99.7th percentile)

            - confidence (float): Probability in [0, 1] range
              - < 0.05: Likely human (below 5th percentile)
              - 0.05-0.95: Uncertain region
              - > 0.95: Likely watermarked (above 95th percentile)
              - > 0.99: Very likely watermarked (above 99th percentile)

        Examples:
            >>> validator = StatisticalValidator("data/baseline/human_stats.pkl")

            >>> # High score (likely watermarked)
            >>> z, conf = validator.validate(0.45)
            >>> print(f"Z: {z:.2f}, Conf: {conf:.3f}")
            Z: 3.00, Conf: 0.999

            >>> # Score at human mean (uncertain)
            >>> z, conf = validator.validate(0.15)
            >>> print(f"Z: {z:.2f}, Conf: {conf:.3f}")
            Z: 0.00, Conf: 0.500

            >>> # Low score (likely human)
            >>> z, conf = validator.validate(0.05)
            >>> print(f"Z: {z:.2f}, Conf: {conf:.3f}")
            Z: -1.00, Conf: 0.159

            >>> # Threshold-based classification
            >>> z, conf = validator.validate(0.42)
            >>> if conf > 0.95:
            ...     label = "WATERMARKED"
            ... elif conf < 0.05:
            ...     label = "HUMAN"
            ... else:
            ...     label = "UNCERTAIN"

        Notes:
            - Tier 1: Simple sequential processing, no caching
            - Both values are returned to allow flexible interpretation
            - The z-score is useful for understanding magnitude of deviation
            - The confidence is useful for threshold-based classification
        """
        # Step 1: Calculate z-score using baseline statistics
        z_score = self.z_score_calculator.calculate_z_score(
            document_score=document_score,
            human_mean=self.human_mean,
            human_std=self.human_std
        )

        # Step 2: Convert z-score to confidence level
        confidence = self.confidence_converter.convert_to_confidence(z_score)

        # Return both for transparency
        return z_score, confidence

    def get_baseline_info(self) -> Dict[str, float]:
        """Get information about the loaded baseline.

        Returns baseline statistics for debugging, logging, or reporting purposes.

        Returns:
            Dictionary with keys:
            - human_mean: Mean echo score for human/natural text
            - human_std: Standard deviation of human/natural text scores
            - n_documents: Number of documents in baseline corpus

        Examples:
            >>> validator = StatisticalValidator()
            >>> info = validator.get_baseline_info()
            >>> print(f"Baseline: μ={info['human_mean']:.3f}, σ={info['human_std']:.3f}")
            Baseline: μ=0.150, σ=0.080

            >>> print(f"Based on {info['n_documents']} documents")
            Based on 127 documents
        """
        return {
            'human_mean': self.human_mean,
            'human_std': self.human_std,
            'n_documents': self.n_documents
        }

    def classify(self, document_score: float, threshold: float = 0.95) -> str:
        """Classify document as HUMAN, WATERMARKED, or UNCERTAIN.

        Convenience method that applies a confidence threshold to classify documents
        into discrete categories. This is useful for simple classification use cases.

        Args:
            document_score: Score from ScoringModule, in [0, 1] range
            threshold: Confidence threshold for watermark classification (default: 0.95)

        Returns:
            Classification label:
            - "HUMAN": confidence < (1 - threshold), below lower percentile
            - "WATERMARKED": confidence > threshold, above upper percentile
            - "UNCERTAIN": confidence in between, unclear classification

        Examples:
            >>> validator = StatisticalValidator()

            >>> # High score → WATERMARKED
            >>> label = validator.classify(0.45)
            >>> print(label)
            WATERMARKED

            >>> # Low score → HUMAN
            >>> label = validator.classify(0.05)
            >>> print(label)
            HUMAN

            >>> # Medium score → UNCERTAIN
            >>> label = validator.classify(0.18)
            >>> print(label)
            UNCERTAIN

            >>> # Custom threshold (99% confidence)
            >>> label = validator.classify(0.35, threshold=0.99)
            >>> print(label)
            UNCERTAIN

        Notes:
            - Tier 1: Simple threshold-based classification
            - Symmetric thresholds: (1-threshold) for human, threshold for watermark
            - For more nuanced analysis, use validate() and inspect z-score/confidence directly
        """
        _, confidence = self.validate(document_score)

        if confidence > threshold:
            return "WATERMARKED"
        elif confidence < (1 - threshold):
            return "HUMAN"
        else:
            return "UNCERTAIN"
