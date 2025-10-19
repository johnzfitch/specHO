"""Statistical Validator module for SpecHO watermark detection.

This module implements statistical validation components that convert document
scores into confidence levels by comparing against baseline distributions.

Components:
    - BaselineCorpusProcessor: Process corpus to establish baseline statistics (Task 6.1)
    - ZScoreCalculator: Calculate z-scores (Task 6.2)
    - ConfidenceConverter: Convert z-scores to confidence levels (Task 6.3)
    - StatisticalValidator: Pipeline orchestrator (Task 6.4)

Tier: 1 (MVP)
Phase: Component 5 - Statistical Validator
Tasks: 6.1, 6.2, 6.3, 6.4
"""

from .baseline_builder import BaselineCorpusProcessor
from .z_score import ZScoreCalculator
from .confidence import ConfidenceConverter
from .pipeline import StatisticalValidator

__all__ = [
    'BaselineCorpusProcessor',
    'ZScoreCalculator',
    'ConfidenceConverter',
    'StatisticalValidator'
]
