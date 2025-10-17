"""Echo analysis pipeline orchestrating all three similarity dimensions.

This module provides the unified interface for analyzing clause pairs across
phonetic, structural, and semantic dimensions. It coordinates the three
specialized analyzers and returns a consolidated EchoScore.

Tier: 1 (MVP)
Task: 4.4
Dependencies: Tasks 4.1, 4.2, 4.3
"""

from typing import Optional
from specHO.models import ClausePair, EchoScore
from specHO.echo_engine.phonetic_analyzer import PhoneticEchoAnalyzer
from specHO.echo_engine.structural_analyzer import StructuralEchoAnalyzer
from specHO.echo_engine.semantic_analyzer import SemanticEchoAnalyzer


class EchoAnalysisEngine:
    """Orchestrates phonetic, structural, and semantic echo analysis.

    This is the main entry point for analyzing clause pairs. It runs all three
    specialized analyzers and combines their results into a consolidated
    EchoScore dataclass.

    Tier 1 Implementation:
    - Simple orchestration (no complex logic)
    - Run all three analyzers sequentially
    - Combine results into EchoScore
    - No error recovery (let exceptions propagate)

    Attributes:
        phonetic_analyzer: Analyzes phonetic similarity (ARPAbet/Levenshtein)
        structural_analyzer: Analyzes POS patterns and syllable counts
        semantic_analyzer: Analyzes semantic similarity (embeddings)
    """

    def __init__(
        self,
        phonetic_analyzer: Optional[PhoneticEchoAnalyzer] = None,
        structural_analyzer: Optional[StructuralEchoAnalyzer] = None,
        semantic_analyzer: Optional[SemanticEchoAnalyzer] = None
    ):
        """Initialize the echo analysis engine with specialized analyzers.

        Args:
            phonetic_analyzer: Phonetic similarity analyzer. If None, creates
                              default instance.
            structural_analyzer: Structural similarity analyzer. If None,
                                creates default instance.
            semantic_analyzer: Semantic similarity analyzer. If None, creates
                              default instance (operates in fallback mode if
                              no embeddings available).
        """
        # Initialize analyzers (use defaults if not provided)
        self.phonetic_analyzer = phonetic_analyzer or PhoneticEchoAnalyzer()
        self.structural_analyzer = structural_analyzer or StructuralEchoAnalyzer()
        self.semantic_analyzer = semantic_analyzer or SemanticEchoAnalyzer()

    def analyze_pair(self, clause_pair: ClausePair) -> EchoScore:
        """Analyze a clause pair across all three similarity dimensions.

        Runs phonetic, structural, and semantic analysis on the clause pair's
        zones and returns a consolidated EchoScore. The combined_score is
        calculated by the scoring module (Task 5.x), so we leave it as 0.0.

        Args:
            clause_pair: ClausePair with zone_a_tokens and zone_b_tokens populated

        Returns:
            EchoScore with all three dimension scores:
            - phonetic_score: Float in [0.0, 1.0]
            - structural_score: Float in [0.0, 1.0]
            - semantic_score: Float in [0.0, 1.0]
            - combined_score: 0.0 (calculated by scoring module)

        Edge Cases:
            - Empty zones: Each analyzer handles this (typically returns 0.0)
            - Missing phonetic data: Phonetic analyzer falls back gracefully
            - Missing embeddings: Semantic analyzer returns 0.5 (neutral)

        Example:
            >>> engine = EchoAnalysisEngine()
            >>> score = engine.analyze_pair(clause_pair)
            >>> print(f"Phonetic: {score.phonetic_score:.3f}")
            >>> print(f"Structural: {score.structural_score:.3f}")
            >>> print(f"Semantic: {score.semantic_score:.3f}")
        """
        # Run all three analyzers on the clause pair zones
        phonetic_score = self.phonetic_analyzer.analyze(
            clause_pair.zone_a_tokens,
            clause_pair.zone_b_tokens
        )

        structural_score = self.structural_analyzer.analyze(
            clause_pair.zone_a_tokens,
            clause_pair.zone_b_tokens
        )

        semantic_score = self.semantic_analyzer.analyze(
            clause_pair.zone_a_tokens,
            clause_pair.zone_b_tokens
        )

        # Return consolidated EchoScore
        # Note: combined_score is 0.0 here - it will be calculated by the
        # scoring module (Task 5.1) which applies weighted combination
        return EchoScore(
            phonetic_score=phonetic_score,
            structural_score=structural_score,
            semantic_score=semantic_score,
            combined_score=0.0  # Calculated by scoring module
        )
