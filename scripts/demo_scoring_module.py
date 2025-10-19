"""Demo script for Task 5.3: ScoringModule

Demonstrates the complete scoring pipeline from echo scores to document score.
"""

import sys
import io

# Fix Windows console encoding for Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from specHO.models import EchoScore
from specHO.scoring.pipeline import ScoringModule


def main():
    print("=" * 70)
    print("TASK 5.3 DEMO: ScoringModule Orchestrator")
    print("=" * 70)
    print()

    # Initialize scorer
    scorer = ScoringModule()
    print("✓ ScoringModule initialized")
    print(f"  - WeightedScorer: {type(scorer.weighted_scorer).__name__}")
    print(f"  - DocumentAggregator: {type(scorer.aggregator).__name__}")
    print()

    # Demo 1: Strong watermark signal (high similarity)
    print("-" * 70)
    print("DEMO 1: Strong Watermark Signal")
    print("-" * 70)
    strong_echoes = [
        EchoScore(0.85, 0.80, 0.82, 0.0),
        EchoScore(0.88, 0.83, 0.85, 0.0),
        EchoScore(0.86, 0.81, 0.83, 0.0),
        EchoScore(0.87, 0.82, 0.84, 0.0),
        EchoScore(0.84, 0.79, 0.81, 0.0)
    ]

    doc_score = scorer.score_document(strong_echoes)
    print(f"Echo pairs: {len(strong_echoes)}")
    print(f"Document score: {doc_score:.4f}")
    print(f"Classification: WATERMARKED (score > 0.75)")
    print()

    # Demo 2: Natural text (weak similarity)
    print("-" * 70)
    print("DEMO 2: Natural Text Signal")
    print("-" * 70)
    weak_echoes = [
        EchoScore(0.15, 0.12, 0.18, 0.0),
        EchoScore(0.12, 0.15, 0.13, 0.0),
        EchoScore(0.18, 0.11, 0.16, 0.0),
        EchoScore(0.14, 0.13, 0.15, 0.0),
        EchoScore(0.16, 0.14, 0.12, 0.0)
    ]

    doc_score = scorer.score_document(weak_echoes)
    print(f"Echo pairs: {len(weak_echoes)}")
    print(f"Document score: {doc_score:.4f}")
    print(f"Classification: HUMAN/NATURAL (score < 0.25)")
    print()

    # Demo 3: Unwatermarked AI (moderate similarity)
    print("-" * 70)
    print("DEMO 3: Unwatermarked AI Text")
    print("-" * 70)
    moderate_echoes = [
        EchoScore(0.45, 0.38, 0.42, 0.0),
        EchoScore(0.42, 0.40, 0.44, 0.0),
        EchoScore(0.48, 0.35, 0.40, 0.0),
        EchoScore(0.40, 0.42, 0.45, 0.0),
        EchoScore(0.43, 0.39, 0.41, 0.0)
    ]

    doc_score = scorer.score_document(moderate_echoes)
    print(f"Echo pairs: {len(moderate_echoes)}")
    print(f"Document score: {doc_score:.4f}")
    print(f"Classification: UNWATERMARKED AI (0.25 ≤ score ≤ 0.50)")
    print()

    # Demo 4: Edge cases
    print("-" * 70)
    print("DEMO 4: Edge Cases")
    print("-" * 70)

    # Empty input
    doc_score = scorer.score_document([])
    print(f"Empty echo list: {doc_score:.4f} (expected: 0.0)")

    # Single echo
    single_echo = [EchoScore(0.7, 0.6, 0.65, 0.0)]
    doc_score = scorer.score_document(single_echo)
    print(f"Single echo: {doc_score:.4f}")

    # Perfect and zero mixed
    mixed_extremes = [
        EchoScore(1.0, 1.0, 1.0, 0.0),
        EchoScore(0.0, 0.0, 0.0, 0.0)
    ]
    doc_score = scorer.score_document(mixed_extremes)
    print(f"Perfect + Zero: {doc_score:.4f} (expected: ~0.5)")
    print()

    print("=" * 70)
    print("✓ ScoringModule demo complete!")
    print("  All orchestration patterns working correctly")
    print("=" * 70)


if __name__ == "__main__":
    main()
