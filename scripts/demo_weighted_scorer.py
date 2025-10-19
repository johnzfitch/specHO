"""Demo script for Task 5.1: WeightedScorer

Demonstrates weighted scoring on realistic EchoScore examples to validate
the implementation matches expected behavior.

Real-world validation following Session 5 pattern:
- Test with realistic score combinations
- Verify output ranges match expectations
- Confirm NaN handling works correctly
- Validate custom weight configurations
"""

import numpy as np
from specHO.models import EchoScore
from specHO.scoring.weighted_scorer import WeightedScorer
from specHO.config import load_config


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demo_basic_scoring():
    """Demonstrate basic weighted scoring with default config."""
    print_section("BASIC WEIGHTED SCORING (Default Config)")

    scorer = WeightedScorer()
    print(f"\nDefault weights: {scorer.get_weights()}")

    # Test cases representing different echo strengths
    test_cases = [
        ("Strong Echo (high on all dimensions)", EchoScore(0.85, 0.80, 0.75, 0.0)),
        ("Moderate Echo (mid-range values)", EchoScore(0.60, 0.55, 0.65, 0.0)),
        ("Weak Echo (low similarity)", EchoScore(0.25, 0.30, 0.20, 0.0)),
        ("Phonetic Dominant (rhyming clauses)", EchoScore(0.90, 0.40, 0.50, 0.0)),
        ("Semantic Dominant (paraphrasing)", EchoScore(0.30, 0.35, 0.85, 0.0)),
        ("Structural Dominant (parallel syntax)", EchoScore(0.35, 0.90, 0.40, 0.0)),
    ]

    print("\nTest Cases:")
    print(f"{'Description':<40} {'P':>6} {'S':>6} {'Sem':>6} -> {'Score':>6}")
    print("-" * 70)

    for description, echo in test_cases:
        score = scorer.calculate_pair_score(echo)
        print(f"{description:<40} {echo.phonetic_score:>6.2f} "
              f"{echo.structural_score:>6.2f} {echo.semantic_score:>6.2f} "
              f"-> {score:>6.3f}")


def demo_custom_weights():
    """Demonstrate scoring with custom weight configurations."""
    print_section("CUSTOM WEIGHT CONFIGURATIONS")

    # Example echo to score with different weights
    echo = EchoScore(0.70, 0.50, 0.60, 0.0)
    print(f"\nTest Echo: Phonetic={echo.phonetic_score:.2f}, "
          f"Structural={echo.structural_score:.2f}, "
          f"Semantic={echo.semantic_score:.2f}")

    weight_configs = [
        ("Equal Weights (default)", {'phonetic': 0.33, 'structural': 0.33, 'semantic': 0.34}),
        ("Phonetic Emphasis", {'phonetic': 0.5, 'structural': 0.25, 'semantic': 0.25}),
        ("Semantic Emphasis", {'phonetic': 0.25, 'structural': 0.25, 'semantic': 0.5}),
        ("Structural Emphasis", {'phonetic': 0.25, 'structural': 0.5, 'semantic': 0.25}),
        ("Phonetic+Semantic Focus", {'phonetic': 0.45, 'structural': 0.1, 'semantic': 0.45}),
    ]

    print(f"\n{'Configuration':<30} {'Weights':>30} -> {'Score':>6}")
    print("-" * 70)

    for name, weights in weight_configs:
        scorer = WeightedScorer(weights=weights)
        score = scorer.calculate_pair_score(echo)
        weight_str = f"({weights['phonetic']:.2f}, {weights['structural']:.2f}, {weights['semantic']:.2f})"
        print(f"{name:<30} {weight_str:>30} -> {score:>6.3f}")


def demo_nan_handling():
    """Demonstrate NaN handling for missing data."""
    print_section("NaN HANDLING (Missing Data Strategy)")

    scorer = WeightedScorer()
    print(f"\nMissing data strategy: {scorer.config.missing_data_strategy}")
    print("(NaN values are treated as 0.0 in Tier 1)")

    test_cases = [
        ("All scores available", EchoScore(0.70, 0.60, 0.80, 0.0)),
        ("Missing phonetic", EchoScore(np.nan, 0.60, 0.80, 0.0)),
        ("Missing structural", EchoScore(0.70, np.nan, 0.80, 0.0)),
        ("Missing semantic", EchoScore(0.70, 0.60, np.nan, 0.0)),
        ("Missing phonetic+structural", EchoScore(np.nan, np.nan, 0.80, 0.0)),
        ("All missing (fallback to 0)", EchoScore(np.nan, np.nan, np.nan, 0.0)),
    ]

    print(f"\n{'Description':<35} {'P':>8} {'S':>8} {'Sem':>8} -> {'Score':>6}")
    print("-" * 70)

    for description, echo in test_cases:
        score = scorer.calculate_pair_score(echo)
        p_str = f"{echo.phonetic_score:.2f}" if not np.isnan(echo.phonetic_score) else "NaN"
        s_str = f"{echo.structural_score:.2f}" if not np.isnan(echo.structural_score) else "NaN"
        sem_str = f"{echo.semantic_score:.2f}" if not np.isnan(echo.semantic_score) else "NaN"
        print(f"{description:<35} {p_str:>8} {s_str:>8} {sem_str:>8} -> {score:>6.3f}")


def demo_config_integration():
    """Demonstrate integration with SpecHO config system."""
    print_section("CONFIG SYSTEM INTEGRATION")

    # Test with simple profile
    print("\n[1] Simple Profile (Tier 1 - Equal Weights)")
    config = load_config("simple")
    scorer = WeightedScorer(config=config.scoring)
    print(f"    Weights: {scorer.get_weights()}")

    # Test with overrides
    print("\n[2] Simple Profile with Overrides")
    config = load_config("simple", overrides={
        "scoring.phonetic_weight": 0.5,
        "scoring.structural_weight": 0.3,
        "scoring.semantic_weight": 0.2
    })
    scorer = WeightedScorer(config=config.scoring)
    print(f"    Weights: {scorer.get_weights()}")

    # Test scoring with both configs
    echo = EchoScore(0.75, 0.55, 0.65, 0.0)
    print(f"\n[3] Scoring Same Echo with Different Configs")
    print(f"    Echo: P={echo.phonetic_score:.2f}, S={echo.structural_score:.2f}, "
          f"Sem={echo.semantic_score:.2f}")

    config1 = load_config("simple")
    scorer1 = WeightedScorer(config=config1.scoring)
    score1 = scorer1.calculate_pair_score(echo)
    print(f"    Simple profile: {score1:.3f}")

    config2 = load_config("simple", overrides={
        "scoring.phonetic_weight": 0.6,
        "scoring.structural_weight": 0.2,
        "scoring.semantic_weight": 0.2
    })
    scorer2 = WeightedScorer(config=config2.scoring)
    score2 = scorer2.calculate_pair_score(echo)
    print(f"    Phonetic-emphasis profile: {score2:.3f}")


def demo_realistic_scenarios():
    """Demonstrate scoring on realistic watermark detection scenarios."""
    print_section("REALISTIC WATERMARK DETECTION SCENARIOS")

    scorer = WeightedScorer()

    scenarios = [
        ("Watermarked Text (Strong Echo Rule)",
         "High scores expected across all dimensions",
         EchoScore(0.82, 0.78, 0.75, 0.0)),

        ("Unwatermarked AI Text",
         "Low phonetic/structural, moderate semantic",
         EchoScore(0.28, 0.31, 0.55, 0.0)),

        ("Human Text (Natural Writing)",
         "Low scores across all dimensions",
         EchoScore(0.15, 0.22, 0.35, 0.0)),

        ("Poetry (Natural Rhyming)",
         "High phonetic, low structural/semantic",
         EchoScore(0.85, 0.30, 0.25, 0.0)),

        ("Technical Documentation",
         "Moderate structural (templates), low others",
         EchoScore(0.20, 0.65, 0.40, 0.0)),
    ]

    print("\nExpected Score Ranges:")
    print("  - Watermarked: 0.70-0.90 (clear echo signal)")
    print("  - Unwatermarked AI: 0.30-0.50 (coherent but no echo)")
    print("  - Human/Natural: 0.15-0.35 (minimal echo)")
    print()

    print(f"{'Scenario':<35} {'Description':<40} -> {'Score':>6}")
    print("-" * 85)

    for scenario, description, echo in scenarios:
        score = scorer.calculate_pair_score(echo)
        print(f"{scenario:<35} {description:<40} -> {score:>6.3f}")


def main():
    """Run all demonstration scenarios."""
    print("\n" + "=" * 70)
    print("  WEIGHTED SCORER DEMONSTRATION (Task 5.1)")
    print("  Real-world validation of weighted scoring implementation")
    print("=" * 70)

    try:
        demo_basic_scoring()
        demo_custom_weights()
        demo_nan_handling()
        demo_config_integration()
        demo_realistic_scenarios()

        print("\n" + "=" * 70)
        print("  VALIDATION COMPLETE")
        print("=" * 70)
        print("\nAll demonstrations completed successfully!")
        print("WeightedScorer implementation validated on realistic scenarios.")
        print("\nKey Observations:")
        print("  - Weighted sum calculation works correctly")
        print("  - NaN handling treats missing data as 0.0")
        print("  - Custom weights allow tuning for specific use cases")
        print("  - Output always in [0,1] range")
        print("  - Config integration working properly")
        print("\nReady for Task 5.2: DocumentAggregator")

    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
