"""Demo script for Statistical Validator (Tasks 6.2-6.4).

This script demonstrates the complete statistical validation workflow:
1. ZScoreCalculator: Convert document scores to z-scores
2. ConfidenceConverter: Convert z-scores to confidence levels
3. StatisticalValidator: Orchestrated pipeline with baseline comparison

The demo creates a test baseline and validates several document scores,
showing how the system distinguishes human from watermarked text.

Usage:
    python scripts/demo_validator.py
"""

import pickle
import tempfile
from pathlib import Path

from specHO.validator.z_score import ZScoreCalculator
from specHO.validator.confidence import ConfidenceConverter
from specHO.validator.pipeline import StatisticalValidator


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def demo_z_score_calculator():
    """Demonstrate ZScoreCalculator usage."""
    print_section("TASK 6.2: Z-Score Calculator")

    calc = ZScoreCalculator()

    # Define a baseline
    human_mean = 0.15
    human_std = 0.10

    print(f"Baseline Statistics:")
    print(f"  Human Mean: {human_mean:.3f}")
    print(f"  Human Std:  {human_std:.3f}\n")

    # Test various document scores
    test_scores = [
        (0.05, "Low score (likely human)"),
        (0.15, "Score at mean (uncertain)"),
        (0.25, "Moderate score (uncertain)"),
        (0.35, "High score (possibly watermarked)"),
        (0.45, "Very high score (likely watermarked)")
    ]

    print("Document Score Analysis:")
    print(f"{'Score':<10} {'Z-Score':<12} {'Description':<40}")
    print("-" * 70)

    for score, description in test_scores:
        z = calc.calculate_z_score(score, human_mean, human_std)
        print(f"{score:<10.3f} {z:<12.2f} {description}")

    print("\n[INSIGHT] Z-score shows how many standard deviations a score")
    print("          is from the human mean. z>2 suggests watermarking.")


def demo_confidence_converter():
    """Demonstrate ConfidenceConverter usage."""
    print_section("TASK 6.3: Confidence Converter")

    converter = ConfidenceConverter()

    # Test z-scores from previous demo
    test_z_scores = [
        (-2.0, "Far below mean"),
        (-1.0, "Below mean"),
        (0.0, "At mean"),
        (1.0, "Above mean"),
        (2.0, "Far above mean (95th percentile)"),
        (3.0, "Very far above mean (99.7th percentile)")
    ]

    print("Z-Score to Confidence Conversion:")
    print(f"{'Z-Score':<12} {'Confidence':<15} {'Percentile':<15} {'Description':<30}")
    print("-" * 80)

    for z, description in test_z_scores:
        confidence = converter.convert_to_confidence(z)
        percentile = converter.z_score_to_percentile(z)

        print(f"{z:<12.2f} {confidence:<15.6f} {percentile:<15.1f} {description}")

    print("\n[INSIGHT] Confidence represents the probability that a score")
    print("          came from the human distribution. High confidence (>0.95)")
    print("          suggests watermarking.")


def demo_statistical_validator():
    """Demonstrate complete StatisticalValidator pipeline."""
    print_section("TASK 6.4: Statistical Validator (Complete Pipeline)")

    # Create a temporary baseline file
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "demo_baseline.pkl"

        # Create baseline statistics
        baseline_stats = {
            'human_mean': 0.15,
            'human_std': 0.10,
            'n_documents': 127  # Simulated corpus size
        }

        with open(baseline_path, 'wb') as f:
            pickle.dump(baseline_stats, f)

        print(f"Created test baseline at: {baseline_path}\n")

        # Initialize validator
        validator = StatisticalValidator(str(baseline_path))

        # Display baseline info
        info = validator.get_baseline_info()
        print("Loaded Baseline Statistics:")
        print(f"  Mean:          {info['human_mean']:.3f}")
        print(f"  Std Dev:       {info['human_std']:.3f}")
        print(f"  N Documents:   {info['n_documents']}")
        print()

        # Test document scores
        test_documents = [
            (0.08, "Human-written article"),
            (0.12, "Human blog post"),
            (0.15, "Typical human text (at mean)"),
            (0.22, "Borderline case"),
            (0.35, "Suspicious (high echoing)"),
            (0.42, "AI-generated with watermark"),
            (0.50, "Strong watermark signal")
        ]

        print("Document Validation Results:")
        print(f"{'Score':<10} {'Z-Score':<12} {'Confidence':<15} {'Label':<15} {'Description':<30}")
        print("-" * 95)

        for score, description in test_documents:
            z_score, confidence = validator.validate(score)
            label = validator.classify(score, threshold=0.95)

            print(f"{score:<10.3f} {z_score:<12.2f} {confidence:<15.6f} {label:<15} {description}")

        print("\n[INSIGHT] The validator orchestrates z-score calculation and")
        print("          confidence conversion to provide a complete classification.")
        print("          Threshold: confidence >0.95 = WATERMARKED, <0.05 = HUMAN")


def demo_threshold_comparison():
    """Demonstrate effect of different classification thresholds."""
    print_section("BONUS: Threshold Sensitivity Analysis")

    # Create temporary baseline
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "demo_baseline.pkl"

        baseline_stats = {
            'human_mean': 0.15,
            'human_std': 0.10,
            'n_documents': 127
        }

        with open(baseline_path, 'wb') as f:
            pickle.dump(baseline_stats, f)

        validator = StatisticalValidator(str(baseline_path))

        # Test score in uncertain region
        test_score = 0.28

        print(f"Document Score: {test_score:.3f}")

        z_score, confidence = validator.validate(test_score)
        print(f"Z-Score:        {z_score:.2f}")
        print(f"Confidence:     {confidence:.6f}\n")

        # Try different thresholds
        thresholds = [0.90, 0.95, 0.975, 0.99]

        print("Classification with Different Thresholds:")
        print(f"{'Threshold':<15} {'Label':<20} {'Interpretation':<40}")
        print("-" * 75)

        for thresh in thresholds:
            label = validator.classify(test_score, threshold=thresh)
            percentile = thresh * 100

            interp = f"{percentile:.1f}th percentile cutoff"
            print(f"{thresh:<15.3f} {label:<20} {interp}")

        print("\n[INSIGHT] Lower thresholds (0.90) classify more as watermarked,")
        print("          while higher thresholds (0.99) are more conservative.")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("  STATISTICAL VALIDATOR DEMONSTRATION")
    print("  Tasks 6.2, 6.3, 6.4 - Complete Validation Pipeline")
    print("="*70)

    # Run demonstrations
    demo_z_score_calculator()
    demo_confidence_converter()
    demo_statistical_validator()
    demo_threshold_comparison()

    # Summary
    print_section("SUMMARY")
    print("[COMPLETE] Task 6.2: ZScoreCalculator - Converts scores to z-scores")
    print("[COMPLETE] Task 6.3: ConfidenceConverter - Converts z-scores to confidence")
    print("[COMPLETE] Task 6.4: StatisticalValidator - Orchestrates complete pipeline")
    print()
    print("The statistical validator provides a principled, threshold-free")
    print("approach to watermark detection by comparing document scores against")
    print("baseline human/natural text distributions.")
    print()
    print("Next Steps:")
    print("  1. Build real baseline corpus using BaselineCorpusProcessor")
    print("  2. Integrate with SpecHODetector for end-to-end detection")
    print("  3. Tune threshold based on false positive/negative requirements")
    print()


if __name__ == "__main__":
    main()
