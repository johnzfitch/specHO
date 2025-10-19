#!/usr/bin/env python
"""Demo script to analyze sample2.md through the complete SpecHO pipeline."""

import sys
import pickle
from pathlib import Path

# Add specHO to path
sys.path.insert(0, str(Path(__file__).parent))

from specHO.detector import SpecHODetector
from specHO.utils import load_text_file, setup_logging


def create_temp_baseline():
    """Create a temporary baseline for demonstration."""
    baseline_stats = {
        'human_mean': 0.15,
        'human_std': 0.10,
        'n_documents': 100
    }

    baseline_path = Path("data/baseline")
    baseline_path.mkdir(parents=True, exist_ok=True)

    baseline_file = baseline_path / "demo_baseline.pkl"
    with open(baseline_file, 'wb') as f:
        pickle.dump(baseline_stats, f)

    return str(baseline_file)


def main():
    # Setup logging
    setup_logging(level="INFO")

    # Create temporary baseline
    print("=" * 80)
    print("Creating temporary baseline statistics...")
    baseline_path = create_temp_baseline()
    print(f"Baseline created at: {baseline_path}")
    print("=" * 80)
    print()

    # Initialize detector
    print("Initializing SpecHO Detector...")
    detector = SpecHODetector(baseline_path)
    print("[OK] Detector initialized")
    print()

    # Load sample text
    print("=" * 80)
    # Check if argument provided
    if len(sys.argv) > 1:
        sample_path = sys.argv[1]
    else:
        sample_path = "specHO/sample2.md"  # Default
    print(f"Loading sample text from: {sample_path}")
    text = load_text_file(sample_path)
    print(f"[OK] Loaded {len(text)} characters")
    print("=" * 80)
    print()

    # Run analysis
    print("Running complete detection pipeline...")
    print("=" * 80)
    analysis = detector.analyze(text)
    print("=" * 80)
    print()

    # Display results
    print("=" * 80)
    print("ANALYSIS RESULTS")
    print("=" * 80)
    print()

    print(f"Document Length:        {len(analysis.text):,} characters")
    print(f"Clause Pairs Found:     {len(analysis.clause_pairs)}")
    print(f"Echo Scores Computed:   {len(analysis.echo_scores)}")
    print()

    print("-" * 80)
    print("DETECTION SCORES")
    print("-" * 80)
    print(f"Final Score:            {analysis.final_score:.4f}")
    print(f"Z-Score:                {analysis.z_score:.4f}")
    print(f"Confidence:             {analysis.confidence:.2%}")
    print()

    # Interpretation
    print("-" * 80)
    print("INTERPRETATION")
    print("-" * 80)

    if analysis.z_score > 3.0:
        verdict = "[!!!] HIGHLY LIKELY watermarked (z > 3.0)"
        interpretation = "Very strong evidence of AI watermarking"
    elif analysis.z_score > 2.0:
        verdict = "[!!] LIKELY watermarked (z > 2.0)"
        interpretation = "Strong evidence of AI watermarking"
    elif analysis.z_score > 1.0:
        verdict = "[?] POSSIBLY watermarked (z > 1.0)"
        interpretation = "Moderate evidence of AI watermarking"
    else:
        verdict = "[OK] UNLIKELY watermarked (z <= 1.0)"
        interpretation = "Little to no evidence of AI watermarking"

    print(f"Verdict:      {verdict}")
    print(f"Explanation:  {interpretation}")
    print()

    # Show sample clause pairs
    if analysis.clause_pairs:
        print("-" * 80)
        print("SAMPLE CLAUSE PAIRS (first 5)")
        print("-" * 80)

        for i, pair in enumerate(analysis.clause_pairs[:5], 1):
            print(f"\nPair {i} ({pair.pair_type}):")

            # Show clause A ending
            if pair.zone_a_tokens:
                zone_a_text = " ".join([t.text for t in pair.zone_a_tokens])
                print(f"  Clause A ending: ...{zone_a_text}")

            # Show clause B beginning
            if pair.zone_b_tokens:
                zone_b_text = " ".join([t.text for t in pair.zone_b_tokens])
                print(f"  Clause B beginning: {zone_b_text}...")

            # Show echo score if available
            if i <= len(analysis.echo_scores):
                score = analysis.echo_scores[i-1]
                print(f"  Echo Score: phonetic={score.phonetic_score:.3f}, "
                      f"structural={score.structural_score:.3f}, "
                      f"semantic={score.semantic_score:.3f}")

    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
