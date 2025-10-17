"""
Quick analysis script for testing PhoneticEchoAnalyzer on sample AI-generated text.

This script demonstrates the complete pipeline from raw text to phonetic echo scores.
"""

from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.phonetic_analyzer import PhoneticEchoAnalyzer


def analyze_sample_text(file_path: str):
    """
    Analyze a text file for phonetic echoes.

    Args:
        file_path: Path to text file to analyze
    """
    print("=" * 80)
    print("PHONETIC ECHO ANALYZER - Sample Text Analysis")
    print("=" * 80)

    # Read the file
    print(f"\n[*] Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"   Text length: {len(text)} characters")
    print(f"   Approximate words: {len(text.split())}")

    # Initialize components
    print("\n[*] Initializing components...")
    preprocessor = LinguisticPreprocessor()
    identifier = ClauseIdentifier()
    phonetic_analyzer = PhoneticEchoAnalyzer()

    # Process text (take first 5000 chars for reasonable processing time)
    sample_text = text[:5000]
    print(f"\n[*] Processing first {len(sample_text)} characters...")

    # Preprocess
    print("   [Stage 1] Linguistic preprocessing...")
    tokens, doc = preprocessor.process(sample_text)
    print(f"      OK: {len(tokens)} tokens extracted")

    # Identify clause pairs
    print("   [Stage 2] Clause identification...")
    pairs = identifier.identify_pairs(tokens, doc)
    print(f"      OK: {len(pairs)} clause pairs identified")

    # Analyze phonetic echoes
    print("   [Stage 3] Phonetic echo analysis...")

    if not pairs:
        print("      WARNING: No clause pairs found - cannot analyze echoes")
        return

    phonetic_scores = []
    for i, pair in enumerate(pairs):
        score = phonetic_analyzer.analyze(pair.zone_a_tokens, pair.zone_b_tokens)
        phonetic_scores.append(score)

        if i < 5:  # Show first 5 examples
            zone_a_text = " ".join([t.text for t in pair.zone_a_tokens])
            zone_b_text = " ".join([t.text for t in pair.zone_b_tokens])
            print(f"\n      Pair {i+1} ({pair.pair_type}):")
            print(f"         Zone A: {zone_a_text}")
            print(f"         Zone B: {zone_b_text}")
            print(f"         Phonetic similarity: {score:.3f}")

    # Calculate statistics
    if phonetic_scores:
        avg_score = sum(phonetic_scores) / len(phonetic_scores)
        max_score = max(phonetic_scores)
        min_score = min(phonetic_scores)

        print(f"\n" + "=" * 80)
        print("[RESULTS]")
        print("=" * 80)
        print(f"\nPhonetic Echo Statistics:")
        print(f"   Total clause pairs analyzed: {len(phonetic_scores)}")
        print(f"   Average phonetic similarity: {avg_score:.3f}")
        print(f"   Maximum similarity: {max_score:.3f}")
        print(f"   Minimum similarity: {min_score:.3f}")

        # Find high-similarity pairs (potential echoes)
        high_similarity = [(i, score) for i, score in enumerate(phonetic_scores) if score > 0.8]

        if high_similarity:
            print(f"\n[HIGH SIMILARITY PAIRS] (>0.8):")
            for idx, score in high_similarity[:5]:  # Show top 5
                pair = pairs[idx]
                zone_a_text = " ".join([t.text for t in pair.zone_a_tokens])
                zone_b_text = " ".join([t.text for t in pair.zone_b_tokens])
                print(f"\n   Pair {idx+1} - Similarity: {score:.3f}")
                print(f"      Zone A: {zone_a_text}")
                print(f"      Zone B: {zone_b_text}")
        else:
            print(f"\n   No pairs with >0.8 similarity found")

        # Interpretation
        print(f"\n" + "=" * 80)
        print("[INTERPRETATION]")
        print("=" * 80)
        print(f"\nFor AI-generated text with deliberate echo watermarking,")
        print(f"we would expect to see:")
        print(f"   - Higher average phonetic similarity (>0.6)")
        print(f"   - Multiple pairs with very high similarity (>0.8)")
        print(f"   - Consistent patterns across clause pairs")
        print(f"\nCurrent average: {avg_score:.3f}")

        if avg_score > 0.6:
            print("   [!] HIGH - Possible watermarking detected!")
        elif avg_score > 0.5:
            print("   [*] MODERATE - Some phonetic echoing present")
        else:
            print("   [OK] LOW - Typical of natural or non-watermarked text")

    print(f"\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    file_path = sys.argv[1] if len(sys.argv) > 1 else "specHO/sample.txt"

    try:
        analyze_sample_text(file_path)
    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
        print("Usage: python scripts/analyze_sample.py <path/to/file>")
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
