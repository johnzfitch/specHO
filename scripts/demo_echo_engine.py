"""Demonstration of complete Echo Engine pipeline with all three analyzers.

This script shows how the EchoAnalysisEngine orchestrates phonetic, structural,
and semantic analysis on real clause pairs from the AI essay sample.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.pipeline import EchoAnalysisEngine
from specHO.echo_engine.semantic_analyzer import SemanticEchoAnalyzer


def main():
    """Demonstrate Echo Engine with all three analyzers on real text."""

    print("=" * 80)
    print("ECHO ENGINE DEMONSTRATION - COMPLETE PIPELINE")
    print("=" * 80)
    print()

    # Load sample text
    sample_path = project_root / "specHO" / "sample.txt"
    print(f"[*] Loading sample: {sample_path}")
    with open(sample_path, 'r', encoding='utf-8') as f:
        text = f.read()

    word_count = len(text.split())
    print(f"    Words: {word_count:,}")
    print()

    # Initialize components
    print("[*] Initializing pipeline components...")
    preprocessor = LinguisticPreprocessor()
    clause_identifier = ClauseIdentifier()

    # Initialize Echo Engine with Sentence Transformers for semantic analysis
    print("[*] Loading Echo Engine...")
    print("    - Phonetic Analyzer: Levenshtein distance on ARPAbet")
    print("    - Structural Analyzer: POS patterns + syllable counts")
    print("    - Semantic Analyzer: Sentence Transformers (all-MiniLM-L6-v2)")

    semantic_analyzer = SemanticEchoAnalyzer(model_path='all-MiniLM-L6-v2')

    if semantic_analyzer.model_type == 'sentence_transformer':
        print("    [OK] Sentence Transformers loaded successfully!")
    else:
        print("    [WARN] Semantic analyzer in fallback mode")

    echo_engine = EchoAnalysisEngine(semantic_analyzer=semantic_analyzer)
    print()

    # Process text
    print("[*] Processing text through preprocessor...")
    tokens, doc = preprocessor.process(text)
    print(f"    [OK] Preprocessor: {len(tokens)} tokens")

    print("[*] Identifying clause pairs...")
    clause_pairs = clause_identifier.identify_pairs(tokens, doc)
    print(f"    [OK] Clause Identifier: {len(clause_pairs)} pairs")
    print()

    # Analyze clause pairs with Echo Engine
    print("[*] Analyzing clause pairs with Echo Engine...")
    print()

    if not clause_pairs:
        print("    [WARN] No clause pairs found in text")
        return

    # Analyze first 10 pairs
    num_to_analyze = min(10, len(clause_pairs))
    echo_scores = []

    print(f"   Analyzing first {num_to_analyze} clause pairs:")
    print()

    for i, pair in enumerate(clause_pairs[:num_to_analyze], 1):
        zone_a_text = " ".join(t.text for t in pair.zone_a_tokens)
        zone_b_text = " ".join(t.text for t in pair.zone_b_tokens)

        if not zone_a_text.strip() or not zone_b_text.strip():
            continue

        # Run complete echo analysis
        score = echo_engine.analyze_pair(pair)
        echo_scores.append(score)

        print(f"   Pair {i}:")
        print(f"      Zone A: {zone_a_text}")
        print(f"      Zone B: {zone_b_text}")
        print(f"      Phonetic:   {score.phonetic_score:.3f}")
        print(f"      Structural: {score.structural_score:.3f}")
        print(f"      Semantic:   {score.semantic_score:.3f}")
        print()

    # Statistics
    if echo_scores:
        avg_phonetic = sum(s.phonetic_score for s in echo_scores) / len(echo_scores)
        avg_structural = sum(s.structural_score for s in echo_scores) / len(echo_scores)
        avg_semantic = sum(s.semantic_score for s in echo_scores) / len(echo_scores)

        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print()
        print(f"   Total pairs analyzed: {len(echo_scores)}")
        print()
        print("   Average Scores:")
        print(f"      Phonetic:   {avg_phonetic:.3f}")
        print(f"      Structural: {avg_structural:.3f}")
        print(f"      Semantic:   {avg_semantic:.3f}")
        print()
        print("[*] INTERPRETATION:")
        print()
        print("   Phonetic Dimension:")
        print("   - Measures sound similarity between zone words")
        print("   - Uses Levenshtein distance on ARPAbet phonetic transcriptions")
        print("   - Lower scores expected for unwatermarked text")
        print()
        print("   Structural Dimension:")
        print("   - Measures POS pattern and syllable count similarity")
        print("   - Captures grammatical structure echoes")
        print("   - Moderate scores indicate some structural patterns")
        print()
        print("   Semantic Dimension:")
        print("   - Measures meaning similarity using modern embeddings")
        print("   - Context-aware Sentence Transformers (2023 SOTA)")
        print("   - Moderate scores normal for coherent text")
        print()
        print("   Combined Analysis:")
        if avg_phonetic > 0.6 and avg_structural > 0.6 and avg_semantic > 0.6:
            print("   [!] High scores across ALL dimensions suggest watermark")
        elif avg_phonetic > 0.7 or avg_structural > 0.7 or avg_semantic > 0.7:
            print("   [?] High score in ONE dimension - investigate further")
        else:
            print("   [OK] Scores within normal range for unwatermarked text")
        print()

    print("=" * 80)
    print("[OK] DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Next Steps:")
    print("   - Task 5.x: Implement weighted scoring and aggregation")
    print("   - Task 6.x: Add statistical validation with baseline corpus")
    print("   - Task 7.x: Create complete detector and CLI")
    print()


if __name__ == "__main__":
    main()
