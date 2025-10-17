"""Test SemanticEchoAnalyzer on real AI-generated essay.

This script demonstrates the semantic analyzer in fallback mode (no embeddings).
For production use, you would load pre-trained Word2Vec or GloVe embeddings.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.semantic_analyzer import SemanticEchoAnalyzer


def main():
    """Test semantic analyzer on AI essay sample."""

    print("=" * 80)
    print("SEMANTIC ECHO ANALYZER TEST - TIER 1 FALLBACK MODE")
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
    semantic_analyzer = SemanticEchoAnalyzer()  # No model = fallback mode
    print("    [OK] Components initialized")
    print()

    # Process text
    print("[*] Processing text through pipeline...")
    tokens, doc = preprocessor.process(text)
    print(f"    [OK] Preprocessor: {len(tokens)} tokens")

    clause_pairs = clause_identifier.identify_pairs(tokens, doc)
    print(f"    [OK] Clause Identifier: {len(clause_pairs)} pairs")
    print()

    # Analyze semantic similarity
    print("[*] Analyzing semantic similarity...")
    print()

    if not clause_pairs:
        print("    [WARN] No clause pairs found in text")
        return

    # Analyze first 10 pairs
    num_to_analyze = min(10, len(clause_pairs))
    similarities = []

    print(f"   Analyzing first {num_to_analyze} clause pairs:")
    print()

    for i, pair in enumerate(clause_pairs[:num_to_analyze], 1):
        zone_a_text = " ".join(t.text for t in pair.zone_a_tokens)
        zone_b_text = " ".join(t.text for t in pair.zone_b_tokens)

        similarity = semantic_analyzer.analyze(pair.zone_a_tokens, pair.zone_b_tokens)
        similarities.append(similarity)

        print(f"   Pair {i}:")
        print(f"      Zone A: {zone_a_text}")
        print(f"      Zone B: {zone_b_text}")
        print(f"      Similarity: {similarity:.3f}")
        print()

    # Statistics
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        min_similarity = min(similarities)
        max_similarity = max(similarities)

        print("=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)
        print()
        print(f"   Total pairs analyzed: {num_to_analyze}")
        print(f"   Average similarity:   {avg_similarity:.3f}")
        print(f"   Min similarity:       {min_similarity:.3f}")
        print(f"   Max similarity:       {max_similarity:.3f}")
        print()
        print("[*] INTERPRETATION:")
        print()
        print("   Since no Word2Vec/GloVe embeddings are loaded, the analyzer")
        print("   returns the fallback value of 0.5 (neutral) for all pairs.")
        print()
        print("   To enable semantic analysis:")
        print("   1. Download pre-trained embeddings:")
        print("      - GloVe: https://nlp.stanford.edu/projects/glove/")
        print("      - Word2Vec: https://code.google.com/archive/p/word2vec/")
        print("   2. Initialize analyzer with model path:")
        print("      analyzer = SemanticEchoAnalyzer(model_path='glove.6B.100d.txt')")
        print()
        print("   Expected behavior with embeddings:")
        print("   - Semantically similar zones: 0.6-1.0")
        print("   - Unrelated zones: 0.3-0.5")
        print("   - Semantically opposite zones: 0.0-0.3")
        print()

    print("=" * 80)
    print("[OK] TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
