"""Test SemanticEchoAnalyzer with Sentence Transformers on real AI essay.

This script demonstrates modern semantic analysis using state-of-the-art
Sentence Transformers embeddings (2023 models).
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
    """Test semantic analyzer with Sentence Transformers on AI essay sample."""

    print("=" * 80)
    print("SEMANTIC ANALYZER TEST - SENTENCE TRANSFORMERS (2023 SOTA)")
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

    # Initialize with Sentence Transformers
    print("[*] Loading Sentence Transformer model: all-MiniLM-L6-v2")
    print("    (This may take a moment on first run...)")
    semantic_analyzer = SemanticEchoAnalyzer(model_path='all-MiniLM-L6-v2')

    if semantic_analyzer.model_type == 'sentence_transformer':
        print("    [OK] Sentence Transformer loaded successfully!")
        print(f"    Model type: {semantic_analyzer.model_type}")
    else:
        print("    [WARN] Fallback mode - Sentence Transformers not available")
        return

    print()

    # Process text
    print("[*] Processing text through pipeline...")
    tokens, doc = preprocessor.process(text)
    print(f"    [OK] Preprocessor: {len(tokens)} tokens")

    clause_pairs = clause_identifier.identify_pairs(tokens, doc)
    print(f"    [OK] Clause Identifier: {len(clause_pairs)} pairs")
    print()

    # Analyze semantic similarity
    print("[*] Analyzing semantic similarity with MODERN embeddings...")
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

        if not zone_a_text.strip() or not zone_b_text.strip():
            continue

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
        print(f"   Total pairs analyzed: {len(similarities)}")
        print(f"   Average similarity:   {avg_similarity:.3f}")
        print(f"   Min similarity:       {min_similarity:.3f}")
        print(f"   Max similarity:       {max_similarity:.3f}")
        print()
        print("[*] INTERPRETATION:")
        print()
        print("   Modern Sentence Transformers provide:")
        print("   - Context-aware embeddings (understands word meaning in context)")
        print("   - Much better semantic understanding than Word2Vec/GloVe")
        print("   - Direct sentence encoding (no averaging needed)")
        print()
        print("   Expected semantic scores for unwatermarked text:")
        print("   - Average: 0.3-0.6 (moderate variation)")
        print("   - High scores (>0.7): Genuinely related concepts")
        print("   - Low scores (<0.3): Unrelated concepts")
        print()
        print("   For watermarked text with Echo Rule:")
        print("   - Expected average: 0.6-0.8+ (artificially high similarity)")
        print("   - Consistent high scores across pairs")
        print()

        # Watermark verdict
        if avg_similarity > 0.65:
            print("   [!] VERDICT: Potential watermark detected (high avg similarity)")
        elif avg_similarity < 0.4:
            print("   [OK] VERDICT: Likely unwatermarked (low avg similarity)")
        else:
            print("   [?] VERDICT: Inconclusive (moderate similarity)")

        print()

    print("=" * 80)
    print("[OK] TEST COMPLETE")
    print("=" * 80)
    print()
    print("Technical Details:")
    print(f"   Model: {semantic_analyzer.model}")
    print(f"   Type: {semantic_analyzer.model_type}")
    print(f"   Embedding dimension: 384")
    print(f"   Context-aware: Yes")
    print()


if __name__ == "__main__":
    main()
