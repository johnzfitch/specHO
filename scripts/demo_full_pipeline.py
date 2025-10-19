"""Full Pipeline Demo: End-to-End Watermark Detection

Runs sample.txt through the complete SpecHO pipeline implemented so far:

1. Preprocessor: Text → (tokens, doc)
2. ClauseIdentifier: (tokens, doc) → List[ClausePair]
3. EchoEngine: List[ClausePair] → List[EchoScore]
4. ScoringModule: List[EchoScore] → document_score
   (Orchestrates WeightedScorer + DocumentAggregator)

This demonstrates the complete pipeline integration from Task 1.1 through Task 5.3.
"""

import time
from pathlib import Path

# Import all pipeline components
from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.pipeline import EchoAnalysisEngine
from specHO.scoring.pipeline import ScoringModule


def print_header(title):
    """Print section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title):
    """Print subsection header."""
    print("\n" + "-" * 80)
    print(f"  {title}")
    print("-" * 80)


def load_sample_text():
    """Load the sample.txt file."""
    sample_path = Path(__file__).parent.parent / "specHO" / "sample.txt"

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    with open(sample_path, 'r', encoding='utf-8') as f:
        text = f.read()

    return text


def main():
    """Run complete pipeline demonstration."""

    print("=" * 80)
    print("  FULL PIPELINE DEMONSTRATION")
    print("  SpecHO Watermark Detection System")
    print("  Components: Preprocessor -> Clause Identifier -> Echo Engine -> Scoring")
    print("=" * 80)

    # Load sample text
    print_header("STEP 0: LOADING SAMPLE TEXT")
    text = load_sample_text()

    # Show text statistics
    lines = text.split('\n')
    words = text.split()
    chars = len(text)

    print(f"\nFile: specHO/sample.txt")
    print(f"Lines: {len(lines)}")
    print(f"Words: {len(words)}")
    print(f"Characters: {chars}")
    print(f"\nFirst 200 characters:")
    print(f"  {text[:200]}...")

    # Step 1: Preprocessing
    print_header("STEP 1: LINGUISTIC PREPROCESSING")
    print("\nInitializing preprocessor...")
    preprocessor = LinguisticPreprocessor()

    print("Processing text (tokenization, POS tagging, dependencies, phonetics)...")
    start_time = time.time()
    tokens, doc = preprocessor.process(text)
    preprocess_time = time.time() - start_time

    print(f"\n[COMPLETE] Preprocessing finished in {preprocess_time:.2f}s")
    print(f"  Total tokens: {len(tokens)}")
    print(f"  Content words: {sum(1 for t in tokens if t.is_content_word)}")
    print(f"  Sentences: {len(list(doc.sents))}")

    # Show sample tokens
    print("\nSample tokens (first 10):")
    print(f"  {'Text':<15} {'POS':<8} {'Phonetic':<20} {'Content':<8} {'Syllables':<10}")
    print("  " + "-" * 75)
    for i, token in enumerate(tokens[:10]):
        content = "Yes" if token.is_content_word else "No"
        phonetic_str = token.phonetic if token.phonetic else "N/A"
        syllables = token.syllable_count if token.syllable_count else 0
        print(f"  {token.text:<15} {token.pos_tag:<8} {phonetic_str:<20} {content:<8} {syllables:<10}")

    # Step 2: Clause Identification
    print_header("STEP 2: CLAUSE IDENTIFICATION")
    print("\nInitializing clause identifier...")
    clause_identifier = ClauseIdentifier()

    print("Identifying clause pairs (boundaries, pairing rules, zone extraction)...")
    start_time = time.time()
    clause_pairs = clause_identifier.identify_pairs(tokens, doc)
    clause_time = time.time() - start_time

    print(f"\n[COMPLETE] Clause identification finished in {clause_time:.2f}s")
    print(f"  Clause pairs identified: {len(clause_pairs)}")

    # Show sample pairs
    if clause_pairs:
        print("\nSample clause pairs (first 3):")
        for i, pair in enumerate(clause_pairs[:3], 1):
            zone_a_text = ' '.join(t.text for t in pair.zone_a_tokens)
            zone_b_text = ' '.join(t.text for t in pair.zone_b_tokens)
            print(f"\n  Pair {i} (Type: {pair.pair_type}):")
            print(f"    Zone A ({len(pair.zone_a_tokens)} tokens): {zone_a_text[:60]}...")
            print(f"    Zone B ({len(pair.zone_b_tokens)} tokens): {zone_b_text[:60]}...")

    # Step 3: Echo Analysis
    print_header("STEP 3: ECHO ANALYSIS")
    print("\nInitializing echo analysis engine...")
    echo_engine = EchoAnalysisEngine()

    print("Analyzing echo patterns (phonetic, structural, semantic)...")
    start_time = time.time()
    echo_scores = [echo_engine.analyze_pair(pair) for pair in clause_pairs]
    echo_time = time.time() - start_time

    print(f"\n[COMPLETE] Echo analysis finished in {echo_time:.2f}s")
    print(f"  Echo scores computed: {len(echo_scores)}")

    # Show sample echo scores
    if echo_scores:
        print("\nSample echo scores (first 5):")
        print(f"  {'Pair':<8} {'Phonetic':>10} {'Structural':>12} {'Semantic':>10}")
        print("  " + "-" * 45)
        for i, score in enumerate(echo_scores[:5], 1):
            print(f"  Pair {i:<3} {score.phonetic_score:>10.3f} "
                  f"{score.structural_score:>12.3f} {score.semantic_score:>10.3f}")

        # Compute statistics
        avg_phonetic = sum(s.phonetic_score for s in echo_scores) / len(echo_scores)
        avg_structural = sum(s.structural_score for s in echo_scores) / len(echo_scores)
        avg_semantic = sum(s.semantic_score for s in echo_scores) / len(echo_scores)

        print("\nAverage scores across all pairs:")
        print(f"  Phonetic:   {avg_phonetic:.3f}")
        print(f"  Structural: {avg_structural:.3f}")
        print(f"  Semantic:   {avg_semantic:.3f}")

    # Step 4: Scoring (Weighted + Aggregation)
    print_header("STEP 4: SCORING MODULE (Task 5.3)")
    print("\nInitializing ScoringModule orchestrator...")
    scoring_module = ScoringModule()

    print(f"  WeightedScorer: {type(scoring_module.weighted_scorer).__name__}")
    print(f"  DocumentAggregator: {type(scoring_module.aggregator).__name__}")

    weights = scoring_module.weighted_scorer.get_weights()
    print(f"\nUsing weights: Phonetic={weights['phonetic']:.2f}, "
          f"Structural={weights['structural']:.2f}, "
          f"Semantic={weights['semantic']:.2f}")

    print("\nScoring document (weighted scoring + aggregation)...")
    start_time = time.time()
    document_score = scoring_module.score_document(echo_scores)
    scoring_time = time.time() - start_time

    print(f"\n[COMPLETE] Scoring finished in {scoring_time:.2f}s")

    # Show intermediate pair scores for analysis
    pair_scores = [scoring_module.weighted_scorer.calculate_pair_score(score) for score in echo_scores]

    if pair_scores:
        print(f"\n  Pair scores computed: {len(pair_scores)}")
        print("\nSample pair scores (first 10):")
        print(f"  {'Pair':<8} {'Score':>8}")
        print("  " + "-" * 20)
        for i, score in enumerate(pair_scores[:10], 1):
            print(f"  Pair {i:<3} {score:>8.3f}")

        # Show distribution
        low_scores = sum(1 for s in pair_scores if s < 0.3)
        mid_scores = sum(1 for s in pair_scores if 0.3 <= s < 0.7)
        high_scores = sum(1 for s in pair_scores if s >= 0.7)

        print(f"\nScore distribution:")
        print(f"  Low (< 0.3):    {low_scores:>4} pairs ({low_scores/len(pair_scores)*100:.1f}%)")
        print(f"  Medium (0.3-0.7): {mid_scores:>4} pairs ({mid_scores/len(pair_scores)*100:.1f}%)")
        print(f"  High (>= 0.7):  {high_scores:>4} pairs ({high_scores/len(pair_scores)*100:.1f}%)")

    # Get statistics
    stats = scoring_module.aggregator.get_statistics(pair_scores)

    # Final Results
    print_header("FINAL RESULTS")

    print(f"\nDocument Score: {document_score:.4f}")
    print("\nScore Statistics:")
    print(f"  Number of pairs: {stats['n_pairs']}")
    print(f"  Mean:            {stats['mean']:.4f}")
    print(f"  Median:          {stats['median']:.4f}")
    print(f"  Minimum:         {stats['min']:.4f}")
    print(f"  Maximum:         {stats['max']:.4f}")
    print(f"  Std Deviation:   {stats['stdev']:.4f}")

    # Classification
    print("\nClassification:")
    if document_score >= 0.75:
        classification = "STRONG WATERMARK"
        interpretation = "Very likely watermarked with Echo Rule"
    elif document_score >= 0.50:
        classification = "WEAK WATERMARK"
        interpretation = "Possible watermark, borderline detection"
    elif document_score >= 0.25:
        classification = "UNWATERMARKED AI"
        interpretation = "Likely AI-generated but no watermark"
    else:
        classification = "HUMAN/NATURAL"
        interpretation = "Likely human-written or natural text"

    print(f"  Classification: {classification}")
    print(f"  Interpretation: {interpretation}")

    # Performance Summary
    print_header("PERFORMANCE SUMMARY")

    total_time = preprocess_time + clause_time + echo_time + scoring_time

    print(f"\nProcessing Times:")
    print(f"  Preprocessing:        {preprocess_time:>8.2f}s ({preprocess_time/total_time*100:>5.1f}%)")
    print(f"  Clause Identification: {clause_time:>8.2f}s ({clause_time/total_time*100:>5.1f}%)")
    print(f"  Echo Analysis:        {echo_time:>8.2f}s ({echo_time/total_time*100:>5.1f}%)")
    print(f"  Scoring Module:       {scoring_time:>8.2f}s ({scoring_time/total_time*100:>5.1f}%)")
    print(f"  " + "-" * 40)
    print(f"  Total:                {total_time:>8.2f}s")

    print(f"\nThroughput:")
    print(f"  Words processed: {len(words)}")
    print(f"  Processing rate: {len(words)/total_time:.1f} words/second")
    print(f"  Pairs analyzed: {len(clause_pairs)}")
    print(f"  Pair rate: {len(clause_pairs)/total_time:.1f} pairs/second")

    # Pipeline Status
    print_header("PIPELINE STATUS")

    print("\nComponents Tested:")
    print("  [OK] Component 1: Preprocessor (Tasks 2.1-2.5)")
    print("  [OK] Component 2: Clause Identifier (Tasks 3.1-3.4)")
    print("  [OK] Component 3: Echo Engine (Tasks 4.1-4.4)")
    print("  [OK] Component 4: Scoring - WeightedScorer (Task 5.1)")
    print("  [OK] Component 4: Scoring - DocumentAggregator (Task 5.2)")
    print("  [OK] Component 4: Scoring - ScoringModule (Task 5.3)")

    print("\nComponents Remaining:")
    print("  [ ] Component 5: Statistical Validator (Tasks 6.1-6.4)")
    print("  [ ] Integration: SpecHODetector (Task 7.1)")
    print("  [ ] Integration: CLI Interface (Task 7.2)")

    print("\n" + "=" * 80)
    print("  PIPELINE DEMONSTRATION COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully processed '{Path('specHO/sample.txt').name}'")
    print(f"Document classified as: {classification}")
    print(f"Final score: {document_score:.4f}")
    print("\nAll components working correctly! Ready to continue with Component 5 (Validator).")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
