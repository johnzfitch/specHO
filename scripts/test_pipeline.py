"""
Comprehensive pipeline diagnostic script.

Tests the complete SpecHO pipeline on sample text and identifies any issues.
"""

import sys
from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.phonetic_analyzer import PhoneticEchoAnalyzer


def test_pipeline(file_path: str, max_chars: int = 2000):
    """
    Test the complete pipeline and report detailed diagnostics.

    Args:
        file_path: Path to text file
        max_chars: Maximum characters to process
    """
    print("=" * 80)
    print("SPECHO PIPELINE DIAGNOSTIC TEST")
    print("=" * 80)

    # Read file
    print(f"\n[1] Reading file: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    sample = text[:max_chars]
    print(f"   Total text length: {len(text)} chars")
    print(f"   Testing with: {len(sample)} chars")
    print(f"   First 100 chars: {sample[:100]}...")

    # Test Preprocessor
    print(f"\n[2] Testing Preprocessor...")
    preprocessor = LinguisticPreprocessor()

    try:
        tokens, doc = preprocessor.process(sample)
        print(f"   SUCCESS: Preprocessor returned {len(tokens)} tokens")
        print(f"   SpaCy doc has {len(doc)} tokens")

        # Check token alignment
        if len(tokens) != len(doc):
            print(f"   [WARNING] Token count mismatch: {len(tokens)} vs {len(doc)}")

        # Check token quality
        tokens_with_phonetics = sum(1 for t in tokens if t.phonetic is not None)
        tokens_with_pos = sum(1 for t in tokens if t.pos_tag is not None)
        content_words = sum(1 for t in tokens if t.is_content_word)

        print(f"\n   Token Quality Report:")
        print(f"      Tokens with phonetics: {tokens_with_phonetics}/{len(tokens)} ({100*tokens_with_phonetics/len(tokens):.1f}%)")
        print(f"      Tokens with POS tags: {tokens_with_pos}/{len(tokens)} ({100*tokens_with_pos/len(tokens):.1f}%)")
        print(f"      Content words: {content_words}/{len(tokens)} ({100*content_words/len(tokens):.1f}%)")

        # Show sample tokens
        print(f"\n   Sample Tokens (first 5):")
        for i, token in enumerate(tokens[:5]):
            print(f"      [{i}] '{token.text}' | POS: {token.pos_tag} | Phonetic: {token.phonetic} | Content: {token.is_content_word}")

    except Exception as e:
        print(f"   [ERROR] Preprocessor failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test Clause Identifier
    print(f"\n[3] Testing Clause Identifier...")
    identifier = ClauseIdentifier()

    try:
        pairs = identifier.identify_pairs(tokens, doc)
        print(f"   SUCCESS: Found {len(pairs)} clause pairs")

        if len(pairs) == 0:
            print(f"   [WARNING] No clause pairs found - this may be normal for short text")
        else:
            # Show pair statistics
            pair_types = {}
            for pair in pairs:
                pair_types[pair.pair_type] = pair_types.get(pair.pair_type, 0) + 1

            print(f"\n   Pair Type Distribution:")
            for ptype, count in pair_types.items():
                print(f"      {ptype}: {count}")

            # Show sample pairs
            print(f"\n   Sample Pairs (first 3):")
            for i, pair in enumerate(pairs[:3]):
                zone_a_text = " ".join([t.text for t in pair.zone_a_tokens[:5]])
                zone_b_text = " ".join([t.text for t in pair.zone_b_tokens[:5]])
                print(f"\n      Pair {i+1} ({pair.pair_type}):")
                print(f"         Clause A: {len(pair.clause_a.tokens)} tokens, Zone: {len(pair.zone_a_tokens)} tokens")
                print(f"         Clause B: {len(pair.clause_b.tokens)} tokens, Zone: {len(pair.zone_b_tokens)} tokens")
                print(f"         Zone A: {zone_a_text}...")
                print(f"         Zone B: {zone_b_text}...")

    except Exception as e:
        print(f"   [ERROR] Clause Identifier failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test Phonetic Analyzer
    print(f"\n[4] Testing Phonetic Echo Analyzer...")
    phonetic_analyzer = PhoneticEchoAnalyzer()

    try:
        if len(pairs) == 0:
            print(f"   [SKIP] No pairs to analyze")
        else:
            scores = []
            valid_pairs = 0
            empty_zones = 0

            for pair in pairs:
                if not pair.zone_a_tokens or not pair.zone_b_tokens:
                    empty_zones += 1
                    continue

                score = phonetic_analyzer.analyze(pair.zone_a_tokens, pair.zone_b_tokens)
                scores.append(score)
                valid_pairs += 1

            print(f"   SUCCESS: Analyzed {valid_pairs} pairs")

            if empty_zones > 0:
                print(f"   [INFO] Skipped {empty_zones} pairs with empty zones")

            if scores:
                avg = sum(scores) / len(scores)
                print(f"\n   Phonetic Similarity Statistics:")
                print(f"      Average: {avg:.3f}")
                print(f"      Max: {max(scores):.3f}")
                print(f"      Min: {min(scores):.3f}")
                print(f"      Scores > 0.5: {sum(1 for s in scores if s > 0.5)}")
                print(f"      Scores > 0.8: {sum(1 for s in scores if s > 0.8)}")

    except Exception as e:
        print(f"   [ERROR] Phonetic Analyzer failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Overall Summary
    print(f"\n" + "=" * 80)
    print("[PIPELINE TEST SUMMARY]")
    print("=" * 80)
    print(f"\n[OK] All pipeline stages completed successfully!")
    print(f"\nStage Results:")
    print(f"   Preprocessor: {len(tokens)} tokens extracted")
    print(f"   Clause Identifier: {len(pairs)} clause pairs found")
    print(f"   Phonetic Analyzer: {valid_pairs if 'valid_pairs' in locals() else 0} pairs analyzed")

    # Check for issues
    issues = []
    if len(tokens) != len(doc):
        issues.append(f"Token count mismatch ({len(tokens)} vs {len(doc)})")
    if tokens_with_phonetics < len(tokens) * 0.9:
        issues.append(f"Low phonetic coverage ({100*tokens_with_phonetics/len(tokens):.1f}%)")
    if len(pairs) == 0:
        issues.append("No clause pairs detected")

    if issues:
        print(f"\n[WARNINGS DETECTED]:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print(f"\n[OK] No issues detected!")

    print(f"\n" + "=" * 80)


def detailed_token_investigation(file_path: str):
    """
    Investigate token mismatch issue in detail.
    """
    print("\n" + "=" * 80)
    print("[DETAILED TOKEN MISMATCH INVESTIGATION]")
    print("=" * 80)

    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()[:500]  # Small sample

    print(f"\nText sample ({len(text)} chars):")
    print(f"   '{text[:200]}...'")

    preprocessor = LinguisticPreprocessor()

    # Test tokenization
    from specHO.preprocessor.tokenizer import Tokenizer
    tokenizer = Tokenizer()

    print(f"\n[Step 1] Tokenizer output:")
    token_objs = tokenizer.tokenize(text)
    print(f"   Tokenizer produced: {len(token_objs)} tokens")
    print(f"   First 10 tokens: {[t.text for t in token_objs[:10]]}")

    # Test POS tagger
    from specHO.preprocessor.pos_tagger import POSTagger
    pos_tagger = POSTagger()

    print(f"\n[Step 2] POSTagger processing:")
    tagged_tokens = pos_tagger.tag(token_objs)
    print(f"   POSTagger returned: {len(tagged_tokens)} tokens")

    # Test dependency parser
    from specHO.preprocessor.dependency_parser import DependencyParser
    parser = DependencyParser()

    print(f"\n[Step 3] DependencyParser processing:")
    doc = parser.parse(text)
    print(f"   SpaCy doc has: {len(doc)} tokens")
    print(f"   SpaCy tokens: {[t.text for t in doc[:10]]}")

    # Compare
    print(f"\n[COMPARISON]:")
    print(f"   Tokenizer count: {len(token_objs)}")
    print(f"   POSTagger count: {len(tagged_tokens)}")
    print(f"   SpaCy Doc count: {len(doc)}")

    if len(token_objs) != len(doc):
        print(f"\n   [MISMATCH DETECTED]")
        print(f"   Difference: {abs(len(token_objs) - len(doc))} tokens")

        # Find differences
        tokenizer_texts = [t.text for t in token_objs]
        spacy_texts = [t.text for t in doc]

        print(f"\n   Tokenizer-only tokens:")
        for i, text in enumerate(tokenizer_texts):
            if i >= len(spacy_texts) or text != spacy_texts[i]:
                print(f"      [{i}] '{text}'")

        print(f"\n   SpaCy-only tokens:")
        for i, text in enumerate(spacy_texts):
            if i >= len(tokenizer_texts) or text != tokenizer_texts[i]:
                print(f"      [{i}] '{text}'")


if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "specHO/sample.txt"

    try:
        # Run main pipeline test
        test_pipeline(file_path, max_chars=2000)

        # Run detailed investigation
        detailed_token_investigation(file_path)

    except FileNotFoundError:
        print(f"[ERROR] File not found: {file_path}")
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
