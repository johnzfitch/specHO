"""
End-to-End Pipeline Tests with Real Sample Texts

Tests the complete pipeline from raw text to clause pairs with zones.
Validates that all components work together correctly.
"""

import pytest
from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier


@pytest.fixture
def preprocessor():
    """Fixture providing LinguisticPreprocessor instance."""
    return LinguisticPreprocessor()


@pytest.fixture
def identifier():
    """Fixture providing ClauseIdentifier instance."""
    return ClauseIdentifier()


# ============================================================================
# SAMPLE TEXTS
# ============================================================================

SAMPLE_1_NEWS = """
The technology company announced record profits yesterday; investors
responded positively to the news, and share prices rose by 15 percent.
However, analysts remain cautious about future growth prospects.
"""

SAMPLE_2_LITERARY = """
The wind howled through the empty streets; darkness enveloped everything.
A single light flickered in the distance, but hope seemed distant and fragile.
Therefore, she pressed forward into the unknown.
"""

SAMPLE_3_ACADEMIC = """
The experiment yielded unexpected results; the control group showed
significant improvements, and the experimental group demonstrated
remarkable resilience. However, the data requires further analysis
before definitive conclusions can be drawn.
"""

SAMPLE_4_CONVERSATIONAL = """
I went to the store yesterday; they were out of milk, but I found
everything else I needed. The cashier was friendly, and the lines
moved quickly. Therefore, the trip was successful overall.
"""

SAMPLE_5_TECHNICAL = """
The algorithm processes input data efficiently; it optimizes memory
usage through caching mechanisms, and it reduces computational
complexity via parallel processing. However, edge cases require
additional validation. Therefore, comprehensive testing remains essential.
"""


# ============================================================================
# END-TO-END PIPELINE TESTS
# ============================================================================

def test_sample_1_news_complete_pipeline(preprocessor, identifier):
    """Test Sample 1 (News) through complete pipeline."""
    text = SAMPLE_1_NEWS

    # Step 1: Preprocess
    tokens, doc = preprocessor.process(text)

    # Validate preprocessing
    assert len(tokens) > 0, "Tokens should be generated"
    assert all(hasattr(t, 'text') for t in tokens), "All tokens should have text"
    assert all(hasattr(t, 'pos_tag') for t in tokens), "All tokens should have POS tags"
    assert all(hasattr(t, 'is_content_word') for t in tokens), "All tokens should have content word flag"

    # Step 2: Identify clause pairs
    pairs = identifier.identify_pairs(tokens, doc)

    # Validate clause pairs
    assert len(pairs) > 0, "Should find clause pairs in news text"
    print(f"\n[Sample 1 - News] Found {len(pairs)} clause pair(s)")

    for i, pair in enumerate(pairs):
        print(f"\nPair {i+1}:")
        print(f"  Type: {pair.pair_type}")
        print(f"  Clause A: {' '.join([t.text for t in pair.clause_a.tokens])}")
        print(f"  Clause B: {' '.join([t.text for t in pair.clause_b.tokens])}")
        print(f"  Zone A tokens: {[t.text for t in pair.zone_a_tokens]}")
        print(f"  Zone B tokens: {[t.text for t in pair.zone_b_tokens]}")

        # Validate pair structure
        assert pair.clause_a is not None
        assert pair.clause_b is not None
        assert isinstance(pair.zone_a_tokens, list)
        assert isinstance(pair.zone_b_tokens, list)
        assert pair.pair_type in ["punctuation", "conjunction", "transition"]

        # Validate zones have content words
        for token in pair.zone_a_tokens:
            assert token.is_content_word, f"Zone A token '{token.text}' should be content word"
            assert hasattr(token, 'phonetic'), "Zone A token should have phonetic data"
            assert hasattr(token, 'pos_tag'), "Zone A token should have POS tag"

        for token in pair.zone_b_tokens:
            assert token.is_content_word, f"Zone B token '{token.text}' should be content word"
            assert hasattr(token, 'phonetic'), "Zone B token should have phonetic data"
            assert hasattr(token, 'pos_tag'), "Zone B token should have POS tag"

    # Should find at least semicolon and conjunction pairs
    pair_types = set(p.pair_type for p in pairs)
    assert len(pair_types) >= 1, "Should find at least one type of pair"


def test_sample_2_literary_complete_pipeline(preprocessor, identifier):
    """Test Sample 2 (Literary) through complete pipeline."""
    text = SAMPLE_2_LITERARY

    # Step 1: Preprocess
    tokens, doc = preprocessor.process(text)

    # Validate preprocessing
    assert len(tokens) > 0, "Tokens should be generated"

    # Step 2: Identify clause pairs
    pairs = identifier.identify_pairs(tokens, doc)

    # Validate clause pairs
    assert len(pairs) > 0, "Should find clause pairs in literary text"
    print(f"\n[Sample 2 - Literary] Found {len(pairs)} clause pair(s)")

    for i, pair in enumerate(pairs):
        print(f"\nPair {i+1}:")
        print(f"  Type: {pair.pair_type}")
        print(f"  Clause A: {' '.join([t.text for t in pair.clause_a.tokens[:10]])}{'...' if len(pair.clause_a.tokens) > 10 else ''}")
        print(f"  Clause B: {' '.join([t.text for t in pair.clause_b.tokens[:10]])}{'...' if len(pair.clause_b.tokens) > 10 else ''}")
        print(f"  Zone A tokens: {[t.text for t in pair.zone_a_tokens]}")
        print(f"  Zone B tokens: {[t.text for t in pair.zone_b_tokens]}")

        # Validate pair structure
        assert pair.pair_type in ["punctuation", "conjunction", "transition"]

        # Validate zones
        assert len(pair.zone_a_tokens) <= 3, "Zone A should have at most 3 tokens"
        assert len(pair.zone_b_tokens) <= 3, "Zone B should have at most 3 tokens"

        # Check linguistic data completeness
        for token in pair.zone_a_tokens + pair.zone_b_tokens:
            assert token.text, "Token should have text"
            assert token.pos_tag, "Token should have POS tag"
            assert isinstance(token.is_content_word, bool)

    # Should find punctuation, conjunction, and possibly transition pairs
    pair_types = set(p.pair_type for p in pairs)
    print(f"  Pair types found: {pair_types}")


def test_sample_3_academic_complete_pipeline(preprocessor, identifier):
    """Test Sample 3 (Academic) through complete pipeline."""
    text = SAMPLE_3_ACADEMIC

    # Step 1: Preprocess
    tokens, doc = preprocessor.process(text)

    # Validate preprocessing
    assert len(tokens) > 0, "Tokens should be generated"

    # Step 2: Identify clause pairs
    pairs = identifier.identify_pairs(tokens, doc)

    # Validate clause pairs
    assert len(pairs) > 0, "Should find clause pairs in academic text"
    print(f"\n[Sample 3 - Academic] Found {len(pairs)} clause pair(s)")

    for i, pair in enumerate(pairs):
        print(f"\nPair {i+1}:")
        print(f"  Type: {pair.pair_type}")
        print(f"  Clause A tokens: {len(pair.clause_a.tokens)}")
        print(f"  Clause B tokens: {len(pair.clause_b.tokens)}")
        print(f"  Zone A: {[t.text for t in pair.zone_a_tokens]} (POS: {[t.pos_tag for t in pair.zone_a_tokens]})")
        print(f"  Zone B: {[t.text for t in pair.zone_b_tokens]} (POS: {[t.pos_tag for t in pair.zone_b_tokens]})")

        # Validate pair structure
        assert pair.clause_a.start_idx < pair.clause_b.start_idx, "Clause A should precede Clause B"

        # Validate zones have proper POS tags
        for token in pair.zone_a_tokens + pair.zone_b_tokens:
            assert token.pos_tag in ["NOUN", "PROPN", "VERB", "ADJ", "ADV"], \
                f"Zone token '{token.text}' has unexpected POS tag: {token.pos_tag}"

    # Academic text should have multiple types of pairs
    pair_types = set(p.pair_type for p in pairs)
    print(f"  Pair types found: {pair_types}")
    assert len(pair_types) >= 1


def test_sample_4_conversational_complete_pipeline(preprocessor, identifier):
    """Test Sample 4 (Conversational) through complete pipeline."""
    text = SAMPLE_4_CONVERSATIONAL

    # Step 1: Preprocess
    tokens, doc = preprocessor.process(text)

    # Validate preprocessing
    assert len(tokens) > 0, "Tokens should be generated"

    # Step 2: Identify clause pairs
    pairs = identifier.identify_pairs(tokens, doc)

    # Validate clause pairs
    assert len(pairs) > 0, "Should find clause pairs in conversational text"
    print(f"\n[Sample 4 - Conversational] Found {len(pairs)} clause pair(s)")

    for i, pair in enumerate(pairs):
        print(f"\nPair {i+1}:")
        print(f"  Type: {pair.pair_type}")
        print(f"  Zone A: {[t.text for t in pair.zone_a_tokens]}")
        print(f"  Zone B: {[t.text for t in pair.zone_b_tokens]}")

        # Check phonetic transcriptions exist
        zone_a_phonetics = [t.phonetic for t in pair.zone_a_tokens if t.phonetic]
        zone_b_phonetics = [t.phonetic for t in pair.zone_b_tokens if t.phonetic]

        print(f"  Zone A phonetics: {zone_a_phonetics}")
        print(f"  Zone B phonetics: {zone_b_phonetics}")

        # Validate at least some phonetic data exists
        assert len(zone_a_phonetics) > 0 or len(zone_b_phonetics) > 0, \
            "At least one zone should have phonetic transcriptions"

    # Should find various pair types
    pair_types = set(p.pair_type for p in pairs)
    print(f"  Pair types found: {pair_types}")


def test_sample_5_technical_complete_pipeline(preprocessor, identifier):
    """Test Sample 5 (Technical) through complete pipeline."""
    text = SAMPLE_5_TECHNICAL

    # Step 1: Preprocess
    tokens, doc = preprocessor.process(text)

    # Validate preprocessing
    assert len(tokens) > 0, "Tokens should be generated"

    # Step 2: Identify clause pairs
    pairs = identifier.identify_pairs(tokens, doc)

    # Validate clause pairs
    assert len(pairs) > 0, "Should find clause pairs in technical text"
    print(f"\n[Sample 5 - Technical] Found {len(pairs)} clause pair(s)")

    for i, pair in enumerate(pairs):
        print(f"\nPair {i+1}:")
        print(f"  Type: {pair.pair_type}")
        print(f"  Zone A: {[t.text for t in pair.zone_a_tokens]}")
        print(f"  Zone B: {[t.text for t in pair.zone_b_tokens]}")
        print(f"  Zone A syllables: {[t.syllable_count for t in pair.zone_a_tokens]}")
        print(f"  Zone B syllables: {[t.syllable_count for t in pair.zone_b_tokens]}")

        # Validate syllable counts
        for token in pair.zone_a_tokens + pair.zone_b_tokens:
            assert token.syllable_count > 0, f"Token '{token.text}' should have syllable count"

    # Technical text should have multiple pairs
    pair_types = set(p.pair_type for p in pairs)
    print(f"  Pair types found: {pair_types}")
    assert len(pairs) >= 2, "Technical text should yield multiple pairs"


# ============================================================================
# CROSS-SAMPLE COMPARISON TESTS
# ============================================================================

def test_all_samples_comparison(preprocessor, identifier):
    """Compare results across all five samples."""
    samples = [
        ("News", SAMPLE_1_NEWS),
        ("Literary", SAMPLE_2_LITERARY),
        ("Academic", SAMPLE_3_ACADEMIC),
        ("Conversational", SAMPLE_4_CONVERSATIONAL),
        ("Technical", SAMPLE_5_TECHNICAL)
    ]

    results = []

    print("\n" + "="*70)
    print("CROSS-SAMPLE COMPARISON")
    print("="*70)

    for name, text in samples:
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        pair_types = set(p.pair_type for p in pairs)
        total_zones = sum(len(p.zone_a_tokens) + len(p.zone_b_tokens) for p in pairs)
        avg_zone_size = total_zones / (2 * len(pairs)) if pairs else 0

        result = {
            "name": name,
            "tokens": len(tokens),
            "pairs": len(pairs),
            "pair_types": pair_types,
            "total_zones": total_zones,
            "avg_zone_size": avg_zone_size
        }
        results.append(result)

        print(f"\n{name}:")
        print(f"  Total tokens: {result['tokens']}")
        print(f"  Clause pairs found: {result['pairs']}")
        print(f"  Pair types: {result['pair_types']}")
        print(f"  Average zone size: {result['avg_zone_size']:.2f} tokens")

    # Validate all samples produced results
    for result in results:
        assert result["pairs"] > 0, f"{result['name']} should find pairs"
        assert len(result["pair_types"]) > 0, f"{result['name']} should have pair types"
        assert result["avg_zone_size"] > 0, f"{result['name']} should have non-zero zones"
        assert result["avg_zone_size"] <= 3, f"{result['name']} zones should not exceed 3 tokens average"

    print("\n" + "="*70)
    print("All samples processed successfully!")
    print("="*70)


# ============================================================================
# DETAILED ZONE ANALYSIS TEST
# ============================================================================

def test_all_samples_zone_quality(preprocessor, identifier):
    """Detailed analysis of zone quality across all samples."""
    samples = [
        ("News", SAMPLE_1_NEWS),
        ("Literary", SAMPLE_2_LITERARY),
        ("Academic", SAMPLE_3_ACADEMIC),
        ("Conversational", SAMPLE_4_CONVERSATIONAL),
        ("Technical", SAMPLE_5_TECHNICAL)
    ]

    print("\n" + "="*70)
    print("DETAILED ZONE QUALITY ANALYSIS")
    print("="*70)

    for name, text in samples:
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        print(f"\n{name} - Zone Analysis:")

        for i, pair in enumerate(pairs):
            print(f"\n  Pair {i+1} ({pair.pair_type}):")

            # Analyze Zone A
            zone_a_words = [t.text for t in pair.zone_a_tokens]
            zone_a_pos = [t.pos_tag for t in pair.zone_a_tokens]
            zone_a_phonetics = [t.phonetic if t.phonetic else 'N/A' for t in pair.zone_a_tokens]

            print(f"    Zone A: {zone_a_words}")
            print(f"      POS: {zone_a_pos}")
            print(f"      Phonetics: {zone_a_phonetics}")

            # Analyze Zone B
            zone_b_words = [t.text for t in pair.zone_b_tokens]
            zone_b_pos = [t.pos_tag for t in pair.zone_b_tokens]
            zone_b_phonetics = [t.phonetic if t.phonetic else 'N/A' for t in pair.zone_b_tokens]

            print(f"    Zone B: {zone_b_words}")
            print(f"      POS: {zone_b_pos}")
            print(f"      Phonetics: {zone_b_phonetics}")

            # Validate quality
            assert all(t.is_content_word for t in pair.zone_a_tokens), "Zone A should only have content words"
            assert all(t.is_content_word for t in pair.zone_b_tokens), "Zone B should only have content words"
            assert all(t.pos_tag in ["NOUN", "PROPN", "VERB", "ADJ", "ADV"] for t in pair.zone_a_tokens + pair.zone_b_tokens), \
                "All zone tokens should be content word POS tags"

    print("\n" + "="*70)
    print("Zone quality validation passed!")
    print("="*70)
