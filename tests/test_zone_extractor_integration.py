"""
Integration Tests for ZoneExtractor (Task 3.3)

Tests ZoneExtractor integration with the full clause identification pipeline:
Preprocessor → BoundaryDetector → PairRulesEngine → ZoneExtractor
"""

import pytest
from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector
from specHO.clause_identifier.pair_rules import PairRulesEngine
from specHO.clause_identifier.zone_extractor import ZoneExtractor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def preprocessor():
    """Create LinguisticPreprocessor instance."""
    return LinguisticPreprocessor()


@pytest.fixture
def detector():
    """Create ClauseBoundaryDetector instance."""
    return ClauseBoundaryDetector()


@pytest.fixture
def pair_engine():
    """Create PairRulesEngine instance."""
    return PairRulesEngine()


@pytest.fixture
def zone_extractor():
    """Create ZoneExtractor instance."""
    return ZoneExtractor()


# ============================================================================
# FULL PIPELINE INTEGRATION TESTS
# ============================================================================

def test_full_pipeline_simple_sentence(preprocessor, detector, pair_engine, zone_extractor):
    """Test full pipeline with simple semicolon-separated clauses."""
    text = "The cat sat quietly; the dog ran quickly."

    # Step 1: Preprocess
    tokens, doc = preprocessor.process(text)
    assert len(tokens) > 0

    # Step 2: Detect clause boundaries
    clauses = detector.identify_clauses(doc, tokens)
    assert len(clauses) >= 2

    # Step 3: Identify clause pairs
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)
    assert len(pairs) >= 1

    # Step 4: Extract zones
    pair = pairs[0]
    zone_a, zone_b = zone_extractor.extract_zones(pair)

    # Verify zones contain content words
    assert len(zone_a) > 0
    assert len(zone_b) > 0
    assert all(t.is_content_word for t in zone_a)
    assert all(t.is_content_word for t in zone_b)


def test_full_pipeline_conjunction(preprocessor, detector, pair_engine, zone_extractor):
    """Test full pipeline with conjunction-based pairing."""
    text = "The ancient castle stood tall, and the modern tower reached high."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    assert len(pairs) >= 1

    # Extract zones from first pair
    zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

    # Should extract content words from both clauses
    assert len(zone_a) <= 3  # At most 3
    assert len(zone_b) <= 3  # At most 3
    assert all(t.is_content_word for t in zone_a)
    assert all(t.is_content_word for t in zone_b)

    # Verify we got meaningful words (nouns, verbs, adjectives)
    zone_a_pos = [t.pos_tag for t in zone_a]
    zone_b_pos = [t.pos_tag for t in zone_b]
    assert any(pos in ["NOUN", "VERB", "ADJ"] for pos in zone_a_pos)
    assert any(pos in ["NOUN", "VERB", "ADJ"] for pos in zone_b_pos)


def test_full_pipeline_transition(preprocessor, detector, pair_engine, zone_extractor):
    """Test full pipeline with transition-based pairing."""
    text = "The system worked correctly. However, the network failed completely."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    # May get pairs from different rules
    assert len(pairs) >= 1

    # Extract zones from all pairs
    for pair in pairs:
        zone_a, zone_b = zone_extractor.extract_zones(pair)

        # Zones should only contain content words
        assert all(t.is_content_word for t in zone_a)
        assert all(t.is_content_word for t in zone_b)


def test_full_pipeline_multiple_pairs(preprocessor, detector, pair_engine, zone_extractor):
    """Test full pipeline with text generating multiple clause pairs."""
    text = "The cat sat; the dog ran, and the bird flew. Therefore, chaos ensued."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    # Should detect multiple pairs (semicolon, conjunction, transition)
    assert len(pairs) >= 2

    # Extract zones from all pairs
    all_zones = []
    for pair in pairs:
        zone_a, zone_b = zone_extractor.extract_zones(pair)
        all_zones.append((zone_a, zone_b))

        # Each zone should contain only content words
        assert all(t.is_content_word for t in zone_a)
        assert all(t.is_content_word for t in zone_b)

    # Verify we extracted multiple zone pairs
    assert len(all_zones) == len(pairs)


# ============================================================================
# ZONE QUALITY TESTS
# ============================================================================

def test_zones_have_phonetic_data(preprocessor, detector, pair_engine, zone_extractor):
    """Test that extracted zone tokens have phonetic transcriptions."""
    text = "The cat sat; the dog ran."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    assert len(pairs) >= 1

    zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

    # All tokens should have phonetic data (populated by preprocessor)
    for token in zone_a:
        assert token.phonetic != ""  # Should have ARPAbet transcription
        assert token.syllable_count > 0

    for token in zone_b:
        assert token.phonetic != ""
        assert token.syllable_count > 0


def test_zones_have_pos_tags(preprocessor, detector, pair_engine, zone_extractor):
    """Test that extracted zone tokens have POS tags."""
    text = "The quick brown fox jumped; the lazy dog slept."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    assert len(pairs) >= 1

    zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

    # All tokens should have POS tags
    for token in zone_a:
        assert token.pos_tag in ["NOUN", "VERB", "ADJ", "PROPN"]

    for token in zone_b:
        assert token.pos_tag in ["NOUN", "VERB", "ADJ", "PROPN"]


def test_zones_maintain_token_order(preprocessor, detector, pair_engine, zone_extractor):
    """Test that zones maintain the original token order."""
    text = "The ancient mysterious castle stood; the modern glass tower rose."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    assert len(pairs) >= 1

    zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

    # Zone A should be last 3 content words in order
    # Zone B should be first 3 content words in order
    # We can verify they appear in the same order as in the text

    zone_a_texts = [t.text.lower() for t in zone_a]
    zone_b_texts = [t.text.lower() for t in zone_b]

    # Check that zone words appear in text order
    text_lower = text.lower()
    if len(zone_a_texts) >= 2:
        assert text_lower.index(zone_a_texts[0]) < text_lower.index(zone_a_texts[1])
    if len(zone_b_texts) >= 2:
        assert text_lower.index(zone_b_texts[0]) < text_lower.index(zone_b_texts[1])


# ============================================================================
# EDGE CASE INTEGRATION TESTS
# ============================================================================

def test_pipeline_with_short_clauses(preprocessor, detector, pair_engine, zone_extractor):
    """Test pipeline with short clauses having few content words."""
    text = "I ran; you sat."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    if len(pairs) >= 1:
        zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

        # Short clauses should return all available content words
        assert len(zone_a) <= 3
        assert len(zone_b) <= 3
        assert all(t.is_content_word for t in zone_a)
        assert all(t.is_content_word for t in zone_b)


def test_pipeline_with_long_clauses(preprocessor, detector, pair_engine, zone_extractor):
    """Test pipeline with long clauses having many content words."""
    text = "The incredibly ancient mysterious dark castle stood majestically; the extremely modern sleek glass tower rose dramatically."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    if len(pairs) >= 1:
        zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

        # Should extract exactly 3 content words from each
        assert len(zone_a) == 3
        assert len(zone_b) == 3


def test_pipeline_preserves_clause_pair_metadata(preprocessor, detector, pair_engine, zone_extractor):
    """Test that zone extraction preserves ClausePair metadata."""
    text = "The cat sat; the dog ran."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    assert len(pairs) >= 1

    original_pair = pairs[0]
    original_type = original_pair.pair_type
    original_clause_a = original_pair.clause_a
    original_clause_b = original_pair.clause_b

    # Extract zones
    zone_a, zone_b = zone_extractor.extract_zones(original_pair)

    # Original pair should be unchanged (zones returned separately)
    assert original_pair.pair_type == original_type
    assert original_pair.clause_a == original_clause_a
    assert original_pair.clause_b == original_clause_b
    assert original_pair.zone_a_tokens == []  # Still empty (we returned zones separately)
    assert original_pair.zone_b_tokens == []


# ============================================================================
# REAL-WORLD TEXT INTEGRATION TESTS
# ============================================================================

def test_pipeline_news_text(preprocessor, detector, pair_engine, zone_extractor):
    """Test full pipeline with news-style text."""
    text = "The stock market crashed dramatically; investors lost billions immediately."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    assert len(pairs) >= 1

    zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

    # Should extract meaningful content words
    assert len(zone_a) > 0
    assert len(zone_b) > 0

    # All should be content words with linguistic data
    for token in zone_a + zone_b:
        assert token.is_content_word
        assert token.pos_tag != ""
        assert token.phonetic != ""


def test_pipeline_literary_text(preprocessor, detector, pair_engine, zone_extractor):
    """Test full pipeline with literary-style text."""
    text = "The ancient tower stood silently, and the dark clouds gathered ominously."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    assert len(pairs) >= 1

    # Extract zones from all pairs
    for pair in pairs:
        zone_a, zone_b = zone_extractor.extract_zones(pair)

        # Verify quality of extracted zones
        assert all(t.is_content_word for t in zone_a)
        assert all(t.is_content_word for t in zone_b)
        assert all(t.syllable_count > 0 for t in zone_a)
        assert all(t.syllable_count > 0 for t in zone_b)


def test_pipeline_conversational_text(preprocessor, detector, pair_engine, zone_extractor):
    """Test full pipeline with conversational text."""
    text = "I completely agree with your point. However, the implementation seems difficult."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    # Should detect at least one pair
    assert len(pairs) >= 1

    for pair in pairs:
        zone_a, zone_b = zone_extractor.extract_zones(pair)

        # Zones should contain content words
        if len(zone_a) > 0:
            assert all(t.is_content_word for t in zone_a)
        if len(zone_b) > 0:
            assert all(t.is_content_word for t in zone_b)


# ============================================================================
# PERFORMANCE AND BATCH PROCESSING TESTS
# ============================================================================

def test_extract_zones_from_multiple_pairs(preprocessor, detector, pair_engine, zone_extractor):
    """Test extracting zones from multiple clause pairs in sequence."""
    text = "The cat sat; the dog ran, and the bird flew; the fish swam."

    tokens, doc = preprocessor.process(text)
    clauses = detector.identify_clauses(doc, tokens)
    pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

    # Extract zones from all pairs
    extracted_zones = []
    for pair in pairs:
        zone_a, zone_b = zone_extractor.extract_zones(pair)
        extracted_zones.append((zone_a, zone_b))

    # Should have extracted zones for all pairs
    assert len(extracted_zones) == len(pairs)

    # All zones should be valid
    for zone_a, zone_b in extracted_zones:
        assert isinstance(zone_a, list)
        assert isinstance(zone_b, list)
        assert all(t.is_content_word for t in zone_a)
        assert all(t.is_content_word for t in zone_b)
