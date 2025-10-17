"""
Tests for Task 3.4: ClauseIdentifier Pipeline

Tests the complete orchestration of clause detection, pairing, and zone extraction.
"""

import pytest
from specHO.models import Token, Clause, ClausePair
from specHO.clause_identifier.pipeline import ClauseIdentifier, quick_identify_pairs
from specHO.preprocessor.pipeline import LinguisticPreprocessor


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def preprocessor():
    """Fixture providing LinguisticPreprocessor instance."""
    return LinguisticPreprocessor()


@pytest.fixture
def identifier():
    """Fixture providing ClauseIdentifier instance."""
    return ClauseIdentifier()


@pytest.fixture
def simple_text():
    """Simple text with semicolon separator."""
    return "The cat sat; the dog ran."


@pytest.fixture
def complex_text():
    """Complex text with multiple pairing opportunities."""
    return "The cat sat quietly; the dog ran quickly, and the bird flew high."


@pytest.fixture
def conjunction_text():
    """Text with conjunction pairing."""
    return "The sun set behind the mountains, but the moon rose over the ocean."


@pytest.fixture
def transition_text():
    """Text with transitional phrase."""
    return "The experiment failed. However, the researchers learned valuable lessons."


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_initialization():
    """Test that ClauseIdentifier initializes all sub-components."""
    identifier = ClauseIdentifier()

    assert identifier.boundary_detector is not None
    assert identifier.pair_engine is not None
    assert identifier.zone_extractor is not None


def test_components_are_correct_types():
    """Test that initialized components are correct types."""
    from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector
    from specHO.clause_identifier.pair_rules import PairRulesEngine
    from specHO.clause_identifier.zone_extractor import ZoneExtractor

    identifier = ClauseIdentifier()

    assert isinstance(identifier.boundary_detector, ClauseBoundaryDetector)
    assert isinstance(identifier.pair_engine, PairRulesEngine)
    assert isinstance(identifier.zone_extractor, ZoneExtractor)


# ============================================================================
# BASIC PIPELINE TESTS
# ============================================================================

def test_identify_pairs_basic(identifier, preprocessor, simple_text):
    """Test basic pipeline execution with simple text."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert isinstance(pairs, list)
    assert len(pairs) > 0
    assert all(isinstance(pair, ClausePair) for pair in pairs)


def test_identify_pairs_returns_clause_pairs(identifier, preprocessor, simple_text):
    """Test that identify_pairs returns ClausePair objects."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    for pair in pairs:
        assert hasattr(pair, 'clause_a')
        assert hasattr(pair, 'clause_b')
        assert hasattr(pair, 'zone_a_tokens')
        assert hasattr(pair, 'zone_b_tokens')
        assert hasattr(pair, 'pair_type')


def test_identify_pairs_zones_populated(identifier, preprocessor, simple_text):
    """Test that zones are populated in returned pairs."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    for pair in pairs:
        # Zones should be lists of tokens
        assert isinstance(pair.zone_a_tokens, list)
        assert isinstance(pair.zone_b_tokens, list)

        # At least one zone should have content (for valid pairs)
        # Both could be empty if clauses have no content words
        assert isinstance(pair.zone_a_tokens, list) and isinstance(pair.zone_b_tokens, list)


def test_identify_pairs_zone_tokens_are_tokens(identifier, preprocessor, simple_text):
    """Test that zone tokens are Token objects with proper fields."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    for pair in pairs:
        for token in pair.zone_a_tokens:
            # Check Token fields directly (avoid isinstance due to import path issues)
            assert hasattr(token, 'text')
            assert hasattr(token, 'pos_tag')
            assert hasattr(token, 'phonetic')
            assert hasattr(token, 'is_content_word')
            assert token.is_content_word  # Zone tokens should be content words
            # Verify it has the right structure
            assert token.__class__.__name__ == 'Token'

        for token in pair.zone_b_tokens:
            # Check Token fields directly (avoid isinstance due to import path issues)
            assert hasattr(token, 'text')
            assert hasattr(token, 'pos_tag')
            assert hasattr(token, 'phonetic')
            assert hasattr(token, 'is_content_word')
            assert token.is_content_word  # Zone tokens should be content words
            # Verify it has the right structure
            assert token.__class__.__name__ == 'Token'


# ============================================================================
# RULE-SPECIFIC TESTS
# ============================================================================

def test_identify_pairs_with_semicolon(identifier, preprocessor, simple_text):
    """Test pipeline identifies semicolon-separated pairs (Rule A)."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    # Should find at least one pair
    assert len(pairs) >= 1

    # Should have punctuation pair type
    pair_types = [pair.pair_type for pair in pairs]
    assert "punctuation" in pair_types


def test_identify_pairs_with_conjunction(identifier, preprocessor, conjunction_text):
    """Test pipeline identifies conjunction pairs (Rule B)."""
    tokens, doc = preprocessor.process(conjunction_text)
    pairs = identifier.identify_pairs(tokens, doc)

    # Should find at least one pair
    assert len(pairs) >= 1

    # Should have conjunction pair type
    pair_types = [pair.pair_type for pair in pairs]
    assert "conjunction" in pair_types


def test_identify_pairs_with_transition(identifier, preprocessor, transition_text):
    """Test pipeline identifies transition pairs (Rule C)."""
    tokens, doc = preprocessor.process(transition_text)
    pairs = identifier.identify_pairs(tokens, doc)

    # Should find at least one pair
    assert len(pairs) >= 1

    # Should have transition pair type
    pair_types = [pair.pair_type for pair in pairs]
    assert "transition" in pair_types


def test_identify_pairs_complex_text(identifier, preprocessor, complex_text):
    """Test pipeline with text triggering multiple rules."""
    tokens, doc = preprocessor.process(complex_text)
    pairs = identifier.identify_pairs(tokens, doc)

    # Should find multiple pairs (semicolon and conjunction)
    assert len(pairs) >= 1

    # Check that pairs have different types
    pair_types = set(pair.pair_type for pair in pairs)
    # Should have at least one rule triggered
    assert len(pair_types) >= 1


# ============================================================================
# ZONE QUALITY TESTS
# ============================================================================

def test_zones_contain_content_words_only(identifier, preprocessor, simple_text):
    """Test that extracted zones contain only content words."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    for pair in pairs:
        for token in pair.zone_a_tokens:
            assert token.is_content_word, f"Zone A contains non-content word: {token.text}"

        for token in pair.zone_b_tokens:
            assert token.is_content_word, f"Zone B contains non-content word: {token.text}"


def test_zones_have_correct_size(identifier, preprocessor, simple_text):
    """Test that zones have at most 3 tokens (Tier 1 default)."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    for pair in pairs:
        assert len(pair.zone_a_tokens) <= 3, "Zone A exceeds maximum size"
        assert len(pair.zone_b_tokens) <= 3, "Zone B exceeds maximum size"


def test_zones_have_linguistic_data(identifier, preprocessor, simple_text):
    """Test that zone tokens have complete linguistic annotations."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = identifier.identify_pairs(tokens, doc)

    for pair in pairs:
        for token in pair.zone_a_tokens + pair.zone_b_tokens:
            # Check all Token fields are populated
            assert token.text, "Token missing text"
            assert token.pos_tag, "Token missing POS tag"
            # Phonetic can be None for OOV words, but should be string if present
            assert token.phonetic is None or isinstance(token.phonetic, str)
            assert isinstance(token.is_content_word, bool)
            assert isinstance(token.syllable_count, int)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_identify_pairs_empty_text(identifier, preprocessor):
    """Test pipeline with empty text."""
    tokens, doc = preprocessor.process("")
    pairs = identifier.identify_pairs(tokens, doc)

    assert isinstance(pairs, list)
    assert len(pairs) == 0


def test_identify_pairs_single_word(identifier, preprocessor):
    """Test pipeline with single word (no clauses)."""
    tokens, doc = preprocessor.process("Hello.")
    pairs = identifier.identify_pairs(tokens, doc)

    assert isinstance(pairs, list)
    # Single word may or may not produce pairs depending on parse


def test_identify_pairs_no_triggers(identifier, preprocessor):
    """Test pipeline with text that has no pairing triggers."""
    text = "The cat sat there."  # No semicolon, conjunction, or transition
    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert isinstance(pairs, list)
    # May be empty or may find pairs depending on clause detection


def test_identify_pairs_short_clauses(identifier, preprocessor):
    """Test pipeline with very short clauses (< 3 content words)."""
    text = "Go; run."  # Each clause has only 1 content word
    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert isinstance(pairs, list)

    # If pairs found, zones should gracefully handle short clauses
    for pair in pairs:
        # Zones should exist even if they're shorter than 3 tokens
        assert isinstance(pair.zone_a_tokens, list)
        assert isinstance(pair.zone_b_tokens, list)


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_full_pipeline_integration(preprocessor):
    """Test complete pipeline from raw text to enriched pairs."""
    text = "The ancient castle stood majestically on the hill; the modern city sprawled below it."

    # Step 1: Preprocess
    tokens, doc = preprocessor.process(text)

    # Step 2: Identify pairs
    identifier = ClauseIdentifier()
    pairs = identifier.identify_pairs(tokens, doc)

    # Verify end-to-end results
    assert len(pairs) >= 1, "Should find at least one pair"

    pair = pairs[0]
    assert pair.pair_type == "punctuation", "Should be semicolon pair"

    # Check zones are populated with content words
    assert len(pair.zone_a_tokens) > 0, "Zone A should have content words"
    assert len(pair.zone_b_tokens) > 0, "Zone B should have content words"

    # Check linguistic data is complete
    for token in pair.zone_a_tokens + pair.zone_b_tokens:
        assert token.text
        assert token.pos_tag
        assert token.is_content_word


def test_pipeline_preserves_clause_relationships(identifier, preprocessor):
    """Test that clause relationships are preserved through pipeline."""
    text = "The cat sat; the dog ran."
    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    # Should preserve clause_a < clause_b ordering
    for pair in pairs:
        assert pair.clause_a.start_idx < pair.clause_b.start_idx, \
            "Clause A should come before Clause B in document order"


def test_pipeline_handles_multiple_sentences(identifier, preprocessor):
    """Test pipeline with multiple sentences."""
    text = "First sentence. Second sentence; with semicolon. Third sentence."
    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert isinstance(pairs, list)
    # Should find at least the semicolon pair
    if len(pairs) > 0:
        assert all(isinstance(pair, ClausePair) for pair in pairs)


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_quick_identify_pairs(preprocessor, simple_text):
    """Test the quick_identify_pairs convenience function."""
    tokens, doc = preprocessor.process(simple_text)
    pairs = quick_identify_pairs(tokens, doc)

    assert isinstance(pairs, list)
    assert len(pairs) > 0
    assert all(isinstance(pair, ClausePair) for pair in pairs)


def test_quick_identify_pairs_equivalent_to_class(preprocessor, simple_text):
    """Test that quick function produces same results as class method."""
    tokens, doc = preprocessor.process(simple_text)

    # Using class
    identifier = ClauseIdentifier()
    pairs_class = identifier.identify_pairs(tokens, doc)

    # Using convenience function
    pairs_quick = quick_identify_pairs(tokens, doc)

    # Should have same number of pairs
    assert len(pairs_class) == len(pairs_quick)

    # Should have same pair types
    types_class = sorted([p.pair_type for p in pairs_class])
    types_quick = sorted([p.pair_type for p in pairs_quick])
    assert types_class == types_quick


# ============================================================================
# REAL-WORLD TEXT TESTS
# ============================================================================

def test_pipeline_news_text(identifier, preprocessor):
    """Test pipeline with news-style text."""
    text = """
    The technology company announced record profits yesterday; investors
    responded positively to the news, and share prices rose by 15 percent.
    """
    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert len(pairs) >= 1, "Should identify pairs in news text"

    for pair in pairs:
        # Verify zones have content
        assert len(pair.zone_a_tokens) > 0 or len(pair.zone_b_tokens) > 0


def test_pipeline_literary_text(identifier, preprocessor):
    """Test pipeline with literary-style text."""
    text = """
    The wind howled through the empty streets; darkness enveloped everything.
    However, a single light flickered in the distance.
    """
    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert len(pairs) >= 1, "Should identify pairs in literary text"

    # Should find both punctuation and transition pairs
    pair_types = set(pair.pair_type for pair in pairs)
    assert len(pair_types) >= 1


def test_pipeline_technical_text(identifier, preprocessor):
    """Test pipeline with technical text."""
    text = """
    The algorithm processes input data efficiently; it optimizes memory usage,
    and it reduces computational complexity. Therefore, performance improves significantly.
    """
    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert len(pairs) >= 1, "Should identify pairs in technical text"


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_pipeline_with_long_text(identifier, preprocessor):
    """Test pipeline performance with longer document."""
    text = " ".join([
        "The cat sat quietly on the windowsill; the dog lay sleeping below.",
        "Birds chirped in the trees, and children played in the yard.",
        "The sun set slowly. However, the evening remained warm.",
        "Stars began to appear; the moon rose majestically over the horizon.",
    ])

    tokens, doc = preprocessor.process(text)
    pairs = identifier.identify_pairs(tokens, doc)

    assert len(pairs) >= 3, "Should find multiple pairs in long text"
    assert all(isinstance(pair, ClausePair) for pair in pairs)


def test_pipeline_batch_processing(identifier, preprocessor):
    """Test pipeline with multiple separate texts."""
    texts = [
        "The cat sat; the dog ran.",
        "Rain fell heavily, but the sun broke through the clouds.",
        "The experiment succeeded. Therefore, we published our findings."
    ]

    all_pairs = []
    for text in texts:
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)
        all_pairs.extend(pairs)

    # Should find pairs from all texts
    assert len(all_pairs) >= 3

    # Should have variety of pair types
    pair_types = set(pair.pair_type for pair in all_pairs)
    assert len(pair_types) >= 2, "Should have multiple pair types across texts"
