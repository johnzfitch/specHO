"""
Tests for Task 4.1: PhoneticEchoAnalyzer

Tests phonetic similarity analysis using Levenshtein distance on ARPAbet transcriptions.
"""

import pytest
from specHO.models import Token
from specHO.echo_engine.phonetic_analyzer import (
    PhoneticEchoAnalyzer,
    quick_phonetic_analysis
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def analyzer():
    """Fixture providing PhoneticEchoAnalyzer instance."""
    return PhoneticEchoAnalyzer()


@pytest.fixture
def identical_tokens():
    """Tokens with identical phonetics."""
    return (
        [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T", is_content_word=True, syllable_count=1)],
        [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T", is_content_word=True, syllable_count=1)]
    )


@pytest.fixture
def similar_tokens():
    """Tokens with similar phonetics (rhyming)."""
    return (
        [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T", is_content_word=True, syllable_count=1)],
        [Token(text="bat", pos_tag="NOUN", phonetic="B AE1 T", is_content_word=True, syllable_count=1)]
    )


@pytest.fixture
def different_tokens():
    """Tokens with completely different phonetics."""
    return (
        [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T", is_content_word=True, syllable_count=1)],
        [Token(text="dog", pos_tag="NOUN", phonetic="D AO1 G", is_content_word=True, syllable_count=1)]
    )


@pytest.fixture
def multi_token_zones():
    """Multi-token zones with varying similarity."""
    zone_a = [
        Token(text="running", pos_tag="VERB", phonetic="R AH1 N IH0 NG", is_content_word=True, syllable_count=2),
        Token(text="quickly", pos_tag="ADV", phonetic="K W IH1 K L IY0", is_content_word=True, syllable_count=2),
        Token(text="fast", pos_tag="ADV", phonetic="F AE1 S T", is_content_word=True, syllable_count=1)
    ]
    zone_b = [
        Token(text="walking", pos_tag="VERB", phonetic="W AO1 K IH0 NG", is_content_word=True, syllable_count=2),
        Token(text="slowly", pos_tag="ADV", phonetic="S L OW1 L IY0", is_content_word=True, syllable_count=2),
        Token(text="past", pos_tag="ADV", phonetic="P AE1 S T", is_content_word=True, syllable_count=1)
    ]
    return zone_a, zone_b


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_initialization(analyzer):
    """Test that PhoneticEchoAnalyzer initializes correctly."""
    assert analyzer is not None


def test_analyzer_has_analyze_method(analyzer):
    """Test that analyzer has required analyze method."""
    assert hasattr(analyzer, 'analyze')
    assert callable(analyzer.analyze)


def test_analyzer_has_calculate_phonetic_similarity(analyzer):
    """Test that analyzer has helper method."""
    assert hasattr(analyzer, 'calculate_phonetic_similarity')
    assert callable(analyzer.calculate_phonetic_similarity)


# ============================================================================
# CALCULATE_PHONETIC_SIMILARITY TESTS
# ============================================================================

class TestCalculatePhoneticSimilarity:
    """Tests for phonetic similarity calculation."""

    def test_identical_phonetics(self, analyzer):
        """Test similarity of identical phonetic strings."""
        phoneme = "K AE1 T"
        similarity = analyzer.calculate_phonetic_similarity(phoneme, phoneme)

        assert similarity == 1.0

    def test_completely_different(self, analyzer):
        """Test similarity of completely different phonetics."""
        phoneme_a = "K AE1 T"
        phoneme_b = "D AO1 G"
        similarity = analyzer.calculate_phonetic_similarity(phoneme_a, phoneme_b)

        # Should be > 0 (not completely dissimilar due to length normalization)
        # but < 0.8 (mostly different)
        assert 0.0 <= similarity < 0.8

    def test_one_phoneme_difference(self, analyzer):
        """Test similarity with single phoneme difference."""
        phoneme_a = "K AE1 T"  # cat
        phoneme_b = "B AE1 T"  # bat
        similarity = analyzer.calculate_phonetic_similarity(phoneme_a, phoneme_b)

        # Distance = 1 (K→B), max_length = 7
        # Expected: 1 - (1/7) ≈ 0.857
        assert 0.8 <= similarity <= 0.9

    def test_empty_string_handling(self, analyzer):
        """Test handling of empty phonetic strings."""
        similarity = analyzer.calculate_phonetic_similarity("", "K AE1 T")
        assert similarity == 0.0

        similarity = analyzer.calculate_phonetic_similarity("K AE1 T", "")
        assert similarity == 0.0

        similarity = analyzer.calculate_phonetic_similarity("", "")
        assert similarity == 0.0

    def test_similarity_in_range(self, analyzer):
        """Test that similarity is always in [0,1] range."""
        test_pairs = [
            ("K AE1 T", "B AE1 T"),
            ("R AH1 N", "W AO1 K"),
            ("HH EH1 L OW0", "G UH1 D B AY1"),
            ("S", "T"),
            ("AH0", "AH1")
        ]

        for phoneme_a, phoneme_b in test_pairs:
            similarity = analyzer.calculate_phonetic_similarity(phoneme_a, phoneme_b)
            assert 0.0 <= similarity <= 1.0, \
                f"Similarity {similarity} out of range for {phoneme_a} vs {phoneme_b}"

    def test_symmetry(self, analyzer):
        """Test that similarity is symmetric."""
        phoneme_a = "K AE1 T"
        phoneme_b = "B AE1 T"

        sim_ab = analyzer.calculate_phonetic_similarity(phoneme_a, phoneme_b)
        sim_ba = analyzer.calculate_phonetic_similarity(phoneme_b, phoneme_a)

        assert sim_ab == sim_ba


# ============================================================================
# ANALYZE METHOD TESTS
# ============================================================================

class TestAnalyze:
    """Tests for zone-level analysis."""

    def test_empty_zones_return_zero(self, analyzer):
        """Test that empty zones return 0 similarity."""
        zone_a = []
        zone_b = [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T",
                       is_content_word=True, syllable_count=1)]

        assert analyzer.analyze(zone_a, zone_b) == 0.0
        assert analyzer.analyze(zone_b, zone_a) == 0.0
        assert analyzer.analyze([], []) == 0.0

    def test_identical_single_token_zones(self, analyzer, identical_tokens):
        """Test identical single-token zones."""
        zone_a, zone_b = identical_tokens
        similarity = analyzer.analyze(zone_a, zone_b)

        assert similarity == 1.0

    def test_similar_single_token_zones(self, analyzer, similar_tokens):
        """Test similar single-token zones (rhyming)."""
        zone_a, zone_b = similar_tokens
        similarity = analyzer.analyze(zone_a, zone_b)

        # Should be high similarity (rhyme = similar phonetics)
        assert 0.7 <= similarity <= 0.95

    def test_different_single_token_zones(self, analyzer, different_tokens):
        """Test different single-token zones."""
        zone_a, zone_b = different_tokens
        similarity = analyzer.analyze(zone_a, zone_b)

        # Should be low to moderate similarity (due to length normalization)
        assert 0.0 <= similarity < 0.8

    def test_multi_token_zones(self, analyzer, multi_token_zones):
        """Test analysis of multi-token zones."""
        zone_a, zone_b = multi_token_zones
        similarity = analyzer.analyze(zone_a, zone_b)

        # Should return a value in [0,1]
        assert 0.0 <= similarity <= 1.0

    def test_best_match_selection(self, analyzer):
        """Test that best match is selected for each token."""
        # Zone A has "cat"
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T",
                       is_content_word=True, syllable_count=1)]

        # Zone B has "bat" (close) and "dog" (far)
        zone_b = [
            Token(text="bat", pos_tag="NOUN", phonetic="B AE1 T",
                 is_content_word=True, syllable_count=1),
            Token(text="dog", pos_tag="NOUN", phonetic="D AO1 G",
                 is_content_word=True, syllable_count=1)
        ]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Should pick "bat" as best match (higher similarity than "dog")
        # Similarity should be close to cat-bat similarity (~0.857)
        assert 0.8 <= similarity <= 0.9

    def test_tokens_without_phonetics(self, analyzer):
        """Test handling of tokens without phonetic transcriptions."""
        # Create tokens with None phonetics (OOV words)
        zone_a = [
            Token(text="xyzzy", pos_tag="NOUN", phonetic=None,
                 is_content_word=True, syllable_count=0),
            Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T",
                 is_content_word=True, syllable_count=1)
        ]

        zone_b = [
            Token(text="bat", pos_tag="NOUN", phonetic="B AE1 T",
                 is_content_word=True, syllable_count=1)
        ]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Should only compare "cat" vs "bat" (skip None phonetics)
        assert 0.8 <= similarity <= 0.9

    def test_all_tokens_without_phonetics(self, analyzer):
        """Test zones where all tokens lack phonetics."""
        zone_a = [
            Token(text="xyzzy", pos_tag="NOUN", phonetic=None,
                 is_content_word=True, syllable_count=0)
        ]

        zone_b = [
            Token(text="qwerty", pos_tag="NOUN", phonetic=None,
                 is_content_word=True, syllable_count=0)
        ]

        similarity = analyzer.analyze(zone_a, zone_b)

        # No valid comparisons → return 0
        assert similarity == 0.0


# ============================================================================
# REAL-WORLD PHONETIC TESTS
# ============================================================================

class TestRealWorldPhonetics:
    """Tests with real-world phonetic patterns."""

    def test_rhyming_words(self, analyzer):
        """Test detection of rhyming words."""
        # "cat" / "bat" / "hat"
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T",
                       is_content_word=True, syllable_count=1)]
        zone_b = [Token(text="hat", pos_tag="NOUN", phonetic="HH AE1 T",
                       is_content_word=True, syllable_count=1)]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Rhyming words should have high similarity
        assert similarity > 0.7

    def test_alliteration(self, analyzer):
        """Test detection of alliterative patterns."""
        # "cat" / "call"
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T",
                       is_content_word=True, syllable_count=1)]
        zone_b = [Token(text="call", pos_tag="VERB", phonetic="K AO1 L",
                       is_content_word=True, syllable_count=1)]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Alliterative words share initial phoneme
        assert similarity > 0.3  # Some similarity due to shared "K"

    def test_homophones(self, analyzer):
        """Test identical pronunciation (homophones)."""
        # "to" / "two" / "too" (all pronounced "T UW1")
        zone_a = [Token(text="to", pos_tag="PREP", phonetic="T UW1",
                       is_content_word=False, syllable_count=1)]
        zone_b = [Token(text="too", pos_tag="ADV", phonetic="T UW1",
                       is_content_word=True, syllable_count=1)]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Homophones should be identical phonetically
        assert similarity == 1.0

    def test_compound_words(self, analyzer):
        """Test multi-syllable compound words."""
        # "understand" / "undermine"
        zone_a = [Token(text="understand", pos_tag="VERB",
                       phonetic="AH2 N D ER0 S T AE1 N D",
                       is_content_word=True, syllable_count=3)]
        zone_b = [Token(text="undermine", pos_tag="VERB",
                       phonetic="AH2 N D ER0 M AY1 N",
                       is_content_word=True, syllable_count=3)]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Share "under" prefix, should have moderate similarity
        assert 0.4 <= similarity <= 0.8


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_phoneme_tokens(self, analyzer):
        """Test tokens with single phonemes."""
        zone_a = [Token(text="a", pos_tag="DET", phonetic="AH0",
                       is_content_word=False, syllable_count=1)]
        zone_b = [Token(text="I", pos_tag="PRON", phonetic="AY1",
                       is_content_word=True, syllable_count=1)]

        similarity = analyzer.analyze(zone_a, zone_b)

        assert 0.0 <= similarity <= 1.0

    def test_very_long_phonetic_strings(self, analyzer):
        """Test handling of long phonetic transcriptions."""
        # Multi-syllable words
        zone_a = [Token(text="extraordinarily", pos_tag="ADV",
                       phonetic="IH0 K S T R AO2 R D AH0 N EH1 R AH0 L IY0",
                       is_content_word=True, syllable_count=6)]
        zone_b = [Token(text="extraordinary", pos_tag="ADJ",
                       phonetic="IH0 K S T R AO1 R D AH0 N EH2 R IY0",
                       is_content_word=True, syllable_count=5)]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Related words should have high similarity
        assert similarity > 0.8

    def test_mixed_zone_sizes(self, analyzer):
        """Test zones of different sizes."""
        zone_a = [
            Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T",
                 is_content_word=True, syllable_count=1)
        ]

        zone_b = [
            Token(text="bat", pos_tag="NOUN", phonetic="B AE1 T",
                 is_content_word=True, syllable_count=1),
            Token(text="hat", pos_tag="NOUN", phonetic="HH AE1 T",
                 is_content_word=True, syllable_count=1),
            Token(text="rat", pos_tag="NOUN", phonetic="R AE1 T",
                 is_content_word=True, syllable_count=1)
        ]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Should find best match from zone_b
        assert 0.7 <= similarity <= 0.9


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_quick_phonetic_analysis(similar_tokens):
    """Test the quick_phonetic_analysis convenience function."""
    zone_a, zone_b = similar_tokens
    similarity = quick_phonetic_analysis(zone_a, zone_b)

    assert isinstance(similarity, float)
    assert 0.0 <= similarity <= 1.0


def test_quick_analysis_matches_class_method(similar_tokens):
    """Test that quick function matches class method."""
    zone_a, zone_b = similar_tokens

    # Using class
    analyzer = PhoneticEchoAnalyzer()
    similarity_class = analyzer.analyze(zone_a, zone_b)

    # Using convenience function
    similarity_quick = quick_phonetic_analysis(zone_a, zone_b)

    assert similarity_class == similarity_quick


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with preprocessor output."""

    def test_with_preprocessor_output(self):
        """Test analyzer with real preprocessor output."""
        from specHO.preprocessor.pipeline import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        text = "The cat sat; the bat ran."

        tokens, doc = preprocessor.process(text)

        # Extract some tokens as zones
        zone_a = [t for t in tokens if t.text in ["cat", "sat"]]
        zone_b = [t for t in tokens if t.text in ["bat", "ran"]]

        # Ensure we have tokens
        assert len(zone_a) > 0
        assert len(zone_b) > 0

        # Analyze
        analyzer = PhoneticEchoAnalyzer()
        similarity = analyzer.analyze(zone_a, zone_b)

        # Should return valid similarity
        assert 0.0 <= similarity <= 1.0

    def test_reusable_analyzer(self):
        """Test that analyzer can be reused across multiple calls."""
        analyzer = PhoneticEchoAnalyzer()

        test_cases = [
            ([Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T",
                   is_content_word=True, syllable_count=1)],
             [Token(text="bat", pos_tag="NOUN", phonetic="B AE1 T",
                   is_content_word=True, syllable_count=1)]),
            ([Token(text="dog", pos_tag="NOUN", phonetic="D AO1 G",
                   is_content_word=True, syllable_count=1)],
             [Token(text="log", pos_tag="NOUN", phonetic="L AO1 G",
                   is_content_word=True, syllable_count=1)]),
        ]

        for zone_a, zone_b in test_cases:
            similarity = analyzer.analyze(zone_a, zone_b)
            assert 0.0 <= similarity <= 1.0
