"""
Tests for StructuralEchoAnalyzer (Task 4.2).

Covers POS pattern comparison, syllable count similarity, and combined scoring.
"""

import pytest
from specHO.models import Token
from specHO.echo_engine.structural_analyzer import (
    StructuralEchoAnalyzer,
    quick_structural_analysis
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def analyzer():
    """Fixture for StructuralEchoAnalyzer instance."""
    return StructuralEchoAnalyzer()


@pytest.fixture
def sample_tokens_identical_pos():
    """Two zones with identical POS patterns and syllable counts."""
    zone_a = [
        Token(text="happy", pos_tag="ADJ", phonetic="HH AE1 P IY0", is_content_word=True, syllable_count=2),
        Token(text="dog", pos_tag="NOUN", phonetic="D AO1 G", is_content_word=True, syllable_count=1),
        Token(text="ran", pos_tag="VERB", phonetic="R AE1 N", is_content_word=True, syllable_count=1),
    ]
    zone_b = [
        Token(text="sad", pos_tag="ADJ", phonetic="S AE1 D", is_content_word=True, syllable_count=1),
        Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T", is_content_word=True, syllable_count=1),
        Token(text="jumped", pos_tag="VERB", phonetic="JH AH1 M P T", is_content_word=True, syllable_count=1),
    ]
    return zone_a, zone_b


@pytest.fixture
def sample_tokens_different_pos():
    """Two zones with different POS patterns."""
    zone_a = [
        Token(text="happy", pos_tag="ADJ", phonetic="HH AE1 P IY0", is_content_word=True, syllable_count=2),
        Token(text="dog", pos_tag="NOUN", phonetic="D AO1 G", is_content_word=True, syllable_count=1),
    ]
    zone_b = [
        Token(text="ran", pos_tag="VERB", phonetic="R AE1 N", is_content_word=True, syllable_count=1),
        Token(text="quickly", pos_tag="ADV", phonetic="K W IH1 K L IY0", is_content_word=True, syllable_count=2),
    ]
    return zone_a, zone_b


@pytest.fixture
def sample_tokens_same_syllables():
    """Two zones with same total syllable count but different POS."""
    zone_a = [
        Token(text="cat", pos_tag="NOUN", phonetic="K AE1 T", is_content_word=True, syllable_count=1),
        Token(text="dog", pos_tag="NOUN", phonetic="D AO1 G", is_content_word=True, syllable_count=1),
        Token(text="ran", pos_tag="VERB", phonetic="R AE1 N", is_content_word=True, syllable_count=1),
    ]  # Total: 3 syllables
    zone_b = [
        Token(text="happy", pos_tag="ADJ", phonetic="HH AE1 P IY0", is_content_word=True, syllable_count=2),
        Token(text="go", pos_tag="VERB", phonetic="G OW1", is_content_word=True, syllable_count=1),
    ]  # Total: 3 syllables
    return zone_a, zone_b


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_initialization():
    """Test that StructuralEchoAnalyzer can be instantiated."""
    analyzer = StructuralEchoAnalyzer()
    assert analyzer is not None


def test_analyzer_has_analyze_method(analyzer):
    """Test that analyzer has the main analyze method."""
    assert hasattr(analyzer, 'analyze')
    assert callable(analyzer.analyze)


def test_analyzer_has_compare_pos_patterns(analyzer):
    """Test that analyzer has compare_pos_patterns method."""
    assert hasattr(analyzer, 'compare_pos_patterns')
    assert callable(analyzer.compare_pos_patterns)


def test_analyzer_has_compare_syllable_counts(analyzer):
    """Test that analyzer has compare_syllable_counts method."""
    assert hasattr(analyzer, 'compare_syllable_counts')
    assert callable(analyzer.compare_syllable_counts)


# ============================================================================
# POS PATTERN COMPARISON TESTS
# ============================================================================

class TestComparePOSPatterns:
    """Tests for compare_pos_patterns method."""

    def test_identical_pos_patterns(self, analyzer, sample_tokens_identical_pos):
        """Identical POS patterns should return 1.0."""
        zone_a, zone_b = sample_tokens_identical_pos
        similarity = analyzer.compare_pos_patterns(zone_a, zone_b)
        assert similarity == 1.0

    def test_different_pos_patterns(self, analyzer, sample_tokens_different_pos):
        """Different POS patterns should return 0.0."""
        zone_a, zone_b = sample_tokens_different_pos
        similarity = analyzer.compare_pos_patterns(zone_a, zone_b)
        assert similarity == 0.0

    def test_empty_zone_a(self, analyzer):
        """Empty zone_a should return 0.0."""
        zone_a = []
        zone_b = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        similarity = analyzer.compare_pos_patterns(zone_a, zone_b)
        assert similarity == 0.0

    def test_empty_zone_b(self, analyzer):
        """Empty zone_b should return 0.0."""
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        zone_b = []
        similarity = analyzer.compare_pos_patterns(zone_a, zone_b)
        assert similarity == 0.0

    def test_both_zones_empty(self, analyzer):
        """Both zones empty should return 0.0."""
        similarity = analyzer.compare_pos_patterns([], [])
        assert similarity == 0.0

    def test_none_pos_tags_skipped(self, analyzer):
        """Tokens with None pos_tag should be skipped."""
        zone_a = [
            Token(text="the", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=1),
            Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1),
        ]
        zone_b = [
            Token(text="a", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=1),
            Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1),
        ]
        similarity = analyzer.compare_pos_patterns(zone_a, zone_b)
        assert similarity == 1.0  # Both have pattern ["NOUN"]

    def test_all_none_pos_tags(self, analyzer):
        """All tokens with None pos_tag should return 0.0."""
        zone_a = [Token(text="the", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=1)]
        zone_b = [Token(text="a", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=1)]
        similarity = analyzer.compare_pos_patterns(zone_a, zone_b)
        assert similarity == 0.0

    def test_different_length_patterns(self, analyzer):
        """Different length POS patterns should return 0.0."""
        zone_a = [
            Token(text="happy", pos_tag="ADJ", phonetic="HAPPY", is_content_word=True, syllable_count=2),
            Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1),
        ]
        zone_b = [
            Token(text="sad", pos_tag="ADJ", phonetic="SAD", is_content_word=True, syllable_count=1),
            Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1),
            Token(text="ran", pos_tag="VERB", phonetic="RAN", is_content_word=True, syllable_count=1),
        ]
        similarity = analyzer.compare_pos_patterns(zone_a, zone_b)
        assert similarity == 0.0


# ============================================================================
# SYLLABLE COUNT COMPARISON TESTS
# ============================================================================

class TestCompareSyllableCounts:
    """Tests for compare_syllable_counts method."""

    def test_identical_syllable_counts(self, analyzer):
        """Identical syllable counts should return 1.0."""
        zone_a = [
            Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1),
            Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1),
        ]  # Total: 2
        zone_b = [
            Token(text="happy", pos_tag="ADJ", phonetic="HAPPY", is_content_word=True, syllable_count=2),
        ]  # Total: 2
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        assert similarity == 1.0

    def test_different_syllable_counts(self, analyzer):
        """Different syllable counts should return proportional similarity."""
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]  # Total: 1
        zone_b = [Token(text="elephant", pos_tag="NOUN", phonetic="ELEPHANT", is_content_word=True, syllable_count=3)]  # Total: 3
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        # Expected: 1 - (|1-3| / max(1,3)) = 1 - (2/3) = 0.333...
        assert 0.33 <= similarity <= 0.34

    def test_empty_zone_a_syllables(self, analyzer):
        """Empty zone_a should return 0.0."""
        zone_a = []
        zone_b = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        assert similarity == 0.0

    def test_empty_zone_b_syllables(self, analyzer):
        """Empty zone_b should return 0.0."""
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        zone_b = []
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        assert similarity == 0.0

    def test_both_zones_empty_syllables(self, analyzer):
        """Both zones empty should return 0.0."""
        similarity = analyzer.compare_syllable_counts([], [])
        assert similarity == 0.0

    def test_none_syllable_counts_skipped(self, analyzer):
        """Tokens with None syllable_count should be skipped."""
        zone_a = [
            Token(text="the", pos_tag="DET", phonetic="THE", is_content_word=True, syllable_count=None),
            Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1),
        ]  # Total: 1
        zone_b = [
            Token(text="a", pos_tag="DET", phonetic="A", is_content_word=True, syllable_count=None),
            Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1),
        ]  # Total: 1
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        assert similarity == 1.0

    def test_all_none_syllable_counts(self, analyzer):
        """All tokens with None syllable_count should return 0.0."""
        zone_a = [Token(text="the", pos_tag="DET", phonetic="THE", is_content_word=True, syllable_count=None)]
        zone_b = [Token(text="a", pos_tag="DET", phonetic="A", is_content_word=True, syllable_count=None)]
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        assert similarity == 0.0

    def test_both_zero_syllables(self, analyzer):
        """Both zones with 0 syllables should return 1.0."""
        zone_a = [Token(text="x", pos_tag="SYM", phonetic="X", is_content_word=True, syllable_count=0)]
        zone_b = [Token(text="y", pos_tag="SYM", phonetic="Y", is_content_word=True, syllable_count=0)]
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        assert similarity == 1.0

    def test_large_syllable_difference(self, analyzer):
        """Large syllable difference should return low similarity."""
        zone_a = [Token(text="I", pos_tag="PRON", phonetic="I", is_content_word=True, syllable_count=1)]  # Total: 1
        zone_b = [
            Token(text="extraordinary", pos_tag="ADJ", phonetic="EXTRAORDINARY", is_content_word=True, syllable_count=5),
            Token(text="revolutionary", pos_tag="ADJ", phonetic="REVOLUTIONARY", is_content_word=True, syllable_count=5),
        ]  # Total: 10
        similarity = analyzer.compare_syllable_counts(zone_a, zone_b)
        # Expected: 1 - (|1-10| / 10) = 1 - 0.9 = 0.1
        assert 0.09 <= similarity <= 0.11


# ============================================================================
# ANALYZE METHOD TESTS
# ============================================================================

class TestAnalyze:
    """Tests for the main analyze method."""

    def test_identical_structure(self, analyzer, sample_tokens_identical_pos):
        """Zones with identical POS should have high similarity."""
        zone_a, zone_b = sample_tokens_identical_pos
        score = analyzer.analyze(zone_a, zone_b)
        # POS match = 1.0, syllable similarity depends on counts
        # zone_a: 2+1+1=4 syllables, zone_b: 1+1+1=3 syllables
        # syllable_sim = 1 - (1/4) = 0.75
        # combined = 0.5*1.0 + 0.5*0.75 = 0.875
        assert 0.85 <= score <= 0.90

    def test_different_structure(self, analyzer, sample_tokens_different_pos):
        """Zones with different POS should have lower similarity."""
        zone_a, zone_b = sample_tokens_different_pos
        score = analyzer.analyze(zone_a, zone_b)
        # POS match = 0.0, syllable: zone_a=3, zone_b=3
        # syllable_sim = 1.0
        # combined = 0.5*0.0 + 0.5*1.0 = 0.5
        assert score == 0.5

    def test_same_syllables_different_pos(self, analyzer, sample_tokens_same_syllables):
        """Same syllable count but different POS should score 0.5."""
        zone_a, zone_b = sample_tokens_same_syllables
        score = analyzer.analyze(zone_a, zone_b)
        # POS different, syllables same
        # Expected: 0.5*0.0 + 0.5*1.0 = 0.5
        assert score == 0.5

    def test_empty_zone_a_returns_zero(self, analyzer):
        """Empty zone_a should return 0.0."""
        zone_a = []
        zone_b = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        score = analyzer.analyze(zone_a, zone_b)
        assert score == 0.0

    def test_empty_zone_b_returns_zero(self, analyzer):
        """Empty zone_b should return 0.0."""
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        zone_b = []
        score = analyzer.analyze(zone_a, zone_b)
        assert score == 0.0

    def test_both_zones_empty_returns_zero(self, analyzer):
        """Both zones empty should return 0.0."""
        score = analyzer.analyze([], [])
        assert score == 0.0

    def test_output_in_valid_range(self, analyzer, sample_tokens_identical_pos):
        """Output should always be in [0, 1] range."""
        zone_a, zone_b = sample_tokens_identical_pos
        score = analyzer.analyze(zone_a, zone_b)
        assert 0.0 <= score <= 1.0

    def test_single_token_zones(self, analyzer):
        """Single token zones should work correctly."""
        zone_a = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        zone_b = [Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1)]
        score = analyzer.analyze(zone_a, zone_b)
        # POS match = 1.0, syllable match = 1.0
        # combined = 0.5*1.0 + 0.5*1.0 = 1.0
        assert score == 1.0


# ============================================================================
# REAL-WORLD STRUCTURAL PATTERNS
# ============================================================================

class TestRealWorldStructures:
    """Tests with realistic structural patterns."""

    def test_parallel_adjective_noun_structure(self, analyzer):
        """Parallel ADJ-NOUN structures should score high."""
        zone_a = [
            Token(text="beautiful", pos_tag="ADJ", phonetic="BEAUTIFUL", is_content_word=True, syllable_count=3),
            Token(text="garden", pos_tag="NOUN", phonetic="GARDEN", is_content_word=True, syllable_count=2),
        ]
        zone_b = [
            Token(text="amazing", pos_tag="ADJ", phonetic="AMAZING", is_content_word=True, syllable_count=3),
            Token(text="forest", pos_tag="NOUN", phonetic="FOREST", is_content_word=True, syllable_count=2),
        ]
        score = analyzer.analyze(zone_a, zone_b)
        # POS match = 1.0, syllables identical (5 each)
        # combined = 1.0
        assert score == 1.0

    def test_different_syntactic_roles(self, analyzer):
        """Different syntactic structures should score low."""
        zone_a = [
            Token(text="quickly", pos_tag="ADV", phonetic="QUICKLY", is_content_word=True, syllable_count=2),
            Token(text="ran", pos_tag="VERB", phonetic="RAN", is_content_word=True, syllable_count=1),
        ]
        zone_b = [
            Token(text="the", pos_tag="DET", phonetic="THE", is_content_word=True, syllable_count=1),
            Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1),
        ]
        score = analyzer.analyze(zone_a, zone_b)
        # POS different (ADV-VERB vs DET-NOUN)
        # Syllables: 3 vs 2, similarity = 1 - (1/3) = 0.667
        # combined = 0.5*0.0 + 0.5*0.667 = 0.333
        assert 0.30 <= score <= 0.35

    def test_verb_heavy_zones(self, analyzer):
        """Matching verb-heavy zones should score high."""
        zone_a = [
            Token(text="jumped", pos_tag="VERB", phonetic="JUMPED", is_content_word=True, syllable_count=1),
            Token(text="ran", pos_tag="VERB", phonetic="RAN", is_content_word=True, syllable_count=1),
            Token(text="fell", pos_tag="VERB", phonetic="FELL", is_content_word=True, syllable_count=1),
        ]
        zone_b = [
            Token(text="walked", pos_tag="VERB", phonetic="WALKED", is_content_word=True, syllable_count=1),
            Token(text="climbed", pos_tag="VERB", phonetic="CLIMBED", is_content_word=True, syllable_count=1),
            Token(text="swam", pos_tag="VERB", phonetic="SWAM", is_content_word=True, syllable_count=1),
        ]
        score = analyzer.analyze(zone_a, zone_b)
        # POS match = 1.0 (all VERB), syllables identical (3 each)
        # combined = 1.0
        assert score == 1.0

    def test_complex_multisyllabic_words(self, analyzer):
        """Complex words with matching structure should score well."""
        zone_a = [
            Token(text="extraordinary", pos_tag="ADJ", phonetic="EXTRAORDINARY", is_content_word=True, syllable_count=5),
            Token(text="circumstance", pos_tag="NOUN", phonetic="CIRCUMSTANCE", is_content_word=True, syllable_count=3),
        ]
        zone_b = [
            Token(text="magnificent", pos_tag="ADJ", phonetic="MAGNIFICENT", is_content_word=True, syllable_count=4),
            Token(text="performance", pos_tag="NOUN", phonetic="PERFORMANCE", is_content_word=True, syllable_count=3),
        ]
        score = analyzer.analyze(zone_a, zone_b)
        # POS match = 1.0 (ADJ-NOUN)
        # Syllables: 8 vs 7, similarity = 1 - (1/8) = 0.875
        # combined = 0.5*1.0 + 0.5*0.875 = 0.9375
        assert 0.93 <= score <= 0.95


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_mixed_none_and_valid_data(self, analyzer):
        """Zones with mix of None and valid data should work."""
        zone_a = [
            Token(text="the", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=None),
            Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1),
            Token(text="ran", pos_tag="VERB", phonetic="RAN", is_content_word=True, syllable_count=1),
        ]
        zone_b = [
            Token(text="a", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=None),
            Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1),
            Token(text="jumped", pos_tag="VERB", phonetic="JUMPED", is_content_word=True, syllable_count=1),
        ]
        score = analyzer.analyze(zone_a, zone_b)
        # After filtering: both have [NOUN, VERB] and 2 syllables
        # POS match = 1.0, syllable match = 1.0
        # combined = 1.0
        assert score == 1.0

    def test_very_long_zones(self, analyzer):
        """Very long zones should work without errors."""
        zone_a = [Token(text=f"word{i}", pos_tag="NOUN", phonetic=f"WORD{i}", is_content_word=True, syllable_count=2)
                 for i in range(20)]
        zone_b = [Token(text=f"term{i}", pos_tag="NOUN", phonetic=f"TERM{i}", is_content_word=True, syllable_count=2)
                 for i in range(20)]
        score = analyzer.analyze(zone_a, zone_b)
        # All NOUN, all 2 syllables
        assert score == 1.0

    def test_all_data_none(self, analyzer):
        """Zones where all fields are None should return 0.0."""
        zone_a = [Token(text="x", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=None)]
        zone_b = [Token(text="y", pos_tag=None, phonetic=None, is_content_word=False, syllable_count=None)]
        score = analyzer.analyze(zone_a, zone_b)
        # No valid POS or syllables = 0.0
        assert score == 0.0


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_quick_structural_analysis(self, sample_tokens_identical_pos):
        """quick_structural_analysis should work."""
        zone_a, zone_b = sample_tokens_identical_pos
        score = quick_structural_analysis(zone_a, zone_b)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests with preprocessor output."""

    def test_with_preprocessor_output(self):
        """Test with tokens from actual preprocessor."""
        from specHO.preprocessor.pipeline import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        text = "The happy dog ran. The sad cat jumped."
        tokens, doc = preprocessor.process(text)

        # Create simple zones
        zone_a = [t for t in tokens if t.text in ["happy", "dog", "ran"]]
        zone_b = [t for t in tokens if t.text in ["sad", "cat", "jumped"]]

        analyzer = StructuralEchoAnalyzer()
        score = analyzer.analyze(zone_a, zone_b)

        # Should have valid score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_reusable_analyzer(self):
        """Test that analyzer can be reused across multiple calls."""
        analyzer = StructuralEchoAnalyzer()

        zone_a1 = [Token(text="cat", pos_tag="NOUN", phonetic="CAT", is_content_word=True, syllable_count=1)]
        zone_b1 = [Token(text="dog", pos_tag="NOUN", phonetic="DOG", is_content_word=True, syllable_count=1)]

        zone_a2 = [Token(text="run", pos_tag="VERB", phonetic="RUN", is_content_word=True, syllable_count=1)]
        zone_b2 = [Token(text="jump", pos_tag="VERB", phonetic="JUMP", is_content_word=True, syllable_count=1)]

        score1 = analyzer.analyze(zone_a1, zone_b1)
        score2 = analyzer.analyze(zone_a2, zone_b2)

        assert score1 == 1.0
        assert score2 == 1.0
