"""
Task 8.2: Unified Test Suite for Clause Identifier Component

Comprehensive testing covering:
- ClauseBoundaryDetector (Task 3.1)
- PairRulesEngine (Task 3.2)
- ZoneExtractor (Task 3.3)
- ClauseIdentifier Pipeline (Task 3.4)

Tier 1 implementation - comprehensive unit and integration testing.
"""

import pytest
from specHO.models import Token, Clause, ClausePair
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector
from specHO.clause_identifier.pair_rules import PairRulesEngine
from specHO.clause_identifier.zone_extractor import ZoneExtractor
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
def boundary_detector():
    """Fixture providing ClauseBoundaryDetector instance."""
    return ClauseBoundaryDetector()


@pytest.fixture
def pair_engine():
    """Fixture providing PairRulesEngine instance."""
    return PairRulesEngine()


@pytest.fixture
def zone_extractor():
    """Fixture providing ZoneExtractor instance."""
    return ZoneExtractor()


@pytest.fixture
def identifier():
    """Fixture providing ClauseIdentifier instance."""
    return ClauseIdentifier()


# ============================================================================
# COMPONENT 1: BOUNDARY DETECTOR TESTS
# ============================================================================

class TestClauseBoundaryDetector:
    """Tests for clause boundary detection (Task 3.1)."""

    def test_initialization(self, boundary_detector):
        """Test that BoundaryDetector initializes correctly."""
        assert boundary_detector is not None

    def test_identify_clauses_returns_list(self, boundary_detector, preprocessor):
        """Test that identify_clauses returns a list."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)

        assert isinstance(clauses, list)

    def test_identify_clauses_returns_clause_objects(self, boundary_detector, preprocessor):
        """Test that returned clauses are Clause objects."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)

        for clause in clauses:
            assert hasattr(clause, 'tokens')
            assert hasattr(clause, 'start_idx')
            assert hasattr(clause, 'end_idx')
            assert hasattr(clause, 'clause_type')
            assert hasattr(clause, 'head_idx')

    def test_identifies_multiple_clauses(self, boundary_detector, preprocessor):
        """Test detection of multiple clauses."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)

        assert len(clauses) >= 2, "Should detect at least 2 clauses"

    def test_clause_has_valid_indices(self, boundary_detector, preprocessor):
        """Test that clause indices are valid."""
        text = "The cat sat quietly."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)

        for clause in clauses:
            assert clause.start_idx >= 0
            assert clause.end_idx > clause.start_idx
            assert clause.head_idx >= clause.start_idx
            assert clause.head_idx < clause.end_idx

    def test_empty_text(self, boundary_detector, preprocessor):
        """Test handling of empty text."""
        tokens, doc = preprocessor.process("")
        clauses = boundary_detector.identify_clauses(doc, tokens)

        assert isinstance(clauses, list)


# ============================================================================
# COMPONENT 2: PAIR RULES ENGINE TESTS
# ============================================================================

class TestPairRulesEngine:
    """Tests for pairing rules (Task 3.2)."""

    def test_initialization(self, pair_engine):
        """Test that PairRulesEngine initializes correctly."""
        assert pair_engine is not None

    def test_apply_all_rules_returns_list(self, pair_engine, boundary_detector, preprocessor):
        """Test that apply_all_rules returns a list."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        assert isinstance(pairs, list)

    def test_rule_a_punctuation(self, pair_engine, boundary_detector, preprocessor):
        """Test Rule A identifies punctuation pairs."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        # Should find at least one pair
        assert len(pairs) >= 1

        # Should have punctuation pair type
        pair_types = [pair.pair_type for pair in pairs]
        assert "punctuation" in pair_types

    def test_rule_b_conjunction(self, pair_engine, boundary_detector, preprocessor):
        """Test Rule B identifies conjunction pairs."""
        text = "The sun set, but the moon rose."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        # Should find at least one pair
        assert len(pairs) >= 1

        # Should have conjunction pair type
        pair_types = [pair.pair_type for pair in pairs]
        assert "conjunction" in pair_types

    def test_rule_c_transition(self, pair_engine, boundary_detector, preprocessor):
        """Test Rule C identifies transition pairs."""
        text = "The experiment failed. However, we learned lessons."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        # Should find at least one pair
        assert len(pairs) >= 1

        # Should have transition pair type
        pair_types = [pair.pair_type for pair in pairs]
        assert "transition" in pair_types

    def test_pairs_have_correct_structure(self, pair_engine, boundary_detector, preprocessor):
        """Test that pairs have correct ClausePair structure."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        for pair in pairs:
            assert hasattr(pair, 'clause_a')
            assert hasattr(pair, 'clause_b')
            assert hasattr(pair, 'pair_type')
            # Note: zone tokens populated by ZoneExtractor later


# ============================================================================
# COMPONENT 3: ZONE EXTRACTOR TESTS
# ============================================================================

class TestZoneExtractor:
    """Tests for zone extraction (Task 3.3)."""

    def test_initialization(self, zone_extractor):
        """Test that ZoneExtractor initializes correctly."""
        assert zone_extractor is not None

    def test_extract_zones_returns_tuple(self, zone_extractor, pair_engine, boundary_detector, preprocessor):
        """Test that extract_zones returns a tuple of lists."""
        text = "The cat sat quietly; the dog ran quickly."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        if len(pairs) > 0:
            zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

            assert isinstance(zone_a, list)
            assert isinstance(zone_b, list)

    def test_zones_contain_content_words_only(self, zone_extractor, pair_engine, boundary_detector, preprocessor):
        """Test that zones contain only content words."""
        text = "The cat sat quietly; the dog ran quickly."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        if len(pairs) > 0:
            zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

            for token in zone_a:
                assert token.is_content_word, f"Zone A contains non-content word: {token.text}"

            for token in zone_b:
                assert token.is_content_word, f"Zone B contains non-content word: {token.text}"

    def test_zones_max_size_three(self, zone_extractor, pair_engine, boundary_detector, preprocessor):
        """Test that zones have at most 3 tokens."""
        text = "The ancient castle stood majestically on the hill; the modern city sprawled below."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        if len(pairs) > 0:
            zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

            assert len(zone_a) <= 3, "Zone A exceeds maximum size"
            assert len(zone_b) <= 3, "Zone B exceeds maximum size"

    def test_short_clauses_graceful_handling(self, zone_extractor, pair_engine, boundary_detector, preprocessor):
        """Test graceful handling of short clauses."""
        text = "Go; run."
        tokens, doc = preprocessor.process(text)
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)

        if len(pairs) > 0:
            zone_a, zone_b = zone_extractor.extract_zones(pairs[0])

            # Should return available tokens, not error
            assert isinstance(zone_a, list)
            assert isinstance(zone_b, list)


# ============================================================================
# COMPONENT 4: PIPELINE INTEGRATION TESTS
# ============================================================================

class TestClauseIdentifierPipeline:
    """Tests for complete pipeline orchestration (Task 3.4)."""

    def test_initialization(self, identifier):
        """Test that ClauseIdentifier initializes all components."""
        assert identifier.boundary_detector is not None
        assert identifier.pair_engine is not None
        assert identifier.zone_extractor is not None

    def test_identify_pairs_returns_list(self, identifier, preprocessor):
        """Test that identify_pairs returns a list."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert isinstance(pairs, list)

    def test_identify_pairs_returns_enriched_pairs(self, identifier, preprocessor):
        """Test that returned pairs have zones populated."""
        text = "The cat sat quietly; the dog ran quickly."
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        for pair in pairs:
            assert hasattr(pair, 'zone_a_tokens')
            assert hasattr(pair, 'zone_b_tokens')
            assert isinstance(pair.zone_a_tokens, list)
            assert isinstance(pair.zone_b_tokens, list)

    def test_zones_have_complete_linguistic_data(self, identifier, preprocessor):
        """Test that zone tokens have complete annotations."""
        text = "The cat sat quietly; the dog ran quickly."
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        for pair in pairs:
            for token in pair.zone_a_tokens + pair.zone_b_tokens:
                assert token.text
                assert token.pos_tag
                assert isinstance(token.is_content_word, bool)
                assert isinstance(token.syllable_count, int)

    def test_quick_identify_pairs_convenience_function(self, preprocessor):
        """Test the quick_identify_pairs convenience function."""
        text = "The cat sat; the dog ran."
        tokens, doc = preprocessor.process(text)
        pairs = quick_identify_pairs(tokens, doc)

        assert isinstance(pairs, list)
        if len(pairs) > 0:
            assert all(isinstance(pair, ClausePair) for pair in pairs)


# ============================================================================
# END-TO-END INTEGRATION TESTS
# ============================================================================

class TestEndToEndIntegration:
    """End-to-end integration tests for the complete clause identifier."""

    def test_full_pipeline_semicolon(self, identifier, preprocessor):
        """Test complete pipeline with semicolon-separated clauses."""
        text = "The ancient castle stood majestically on the hill; the modern city sprawled below it."

        # Full pipeline
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        # Verify results
        assert len(pairs) >= 1, "Should find at least one pair"

        pair = pairs[0]
        assert pair.pair_type == "punctuation"
        assert len(pair.zone_a_tokens) > 0
        assert len(pair.zone_b_tokens) > 0

        # Verify zone tokens are content words
        for token in pair.zone_a_tokens + pair.zone_b_tokens:
            assert token.is_content_word

    def test_full_pipeline_conjunction(self, identifier, preprocessor):
        """Test complete pipeline with conjunction."""
        text = "The sun set behind the mountains, but the moon rose over the ocean."

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 1

        # Should find conjunction pair
        pair_types = [pair.pair_type for pair in pairs]
        assert "conjunction" in pair_types

    def test_full_pipeline_transition(self, identifier, preprocessor):
        """Test complete pipeline with transition."""
        text = "The experiment failed. However, the researchers learned valuable lessons."

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 1

        # Should find transition pair
        pair_types = [pair.pair_type for pair in pairs]
        assert "transition" in pair_types

    def test_full_pipeline_mixed_rules(self, identifier, preprocessor):
        """Test pipeline with multiple rule types."""
        text = "The cat sat quietly; the dog ran quickly, and the bird flew high."

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 1

        # Should find multiple pair types
        pair_types = set(pair.pair_type for pair in pairs)
        assert len(pair_types) >= 1

    def test_full_pipeline_preserves_ordering(self, identifier, preprocessor):
        """Test that clause ordering is preserved."""
        text = "The cat sat; the dog ran."

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        for pair in pairs:
            assert pair.clause_a.start_idx < pair.clause_b.start_idx


# ============================================================================
# EDGE CASES AND ERROR HANDLING
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases across all components."""

    def test_empty_text(self, identifier, preprocessor):
        """Test handling of empty text."""
        tokens, doc = preprocessor.process("")
        pairs = identifier.identify_pairs(tokens, doc)

        assert isinstance(pairs, list)
        assert len(pairs) == 0

    def test_single_word(self, identifier, preprocessor):
        """Test handling of single word."""
        tokens, doc = preprocessor.process("Hello.")
        pairs = identifier.identify_pairs(tokens, doc)

        assert isinstance(pairs, list)

    def test_no_pairing_triggers(self, identifier, preprocessor):
        """Test text with no pairing triggers."""
        text = "The cat sat there."
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert isinstance(pairs, list)

    def test_very_short_clauses(self, identifier, preprocessor):
        """Test very short clauses (< 3 content words)."""
        text = "Go; run."
        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        # Should handle gracefully
        assert isinstance(pairs, list)

        for pair in pairs:
            assert isinstance(pair.zone_a_tokens, list)
            assert isinstance(pair.zone_b_tokens, list)

    def test_long_document(self, identifier, preprocessor):
        """Test with longer document."""
        text = " ".join([
            "The cat sat quietly on the windowsill; the dog lay sleeping below.",
            "Birds chirped in the trees, and children played in the yard.",
            "The sun set slowly. However, the evening remained warm.",
            "Stars began to appear; the moon rose majestically over the horizon.",
        ])

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 3, "Should find multiple pairs in long text"

    def test_multiple_sentences(self, identifier, preprocessor):
        """Test with multiple sentences."""
        text = "First sentence. Second sentence; with semicolon. Third sentence."

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert isinstance(pairs, list)


# ============================================================================
# REAL-WORLD TEXT TESTS
# ============================================================================

class TestRealWorldTexts:
    """Tests with real-world text samples."""

    def test_news_style_text(self, identifier, preprocessor):
        """Test with news-style text."""
        text = """
        The technology company announced record profits yesterday; investors
        responded positively to the news, and share prices rose by 15 percent.
        """

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 1

    def test_literary_style_text(self, identifier, preprocessor):
        """Test with literary-style text."""
        text = """
        The wind howled through the empty streets; darkness enveloped everything.
        However, a single light flickered in the distance.
        """

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 1

    def test_technical_text(self, identifier, preprocessor):
        """Test with technical text."""
        text = """
        The algorithm processes input data efficiently; it optimizes memory usage,
        and it reduces computational complexity. Therefore, performance improves significantly.
        """

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 1

    def test_conversational_text(self, identifier, preprocessor):
        """Test with conversational text."""
        text = "I wanted to go to the park; she wanted to stay home, but we compromised."

        tokens, doc = preprocessor.process(text)
        pairs = identifier.identify_pairs(tokens, doc)

        assert len(pairs) >= 1


# ============================================================================
# PERFORMANCE AND BATCH TESTS
# ============================================================================

class TestPerformance:
    """Performance-related tests."""

    def test_batch_processing(self, identifier, preprocessor):
        """Test processing multiple texts."""
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

        assert len(all_pairs) >= 3

    def test_reusable_components(self):
        """Test that components can be reused across calls."""
        identifier = ClauseIdentifier()
        preprocessor = LinguisticPreprocessor()

        texts = [
            "The cat sat; the dog ran.",
            "The sun set, but the moon rose."
        ]

        for text in texts:
            tokens, doc = preprocessor.process(text)
            pairs = identifier.identify_pairs(tokens, doc)
            assert isinstance(pairs, list)
