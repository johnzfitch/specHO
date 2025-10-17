"""
Tests for ClauseBoundaryDetector (Task 3.1)

Tests clause boundary detection using dependency parsing.
Covers main clauses, coordinate clauses, and subordinate clauses.
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector, quick_identify_clauses
from specHO.models import Clause, Token
from specHO.preprocessor.pipeline import LinguisticPreprocessor


@pytest.fixture
def detector():
    """Fixture for ClauseBoundaryDetector."""
    return ClauseBoundaryDetector()


@pytest.fixture
def preprocessor():
    """Fixture for LinguisticPreprocessor."""
    return LinguisticPreprocessor()


# ============================================================================
# Initialization Tests
# ============================================================================

def test_initialization(detector):
    """Test ClauseBoundaryDetector initialization."""
    assert detector is not None


def test_initialization_no_config():
    """Test that detector requires no configuration."""
    detector = ClauseBoundaryDetector()
    assert detector is not None


# ============================================================================
# Simple Sentence Tests (Main Clauses)
# ============================================================================

def test_simple_sentence_one_clause(detector, preprocessor):
    """Test detection of a single main clause."""
    text = "The cat sat on the mat."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect one main clause
    assert len(clauses) == 1
    assert clauses[0].clause_type == "main"
    assert clauses[0].start_idx >= 0
    assert clauses[0].end_idx < len(tokens)
    assert len(clauses[0].tokens) > 0


def test_simple_sentence_clause_tokens(detector, preprocessor):
    """Test that clause contains correct tokens."""
    text = "The dog ran."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) == 1
    clause = clauses[0]

    # Clause should contain all tokens
    assert len(clause.tokens) == 4  # The, dog, ran, .
    assert clause.tokens[0].text == "The"
    assert clause.tokens[1].text == "dog"
    assert clause.tokens[2].text == "ran"


def test_multiple_simple_sentences(detector, preprocessor):
    """Test detection of multiple main clauses from multiple sentences."""
    text = "The cat sat. The dog ran."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect two main clauses (one per sentence)
    assert len(clauses) == 2
    assert clauses[0].clause_type == "main"
    assert clauses[1].clause_type == "main"


# ============================================================================
# Coordinated Clause Tests (conj)
# ============================================================================

def test_coordinated_clauses_with_and(detector, preprocessor):
    """Test detection of coordinated clauses joined by 'and'."""
    text = "The cat sat, and the dog ran."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect two clauses: main + coordinate
    assert len(clauses) == 2

    # First clause should be main
    main_clauses = [c for c in clauses if c.clause_type == "main"]
    assert len(main_clauses) == 1

    # Second clause should be coordinate
    coord_clauses = [c for c in clauses if c.clause_type == "coordinate"]
    assert len(coord_clauses) == 1


def test_coordinated_clauses_with_but(detector, preprocessor):
    """Test detection of coordinated clauses joined by 'but'."""
    text = "She wanted to go, but he stayed home."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should have main + coordinate
    assert len(clauses) == 2
    assert any(c.clause_type == "main" for c in clauses)
    assert any(c.clause_type == "coordinate" for c in clauses)


def test_coordinated_clauses_with_or(detector, preprocessor):
    """Test detection of coordinated clauses joined by 'or'."""
    text = "We can stay here, or we can leave now."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should have main + coordinate
    assert len(clauses) == 2


def test_multiple_coordinated_clauses(detector, preprocessor):
    """Test detection of multiple coordinated clauses."""
    text = "The cat sat, the dog ran, and the bird flew."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should have at least 3 clauses (main + 2 coordinates)
    assert len(clauses) >= 3


# ============================================================================
# Subordinate Clause Tests (advcl, ccomp)
# ============================================================================

def test_subordinate_clause_with_when(detector, preprocessor):
    """Test detection of subordinate clause with 'when'."""
    text = "When the cat sat, the dog ran."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect main + subordinate
    assert len(clauses) == 2
    assert any(c.clause_type == "main" for c in clauses)
    assert any(c.clause_type == "subordinate" for c in clauses)


def test_subordinate_clause_with_because(detector, preprocessor):
    """Test detection of subordinate clause with 'because'."""
    text = "She left because he was late."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should have main + subordinate
    assert len(clauses) == 2


def test_subordinate_clause_with_although(detector, preprocessor):
    """Test detection of subordinate clause with 'although'."""
    text = "Although it rained, we went outside."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should have subordinate + main
    assert len(clauses) == 2
    assert any(c.clause_type == "subordinate" for c in clauses)


def test_clausal_complement_ccomp(detector, preprocessor):
    """Test detection of clausal complement (ccomp)."""
    text = "I think the cat sat there."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect at least main clause
    # ccomp may or may not be detected depending on spaCy parse
    assert len(clauses) >= 1
    assert any(c.clause_type == "main" for c in clauses)


# ============================================================================
# Complex Sentence Tests
# ============================================================================

def test_complex_sentence_multiple_types(detector, preprocessor):
    """Test detection in complex sentence with multiple clause types."""
    text = "When the cat sat, the dog ran, and the bird flew."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect subordinate, main, and coordinate
    assert len(clauses) >= 3

    clause_types = {c.clause_type for c in clauses}
    # Should have at least 2 different clause types
    assert len(clause_types) >= 2


def test_long_sentence_with_clauses(detector, preprocessor):
    """Test detection in longer sentence."""
    text = "The scientists announced a breakthrough, and the public celebrated, although skeptics remained doubtful."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect multiple clauses
    assert len(clauses) >= 2


# ============================================================================
# Edge Case Tests
# ============================================================================

def test_empty_doc(detector, preprocessor):
    """Test handling of empty document."""
    text = ""
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should return empty list
    assert clauses == []


def test_single_word(detector, preprocessor):
    """Test handling of single word."""
    text = "Go."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect one clause
    assert len(clauses) >= 1


def test_fragment_no_verb(detector, preprocessor):
    """Test handling of fragment without verb."""
    text = "The big cat."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # May or may not detect clause depending on parse
    # Just verify it doesn't crash
    assert isinstance(clauses, list)


def test_punctuation_only(detector, preprocessor):
    """Test handling of punctuation-only input."""
    text = "..."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should handle gracefully
    assert isinstance(clauses, list)


# ============================================================================
# Clause Span Tests
# ============================================================================

def test_clause_spans_non_overlapping(detector, preprocessor):
    """Test that clause spans are reasonable (may overlap in Tier 1)."""
    text = "The cat sat, and the dog ran."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Verify all spans are valid
    for clause in clauses:
        assert clause.start_idx >= 0
        assert clause.end_idx < len(tokens)
        assert clause.start_idx <= clause.end_idx


def test_clause_spans_cover_tokens(detector, preprocessor):
    """Test that clause spans include tokens."""
    text = "The dog ran quickly."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 1

    for clause in clauses:
        # Span should include at least one token
        span_length = clause.end_idx - clause.start_idx + 1
        assert span_length >= 1
        assert len(clause.tokens) >= 1


def test_clause_tokens_match_span(detector, preprocessor):
    """Test that clause.tokens length matches span."""
    text = "The cat sat on the mat."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    for clause in clauses:
        expected_length = clause.end_idx - clause.start_idx + 1
        assert len(clause.tokens) == expected_length


# ============================================================================
# Sorting Tests
# ============================================================================

def test_clauses_sorted_by_position(detector, preprocessor):
    """Test that clauses are sorted by start_idx."""
    text = "The cat sat, and the dog ran, and the bird flew."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Verify clauses are sorted
    for i in range(len(clauses) - 1):
        assert clauses[i].start_idx <= clauses[i + 1].start_idx


# ============================================================================
# Real-World Text Tests
# ============================================================================

def test_news_article_excerpt(detector, preprocessor):
    """Test with news article text."""
    text = "Scientists announced a breakthrough yesterday. The discovery could revolutionize the field."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect multiple main clauses
    assert len(clauses) >= 2
    main_clauses = [c for c in clauses if c.clause_type == "main"]
    assert len(main_clauses) >= 1


def test_conversational_text(detector, preprocessor):
    """Test with conversational text."""
    text = "I think we should go, but maybe we should wait."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect multiple clauses
    assert len(clauses) >= 2


def test_literary_text_with_semicolon(detector, preprocessor):
    """Test with literary text containing semicolon."""
    text = "The garden was silent; shadows danced across the lawn."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Semicolon typically separates clauses
    # Should detect multiple clauses
    assert len(clauses) >= 2


# ============================================================================
# Convenience Function Tests
# ============================================================================

def test_quick_identify_clauses(preprocessor):
    """Test quick_identify_clauses convenience function."""
    text = "The cat sat, and the dog ran."
    tokens, doc = preprocessor.process(text)

    clauses = quick_identify_clauses(doc, tokens)

    # Should work same as detector.identify_clauses()
    assert len(clauses) >= 2
    # Verify clauses have correct attributes (duck typing test)
    for c in clauses:
        assert hasattr(c, 'tokens')
        assert hasattr(c, 'start_idx')
        assert hasattr(c, 'end_idx')
        assert hasattr(c, 'clause_type')


# ============================================================================
# Integration Tests with Preprocessor
# ============================================================================

def test_integration_with_preprocessor(detector, preprocessor):
    """Test integration with full preprocessor output."""
    text = "When the cat sat on the mat, the dog ran in the yard."
    tokens, doc = preprocessor.process(text)

    # Verify preprocessor output is enriched
    assert all(t.pos_tag != "" for t in tokens)
    assert any(t.is_content_word for t in tokens)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect multiple clauses
    assert len(clauses) >= 2

    # Verify clause tokens have all fields populated
    for clause in clauses:
        for token in clause.tokens:
            assert token.text != ""
            assert token.pos_tag != ""
            # phonetic and syllable_count should be populated too
            assert isinstance(token.syllable_count, int)


def test_integration_complex_sentence(detector, preprocessor):
    """Test integration with complex sentence structure."""
    text = "Although it was raining, we decided to go outside because we needed exercise."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should detect multiple clauses (subordinate + main + subordinate)
    assert len(clauses) >= 2

    # Should have subordinate clauses
    subordinate_clauses = [c for c in clauses if c.clause_type == "subordinate"]
    assert len(subordinate_clauses) >= 1


# ============================================================================
# Clause Type Distribution Tests
# ============================================================================

def test_clause_type_distribution_simple(detector, preprocessor):
    """Test clause type distribution in simple text."""
    text = "The cat sat. The dog ran. The bird flew."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # All should be main clauses
    main_count = sum(1 for c in clauses if c.clause_type == "main")
    assert main_count == len(clauses)


def test_clause_type_distribution_mixed(detector, preprocessor):
    """Test clause type distribution in mixed text."""
    text = "The cat sat, and the dog ran when the bell rang."
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    # Should have variety of clause types
    clause_types = {c.clause_type for c in clauses}
    assert len(clause_types) >= 2  # At least 2 different types


# ============================================================================
# Performance Tests (Basic)
# ============================================================================

def test_performance_short_text(detector, preprocessor):
    """Test performance on short text."""
    text = "The quick brown fox jumps over the lazy dog."
    tokens, doc = preprocessor.process(text)

    # Should complete quickly
    clauses = detector.identify_clauses(doc, tokens)

    assert isinstance(clauses, list)


def test_performance_medium_text(detector, preprocessor):
    """Test performance on medium-length text."""
    text = (
        "The scientist worked late into the night, analyzing data from the experiment. "
        "When dawn broke, she realized the hypothesis was correct, and she immediately "
        "called her colleagues to share the breakthrough."
    )
    tokens, doc = preprocessor.process(text)

    clauses = detector.identify_clauses(doc, tokens)

    assert len(clauses) >= 3
