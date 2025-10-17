"""
Tests for ZoneExtractor (Task 3.3)

Tests the zone extraction component that identifies terminal and initial zones
from clause pairs for echo analysis.
"""

import pytest
from specHO.models import Token, Clause, ClausePair
from specHO.clause_identifier.zone_extractor import ZoneExtractor, quick_extract_zones


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def extractor():
    """Create a ZoneExtractor instance."""
    return ZoneExtractor()


@pytest.fixture
def sample_tokens_a():
    """Create sample tokens for clause A: 'The quick brown fox jumped'."""
    return [
        Token("The", "DET", "DH AH0", False, 1),
        Token("quick", "ADJ", "K W IH1 K", True, 1),
        Token("brown", "ADJ", "B R AW1 N", True, 1),
        Token("fox", "NOUN", "F AA1 K S", True, 1),
        Token("jumped", "VERB", "JH AH1 M P T", True, 1),
    ]


@pytest.fixture
def sample_tokens_b():
    """Create sample tokens for clause B: 'over the lazy dog'."""
    return [
        Token("over", "ADP", "OW1 V ER0", False, 2),
        Token("the", "DET", "DH AH0", False, 1),
        Token("lazy", "ADJ", "L EY1 Z IY0", True, 2),
        Token("dog", "NOUN", "D AO1 G", True, 1),
    ]


@pytest.fixture
def clause_a(sample_tokens_a):
    """Create clause A with 4 content words."""
    return Clause(
        tokens=sample_tokens_a,
        start_idx=0,
        end_idx=5,
        clause_type="main",
        head_idx=4
    )


@pytest.fixture
def clause_b(sample_tokens_b):
    """Create clause B with 2 content words."""
    return Clause(
        tokens=sample_tokens_b,
        start_idx=5,
        end_idx=9,
        clause_type="subordinate",
        head_idx=5
    )


@pytest.fixture
def clause_pair(clause_a, clause_b):
    """Create a clause pair."""
    return ClausePair(
        clause_a=clause_a,
        clause_b=clause_b,
        zone_a_tokens=[],
        zone_b_tokens=[],
        pair_type="punctuation"
    )


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_initialization(extractor):
    """Test ZoneExtractor initializes correctly."""
    assert extractor is not None
    assert isinstance(extractor, ZoneExtractor)


# ============================================================================
# GET_TERMINAL_CONTENT_WORDS TESTS
# ============================================================================

def test_get_terminal_content_words_basic(extractor, clause_a):
    """Test extracting last 3 content words from clause with 4 content words."""
    result = extractor.get_terminal_content_words(clause_a, n=3)

    assert len(result) == 3
    assert all(isinstance(t, Token) for t in result)
    assert [t.text for t in result] == ["brown", "fox", "jumped"]
    assert all(t.is_content_word for t in result)


def test_get_terminal_content_words_fewer_than_n(extractor, clause_b):
    """Test extracting last 3 content words from clause with only 2 content words."""
    result = extractor.get_terminal_content_words(clause_b, n=3)

    assert len(result) == 2  # Only 2 content words available
    assert [t.text for t in result] == ["lazy", "dog"]


def test_get_terminal_content_words_exactly_n(extractor):
    """Test extracting last 3 content words from clause with exactly 3 content words."""
    tokens = [
        Token("The", "DET", "", False, 1),
        Token("cat", "NOUN", "", True, 1),
        Token("sat", "VERB", "", True, 1),
        Token("very", "ADV", "", False, 1),
        Token("quietly", "ADV", "", False, 1),
    ]
    clause = Clause(tokens, 0, 5, "main", 1)

    result = extractor.get_terminal_content_words(clause, n=3)

    assert len(result) == 2  # Only 2 content words total
    assert [t.text for t in result] == ["cat", "sat"]


def test_get_terminal_content_words_empty_clause(extractor):
    """Test extracting from empty clause."""
    clause = Clause(tokens=[], start_idx=0, end_idx=0, clause_type="main", head_idx=0)
    result = extractor.get_terminal_content_words(clause, n=3)

    assert result == []


def test_get_terminal_content_words_no_content_words(extractor):
    """Test extracting from clause with no content words."""
    tokens = [
        Token("the", "DET", "", False, 1),
        Token("of", "ADP", "", False, 1),
        Token("a", "DET", "", False, 1),
    ]
    clause = Clause(tokens, 0, 3, "main", 0)

    result = extractor.get_terminal_content_words(clause, n=3)

    assert result == []


def test_get_terminal_content_words_custom_n(extractor, clause_a):
    """Test extracting with custom window size."""
    result = extractor.get_terminal_content_words(clause_a, n=2)

    assert len(result) == 2
    assert [t.text for t in result] == ["fox", "jumped"]


def test_get_terminal_content_words_n_equals_1(extractor, clause_a):
    """Test extracting single terminal word."""
    result = extractor.get_terminal_content_words(clause_a, n=1)

    assert len(result) == 1
    assert result[0].text == "jumped"


def test_get_terminal_content_words_preserves_token_fields(extractor, clause_a):
    """Test that extracted tokens preserve all fields."""
    result = extractor.get_terminal_content_words(clause_a, n=1)

    token = result[0]
    assert token.text == "jumped"
    assert token.pos_tag == "VERB"
    assert token.phonetic == "JH AH1 M P T"
    assert token.is_content_word is True
    assert token.syllable_count == 1


# ============================================================================
# GET_INITIAL_CONTENT_WORDS TESTS
# ============================================================================

def test_get_initial_content_words_basic(extractor, clause_a):
    """Test extracting first 3 content words from clause with 4 content words."""
    result = extractor.get_initial_content_words(clause_a, n=3)

    assert len(result) == 3
    assert all(isinstance(t, Token) for t in result)
    assert [t.text for t in result] == ["quick", "brown", "fox"]
    assert all(t.is_content_word for t in result)


def test_get_initial_content_words_fewer_than_n(extractor, clause_b):
    """Test extracting first 3 content words from clause with only 2 content words."""
    result = extractor.get_initial_content_words(clause_b, n=3)

    assert len(result) == 2  # Only 2 content words available
    assert [t.text for t in result] == ["lazy", "dog"]


def test_get_initial_content_words_exactly_n(extractor):
    """Test extracting first 3 content words from clause with exactly 3 content words."""
    tokens = [
        Token("The", "DET", "", False, 1),
        Token("cat", "NOUN", "", True, 1),
        Token("sat", "VERB", "", True, 1),
        Token("on", "ADP", "", False, 1),
        Token("mat", "NOUN", "", True, 1),
    ]
    clause = Clause(tokens, 0, 5, "main", 1)

    result = extractor.get_initial_content_words(clause, n=3)

    assert len(result) == 3
    assert [t.text for t in result] == ["cat", "sat", "mat"]


def test_get_initial_content_words_empty_clause(extractor):
    """Test extracting from empty clause."""
    clause = Clause(tokens=[], start_idx=0, end_idx=0, clause_type="main", head_idx=0)
    result = extractor.get_initial_content_words(clause, n=3)

    assert result == []


def test_get_initial_content_words_no_content_words(extractor):
    """Test extracting from clause with no content words."""
    tokens = [
        Token("the", "DET", "", False, 1),
        Token("of", "ADP", "", False, 1),
        Token("a", "DET", "", False, 1),
    ]
    clause = Clause(tokens, 0, 3, "main", 0)

    result = extractor.get_initial_content_words(clause, n=3)

    assert result == []


def test_get_initial_content_words_custom_n(extractor, clause_a):
    """Test extracting with custom window size."""
    result = extractor.get_initial_content_words(clause_a, n=2)

    assert len(result) == 2
    assert [t.text for t in result] == ["quick", "brown"]


def test_get_initial_content_words_n_equals_1(extractor, clause_a):
    """Test extracting single initial word."""
    result = extractor.get_initial_content_words(clause_a, n=1)

    assert len(result) == 1
    assert result[0].text == "quick"


def test_get_initial_content_words_preserves_token_fields(extractor, clause_b):
    """Test that extracted tokens preserve all fields."""
    result = extractor.get_initial_content_words(clause_b, n=1)

    token = result[0]
    assert token.text == "lazy"
    assert token.pos_tag == "ADJ"
    assert token.phonetic == "L EY1 Z IY0"
    assert token.is_content_word is True
    assert token.syllable_count == 2


# ============================================================================
# EXTRACT_ZONES TESTS
# ============================================================================

def test_extract_zones_basic(extractor, clause_pair):
    """Test extracting zones from a clause pair."""
    zone_a, zone_b = extractor.extract_zones(clause_pair)

    assert isinstance(zone_a, list)
    assert isinstance(zone_b, list)
    assert len(zone_a) == 3  # Last 3 content words from clause_a
    assert len(zone_b) == 2  # All 2 content words from clause_b (< 3 available)

    assert [t.text for t in zone_a] == ["brown", "fox", "jumped"]
    assert [t.text for t in zone_b] == ["lazy", "dog"]


def test_extract_zones_returns_tuple(extractor, clause_pair):
    """Test that extract_zones returns a tuple."""
    result = extractor.extract_zones(clause_pair)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_extract_zones_both_clauses_short(extractor):
    """Test extracting zones when both clauses have < 3 content words."""
    tokens_a = [
        Token("I", "PRON", "", True, 1),
        Token("ran", "VERB", "", True, 1),
    ]
    tokens_b = [
        Token("you", "PRON", "", True, 1),
        Token("sat", "VERB", "", True, 1),
    ]

    clause_a = Clause(tokens_a, 0, 2, "main", 1)
    clause_b = Clause(tokens_b, 2, 4, "main", 3)
    pair = ClausePair(clause_a, clause_b, [], [], "conjunction")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert len(zone_a) == 2
    assert len(zone_b) == 2
    assert [t.text for t in zone_a] == ["I", "ran"]
    assert [t.text for t in zone_b] == ["you", "sat"]


def test_extract_zones_empty_clauses(extractor):
    """Test extracting zones from empty clauses."""
    clause_a = Clause([], 0, 0, "main", 0)
    clause_b = Clause([], 0, 0, "main", 0)
    pair = ClausePair(clause_a, clause_b, [], [], "punctuation")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert zone_a == []
    assert zone_b == []


def test_extract_zones_one_empty_clause(extractor, clause_a):
    """Test extracting zones when one clause is empty."""
    clause_b = Clause([], 5, 5, "main", 5)
    pair = ClausePair(clause_a, clause_b, [], [], "punctuation")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert len(zone_a) == 3  # clause_a has content words
    assert zone_b == []  # clause_b is empty


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_extract_zones_with_mixed_content_words(extractor):
    """Test extraction from clauses with mixed content/function words."""
    tokens_a = [
        Token("The", "DET", "", False, 1),
        Token("very", "ADV", "", False, 1),
        Token("big", "ADJ", "", True, 1),
        Token("red", "ADJ", "", True, 1),
        Token("car", "NOUN", "", True, 1),
        Token("was", "AUX", "", False, 1),
        Token("driving", "VERB", "", True, 1),
    ]
    tokens_b = [
        Token("and", "CCONJ", "", False, 1),
        Token("the", "DET", "", False, 1),
        Token("small", "ADJ", "", True, 1),
        Token("blue", "ADJ", "", True, 1),
        Token("truck", "NOUN", "", True, 1),
        Token("was", "AUX", "", False, 1),
        Token("stopping", "VERB", "", True, 1),
    ]

    clause_a = Clause(tokens_a, 0, 7, "main", 6)
    clause_b = Clause(tokens_b, 7, 14, "coordinate", 13)
    pair = ClausePair(clause_a, clause_b, [], [], "conjunction")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert len(zone_a) == 3
    assert len(zone_b) == 3
    assert [t.text for t in zone_a] == ["red", "car", "driving"]
    assert [t.text for t in zone_b] == ["small", "blue", "truck"]


def test_extract_zones_preserves_original_clause_pair(extractor, clause_pair):
    """Test that extract_zones doesn't modify the original ClausePair."""
    original_zone_a = clause_pair.zone_a_tokens.copy()
    original_zone_b = clause_pair.zone_b_tokens.copy()

    zone_a, zone_b = extractor.extract_zones(clause_pair)

    # Original should be unchanged
    assert clause_pair.zone_a_tokens == original_zone_a
    assert clause_pair.zone_b_tokens == original_zone_b


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_quick_extract_zones(clause_pair):
    """Test quick_extract_zones convenience function."""
    zone_a, zone_b = quick_extract_zones(clause_pair)

    assert isinstance(zone_a, list)
    assert isinstance(zone_b, list)
    assert len(zone_a) == 3
    assert len(zone_b) == 2


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_extract_zones_single_content_word_each(extractor):
    """Test extraction when each clause has exactly 1 content word."""
    tokens_a = [Token("The", "DET", "", False, 1), Token("cat", "NOUN", "", True, 1)]
    tokens_b = [Token("a", "DET", "", False, 1), Token("dog", "NOUN", "", True, 1)]

    clause_a = Clause(tokens_a, 0, 2, "main", 1)
    clause_b = Clause(tokens_b, 2, 4, "main", 3)
    pair = ClausePair(clause_a, clause_b, [], [], "punctuation")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert len(zone_a) == 1
    assert len(zone_b) == 1
    assert zone_a[0].text == "cat"
    assert zone_b[0].text == "dog"


def test_extract_zones_many_content_words(extractor):
    """Test extraction from clauses with many content words."""
    tokens_a = [Token(f"word{i}", "NOUN", "", True, 1) for i in range(10)]
    tokens_b = [Token(f"word{i}", "VERB", "", True, 1) for i in range(10, 20)]

    clause_a = Clause(tokens_a, 0, 10, "main", 9)
    clause_b = Clause(tokens_b, 10, 20, "main", 19)
    pair = ClausePair(clause_a, clause_b, [], [], "conjunction")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert len(zone_a) == 3  # Last 3 of 10
    assert len(zone_b) == 3  # First 3 of 10
    assert [t.text for t in zone_a] == ["word7", "word8", "word9"]
    assert [t.text for t in zone_b] == ["word10", "word11", "word12"]


def test_extract_zones_all_function_words(extractor):
    """Test extraction when clauses contain only function words."""
    tokens_a = [
        Token("the", "DET", "", False, 1),
        Token("of", "ADP", "", False, 1),
        Token("and", "CCONJ", "", False, 1),
    ]
    tokens_b = [
        Token("a", "DET", "", False, 1),
        Token("in", "ADP", "", False, 1),
        Token("but", "CCONJ", "", False, 1),
    ]

    clause_a = Clause(tokens_a, 0, 3, "main", 0)
    clause_b = Clause(tokens_b, 3, 6, "main", 3)
    pair = ClausePair(clause_a, clause_b, [], [], "conjunction")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert zone_a == []
    assert zone_b == []


# ============================================================================
# REAL-WORLD SCENARIO TESTS
# ============================================================================

def test_extract_zones_literary_text(extractor):
    """Test extraction from literary text-like clauses."""
    # "The ancient castle stood majestically; its towers reached skyward."
    tokens_a = [
        Token("The", "DET", "", False, 1),
        Token("ancient", "ADJ", "", True, 2),
        Token("castle", "NOUN", "", True, 2),
        Token("stood", "VERB", "", True, 1),
        Token("majestically", "ADV", "", False, 5),
    ]
    tokens_b = [
        Token("its", "PRON", "", False, 1),
        Token("towers", "NOUN", "", True, 2),
        Token("reached", "VERB", "", True, 1),
        Token("skyward", "ADV", "", True, 2),  # Some adverbs can be content words
    ]

    clause_a = Clause(tokens_a, 0, 5, "main", 3)
    clause_b = Clause(tokens_b, 5, 9, "main", 7)
    pair = ClausePair(clause_a, clause_b, [], [], "punctuation")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert len(zone_a) == 3
    assert len(zone_b) == 3
    assert [t.text for t in zone_a] == ["ancient", "castle", "stood"]
    assert [t.text for t in zone_b] == ["towers", "reached", "skyward"]


def test_extract_zones_conversational_text(extractor):
    """Test extraction from conversational text-like clauses."""
    # "I totally agree; you make excellent points."
    tokens_a = [
        Token("I", "PRON", "", True, 1),
        Token("totally", "ADV", "", False, 3),
        Token("agree", "VERB", "", True, 2),
    ]
    tokens_b = [
        Token("you", "PRON", "", True, 1),
        Token("make", "VERB", "", True, 1),
        Token("excellent", "ADJ", "", True, 3),
        Token("points", "NOUN", "", True, 1),
    ]

    clause_a = Clause(tokens_a, 0, 3, "main", 2)
    clause_b = Clause(tokens_b, 3, 7, "main", 4)
    pair = ClausePair(clause_a, clause_b, [], [], "punctuation")

    zone_a, zone_b = extractor.extract_zones(pair)

    assert len(zone_a) == 2  # Only 2 content words
    assert len(zone_b) == 3
    assert [t.text for t in zone_a] == ["I", "agree"]
    assert [t.text for t in zone_b] == ["you", "make", "excellent"]
