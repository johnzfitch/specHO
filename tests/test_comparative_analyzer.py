"""
Tests for ComparativeClusterAnalyzer.

Tests the comparative term clustering detection feature from toolkit analysis.
"""

import pytest
from specHO.echo_engine.comparative_analyzer import (
    ComparativeClusterAnalyzer,
    quick_comparative_analysis
)
from specHO.models import Token


@pytest.fixture
def analyzer():
    """Create a ComparativeClusterAnalyzer instance."""
    return ComparativeClusterAnalyzer()


@pytest.fixture
def sample_tokens():
    """Create sample tokens for testing."""
    return [
        Token('less', 'ADJ', 'L EH S', True, 1),
        Token('effort', 'NOUN', 'EH F ER T', True, 2),
        Token('more', 'ADJ', 'M AO R', True, 1),
        Token('generic', 'ADJ', 'JH AH N EH R IH K', True, 3),
        Token('shorter', 'ADJ', 'SH AO R T ER', True, 2),
        Token('text', 'NOUN', 'T EH K S T', True, 1),
    ]


class TestComparativeClusterAnalyzer:
    """Test suite for ComparativeClusterAnalyzer."""

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None
        assert isinstance(analyzer.COMPARATIVE_TERMS, set)
        assert 'less' in analyzer.COMPARATIVE_TERMS
        assert 'more' in analyzer.COMPARATIVE_TERMS
        assert 'shorter' in analyzer.COMPARATIVE_TERMS

    def test_no_comparatives(self, analyzer):
        """Test with no comparative terms."""
        tokens = [
            Token('the', 'DET', 'DH AH', False, 1),
            Token('cat', 'NOUN', 'K AE T', True, 1),
            Token('sat', 'VERB', 'S AE T', True, 1),
        ]
        score = analyzer.analyze(tokens, [])
        assert score == 0.0

    def test_one_comparative(self, analyzer):
        """Test with one comparative term."""
        tokens = [
            Token('less', 'ADJ', 'L EH S', True, 1),
            Token('effort', 'NOUN', 'EH F ER T', True, 2),
        ]
        score = analyzer.analyze(tokens, [])
        assert 0.1 <= score <= 0.2

    def test_two_comparatives(self, analyzer):
        """Test with two comparative terms."""
        zone_a = [Token('less', 'ADJ', 'L EH S', True, 1)]
        zone_b = [Token('more', 'ADJ', 'M AO R', True, 1)]
        score = analyzer.analyze(zone_a, zone_b)
        assert score == 0.3

    def test_three_comparatives(self, analyzer, sample_tokens):
        """Test with three comparative terms - moderate suspicion."""
        zone_a = [sample_tokens[0], sample_tokens[1]]  # less, effort
        zone_b = [sample_tokens[2], sample_tokens[4]]  # more, shorter
        score = analyzer.analyze(zone_a, zone_b)
        # Should have 3 comparatives: less, more, shorter
        assert score == 0.5

    def test_four_comparatives(self, analyzer):
        """Test with four comparative terms - high suspicion."""
        zone_a = [
            Token('less', 'ADJ', 'L EH S', True, 1),
            Token('more', 'ADJ', 'M AO R', True, 1),
        ]
        zone_b = [
            Token('shorter', 'ADJ', 'SH AO R T ER', True, 2),
            Token('better', 'ADJ', 'B EH T ER', True, 2),
        ]
        score = analyzer.analyze(zone_a, zone_b)
        assert score == 0.8

    def test_five_comparatives_extreme(self, analyzer):
        """Test with five comparative terms - extreme suspicion (AI tell)."""
        zone_a = [
            Token('less', 'ADJ', 'L EH S', True, 1),
            Token('more', 'ADJ', 'M AO R', True, 1),
            Token('shorter', 'ADJ', 'SH AO R T ER', True, 2),
        ]
        zone_b = [
            Token('better', 'ADJ', 'B EH T ER', True, 2),
            Token('fewer', 'ADJ', 'F Y UW ER', True, 2),
        ]
        score = analyzer.analyze(zone_a, zone_b)
        assert score >= 0.9

    def test_case_insensitive(self, analyzer):
        """Test that comparison is case-insensitive."""
        tokens = [
            Token('LESS', 'ADJ', 'L EH S', True, 1),
            Token('More', 'ADJ', 'M AO R', True, 1),
        ]
        score = analyzer.analyze(tokens, [])
        assert score == 0.3

    def test_empty_zones(self, analyzer):
        """Test with empty zones."""
        score = analyzer.analyze([], [])
        assert score == 0.0

    def test_one_empty_zone(self, analyzer):
        """Test with one empty zone."""
        tokens = [Token('less', 'ADJ', 'L EH S', True, 1)]
        score = analyzer.analyze(tokens, [])
        assert 0.1 <= score <= 0.2

    def test_get_comparatives_in_zones(self, analyzer, sample_tokens):
        """Test getting list of comparative terms found."""
        zone_a = [sample_tokens[0], sample_tokens[1]]  # less, effort
        zone_b = [sample_tokens[2]]  # more
        
        comparatives = analyzer.get_comparatives_in_zones(zone_a, zone_b)
        assert 'less' in comparatives
        assert 'more' in comparatives
        assert len(comparatives) == 2

    def test_superlatives(self, analyzer):
        """Test that superlatives are also detected."""
        tokens = [
            Token('best', 'ADJ', 'B EH S T', True, 1),
            Token('worst', 'ADJ', 'W ER S T', True, 1),
            Token('least', 'ADJ', 'L IY S T', True, 1),
        ]
        score = analyzer.analyze(tokens, [])
        assert score == 0.5  # 3 comparatives

    def test_diverse_comparatives(self, analyzer):
        """Test with diverse comparative forms."""
        tokens = [
            Token('higher', 'ADJ', 'HH AY ER', True, 2),
            Token('lower', 'ADJ', 'L OW ER', True, 2),
            Token('faster', 'ADJ', 'F AE S T ER', True, 2),
            Token('slower', 'ADJ', 'S L OW ER', True, 2),
        ]
        score = analyzer.analyze(tokens, [])
        assert score == 0.8  # 4 comparatives


class TestQuickComparativeAnalysis:
    """Test convenience function."""

    def test_quick_analysis(self):
        """Test quick_comparative_analysis convenience function."""
        zone_a = [Token('less', 'ADJ', 'L EH S', True, 1)]
        zone_b = [Token('more', 'ADJ', 'M AO R', True, 1)]
        
        score = quick_comparative_analysis(zone_a, zone_b)
        assert score == 0.3


class TestComparativeTermsCoverage:
    """Test that comprehensive set of comparative terms is covered."""

    def test_basic_comparatives_present(self, analyzer):
        """Test basic comparative terms are in the set."""
        basic_terms = ['less', 'more', 'fewer', 'greater', 'smaller', 
                      'larger', 'shorter', 'longer', 'better', 'worse']
        for term in basic_terms:
            assert term in analyzer.COMPARATIVE_TERMS

    def test_superlatives_present(self, analyzer):
        """Test superlative terms are in the set."""
        superlatives = ['least', 'most', 'best', 'worst', 'smallest', 
                       'largest', 'shortest', 'longest']
        for term in superlatives:
            assert term in analyzer.COMPARATIVE_TERMS

    def test_adjective_comparatives_present(self, analyzer):
        """Test adjective comparative forms are in the set."""
        adjectives = ['simpler', 'clearer', 'broader', 'narrower', 
                     'richer', 'poorer', 'newer', 'older']
        for term in adjectives:
            assert term in analyzer.COMPARATIVE_TERMS
