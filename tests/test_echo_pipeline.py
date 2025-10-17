"""Tests for Echo Analysis Engine pipeline orchestration.

This test module validates that the EchoAnalysisEngine correctly orchestrates
all three analyzers (phonetic, structural, semantic) and produces consolidated
EchoScore objects.

Tier: 1 (MVP)
Task: 8.3 (partial - pipeline integration tests)
"""

import pytest
from unittest.mock import Mock, MagicMock
from specHO.models import Token, Clause, ClausePair, EchoScore
from specHO.echo_engine.pipeline import EchoAnalysisEngine
from specHO.echo_engine.phonetic_analyzer import PhoneticEchoAnalyzer
from specHO.echo_engine.structural_analyzer import StructuralEchoAnalyzer
from specHO.echo_engine.semantic_analyzer import SemanticEchoAnalyzer


@pytest.fixture
def sample_tokens_a():
    """Sample tokens for zone A."""
    return [
        Token(text="quick", pos_tag="ADJ", phonetic="K W IH1 K", is_content_word=True, syllable_count=1),
        Token(text="brown", pos_tag="ADJ", phonetic="B R AW1 N", is_content_word=True, syllable_count=1),
        Token(text="fox", pos_tag="NOUN", phonetic="F AA1 K S", is_content_word=True, syllable_count=1),
    ]


@pytest.fixture
def sample_tokens_b():
    """Sample tokens for zone B."""
    return [
        Token(text="swift", pos_tag="ADJ", phonetic="S W IH1 F T", is_content_word=True, syllable_count=1),
        Token(text="red", pos_tag="ADJ", phonetic="R EH1 D", is_content_word=True, syllable_count=1),
        Token(text="dog", pos_tag="NOUN", phonetic="D AO1 G", is_content_word=True, syllable_count=1),
    ]


@pytest.fixture
def sample_clause_pair(sample_tokens_a, sample_tokens_b):
    """Sample clause pair with zones populated."""
    clause_a = Clause(
        tokens=sample_tokens_a,
        start_idx=0,
        end_idx=3,
        clause_type="main",
        head_idx=0
    )

    clause_b = Clause(
        tokens=sample_tokens_b,
        start_idx=3,
        end_idx=6,
        clause_type="main",
        head_idx=3
    )

    return ClausePair(
        clause_a=clause_a,
        clause_b=clause_b,
        zone_a_tokens=sample_tokens_a,
        zone_b_tokens=sample_tokens_b,
        pair_type="conjunction"
    )


@pytest.fixture
def mock_analyzers():
    """Mock analyzers with predictable outputs."""
    phonetic = Mock(spec=PhoneticEchoAnalyzer)
    phonetic.analyze = Mock(return_value=0.75)

    structural = Mock(spec=StructuralEchoAnalyzer)
    structural.analyze = Mock(return_value=0.85)

    semantic = Mock(spec=SemanticEchoAnalyzer)
    semantic.analyze = Mock(return_value=0.65)

    return {
        'phonetic': phonetic,
        'structural': structural,
        'semantic': semantic
    }


# ============================================================================
# Initialization Tests
# ============================================================================

class TestEchoAnalysisEngineInit:
    """Test EchoAnalysisEngine initialization."""

    def test_default_initialization(self):
        """Test that engine initializes with default analyzers."""
        engine = EchoAnalysisEngine()

        assert isinstance(engine.phonetic_analyzer, PhoneticEchoAnalyzer)
        assert isinstance(engine.structural_analyzer, StructuralEchoAnalyzer)
        assert isinstance(engine.semantic_analyzer, SemanticEchoAnalyzer)

    def test_custom_analyzers(self, mock_analyzers):
        """Test that engine accepts custom analyzer instances."""
        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_analyzers['phonetic'],
            structural_analyzer=mock_analyzers['structural'],
            semantic_analyzer=mock_analyzers['semantic']
        )

        assert engine.phonetic_analyzer == mock_analyzers['phonetic']
        assert engine.structural_analyzer == mock_analyzers['structural']
        assert engine.semantic_analyzer == mock_analyzers['semantic']

    def test_partial_custom_analyzers(self, mock_analyzers):
        """Test that engine accepts mix of custom and default analyzers."""
        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_analyzers['phonetic']
        )

        assert engine.phonetic_analyzer == mock_analyzers['phonetic']
        assert isinstance(engine.structural_analyzer, StructuralEchoAnalyzer)
        assert isinstance(engine.semantic_analyzer, SemanticEchoAnalyzer)


# ============================================================================
# Orchestration Tests
# ============================================================================

class TestEchoAnalysisOrchestration:
    """Test that engine correctly orchestrates all three analyzers."""

    def test_analyze_pair_calls_all_analyzers(self, sample_clause_pair, mock_analyzers):
        """Test that analyze_pair calls all three analyzers."""
        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_analyzers['phonetic'],
            structural_analyzer=mock_analyzers['structural'],
            semantic_analyzer=mock_analyzers['semantic']
        )

        result = engine.analyze_pair(sample_clause_pair)

        # Verify all analyzers were called with correct arguments
        mock_analyzers['phonetic'].analyze.assert_called_once_with(
            sample_clause_pair.zone_a_tokens,
            sample_clause_pair.zone_b_tokens
        )
        mock_analyzers['structural'].analyze.assert_called_once_with(
            sample_clause_pair.zone_a_tokens,
            sample_clause_pair.zone_b_tokens
        )
        mock_analyzers['semantic'].analyze.assert_called_once_with(
            sample_clause_pair.zone_a_tokens,
            sample_clause_pair.zone_b_tokens
        )

    def test_analyze_pair_returns_echo_score(self, sample_clause_pair, mock_analyzers):
        """Test that analyze_pair returns properly structured EchoScore."""
        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_analyzers['phonetic'],
            structural_analyzer=mock_analyzers['structural'],
            semantic_analyzer=mock_analyzers['semantic']
        )

        result = engine.analyze_pair(sample_clause_pair)

        assert isinstance(result, EchoScore)
        assert result.phonetic_score == 0.75
        assert result.structural_score == 0.85
        assert result.semantic_score == 0.65
        assert result.combined_score == 0.0  # Not calculated by engine

    def test_analyze_pair_preserves_analyzer_scores(self, sample_clause_pair, mock_analyzers):
        """Test that scores from analyzers are correctly preserved."""
        # Set different scores
        mock_analyzers['phonetic'].analyze.return_value = 0.123
        mock_analyzers['structural'].analyze.return_value = 0.456
        mock_analyzers['semantic'].analyze.return_value = 0.789

        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_analyzers['phonetic'],
            structural_analyzer=mock_analyzers['structural'],
            semantic_analyzer=mock_analyzers['semantic']
        )

        result = engine.analyze_pair(sample_clause_pair)

        assert result.phonetic_score == 0.123
        assert result.structural_score == 0.456
        assert result.semantic_score == 0.789


# ============================================================================
# Integration Tests with Real Analyzers
# ============================================================================

class TestEchoAnalysisIntegration:
    """Test engine with real analyzer instances."""

    def test_real_analyzers_with_sample_data(self, sample_clause_pair):
        """Test that real analyzers work together through engine."""
        engine = EchoAnalysisEngine()

        result = engine.analyze_pair(sample_clause_pair)

        # Verify result structure
        assert isinstance(result, EchoScore)
        assert 0.0 <= result.phonetic_score <= 1.0
        assert 0.0 <= result.structural_score <= 1.0
        assert 0.0 <= result.semantic_score <= 1.0
        assert result.combined_score == 0.0

    def test_real_analyzers_with_identical_zones(self):
        """Test engine with identical zones (should produce high scores)."""
        tokens = [
            Token(text="the", pos_tag="DET", phonetic="DH AH0", is_content_word=False, syllable_count=1),
            Token(text="quick", pos_tag="ADJ", phonetic="K W IH1 K", is_content_word=True, syllable_count=1),
            Token(text="fox", pos_tag="NOUN", phonetic="F AA1 K S", is_content_word=True, syllable_count=1),
        ]

        clause_a = Clause(tokens=tokens, start_idx=0, end_idx=3, clause_type="main", head_idx=0)
        clause_b = Clause(tokens=tokens, start_idx=3, end_idx=6, clause_type="main", head_idx=3)

        pair = ClausePair(
            clause_a=clause_a,
            clause_b=clause_b,
            zone_a_tokens=tokens,
            zone_b_tokens=tokens,
            pair_type="conjunction"
        )

        engine = EchoAnalysisEngine()
        result = engine.analyze_pair(pair)

        # Identical zones should have high scores
        assert result.phonetic_score >= 0.9
        assert result.structural_score >= 0.9
        # Semantic may be lower if no embeddings available (fallback 0.5)

    def test_real_analyzers_with_empty_zones(self):
        """Test engine with empty zones."""
        clause_a = Clause(tokens=[], start_idx=0, end_idx=0, clause_type="main", head_idx=0)
        clause_b = Clause(tokens=[], start_idx=0, end_idx=0, clause_type="main", head_idx=0)

        pair = ClausePair(
            clause_a=clause_a,
            clause_b=clause_b,
            zone_a_tokens=[],
            zone_b_tokens=[],
            pair_type="conjunction"
        )

        engine = EchoAnalysisEngine()
        result = engine.analyze_pair(pair)

        # Empty zones should produce 0.0 scores (except semantic fallback)
        assert result.phonetic_score == 0.0
        assert result.structural_score == 0.0
        assert result.semantic_score == 0.0  # Empty zones


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEchoAnalysisEdgeCases:
    """Test engine behavior with edge cases."""

    def test_analyzer_returning_zero(self, sample_clause_pair):
        """Test that engine handles zero scores correctly."""
        mock_phonetic = Mock(spec=PhoneticEchoAnalyzer)
        mock_phonetic.analyze = Mock(return_value=0.0)

        mock_structural = Mock(spec=StructuralEchoAnalyzer)
        mock_structural.analyze = Mock(return_value=0.0)

        mock_semantic = Mock(spec=SemanticEchoAnalyzer)
        mock_semantic.analyze = Mock(return_value=0.0)

        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_phonetic,
            structural_analyzer=mock_structural,
            semantic_analyzer=mock_semantic
        )

        result = engine.analyze_pair(sample_clause_pair)

        assert result.phonetic_score == 0.0
        assert result.structural_score == 0.0
        assert result.semantic_score == 0.0

    def test_analyzer_returning_one(self, sample_clause_pair):
        """Test that engine handles maximum scores correctly."""
        mock_phonetic = Mock(spec=PhoneticEchoAnalyzer)
        mock_phonetic.analyze = Mock(return_value=1.0)

        mock_structural = Mock(spec=StructuralEchoAnalyzer)
        mock_structural.analyze = Mock(return_value=1.0)

        mock_semantic = Mock(spec=SemanticEchoAnalyzer)
        mock_semantic.analyze = Mock(return_value=1.0)

        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_phonetic,
            structural_analyzer=mock_structural,
            semantic_analyzer=mock_semantic
        )

        result = engine.analyze_pair(sample_clause_pair)

        assert result.phonetic_score == 1.0
        assert result.structural_score == 1.0
        assert result.semantic_score == 1.0

    def test_mixed_score_values(self, sample_clause_pair):
        """Test engine with varied score values."""
        mock_phonetic = Mock(spec=PhoneticEchoAnalyzer)
        mock_phonetic.analyze = Mock(return_value=0.2)

        mock_structural = Mock(spec=StructuralEchoAnalyzer)
        mock_structural.analyze = Mock(return_value=0.8)

        mock_semantic = Mock(spec=SemanticEchoAnalyzer)
        mock_semantic.analyze = Mock(return_value=0.5)

        engine = EchoAnalysisEngine(
            phonetic_analyzer=mock_phonetic,
            structural_analyzer=mock_structural,
            semantic_analyzer=mock_semantic
        )

        result = engine.analyze_pair(sample_clause_pair)

        assert result.phonetic_score == 0.2
        assert result.structural_score == 0.8
        assert result.semantic_score == 0.5


# ============================================================================
# Score Range Validation Tests
# ============================================================================

class TestEchoScoreRanges:
    """Verify all scores are in valid [0,1] range."""

    def test_all_scores_in_valid_range(self, sample_clause_pair):
        """Test that all scores are within [0,1] range."""
        engine = EchoAnalysisEngine()
        result = engine.analyze_pair(sample_clause_pair)

        # Validate ranges
        assert 0.0 <= result.phonetic_score <= 1.0, "Phonetic score out of range"
        assert 0.0 <= result.structural_score <= 1.0, "Structural score out of range"
        assert 0.0 <= result.semantic_score <= 1.0, "Semantic score out of range"

    def test_combined_score_is_zero(self, sample_clause_pair):
        """Test that combined_score is 0.0 (calculated by scoring module)."""
        engine = EchoAnalysisEngine()
        result = engine.analyze_pair(sample_clause_pair)

        assert result.combined_score == 0.0, "Combined score should be 0.0"
