"""Tests for core data models (models.py).

Tests the basic functionality of the five foundational dataclasses:
Token, Clause, ClausePair, EchoScore, and DocumentAnalysis.

Tier: 1 (MVP)
Coverage: Basic instantiation and field access
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from models import Token, Clause, ClausePair, EchoScore, DocumentAnalysis


class TestToken:
    """Tests for Token dataclass."""

    def test_token_creation(self):
        """Test basic Token instantiation."""
        token = Token(
            text="hello",
            pos_tag="NOUN",
            phonetic="HH EH L OW",
            is_content_word=True,
            syllable_count=2
        )
        assert token.text == "hello"
        assert token.pos_tag == "NOUN"
        assert token.phonetic == "HH EH L OW"
        assert token.is_content_word is True
        assert token.syllable_count == 2

    def test_token_content_word_false(self):
        """Test Token with function word (is_content_word=False)."""
        token = Token("the", "DET", "DH AH", False, 1)
        assert token.is_content_word is False

    def test_token_fields_immutable_types(self):
        """Test that Token fields have correct types."""
        token = Token("test", "VERB", "T EH S T", True, 1)
        assert isinstance(token.text, str)
        assert isinstance(token.pos_tag, str)
        assert isinstance(token.phonetic, str)
        assert isinstance(token.is_content_word, bool)
        assert isinstance(token.syllable_count, int)


class TestClause:
    """Tests for Clause dataclass."""

    def test_clause_creation(self):
        """Test basic Clause instantiation."""
        tokens = [
            Token("the", "DET", "DH AH", False, 1),
            Token("cat", "NOUN", "K AE T", True, 1),
        ]
        clause = Clause(tokens=tokens, start_idx=0, end_idx=2, clause_type="main", head_idx=1)
        assert len(clause.tokens) == 2
        assert clause.start_idx == 0
        assert clause.end_idx == 2
        assert clause.clause_type == "main"
        assert clause.head_idx == 1

    def test_clause_subordinate_type(self):
        """Test Clause with subordinate type."""
        tokens = [Token("although", "SCONJ", "AO L DH OW", False, 2)]
        clause = Clause(tokens, 5, 10, "subordinate", 5)
        assert clause.clause_type == "subordinate"
        assert clause.head_idx == 5

    def test_clause_empty_tokens(self):
        """Test Clause with empty token list."""
        clause = Clause([], 0, 0, "fragment", 0)
        assert len(clause.tokens) == 0


class TestClausePair:
    """Tests for ClausePair dataclass."""

    def test_clause_pair_creation(self):
        """Test basic ClausePair instantiation."""
        token1 = Token("test", "NOUN", "T EH S T", True, 1)
        token2 = Token("word", "NOUN", "W ER D", True, 1)

        clause_a = Clause([token1], 0, 1, "main", 0)
        clause_b = Clause([token2], 2, 3, "main", 2)

        pair = ClausePair(
            clause_a=clause_a,
            clause_b=clause_b,
            zone_a_tokens=[token1],
            zone_b_tokens=[token2],
            pair_type="punctuation"
        )

        assert pair.clause_a == clause_a
        assert pair.clause_b == clause_b
        assert len(pair.zone_a_tokens) == 1
        assert len(pair.zone_b_tokens) == 1
        assert pair.pair_type == "punctuation"

    def test_clause_pair_types(self):
        """Test different pair types."""
        token = Token("test", "NOUN", "T EH S T", True, 1)
        clause = Clause([token], 0, 1, "main", 0)

        for pair_type in ["punctuation", "conjunction", "transition"]:
            pair = ClausePair(clause, clause, [token], [token], pair_type)
            assert pair.pair_type == pair_type

    def test_clause_pair_different_zone_sizes(self):
        """Test ClausePair with different zone sizes."""
        tokens_a = [Token(f"word{i}", "NOUN", "W ER D", True, 1) for i in range(3)]
        tokens_b = [Token(f"test{i}", "VERB", "T EH S T", True, 1) for i in range(2)]

        clause = Clause(tokens_a, 0, 3, "main", 1)
        pair = ClausePair(clause, clause, tokens_a, tokens_b, "conjunction")

        assert len(pair.zone_a_tokens) == 3
        assert len(pair.zone_b_tokens) == 2


class TestEchoScore:
    """Tests for EchoScore dataclass."""

    def test_echo_score_creation(self):
        """Test basic EchoScore instantiation."""
        score = EchoScore(
            phonetic_score=0.8,
            structural_score=0.7,
            semantic_score=0.9,
            combined_score=0.8
        )
        assert score.phonetic_score == 0.8
        assert score.structural_score == 0.7
        assert score.semantic_score == 0.9
        assert score.combined_score == 0.8

    def test_echo_score_range_zero_to_one(self):
        """Test that scores are in valid range [0.0, 1.0]."""
        score = EchoScore(0.0, 0.5, 1.0, 0.5)
        assert 0.0 <= score.phonetic_score <= 1.0
        assert 0.0 <= score.structural_score <= 1.0
        assert 0.0 <= score.semantic_score <= 1.0
        assert 0.0 <= score.combined_score <= 1.0

    def test_echo_score_all_zeros(self):
        """Test EchoScore with all zero scores (no similarity)."""
        score = EchoScore(0.0, 0.0, 0.0, 0.0)
        assert score.combined_score == 0.0

    def test_echo_score_all_ones(self):
        """Test EchoScore with perfect similarity."""
        score = EchoScore(1.0, 1.0, 1.0, 1.0)
        assert score.combined_score == 1.0


class TestDocumentAnalysis:
    """Tests for DocumentAnalysis dataclass."""

    def test_document_analysis_creation(self):
        """Test basic DocumentAnalysis instantiation."""
        token = Token("test", "NOUN", "T EH S T", True, 1)
        clause = Clause([token], 0, 1, "main", 0)
        pair = ClausePair(clause, clause, [token], [token], "punctuation")
        score = EchoScore(0.8, 0.7, 0.9, 0.8)

        analysis = DocumentAnalysis(
            text="Test document text.",
            clause_pairs=[pair],
            echo_scores=[score],
            final_score=0.8,
            z_score=2.5,
            confidence=0.95
        )

        assert analysis.text == "Test document text."
        assert len(analysis.clause_pairs) == 1
        assert len(analysis.echo_scores) == 1
        assert analysis.final_score == 0.8
        assert analysis.z_score == 2.5
        assert analysis.confidence == 0.95

    def test_document_analysis_empty_pairs(self):
        """Test DocumentAnalysis with no clause pairs."""
        analysis = DocumentAnalysis(
            text="Short text.",
            clause_pairs=[],
            echo_scores=[],
            final_score=0.0,
            z_score=0.0,
            confidence=0.5
        )
        assert len(analysis.clause_pairs) == 0
        assert len(analysis.echo_scores) == 0

    def test_document_analysis_multiple_pairs(self):
        """Test DocumentAnalysis with multiple clause pairs and scores."""
        token = Token("test", "NOUN", "T EH S T", True, 1)
        clause = Clause([token], 0, 1, "main", 0)

        pairs = [
            ClausePair(clause, clause, [token], [token], "punctuation"),
            ClausePair(clause, clause, [token], [token], "conjunction"),
            ClausePair(clause, clause, [token], [token], "transition"),
        ]

        scores = [
            EchoScore(0.8, 0.7, 0.9, 0.8),
            EchoScore(0.6, 0.5, 0.7, 0.6),
            EchoScore(0.9, 0.8, 0.95, 0.88),
        ]

        analysis = DocumentAnalysis(
            text="Longer document with multiple clauses.",
            clause_pairs=pairs,
            echo_scores=scores,
            final_score=0.76,
            z_score=1.8,
            confidence=0.85
        )

        assert len(analysis.clause_pairs) == 3
        assert len(analysis.echo_scores) == 3
        assert analysis.final_score == 0.76

    def test_document_analysis_high_confidence(self):
        """Test DocumentAnalysis with high watermark confidence."""
        token = Token("test", "NOUN", "T EH S T", True, 1)
        clause = Clause([token], 0, 1, "main", 0)
        pair = ClausePair(clause, clause, [token], [token], "punctuation")
        score = EchoScore(0.95, 0.92, 0.98, 0.95)

        analysis = DocumentAnalysis(
            text="Watermarked text.",
            clause_pairs=[pair],
            echo_scores=[score],
            final_score=0.95,
            z_score=3.5,
            confidence=0.998
        )

        assert analysis.z_score > 3.0
        assert analysis.confidence > 0.99


class TestDataModelIntegration:
    """Integration tests for data models working together."""

    def test_full_pipeline_data_flow(self):
        """Test complete data flow through all models."""
        # Create tokens
        tokens_a = [
            Token("technology", "NOUN", "T EH K N AA L AH JH IY", True, 4),
            Token("became", "VERB", "B IH K EY M", True, 2),
            Token("obsolete", "ADJ", "AA B S AH L IY T", True, 3),
        ]

        tokens_b = [
            Token("it", "PRON", "IH T", False, 1),
            Token("expanded", "VERB", "IH K S P AE N D IH D", True, 3),
            Token("creativity", "NOUN", "K R IY EY T IH V IH T IY", True, 5),
        ]

        # Create clauses
        clause_a = Clause(tokens_a, 0, 3, "main", 1)
        clause_b = Clause(tokens_b, 4, 7, "main", 5)

        # Create clause pair
        pair = ClausePair(
            clause_a=clause_a,
            clause_b=clause_b,
            zone_a_tokens=[tokens_a[-1]],  # "obsolete"
            zone_b_tokens=[tokens_b[1]],   # "expanded"
            pair_type="conjunction"
        )

        # Create echo score
        score = EchoScore(
            phonetic_score=0.75,
            structural_score=0.68,
            semantic_score=0.82,
            combined_score=0.75
        )

        # Create document analysis
        analysis = DocumentAnalysis(
            text="The technology became obsolete, but it expanded their creativity.",
            clause_pairs=[pair],
            echo_scores=[score],
            final_score=0.75,
            z_score=2.1,
            confidence=0.92
        )

        # Verify complete pipeline
        assert len(analysis.clause_pairs) == 1
        assert len(analysis.clause_pairs[0].zone_a_tokens) == 1
        assert analysis.clause_pairs[0].zone_a_tokens[0].text == "obsolete"
        assert analysis.echo_scores[0].combined_score == 0.75
        assert analysis.z_score > 2.0

    def test_dataclass_equality(self):
        """Test that dataclasses support equality comparison."""
        token1 = Token("test", "NOUN", "T EH S T", True, 1)
        token2 = Token("test", "NOUN", "T EH S T", True, 1)
        assert token1 == token2

        score1 = EchoScore(0.8, 0.7, 0.9, 0.8)
        score2 = EchoScore(0.8, 0.7, 0.9, 0.8)
        assert score1 == score2
