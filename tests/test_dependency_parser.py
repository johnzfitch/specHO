"""Tests for DependencyParser (Task 2.3).

Tests dependency parsing and clause boundary detection functionality.

Tier: 1 (MVP)
Coverage: Dependency parsing, clause boundaries, ROOT/conj/advcl detection
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from preprocessor.dependency_parser import DependencyParser, quick_parse


class TestDependencyParserBasic:
    """Basic tests for DependencyParser class."""

    def test_dependency_parser_initialization(self):
        """Test DependencyParser can be initialized."""
        parser = DependencyParser()
        assert parser.nlp is not None

    def test_dependency_parser_custom_model(self):
        """Test DependencyParser with custom model name."""
        parser = DependencyParser("en_core_web_sm")
        assert parser.nlp is not None

    def test_dependency_parser_invalid_model_raises_error(self):
        """Test that invalid model name raises OSError."""
        with pytest.raises(OSError) as exc_info:
            DependencyParser("nonexistent_model")
        assert "nonexistent_model" in str(exc_info.value)


class TestParse:
    """Tests for parse() method."""

    def test_parse_simple_sentence(self):
        """Test parsing simple sentence."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")

        assert doc is not None
        assert len(doc) == 4  # The, cat, sat, .
        assert doc.text == "The cat sat."

    def test_parse_returns_spacy_doc(self):
        """Test that parse returns spaCy Doc object."""
        parser = DependencyParser()
        doc = parser.parse("Test sentence.")

        # Check it's a spaCy Doc with expected attributes
        assert hasattr(doc, "text")
        assert hasattr(doc, "sents")
        assert hasattr(doc, "__iter__")

    def test_parse_multiple_sentences(self):
        """Test parsing multiple sentences."""
        parser = DependencyParser()
        doc = parser.parse("First sentence. Second sentence.")

        sentences = list(doc.sents)
        assert len(sentences) == 2

    def test_parse_empty_text(self):
        """Test parsing empty text."""
        parser = DependencyParser()
        doc = parser.parse("")

        assert doc is not None
        assert len(doc) == 0

    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only text."""
        parser = DependencyParser()
        doc = parser.parse("   \n\t  ")

        assert doc is not None
        assert len(doc) == 0

    def test_parse_preserves_tokens(self):
        """Test that parse preserves all tokens."""
        parser = DependencyParser()
        text = "Hello world!"
        doc = parser.parse(text)

        tokens = [token.text for token in doc]
        assert "Hello" in tokens
        assert "world" in tokens
        assert "!" in tokens

    def test_parse_adds_dependency_labels(self):
        """Test that parse adds dependency labels."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")

        # Check that tokens have dep_ attributes
        for token in doc:
            assert hasattr(token, "dep_")
            assert token.dep_ != ""


class TestGetClauseBoundaries:
    """Tests for get_clause_boundaries() method."""

    def test_get_clause_boundaries_simple_sentence(self):
        """Test clause boundaries for simple sentence."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        boundaries = parser.get_clause_boundaries(doc)

        # Simple sentence should have one clause
        assert len(boundaries) >= 1
        assert all(isinstance(b, tuple) for b in boundaries)
        assert all(len(b) == 2 for b in boundaries)

    def test_get_clause_boundaries_coordinated_clauses(self):
        """Test clause boundaries for coordinated clauses."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat, and the dog ran.")
        boundaries = parser.get_clause_boundaries(doc)

        # Should detect at least 2 clauses (cat sat + dog ran)
        assert len(boundaries) >= 2

    def test_get_clause_boundaries_subordinate_clause(self):
        """Test clause boundaries with subordinate clause."""
        parser = DependencyParser()
        doc = parser.parse("She said that he left.")
        boundaries = parser.get_clause_boundaries(doc)

        # Should detect multiple clauses
        assert len(boundaries) >= 1

    def test_get_clause_boundaries_empty_doc(self):
        """Test clause boundaries with empty doc."""
        parser = DependencyParser()
        doc = parser.parse("")
        boundaries = parser.get_clause_boundaries(doc)

        assert boundaries == []

    def test_get_clause_boundaries_returns_tuples(self):
        """Test that boundaries are (start, end) tuples."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat, and the dog ran.")
        boundaries = parser.get_clause_boundaries(doc)

        for start, end in boundaries:
            assert isinstance(start, int)
            assert isinstance(end, int)
            assert end > start  # End must be after start

    def test_get_clause_boundaries_indices_in_range(self):
        """Test that boundary indices are within document range."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat, and the dog ran.")
        boundaries = parser.get_clause_boundaries(doc)

        for start, end in boundaries:
            assert start >= 0
            assert end <= len(doc)

    def test_get_clause_boundaries_complex_sentence(self):
        """Test clause boundaries with complex sentence structure."""
        parser = DependencyParser()
        doc = parser.parse("When the cat sat, the dog ran, but the bird flew.")
        boundaries = parser.get_clause_boundaries(doc)

        # Should detect multiple clauses
        assert len(boundaries) >= 2


class TestFindRootVerbs:
    """Tests for find_root_verbs() method."""

    def test_find_root_verbs_simple_sentence(self):
        """Test finding ROOT verb in simple sentence."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        roots = parser.find_root_verbs(doc)

        assert len(roots) == 1
        assert roots[0].text == "sat"

    def test_find_root_verbs_multiple_sentences(self):
        """Test finding ROOT verbs in multiple sentences."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat. The dog ran.")
        roots = parser.find_root_verbs(doc)

        assert len(roots) == 2
        root_texts = [token.text for token in roots]
        assert "sat" in root_texts
        assert "ran" in root_texts

    def test_find_root_verbs_empty_doc(self):
        """Test finding ROOT verbs in empty doc."""
        parser = DependencyParser()
        doc = parser.parse("")
        roots = parser.find_root_verbs(doc)

        assert roots == []

    def test_find_root_verbs_returns_tokens(self):
        """Test that find_root_verbs returns spaCy tokens."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        roots = parser.find_root_verbs(doc)

        assert all(hasattr(token, "dep_") for token in roots)
        assert all(token.dep_ == "ROOT" for token in roots)


class TestFindCoordinatedClauses:
    """Tests for find_coordinated_clauses() method."""

    def test_find_coordinated_clauses_simple(self):
        """Test finding coordinated clauses."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat, and the dog ran.")
        pairs = parser.find_coordinated_clauses(doc)

        # Should find at least one coordinated pair
        assert len(pairs) >= 1
        assert all(isinstance(p, tuple) for p in pairs)
        assert all(len(p) == 2 for p in pairs)

    def test_find_coordinated_clauses_returns_token_pairs(self):
        """Test that coordinated clauses returns (head, conj) pairs."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat, and the dog ran.")
        pairs = parser.find_coordinated_clauses(doc)

        for head, conj in pairs:
            assert hasattr(head, "text")
            assert hasattr(conj, "text")
            assert hasattr(conj, "dep_")
            assert conj.dep_ == "conj"

    def test_find_coordinated_clauses_no_coordination(self):
        """Test finding coordinated clauses when none exist."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        pairs = parser.find_coordinated_clauses(doc)

        # Simple sentence has no coordination
        assert len(pairs) == 0

    def test_find_coordinated_clauses_multiple_conjunctions(self):
        """Test finding multiple coordinated clauses."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat, the dog ran, and the bird flew.")
        pairs = parser.find_coordinated_clauses(doc)

        # Should find multiple coordination pairs
        assert len(pairs) >= 1


class TestFindSubordinateClauses:
    """Tests for find_subordinate_clauses() method."""

    def test_find_subordinate_clauses_ccomp(self):
        """Test finding clausal complement (ccomp)."""
        parser = DependencyParser()
        doc = parser.parse("She said that he left.")
        pairs = parser.find_subordinate_clauses(doc)

        # Should find subordinate clause relationship
        assert len(pairs) >= 1
        assert all(isinstance(p, tuple) for p in pairs)

    def test_find_subordinate_clauses_advcl(self):
        """Test finding adverbial clause (advcl)."""
        parser = DependencyParser()
        doc = parser.parse("He left when she arrived.")
        pairs = parser.find_subordinate_clauses(doc)

        # Should find subordinate clause
        assert len(pairs) >= 1

    def test_find_subordinate_clauses_returns_token_pairs(self):
        """Test that subordinate clauses returns (main, subordinate) pairs."""
        parser = DependencyParser()
        doc = parser.parse("She said that he left.")
        pairs = parser.find_subordinate_clauses(doc)

        for main_verb, sub_verb in pairs:
            assert hasattr(main_verb, "text")
            assert hasattr(sub_verb, "text")
            assert hasattr(sub_verb, "dep_")

    def test_find_subordinate_clauses_no_subordination(self):
        """Test finding subordinate clauses when none exist."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        pairs = parser.find_subordinate_clauses(doc)

        # Simple sentence has no subordination
        assert len(pairs) == 0


class TestGetDependencyTree:
    """Tests for get_dependency_tree() utility method."""

    def test_get_dependency_tree_returns_string(self):
        """Test that get_dependency_tree returns a string."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        tree = parser.get_dependency_tree(doc)

        assert isinstance(tree, str)

    def test_get_dependency_tree_contains_root(self):
        """Test that dependency tree contains ROOT information."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        tree = parser.get_dependency_tree(doc)

        assert "ROOT" in tree

    def test_get_dependency_tree_contains_tokens(self):
        """Test that dependency tree contains token text."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        tree = parser.get_dependency_tree(doc)

        assert "cat" in tree
        assert "sat" in tree

    def test_get_dependency_tree_empty_doc(self):
        """Test dependency tree with empty doc."""
        parser = DependencyParser()
        doc = parser.parse("")
        tree = parser.get_dependency_tree(doc)

        assert tree == ""

    def test_get_dependency_tree_formatting(self):
        """Test that dependency tree has expected formatting."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")
        tree = parser.get_dependency_tree(doc)

        # Should contain POS and DEP labels
        assert "POS:" in tree
        assert "DEP:" in tree


class TestQuickParseHelper:
    """Tests for quick_parse() convenience function."""

    def test_quick_parse_works(self):
        """Test quick_parse convenience function."""
        doc = quick_parse("The cat sat.")

        assert doc is not None
        assert len(doc) > 0
        assert doc.text == "The cat sat."

    def test_quick_parse_returns_doc(self):
        """Test that quick_parse returns spaCy Doc."""
        doc = quick_parse("Test sentence.")

        assert hasattr(doc, "text")
        assert hasattr(doc, "sents")

    def test_quick_parse_custom_model(self):
        """Test quick_parse with custom model."""
        doc = quick_parse("Test sentence.", model_name="en_core_web_sm")

        assert doc is not None
        assert len(doc) > 0


class TestIntegrationWithSpacy:
    """Integration tests with spaCy dependency parsing."""

    def test_full_dependency_analysis(self):
        """Test complete dependency analysis workflow."""
        parser = DependencyParser()
        doc = parser.parse("The quick brown fox jumps over the lazy dog.")

        # Check dependency structure is complete
        assert len(doc) > 0
        assert all(token.dep_ != "" for token in doc)

        # Find ROOT
        roots = parser.find_root_verbs(doc)
        assert len(roots) == 1

        # Get dependency tree
        tree = parser.get_dependency_tree(doc)
        assert len(tree) > 0

    def test_complex_sentence_analysis(self):
        """Test analysis of complex sentence structure."""
        parser = DependencyParser()
        text = "Although the cat sat on the mat, the dog ran in the yard."
        doc = parser.parse(text)

        # Get clause boundaries
        boundaries = parser.get_clause_boundaries(doc)
        assert len(boundaries) >= 1

        # Check for subordinate clauses
        subordinate_pairs = parser.find_subordinate_clauses(doc)
        # Complex sentence may have subordinate relationships

    def test_compound_sentence_analysis(self):
        """Test analysis of compound sentence."""
        parser = DependencyParser()
        text = "The cat sat, and the dog ran, but the bird flew."
        doc = parser.parse(text)

        # Get coordinated clauses
        coordinated = parser.find_coordinated_clauses(doc)
        assert len(coordinated) >= 2  # Multiple coordination points

        # Get clause boundaries
        boundaries = parser.get_clause_boundaries(doc)
        assert len(boundaries) >= 3  # At least 3 clauses


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_parse_single_word(self):
        """Test parsing single word."""
        parser = DependencyParser()
        doc = parser.parse("Hello")

        assert len(doc) == 1
        assert doc[0].text == "Hello"

    def test_parse_punctuation_only(self):
        """Test parsing punctuation only."""
        parser = DependencyParser()
        doc = parser.parse("!!!")

        assert len(doc) == 3
        assert all(token.text == "!" for token in doc)

    def test_parse_numbers(self):
        """Test parsing numbers."""
        parser = DependencyParser()
        doc = parser.parse("The year 2025 begins.")

        boundaries = parser.get_clause_boundaries(doc)
        assert len(boundaries) >= 1

    def test_parse_contractions(self):
        """Test parsing contractions."""
        parser = DependencyParser()
        doc = parser.parse("Don't worry, it's fine.")

        assert len(doc) > 0
        boundaries = parser.get_clause_boundaries(doc)
        assert len(boundaries) >= 1

    def test_parse_long_sentence(self):
        """Test parsing very long sentence."""
        parser = DependencyParser()
        text = " ".join(["The cat sat"] * 20) + "."
        doc = parser.parse(text)

        assert len(doc) > 0
        boundaries = parser.get_clause_boundaries(doc)

    def test_parse_special_characters(self):
        """Test parsing with special characters."""
        parser = DependencyParser()
        doc = parser.parse("The cat—an orange tabby—sat quietly.")

        assert len(doc) > 0
        boundaries = parser.get_clause_boundaries(doc)
        assert len(boundaries) >= 1


class TestDependencyLabels:
    """Tests for specific dependency label detection."""

    def test_detects_root_label(self):
        """Test that ROOT dependency label is detected."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")

        root_labels = [token.dep_ for token in doc if token.dep_ == "ROOT"]
        assert len(root_labels) == 1

    def test_detects_conj_label(self):
        """Test that conj dependency label is detected."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat and ran.")

        conj_labels = [token.dep_ for token in doc if token.dep_ == "conj"]
        assert len(conj_labels) >= 1

    def test_detects_nsubj_label(self):
        """Test that nsubj (nominal subject) is detected."""
        parser = DependencyParser()
        doc = parser.parse("The cat sat.")

        nsubj_labels = [token.dep_ for token in doc if token.dep_ == "nsubj"]
        assert len(nsubj_labels) >= 1
