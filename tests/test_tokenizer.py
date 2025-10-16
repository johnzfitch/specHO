"""Basic tests for Tokenizer (Task 2.1).

Comprehensive preprocessor tests will be in test_preprocessor.py (Task 8.1)
after all preprocessor components are implemented.

Tier: 1 (MVP)
Coverage: Basic tokenization functionality
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from preprocessor.tokenizer import Tokenizer, quick_tokenize
from models import Token


class TestTokenizerBasic:
    """Basic tests for Tokenizer class."""

    def test_tokenizer_initialization(self):
        """Test Tokenizer can be initialized."""
        tokenizer = Tokenizer()
        assert tokenizer.nlp is not None

    def test_tokenize_simple_text(self):
        """Test tokenizing simple text."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello world")

        assert len(tokens) == 2
        assert tokens[0].text == "Hello"
        assert tokens[1].text == "world"

    def test_tokenize_with_punctuation(self):
        """Test tokenizing text with punctuation."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Hello, world!")

        assert len(tokens) == 4
        assert tokens[0].text == "Hello"
        assert tokens[1].text == ","
        assert tokens[2].text == "world"
        assert tokens[3].text == "!"

    def test_tokenize_contractions(self):
        """Test that contractions are split correctly."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Don't worry")

        assert len(tokens) == 3
        assert tokens[0].text == "Do"
        assert tokens[1].text == "n't"
        assert tokens[2].text == "worry"

    def test_tokenize_returns_token_objects(self):
        """Test that tokenize returns Token objects."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Test")

        assert len(tokens) == 1
        assert isinstance(tokens[0], Token)

    def test_token_placeholder_fields(self):
        """Test that Token placeholder fields are set correctly."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("test")
        token = tokens[0]

        assert token.text == "test"
        assert token.pos_tag == ""
        assert token.phonetic == ""
        assert token.is_content_word is False
        assert token.syllable_count == 0

    def test_tokenize_empty_string(self):
        """Test tokenizing empty string returns empty list."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("")
        assert tokens == []

    def test_tokenize_whitespace_only(self):
        """Test tokenizing whitespace returns empty list."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("   ")
        assert tokens == []

    def test_tokenize_multiline_text(self):
        """Test tokenizing multiline text."""
        tokenizer = Tokenizer()
        text = "First line.\nSecond line."
        tokens = tokenizer.tokenize(text)

        assert len(tokens) > 0
        token_texts = [t.text for t in tokens]
        assert "First" in token_texts
        assert "Second" in token_texts


class TestTokenizerWithDoc:
    """Tests for tokenize_with_doc method."""

    def test_tokenize_with_doc_returns_tuple(self):
        """Test that tokenize_with_doc returns (tokens, doc) tuple."""
        tokenizer = Tokenizer()
        result = tokenizer.tokenize_with_doc("Hello world")

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tokenize_with_doc_tokens_match(self):
        """Test that tokens from tokenize_with_doc match tokenize."""
        tokenizer = Tokenizer()
        tokens1 = tokenizer.tokenize("Hello world")
        tokens2, doc = tokenizer.tokenize_with_doc("Hello world")

        assert len(tokens1) == len(tokens2)
        for t1, t2 in zip(tokens1, tokens2):
            assert t1.text == t2.text

    def test_tokenize_with_doc_returns_spacy_doc(self):
        """Test that doc object is a spaCy Doc."""
        tokenizer = Tokenizer()
        tokens, doc = tokenizer.tokenize_with_doc("Hello world")

        # Check it's a spaCy doc by checking it has spaCy doc attributes
        assert hasattr(doc, 'text')
        assert hasattr(doc, '__iter__')
        assert len(list(doc)) == len(tokens)

    def test_tokenize_with_doc_empty_string(self):
        """Test tokenize_with_doc with empty string."""
        tokenizer = Tokenizer()
        tokens, doc = tokenizer.tokenize_with_doc("")

        assert tokens == []
        assert doc is not None


class TestQuickTokenize:
    """Tests for quick_tokenize convenience function."""

    def test_quick_tokenize_works(self):
        """Test quick_tokenize convenience function."""
        tokens = quick_tokenize("Hello world")

        assert len(tokens) == 2
        assert tokens[0].text == "Hello"
        assert tokens[1].text == "world"

    def test_quick_tokenize_returns_tokens(self):
        """Test that quick_tokenize returns Token objects."""
        tokens = quick_tokenize("Test")
        assert isinstance(tokens[0], Token)


class TestTokenizerEdgeCases:
    """Tests for edge cases and special inputs."""

    def test_tokenize_numbers(self):
        """Test tokenizing numbers."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("The year 2025")

        token_texts = [t.text for t in tokens]
        assert "The" in token_texts
        assert "year" in token_texts
        assert "2025" in token_texts

    def test_tokenize_special_characters(self):
        """Test tokenizing special characters."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("Cost is $100")

        token_texts = [t.text for t in tokens]
        assert "Cost" in token_texts
        assert "$" in token_texts
        assert "100" in token_texts

    def test_tokenize_hyphenated_words(self):
        """Test tokenizing hyphenated words."""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize("well-known fact")

        # spaCy may handle hyphens differently, just verify we get tokens
        assert len(tokens) >= 3
        token_texts = [t.text for t in tokens]
        assert "well" in token_texts or "well-known" in token_texts

    def test_tokenize_long_text(self):
        """Test tokenizing longer text."""
        tokenizer = Tokenizer()
        text = "The quick brown fox jumps over the lazy dog. This is a longer sentence."
        tokens = tokenizer.tokenize(text)

        assert len(tokens) > 10
        assert all(isinstance(t, Token) for t in tokens)

    def test_tokenize_apostrophes(self):
        """Test various apostrophe usages."""
        tokenizer = Tokenizer()

        # Possessive
        tokens1 = tokenizer.tokenize("John's book")
        token_texts1 = [t.text for t in tokens1]
        assert "John" in token_texts1

        # It's (contraction)
        tokens2 = tokenizer.tokenize("It's working")
        token_texts2 = [t.text for t in tokens2]
        assert "It" in token_texts2 or "it" in token_texts2
