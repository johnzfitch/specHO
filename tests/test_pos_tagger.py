"""Tests for POSTagger (Task 2.2).

Tests POS tagging and content word identification functionality.

Tier: 1 (MVP)
Coverage: POS tagging, content word filtering
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from preprocessor.tokenizer import Tokenizer
from preprocessor.pos_tagger import POSTagger
from models import Token


class TestPOSTaggerBasic:
    """Basic tests for POSTagger class."""

    def test_pos_tagger_initialization(self):
        """Test POSTagger can be initialized."""
        pos_tagger = POSTagger()
        assert pos_tagger.nlp is not None
        assert pos_tagger.content_pos_tags is not None

    def test_content_pos_tags_defined(self):
        """Test that content POS tags are properly defined."""
        pos_tagger = POSTagger()
        assert "NOUN" in pos_tagger.content_pos_tags
        assert "VERB" in pos_tagger.content_pos_tags
        assert "ADJ" in pos_tagger.content_pos_tags
        assert "ADV" in pos_tagger.content_pos_tags


class TestPOSTagging:
    """Tests for tag() method."""

    def test_tag_simple_tokens(self):
        """Test tagging simple tokens."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("The cat sat")
        tagged_tokens = pos_tagger.tag(tokens)

        assert len(tagged_tokens) == 3
        assert all(t.pos_tag != "" for t in tagged_tokens)

    def test_tag_preserves_text(self):
        """Test that tagging preserves token text."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("Hello world")
        tagged_tokens = pos_tagger.tag(tokens)

        assert tagged_tokens[0].text == "Hello"
        assert tagged_tokens[1].text == "world"

    def test_tag_noun_correctly(self):
        """Test that nouns are tagged correctly."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("The cat")
        tagged_tokens = pos_tagger.tag(tokens)

        # "cat" should be tagged as NOUN
        cat_token = next(t for t in tagged_tokens if t.text == "cat")
        assert cat_token.pos_tag == "NOUN"

    def test_tag_verb_correctly(self):
        """Test that verbs are tagged correctly."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("The cat runs")
        tagged_tokens = pos_tagger.tag(tokens)

        # "runs" should be tagged as VERB
        verb_token = next(t for t in tagged_tokens if t.text == "runs")
        assert verb_token.pos_tag == "VERB"

    def test_tag_adjective_correctly(self):
        """Test that adjectives are tagged correctly."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("The quick fox")
        tagged_tokens = pos_tagger.tag(tokens)

        # "quick" should be tagged as ADJ
        adj_token = next(t for t in tagged_tokens if t.text == "quick")
        assert adj_token.pos_tag == "ADJ"

    def test_tag_determiner_correctly(self):
        """Test that determiners are tagged correctly."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("The cat")
        tagged_tokens = pos_tagger.tag(tokens)

        # "The" should be tagged as DET
        det_token = next(t for t in tagged_tokens if t.text == "The")
        assert det_token.pos_tag == "DET"

    def test_tag_empty_list(self):
        """Test tagging empty token list."""
        pos_tagger = POSTagger()
        tagged_tokens = pos_tagger.tag([])
        assert tagged_tokens == []

    def test_tag_preserves_other_fields(self):
        """Test that tagging preserves phonetic and syllable_count fields."""
        token = Token("test", "", "T EH S T", False, 2)
        pos_tagger = POSTagger()

        tagged_tokens = pos_tagger.tag([token])

        assert tagged_tokens[0].phonetic == "T EH S T"
        assert tagged_tokens[0].syllable_count == 2


class TestContentWordIdentification:
    """Tests for content word identification."""

    def test_is_content_word_noun(self):
        """Test that nouns are identified as content words."""
        pos_tagger = POSTagger()
        token = Token("cat", "NOUN", "", False, 0)

        assert pos_tagger.is_content_word(token) is True

    def test_is_content_word_verb(self):
        """Test that verbs are identified as content words."""
        pos_tagger = POSTagger()
        token = Token("run", "VERB", "", False, 0)

        assert pos_tagger.is_content_word(token) is True

    def test_is_content_word_adjective(self):
        """Test that adjectives are identified as content words."""
        pos_tagger = POSTagger()
        token = Token("quick", "ADJ", "", False, 0)

        assert pos_tagger.is_content_word(token) is True

    def test_is_content_word_adverb(self):
        """Test that adverbs are identified as content words."""
        pos_tagger = POSTagger()
        token = Token("quickly", "ADV", "", False, 0)

        assert pos_tagger.is_content_word(token) is True

    def test_is_not_content_word_determiner(self):
        """Test that determiners are not content words."""
        pos_tagger = POSTagger()
        token = Token("the", "DET", "", False, 0)

        assert pos_tagger.is_content_word(token) is False

    def test_is_not_content_word_preposition(self):
        """Test that prepositions are not content words."""
        pos_tagger = POSTagger()
        token = Token("in", "ADP", "", False, 0)

        assert pos_tagger.is_content_word(token) is False

    def test_is_not_content_word_pronoun(self):
        """Test that pronouns are not content words."""
        pos_tagger = POSTagger()
        token = Token("it", "PRON", "", False, 0)

        assert pos_tagger.is_content_word(token) is False

    def test_is_not_content_word_punctuation(self):
        """Test that punctuation is not content word."""
        pos_tagger = POSTagger()
        token = Token(".", "PUNCT", "", False, 0)

        assert pos_tagger.is_content_word(token) is False

    def test_is_content_word_from_pos_directly(self):
        """Test is_content_word_from_pos method."""
        pos_tagger = POSTagger()

        assert pos_tagger.is_content_word_from_pos("NOUN") is True
        assert pos_tagger.is_content_word_from_pos("VERB") is True
        assert pos_tagger.is_content_word_from_pos("ADJ") is True
        assert pos_tagger.is_content_word_from_pos("DET") is False
        assert pos_tagger.is_content_word_from_pos("ADP") is False


class TestContentWordSetting:
    """Tests for automatic is_content_word field setting."""

    def test_tag_sets_content_word_flag_true(self):
        """Test that tag() sets is_content_word to True for content words."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("cat")
        tagged_tokens = pos_tagger.tag(tokens)

        assert tagged_tokens[0].is_content_word is True

    def test_tag_sets_content_word_flag_false(self):
        """Test that tag() sets is_content_word to False for function words."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("the")
        tagged_tokens = pos_tagger.tag(tokens)

        assert tagged_tokens[0].is_content_word is False

    def test_tag_mixed_content_and_function_words(self):
        """Test tagging sentence with both content and function words."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("The cat sat on the mat")
        tagged_tokens = pos_tagger.tag(tokens)

        # Content words: cat, sat, mat
        content_words = [t for t in tagged_tokens if t.is_content_word]
        assert len(content_words) >= 3

        # Function words: The, on, the
        function_words = [t for t in tagged_tokens if not t.is_content_word]
        assert len(function_words) >= 3


class TestGetContentWords:
    """Tests for get_content_words() method."""

    def test_get_content_words_filters_correctly(self):
        """Test that get_content_words returns only content words."""
        tokens = [
            Token("the", "DET", "", False, 0),
            Token("cat", "NOUN", "", True, 0),
            Token("sat", "VERB", "", True, 0),
            Token("on", "ADP", "", False, 0),
        ]

        pos_tagger = POSTagger()
        content = pos_tagger.get_content_words(tokens)

        assert len(content) == 2
        assert content[0].text == "cat"
        assert content[1].text == "sat"

    def test_get_content_words_empty_list(self):
        """Test get_content_words with empty list."""
        pos_tagger = POSTagger()
        content = pos_tagger.get_content_words([])
        assert content == []

    def test_get_content_words_all_function_words(self):
        """Test get_content_words with only function words."""
        tokens = [
            Token("the", "DET", "", False, 0),
            Token("of", "ADP", "", False, 0),
            Token("a", "DET", "", False, 0),
        ]

        pos_tagger = POSTagger()
        content = pos_tagger.get_content_words(tokens)
        assert content == []

    def test_get_content_words_all_content_words(self):
        """Test get_content_words with only content words."""
        tokens = [
            Token("cat", "NOUN", "", True, 0),
            Token("runs", "VERB", "", True, 0),
            Token("quickly", "ADV", "", True, 0),
        ]

        pos_tagger = POSTagger()
        content = pos_tagger.get_content_words(tokens)
        assert len(content) == 3


class TestTagTextDirectly:
    """Tests for tag_text_directly() convenience method."""

    def test_tag_text_directly_works(self):
        """Test tag_text_directly convenience method."""
        pos_tagger = POSTagger()
        tokens = pos_tagger.tag_text_directly("The cat sat")

        assert len(tokens) == 3
        assert all(t.pos_tag != "" for t in tokens)

    def test_tag_text_directly_returns_tokens(self):
        """Test that tag_text_directly returns Token objects."""
        pos_tagger = POSTagger()
        tokens = pos_tagger.tag_text_directly("Hello")

        assert isinstance(tokens[0], Token)

    def test_tag_text_directly_sets_content_words(self):
        """Test that tag_text_directly sets is_content_word correctly."""
        pos_tagger = POSTagger()
        tokens = pos_tagger.tag_text_directly("The cat sat")

        content_words = [t for t in tokens if t.is_content_word]
        assert len(content_words) >= 2  # cat, sat


class TestIntegrationWithTokenizer:
    """Integration tests with Tokenizer."""

    def test_full_pipeline_tokenizer_to_pos_tagger(self):
        """Test complete tokenization → POS tagging pipeline."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.tokenize(text)
        tagged_tokens = pos_tagger.tag(tokens)

        # Verify all tokens have POS tags
        assert all(t.pos_tag != "" for t in tagged_tokens)

        # Verify content words are identified
        content_words = pos_tagger.get_content_words(tagged_tokens)
        assert len(content_words) > 0

        # Verify specific words
        quick_token = next(t for t in tagged_tokens if t.text == "quick")
        assert quick_token.pos_tag == "ADJ"
        assert quick_token.is_content_word is True

        the_token = next(t for t in tagged_tokens if t.text == "The")
        assert the_token.pos_tag == "DET"
        assert the_token.is_content_word is False

    def test_pipeline_with_contractions(self):
        """Test pipeline handles contractions correctly."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("Don't worry")
        tagged_tokens = pos_tagger.tag(tokens)

        # Verify all parts of contraction are tagged
        assert all(t.pos_tag != "" for t in tagged_tokens)

    def test_pipeline_preserves_token_count(self):
        """Test that POS tagging preserves token count."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        text = "Test sentence with five words."
        tokens = tokenizer.tokenize(text)
        tagged_tokens = pos_tagger.tag(tokens)

        assert len(tokens) == len(tagged_tokens)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_tag_single_token(self):
        """Test tagging single token."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("Hello")
        tagged_tokens = pos_tagger.tag(tokens)

        assert len(tagged_tokens) == 1
        assert tagged_tokens[0].pos_tag != ""

    def test_tag_punctuation_only(self):
        """Test tagging punctuation."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("!")
        tagged_tokens = pos_tagger.tag(tokens)

        assert tagged_tokens[0].pos_tag == "PUNCT"
        assert tagged_tokens[0].is_content_word is False

    def test_tag_numbers(self):
        """Test tagging numbers."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("The year 2025")
        tagged_tokens = pos_tagger.tag(tokens)

        # Verify number token has POS tag
        num_token = next(t for t in tagged_tokens if t.text == "2025")
        assert num_token.pos_tag != ""

    def test_tag_proper_nouns(self):
        """Test tagging proper nouns."""
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()

        tokens = tokenizer.tokenize("John lives in London")
        tagged_tokens = pos_tagger.tag(tokens)

        # Proper nouns should be tagged as PROPN
        john_token = next(t for t in tagged_tokens if t.text == "John")
        assert john_token.pos_tag == "PROPN"
        assert john_token.is_content_word is True  # PROPN is a content word
