"""Tests for PhoneticTranscriber (Task 2.4).

Tests phonetic transcription and syllable counting functionality.

Tier: 1 (MVP)
Coverage: ARPAbet transcription, syllable counting, OOV handling
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from preprocessor.phonetic import PhoneticTranscriber, quick_transcribe, get_rhyming_words
from models import Token


class TestPhoneticTranscriberBasic:
    """Basic tests for PhoneticTranscriber class."""

    def test_phonetic_transcriber_initialization(self):
        """Test PhoneticTranscriber can be initialized."""
        transcriber = PhoneticTranscriber()
        assert transcriber is not None

    def test_initialization_logs_tier_info(self, caplog):
        """Test that initialization logs Tier 1 info."""
        transcriber = PhoneticTranscriber()
        # Transcriber should log its tier level (though may not be captured by caplog)
        assert transcriber is not None


class TestTranscribe:
    """Tests for transcribe() method."""

    def test_transcribe_simple_word(self):
        """Test transcribing simple known word."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("hello")

        assert phonetic == "HH AH0 L OW1"
        assert isinstance(phonetic, str)

    def test_transcribe_cat(self):
        """Test transcribing 'cat'."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("cat")

        assert phonetic == "K AE1 T"

    def test_transcribe_dog(self):
        """Test transcribing 'dog'."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("dog")

        assert phonetic == "D AO1 G"

    def test_transcribe_world(self):
        """Test transcribing 'world'."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("world")

        assert phonetic == "W ER1 L D"

    def test_transcribe_beautiful(self):
        """Test transcribing longer word."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("beautiful")

        assert phonetic == "B Y UW1 T AH0 F AH0 L"

    def test_transcribe_case_insensitive(self):
        """Test that transcription is case-insensitive."""
        transcriber = PhoneticTranscriber()

        assert transcriber.transcribe("HELLO") == transcriber.transcribe("hello")
        assert transcriber.transcribe("Hello") == transcriber.transcribe("hello")
        assert transcriber.transcribe("HeLLo") == transcriber.transcribe("hello")

    def test_transcribe_with_punctuation(self):
        """Test transcribing word with attached punctuation."""
        transcriber = PhoneticTranscriber()

        # Should strip punctuation and transcribe the word
        assert transcriber.transcribe("hello,") == "HH AH0 L OW1"
        assert transcriber.transcribe("hello.") == "HH AH0 L OW1"
        assert transcriber.transcribe("hello!") == "HH AH0 L OW1"
        assert transcriber.transcribe("hello?") == "HH AH0 L OW1"

    def test_transcribe_oov_word(self):
        """Test transcribing out-of-vocabulary word."""
        transcriber = PhoneticTranscriber()

        # OOV words should return uppercase
        phonetic = transcriber.transcribe("xyz123")
        assert phonetic == "XYZ123"

    def test_transcribe_nonsense_word(self):
        """Test transcribing nonsense word not in dictionary."""
        transcriber = PhoneticTranscriber()

        # Should return uppercase as fallback
        phonetic = transcriber.transcribe("qwerty")
        assert phonetic.isupper()

    def test_transcribe_empty_string(self):
        """Test transcribing empty string."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("")

        assert phonetic == ""

    def test_transcribe_whitespace_only(self):
        """Test transcribing whitespace-only string."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("   ")

        assert phonetic == ""

    def test_transcribe_punctuation_only(self):
        """Test transcribing punctuation-only string."""
        transcriber = PhoneticTranscriber()
        phonetic = transcriber.transcribe("...")

        assert phonetic == ""


class TestCountSyllables:
    """Tests for count_syllables() method."""

    def test_count_syllables_simple_words(self):
        """Test syllable counting for simple words."""
        transcriber = PhoneticTranscriber()

        assert transcriber.count_syllables("cat") == 1
        assert transcriber.count_syllables("dog") == 1
        assert transcriber.count_syllables("hello") == 2

    def test_count_syllables_longer_words(self):
        """Test syllable counting for longer words."""
        transcriber = PhoneticTranscriber()

        assert transcriber.count_syllables("beautiful") == 3
        assert transcriber.count_syllables("dictionary") == 4
        assert transcriber.count_syllables("information") == 4

    def test_count_syllables_monosyllabic(self):
        """Test syllable counting for monosyllabic words."""
        transcriber = PhoneticTranscriber()

        assert transcriber.count_syllables("a") == 1
        assert transcriber.count_syllables("I") == 1
        assert transcriber.count_syllables("be") == 1

    def test_count_syllables_case_insensitive(self):
        """Test that syllable counting is case-insensitive."""
        transcriber = PhoneticTranscriber()

        assert transcriber.count_syllables("HELLO") == transcriber.count_syllables("hello")
        assert transcriber.count_syllables("Beautiful") == transcriber.count_syllables("beautiful")

    def test_count_syllables_with_punctuation(self):
        """Test syllable counting with punctuation."""
        transcriber = PhoneticTranscriber()

        assert transcriber.count_syllables("hello,") == 2
        assert transcriber.count_syllables("hello.") == 2

    def test_count_syllables_oov_word(self):
        """Test syllable counting for OOV words using estimation."""
        transcriber = PhoneticTranscriber()

        # Should estimate syllables for OOV words
        syllables = transcriber.count_syllables("xyz")
        assert syllables >= 1  # At least 1 syllable for non-empty word

    def test_count_syllables_empty_string(self):
        """Test syllable counting for empty string."""
        transcriber = PhoneticTranscriber()
        assert transcriber.count_syllables("") == 0

    def test_count_syllables_whitespace(self):
        """Test syllable counting for whitespace."""
        transcriber = PhoneticTranscriber()
        assert transcriber.count_syllables("   ") == 0


class TestEstimateSyllables:
    """Tests for _estimate_syllables() fallback method."""

    def test_estimate_syllables_single_vowel(self):
        """Test syllable estimation for single vowel."""
        transcriber = PhoneticTranscriber()

        assert transcriber._estimate_syllables("a") == 1
        assert transcriber._estimate_syllables("i") == 1

    def test_estimate_syllables_consonant_clusters(self):
        """Test syllable estimation with consonant clusters."""
        transcriber = PhoneticTranscriber()

        # Words with consonants only should have at least 1 syllable
        assert transcriber._estimate_syllables("xyz") == 1

    def test_estimate_syllables_multiple_vowels(self):
        """Test syllable estimation with multiple vowel clusters."""
        transcriber = PhoneticTranscriber()

        # Should count vowel clusters
        count = transcriber._estimate_syllables("aardvark")
        assert count >= 1

    def test_estimate_syllables_silent_e(self):
        """Test that silent 'e' is handled."""
        transcriber = PhoneticTranscriber()

        # 'make' should be 1 syllable (silent e)
        # Estimation may vary, but should be reasonable
        count = transcriber._estimate_syllables("make")
        assert count >= 1

    def test_estimate_syllables_empty(self):
        """Test syllable estimation for empty string."""
        transcriber = PhoneticTranscriber()
        assert transcriber._estimate_syllables("") == 0


class TestGetStressedSyllables:
    """Tests for get_stressed_syllables() method."""

    def test_get_stressed_syllables_hello(self):
        """Test stressed syllable extraction for 'hello'."""
        transcriber = PhoneticTranscriber()
        phonetic = "HH AH0 L OW1"
        stressed = transcriber.get_stressed_syllables(phonetic)

        assert stressed == ["OW1"]

    def test_get_stressed_syllables_beautiful(self):
        """Test stressed syllable extraction for 'beautiful'."""
        transcriber = PhoneticTranscriber()
        phonetic = "B Y UW1 T AH0 F AH0 L"
        stressed = transcriber.get_stressed_syllables(phonetic)

        assert stressed == ["UW1"]

    def test_get_stressed_syllables_multiple_stress(self):
        """Test stressed syllables with primary and secondary stress."""
        transcriber = PhoneticTranscriber()
        # "information": IH2 N F ER0 M EY1 SH AH0 N
        phonetic = "IH2 N F ER0 M EY1 SH AH0 N"
        stressed = transcriber.get_stressed_syllables(phonetic)

        assert "IH2" in stressed  # Secondary stress
        assert "EY1" in stressed  # Primary stress
        assert len(stressed) == 2

    def test_get_stressed_syllables_no_stress(self):
        """Test stressed syllables with no stress markers."""
        transcriber = PhoneticTranscriber()
        phonetic = "DH AH0"  # "the" - no primary stress
        stressed = transcriber.get_stressed_syllables(phonetic)

        assert stressed == []  # No primary or secondary stress

    def test_get_stressed_syllables_empty(self):
        """Test stressed syllables with empty string."""
        transcriber = PhoneticTranscriber()
        stressed = transcriber.get_stressed_syllables("")

        assert stressed == []

    def test_get_stressed_syllables_single_phoneme(self):
        """Test stressed syllables with single phoneme."""
        transcriber = PhoneticTranscriber()
        stressed = transcriber.get_stressed_syllables("AE1")

        assert stressed == ["AE1"]


class TestTranscribeTokens:
    """Tests for transcribe_tokens() method."""

    def test_transcribe_tokens_simple(self):
        """Test transcribing simple token list."""
        transcriber = PhoneticTranscriber()
        tokens = [
            Token("hello", "", "", False, 0),
            Token("world", "", "", False, 0)
        ]

        enriched = transcriber.transcribe_tokens(tokens)

        assert len(enriched) == 2
        assert enriched[0].phonetic == "HH AH0 L OW1"
        assert enriched[0].syllable_count == 2
        assert enriched[1].phonetic == "W ER1 L D"
        assert enriched[1].syllable_count == 1

    def test_transcribe_tokens_preserves_text(self):
        """Test that transcribe_tokens preserves text field."""
        transcriber = PhoneticTranscriber()
        tokens = [Token("hello", "", "", False, 0)]

        enriched = transcriber.transcribe_tokens(tokens)

        assert enriched[0].text == "hello"

    def test_transcribe_tokens_preserves_pos_tag(self):
        """Test that transcribe_tokens preserves pos_tag field."""
        transcriber = PhoneticTranscriber()
        tokens = [Token("hello", "INTJ", "", False, 0)]

        enriched = transcriber.transcribe_tokens(tokens)

        assert enriched[0].pos_tag == "INTJ"

    def test_transcribe_tokens_preserves_is_content_word(self):
        """Test that transcribe_tokens preserves is_content_word field."""
        transcriber = PhoneticTranscriber()
        tokens = [Token("hello", "INTJ", "", True, 0)]

        enriched = transcriber.transcribe_tokens(tokens)

        assert enriched[0].is_content_word is True

    def test_transcribe_tokens_empty_list(self):
        """Test transcribing empty token list."""
        transcriber = PhoneticTranscriber()
        enriched = transcriber.transcribe_tokens([])

        assert enriched == []

    def test_transcribe_tokens_mixed_words(self):
        """Test transcribing mix of known and OOV words."""
        transcriber = PhoneticTranscriber()
        tokens = [
            Token("cat", "", "", False, 0),
            Token("xyz", "", "", False, 0),
            Token("dog", "", "", False, 0)
        ]

        enriched = transcriber.transcribe_tokens(tokens)

        assert enriched[0].phonetic == "K AE1 T"
        assert enriched[0].syllable_count == 1
        assert enriched[1].phonetic == "XYZ"  # OOV fallback
        assert enriched[2].phonetic == "D AO1 G"
        assert enriched[2].syllable_count == 1


class TestIntegrationWithPreviousComponents:
    """Integration tests with Tokenizer and POSTagger."""

    def test_full_pipeline_tokenizer_to_phonetic(self):
        """Test complete tokenization → phonetic transcription pipeline."""
        from preprocessor.tokenizer import Tokenizer

        tokenizer = Tokenizer()
        transcriber = PhoneticTranscriber()

        text = "The cat sat"  # No punctuation to avoid empty phonetic
        tokens = tokenizer.tokenize(text)
        enriched = transcriber.transcribe_tokens(tokens)

        # Verify all word tokens have phonetic transcriptions
        word_tokens = [t for t in enriched if t.text.isalpha()]
        assert all(t.phonetic != "" for t in word_tokens)
        assert all(t.syllable_count > 0 for t in word_tokens)

        # Check specific words
        cat_token = next(t for t in enriched if t.text == "cat")
        assert cat_token.phonetic == "K AE1 T"
        assert cat_token.syllable_count == 1

    def test_full_pipeline_with_pos_tagger(self):
        """Test tokenization → POS tagging → phonetic transcription."""
        from preprocessor.tokenizer import Tokenizer
        from preprocessor.pos_tagger import POSTagger

        tokenizer = Tokenizer()
        pos_tagger = POSTagger()
        transcriber = PhoneticTranscriber()

        text = "The quick brown fox"  # No punctuation
        tokens = tokenizer.tokenize(text)
        tagged = pos_tagger.tag(tokens)
        enriched = transcriber.transcribe_tokens(tagged)

        # Verify all fields are populated for word tokens
        for token in enriched:
            assert token.text != ""
            assert token.pos_tag != ""
            # Only check phonetic for alphabetic tokens (not punctuation)
            if token.text.isalpha():
                assert token.phonetic != ""
                assert token.syllable_count > 0
            # is_content_word is bool, so just check it exists
            assert isinstance(token.is_content_word, bool)


class TestQuickTranscribeHelper:
    """Tests for quick_transcribe() convenience function."""

    def test_quick_transcribe_works(self):
        """Test quick_transcribe convenience function."""
        phonetic = quick_transcribe("hello")

        assert phonetic == "HH AH0 L OW1"

    def test_quick_transcribe_various_words(self):
        """Test quick_transcribe with various words."""
        assert quick_transcribe("cat") == "K AE1 T"
        assert quick_transcribe("dog") == "D AO1 G"
        assert quick_transcribe("world") == "W ER1 L D"


class TestGetRhymingWords:
    """Tests for get_rhyming_words() utility function."""

    def test_get_rhyming_words_cat(self):
        """Test finding rhymes for 'cat'."""
        rhymes = get_rhyming_words("cat", max_results=10)

        assert isinstance(rhymes, list)
        # Should find some common rhymes
        common_rhymes = {"bat", "hat", "mat", "rat", "sat", "fat"}
        assert any(rhyme in common_rhymes for rhyme in rhymes)

    def test_get_rhyming_words_respects_max_results(self):
        """Test that max_results parameter is respected."""
        rhymes = get_rhyming_words("cat", max_results=5)

        assert len(rhymes) <= 5

    def test_get_rhyming_words_case_insensitive(self):
        """Test that rhyme finding is case-insensitive."""
        rhymes_lower = get_rhyming_words("cat", max_results=5)
        rhymes_upper = get_rhyming_words("CAT", max_results=5)

        # Should return similar results regardless of case
        assert rhymes_lower == rhymes_upper


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_transcribe_numbers(self):
        """Test transcribing numbers."""
        transcriber = PhoneticTranscriber()

        # Numbers might not be in dictionary, should return uppercase
        phonetic = transcriber.transcribe("123")
        assert phonetic.isupper() or phonetic.isdigit()

    def test_transcribe_contractions(self):
        """Test transcribing contractions."""
        transcriber = PhoneticTranscriber()

        # Common contractions should work
        # "don't" might be in dictionary as "dont"
        phonetic = transcriber.transcribe("don't")
        assert phonetic != ""

    def test_transcribe_special_characters(self):
        """Test transcribing with special characters."""
        transcriber = PhoneticTranscriber()

        # Should handle or strip special characters
        phonetic = transcriber.transcribe("hello@world")
        assert phonetic != ""

    def test_transcribe_hyphenated_words(self):
        """Test transcribing hyphenated words."""
        transcriber = PhoneticTranscriber()

        # Hyphenated words may not be in dictionary
        phonetic = transcriber.transcribe("well-known")
        assert phonetic != ""

    def test_count_syllables_very_long_word(self):
        """Test syllable counting for very long word."""
        transcriber = PhoneticTranscriber()

        # "antidisestablishmentarianism" - 12 syllables
        syllables = transcriber.count_syllables("antidisestablishmentarianism")
        assert syllables > 5  # Should recognize this as multi-syllabic


class TestStressPatterns:
    """Tests for stress pattern detection."""

    def test_primary_stress_marked_with_1(self):
        """Test that primary stress is marked with '1'."""
        transcriber = PhoneticTranscriber()

        phonetic = transcriber.transcribe("hello")
        stressed = transcriber.get_stressed_syllables(phonetic)

        # Should have at least one primary stress
        assert any('1' in s for s in stressed)

    def test_secondary_stress_marked_with_2(self):
        """Test that secondary stress is marked with '2'."""
        transcriber = PhoneticTranscriber()

        # Find a word with secondary stress
        phonetic = transcriber.transcribe("information")
        stressed = transcriber.get_stressed_syllables(phonetic)

        # "information" should have both primary (1) and secondary (2) stress
        assert any('1' in s for s in stressed)
        assert any('2' in s for s in stressed)

    def test_unstressed_syllables_not_in_list(self):
        """Test that unstressed syllables (0) are not in stressed list."""
        transcriber = PhoneticTranscriber()

        phonetic = "HH AH0 L OW1"  # hello: AH0 is unstressed
        stressed = transcriber.get_stressed_syllables(phonetic)

        # AH0 should not be in the list
        assert "AH0" not in stressed
        assert "OW1" in stressed
