"""Tests for LinguisticPreprocessor Pipeline (Task 2.5).

Tests the orchestrator that chains all preprocessor components together.
This is a critical integration point that validates data flow through
the entire preprocessing pipeline.

Tier: 1 (MVP)
Coverage: Component integration, data flow, field population, output validation
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from preprocessor.pipeline import LinguisticPreprocessor, quick_process
from models import Token


class TestLinguisticPreprocessorBasic:
    """Basic tests for LinguisticPreprocessor class."""

    def test_linguistic_preprocessor_initialization(self):
        """Test LinguisticPreprocessor can be initialized."""
        preprocessor = LinguisticPreprocessor()
        assert preprocessor is not None
        assert preprocessor.tokenizer is not None
        assert preprocessor.pos_tagger is not None
        assert preprocessor.dependency_parser is not None
        assert preprocessor.phonetic_transcriber is not None

    def test_subcomponents_initialized(self):
        """Test that all subcomponents are properly initialized."""
        preprocessor = LinguisticPreprocessor()

        # Verify each component has expected attributes
        assert hasattr(preprocessor.tokenizer, 'nlp')
        assert hasattr(preprocessor.pos_tagger, 'nlp')
        assert hasattr(preprocessor.dependency_parser, 'nlp')
        # PhoneticTranscriber doesn't need nlp attribute


class TestProcess:
    """Tests for process() method."""

    def test_process_simple_sentence(self):
        """Test processing simple sentence."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat sat.")

        assert len(tokens) > 0
        assert doc is not None
        assert len(list(doc.sents)) == 1

    def test_process_returns_tuple(self):
        """Test that process returns (tokens, doc) tuple."""
        preprocessor = LinguisticPreprocessor()
        result = preprocessor.process("Test sentence.")

        assert isinstance(result, tuple)
        assert len(result) == 2
        tokens, doc = result
        assert isinstance(tokens, list)

    def test_process_populates_all_token_fields(self):
        """Test that all Token fields are populated by the pipeline."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat sat.")

        # Check a content word (cat)
        cat_token = next(t for t in tokens if t.text == "cat")

        # All fields should be populated
        assert cat_token.text == "cat"
        assert cat_token.pos_tag == "NOUN"
        assert cat_token.phonetic == "K AE1 T"
        assert cat_token.is_content_word is True
        assert cat_token.syllable_count == 1

    def test_process_identifies_content_words(self):
        """Test that content words are correctly identified."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The quick brown fox.")

        content_words = [t for t in tokens if t.is_content_word]
        content_texts = [t.text for t in content_words]

        # Should identify nouns, adjectives
        assert "quick" in content_texts or "brown" in content_texts
        assert "fox" in content_texts

        # Should NOT identify articles as content words
        assert "The" not in content_texts

    def test_process_creates_dependency_parse(self):
        """Test that dependency parse is created."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat sat on the mat.")

        # Doc should have dependency information
        assert len(doc) > 0
        assert all(hasattr(token, 'dep_') for token in doc)
        assert all(hasattr(token, 'head') for token in doc)

    def test_process_multiple_sentences(self):
        """Test processing text with multiple sentences."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("First sentence. Second sentence.")

        # Should have 2 sentences
        sentences = list(doc.sents)
        assert len(sentences) == 2

        # Should have tokens from both sentences
        assert len(tokens) > 4  # At least 2 words per sentence + punctuation

    def test_process_empty_string(self):
        """Test processing empty string."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("")

        assert tokens == []
        assert len(doc) == 0

    def test_process_whitespace_only(self):
        """Test processing whitespace-only string."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("   \n\t  ")

        assert tokens == []
        assert len(doc) == 0


class TestDataFlow:
    """Tests for data flow through the pipeline."""

    def test_sequential_enrichment(self):
        """Test that each component enriches tokens sequentially."""
        preprocessor = LinguisticPreprocessor()

        # After tokenization (step 1): only text is populated
        tokens_step1 = preprocessor.tokenizer.tokenize("Hello world")
        assert all(t.text != "" for t in tokens_step1)
        assert all(t.pos_tag == "" for t in tokens_step1)
        assert all(t.phonetic == "" for t in tokens_step1)

        # After POS tagging (step 2): text + pos_tag + is_content_word
        tokens_step2 = preprocessor.pos_tagger.tag(tokens_step1)
        assert all(t.pos_tag != "" for t in tokens_step2)
        # phonetic still empty at this stage
        assert all(t.phonetic == "" for t in tokens_step2)

        # After phonetic (step 3): all fields populated
        tokens_step3 = preprocessor.phonetic_transcriber.transcribe_tokens(tokens_step2)
        word_tokens = [t for t in tokens_step3 if t.text.isalpha()]
        assert all(t.phonetic != "" for t in word_tokens)
        assert all(t.syllable_count > 0 for t in word_tokens)

    def test_complete_pipeline_integration(self):
        """Test complete pipeline produces fully enriched output."""
        preprocessor = LinguisticPreprocessor()
        text = "The quick brown fox jumps over the lazy dog."
        tokens, doc = preprocessor.process(text)

        # Every alphabetic token should have all fields
        for token in tokens:
            if token.text.isalpha():
                assert token.text != ""
                assert token.pos_tag != ""
                assert token.phonetic != ""
                assert isinstance(token.is_content_word, bool)
                assert token.syllable_count > 0

        # Dependency parse should cover same text
        assert doc.text == text

    def test_token_count_matches_reasonable_expectation(self):
        """Test that token count is reasonable for input."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("One two three four five.")

        # Should have 6 tokens (5 words + 1 punctuation)
        assert len(tokens) == 6

        # spaCy doc should have same token count
        assert len(doc) == 6


class TestContentWordFiltering:
    """Tests for content word identification throughout pipeline."""

    def test_content_word_ratio_reasonable(self):
        """Test that content word ratio is within expected range."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The quick brown fox jumps over the lazy dog.")

        content_words = sum(1 for t in tokens if t.is_content_word)
        total_words = len(tokens)

        # Expect 30-70% content words in normal English
        content_ratio = content_words / total_words
        assert 0.3 <= content_ratio <= 0.7

    def test_function_words_not_marked_as_content(self):
        """Test that function words are not marked as content words."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat sat on the mat.")

        # Find function words
        the_tokens = [t for t in tokens if t.text.lower() == "the"]
        on_token = next((t for t in tokens if t.text == "on"), None)

        # Function words should NOT be content words
        assert all(not t.is_content_word for t in the_tokens)
        if on_token:
            assert on_token.is_content_word is False

    def test_nouns_marked_as_content(self):
        """Test that nouns are marked as content words."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat and dog.")

        cat_token = next(t for t in tokens if t.text == "cat")
        dog_token = next(t for t in tokens if t.text == "dog")

        assert cat_token.is_content_word is True
        assert dog_token.is_content_word is True

    def test_verbs_marked_as_content(self):
        """Test that verbs are marked as content words."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat runs and jumps.")

        runs_token = next(t for t in tokens if t.text == "runs")
        jumps_token = next(t for t in tokens if t.text == "jumps")

        assert runs_token.is_content_word is True
        assert jumps_token.is_content_word is True


class TestPhoneticEnrichment:
    """Tests for phonetic transcription in the pipeline."""

    def test_phonetic_fields_populated(self):
        """Test that phonetic fields are populated for all words."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("Hello world.")

        word_tokens = [t for t in tokens if t.text.isalpha()]

        # All word tokens should have phonetic transcriptions
        assert all(t.phonetic != "" for t in word_tokens)

        # All word tokens should have syllable counts > 0
        assert all(t.syllable_count > 0 for t in word_tokens)

    def test_phonetic_transcriptions_accurate(self):
        """Test that phonetic transcriptions are accurate."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("cat dog hello")

        cat_token = next(t for t in tokens if t.text == "cat")
        dog_token = next(t for t in tokens if t.text == "dog")
        hello_token = next(t for t in tokens if t.text == "hello")

        assert cat_token.phonetic == "K AE1 T"
        assert dog_token.phonetic == "D AO1 G"
        assert hello_token.phonetic == "HH AH0 L OW1"

    def test_syllable_counts_accurate(self):
        """Test that syllable counts are accurate."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("cat beautiful information")

        cat_token = next(t for t in tokens if t.text == "cat")
        beautiful_token = next(t for t in tokens if t.text == "beautiful")
        information_token = next(t for t in tokens if t.text == "information")

        assert cat_token.syllable_count == 1
        assert beautiful_token.syllable_count == 3
        assert information_token.syllable_count == 4


class TestDependencyParse:
    """Tests for dependency parse output."""

    def test_dependency_parse_structure(self):
        """Test that dependency parse has expected structure."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat sat.")

        # Should find ROOT verb
        root_tokens = [t for t in doc if t.dep_ == "ROOT"]
        assert len(root_tokens) == 1
        assert root_tokens[0].text == "sat"

    def test_dependency_parse_complex_sentence(self):
        """Test dependency parse on complex sentence."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The cat sat, and the dog ran.")

        # Should find coordination
        conj_tokens = [t for t in doc if t.dep_ == "conj"]
        assert len(conj_tokens) >= 1

    def test_dependency_parse_subordinate_clause(self):
        """Test dependency parse identifies subordinate clauses."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("She said that he left.")

        # Should have clausal relationships
        deps = [t.dep_ for t in doc]
        # May have ccomp or other clause-indicating dependencies
        assert any(dep in ["ccomp", "advcl", "acl"] for dep in deps)


class TestBatchProcessing:
    """Tests for process_batch() method."""

    def test_process_batch_multiple_texts(self):
        """Test batch processing of multiple texts."""
        preprocessor = LinguisticPreprocessor()
        texts = ["First text.", "Second text.", "Third text."]

        results = preprocessor.process_batch(texts)

        assert len(results) == 3
        assert all(isinstance(r, tuple) for r in results)
        assert all(len(r) == 2 for r in results)

    def test_process_batch_empty_list(self):
        """Test batch processing with empty list."""
        preprocessor = LinguisticPreprocessor()
        results = preprocessor.process_batch([])

        assert results == []

    def test_process_batch_single_text(self):
        """Test batch processing with single text."""
        preprocessor = LinguisticPreprocessor()
        results = preprocessor.process_batch(["Single text."])

        assert len(results) == 1
        tokens, doc = results[0]
        assert len(tokens) > 0


class TestGetTokenCount:
    """Tests for get_token_count() utility method."""

    def test_get_token_count_simple(self):
        """Test token counting."""
        preprocessor = LinguisticPreprocessor()
        count = preprocessor.get_token_count("Hello world!")

        assert count == 3  # Hello, world, !

    def test_get_token_count_empty(self):
        """Test token counting with empty string."""
        preprocessor = LinguisticPreprocessor()
        count = preprocessor.get_token_count("")

        assert count == 0

    def test_get_token_count_matches_process(self):
        """Test that get_token_count matches process() output."""
        preprocessor = LinguisticPreprocessor()
        text = "The quick brown fox."

        count = preprocessor.get_token_count(text)
        tokens, doc = preprocessor.process(text)

        assert count == len(tokens)


class TestQuickProcessHelper:
    """Tests for quick_process() convenience function."""

    def test_quick_process_works(self):
        """Test quick_process convenience function."""
        tokens, doc = quick_process("The cat sat.")

        assert len(tokens) > 0
        assert doc is not None

    def test_quick_process_returns_correct_types(self):
        """Test that quick_process returns correct types."""
        result = quick_process("Test sentence.")

        assert isinstance(result, tuple)
        assert len(result) == 2
        tokens, doc = result
        assert isinstance(tokens, list)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_process_very_long_sentence(self):
        """Test processing very long sentence."""
        preprocessor = LinguisticPreprocessor()
        long_text = " ".join(["word"] * 100) + "."

        tokens, doc = preprocessor.process(long_text)

        # Should handle long text without errors
        assert len(tokens) > 0

    def test_process_special_characters(self):
        """Test processing text with special characters."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("Hello! How are you? I'm fine.")

        # Should handle punctuation and contractions
        assert len(tokens) > 0

    def test_process_numbers(self):
        """Test processing text with numbers."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("The year 2025 begins soon.")

        # Should handle numbers
        assert len(tokens) > 0

    def test_process_mixed_case(self):
        """Test processing text with mixed case."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("HELLO world TeSt")

        # Should handle mixed case
        assert len(tokens) == 3

    def test_process_unicode(self):
        """Test processing text with unicode characters."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process("café résumé naïve")

        # Should handle unicode (may not transcribe perfectly but shouldn't crash)
        assert len(tokens) >= 3


class TestValidation:
    """Tests for _validate_output() internal method."""

    def test_validation_passes_for_good_output(self):
        """Test that validation passes for typical output."""
        preprocessor = LinguisticPreprocessor()

        # This should complete without raising exceptions
        tokens, doc = preprocessor.process("The quick brown fox jumps over the lazy dog.")

        # If we got here, validation passed
        assert True

    def test_validation_handles_empty_input(self):
        """Test that validation handles empty input gracefully."""
        preprocessor = LinguisticPreprocessor()

        # Should not raise exceptions for empty input
        tokens, doc = preprocessor.process("")

        assert tokens == []


class TestRealWorldTexts:
    """Tests with realistic text samples."""

    def test_process_short_paragraph(self):
        """Test processing a short paragraph."""
        preprocessor = LinguisticPreprocessor()
        text = ("The Echo Rule is a linguistic watermarking technique. "
                "It operates at the level of clause relationships. "
                "This creates a detectable signal in generated text.")

        tokens, doc = preprocessor.process(text)

        # Should handle multi-sentence text
        assert len(list(doc.sents)) == 3
        assert len(tokens) > 20

        # Should have mix of content and function words
        content_words = sum(1 for t in tokens if t.is_content_word)
        assert content_words > 10

    def test_process_complex_sentence_structure(self):
        """Test processing complex sentence with multiple clauses."""
        preprocessor = LinguisticPreprocessor()
        text = "Although the cat sat on the mat, the dog ran in the yard."

        tokens, doc = preprocessor.process(text)

        # Should handle complex structure
        assert len(tokens) > 10

        # Dependency parse should identify clause structure
        deps = [t.dep_ for t in doc]
        # Should have some subordinate clause markers
        assert len(set(deps)) > 5  # Various dependency types

    def test_process_with_conjunctions(self):
        """Test processing text with coordinating conjunctions."""
        preprocessor = LinguisticPreprocessor()
        text = "The cat sat, and the dog ran, but the bird flew."

        tokens, doc = preprocessor.process(text)

        # Should identify conjunctions
        and_token = next((t for t in tokens if t.text == "and"), None)
        but_token = next((t for t in tokens if t.text == "but"), None)

        assert and_token is not None
        assert but_token is not None

    def test_process_news_article_excerpt(self):
        """Test processing news article excerpt with formal style."""
        preprocessor = LinguisticPreprocessor()
        text = (
            "Scientists announced a breakthrough in renewable energy yesterday. "
            "The new solar panel design increases efficiency by 40 percent, and "
            "researchers believe it could revolutionize the industry. However, "
            "commercial production remains years away."
        )

        tokens, doc = preprocessor.process(text)

        # Should handle formal news style
        assert len(list(doc.sents)) == 3
        assert len(tokens) > 30

        # Should identify technical vocabulary as content words
        content_words = [t.text for t in tokens if t.is_content_word]
        assert any(word in ["Scientists", "breakthrough", "efficiency"] for word in content_words)

        # Verify phonetic transcription for technical terms
        technical_tokens = [t for t in tokens if t.text.lower() in ["solar", "researchers", "production"]]
        assert all(t.phonetic != "" for t in technical_tokens)

    def test_process_conversational_text(self):
        """Test processing informal conversational text with contractions."""
        preprocessor = LinguisticPreprocessor()
        text = (
            "I don't think we're ready yet. She's been working on this for months, "
            "but there's still more to do. Won't you help us finish?"
        )

        tokens, doc = preprocessor.process(text)

        # Should handle contractions
        assert len(list(doc.sents)) == 3

        # Contractions should be split and processed
        contraction_bases = [t.text for t in tokens if t.text in ["do", "are", "has", "is", "will"]]
        assert len(contraction_bases) > 0

        # Personal pronouns should be identified as function words
        pronouns = [t for t in tokens if t.text.lower() in ["i", "we", "she", "us", "you"]]
        assert all(not t.is_content_word for t in pronouns)

    def test_process_literary_excerpt(self):
        """Test processing literary text with descriptive language."""
        preprocessor = LinguisticPreprocessor()
        text = (
            "The ancient oak stood silent beneath the pale moonlight; "
            "its gnarled branches reached skyward like desperate fingers. "
            "A cool wind whispered through the leaves, carrying secrets "
            "from distant shores."
        )

        tokens, doc = preprocessor.process(text)

        # Should handle complex descriptive sentences
        # Note: semicolon doesn't create new sentence in spaCy
        assert len(list(doc.sents)) == 2
        assert len(tokens) > 30  # Rich descriptive text

        # Rich vocabulary should be mostly content words
        content_ratio = sum(1 for t in tokens if t.is_content_word) / len(tokens)
        assert content_ratio > 0.4  # Literary text is content-rich

        # Verify adjectives are identified as content words
        adjectives = [t for t in tokens if t.pos_tag == "ADJ"]
        assert len(adjectives) >= 4  # ancient, pale, gnarled, cool, distant
        assert all(t.is_content_word for t in adjectives)

    def test_process_technical_documentation(self):
        """Test processing technical documentation with specialized terms."""
        preprocessor = LinguisticPreprocessor()
        text = (
            "The API endpoint accepts JSON payloads with authentication tokens. "
            "Configure the middleware by setting environment variables in the "
            "deployment configuration file. The system validates all requests "
            "before processing."
        )

        tokens, doc = preprocessor.process(text)

        # Should handle technical jargon
        assert len(list(doc.sents)) == 3

        # Technical terms should have fallback phonetic handling
        technical_terms = [t for t in tokens if t.text in ["API", "JSON", "middleware"]]
        assert all(t.phonetic != "" for t in technical_terms)  # Uppercase fallback

        # Should identify technical nouns as content words
        tech_content = [t for t in tokens if t.text.lower() in ["endpoint", "payloads", "tokens", "variables"]]
        assert all(t.is_content_word for t in tech_content)

    def test_process_academic_writing(self):
        """Test processing academic writing with complex structures."""
        preprocessor = LinguisticPreprocessor()
        text = (
            "While previous research has focused on individual components, "
            "this study examines the interaction between multiple factors. "
            "The findings suggest that contextual variables significantly "
            "influence outcomes; therefore, future work should consider "
            "holistic approaches."
        )

        tokens, doc = preprocessor.process(text)

        # Should handle academic sentence structure
        # Note: semicolon doesn't create new sentence in spaCy
        assert len(list(doc.sents)) == 2

        # Should identify subordinate clauses
        deps = [t.dep_ for t in doc]
        assert any(dep in ["mark", "advcl", "ccomp"] for dep in deps)

        # Academic vocabulary should be processed correctly
        academic_terms = [t for t in tokens if t.text.lower() in
                         ["research", "study", "findings", "variables", "outcomes"]]
        assert all(t.is_content_word for t in academic_terms)
        assert all(t.syllable_count >= 2 for t in academic_terms)  # Multisyllabic

    def test_process_dialogue_with_quotations(self):
        """Test processing text with dialogue and quotation marks."""
        preprocessor = LinguisticPreprocessor()
        text = (
            '"We need to finish this today," she said firmly. '
            'He replied, "I understand, but we should test it thoroughly first." '
            'The manager nodded and said, "Let\'s compromise—finish the core, '
            'then test tomorrow."'
        )

        tokens, doc = preprocessor.process(text)

        # Should handle quotation marks and dialogue
        assert len(list(doc.sents)) == 3

        # Dialogue verbs should be identified
        dialogue_verbs = [t for t in tokens if t.text.lower() in ["said", "replied", "nodded"]]
        assert len(dialogue_verbs) >= 2
        assert all(t.is_content_word for t in dialogue_verbs)

        # Should handle mixed punctuation (em dashes, apostrophes)
        assert len(tokens) > 30

        # Verify proper handling of contractions in dialogue
        lets_token = next((t for t in tokens if t.text.lower() == "let"), None)
        assert lets_token is not None  # "Let's" should be split
