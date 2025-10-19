"""
Integration tests using real data to catch preprocessing bugs.

Purpose:
    Prevent regressions like the POSTagger alignment bug that caused:
    - 89.5% empty POS tags
    - 4.9% content word rate (should be 30-70%)
    - False negatives (AI text scoring 0.0381 instead of 0.25-0.50)

These tests use sample.txt (real AI-generated text) to validate the
entire preprocessing pipeline on realistic data, not just mocked data.

Session 6 Bug Fix:
    Fixed POSTagger alignment by running DependencyParser first and
    passing its spaCy doc to POSTagger, ensuring perfect alignment.
"""

import pytest
from pathlib import Path
from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.pipeline import EchoAnalysisEngine
from specHO.scoring.weighted_scorer import WeightedScorer
from specHO.scoring.aggregator import DocumentAggregator


# Load sample.txt once for all tests
@pytest.fixture(scope="module")
def sample_text():
    """Load the sample AI-generated text."""
    sample_path = Path(__file__).parent.parent / "specHO" / "sample.txt"
    if not sample_path.exists():
        pytest.skip(f"sample.txt not found at {sample_path}")
    return sample_path.read_text(encoding='utf-8')


class TestPreprocessingRealData:
    """Test preprocessing on real data to catch alignment bugs."""

    def test_content_word_rate_realistic(self, sample_text):
        """Content word rate should be 30-70% on real text (Session 6 bug test)."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process(sample_text)

        content_words = [t for t in tokens if t.is_content_word]
        content_rate = len(content_words) / len(tokens) if tokens else 0

        # Session 6 bug: was 4.9% (should be 30-70%)
        assert 0.30 <= content_rate <= 0.70, (
            f"Content word rate {content_rate*100:.1f}% is outside 30-70% range. "
            f"This indicates POSTagger alignment failure (Session 6 bug)."
        )

    def test_field_population_high(self, sample_text):
        """Most tokens should have all fields populated (Session 6 bug test)."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process(sample_text)

        fully_populated = [
            t for t in tokens
            if t.pos_tag and t.phonetic and t.is_content_word is not None
        ]
        population_rate = len(fully_populated) / len(tokens) if tokens else 0

        # Session 6 bug: was 9.1% (should be >80%)
        assert population_rate > 0.80, (
            f"Field population {population_rate*100:.1f}% is below 80%. "
            f"This indicates tokens not being properly enriched."
        )

    def test_no_empty_pos_tags_majority(self, sample_text):
        """POS tags should be populated for most tokens (Session 6 bug test)."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process(sample_text)

        empty_pos = [t for t in tokens if not t.pos_tag or t.pos_tag == ""]
        empty_rate = len(empty_pos) / len(tokens) if tokens else 0

        # Session 6 bug: 89.5% had empty POS tags (should be <20%)
        assert empty_rate < 0.20, (
            f"Empty POS tag rate {empty_rate*100:.1f}% is above 20%. "
            f"Session 6 bug: POSTagger was failing to tag most tokens."
        )

    def test_pos_distribution_realistic(self, sample_text):
        """POS tag distribution should be realistic for English text."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process(sample_text)

        from collections import Counter
        pos_counts = Counter(t.pos_tag for t in tokens if t.pos_tag)

        # Nouns and verbs should be the most common content words
        noun_count = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)
        verb_count = pos_counts.get("VERB", 0)

        noun_rate = noun_count / len(tokens) if tokens else 0
        verb_rate = verb_count / len(tokens) if tokens else 0

        # Expect 15-25% nouns and 10-20% verbs in typical text
        assert 0.10 <= noun_rate <= 0.30, f"Noun rate {noun_rate*100:.1f}% is unrealistic"
        assert 0.05 <= verb_rate <= 0.25, f"Verb rate {verb_rate*100:.1f}% is unrealistic"

    def test_token_doc_alignment(self, sample_text):
        """Custom tokens and spaCy doc should have similar lengths."""
        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process(sample_text)

        # Allow small differences due to tokenization variations
        diff = abs(len(tokens) - len(doc))
        diff_percent = diff / len(tokens) if tokens else 0

        assert diff_percent < 0.05, (
            f"Token/doc mismatch: {len(tokens)} tokens vs {len(doc)} doc tokens "
            f"({diff_percent*100:.1f}% difference). Indicates alignment issues."
        )


class TestFullPipelineRealData:
    """Test full detection pipeline on real data."""

    def test_unwatermarked_ai_detection(self, sample_text):
        """Sample.txt should score 0.25-0.50 (unwatermarked AI range)."""
        # Run full pipeline
        preprocessor = LinguisticPreprocessor()
        clause_identifier = ClauseIdentifier()
        echo_engine = EchoAnalysisEngine()
        scorer = WeightedScorer()
        aggregator = DocumentAggregator()

        # Process
        tokens, doc = preprocessor.process(sample_text)
        pairs = clause_identifier.identify_pairs(tokens, doc)
        echo_scores = [echo_engine.analyze_pair(pair) for pair in pairs]
        pair_scores = [scorer.calculate_pair_score(es) for es in echo_scores]
        document_score = aggregator.aggregate_scores(pair_scores)

        # Session 6 bug: scored 0.0381 (should be 0.25-0.50)
        assert 0.25 <= document_score <= 0.50, (
            f"Document score {document_score:.4f} is outside 0.25-0.50 range. "
            f"Expected UNWATERMARKED_AI classification. "
            f"Session 6 bug: preprocessing failure caused score of 0.0381."
        )

    def test_sufficient_clause_pairs_found(self, sample_text):
        """Should find many clause pairs in real text (not just a few)."""
        preprocessor = LinguisticPreprocessor()
        clause_identifier = ClauseIdentifier()

        tokens, doc = preprocessor.process(sample_text)
        pairs = clause_identifier.identify_pairs(tokens, doc)

        # sample.txt is ~4000 words, should have 200+ clause pairs
        assert len(pairs) > 200, (
            f"Only found {len(pairs)} clause pairs. Expected 200+ for ~4000 word text. "
            f"May indicate clause detection or preprocessing issues."
        )

    def test_zones_have_sufficient_content_words(self, sample_text):
        """Zones should contain enough content words for reliable comparison."""
        preprocessor = LinguisticPreprocessor()
        clause_identifier = ClauseIdentifier()

        tokens, doc = preprocessor.process(sample_text)
        pairs = clause_identifier.identify_pairs(tokens, doc)

        # Check zone sizes
        zone_sizes = []
        for pair in pairs:
            if pair.zone_a_tokens:
                zone_sizes.append(len(pair.zone_a_tokens))
            if pair.zone_b_tokens:
                zone_sizes.append(len(pair.zone_b_tokens))

        avg_zone_size = sum(zone_sizes) / len(zone_sizes) if zone_sizes else 0

        # Average zone should be ~3 tokens (configured window size)
        assert avg_zone_size >= 2.0, (
            f"Average zone size {avg_zone_size:.1f} is too small. "
            f"Expected ~3 tokens. May indicate content word identification issues."
        )

    def test_echo_scores_reasonable_distribution(self, sample_text):
        """Echo scores should have reasonable variance (not all zeros)."""
        preprocessor = LinguisticPreprocessor()
        clause_identifier = ClauseIdentifier()
        echo_engine = EchoAnalysisEngine()

        tokens, doc = preprocessor.process(sample_text)
        pairs = clause_identifier.identify_pairs(tokens, doc)
        echo_scores = [echo_engine.analyze_pair(pair) for pair in pairs[:100]]  # Sample 100

        # Extract phonetic scores
        phonetic_scores = [es.phonetic_score for es in echo_scores if es.phonetic_score is not None]

        # Should have variance, not all zeros
        avg_phonetic = sum(phonetic_scores) / len(phonetic_scores) if phonetic_scores else 0

        assert avg_phonetic > 0.05, (
            f"Average phonetic score {avg_phonetic:.4f} is too low. "
            f"Session 6 bug: preprocessing failure led to artificially low scores."
        )


class TestRegressionPrevention:
    """Tests specifically designed to catch future regressions."""

    def test_pos_tagger_accepts_spacy_doc(self):
        """POSTagger should accept optional spacy_doc parameter (Session 6 fix)."""
        from specHO.preprocessor.pos_tagger import POSTagger
        from specHO.preprocessor.tokenizer import Tokenizer
        from specHO.preprocessor.dependency_parser import DependencyParser

        text = "The cat sat on the mat."
        tokenizer = Tokenizer()
        pos_tagger = POSTagger()
        dep_parser = DependencyParser()

        tokens = tokenizer.tokenize(text)
        doc = dep_parser.parse(text)

        # This should work without errors (Session 6 fix)
        tagged_tokens = pos_tagger.tag(tokens, spacy_doc=doc)

        assert all(t.pos_tag for t in tagged_tokens), "All tokens should have POS tags"
        assert len(tagged_tokens) == len(tokens), "Token count should be preserved"

    def test_preprocessor_uses_shared_spacy_doc(self, sample_text):
        """LinguisticPreprocessor should use shared spaCy doc (Session 6 fix)."""
        from specHO.preprocessor.pipeline import LinguisticPreprocessor

        preprocessor = LinguisticPreprocessor()
        tokens, doc = preprocessor.process(sample_text)

        # The dependency doc should be created FIRST, then used for POS tagging
        # This ensures perfect alignment (Session 6 fix)

        # Verify most tokens have POS tags (proves shared doc works)
        with_pos = [t for t in tokens if t.pos_tag and t.pos_tag != ""]
        pos_rate = len(with_pos) / len(tokens) if tokens else 0

        assert pos_rate > 0.90, (
            f"POS tag population {pos_rate*100:.1f}% is below 90%. "
            f"Shared spaCy doc fix may not be working correctly."
        )


class TestPerformanceRealData:
    """Performance tests on real data."""

    def test_preprocessing_completes_quickly(self, sample_text):
        """Preprocessing should complete in reasonable time."""
        import time
        preprocessor = LinguisticPreprocessor()

        start = time.time()
        tokens, doc = preprocessor.process(sample_text)
        duration = time.time() - start

        # ~4000 words should process in <5 seconds
        assert duration < 5.0, (
            f"Preprocessing took {duration:.2f}s for ~4000 words. "
            f"Expected <5s. Performance regression detected."
        )

    def test_full_pipeline_throughput(self, sample_text):
        """Full pipeline should maintain good throughput."""
        import time
        preprocessor = LinguisticPreprocessor()
        clause_identifier = ClauseIdentifier()
        echo_engine = EchoAnalysisEngine()
        scorer = WeightedScorer()
        aggregator = DocumentAggregator()

        start = time.time()
        tokens, doc = preprocessor.process(sample_text)
        pairs = clause_identifier.identify_pairs(tokens, doc)
        echo_scores = [echo_engine.analyze_pair(pair) for pair in pairs]
        pair_scores = [scorer.calculate_pair_score(es) for es in echo_scores]
        document_score = aggregator.aggregate_scores(pair_scores)
        duration = time.time() - start

        word_count = len(sample_text.split())
        throughput = word_count / duration

        # Should process >1000 words/second
        assert throughput > 1000, (
            f"Throughput {throughput:.1f} words/second is below 1000. "
            f"Performance regression detected."
        )
