"""Tests for SemanticEchoAnalyzer (Task 4.3).

Tests the semantic similarity analysis using word embeddings.
Covers Tier 1 implementation with mock embeddings and edge cases.

Tier: 1
Task: 8.3 (partial - semantic analyzer tests only)
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from specHO.models import Token
from specHO.echo_engine.semantic_analyzer import SemanticEchoAnalyzer


# Test Fixtures
@pytest.fixture
def mock_embedding_model():
    """Mock gensim KeyedVectors model with sample embeddings."""
    model = Mock()

    # Create deterministic embeddings for test words
    embeddings = {
        'cat': np.array([1.0, 0.0, 0.0]),  # orthogonal to dog
        'dog': np.array([0.0, 1.0, 0.0]),  # orthogonal to cat
        'puppy': np.array([0.0, 0.9, 0.1]),  # similar to dog
        'kitten': np.array([0.9, 0.0, 0.1]),  # similar to cat
        'hello': np.array([0.5, 0.5, 0.0]),
        'world': np.array([0.5, 0.5, 0.0]),
        'run': np.array([1.0, 0.0, 0.0]),
        'walk': np.array([0.8, 0.2, 0.0]),  # similar to run
    }

    # Mock __contains__ and __getitem__
    model.__contains__ = lambda self, word: word in embeddings
    model.__getitem__ = lambda self, word: embeddings[word]

    return model


@pytest.fixture
def sample_zone_synonyms():
    """Sample zones with semantically similar words."""
    zone_a = [
        Token("dog", "NOUN", "D AO1 G", True, 1),
        Token("runs", "VERB", "R AH1 N Z", True, 1),
    ]
    zone_b = [
        Token("puppy", "NOUN", "P AH1 P IY0", True, 2),
        Token("walks", "VERB", "W AO1 K S", True, 1),
    ]
    return zone_a, zone_b


@pytest.fixture
def sample_zone_unrelated():
    """Sample zones with semantically unrelated words."""
    zone_a = [
        Token("cat", "NOUN", "K AE1 T", True, 1),
    ]
    zone_b = [
        Token("dog", "NOUN", "D AO1 G", True, 1),
    ]
    return zone_a, zone_b


@pytest.fixture
def sample_zone_identical():
    """Sample zones with identical words."""
    zone_a = [
        Token("hello", "INTJ", "HH AH0 L OW1", True, 2),
        Token("world", "NOUN", "W ER1 L D", True, 1),
    ]
    zone_b = [
        Token("hello", "INTJ", "HH AH0 L OW1", True, 2),
        Token("world", "NOUN", "W ER1 L D", True, 1),
    ]
    return zone_a, zone_b


class TestSemanticAnalyzerInitialization:
    """Test SemanticEchoAnalyzer initialization and model loading."""

    def test_initialization_without_model(self):
        """Test initialization in fallback mode (no model)."""
        analyzer = SemanticEchoAnalyzer()
        assert analyzer.model is None

    def test_initialization_with_invalid_path(self):
        """Test initialization with non-existent model path."""
        analyzer = SemanticEchoAnalyzer(model_path="nonexistent.bin")
        assert analyzer.model is None  # Should gracefully fall back


class TestSemanticAnalysisWithMockModel:
    """Test semantic analysis with mocked embeddings."""

    def test_analyze_synonyms(self, mock_embedding_model, sample_zone_synonyms):
        """Test analysis of semantically similar zones."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'
        analyzer.model_type = 'gensim'  # Required for mock model to work

        zone_a, zone_b = sample_zone_synonyms
        similarity = analyzer.analyze(zone_a, zone_b)

        # dog/puppy and run/walk are similar → high similarity expected
        assert 0.5 < similarity <= 1.0
        assert isinstance(similarity, (float, np.floating))

    def test_analyze_unrelated(self, mock_embedding_model, sample_zone_unrelated):
        """Test analysis of semantically unrelated zones."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a, zone_b = sample_zone_unrelated
        similarity = analyzer.analyze(zone_a, zone_b)

        # cat and dog are orthogonal in our mock → low similarity expected
        assert 0.0 <= similarity <= 0.5
        assert isinstance(similarity, (float, np.floating))

    def test_analyze_identical(self, mock_embedding_model, sample_zone_identical):
        """Test analysis of identical zones."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a, zone_b = sample_zone_identical
        similarity = analyzer.analyze(zone_a, zone_b)

        # Identical words → very high similarity (close to 1.0)
        assert similarity >= 0.9
        assert isinstance(similarity, (float, np.floating))


class TestSemanticAnalysisEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_zone_a(self, mock_embedding_model):
        """Test with empty zone A."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a = []
        zone_b = [Token("hello", "INTJ", "HH AH0 L OW1", True, 2)]

        similarity = analyzer.analyze(zone_a, zone_b)
        assert similarity == 0.0

    def test_empty_zone_b(self, mock_embedding_model):
        """Test with empty zone B."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a = [Token("hello", "INTJ", "HH AH0 L OW1", True, 2)]
        zone_b = []

        similarity = analyzer.analyze(zone_a, zone_b)
        assert similarity == 0.0

    def test_both_zones_empty(self, mock_embedding_model):
        """Test with both zones empty."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        similarity = analyzer.analyze([], [])
        assert similarity == 0.0

    def test_no_model_fallback(self):
        """Test fallback behavior when no model is available."""
        analyzer = SemanticEchoAnalyzer()
        # analyzer.model is None (no model loaded)

        zone_a = [Token("hello", "INTJ", "HH AH0 L OW1", True, 2)]
        zone_b = [Token("world", "NOUN", "W ER1 L D", True, 1)]

        similarity = analyzer.analyze(zone_a, zone_b)
        assert similarity == 0.5  # Neutral fallback

    def test_all_tokens_oov(self, mock_embedding_model):
        """Test when all tokens are out-of-vocabulary."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a = [Token("xyzabc", "NOUN", "X IH1 Z AE2 B K", True, 2)]
        zone_b = [Token("qwerty", "NOUN", "K W ER1 T IY0", True, 2)]

        similarity = analyzer.analyze(zone_a, zone_b)
        assert similarity == 0.5  # Fallback when no embeddings found

    def test_partial_oov(self, mock_embedding_model):
        """Test when some tokens are OOV (should use available tokens)."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a = [
            Token("cat", "NOUN", "K AE1 T", True, 1),
            Token("xyzabc", "NOUN", "X IH1 Z", True, 2),  # OOV
        ]
        zone_b = [
            Token("kitten", "NOUN", "K IH1 T AH0 N", True, 2),
            Token("qwerty", "NOUN", "K W ER1 T IY0", True, 2),  # OOV
        ]

        similarity = analyzer.analyze(zone_a, zone_b)
        # Should compute similarity using cat/kitten only
        assert 0.5 < similarity <= 1.0


class TestMeanPooling:
    """Test mean pooling implementation."""

    def test_get_zone_vector_single_token(self, mock_embedding_model):
        """Test zone vector for single token."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone = [Token("cat", "NOUN", "K AE1 T", True, 1)]
        vector = analyzer._get_zone_vector(zone)

        assert vector is not None
        assert isinstance(vector, np.ndarray)
        # Single token → vector equals embedding
        np.testing.assert_array_equal(vector, np.array([1.0, 0.0, 0.0]))

    def test_get_zone_vector_multiple_tokens(self, mock_embedding_model):
        """Test zone vector for multiple tokens (mean pooling)."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone = [
            Token("cat", "NOUN", "K AE1 T", True, 1),  # [1.0, 0.0, 0.0]
            Token("dog", "NOUN", "D AO1 G", True, 1),  # [0.0, 1.0, 0.0]
        ]
        vector = analyzer._get_zone_vector(zone)

        assert vector is not None
        # Mean of [1,0,0] and [0,1,0] = [0.5, 0.5, 0.0]
        expected = np.array([0.5, 0.5, 0.0])
        np.testing.assert_array_almost_equal(vector, expected)

    def test_get_zone_vector_all_oov(self, mock_embedding_model):
        """Test zone vector when all tokens are OOV."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone = [Token("xyzabc", "NOUN", "X IH1 Z", True, 2)]
        vector = analyzer._get_zone_vector(zone)

        assert vector is None  # No embeddings found

    def test_get_zone_vector_case_insensitive(self, mock_embedding_model):
        """Test that lookup is case-insensitive."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_lower = [Token("cat", "NOUN", "K AE1 T", True, 1)]
        zone_upper = [Token("CAT", "NOUN", "K AE1 T", True, 1)]

        vec_lower = analyzer._get_zone_vector(zone_lower)
        vec_upper = analyzer._get_zone_vector(zone_upper)

        np.testing.assert_array_equal(vec_lower, vec_upper)


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_cosine_similarity_identical_vectors(self):
        """Test cosine similarity of identical vectors."""
        analyzer = SemanticEchoAnalyzer()

        vec = np.array([1.0, 2.0, 3.0])
        similarity = analyzer._calculate_cosine_similarity(vec, vec)

        # Identical vectors → cosine = 1 → mapped to 1.0
        assert similarity == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self):
        """Test cosine similarity of orthogonal vectors."""
        analyzer = SemanticEchoAnalyzer()

        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([0.0, 1.0, 0.0])
        similarity = analyzer._calculate_cosine_similarity(vec_a, vec_b)

        # Orthogonal → cosine = 0 → mapped to 0.5
        assert similarity == pytest.approx(0.5)

    def test_cosine_similarity_opposite_vectors(self):
        """Test cosine similarity of opposite vectors."""
        analyzer = SemanticEchoAnalyzer()

        vec_a = np.array([1.0, 0.0, 0.0])
        vec_b = np.array([-1.0, 0.0, 0.0])
        similarity = analyzer._calculate_cosine_similarity(vec_a, vec_b)

        # Opposite → cosine = -1 → mapped to 0.0
        assert similarity == pytest.approx(0.0)

    def test_cosine_similarity_zero_vector_a(self):
        """Test handling of zero vector in vec_a."""
        analyzer = SemanticEchoAnalyzer()

        vec_a = np.array([0.0, 0.0, 0.0])
        vec_b = np.array([1.0, 2.0, 3.0])
        similarity = analyzer._calculate_cosine_similarity(vec_a, vec_b)

        # Undefined cosine → fallback to 0.5
        assert similarity == 0.5

    def test_cosine_similarity_zero_vector_b(self):
        """Test handling of zero vector in vec_b."""
        analyzer = SemanticEchoAnalyzer()

        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        similarity = analyzer._calculate_cosine_similarity(vec_a, vec_b)

        # Undefined cosine → fallback to 0.5
        assert similarity == 0.5

    def test_cosine_similarity_range_clipping(self):
        """Test that similarity is clipped to [0,1] range."""
        analyzer = SemanticEchoAnalyzer()

        # Various test vectors
        test_cases = [
            (np.array([1.0, 0.0]), np.array([1.0, 0.0])),  # identical
            (np.array([1.0, 0.0]), np.array([0.5, 0.5])),  # acute angle
            (np.array([1.0, 0.0]), np.array([0.0, 1.0])),  # orthogonal
            (np.array([1.0, 0.0]), np.array([-1.0, 0.0])),  # opposite
        ]

        for vec_a, vec_b in test_cases:
            similarity = analyzer._calculate_cosine_similarity(vec_a, vec_b)
            assert 0.0 <= similarity <= 1.0


class TestIntegrationWithRealScenarios:
    """Integration tests with realistic scenarios."""

    def test_high_similarity_scenario(self, mock_embedding_model):
        """Test scenario with high semantic similarity."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        # "dog walks" vs "puppy runs"
        zone_a = [
            Token("dog", "NOUN", "D AO1 G", True, 1),
            Token("walks", "VERB", "W AO1 K S", True, 1),
        ]
        zone_b = [
            Token("puppy", "NOUN", "P AH1 P IY0", True, 2),
            Token("run", "VERB", "R AH1 N", True, 1),
        ]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Both pairs are semantically related
        assert similarity > 0.6

    def test_low_similarity_scenario(self, mock_embedding_model):
        """Test scenario with low semantic similarity."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        # "cat" vs "dog" (orthogonal in our mock)
        zone_a = [Token("cat", "NOUN", "K AE1 T", True, 1)]
        zone_b = [Token("dog", "NOUN", "D AO1 G", True, 1)]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Orthogonal embeddings → cosine = 0 → similarity = 0.5
        assert similarity == pytest.approx(0.5)

    def test_mixed_similarity_scenario(self, mock_embedding_model):
        """Test scenario with mixed semantic relationships."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        # Mix of related and unrelated
        zone_a = [
            Token("cat", "NOUN", "K AE1 T", True, 1),  # unrelated to dog
            Token("walks", "VERB", "W AO1 K S", True, 1),  # related to run
        ]
        zone_b = [
            Token("dog", "NOUN", "D AO1 G", True, 1),
            Token("run", "VERB", "R AH1 N", True, 1),
        ]

        similarity = analyzer.analyze(zone_a, zone_b)

        # Mean pooling averages vectors, creating moderate-to-high similarity
        # even with mixed relationships (depends on geometric relationships)
        assert 0.3 <= similarity <= 1.0


class TestReturnValueConstraints:
    """Test that return values meet specification constraints."""

    def test_similarity_in_valid_range(self, mock_embedding_model):
        """Test that all similarities are in [0,1] range."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        test_zones = [
            ([Token("cat", "NOUN", "K AE1 T", True, 1)],
             [Token("dog", "NOUN", "D AO1 G", True, 1)]),
            ([Token("dog", "NOUN", "D AO1 G", True, 1)],
             [Token("puppy", "NOUN", "P AH1 P IY0", True, 2)]),
            ([Token("hello", "INTJ", "HH AH0 L OW1", True, 2)],
             [Token("world", "NOUN", "W ER1 L D", True, 1)]),
        ]

        for zone_a, zone_b in test_zones:
            similarity = analyzer.analyze(zone_a, zone_b)
            assert 0.0 <= similarity <= 1.0

    def test_similarity_is_float(self, mock_embedding_model):
        """Test that similarity is returned as float."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a = [Token("cat", "NOUN", "K AE1 T", True, 1)]
        zone_b = [Token("dog", "NOUN", "D AO1 G", True, 1)]

        similarity = analyzer.analyze(zone_a, zone_b)
        assert isinstance(similarity, (float, np.floating))

    def test_deterministic_results(self, mock_embedding_model):
        """Test that repeated calls return identical results."""
        analyzer = SemanticEchoAnalyzer()
        analyzer.model = mock_embedding_model
        analyzer.model_type = 'gensim'

        zone_a = [Token("cat", "NOUN", "K AE1 T", True, 1)]
        zone_b = [Token("kitten", "NOUN", "K IH1 T AH0 N", True, 2)]

        sim1 = analyzer.analyze(zone_a, zone_b)
        sim2 = analyzer.analyze(zone_a, zone_b)
        sim3 = analyzer.analyze(zone_a, zone_b)

        assert sim1 == sim2 == sim3
