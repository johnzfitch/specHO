"""
Tests for Task 5.3: ScoringModule

Tests the orchestration of weighted scoring and aggregation pipeline.
ScoringModule chains WeightedScorer and DocumentAggregator to provide
a unified interface for converting echo scores to document scores.
"""

import pytest
import numpy as np
from specHO.models import EchoScore
from specHO.scoring.pipeline import ScoringModule
from specHO.scoring.weighted_scorer import WeightedScorer
from specHO.scoring.aggregator import DocumentAggregator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def scorer():
    """Fixture providing ScoringModule instance."""
    return ScoringModule()


@pytest.fixture
def perfect_echo():
    """EchoScore with all 1.0 values."""
    return EchoScore(
        phonetic_score=1.0,
        structural_score=1.0,
        semantic_score=1.0,
        combined_score=0.0  # Not used in Tier 1
    )


@pytest.fixture
def zero_echo():
    """EchoScore with all 0.0 values."""
    return EchoScore(
        phonetic_score=0.0,
        structural_score=0.0,
        semantic_score=0.0,
        combined_score=0.0
    )


@pytest.fixture
def mixed_echo():
    """EchoScore with varied values."""
    return EchoScore(
        phonetic_score=0.8,
        structural_score=0.6,
        semantic_score=0.7,
        combined_score=0.0
    )


@pytest.fixture
def sample_echo_scores():
    """List of varied EchoScore objects for testing."""
    return [
        EchoScore(0.8, 0.6, 0.7, 0.0),
        EchoScore(0.75, 0.65, 0.72, 0.0),
        EchoScore(0.82, 0.58, 0.68, 0.0),
        EchoScore(0.78, 0.62, 0.71, 0.0)
    ]


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_initialization(scorer):
    """Test ScoringModule initializes correctly."""
    assert isinstance(scorer, ScoringModule)
    assert isinstance(scorer.weighted_scorer, WeightedScorer)
    assert isinstance(scorer.aggregator, DocumentAggregator)


def test_has_required_methods(scorer):
    """Test ScoringModule has required API methods."""
    assert hasattr(scorer, 'score_document')
    assert callable(scorer.score_document)


# ============================================================================
# BASIC FUNCTIONALITY TESTS
# ============================================================================

def test_score_document_single_perfect_echo(scorer, perfect_echo):
    """Test scoring a single perfect echo (all 1.0)."""
    doc_score = scorer.score_document([perfect_echo])
    assert doc_score == pytest.approx(1.0, abs=0.001)


def test_score_document_single_zero_echo(scorer, zero_echo):
    """Test scoring a single zero echo (all 0.0)."""
    doc_score = scorer.score_document([zero_echo])
    assert doc_score == pytest.approx(0.0, abs=0.001)


def test_score_document_single_mixed_echo(scorer, mixed_echo):
    """Test scoring a single mixed echo."""
    doc_score = scorer.score_document([mixed_echo])
    # Expected: (0.33*0.8 + 0.33*0.6 + 0.34*0.7) = 0.700
    assert doc_score == pytest.approx(0.700, abs=0.001)


def test_score_document_multiple_echoes(scorer, sample_echo_scores):
    """Test scoring multiple echo scores."""
    doc_score = scorer.score_document(sample_echo_scores)

    # Manually calculate expected score:
    # Pair 1: (0.33*0.8 + 0.33*0.6 + 0.34*0.7) = 0.700
    # Pair 2: (0.33*0.75 + 0.33*0.65 + 0.34*0.72) = 0.707
    # Pair 3: (0.33*0.82 + 0.33*0.58 + 0.34*0.68) = 0.693
    # Pair 4: (0.33*0.78 + 0.33*0.62 + 0.34*0.71) = 0.704
    # Mean: (0.700 + 0.707 + 0.693 + 0.704) / 4 = 0.701

    assert doc_score == pytest.approx(0.701, abs=0.01)


def test_score_document_empty_list(scorer):
    """Test scoring empty echo list returns 0.0."""
    doc_score = scorer.score_document([])
    assert doc_score == 0.0


# ============================================================================
# EDGE CASES
# ============================================================================

def test_score_document_with_nan_values(scorer):
    """Test scoring with NaN values in echo scores."""
    echo_with_nan = EchoScore(
        phonetic_score=0.8,
        structural_score=np.nan,
        semantic_score=0.7,
        combined_score=0.0
    )

    doc_score = scorer.score_document([echo_with_nan])
    # WeightedScorer treats NaN as 0.0
    # Expected: (0.33*0.8 + 0.33*0.0 + 0.34*0.7) = 0.502
    assert doc_score == pytest.approx(0.502, abs=0.01)


def test_score_document_all_nan_scores(scorer):
    """Test scoring with all NaN echo scores."""
    all_nan_echo = EchoScore(
        phonetic_score=np.nan,
        structural_score=np.nan,
        semantic_score=np.nan,
        combined_score=0.0
    )

    doc_score = scorer.score_document([all_nan_echo])
    # All NaN treated as 0.0
    assert doc_score == pytest.approx(0.0, abs=0.001)


def test_score_document_mixed_perfect_and_zero(scorer, perfect_echo, zero_echo):
    """Test scoring with mix of perfect and zero echoes."""
    doc_score = scorer.score_document([perfect_echo, zero_echo])
    # Mean of 1.0 and 0.0
    assert doc_score == pytest.approx(0.5, abs=0.001)


def test_score_document_many_echoes(scorer):
    """Test scoring with large number of echo scores."""
    # Create 100 echo scores with values around 0.7
    echoes = [
        EchoScore(0.7, 0.7, 0.7, 0.0)
        for _ in range(100)
    ]

    doc_score = scorer.score_document(echoes)
    # All identical scores should average to that score
    assert doc_score == pytest.approx(0.7, abs=0.01)


# ============================================================================
# ORCHESTRATION VERIFICATION TESTS
# ============================================================================

def test_delegates_to_weighted_scorer(scorer, mixed_echo, monkeypatch):
    """Test that ScoringModule calls WeightedScorer.calculate_pair_score."""
    call_count = 0
    original_method = WeightedScorer.calculate_pair_score

    def mock_calculate_pair_score(self, echo_score):
        nonlocal call_count
        call_count += 1
        return original_method(self, echo_score)

    monkeypatch.setattr(WeightedScorer, 'calculate_pair_score', mock_calculate_pair_score)

    scorer.score_document([mixed_echo])
    assert call_count == 1


def test_delegates_to_aggregator(scorer, sample_echo_scores, monkeypatch):
    """Test that ScoringModule calls DocumentAggregator.aggregate_scores."""
    call_count = 0
    original_method = DocumentAggregator.aggregate_scores

    def mock_aggregate_scores(self, pair_scores):
        nonlocal call_count
        call_count += 1
        return original_method(self, pair_scores)

    monkeypatch.setattr(DocumentAggregator, 'aggregate_scores', mock_aggregate_scores)

    scorer.score_document(sample_echo_scores)
    assert call_count == 1


def test_correct_number_of_pair_scores_generated(scorer, sample_echo_scores, monkeypatch):
    """Test that ScoringModule generates correct number of pair scores."""
    captured_pair_scores = []

    def mock_aggregate_scores(self, pair_scores):
        captured_pair_scores.extend(pair_scores)
        return sum(pair_scores) / len(pair_scores) if pair_scores else 0.0

    monkeypatch.setattr(DocumentAggregator, 'aggregate_scores', mock_aggregate_scores)

    scorer.score_document(sample_echo_scores)

    # Should have 1 pair score per echo score
    assert len(captured_pair_scores) == len(sample_echo_scores)


# ============================================================================
# OUTPUT VALIDATION TESTS
# ============================================================================

def test_output_is_float(scorer, mixed_echo):
    """Test that score_document returns a float."""
    doc_score = scorer.score_document([mixed_echo])
    assert isinstance(doc_score, float)


def test_output_in_valid_range(scorer, sample_echo_scores):
    """Test that output score is in valid [0,1] range."""
    doc_score = scorer.score_document(sample_echo_scores)
    assert 0.0 <= doc_score <= 1.0


def test_consistent_output(scorer, mixed_echo):
    """Test that repeated calls with same input produce same output."""
    score1 = scorer.score_document([mixed_echo])
    score2 = scorer.score_document([mixed_echo])
    assert score1 == score2


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end_watermarked_signal(scorer):
    """Test end-to-end scoring of strong watermark signal."""
    # Simulate strong echo signal (high similarity scores)
    strong_echoes = [
        EchoScore(0.85, 0.80, 0.82, 0.0),
        EchoScore(0.88, 0.83, 0.85, 0.0),
        EchoScore(0.86, 0.81, 0.83, 0.0),
        EchoScore(0.87, 0.82, 0.84, 0.0)
    ]

    doc_score = scorer.score_document(strong_echoes)

    # Should produce high document score (>0.75)
    assert doc_score > 0.75


def test_end_to_end_natural_text_signal(scorer):
    """Test end-to-end scoring of natural text (weak echo signal)."""
    # Simulate weak/no echo signal (low similarity scores)
    weak_echoes = [
        EchoScore(0.15, 0.12, 0.18, 0.0),
        EchoScore(0.12, 0.15, 0.13, 0.0),
        EchoScore(0.18, 0.11, 0.16, 0.0),
        EchoScore(0.14, 0.13, 0.15, 0.0)
    ]

    doc_score = scorer.score_document(weak_echoes)

    # Should produce low document score (<0.25)
    assert doc_score < 0.25


def test_end_to_end_unwatermarked_ai_signal(scorer):
    """Test end-to-end scoring of unwatermarked AI text (moderate echo)."""
    # Simulate moderate echo signal (some accidental similarity)
    moderate_echoes = [
        EchoScore(0.45, 0.38, 0.42, 0.0),
        EchoScore(0.42, 0.40, 0.44, 0.0),
        EchoScore(0.48, 0.35, 0.40, 0.0),
        EchoScore(0.40, 0.42, 0.45, 0.0)
    ]

    doc_score = scorer.score_document(moderate_echoes)

    # Should produce moderate document score (0.25-0.50)
    assert 0.25 <= doc_score <= 0.50


# ============================================================================
# STATISTICAL PROPERTIES TESTS
# ============================================================================

def test_mean_property_preserved(scorer):
    """Test that averaging property is preserved through pipeline."""
    # Create echo scores that should produce known pair scores
    echoes = [
        EchoScore(1.0, 1.0, 1.0, 0.0),  # Pair score: 1.0
        EchoScore(0.0, 0.0, 0.0, 0.0)   # Pair score: 0.0
    ]

    doc_score = scorer.score_document(echoes)

    # Document score should be mean of pair scores: (1.0 + 0.0) / 2 = 0.5
    assert doc_score == pytest.approx(0.5, abs=0.01)


def test_variance_in_scores(scorer):
    """Test that variance in echo scores produces variance in document score."""
    # Low variance echo scores
    low_variance = [EchoScore(0.5, 0.5, 0.5, 0.0) for _ in range(10)]

    # High variance echo scores
    high_variance = [
        EchoScore(0.9, 0.9, 0.9, 0.0) if i % 2 == 0
        else EchoScore(0.1, 0.1, 0.1, 0.0)
        for i in range(10)
    ]

    low_var_score = scorer.score_document(low_variance)
    high_var_score = scorer.score_document(high_variance)

    # Both should have same mean (0.5), but variance shouldn't affect Tier 1 mean
    assert low_var_score == pytest.approx(0.5, abs=0.01)
    assert high_var_score == pytest.approx(0.5, abs=0.01)
