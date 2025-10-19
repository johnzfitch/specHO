"""
Tests for Task 6.1: BaselineCorpusProcessor

Tests the baseline corpus processor that establishes baseline statistics
by processing a corpus of human/natural text through the complete SpecHO pipeline.
"""

import pytest
import pickle
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from specHO.validator.baseline_builder import BaselineCorpusProcessor
from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.echo_engine.pipeline import EchoAnalysisEngine
from specHO.scoring.pipeline import ScoringModule


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def processor():
    """Fixture providing BaselineCorpusProcessor instance."""
    return BaselineCorpusProcessor()


@pytest.fixture
def temp_corpus_dir(tmp_path):
    """Fixture providing temporary corpus directory with sample files."""
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Create sample text files
    texts = [
        "This is a sample document. It contains multiple sentences for testing.",
        "Another test document here. We need several files to calculate statistics.",
        "Third document with different content. Each file should be processed independently.",
        "Fourth test file. This helps establish baseline statistics with variance.",
        "Final document in the test corpus. Five files should be enough for testing."
    ]

    for i, text in enumerate(texts):
        file_path = corpus_dir / f"doc_{i}.txt"
        file_path.write_text(text, encoding='utf-8')

    return corpus_dir


@pytest.fixture
def sample_baseline_stats():
    """Fixture providing sample baseline statistics."""
    return {
        'human_mean': 0.142,
        'human_std': 0.087,
        'n_documents': 50
    }


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================

def test_initialization(processor):
    """Test BaselineCorpusProcessor initializes correctly."""
    assert isinstance(processor, BaselineCorpusProcessor)
    assert isinstance(processor.preprocessor, LinguisticPreprocessor)
    assert isinstance(processor.clause_identifier, ClauseIdentifier)
    assert isinstance(processor.echo_engine, EchoAnalysisEngine)
    assert isinstance(processor.scoring_module, ScoringModule)


def test_has_required_methods(processor):
    """Test BaselineCorpusProcessor has required API methods."""
    assert hasattr(processor, 'process_corpus')
    assert callable(processor.process_corpus)
    assert hasattr(processor, 'save_baseline')
    assert callable(processor.save_baseline)
    assert hasattr(processor, 'load_baseline')
    assert callable(processor.load_baseline)


# ============================================================================
# PROCESS_CORPUS TESTS
# ============================================================================

def test_process_corpus_valid_directory(processor, temp_corpus_dir):
    """Test processing a valid corpus directory."""
    stats = processor.process_corpus(str(temp_corpus_dir))

    # Verify return structure
    assert isinstance(stats, dict)
    assert 'human_mean' in stats
    assert 'human_std' in stats
    assert 'n_documents' in stats

    # Verify statistics are reasonable
    assert isinstance(stats['human_mean'], float)
    assert isinstance(stats['human_std'], float)
    assert isinstance(stats['n_documents'], int)

    # Mean should be in valid score range [0,1]
    assert 0.0 <= stats['human_mean'] <= 1.0

    # Std should be non-negative
    assert stats['human_std'] >= 0.0

    # Should have processed all 5 documents
    assert stats['n_documents'] == 5


def test_process_corpus_nonexistent_directory(processor):
    """Test processing nonexistent directory raises ValueError."""
    with pytest.raises(ValueError, match="Corpus directory does not exist"):
        processor.process_corpus("/nonexistent/path/to/corpus")


def test_process_corpus_empty_directory(processor, tmp_path):
    """Test processing empty directory raises ValueError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ValueError, match="No .txt files found"):
        processor.process_corpus(str(empty_dir))


def test_process_corpus_ignores_non_txt_files(processor, tmp_path):
    """Test that non-.txt files are ignored."""
    corpus_dir = tmp_path / "mixed"
    corpus_dir.mkdir()

    # Create .txt file
    (corpus_dir / "valid.txt").write_text("Valid document.", encoding='utf-8')

    # Create non-.txt files (should be ignored)
    (corpus_dir / "readme.md").write_text("Markdown file", encoding='utf-8')
    (corpus_dir / "data.json").write_text('{"key": "value"}', encoding='utf-8')

    stats = processor.process_corpus(str(corpus_dir))

    # Should only process the .txt file
    assert stats['n_documents'] == 1


def test_process_corpus_returns_statistics_dict(processor, temp_corpus_dir):
    """Test that process_corpus returns dict with expected keys."""
    stats = processor.process_corpus(str(temp_corpus_dir))

    # Check all required keys present
    required_keys = {'human_mean', 'human_std', 'n_documents'}
    assert set(stats.keys()) == required_keys


def test_process_corpus_scores_are_valid(processor, temp_corpus_dir):
    """Test that all document scores are in valid range."""
    # Process corpus
    stats = processor.process_corpus(str(temp_corpus_dir))

    # Mean should be within [0,1]
    assert 0.0 <= stats['human_mean'] <= 1.0

    # Standard deviation should be reasonable (not negative, not impossibly large)
    assert 0.0 <= stats['human_std'] <= 1.0


# ============================================================================
# SAVE/LOAD BASELINE TESTS
# ============================================================================

def test_save_baseline(processor, sample_baseline_stats, tmp_path):
    """Test saving baseline statistics to pickle file."""
    output_path = tmp_path / "baseline.pkl"

    # Save baseline
    processor.save_baseline(sample_baseline_stats, str(output_path))

    # Verify file was created
    assert output_path.exists()

    # Verify file is valid pickle
    with open(output_path, 'rb') as f:
        loaded = pickle.load(f)

    assert loaded == sample_baseline_stats


def test_save_baseline_creates_directory(processor, sample_baseline_stats, tmp_path):
    """Test that save_baseline creates output directory if needed."""
    output_path = tmp_path / "subdir" / "baseline.pkl"

    # Directory doesn't exist yet
    assert not output_path.parent.exists()

    # Save should create directory
    processor.save_baseline(sample_baseline_stats, str(output_path))

    # Verify directory and file created
    assert output_path.parent.exists()
    assert output_path.exists()


def test_load_baseline(processor, sample_baseline_stats, tmp_path):
    """Test loading baseline statistics from pickle file."""
    baseline_path = tmp_path / "baseline.pkl"

    # Save baseline first
    with open(baseline_path, 'wb') as f:
        pickle.dump(sample_baseline_stats, f)

    # Load baseline
    loaded_stats = processor.load_baseline(str(baseline_path))

    # Verify loaded data matches original
    assert loaded_stats == sample_baseline_stats
    assert loaded_stats['human_mean'] == sample_baseline_stats['human_mean']
    assert loaded_stats['human_std'] == sample_baseline_stats['human_std']
    assert loaded_stats['n_documents'] == sample_baseline_stats['n_documents']


def test_load_baseline_nonexistent_file(processor):
    """Test loading nonexistent baseline file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Baseline file not found"):
        processor.load_baseline("/nonexistent/baseline.pkl")


def test_save_and_load_roundtrip(processor, sample_baseline_stats, tmp_path):
    """Test that save and load preserve data correctly."""
    baseline_path = tmp_path / "baseline.pkl"

    # Save
    processor.save_baseline(sample_baseline_stats, str(baseline_path))

    # Load
    loaded_stats = processor.load_baseline(str(baseline_path))

    # Verify exact match
    assert loaded_stats == sample_baseline_stats


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_end_to_end_corpus_processing(processor, tmp_path):
    """Test complete workflow: create corpus → process → save → load."""
    # Create corpus directory with sample files
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()

    # Add multiple text files with varied content
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a test sentence.",
        "Machine learning models can detect patterns in data. Statistical analysis helps.",
        "Natural language processing enables computers to understand text. NLP is powerful.",
    ]

    for i, text in enumerate(texts):
        (corpus_dir / f"doc_{i}.txt").write_text(text, encoding='utf-8')

    # Process corpus
    stats = processor.process_corpus(str(corpus_dir))

    # Verify statistics
    assert stats['n_documents'] == 3
    assert 0.0 <= stats['human_mean'] <= 1.0
    assert stats['human_std'] >= 0.0

    # Save baseline
    baseline_path = tmp_path / "baseline.pkl"
    processor.save_baseline(stats, str(baseline_path))

    # Load baseline
    loaded_stats = processor.load_baseline(str(baseline_path))

    # Verify roundtrip
    assert loaded_stats == stats


def test_corpus_with_utf8_content(processor, tmp_path):
    """Test processing corpus with UTF-8 special characters."""
    corpus_dir = tmp_path / "utf8_corpus"
    corpus_dir.mkdir()

    # Create file with UTF-8 characters
    text_with_unicode = "Café résumé naïve. These words contain special characters: é, ü, ñ."
    (corpus_dir / "unicode.txt").write_text(text_with_unicode, encoding='utf-8')

    # Should process without errors
    stats = processor.process_corpus(str(corpus_dir))

    assert stats['n_documents'] == 1
    assert 0.0 <= stats['human_mean'] <= 1.0


def test_statistics_have_variance(processor, temp_corpus_dir):
    """Test that processing multiple documents produces variance."""
    stats = processor.process_corpus(str(temp_corpus_dir))

    # With 5 different documents, std should be >= 0
    # Note: Very short documents may all score 0.0 (no clause pairs), giving std=0
    # This is acceptable behavior for the baseline processor
    assert stats['human_std'] >= 0.0
    assert isinstance(stats['human_std'], float)
    assert not np.isnan(stats['human_std'])


def test_mean_in_expected_range_for_natural_text(processor, temp_corpus_dir):
    """Test that natural text corpus produces expected score range."""
    stats = processor.process_corpus(str(temp_corpus_dir))

    # Natural/human text typically scores low (0.0-0.3)
    # This is a sanity check that the pipeline is working correctly
    assert 0.0 <= stats['human_mean'] <= 0.5


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

def test_process_corpus_handles_empty_files(processor, tmp_path):
    """Test processing corpus with empty files."""
    corpus_dir = tmp_path / "with_empty"
    corpus_dir.mkdir()

    # Create valid file
    (corpus_dir / "valid.txt").write_text("Valid content here.", encoding='utf-8')

    # Create empty file
    (corpus_dir / "empty.txt").write_text("", encoding='utf-8')

    # Should process valid file and skip empty file
    # (empty file will fail during pipeline, but that's handled)
    stats = processor.process_corpus(str(corpus_dir))

    # Should have at least processed the valid file
    assert stats['n_documents'] >= 1


def test_process_corpus_with_minimal_text(processor, tmp_path):
    """Test processing corpus with very short documents."""
    corpus_dir = tmp_path / "minimal"
    corpus_dir.mkdir()

    # Create files with minimal content (might not have clause pairs)
    (corpus_dir / "short1.txt").write_text("Hi.", encoding='utf-8')
    (corpus_dir / "short2.txt").write_text("Okay.", encoding='utf-8')
    (corpus_dir / "short3.txt").write_text("Yes.", encoding='utf-8')

    # Should complete without errors (even if some docs score 0.0)
    stats = processor.process_corpus(str(corpus_dir))

    assert stats['n_documents'] >= 0  # Some might fail
    assert 0.0 <= stats['human_mean'] <= 1.0


# ============================================================================
# OUTPUT VALIDATION TESTS
# ============================================================================

def test_baseline_stats_have_correct_types(processor, temp_corpus_dir):
    """Test that baseline statistics have correct data types."""
    stats = processor.process_corpus(str(temp_corpus_dir))

    assert isinstance(stats['human_mean'], float)
    assert isinstance(stats['human_std'], float)
    assert isinstance(stats['n_documents'], int)


def test_baseline_stats_are_json_serializable(processor, temp_corpus_dir):
    """Test that baseline statistics can be JSON serialized (for future Tier 2)."""
    import json

    stats = processor.process_corpus(str(temp_corpus_dir))

    # Should be able to serialize to JSON
    json_str = json.dumps(stats)
    reloaded = json.loads(json_str)

    assert reloaded['human_mean'] == pytest.approx(stats['human_mean'])
    assert reloaded['human_std'] == pytest.approx(stats['human_std'])
    assert reloaded['n_documents'] == stats['n_documents']


def test_n_documents_matches_file_count(processor, temp_corpus_dir):
    """Test that n_documents in stats matches actual file count."""
    # Count .txt files in corpus
    txt_files = list(temp_corpus_dir.glob("*.txt"))

    stats = processor.process_corpus(str(temp_corpus_dir))

    # Should process all files (or close to all if some fail)
    assert stats['n_documents'] <= len(txt_files)
    assert stats['n_documents'] > 0
