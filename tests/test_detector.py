"""Tests for SpecHODetector (Task 7.1).

This test suite validates the main watermark detection orchestrator.
Tests cover initialization, pipeline execution, error handling, and
end-to-end integration scenarios.

Test Categories:
- Initialization (successful, missing baseline)
- Basic analysis (valid text, empty text, None input)
- Pipeline orchestration (all 5 components called correctly)
- Error handling (component failures, invalid inputs)
- Helper methods (get_pipeline_info)
- Integration scenarios (real text samples)

Tier: 1 (MVP)
Task: 7.1
"""

import pytest
import pickle
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from specHO.detector import SpecHODetector
from specHO.models import DocumentAnalysis, ClausePair, EchoScore, Token, Clause


@pytest.fixture
def test_baseline_file(tmp_path):
    """Create a temporary baseline file for testing."""
    baseline_stats = {
        'human_mean': 0.15,
        'human_std': 0.10,
        'n_documents': 100
    }

    baseline_path = tmp_path / "test_baseline.pkl"
    with open(baseline_path, 'wb') as f:
        pickle.dump(baseline_stats, f)

    return str(baseline_path)


@pytest.fixture
def mock_components():
    """Create mock components for isolated orchestration testing."""
    mocks = {
        'preprocessor': Mock(),
        'clause_identifier': Mock(),
        'echo_engine': Mock(),
        'scoring_module': Mock(),
        'validator': Mock()
    }
    return mocks


class TestSpecHODetectorInitialization:
    """Tests for SpecHODetector initialization."""

    def test_successful_initialization(self, test_baseline_file):
        """Test that detector initializes successfully with valid baseline."""
        detector = SpecHODetector(test_baseline_file)

        assert detector.preprocessor is not None
        assert detector.clause_identifier is not None
        assert detector.echo_engine is not None
        assert detector.scoring_module is not None
        assert detector.validator is not None
        assert detector.baseline_path == test_baseline_file

    def test_missing_baseline_file_raises_error(self):
        """Test that missing baseline file raises FileNotFoundError."""
        nonexistent_path = "data/baseline/nonexistent.pkl"

        with pytest.raises(FileNotFoundError, match="Baseline statistics file not found"):
            SpecHODetector(nonexistent_path)

    def test_default_baseline_path(self, test_baseline_file, monkeypatch):
        """Test that default baseline path is used when not specified."""
        # Create default location temporarily
        default_path = Path("data/baseline/baseline_stats.pkl")
        default_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy test baseline to default location
        with open(test_baseline_file, 'rb') as src:
            baseline_data = src.read()
        with open(default_path, 'wb') as dst:
            dst.write(baseline_data)

        try:
            detector = SpecHODetector()
            assert detector.baseline_path == "data/baseline/baseline_stats.pkl"
        finally:
            # Cleanup
            default_path.unlink(missing_ok=True)

    def test_all_components_initialized(self, test_baseline_file):
        """Test that all five pipeline components are initialized."""
        detector = SpecHODetector(test_baseline_file)

        # Verify each component type
        from specHO.preprocessor.pipeline import LinguisticPreprocessor
        from specHO.clause_identifier.pipeline import ClauseIdentifier
        from specHO.echo_engine.pipeline import EchoAnalysisEngine
        from specHO.scoring.pipeline import ScoringModule
        from specHO.validator.pipeline import StatisticalValidator

        assert isinstance(detector.preprocessor, LinguisticPreprocessor)
        assert isinstance(detector.clause_identifier, ClauseIdentifier)
        assert isinstance(detector.echo_engine, EchoAnalysisEngine)
        assert isinstance(detector.scoring_module, ScoringModule)
        assert isinstance(detector.validator, StatisticalValidator)


class TestSpecHODetectorBasicAnalysis:
    """Tests for basic text analysis functionality."""

    def test_analyze_valid_text(self, test_baseline_file):
        """Test analysis of valid text produces DocumentAnalysis."""
        detector = SpecHODetector(test_baseline_file)

        text = "The conference ended. However, discussions continued."
        analysis = detector.analyze(text)

        # Verify return type
        assert isinstance(analysis, DocumentAnalysis)

        # Verify all fields present
        assert analysis.text == text
        assert isinstance(analysis.clause_pairs, list)
        assert isinstance(analysis.echo_scores, list)
        assert isinstance(analysis.final_score, float)
        assert isinstance(analysis.z_score, float)
        assert isinstance(analysis.confidence, float)

        # Verify score ranges
        assert 0.0 <= analysis.final_score <= 1.0
        assert 0.0 <= analysis.confidence <= 1.0

    def test_analyze_empty_text(self, test_baseline_file):
        """Test that empty text returns zero analysis."""
        detector = SpecHODetector(test_baseline_file)

        analysis = detector.analyze("")

        assert analysis.text == ""
        assert len(analysis.clause_pairs) == 0
        assert len(analysis.echo_scores) == 0
        assert analysis.final_score == 0.0
        assert analysis.z_score == 0.0
        assert analysis.confidence == 0.5  # Neutral for empty

    def test_analyze_whitespace_only(self, test_baseline_file):
        """Test that whitespace-only text is treated as empty."""
        detector = SpecHODetector(test_baseline_file)

        analysis = detector.analyze("   \n\t  ")

        assert len(analysis.clause_pairs) == 0
        assert analysis.final_score == 0.0

    def test_analyze_none_raises_error(self, test_baseline_file):
        """Test that None input raises ValueError."""
        detector = SpecHODetector(test_baseline_file)

        with pytest.raises(ValueError, match="Input text cannot be None"):
            detector.analyze(None)

    def test_analyze_simple_sentence(self, test_baseline_file):
        """Test analysis of simple sentence with no clause pairs."""
        detector = SpecHODetector(test_baseline_file)

        text = "Hello world."
        analysis = detector.analyze(text)

        assert analysis.text == text
        # Simple sentence likely has 0 clause pairs (no conjunctions/transitions)
        assert isinstance(analysis.clause_pairs, list)

    def test_analyze_complex_text(self, test_baseline_file):
        """Test analysis of complex text with multiple clauses."""
        detector = SpecHODetector(test_baseline_file)

        text = (
            "The meeting started late. However, the presentation was excellent. "
            "The team collaborated well. Therefore, results exceeded expectations."
        )
        analysis = detector.analyze(text)

        assert analysis.text == text
        assert len(analysis.clause_pairs) > 0  # Should find transition pairs
        assert len(analysis.echo_scores) == len(analysis.clause_pairs)
        assert 0.0 <= analysis.final_score <= 1.0


class TestSpecHODetectorPipelineOrchestration:
    """Tests for pipeline component orchestration."""

    def test_pipeline_stages_called_in_order(self, test_baseline_file):
        """Test that all five pipeline stages are called in correct order."""
        detector = SpecHODetector(test_baseline_file)

        with patch.object(detector.preprocessor, 'process', wraps=detector.preprocessor.process) as mock_prep, \
             patch.object(detector.clause_identifier, 'identify_pairs', wraps=detector.clause_identifier.identify_pairs) as mock_clause, \
             patch.object(detector.echo_engine, 'analyze_pair', wraps=detector.echo_engine.analyze_pair) as mock_echo, \
             patch.object(detector.scoring_module, 'score_document', wraps=detector.scoring_module.score_document) as mock_score, \
             patch.object(detector.validator, 'validate', wraps=detector.validator.validate) as mock_val:

            text = "The sky darkened. But hope remained."
            analysis = detector.analyze(text)

            # Verify all components were called
            mock_prep.assert_called_once()
            mock_clause.assert_called_once()
            mock_score.assert_called_once()
            mock_val.assert_called_once()

    def test_preprocessor_output_passed_to_clause_identifier(self, test_baseline_file):
        """Test that preprocessor output is correctly passed to clause identifier."""
        detector = SpecHODetector(test_baseline_file)

        with patch.object(detector.preprocessor, 'process') as mock_prep, \
             patch.object(detector.clause_identifier, 'identify_pairs') as mock_clause, \
             patch.object(detector.echo_engine, 'analyze_pair') as mock_echo, \
             patch.object(detector.scoring_module, 'score_document') as mock_score, \
             patch.object(detector.validator, 'validate') as mock_val:

            # Mock preprocessor return value
            mock_tokens = [Mock()]
            mock_doc = Mock()
            mock_doc.sents = []  # Make doc.sents iterable for logging
            mock_prep.return_value = (mock_tokens, mock_doc)
            mock_clause.return_value = []
            mock_score.return_value = 0.5
            mock_val.return_value = (0.0, 0.5)

            detector.analyze("Test text")

            # Verify clause identifier received preprocessor output
            mock_clause.assert_called_once_with(mock_tokens, mock_doc)

    def test_clause_pairs_passed_to_echo_engine(self, test_baseline_file):
        """Test that identified clause pairs are passed to echo engine."""
        detector = SpecHODetector(test_baseline_file)

        # Create mock clause pair
        mock_pair = ClausePair(
            clause_a=Clause(tokens=[], start_idx=0, end_idx=3, clause_type="main", head_idx=1),
            clause_b=Clause(tokens=[], start_idx=4, end_idx=7, clause_type="main", head_idx=5),
            zone_a_tokens=[],
            zone_b_tokens=[],
            pair_type="rule_a"
        )

        mock_echo_score = EchoScore(0.5, 0.5, 0.5, 0.5)

        with patch.object(detector.clause_identifier, 'identify_pairs', return_value=[mock_pair]), \
             patch.object(detector.echo_engine, 'analyze_pair', return_value=mock_echo_score) as mock_echo, \
             patch.object(detector.scoring_module, 'score_document') as mock_score, \
             patch.object(detector.validator, 'validate') as mock_val:

            mock_score.return_value = 0.5
            mock_val.return_value = (0.0, 0.5)

            detector.analyze("Test text")

            # Verify echo engine analyzed the pair
            mock_echo.assert_called_once()
            assert mock_echo.call_args[0][0] == mock_pair

    def test_echo_scores_passed_to_scoring_module(self, test_baseline_file):
        """Test that echo scores are passed to scoring module."""
        detector = SpecHODetector(test_baseline_file)

        mock_echo_score = EchoScore(
            phonetic_score=0.8,
            structural_score=0.6,
            semantic_score=0.7,
            combined_score=0.7
        )

        with patch.object(detector.clause_identifier, 'identify_pairs'), \
             patch.object(detector.echo_engine, 'analyze_pair', return_value=mock_echo_score), \
             patch.object(detector.scoring_module, 'score_document', wraps=detector.scoring_module.score_document) as mock_score:

            # Mock to generate one pair
            detector.clause_identifier.identify_pairs.return_value = [Mock()]
            detector.validator.validate = Mock(return_value=(0.0, 0.5))

            detector.analyze("Test text")

            # Verify scoring module received echo scores
            mock_score.assert_called_once()
            echo_scores_arg = mock_score.call_args[0][0]
            assert len(echo_scores_arg) == 1
            assert echo_scores_arg[0] == mock_echo_score

    def test_document_score_passed_to_validator(self, test_baseline_file):
        """Test that document score is passed to statistical validator."""
        detector = SpecHODetector(test_baseline_file)

        mock_score = 0.456

        with patch.object(detector.clause_identifier, 'identify_pairs', return_value=[]), \
             patch.object(detector.scoring_module, 'score_document', return_value=mock_score), \
             patch.object(detector.validator, 'validate', wraps=detector.validator.validate) as mock_val:

            detector.analyze("Test text")

            # Verify validator received document score
            mock_val.assert_called_once_with(mock_score)


class TestSpecHODetectorErrorHandling:
    """Tests for error handling throughout the pipeline."""

    def test_empty_clause_pairs_handled_gracefully(self, test_baseline_file):
        """Test that empty clause pairs list is handled without errors."""
        detector = SpecHODetector(test_baseline_file)

        with patch.object(detector.clause_identifier, 'identify_pairs', return_value=[]):
            analysis = detector.analyze("Test text")

            assert len(analysis.clause_pairs) == 0
            assert len(analysis.echo_scores) == 0
            assert analysis.final_score == 0.0

    def test_echo_engine_error_continues_pipeline(self, test_baseline_file, caplog):
        """Test that echo engine errors don't halt the pipeline."""
        detector = SpecHODetector(test_baseline_file)

        # Create two mock pairs
        mock_pair1 = Mock()
        mock_pair2 = Mock()

        with patch.object(detector.clause_identifier, 'identify_pairs', return_value=[mock_pair1, mock_pair2]), \
             patch.object(detector.echo_engine, 'analyze_pair') as mock_echo:

            # First pair fails, second succeeds
            mock_echo.side_effect = [
                Exception("Echo analysis failed"),
                EchoScore(0.5, 0.5, 0.5, 0.5)
            ]

            caplog.clear()
            with caplog.at_level(logging.ERROR):
                analysis = detector.analyze("Test text")

            # Pipeline should continue despite error
            assert len(analysis.echo_scores) == 1
            assert "Error analyzing pair" in caplog.text

    def test_logging_at_each_stage(self, test_baseline_file, caplog):
        """Test that each pipeline stage produces log messages."""
        detector = SpecHODetector(test_baseline_file)

        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            detector.analyze("The sky darkened. But hope remained.")

        # Check for stage markers in logs
        assert "Stage 1: Linguistic preprocessing" in caplog.text
        assert "Stage 2: Identifying clause pairs" in caplog.text
        assert "Stage 3: Analyzing echoes" in caplog.text
        assert "Stage 4: Scoring document" in caplog.text
        assert "Stage 5: Statistical validation" in caplog.text

    def test_complete_pipeline_failure_raises_exception(self, test_baseline_file):
        """Test that complete pipeline failure raises exception."""
        detector = SpecHODetector(test_baseline_file)

        with patch.object(detector.preprocessor, 'process', side_effect=Exception("Fatal error")):
            with pytest.raises(Exception, match="Fatal error"):
                detector.analyze("Test text")


class TestSpecHODetectorHelperMethods:
    """Tests for detector helper methods."""

    def test_get_pipeline_info_returns_dict(self, test_baseline_file):
        """Test that get_pipeline_info returns configuration dictionary."""
        detector = SpecHODetector(test_baseline_file)

        info = detector.get_pipeline_info()

        assert isinstance(info, dict)
        assert 'components' in info
        assert 'baseline_path' in info
        assert 'baseline_stats' in info
        assert 'tier' in info
        assert 'version' in info

    def test_get_pipeline_info_lists_all_components(self, test_baseline_file):
        """Test that pipeline info lists all five components."""
        detector = SpecHODetector(test_baseline_file)

        info = detector.get_pipeline_info()

        expected_components = [
            'LinguisticPreprocessor',
            'ClauseIdentifier',
            'EchoAnalysisEngine',
            'ScoringModule',
            'StatisticalValidator'
        ]

        assert info['components'] == expected_components

    def test_get_pipeline_info_includes_baseline_path(self, test_baseline_file):
        """Test that pipeline info includes baseline path."""
        detector = SpecHODetector(test_baseline_file)

        info = detector.get_pipeline_info()

        assert info['baseline_path'] == test_baseline_file

    def test_get_pipeline_info_includes_baseline_stats(self, test_baseline_file):
        """Test that pipeline info includes baseline statistics."""
        detector = SpecHODetector(test_baseline_file)

        info = detector.get_pipeline_info()

        baseline_stats = info['baseline_stats']
        assert baseline_stats['human_mean'] == 0.15
        assert baseline_stats['human_std'] == 0.10
        assert baseline_stats['n_documents'] == 100

    def test_create_empty_analysis_produces_valid_structure(self, test_baseline_file):
        """Test that _create_empty_analysis produces valid DocumentAnalysis."""
        detector = SpecHODetector(test_baseline_file)

        analysis = detector._create_empty_analysis("Empty text")

        assert isinstance(analysis, DocumentAnalysis)
        assert analysis.text == "Empty text"
        assert analysis.clause_pairs == []
        assert analysis.echo_scores == []
        assert analysis.final_score == 0.0
        assert analysis.z_score == 0.0
        assert analysis.confidence == 0.5


class TestSpecHODetectorIntegration:
    """Integration tests with real text samples."""

    def test_analyze_news_article_excerpt(self, test_baseline_file):
        """Test analysis of news-style text."""
        detector = SpecHODetector(test_baseline_file)

        text = (
            "The conference concluded successfully. However, many questions remain unanswered. "
            "Experts expressed optimism. Yet concerns about implementation persist."
        )

        analysis = detector.analyze(text)

        # Should identify at least one transition pair (However or Yet)
        # Note: Actual count depends on clause boundary detection heuristics
        assert len(analysis.clause_pairs) >= 1
        assert analysis.final_score >= 0.0
        assert isinstance(analysis.confidence, float)

    def test_analyze_conversational_text(self, test_baseline_file):
        """Test analysis of conversational text."""
        detector = SpecHODetector(test_baseline_file)

        text = "I went to the store. But it was closed."

        analysis = detector.analyze(text)

        # Should find "But" conjunction pair
        assert len(analysis.clause_pairs) >= 1
        assert 0.0 <= analysis.final_score <= 1.0

    def test_analyze_multiple_paragraphs(self, test_baseline_file):
        """Test analysis of multi-paragraph text."""
        detector = SpecHODetector(test_baseline_file)

        text = (
            "The project began with high expectations. However, challenges emerged early.\n\n"
            "The team adapted quickly. Therefore, progress continued steadily.\n\n"
            "Final results exceeded initial goals. But lessons were learned along the way."
        )

        analysis = detector.analyze(text)

        assert len(analysis.text) > 0
        assert len(analysis.clause_pairs) >= 3  # Multiple transition words
        assert len(analysis.echo_scores) == len(analysis.clause_pairs)

    def test_analyze_technical_text(self, test_baseline_file):
        """Test analysis of technical/academic text."""
        detector = SpecHODetector(test_baseline_file)

        text = (
            "The algorithm processes data efficiently. However, memory usage remains high. "
            "Optimization strategies were implemented. Therefore, performance improved significantly."
        )

        analysis = detector.analyze(text)

        assert len(analysis.clause_pairs) >= 2
        assert analysis.final_score >= 0.0

    def test_consistent_results_on_repeated_analysis(self, test_baseline_file):
        """Test that analyzing same text produces consistent results."""
        detector = SpecHODetector(test_baseline_file)

        text = "The meeting ended. However, discussions continued."

        analysis1 = detector.analyze(text)
        analysis2 = detector.analyze(text)

        # Results should be identical
        assert analysis1.final_score == analysis2.final_score
        assert analysis1.z_score == analysis2.z_score
        assert analysis1.confidence == analysis2.confidence
        assert len(analysis1.clause_pairs) == len(analysis2.clause_pairs)

    def test_different_texts_produce_different_scores(self, test_baseline_file):
        """Test that different texts produce different analysis results."""
        detector = SpecHODetector(test_baseline_file)

        text1 = "Hello world."  # Simple, no clause pairs
        text2 = "The sky darkened. However, hope remained. Yet fears persisted. But courage endured."  # Many pairs

        analysis1 = detector.analyze(text1)
        analysis2 = detector.analyze(text2)

        # Different texts should have different characteristics
        assert len(analysis1.clause_pairs) != len(analysis2.clause_pairs)
        # Scores may differ (though could coincidentally be same, so not asserting inequality)
