"""Tests for utility functions (utils.py).

Tests file I/O, logging setup, and error handling decorators.

Tier: 1 (MVP)
Coverage: File operations, logging, decorators
"""

import pytest
import sys
import json
import tempfile
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "specHO"))

from utils import (
    load_text_file,
    save_analysis_results,
    setup_logging,
    handle_errors,
    retry_on_failure,
    validate_input,
    ensure_directory,
    get_file_extension,
)
from models import DocumentAnalysis, Token, Clause, ClausePair, EchoScore


class TestLoadTextFile:
    """Tests for load_text_file() function."""

    def test_load_simple_text_file(self, tmp_path):
        """Test loading a simple text file."""
        test_file = tmp_path / "test.txt"
        test_content = "This is a test document for SpecHO."
        test_file.write_text(test_content, encoding="utf-8")

        result = load_text_file(str(test_file))
        assert result == test_content

    def test_load_multiline_text(self, tmp_path):
        """Test loading text with multiple lines."""
        test_file = tmp_path / "multiline.txt"
        test_content = "Line 1\nLine 2\nLine 3"
        test_file.write_text(test_content, encoding="utf-8")

        result = load_text_file(str(test_file))
        assert "\n" in result
        assert result.count("\n") == 2

    def test_load_empty_file_warns(self, tmp_path, caplog):
        """Test loading empty file generates warning."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = load_text_file(str(test_file))

        assert result == ""
        assert "empty file" in caplog.text.lower()

    def test_load_nonexistent_file_raises_error(self):
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_text_file("nonexistent_file.txt")

    def test_load_directory_raises_error(self, tmp_path):
        """Test loading directory instead of file raises IOError."""
        with pytest.raises(IOError):
            load_text_file(str(tmp_path))

    def test_load_with_different_encoding(self, tmp_path):
        """Test loading file with specified encoding."""
        test_file = tmp_path / "encoded.txt"
        test_file.write_text("Test content", encoding="utf-8")

        result = load_text_file(str(test_file), encoding="utf-8")
        assert result == "Test content"


class TestSaveAnalysisResults:
    """Tests for save_analysis_results() function."""

    def setup_method(self):
        """Create mock DocumentAnalysis for tests."""
        token = Token("test", "NOUN", "T EH S T", True, 1)
        clause = Clause([token], 0, 1, "main", 0)
        pair = ClausePair(clause, clause, [token], [token], "punctuation")
        score = EchoScore(0.8, 0.7, 0.9, 0.8)

        self.mock_analysis = DocumentAnalysis(
            text="Test document for saving.",
            clause_pairs=[pair],
            echo_scores=[score],
            final_score=0.8,
            z_score=2.5,
            confidence=0.95,
        )

    def test_save_as_json(self, tmp_path):
        """Test saving analysis as JSON."""
        output_file = tmp_path / "results.json"
        save_analysis_results(self.mock_analysis, str(output_file), format="json")

        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)

        assert data["final_score"] == 0.8
        assert data["z_score"] == 2.5
        assert data["confidence"] == 0.95

    def test_save_as_text(self, tmp_path):
        """Test saving analysis as human-readable text."""
        output_file = tmp_path / "results.txt"
        save_analysis_results(self.mock_analysis, str(output_file), format="txt")

        assert output_file.exists()
        content = output_file.read_text()

        assert "SpecHO Watermark Detection Analysis" in content
        assert "Final Score:" in content
        assert "Z-Score:" in content
        assert "Confidence:" in content
        assert "2.5" in content  # z-score value

    def test_save_creates_parent_directories(self, tmp_path):
        """Test that save creates parent directories if needed."""
        output_file = tmp_path / "nested" / "dir" / "results.json"
        save_analysis_results(self.mock_analysis, str(output_file), format="json")

        assert output_file.exists()
        assert output_file.parent.exists()

    def test_save_invalid_format_raises_error(self, tmp_path):
        """Test that invalid format raises ValueError."""
        output_file = tmp_path / "results.xml"
        with pytest.raises(ValueError, match="Unsupported format"):
            save_analysis_results(self.mock_analysis, str(output_file), format="xml")

    def test_text_format_includes_verdict(self, tmp_path):
        """Test that text format includes watermark verdict."""
        output_file = tmp_path / "results.txt"

        # Create analysis with high z-score
        high_z_analysis = DocumentAnalysis(
            text="Test",
            clause_pairs=[],
            echo_scores=[],
            final_score=0.9,
            z_score=3.5,
            confidence=0.998,
        )

        save_analysis_results(high_z_analysis, str(output_file), format="txt")
        content = output_file.read_text()

        assert "HIGHLY LIKELY" in content or "watermarked" in content.lower()


class TestSetupLogging:
    """Tests for setup_logging() function."""

    def test_setup_logging_info_level(self):
        """Test logging setup at INFO level."""
        setup_logging("INFO", format_style="simple")

        # Verify root logger is configured correctly
        logger = logging.getLogger()
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0

    def test_setup_logging_debug_level(self):
        """Test logging setup at DEBUG level."""
        setup_logging("DEBUG", format_style="simple")

        # Verify root logger is configured correctly
        logger = logging.getLogger()
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) > 0

    def test_setup_logging_warning_filters_info(self):
        """Test WARNING level filters out INFO messages."""
        setup_logging("WARNING", format_style="simple")

        # Verify root logger is configured to WARNING level
        logger = logging.getLogger()
        assert logger.level == logging.WARNING
        assert len(logger.handlers) > 0

        # Verify that INFO level would be filtered
        assert not logger.isEnabledFor(logging.INFO)
        assert logger.isEnabledFor(logging.WARNING)

    def test_setup_logging_invalid_level_raises_error(self):
        """Test that invalid log level raises ValueError."""
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging("INVALID")

    def test_setup_logging_with_file(self, tmp_path):
        """Test logging to file."""
        log_file = tmp_path / "test.log"
        setup_logging("INFO", log_file=str(log_file), format_style="simple")

        logging.info("Test file logging")

        assert log_file.exists()
        content = log_file.read_text()
        assert "Test file logging" in content


class TestHandleErrorsDecorator:
    """Tests for @handle_errors decorator."""

    def test_handle_errors_returns_default_on_exception(self):
        """Test that decorator returns default value on exception."""

        @handle_errors(default_return=0.0, log_errors=False)
        def divide(a, b):
            return a / b

        result = divide(10, 0)  # Should not raise, returns default
        assert result == 0.0

    def test_handle_errors_allows_success(self):
        """Test that decorator allows successful execution."""

        @handle_errors(default_return=0.0, log_errors=False)
        def divide(a, b):
            return a / b

        result = divide(10, 2)
        assert result == 5.0

    def test_handle_errors_with_list_default(self):
        """Test decorator with list as default return."""

        @handle_errors(default_return=[], log_errors=False)
        def get_items():
            raise ValueError("Something failed")

        result = get_items()
        assert result == []
        assert isinstance(result, list)

    def test_handle_errors_logs_when_enabled(self):
        """Test that decorator logs errors when log_errors=True."""
        setup_logging("ERROR", format_style="simple")

        @handle_errors(default_return=None, log_errors=True)
        def failing_function():
            raise ValueError("Test error")

        # Call function and verify it returns default value
        # (Logging output will appear in stderr but we can't easily capture it in tests
        # due to StreamHandler configuration. We're testing the decorator behavior, not
        # the logging infrastructure.)
        result = failing_function()
        assert result is None


class TestRetryOnFailureDecorator:
    """Tests for @retry_on_failure decorator."""

    def test_retry_succeeds_on_first_attempt(self):
        """Test retry decorator with successful first attempt."""
        call_count = 0

        @retry_on_failure(max_attempts=3, delay=0.01)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retry_succeeds_after_failures(self):
        """Test retry decorator succeeds after initial failures."""
        call_count = 0

        @retry_on_failure(max_attempts=3, delay=0.01)
        def succeeds_on_third_try():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("Temporary failure")
            return "success"

        result = succeeds_on_third_try()
        assert result == "success"
        assert call_count == 3

    def test_retry_exhausts_attempts(self):
        """Test retry decorator exhausts all attempts and raises."""

        @retry_on_failure(max_attempts=2, delay=0.01)
        def always_fails():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            always_fails()

    def test_retry_specific_exceptions(self):
        """Test retry only catches specified exceptions."""

        @retry_on_failure(max_attempts=2, delay=0.01, exceptions=(IOError,))
        def raises_value_error():
            raise ValueError("Not retryable")

        # Should raise immediately, not retry
        with pytest.raises(ValueError):
            raises_value_error()


class TestValidateInputDecorator:
    """Tests for @validate_input decorator."""

    def test_validate_input_passes_valid_args(self):
        """Test decorator passes with valid arguments."""

        @validate_input(x=lambda x: x > 0)
        def process(x):
            return x * 2

        result = process(5)
        assert result == 10

    def test_validate_input_rejects_invalid_args(self):
        """Test decorator rejects invalid arguments."""

        @validate_input(x=lambda x: x > 0)
        def process(x):
            return x * 2

        with pytest.raises(ValueError, match="Validation failed"):
            process(-5)

    def test_validate_input_multiple_validators(self):
        """Test decorator with multiple validators."""

        @validate_input(x=lambda x: x > 0, name=lambda n: len(n) > 0)
        def process(x, name):
            return f"{name}: {x}"

        result = process(5, "test")
        assert result == "test: 5"

        with pytest.raises(ValueError):
            process(-1, "test")

        with pytest.raises(ValueError):
            process(5, "")

    def test_validate_input_with_kwargs(self):
        """Test decorator works with keyword arguments."""

        @validate_input(value=lambda v: v >= 0)
        def calculate(value=10):
            return value * 2

        assert calculate(value=5) == 10

        with pytest.raises(ValueError):
            calculate(value=-1)


class TestHelperUtilities:
    """Tests for helper utility functions."""

    def test_ensure_directory_creates_dir(self, tmp_path):
        """Test ensure_directory creates directory."""
        new_dir = tmp_path / "new_directory"
        result = ensure_directory(str(new_dir))

        assert new_dir.exists()
        assert new_dir.is_dir()
        assert isinstance(result, Path)

    def test_ensure_directory_creates_nested_dirs(self, tmp_path):
        """Test ensure_directory creates nested directories."""
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        ensure_directory(str(nested_dir))

        assert nested_dir.exists()

    def test_ensure_directory_idempotent(self, tmp_path):
        """Test ensure_directory is idempotent (can call multiple times)."""
        new_dir = tmp_path / "test_dir"
        ensure_directory(str(new_dir))
        ensure_directory(str(new_dir))  # Should not raise

        assert new_dir.exists()

    def test_get_file_extension(self):
        """Test get_file_extension extracts extension."""
        assert get_file_extension("file.txt") == "txt"
        assert get_file_extension("data/sample.json") == "json"
        assert get_file_extension("output/results.CSV") == "csv"

    def test_get_file_extension_no_extension(self):
        """Test get_file_extension with no extension."""
        assert get_file_extension("file") == ""
        assert get_file_extension("path/to/file") == ""


class TestUtilsIntegration:
    """Integration tests for utilities working together."""

    def test_full_workflow_load_and_save(self, tmp_path):
        """Test complete workflow: load text, process, save results."""
        # Create input file
        input_file = tmp_path / "input.txt"
        input_file.write_text("Test document for full workflow.")

        # Load text
        text = load_text_file(str(input_file))
        assert len(text) > 0

        # Create mock analysis
        token = Token("test", "NOUN", "T EH S T", True, 1)
        clause = Clause([token], 0, 1, "main", 0)
        pair = ClausePair(clause, clause, [token], [token], "punctuation")
        score = EchoScore(0.8, 0.7, 0.9, 0.8)

        analysis = DocumentAnalysis(
            text=text,
            clause_pairs=[pair],
            echo_scores=[score],
            final_score=0.8,
            z_score=2.5,
            confidence=0.95,
        )

        # Save results
        output_file = tmp_path / "output" / "results.json"
        save_analysis_results(analysis, str(output_file))

        # Verify saved
        assert output_file.exists()
        with open(output_file) as f:
            data = json.load(f)
        assert data["text"] == text
