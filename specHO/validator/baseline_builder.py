"""Baseline corpus processor for statistical validation.

This module implements the BaselineCorpusProcessor class which processes a corpus
of human/natural text to establish baseline statistics (mean and standard deviation).
These statistics are used by the Statistical Validator to calculate z-scores and
confidence levels for watermark detection.

Tier 1 processes the corpus through the complete SpecHO pipeline, collects document
scores, and calculates basic statistics. The baseline is saved as a pickle file for
fast loading during detection.

Tier: 1 (MVP)
Task: 6.1
Dependencies: All prior components (complete pipeline required)
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import logging

from ..preprocessor.pipeline import LinguisticPreprocessor
from ..clause_identifier.pipeline import ClauseIdentifier
from ..echo_engine.pipeline import EchoAnalysisEngine
from ..scoring.pipeline import ScoringModule


class BaselineCorpusProcessor:
    """Process corpus of human/natural text to establish baseline statistics.

    The BaselineCorpusProcessor runs the complete SpecHO detection pipeline on
    a corpus of text files (typically human-written or natural text) to establish
    baseline statistics. These statistics represent the expected echo scores for
    text WITHOUT the watermark, providing a reference distribution for statistical
    validation.

    Tier 1 Implementation:
    - Process all .txt files in corpus directory
    - Run complete pipeline on each file
    - Calculate mean and standard deviation of document scores
    - Save/load statistics using pickle
    - Progress tracking with tqdm

    The baseline typically consists of:
    - human_mean: Average echo score for human/natural text (typically 0.1-0.2)
    - human_std: Standard deviation of scores (typically 0.05-0.15)

    These statistics are used by the Statistical Validator to calculate:
    - z_score = (document_score - human_mean) / human_std
    - confidence = norm.cdf(z_score)

    Examples:
        >>> # Process a corpus of human-written text
        >>> processor = BaselineCorpusProcessor()
        >>> stats = processor.process_corpus("data/corpus/human/")
        >>> print(stats)
        {'human_mean': 0.142, 'human_std': 0.087, 'n_documents': 50}

        >>> # Save baseline for use in detection
        >>> processor.save_baseline(stats, "data/baseline/human_stats.pkl")

        >>> # Load baseline in detector
        >>> loaded_stats = processor.load_baseline("data/baseline/human_stats.pkl")
        >>> print(loaded_stats['human_mean'])
        0.142

    Attributes:
        preprocessor: LinguisticPreprocessor instance
        clause_identifier: ClauseIdentifier instance
        echo_engine: EchoAnalysisEngine instance
        scoring_module: ScoringModule instance
        logger: Logger for progress and diagnostics
    """

    def __init__(self):
        """Initialize the baseline corpus processor with complete pipeline.

        Creates instances of all pipeline components. This is expensive
        (especially spaCy model loading), so BaselineCorpusProcessor should
        be instantiated once and reused for multiple corpus directories.

        Tier 1 uses default configurations for all components.
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Initialize complete pipeline
        self.logger.info("Initializing SpecHO pipeline for baseline processing...")
        self.preprocessor = LinguisticPreprocessor()
        self.clause_identifier = ClauseIdentifier()
        self.echo_engine = EchoAnalysisEngine()
        self.scoring_module = ScoringModule()
        self.logger.info("Pipeline initialization complete")

    def process_corpus(self, corpus_path: str) -> Dict[str, float]:
        """Process corpus directory to calculate baseline statistics.

        Processes all .txt files in the corpus directory through the complete
        SpecHO pipeline, collects document scores, and calculates mean and
        standard deviation. These statistics represent the baseline distribution
        for non-watermarked text.

        Algorithm (Tier 1):
        1. Find all .txt files in corpus_path
        2. For each file:
           a. Read text content
           b. Run through complete pipeline (preprocess → identify clauses →
              analyze echoes → score document)
           c. Collect document score
        3. Calculate mean and std of all scores
        4. Return statistics dictionary

        Args:
            corpus_path: Path to directory containing .txt corpus files.
                        Files should be UTF-8 encoded text files.

        Returns:
            Dictionary containing:
                - 'human_mean': Mean document score across corpus
                - 'human_std': Standard deviation of document scores
                - 'n_documents': Number of documents processed

        Raises:
            ValueError: If corpus_path doesn't exist or contains no .txt files
            RuntimeError: If pipeline fails on too many documents (>50%)

        Examples:
            >>> processor = BaselineCorpusProcessor()
            >>> stats = processor.process_corpus("data/corpus/human/")
            >>> stats['human_mean']
            0.142
            >>> stats['n_documents']
            50

            >>> # Check if baseline is reasonable
            >>> assert 0.0 <= stats['human_mean'] <= 0.3  # Human text should score low
            >>> assert stats['human_std'] > 0.0  # Should have some variance
        """
        # Validate corpus path
        corpus_dir = Path(corpus_path)
        if not corpus_dir.exists():
            raise ValueError(f"Corpus directory does not exist: {corpus_path}")

        # Find all .txt files
        txt_files = list(corpus_dir.glob("*.txt"))
        if len(txt_files) == 0:
            raise ValueError(f"No .txt files found in corpus directory: {corpus_path}")

        self.logger.info(f"Found {len(txt_files)} .txt files in {corpus_path}")

        # Process each file and collect scores
        document_scores = []
        failed_count = 0

        for txt_file in tqdm(txt_files, desc="Processing corpus", unit="files"):
            try:
                # Read file content
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Run through complete pipeline
                score = self._process_single_document(text)
                document_scores.append(score)

            except Exception as e:
                self.logger.warning(f"Failed to process {txt_file.name}: {e}")
                failed_count += 1

        # Check failure rate
        failure_rate = failed_count / len(txt_files) if txt_files else 0
        if failure_rate > 0.5:
            raise RuntimeError(
                f"Pipeline failed on {failure_rate:.1%} of documents. "
                f"This is too high for reliable baseline statistics."
            )

        # Calculate statistics
        if not document_scores:
            raise RuntimeError("No documents were successfully processed")

        scores_array = np.array(document_scores)
        baseline_stats = {
            'human_mean': float(np.mean(scores_array)),
            'human_std': float(np.std(scores_array, ddof=1)),  # Sample std deviation
            'n_documents': len(document_scores)
        }

        self.logger.info(
            f"Baseline statistics: mean={baseline_stats['human_mean']:.4f}, "
            f"std={baseline_stats['human_std']:.4f}, n={baseline_stats['n_documents']}"
        )

        return baseline_stats

    def _process_single_document(self, text: str) -> float:
        """Process a single document through the complete pipeline.

        Internal helper method that runs text through all pipeline components
        and returns the document score.

        Args:
            text: Raw text string to process

        Returns:
            Document score (float in [0,1] range)

        Raises:
            Exception: If any pipeline component fails
        """
        # Step 1: Linguistic preprocessing
        tokens, spacy_doc = self.preprocessor.process(text)

        # Step 2: Clause identification
        clause_pairs = self.clause_identifier.identify_pairs(tokens, spacy_doc)

        # Step 3: Echo analysis (analyze each pair individually)
        echo_scores = [
            self.echo_engine.analyze_pair(clause_pair)
            for clause_pair in clause_pairs
        ]

        # Step 4: Document scoring
        document_score = self.scoring_module.score_document(echo_scores)

        return document_score

    def save_baseline(self, baseline_stats: Dict[str, float], output_path: str) -> None:
        """Save baseline statistics to pickle file.

        Serializes the baseline statistics dictionary to a pickle file for
        fast loading during detection. Tier 1 uses pickle for simplicity;
        Tier 2 will add JSON support and versioning.

        Args:
            baseline_stats: Dictionary containing 'human_mean', 'human_std', 'n_documents'
            output_path: Path where baseline pickle file will be saved

        Examples:
            >>> processor = BaselineCorpusProcessor()
            >>> stats = {'human_mean': 0.142, 'human_std': 0.087, 'n_documents': 50}
            >>> processor.save_baseline(stats, "data/baseline/human_stats.pkl")
        """
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Save with pickle
        with open(output_path, 'wb') as f:
            pickle.dump(baseline_stats, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(f"Baseline statistics saved to {output_path}")

    def load_baseline(self, baseline_path: str) -> Dict[str, float]:
        """Load baseline statistics from pickle file.

        Deserializes baseline statistics from a pickle file. This is the
        standard way to load pre-computed baseline statistics for detection.

        Args:
            baseline_path: Path to baseline pickle file

        Returns:
            Dictionary containing 'human_mean', 'human_std', 'n_documents'

        Raises:
            FileNotFoundError: If baseline_path doesn't exist
            pickle.UnpicklingError: If file is corrupted or invalid

        Examples:
            >>> processor = BaselineCorpusProcessor()
            >>> stats = processor.load_baseline("data/baseline/human_stats.pkl")
            >>> print(f"Mean: {stats['human_mean']:.4f}")
            Mean: 0.1420
        """
        baseline_file = Path(baseline_path)
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {baseline_path}")

        with open(baseline_path, 'rb') as f:
            baseline_stats = pickle.load(f)

        self.logger.info(
            f"Loaded baseline: mean={baseline_stats['human_mean']:.4f}, "
            f"std={baseline_stats['human_std']:.4f}, "
            f"n={baseline_stats['n_documents']}"
        )

        return baseline_stats
