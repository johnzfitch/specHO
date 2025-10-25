"""Build baseline statistics from corpus of human-written text.

Tier 1 implementation for processing corpus and saving baseline statistics.
This script should be run ONCE before using the detector to establish the
statistical baseline for comparison.

Task: 7.4
Dependencies: Task 6.1 (BaselineCorpusProcessor), Task 7.1 (SpecHODetector)
Libraries: argparse, tqdm, pickle

Usage:
    python scripts/build_baseline.py --corpus data/corpus/ --output data/baseline/baseline_stats.pkl
    python scripts/build_baseline.py --corpus data/corpus/ --output data/baseline/baseline_stats.pkl --limit 100
"""

import argparse
import pickle
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from specHO.validator.baseline_builder import BaselineCorpusProcessor
from specHO.config import load_config
from tqdm import tqdm


def create_parser():
    """Create argument parser for baseline builder.

    Returns:
        argparse.ArgumentParser: Configured parser
    """
    parser = argparse.ArgumentParser(
        description="Build baseline statistics from human-written text corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Build baseline from corpus directory:
    python scripts/build_baseline.py --corpus data/corpus/ --output data/baseline/baseline_stats.pkl

  Process only first 100 files:
    python scripts/build_baseline.py --corpus data/corpus/ --output data/baseline/baseline_stats.pkl --limit 100

  Use robust config profile:
    python scripts/build_baseline.py --corpus data/corpus/ --output data/baseline/baseline_stats.pkl --config robust

Notes:
  - Corpus directory should contain .txt or .md files
  - Files should be human-written, not AI-generated
  - Recommended: At least 50-100 documents for reliable baseline
  - This only needs to be run once (unless corpus changes)
        """
    )

    parser.add_argument(
        '--corpus', '-c',
        type=str,
        required=True,
        help='Path to directory containing human-written text files'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output path for baseline statistics file (.pkl)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of files to process (for testing)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='simple',
        choices=['simple', 'robust', 'research'],
        help='Configuration profile to use (default: simple)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed progress and statistics'
    )
    parser.add_argument(
        '--extensions',
        type=str,
        nargs='+',
        default=['.txt', '.md'],
        help='File extensions to process (default: .txt .md)'
    )

    return parser


def find_corpus_files(corpus_dir: Path, extensions: list, limit: int = None):
    """Find all text files in corpus directory.

    Args:
        corpus_dir: Path to corpus directory
        extensions: List of file extensions to include
        limit: Maximum number of files to return (None for all)

    Returns:
        List[Path]: List of file paths

    Raises:
        FileNotFoundError: If corpus directory doesn't exist
        ValueError: If no files found
    """
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

    if not corpus_dir.is_dir():
        raise ValueError(f"Not a directory: {corpus_dir}")

    # Find all files with matching extensions
    files = []
    for ext in extensions:
        files.extend(corpus_dir.rglob(f'*{ext}'))

    if not files:
        raise ValueError(f"No files with extensions {extensions} found in {corpus_dir}")

    # Sort for reproducibility
    files = sorted(files)

    # Apply limit if specified
    if limit is not None:
        files = files[:limit]

    return files


def validate_output_path(output_path: Path):
    """Validate output path is writable.

    Args:
        output_path: Path to output file

    Raises:
        ValueError: If output path is invalid
    """
    # Check parent directory exists
    if not output_path.parent.exists():
        raise ValueError(f"Output directory does not exist: {output_path.parent}")

    # Check we can write to it
    if not output_path.parent.is_dir():
        raise ValueError(f"Parent path is not a directory: {output_path.parent}")

    # Warn if file already exists
    if output_path.exists():
        print(f"Warning: Output file already exists and will be overwritten: {output_path}")


def main():
    """Main entry point for baseline builder."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        print("=" * 80)
        print("SpecHO Baseline Builder")
        print("=" * 80)
        print()

        # Validate paths
        corpus_dir = Path(args.corpus)
        output_path = Path(args.output)

        print(f"Corpus directory: {corpus_dir}")
        print(f"Output file: {output_path}")
        print(f"Configuration: {args.config}")
        print()

        # Find files
        print("Scanning corpus directory...")
        files = find_corpus_files(corpus_dir, args.extensions, args.limit)
        print(f"Found {len(files)} files to process")

        if args.limit:
            print(f"(Limited to first {args.limit} files)")
        print()

        # Validate output path
        validate_output_path(output_path)

        # Initialize processor
        print("Initializing baseline processor...")
        config = load_config(args.config)
        processor = BaselineCorpusProcessor(config)
        print("Processor initialized")
        print()

        # Process corpus
        print("Processing corpus files...")
        print("(This may take several minutes depending on corpus size)")
        print()

        # Create text list for processor
        texts = []
        failed_files = []

        # Read all files with progress bar
        for file_path in tqdm(files, desc="Reading files", unit="file"):
            try:
                text = file_path.read_text(encoding='utf-8')
                texts.append(text)
            except UnicodeDecodeError:
                try:
                    text = file_path.read_text(encoding='latin-1')
                    texts.append(text)
                except Exception as e:
                    if args.verbose:
                        print(f"  [ERROR] Failed to read {file_path.name}: {e}")
                    failed_files.append(file_path)
            except Exception as e:
                if args.verbose:
                    print(f"  [ERROR] Failed to read {file_path.name}: {e}")
                failed_files.append(file_path)

        if failed_files:
            print(f"\nWarning: Failed to read {len(failed_files)} files")
            if args.verbose:
                for f in failed_files[:10]:
                    print(f"  - {f.name}")
                if len(failed_files) > 10:
                    print(f"  ... and {len(failed_files) - 10} more")
            print()

        if not texts:
            print("Error: No texts successfully loaded from corpus")
            return 1

        # Process corpus
        print(f"Processing {len(texts)} documents through pipeline...")
        baseline_stats = processor.build_baseline(texts)

        # Display statistics
        print()
        print("=" * 80)
        print("BASELINE STATISTICS")
        print("=" * 80)
        print()
        print(f"  Documents processed: {baseline_stats.num_documents}")
        print(f"  Mean score:          {baseline_stats.mean:.4f}")
        print(f"  Std deviation:       {baseline_stats.std_dev:.4f}")
        print()

        if args.verbose and len(baseline_stats.document_scores) > 0:
            scores = baseline_stats.document_scores
            print("  Score distribution:")
            print(f"    Min:     {min(scores):.4f}")
            print(f"    25th %:  {sorted(scores)[len(scores) // 4]:.4f}")
            print(f"    Median:  {sorted(scores)[len(scores) // 2]:.4f}")
            print(f"    75th %:  {sorted(scores)[3 * len(scores) // 4]:.4f}")
            print(f"    Max:     {max(scores):.4f}")
            print()

        # Save baseline
        print(f"Saving baseline statistics to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            pickle.dump(baseline_stats, f)

        print("Baseline statistics saved successfully")
        print()

        # Verification
        file_size = output_path.stat().st_size
        print(f"Output file size: {file_size:,} bytes")
        print()

        # Success message
        print("=" * 80)
        print("SUCCESS - Baseline built successfully")
        print("=" * 80)
        print()
        print("You can now use SpecHO detector with this baseline:")
        print(f"  1. The baseline file is located at: {output_path}")
        print(f"  2. Configure your detector to use this baseline")
        print(f"  3. Run analysis on new documents")
        print()

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
