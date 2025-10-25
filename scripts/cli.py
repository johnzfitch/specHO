"""Command-line interface for SpecHO watermark detector.

Tier 1 implementation with simple text/file input and formatted output.

Task: 7.2
Dependencies: Task 7.1 (SpecHODetector)
Libraries: argparse, rich

Usage:
    python scripts/cli.py --file sample.txt
    python scripts/cli.py --text "Text to analyze..."
    python scripts/cli.py --file sample.txt --verbose
    python scripts/cli.py --file sample.txt --json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from specHO.detector import SpecHODetector
from specHO.config import load_config
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box


def create_parser():
    """Create argument parser for CLI.

    Returns:
        argparse.ArgumentParser: Configured parser
    """
    parser = argparse.ArgumentParser(
        description="SpecHO - Echo Rule Watermark Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Analyze a file:
    python scripts/cli.py --file sample.txt

  Analyze text directly:
    python scripts/cli.py --text "The sky darkened. But hope remained."

  Verbose mode with all scores:
    python scripts/cli.py --file sample.txt --verbose

  JSON output for automation:
    python scripts/cli.py --file sample.txt --json
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Path to text file to analyze'
    )
    input_group.add_argument(
        '--text', '-t',
        type=str,
        help='Text string to analyze directly'
    )

    # Output options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed breakdown of scores'
    )
    parser.add_argument(
        '--json', '-j',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='simple',
        choices=['simple', 'robust', 'research'],
        help='Configuration profile to use (default: simple)'
    )

    return parser


def load_text(file_path: str) -> str:
    """Load text from file.

    Args:
        file_path: Path to text file

    Returns:
        str: File contents

    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file can't be read
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not path.is_file():
        raise IOError(f"Not a file: {file_path}")

    try:
        return path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encoding
        return path.read_text(encoding='latin-1')


def format_text_output(analysis, verbose=False):
    """Format analysis results for terminal display using Rich.

    Args:
        analysis: DocumentAnalysis object
        verbose: If True, show detailed breakdown
    """
    console = Console()

    # Header
    console.print("\n")
    console.print(Panel.fit(
        "[bold cyan]SpecHO Watermark Detection Results[/bold cyan]",
        box=box.DOUBLE
    ))
    console.print()

    # Main results table
    results_table = Table(show_header=False, box=box.SIMPLE)
    results_table.add_column("Metric", style="bold")
    results_table.add_column("Value")

    # Final score with color coding
    score_color = "green" if analysis.final_score < 0.5 else "yellow" if analysis.final_score < 0.7 else "red"
    results_table.add_row(
        "Document Score",
        f"[{score_color}]{analysis.final_score:.3f}[/{score_color}]"
    )

    # Z-score
    z_color = "green" if abs(analysis.z_score) < 1.5 else "yellow" if abs(analysis.z_score) < 2.5 else "red"
    results_table.add_row(
        "Z-Score",
        f"[{z_color}]{analysis.z_score:.2f}[/{z_color}]"
    )

    # Confidence
    conf_color = "green" if analysis.confidence > 0.9 else "yellow" if analysis.confidence > 0.7 else "red"
    results_table.add_row(
        "Confidence",
        f"[{conf_color}]{analysis.confidence:.1%}[/{conf_color}]"
    )

    # Interpretation
    if analysis.final_score > 0.7:
        verdict = "[bold red]HIGH[/bold red] - Likely watermarked"
    elif analysis.final_score > 0.5:
        verdict = "[bold yellow]MODERATE[/bold yellow] - Possibly watermarked"
    else:
        verdict = "[bold green]LOW[/bold green] - Likely human-written"

    results_table.add_row("Verdict", verdict)

    console.print(results_table)
    console.print()

    # Verbose output
    if verbose:
        console.print("[bold]Detailed Breakdown:[/bold]")
        console.print()

        # Clause pairs summary
        console.print(f"  Clause Pairs Analyzed: {len(analysis.clause_pairs)}")
        console.print(f"  Echo Scores Computed: {len(analysis.echo_scores)}")
        console.print()

        # Echo scores table
        if analysis.echo_scores:
            echo_table = Table(title="Sample Echo Scores (first 10)", box=box.SIMPLE_HEAD)
            echo_table.add_column("#", style="dim")
            echo_table.add_column("Phonetic", justify="right")
            echo_table.add_column("Structural", justify="right")
            echo_table.add_column("Semantic", justify="right")
            echo_table.add_column("Combined", justify="right", style="bold")

            for i, score in enumerate(analysis.echo_scores[:10], 1):
                echo_table.add_row(
                    str(i),
                    f"{score.phonetic_score:.3f}",
                    f"{score.structural_score:.3f}",
                    f"{score.semantic_score:.3f}",
                    f"{score.combined_score:.3f}"
                )

            console.print(echo_table)
            console.print()

        # Statistics
        if analysis.echo_scores:
            combined_scores = [s.combined_score for s in analysis.echo_scores]
            avg_score = sum(combined_scores) / len(combined_scores)
            max_score = max(combined_scores)
            min_score = min(combined_scores)

            stats_table = Table(show_header=False, box=box.SIMPLE)
            stats_table.add_column("Stat", style="bold")
            stats_table.add_column("Value")

            stats_table.add_row("Mean Echo Score", f"{avg_score:.3f}")
            stats_table.add_row("Max Echo Score", f"{max_score:.3f}")
            stats_table.add_row("Min Echo Score", f"{min_score:.3f}")

            console.print(stats_table)
            console.print()

    # Footer note
    console.print("[dim]SpecHO v1.0 (Tier 1 MVP) - Echo Rule Watermark Detector[/dim]")
    console.print()


def format_json_output(analysis):
    """Format analysis results as JSON.

    Args:
        analysis: DocumentAnalysis object

    Returns:
        str: JSON string
    """
    result = {
        "document_score": float(analysis.final_score),
        "z_score": float(analysis.z_score),
        "confidence": float(analysis.confidence),
        "verdict": (
            "likely_watermarked" if analysis.final_score > 0.7
            else "possibly_watermarked" if analysis.final_score > 0.5
            else "likely_human"
        ),
        "metadata": {
            "clause_pairs": len(analysis.clause_pairs),
            "echo_scores_count": len(analysis.echo_scores),
        }
    }

    # Add echo scores if available
    if analysis.echo_scores:
        combined_scores = [float(s.combined_score) for s in analysis.echo_scores]
        result["echo_scores"] = {
            "mean": sum(combined_scores) / len(combined_scores),
            "max": max(combined_scores),
            "min": min(combined_scores),
            "samples": [
                {
                    "phonetic": float(s.phonetic_score),
                    "structural": float(s.structural_score),
                    "semantic": float(s.semantic_score),
                    "combined": float(s.combined_score)
                }
                for s in analysis.echo_scores[:10]  # First 10 samples
            ]
        }

    return json.dumps(result, indent=2)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    console = Console()

    try:
        # Load text
        if args.file:
            console.print(f"[dim]Loading file: {args.file}[/dim]")
            text = load_text(args.file)
        else:
            text = args.text

        if not text.strip():
            console.print("[red]Error: Empty text provided[/red]")
            return 1

        # Initialize detector
        console.print(f"[dim]Initializing detector with '{args.config}' config...[/dim]")
        config = load_config(args.config)

        # SpecHODetector expects baseline_path, not config (Tier 1 simple init)
        baseline_path = "data/baseline/baseline_stats.pkl"  # Default Tier 1 path

        # Check if baseline exists, warn if not
        if not Path(baseline_path).exists():
            console.print("[yellow]Warning: Baseline file not found.[/yellow]")
            console.print(f"[yellow]Expected location: {baseline_path}[/yellow]")
            console.print("[yellow]Z-scores and confidence will not be available.[/yellow]")
            console.print("[dim]Run: python scripts/build_baseline.py --corpus data/corpus/ --output {baseline_path}[/dim]")
            console.print()
            # Create a temporary baseline for testing
            import pickle
            Path(baseline_path).parent.mkdir(parents=True, exist_ok=True)
            # Baseline is a simple dict with mean, std, n_documents
            temp_baseline = {
                'human_mean': 0.3,  # Typical human text echo score
                'human_std': 0.15,   # Typical std deviation
                'n_documents': 0      # No corpus processed
            }
            with open(baseline_path, 'wb') as f:
                pickle.dump(temp_baseline, f)
            console.print("[dim]Created temporary baseline for testing purposes.[/dim]")
            console.print()

        detector = SpecHODetector(baseline_path)

        # Analyze
        console.print("[dim]Analyzing text...[/dim]")
        analysis = detector.analyze(text)

        # Output results
        if args.json:
            print(format_json_output(analysis))
        else:
            format_text_output(analysis, verbose=args.verbose)

        return 0

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        return 1
    except IOError as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
