#!/usr/bin/env python3
"""SpecHO Command Center - Interactive TUI for pipeline testing and analysis.

A terminal-based command center for running the SpecHO watermark detection pipeline
with interactive controls, real-time logging, and detailed analysis visualization.

Usage:
    python scripts/command_center.py

Requirements:
    pip install textual
"""

import sys
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime
import time

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Button, Static, Label,
    DataTable, ProgressBar, Log, TabbedContent, TabPane
)
from textual.reactive import reactive
from textual import work
from rich.text import Text
from rich.panel import Panel
from rich.table import Table as RichTable

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from specHO.detector import SpecHODetector
from specHO.models import DocumentAnalysis


class PipelineStats(Static):
    """Widget to display pipeline statistics."""

    runs_completed = reactive(0)
    current_file = reactive("")

    def render(self) -> Panel:
        """Render the statistics panel."""
        table = RichTable.grid(padding=(0, 2))
        table.add_column(justify="right", style="cyan")
        table.add_column(justify="left", style="green")

        table.add_row("Total Runs:", str(self.runs_completed))
        table.add_row("Current File:", self.current_file or "None")
        table.add_row("Status:", "Ready" if not self.current_file else "Processing...")

        return Panel(
            table,
            title="[bold cyan]Pipeline Statistics[/bold cyan]",
            border_style="cyan"
        )


class ResultsDisplay(Static):
    """Widget to display analysis results."""

    latest_result = reactive(None)

    def render(self) -> Panel:
        """Render the results panel."""
        if not self.latest_result:
            content = "[dim]No results yet. Run analysis on a file to see results.[/dim]"
        else:
            result: DocumentAnalysis = self.latest_result

            table = RichTable.grid(padding=(0, 2))
            table.add_column(justify="right", style="cyan")
            table.add_column(justify="left")

            # Document score
            score_color = "green" if result.document_score < 0.3 else "yellow" if result.document_score < 0.6 else "red"
            table.add_row("Document Score:", f"[{score_color}]{result.document_score:.4f}[/{score_color}]")

            # Z-score
            z_color = "green" if abs(result.z_score) < 2 else "yellow" if abs(result.z_score) < 3 else "red"
            table.add_row("Z-Score:", f"[{z_color}]{result.z_score:.4f}[/{z_color}]")

            # Confidence
            conf_color = "green" if result.confidence < 0.8 else "yellow" if result.confidence < 0.95 else "red"
            table.add_row("Confidence:", f"[{conf_color}]{result.confidence:.4f}[/{conf_color}]")

            # Classification
            classification = "HUMAN" if result.confidence < 0.5 else "UNCERTAIN" if result.confidence < 0.95 else "WATERMARKED"
            class_color = "green" if classification == "HUMAN" else "yellow" if classification == "UNCERTAIN" else "red"
            table.add_row("Classification:", f"[bold {class_color}]{classification}[/bold {class_color}]")

            table.add_row("", "")
            table.add_row("Tokens:", str(len(result.tokens)))
            table.add_row("Clause Pairs:", str(len(result.clause_pairs)))
            table.add_row("Echo Scores:", str(len(result.echo_scores)))

            content = table

        return Panel(
            content,
            title="[bold green]Latest Results[/bold green]",
            border_style="green"
        )


class DetailedAnalysis(Static):
    """Widget to display detailed component-by-component analysis."""

    analysis_data = reactive(None)

    def render(self) -> Panel:
        """Render detailed analysis."""
        if not self.analysis_data:
            content = "[dim]Run an analysis to see detailed breakdown.[/dim]"
        else:
            result: DocumentAnalysis = self.analysis_data

            lines = []
            lines.append("[bold cyan]PREPROCESSING[/bold cyan]")
            lines.append(f"  Tokens: {len(result.tokens)}")
            lines.append(f"  Avg token length: {sum(len(t.text) for t in result.tokens) / len(result.tokens):.1f} chars")
            lines.append("")

            lines.append("[bold cyan]CLAUSE IDENTIFICATION[/bold cyan]")
            lines.append(f"  Clause pairs: {len(result.clause_pairs)}")
            if result.clause_pairs:
                avg_zone = sum(abs(cp.zone_index) for cp in result.clause_pairs) / len(result.clause_pairs)
                lines.append(f"  Avg zone distance: {avg_zone:.2f}")
            lines.append("")

            lines.append("[bold cyan]ECHO ANALYSIS[/bold cyan]")
            lines.append(f"  Echo scores: {len(result.echo_scores)}")
            if result.echo_scores:
                avg_phon = sum(es.phonetic_score for es in result.echo_scores) / len(result.echo_scores)
                avg_struct = sum(es.structural_score for es in result.echo_scores) / len(result.echo_scores)
                avg_sem = sum(es.semantic_score for es in result.echo_scores) / len(result.echo_scores)
                lines.append(f"  Avg phonetic: {avg_phon:.3f}")
                lines.append(f"  Avg structural: {avg_struct:.3f}")
                lines.append(f"  Avg semantic: {avg_sem:.3f}")
            lines.append("")

            lines.append("[bold cyan]SCORING[/bold cyan]")
            lines.append(f"  Document score: {result.document_score:.4f}")
            lines.append("")

            lines.append("[bold cyan]VALIDATION[/bold cyan]")
            lines.append(f"  Z-score: {result.z_score:.4f}")
            lines.append(f"  Confidence: {result.confidence:.4f}")

            content = "\n".join(lines)

        return Panel(
            content,
            title="[bold yellow]Detailed Analysis[/bold yellow]",
            border_style="yellow"
        )


class CommandCenter(App):
    """Main application class for the SpecHO Command Center."""

    CSS = """
    Screen {
        background: $surface;
    }

    #header-banner {
        height: 3;
        content-align: center middle;
        background: $primary;
        color: $text;
    }

    #main-container {
        height: 1fr;
    }

    #left-panel {
        width: 30%;
        height: 100%;
    }

    #right-panel {
        width: 70%;
        height: 100%;
    }

    #controls {
        height: auto;
        padding: 1;
        border: solid $primary;
    }

    #stats-panel {
        height: 8;
        padding: 0;
    }

    #results-panel {
        height: 20;
        padding: 0;
    }

    #detailed-panel {
        height: 1fr;
        padding: 0;
    }

    Button {
        margin: 1;
        width: 100%;
    }

    Button.run-button {
        background: $success;
    }

    Button.run-button:hover {
        background: $success-darken-1;
    }

    Button.clear-button {
        background: $warning;
    }

    Button.exit-button {
        background: $error;
    }

    Log {
        height: 100%;
        border: solid $accent;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("1", "run_sample1", "Run sample.txt"),
        ("2", "run_sample2", "Run sample2.md"),
        ("c", "clear_log", "Clear Log"),
    ]

    def __init__(self):
        super().__init__()
        self.detector: Optional[SpecHODetector] = None
        self.sample1_path = Path("specHO/sample.txt")
        self.sample2_path = Path("specHO/sample2.md")
        self.baseline_path = Path("data/baseline/baseline_stats.pkl")

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header(show_clock=True)

        # ASCII art banner
        banner = Static(
            "[bold cyan]╔═══════════════════════════════════════════════════════════╗[/bold cyan]\n"
            "[bold cyan]║[/bold cyan]  [bold white]SpecHO COMMAND CENTER[/bold white] - Echo Rule Detection  [bold cyan]║[/bold cyan]\n"
            "[bold cyan]╚═══════════════════════════════════════════════════════════╝[/bold cyan]",
            id="header-banner"
        )
        yield banner

        # Main container with left and right panels
        with Container(id="main-container"):
            with Horizontal():
                # Left panel: controls and stats
                with Vertical(id="left-panel"):
                    # Control buttons
                    with Container(id="controls"):
                        yield Static("[bold]Quick Actions[/bold]", classes="section-title")
                        yield Button("🚀 Run sample.txt (3x)", id="run-sample1", classes="run-button")
                        yield Button("🚀 Run sample2.md (3x)", id="run-sample2", classes="run-button")
                        yield Button("🔍 Single: sample.txt", id="single-sample1", classes="run-button")
                        yield Button("🔍 Single: sample2.md", id="single-sample2", classes="run-button")
                        yield Button("🧹 Clear Log", id="clear-log", classes="clear-button")
                        yield Button("❌ Exit", id="exit-app", classes="exit-button")

                    # Statistics
                    yield PipelineStats(id="stats-panel")

                    # Results
                    yield ResultsDisplay(id="results-panel")

                    # Detailed analysis
                    yield DetailedAnalysis(id="detailed-panel")

                # Right panel: logs and detailed output
                with Vertical(id="right-panel"):
                    yield Log(id="pipeline-log")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the application."""
        self.title = "SpecHO Command Center"
        self.sub_title = "Interactive Pipeline Testing"

        log = self.query_one("#pipeline-log", Log)
        log.write_line("[bold green]═══ SpecHO Command Center Initialized ═══[/bold green]")
        log.write_line("")
        log.write_line("Welcome to the SpecHO watermark detection command center!")
        log.write_line("")
        log.write_line("[cyan]Available actions:[/cyan]")
        log.write_line("  • Click buttons on the left to run analyses")
        log.write_line("  • Press keyboard shortcuts (1, 2, c, q)")
        log.write_line("  • Watch real-time results in the panels")
        log.write_line("")

        # Initialize detector
        try:
            self.detector = SpecHODetector(baseline_path=str(self.baseline_path))
            log.write_line(f"[green]✓[/green] Detector initialized with baseline: {self.baseline_path}")
        except Exception as e:
            log.write_line(f"[red]✗[/red] Error initializing detector: {e}")
            log.write_line("[yellow]⚠[/yellow] You may need to build the baseline first")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id

        if button_id == "run-sample1":
            self.action_run_sample1()
        elif button_id == "run-sample2":
            self.action_run_sample2()
        elif button_id == "single-sample1":
            self.run_single_analysis(self.sample1_path)
        elif button_id == "single-sample2":
            self.run_single_analysis(self.sample2_path)
        elif button_id == "clear-log":
            self.action_clear_log()
        elif button_id == "exit-app":
            self.action_quit()

    @work(thread=True)
    def run_single_analysis(self, file_path: Path) -> None:
        """Run analysis on a single file."""
        log = self.query_one("#pipeline-log", Log)
        stats = self.query_one("#stats-panel", PipelineStats)
        results = self.query_one("#results-panel", ResultsDisplay)
        detailed = self.query_one("#detailed-panel", DetailedAnalysis)

        if not self.detector:
            log.write_line("[red]Error: Detector not initialized[/red]")
            return

        if not file_path.exists():
            log.write_line(f"[red]Error: File not found: {file_path}[/red]")
            return

        # Update stats
        stats.current_file = file_path.name

        log.write_line("")
        log.write_line(f"[bold cyan]{'─' * 60}[/bold cyan]")
        log.write_line(f"[bold]▶ Running analysis: {file_path.name}[/bold]")
        log.write_line(f"[bold cyan]{'─' * 60}[/bold cyan]")

        try:
            # Read file
            text = file_path.read_text(encoding='utf-8')
            log.write_line(f"[green]✓[/green] Loaded {len(text)} characters")

            # Run pipeline with timing
            start_time = time.time()
            log.write_line("[cyan]⚙ Processing through pipeline...[/cyan]")

            result = self.detector.analyze(text)

            elapsed = time.time() - start_time
            log.write_line(f"[green]✓[/green] Analysis complete in {elapsed:.2f}s")
            log.write_line("")

            # Display results
            log.write_line("[bold yellow]Results:[/bold yellow]")
            log.write_line(f"  Document Score: {result.document_score:.4f}")
            log.write_line(f"  Z-Score: {result.z_score:.4f}")
            log.write_line(f"  Confidence: {result.confidence:.4f}")

            classification = "HUMAN" if result.confidence < 0.5 else "UNCERTAIN" if result.confidence < 0.95 else "WATERMARKED"
            log.write_line(f"  Classification: [bold]{classification}[/bold]")
            log.write_line("")

            log.write_line("[bold yellow]Component Stats:[/bold yellow]")
            log.write_line(f"  Tokens: {len(result.tokens)}")
            log.write_line(f"  Clause Pairs: {len(result.clause_pairs)}")
            log.write_line(f"  Echo Scores: {len(result.echo_scores)}")

            # Update reactive widgets
            results.latest_result = result
            detailed.analysis_data = result
            stats.runs_completed += 1
            stats.current_file = ""

        except Exception as e:
            log.write_line(f"[red]✗ Error during analysis: {e}[/red]")
            import traceback
            log.write_line(f"[dim]{traceback.format_exc()}[/dim]")
            stats.current_file = ""

    def action_run_sample1(self) -> None:
        """Run sample.txt 3 times."""
        log = self.query_one("#pipeline-log", Log)
        log.write_line("")
        log.write_line("[bold green]═══ Running sample.txt 3 times ═══[/bold green]")

        for i in range(3):
            log.write_line(f"\n[bold]Run {i+1}/3[/bold]")
            self.run_single_analysis(self.sample1_path)

    def action_run_sample2(self) -> None:
        """Run sample2.md 3 times."""
        log = self.query_one("#pipeline-log", Log)
        log.write_line("")
        log.write_line("[bold green]═══ Running sample2.md 3 times ═══[/bold green]")

        for i in range(3):
            log.write_line(f"\n[bold]Run {i+1}/3[/bold]")
            self.run_single_analysis(self.sample2_path)

    def action_clear_log(self) -> None:
        """Clear the log panel."""
        log = self.query_one("#pipeline-log", Log)
        log.clear()
        log.write_line("[bold green]═══ Log Cleared ═══[/bold green]")


def main():
    """Run the command center."""
    app = CommandCenter()
    app.run()


if __name__ == "__main__":
    main()
