#!/usr/bin/env python3
"""Batch pipeline testing script for comprehensive analysis.

Runs sample files through the pipeline multiple times and analyzes:
- Consistency across runs
- Data integrity at each stage
- Performance metrics
- Potential bugs or inefficiencies

Outputs detailed report to console and file.
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from specHO.detector import SpecHODetector
from specHO.models import DocumentAnalysis


@dataclass
class PipelineRun:
    """Results from a single pipeline run."""
    file_name: str
    run_number: int
    duration: float
    final_score: float
    z_score: float
    confidence: float
    clause_pair_count: int
    echo_score_count: int
    classification: str
    errors: List[str] = field(default_factory=list)


@dataclass
class AnalysisIssue:
    """A potential issue found during analysis."""
    severity: str  # "ERROR", "WARNING", "INFO"
    component: str
    description: str
    details: Dict[str, Any] = field(default_factory=dict)


class BatchTester:
    """Comprehensive batch testing for the pipeline."""

    def __init__(self, baseline_path: Path):
        self.detector = SpecHODetector(baseline_path=str(baseline_path))
        self.runs: List[PipelineRun] = []
        self.issues: List[AnalysisIssue] = []

    def run_file_analysis(self, file_path: Path, repetitions: int = 3) -> List[PipelineRun]:
        """Run analysis on a file multiple times."""
        print(f"\n{'='*70}")
        print(f"TESTING: {file_path.name} ({repetitions} runs)")
        print(f"{'='*70}\n")

        text = file_path.read_text(encoding='utf-8')
        file_runs = []

        for run_num in range(1, repetitions + 1):
            print(f"Run {run_num}/{repetitions}...", end=" ", flush=True)

            start_time = time.time()
            try:
                result = self.detector.analyze(text)
                duration = time.time() - start_time

                classification = self._classify_result(result)

                run = PipelineRun(
                    file_name=file_path.name,
                    run_number=run_num,
                    duration=duration,
                    final_score=result.final_score,
                    z_score=result.z_score,
                    confidence=result.confidence,
                    clause_pair_count=len(result.clause_pairs),
                    echo_score_count=len(result.echo_scores),
                    classification=classification
                )

                file_runs.append(run)
                self.runs.append(run)

                print(f"OK {duration:.2f}s | Score: {result.final_score:.4f} | {classification}")

            except Exception as e:
                duration = time.time() - start_time
                print(f"ERROR: {e}")

                run = PipelineRun(
                    file_name=file_path.name,
                    run_number=run_num,
                    duration=duration,
                    final_score=0.0,
                    z_score=0.0,
                    confidence=0.0,
                    clause_pair_count=0,
                    echo_score_count=0,
                    classification="ERROR",
                    errors=[str(e)]
                )
                file_runs.append(run)
                self.runs.append(run)

                self.issues.append(AnalysisIssue(
                    severity="ERROR",
                    component="Pipeline",
                    description=f"Analysis failed on {file_path.name} run {run_num}",
                    details={"error": str(e)}
                ))

        return file_runs

    def _classify_result(self, result: DocumentAnalysis) -> str:
        """Classify result based on confidence."""
        if result.confidence < 0.5:
            return "HUMAN"
        elif result.confidence < 0.95:
            return "UNCERTAIN"
        else:
            return "WATERMARKED"

    def check_consistency(self, runs: List[PipelineRun]) -> None:
        """Check for consistency issues across runs."""
        if len(runs) < 2:
            return

        print(f"\n--- Consistency Analysis ---")

        # Check final scores
        scores = [r.final_score for r in runs if r.classification != "ERROR"]
        if scores:
            score_variance = max(scores) - min(scores)
            print(f"Final Score Variance: {score_variance:.6f}")
            if score_variance > 0.0001:  # Should be identical
                self.issues.append(AnalysisIssue(
                    severity="ERROR",
                    component="Scoring",
                    description=f"Final scores vary across runs for {runs[0].file_name}",
                    details={
                        "min": min(scores),
                        "max": max(scores),
                        "variance": score_variance
                    }
                ))
                print(f"  WARNING: Scores should be identical but vary by {score_variance:.6f}")

        # Check token counts
        token_counts = [r.token_count for r in runs if r.classification != "ERROR"]
        if token_counts and len(set(token_counts)) > 1:
            self.issues.append(AnalysisIssue(
                severity="ERROR",
                component="Preprocessor",
                description=f"Token counts vary across runs for {runs[0].file_name}",
                details={"counts": token_counts}
            ))
            print(f"  ⚠️ ERROR: Token counts vary: {set(token_counts)}")

        # Check clause pair counts
        pair_counts = [r.clause_pair_count for r in runs if r.classification != "ERROR"]
        if pair_counts and len(set(pair_counts)) > 1:
            self.issues.append(AnalysisIssue(
                severity="ERROR",
                component="ClauseIdentifier",
                description=f"Clause pair counts vary across runs for {runs[0].file_name}",
                details={"counts": pair_counts}
            ))
            print(f"  ⚠️ ERROR: Clause pair counts vary: {set(pair_counts)}")

        # Check performance variance
        durations = [r.duration for r in runs if r.classification != "ERROR"]
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_deviation = max(abs(d - avg_duration) for d in durations)
            print(f"Performance: {avg_duration:.2f}s avg, max deviation {max_deviation:.2f}s")

            if max_deviation > avg_duration * 0.5:  # >50% deviation
                self.issues.append(AnalysisIssue(
                    severity="WARNING",
                    component="Performance",
                    description=f"High performance variance for {runs[0].file_name}",
                    details={
                        "avg": avg_duration,
                        "max_deviation": max_deviation,
                        "all_durations": durations
                    }
                ))
                print(f"  ⚠️ WARNING: Performance variance exceeds 50%")

        if not any(issue.component in ["Scoring", "Preprocessor", "ClauseIdentifier"]
                   for issue in self.issues if issue.severity == "ERROR"):
            print("  ✓ All runs produced consistent results")

    def analyze_preprocessing(self, runs: List[PipelineRun]) -> None:
        """Analyze preprocessing stage for issues."""
        print(f"\n--- Preprocessing Analysis ---")

        for run in runs:
            if run.classification == "ERROR":
                continue

            # Check for zero tokens
            if run.token_count == 0:
                self.issues.append(AnalysisIssue(
                    severity="ERROR",
                    component="Preprocessor",
                    description=f"Zero tokens produced for {run.file_name}",
                    details={"run": run.run_number}
                ))
                print(f"  ✗ Run {run.run_number}: Zero tokens!")

            # Check for suspiciously low clause pairs
            if run.clause_pair_count < run.token_count * 0.01:  # Less than 1% seems low
                self.issues.append(AnalysisIssue(
                    severity="WARNING",
                    component="ClauseIdentifier",
                    description=f"Suspiciously low clause pair count for {run.file_name}",
                    details={
                        "run": run.run_number,
                        "tokens": run.token_count,
                        "pairs": run.clause_pair_count,
                        "ratio": run.clause_pair_count / run.token_count if run.token_count > 0 else 0
                    }
                ))
                print(f"  ⚠️ Run {run.run_number}: Low clause pair ratio "
                      f"({run.clause_pair_count}/{run.token_count} = "
                      f"{100*run.clause_pair_count/run.token_count:.1f}%)")

            # Check echo score count matches clause pairs
            if run.echo_score_count != run.clause_pair_count:
                self.issues.append(AnalysisIssue(
                    severity="ERROR",
                    component="EchoEngine",
                    description=f"Echo score count mismatch for {run.file_name}",
                    details={
                        "run": run.run_number,
                        "clause_pairs": run.clause_pair_count,
                        "echo_scores": run.echo_score_count
                    }
                ))
                print(f"  ✗ Run {run.run_number}: Echo scores ({run.echo_score_count}) "
                      f"!= clause pairs ({run.clause_pair_count})")

    def print_summary(self) -> None:
        """Print comprehensive test summary."""
        print(f"\n{'='*70}")
        print("TEST SUMMARY")
        print(f"{'='*70}\n")

        # Count issues by severity
        errors = [i for i in self.issues if i.severity == "ERROR"]
        warnings = [i for i in self.issues if i.severity == "WARNING"]

        print(f"Total Runs: {len(self.runs)}")
        print(f"Successful: {len([r for r in self.runs if r.classification != 'ERROR'])}")
        print(f"Errors: {len(errors)}")
        print(f"Warnings: {len(warnings)}")
        print()

        if errors:
            print("ERRORS FOUND:")
            for issue in errors:
                print(f"  ❌ [{issue.component}] {issue.description}")
                if issue.details:
                    print(f"     Details: {issue.details}")
            print()

        if warnings:
            print("WARNINGS:")
            for issue in warnings:
                print(f"  ⚠️  [{issue.component}] {issue.description}")
                if issue.details:
                    print(f"     Details: {issue.details}")
            print()

        if not errors and not warnings:
            print("✓ No issues found - pipeline is functioning correctly!")

        # Performance summary
        durations = [r.duration for r in self.runs if r.classification != "ERROR"]
        if durations:
            print(f"\nPerformance:")
            print(f"  Average: {sum(durations)/len(durations):.2f}s")
            print(f"  Min: {min(durations):.2f}s")
            print(f"  Max: {max(durations):.2f}s")

    def save_detailed_report(self, output_path: Path) -> None:
        """Save detailed JSON report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "runs": [
                {
                    "file_name": r.file_name,
                    "run_number": r.run_number,
                    "duration": r.duration,
                    "document_score": r.document_score,
                    "z_score": r.z_score,
                    "confidence": r.confidence,
                    "token_count": r.token_count,
                    "clause_pair_count": r.clause_pair_count,
                    "echo_score_count": r.echo_score_count,
                    "classification": r.classification,
                    "errors": r.errors
                }
                for r in self.runs
            ],
            "issues": [
                {
                    "severity": i.severity,
                    "component": i.component,
                    "description": i.description,
                    "details": i.details
                }
                for i in self.issues
            ],
            "summary": {
                "total_runs": len(self.runs),
                "successful_runs": len([r for r in self.runs if r.classification != "ERROR"]),
                "error_count": len([i for i in self.issues if i.severity == "ERROR"]),
                "warning_count": len([i for i in self.issues if i.severity == "WARNING"])
            }
        }

        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n📄 Detailed report saved to: {output_path}")


def main():
    """Run batch pipeline tests."""
    print("SpecHO Pipeline Batch Tester")
    print("=" * 70)

    # Paths
    baseline_path = Path("data/baseline/demo_baseline.pkl")
    sample1_path = Path("specHO/sample.txt")
    sample2_path = Path("specHO/sample2.md")
    output_path = Path("test_results.json")

    # Initialize tester
    tester = BatchTester(baseline_path)

    # Test sample.txt
    if sample1_path.exists():
        runs1 = tester.run_file_analysis(sample1_path, repetitions=3)
        tester.check_consistency(runs1)
        tester.analyze_preprocessing(runs1)
    else:
        print(f"⚠️ {sample1_path} not found, skipping")

    # Test sample2.md
    if sample2_path.exists():
        runs2 = tester.run_file_analysis(sample2_path, repetitions=3)
        tester.check_consistency(runs2)
        tester.analyze_preprocessing(runs2)
    else:
        print(f"⚠️ {sample2_path} not found, skipping")

    # Print summary
    tester.print_summary()

    # Save detailed report
    tester.save_detailed_report(output_path)


if __name__ == "__main__":
    main()
