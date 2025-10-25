# Score Tracking & Storage Enhancement Proposal

## Current State

Currently, SpecHO outputs analysis results to:
1. **Console** (via CLI with Rich formatting)
2. **Python return values** (DocumentAnalysis dataclass)
3. **Test artifacts** (test_results.log, various output files)

**Limitation**: No persistent storage or historical tracking of analysis results.

---

## Proposed Enhancement: Analysis History Database

### Option 1: JSON-based Result Log (Simple)

**File**: `data/analysis_history.jsonl` (JSON Lines format)

**Benefits**:
- No additional dependencies
- Human-readable
- Easy to append
- Simple to parse

**Structure**:
```json
{
  "timestamp": "2025-10-25T09:30:45Z",
  "version": "1.0.0",
  "input": {
    "source": "test_sample_human.txt",
    "text_hash": "sha256:abc123...",
    "word_count": 135,
    "char_count": 750
  },
  "results": {
    "document_score": 0.303,
    "z_score": 0.02,
    "confidence": 0.509,
    "verdict": "LOW",
    "classification": "human-written"
  },
  "metrics": {
    "clause_pairs": 6,
    "processing_time": 1.82,
    "echo_scores": {
      "mean_phonetic": 0.291,
      "mean_structural": 0.199,
      "mean_semantic": 0.417,
      "distribution": {
        "low": 2,
        "medium": 3,
        "high": 1
      }
    }
  },
  "config": {
    "profile": "simple",
    "weights": {
      "phonetic": 0.4,
      "structural": 0.3,
      "semantic": 0.3
    }
  }
}
```

**Implementation**:
```python
# In specHO/utils.py
def save_analysis_to_history(analysis: DocumentAnalysis,
                               source: str,
                               history_path: Path = None):
    """Append analysis results to history log."""
    if history_path is None:
        history_path = Path("data/analysis_history.jsonl")

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
        "input": {
            "source": source,
            "text_hash": hashlib.sha256(analysis.text.encode()).hexdigest()[:16],
            "word_count": len(analysis.text.split()),
            "char_count": len(analysis.text)
        },
        "results": {
            "document_score": analysis.final_score,
            "z_score": analysis.z_score,
            "confidence": analysis.confidence,
            "verdict": classify_score(analysis.final_score),
            "classification": interpret_confidence(analysis.confidence)
        },
        "metrics": extract_metrics(analysis),
        "config": analysis.config_profile
    }

    with open(history_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')


# CLI integration
def main():
    # ... existing code ...
    analysis = detector.analyze(text)

    # Save to history if --save flag
    if args.save:
        save_analysis_to_history(analysis, args.file or "stdin")

    # ... output code ...
```

**Usage**:
```bash
# Analyze and save to history
python scripts/cli.py --file test.txt --save

# View history
python scripts/view_history.py --tail 10
python scripts/view_history.py --source "test_sample_human.txt"
python scripts/view_history.py --date 2025-10-25
```

---

### Option 2: SQLite Database (Robust)

**File**: `data/specho.db`

**Benefits**:
- Structured queries
- Better for large datasets
- Built-in indexing
- ACID compliance

**Schema**:
```sql
CREATE TABLE analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    version TEXT,
    source TEXT,
    text_hash TEXT,
    word_count INTEGER,
    char_count INTEGER,
    document_score REAL,
    z_score REAL,
    confidence REAL,
    verdict TEXT,
    clause_pairs INTEGER,
    processing_time REAL,
    config_profile TEXT,
    UNIQUE(text_hash, timestamp)
);

CREATE TABLE echo_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    pair_index INTEGER,
    phonetic_score REAL,
    structural_score REAL,
    semantic_score REAL,
    combined_score REAL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE TABLE performance_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id INTEGER,
    stage TEXT,  -- 'preprocessing', 'clause_id', 'echo', 'scoring', 'validation'
    duration REAL,
    FOREIGN KEY (analysis_id) REFERENCES analyses(id)
);

CREATE INDEX idx_timestamp ON analyses(timestamp);
CREATE INDEX idx_source ON analyses(source);
CREATE INDEX idx_verdict ON analyses(verdict);
CREATE INDEX idx_text_hash ON analyses(text_hash);
```

**Implementation**:
```python
# specHO/tracking/database.py
import sqlite3
from pathlib import Path
from typing import List, Dict
from dataclasses import asdict

class AnalysisTracker:
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = Path("data/specho.db")

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def save_analysis(self, analysis: DocumentAnalysis, source: str,
                      metrics: Dict = None):
        """Save analysis to database."""
        cursor = self.conn.cursor()

        # Insert main analysis
        cursor.execute("""
            INSERT INTO analyses (
                timestamp, version, source, text_hash,
                word_count, char_count, document_score,
                z_score, confidence, verdict,
                clause_pairs, processing_time, config_profile
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            "1.0.0",
            source,
            hashlib.sha256(analysis.text.encode()).hexdigest()[:16],
            len(analysis.text.split()),
            len(analysis.text),
            analysis.final_score,
            analysis.z_score,
            analysis.confidence,
            classify_score(analysis.final_score),
            len(analysis.clause_pairs),
            metrics.get('total_time', 0) if metrics else None,
            "simple"
        ))

        analysis_id = cursor.lastrowid

        # Insert echo scores
        for i, echo_score in enumerate(analysis.echo_scores):
            cursor.execute("""
                INSERT INTO echo_scores (
                    analysis_id, pair_index,
                    phonetic_score, structural_score,
                    semantic_score, combined_score
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                analysis_id, i,
                echo_score.phonetic_score,
                echo_score.structural_score,
                echo_score.semantic_score,
                echo_score.combined_score
            ))

        self.conn.commit()
        return analysis_id

    def get_history(self, limit: int = 10, source: str = None):
        """Retrieve analysis history."""
        cursor = self.conn.cursor()

        query = "SELECT * FROM analyses"
        params = []

        if source:
            query += " WHERE source = ?"
            params.append(source)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        return cursor.execute(query, params).fetchall()

    def get_statistics(self, days: int = 7):
        """Get aggregate statistics."""
        cursor = self.conn.cursor()

        return cursor.execute("""
            SELECT
                COUNT(*) as total_analyses,
                AVG(document_score) as mean_score,
                AVG(confidence) as mean_confidence,
                SUM(CASE WHEN verdict = 'HIGH' THEN 1 ELSE 0 END) as high_watermark_count,
                SUM(CASE WHEN verdict = 'MEDIUM' THEN 1 ELSE 0 END) as medium_watermark_count,
                SUM(CASE WHEN verdict = 'LOW' THEN 1 ELSE 0 END) as low_watermark_count
            FROM analyses
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
        """, (days,)).fetchone()
```

**Usage**:
```python
from specHO.tracking.database import AnalysisTracker

tracker = AnalysisTracker()
analysis_id = tracker.save_analysis(analysis, "test.txt", metrics)

# Query history
recent = tracker.get_history(limit=10)
stats = tracker.get_statistics(days=7)
```

---

### Option 3: Hybrid Approach (Recommended)

**Combine both**: JSON for simple exports, SQLite for querying

**Benefits**:
- SQLite for primary storage and queries
- JSON export for sharing and archiving
- Best of both worlds

**Implementation**:
```python
class AnalysisTracker:
    # ... SQLite methods from Option 2 ...

    def export_to_json(self, output_path: Path, since: datetime = None):
        """Export database to JSONL format."""
        cursor = self.conn.cursor()

        query = "SELECT * FROM analyses"
        params = []

        if since:
            query += " WHERE timestamp >= ?"
            params.append(since.isoformat())

        analyses = cursor.execute(query, params).fetchall()

        with open(output_path, 'w', encoding='utf-8') as f:
            for analysis in analyses:
                # Get associated echo scores
                echo_scores = cursor.execute("""
                    SELECT * FROM echo_scores WHERE analysis_id = ?
                """, (analysis[0],)).fetchall()

                record = format_as_json(analysis, echo_scores)
                f.write(json.dumps(record) + '\n')

    def import_from_json(self, input_path: Path):
        """Import JSONL history into database."""
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                self.save_from_json(record)
```

---

## CLI Integration

### New Flags

```bash
# Save to database
python scripts/cli.py --file test.txt --save

# Query history
python scripts/cli.py --history 10
python scripts/cli.py --history --source test.txt

# Export history
python scripts/cli.py --export data/history_export.jsonl --since 2025-10-01

# Show statistics
python scripts/cli.py --stats --days 7
```

### Updated Argparser

```python
parser.add_argument(
    '--save', '-s',
    action='store_true',
    help='Save analysis results to history database'
)

parser.add_argument(
    '--history', '-H',
    type=int,
    nargs='?',
    const=10,
    help='Show recent analysis history (default: 10)'
)

parser.add_argument(
    '--stats',
    action='store_true',
    help='Show aggregate statistics'
)

parser.add_argument(
    '--export',
    type=str,
    help='Export history to JSONL file'
)

parser.add_argument(
    '--since',
    type=str,
    help='Filter history since date (ISO format: 2025-10-01)'
)
```

---

## Visualization Scripts

### Script: `scripts/visualize_history.py`

```python
"""Visualize analysis history with matplotlib."""

import matplotlib.pyplot as plt
from specHO.tracking.database import AnalysisTracker

def plot_score_distribution(tracker: AnalysisTracker, days: int = 30):
    """Plot distribution of document scores over time."""
    history = tracker.get_history(limit=1000)

    timestamps = [h[1] for h in history]  # timestamp column
    scores = [h[7] for h in history]      # document_score column

    plt.figure(figsize=(12, 6))
    plt.scatter(timestamps, scores, alpha=0.6)
    plt.xlabel('Timestamp')
    plt.ylabel('Document Score')
    plt.title(f'Document Score Distribution (Last {days} days)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('data/score_distribution.png')

def plot_verdict_counts(tracker: AnalysisTracker, days: int = 30):
    """Plot pie chart of verdict distribution."""
    stats = tracker.get_statistics(days=days)

    labels = ['High Watermark', 'Medium Watermark', 'Low Watermark']
    sizes = [stats[3], stats[4], stats[5]]  # high, medium, low counts

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'Watermark Detection Results (Last {days} days)')
    plt.tight_layout()
    plt.savefig('data/verdict_distribution.png')
```

**Usage**:
```bash
python scripts/visualize_history.py --days 30
```

---

## Benefits of Tracking System

### For Development

1. **Regression Testing**: Track if changes improve/degrade scores
2. **Baseline Validation**: Verify baseline corpus effectiveness
3. **Performance Monitoring**: Identify slowdowns over time
4. **Edge Case Discovery**: Find problematic documents

### For Production

1. **Audit Trail**: Full history of all analyses
2. **Trend Analysis**: Detect shifts in AI-generated text patterns
3. **False Positive/Negative Tracking**: Measure accuracy over time
4. **User Feedback Integration**: Correlate results with ground truth

### For Research

1. **Dataset Building**: Collect labeled examples
2. **Algorithm Comparison**: A/B test different configurations
3. **Statistical Analysis**: Aggregate metrics for papers
4. **Reproducibility**: Exact record of all experiments

---

## Implementation Plan

### Phase 1: Basic JSON Logging (1 hour)
- Add `save_analysis_to_history()` to `specHO/utils.py`
- Add `--save` flag to CLI
- Test with sample documents

### Phase 2: SQLite Database (2 hours)
- Create `specHO/tracking/database.py`
- Implement `AnalysisTracker` class
- Migrate to SQLite-based storage
- Add `--history` and `--stats` flags

### Phase 3: Visualization (1.5 hours)
- Create `scripts/visualize_history.py`
- Implement score distribution plots
- Implement verdict pie charts
- Add export/import functionality

### Phase 4: Advanced Queries (1 hour)
- Add filtering by date range
- Add filtering by score threshold
- Add comparison queries
- Add batch export

**Total Estimated Time**: 5.5 hours

---

## Recommendation

**Start with Option 3 (Hybrid)**:
1. Implement SQLite database for primary storage (better queries)
2. Add JSON export for sharing/archiving
3. Add basic visualization scripts

This provides maximum flexibility with minimal overhead. The SQLite database adds only ~50KB overhead and requires no external dependencies (built into Python).

---

## Example Workflow

```bash
# Analyze document and save
$ python scripts/cli.py --file document.txt --save
✓ Analysis saved to database (ID: 42)

# View recent history
$ python scripts/cli.py --history 5
Recent Analyses:
  [42] 2025-10-25 09:30:45 | document.txt | Score: 0.303 | Verdict: LOW
  [41] 2025-10-25 08:15:22 | sample2.txt  | Score: 0.782 | Verdict: HIGH
  [40] 2025-10-24 16:45:10 | test.txt     | Score: 0.521 | Verdict: MEDIUM
  [39] 2025-10-24 14:20:33 | article.txt  | Score: 0.198 | Verdict: LOW
  [38] 2025-10-24 11:05:15 | blog.txt     | Score: 0.655 | Verdict: MEDIUM

# Show statistics
$ python scripts/cli.py --stats --days 7
Statistics (Last 7 days):
  Total Analyses:     156
  Mean Score:         0.487
  Mean Confidence:    62.3%
  High Watermark:     32 (20.5%)
  Medium Watermark:   58 (37.2%)
  Low Watermark:      66 (42.3%)

# Export history
$ python scripts/cli.py --export data/october_results.jsonl --since 2025-10-01
✓ Exported 156 analyses to data/october_results.jsonl

# Visualize trends
$ python scripts/visualize_history.py --days 30
✓ Generated data/score_distribution.png
✓ Generated data/verdict_distribution.png
```

---

**Next Steps**: Would you like me to implement any of these tracking options?
