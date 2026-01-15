# SpecHO - Echo Rule Watermark Detection

> **ARCHIVED REPOSITORY**
>
> This project has been retired and archived. It represents completed research into AI text detection and source attribution. The codebase is functional but no longer under active development.
>
> For historical context, see `docs/archive/`.

---

## Project Evolution

### What It Started As

SpecHO began as an LLM-generated text detection tool, tackling the classic "is this AI-written?" problem. The approach was to detect a specific watermarking pattern called the "Echo Rule," where AI-generated text exhibits subtle phonetic, structural, and semantic echoes between related clauses.

### What It Became

Through development and experimentation, the project evolved into something more ambitious: **LLM fingerprinting and source identification**. Not just "is this AI?" but "WHICH AI?" The system performs model-level attribution through statistical fingerprinting of writing patterns.

The five-component pipeline analyzes:
- **Phonetic echoes** - Sound correspondence across clause boundaries
- **Structural echoes** - Grammatical pattern alignment (POS sequences)
- **Semantic echoes** - Meaning relationships via word embeddings

By measuring these patterns across many clause pairs, the system builds a statistical profile that can distinguish between different text sources.

---

## The Counterintuitive Finding

During development, a surprising pattern emerged:

> **When treating human-written text as just another "model" in the classification set, humans emerged as the most predictable and identifiable source.**

LLMs exhibit more variance in their outputs than humans do. This inverts the common assumption that humans are the baseline of unpredictability. Human writing, it turns out, is highly fingerprintable.

### Why This Matters

- **Content Provenance**: Attribution becomes a pattern-matching problem, not a detection problem
- **Authenticity Verification**: Human "signatures" may be more reliable than expected
- **Philosophical Implications**: What does "human writing" mean when it's more predictable than machine output?

---

## Current State

### Implementation Status

- **Tier 1 MVP**: Complete (32/32 tasks)
- **Test Coverage**: 830 tests passing (100% pass rate)
- **Performance**: ~75 words/second throughput

### Pipeline Components

1. **Linguistic Preprocessor** - Tokenization, POS tagging, phonetic transcription
2. **Clause Identifier** - Boundary detection, pair rules, zone extraction
3. **Echo Engine** - Phonetic, structural, semantic analyzers
4. **Scoring Module** - Weighted combination, aggregation
5. **Statistical Validator** - Z-score calculation, confidence conversion

### Accuracy Scaling

Current accuracy is based on a ~500 sample training set. The key insight: **this is now a data collection problem, not an algorithm problem**. Accuracy improves predictably with more fingerprint samples. The detection algorithms work; they just need more training data to refine thresholds.

---

## Data Philosophy

The project emphasizes several data principles that emerged during development:

### Token-Efficient Structures

Data formats should minimize wasted context. Avoid verbose boilerplate; structure data for machine consumption, not human readability.

### ML-Ready Pipelines

Shape data for training from the start. The pipeline outputs (clause pairs, echo scores, aggregate metrics) are designed to feed directly into classification models.

### Provenance on Data

Know where your fingerprints came from. Track source documents, model versions, generation parameters. Without provenance, fingerprint data loses its value.

---

## Quick Start

```bash
# Setup
git clone https://github.com/your-repo/specHO.git
cd specHO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Analyze text
python scripts/cli.py --file data/samples/sample.txt --verbose

# Run tests
pytest tests/ -v
```

### Python API

```python
from specHO.detector import SpecHODetector

detector = SpecHODetector()
result = detector.analyze("Text to analyze...")

print(f"Score: {result.final_score:.3f}")
print(f"Confidence: {result.confidence:.1%}")
```

---

## Repository Structure

```
specHO/
├── specHO/              # Core implementation (5 components)
│   ├── preprocessor/    # Tokenization, POS, dependencies, phonetics
│   ├── clause_identifier/  # Boundary detection, pairing rules
│   ├── echo_engine/     # Phonetic, structural, semantic analysis
│   ├── scoring/         # Weighted scoring, aggregation
│   └── validator/       # Z-score, confidence calculation
├── tests/               # 830 tests
├── scripts/             # CLI, demos, utilities
├── data/                # Samples, baseline corpus
├── docs/                # Active documentation
│   └── archive/         # Historical docs, experiments
└── architecture.md      # Echo Rule theory
```

---

## Documentation

| Document | Purpose |
|----------|---------|
| [architecture.md](architecture.md) | Echo Rule theory and detection methodology |
| [docs/TASKS.md](docs/TASKS.md) | All 32 task specifications |
| [docs/SPECS.md](docs/SPECS.md) | Tier specifications |
| [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Implementation learnings |
| [docs/archive/](docs/archive/) | Historical development docs |

---

## Requirements

- Python 3.11+
- spaCy with `en_core_web_sm`
- See `requirements.txt` for full dependencies

---

## License

MIT

---

## Archive Note

This repository is archived as of January 2026. The research achieved its primary objectives:

1. Demonstrated feasibility of multi-dimensional echo detection
2. Built a working 5-component detection pipeline
3. Discovered a new working hypothesis. (Human writing being the mlst predictable) 
4. Established that scaling requires data collection, not algorithm refinement

For anyone continuing this research: focus on building larger, well-documented fingerprint corpora. The detection methodology is sound.

---

*Contributors: Human + AI collaboration*
*Originally developed: 2025*
*Archived: 2026-01-14*
