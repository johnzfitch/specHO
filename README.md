# <img src="icons/sound.png" width="32" height="32"> SpecHO - Echo Rule Watermark Detector

Watermark detection system for AI-generated text using phonetic, structural, and semantic echo analysis.

## <img src="icons/rocket.png" width="24" height="24"> Quick Start

```bash
# Clone and setup
git clone <your-repo-url>
cd SpecHO
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Analyze text
python scripts/cli.py --file sample.txt
```

## <img src="icons/bar-chart.png" width="24" height="24"> Project Status

**Current:** Tier 1 (MVP) - Clause Identifier in progress
**Completed:** Preprocessor (98.6% tests passing)
**See:** [docs/STATUS.md](docs/STATUS.md) for current state

## <img src="icons/blueprint.png" width="24" height="24"> Architecture

Five-component sequential pipeline:
1. **Preprocessor** - Tokenization, POS tagging, phonetic transcription
2. **Clause Identifier** - Find related clause pairs (in progress)
3. **Echo Engine** - Phonetic, structural, semantic similarity
4. **Scoring** - Weighted combination of echo scores
5. **Validator** - Z-score comparison against baseline

## <img src="icons/document.png" width="24" height="24"> Documentation

| Document | Purpose |
|----------|---------|
| <img src="icons/compass.png" width="16" height="16"> [CLAUDE.md](CLAUDE.md) | AI instructions + documentation protocol |
| <img src="icons/bar-chart.png" width="16" height="16"> [docs/STATUS.md](docs/STATUS.md) | Current state + AI context |
| <img src="icons/task-list.png" width="16" height="16"> [docs/TASKS.md](docs/TASKS.md) | All 32 task specifications |
| <img src="icons/document.png" width="16" height="16"> [docs/SPECS.md](docs/SPECS.md) | Tier specifications |
| <img src="icons/wrench.png" width="16" height="16"> [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) | Learnings + gotchas |
| <img src="icons/rocket.png" width="16" height="16"> [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Operations guide |
| <img src="icons/blueprint.png" width="16" height="16"> [architecture.md](architecture.md) | Echo Rule theory |

**Archive:** Historical session docs in `docs/archive/`

## <img src="icons/wrench.png" width="24" height="24"> Usage

```python
from SpecHO import SpecHODetector, load_config

detector = SpecHODetector(load_config("simple"))
result = detector.analyze("Text to analyze...")

print(f"Confidence: {result.confidence:.2%}")
print(f"Z-Score: {result.z_score:.2f}")
```

## <img src="icons/task-list.png" width="24" height="24"> Requirements

- Python 3.11+
- spaCy with en_core_web_sm model
- See `requirements.txt` for full list

## License

MIT

## Contributing

This project follows a strict three-tier development process. See `CLAUDE.md` for contribution guidelines.
