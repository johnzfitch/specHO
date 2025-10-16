# SpecHO - Echo Rule Watermark Detector

Watermark detection system for AI-generated text using phonetic, structural, and semantic echo analysis.

## Quick Start

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

## Project Status

**Current:** Tier 1 (MVP) - In Development  
**Tasks Complete:** 0/32  
**Test Coverage:** 0%

## Architecture

Five-component sequential pipeline:
1. **Linguistic Preprocessor** - Tokenization, POS tagging, phonetic transcription
2. **Clause Pair Identifier** - Find related clause pairs using rule-based system
3. **Echo Analysis Engine** - Phonetic, structural, and semantic similarity scoring
4. **Scoring & Aggregation** - Weighted combination of echo scores
5. **Statistical Validator** - Z-score comparison against human baseline

## Development

Using three-tier approach:
- **Tier 1 (Weeks 1-12):** MVP with simple algorithms
- **Tier 2 (Weeks 13-17):** Production hardening
- **Tier 3 (Week 18+):** Research features

**Documentation:**
- `CLAUDE.md` - Main development guide
- `docs/QUICKSTART.md` - Setup and first task
- `docs/TASKS.md` - All 32 task specifications
- `architecture.md` - Original watermark design

## Usage

```python
from SpecHO import SpecHODetector, load_config

detector = SpecHODetector(load_config("simple"))
result = detector.analyze("Text to analyze...")

print(f"Confidence: {result.confidence:.2%}")
print(f"Z-Score: {result.z_score:.2f}")
```

## Requirements

- Python 3.11+
- spaCy with en_core_web_sm model
- See `requirements.txt` for full list

## License

[Your License Here]

## Contributing

This project follows a strict three-tier development process. See `CLAUDE.md` for contribution guidelines.
