# SpecHO Text Analysis Tools
**Spectral Harmonics of Text - AI Watermark Detection**

## Overview

This toolkit implements "The Echo Rule" methodology for detecting AI-generated text through analysis of:
- **Phonetic patterns** (syllable stress, rhythm)
- **Structural parallelism** (POS tagging, clause structure)
- **Semantic echoing** (embedding similarity, conceptual mirroring)

## Files Included

### Analysis Scripts

1. **specho_analyzer.py**
   - Basic SpecHO analysis with overall statistics
   - Automated echo pattern detection
   - JSON output of results
   - Usage: `python specho_analyzer.py`

2. **specho_detailed.py**
   - Detailed clause-level breakdown
   - Focus on specific suspicious sentences
   - Parallel structure analysis
   - Usage: `python specho_detailed.py`

3. **spececho_final.py**
   - Comprehensive analysis combining all methods
   - Comparative clustering detection
   - Smooth transition analysis
   - Final verdict with confidence levels
   - Usage: `python spececho_final.py`

### Data Files

4. **article.txt**
   - The Conversation article being analyzed
   - "Learning with AI falls short compared to old-fashioned web search"
   - By Shiri Melumad (Wharton)

## Installation

### Requirements

```bash
pip install nltk numpy --break-system-packages
```

### NLTK Data

The scripts will automatically download required NLTK data, but you can manually download with:

```python
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('cmudict')
```

## Usage

### Quick Analysis

Run the comprehensive analysis:

```bash
python spececho_final.py
```

This will output:
- Comparative clustering detection
- Parallel verb structure analysis
- Semantic echo patterns
- Smooth transition analysis
- Overall AI probability verdict

### Detailed Breakdown

For sentence-by-sentence analysis:

```bash
python specho_detailed.py
```

### Full Statistics

For complete statistical analysis with JSON output:

```bash
python specho_analyzer.py
```

## Understanding the Output

### Key Metrics

**Smooth Transitions**
- Rate per sentence
- Human typical: <0.15
- AI typical: >0.25

**Parallel Structures**
- Rate per sentence
- Human typical: <0.2
- AI typical: >0.3

**Comparative Clustering**
- Number of comparatives in single sentence
- Human typical: <3
- AI typical: >3

**Em-dash Frequency**
- Rate per sentence
- Human typical: <0.3
- AI typical: >0.5

### AI Probability Levels

- 🔴 **HIGH** (>0.7): Strong indicators present
- 🟡 **MODERATE** (0.4-0.7): Multiple indicators present
- 🟢 **LOW** (<0.4): Few or weak indicators

## The Echo Rule Methodology

### What is "Harmonic Oscillation"?

LLMs create detectable patterns where concepts "echo" across clause pairs:

```
"learned less" 
    ↓ (echo: less)
"less effort"
    ↓ (echo: comparative)
"shorter" (= less)
    ↓ (echo: less)
"less factual"
    ↓ (echo: more/comparative)
"more generic"
```

This creates a semantic rhythm that human writers rarely sustain.

### Detection Methods

1. **Phonetic Analysis**
   - Syllable counting using CMU Pronouncing Dictionary
   - Stress pattern comparison across clauses
   - Rhythmic cadence detection

2. **Structural Analysis**
   - POS (Part-of-Speech) tagging with NLTK
   - Parallel construction frequency
   - Repetitive verb pattern detection

3. **Semantic Analysis**
   - Word overlap calculation (Jaccard similarity)
   - Conceptual echoing detection
   - Comparative term clustering

## Analyzing Your Own Text

To analyze a different text file:

1. Replace the content in `article.txt` with your text
2. Run any of the analysis scripts
3. Review the output for AI indicators

Or modify the scripts to read from a different file:

```python
with open('your_file.txt', 'r') as f:
    text = f.read()
```

## Results Interpretation

### For The Conversation Article

**Verdict**: MODERATE-HIGH probability of AI assistance

**Key Findings**:
- Comparative clustering: 5 in one sentence (EXTREME)
- Smooth transitions: 0.30 per sentence (HIGH)
- Parallel structures: 0.37 per sentence (MODERATE)
- Em-dash frequency: 0.23 per sentence (LOW)

**Smoking Gun**: The sentence with 5 comparative terms creating harmonic oscillation is nearly impossible to explain as pure human writing.

## Limitations

- Best for formal/academic writing analysis
- May flag heavily-edited human text
- Requires substantial text (>500 words) for reliable results
- Not a definitive proof, but probabilistic indicator

## Technical Details

### Text Processing Pipeline

1. Sentence tokenization
2. Clause boundary detection (punctuation-based)
3. POS tagging for structural analysis
4. Syllable counting for phonetic patterns
5. Semantic similarity calculation
6. Composite score generation

### Scoring System

Each indicator receives a 0-1 score:
- Phonetic: 1.0 - (syllable_difference)
- Structural: POS_pattern_match_ratio
- Semantic: Jaccard_similarity (optimal 0.3-0.5)

Composite score = mean of all indicators

## Citation

If you use this methodology, please cite:

```
SpecHO (Spectral Harmonics of Text) Analysis
The Echo Rule Methodology for AI Watermark Detection
Developed: November 2025
```

## License

This toolkit is provided as-is for educational and research purposes.

## Contact

For questions about the methodology or results, refer to the analysis documentation included in the output files.

---

**Remember**: This is a probabilistic tool. High scores suggest AI involvement but don't prove it. Always consider context and use multiple lines of evidence.
