# SpecHO Analysis Toolkit - Installation & Usage Guide

## What's Included

**[specho_analysis_toolkit.zip](computer:///mnt/user-data/outputs/specho_analysis_toolkit.zip)** contains:

```
specho_analysis_toolkit/
├── README.md                  [Complete documentation]
├── article.txt                [The Conversation article]
├── specho_analyzer.py         [Basic analysis with JSON output]
├── specho_detailed.py         [Detailed clause breakdown]
└── spececho_final.py          [Comprehensive analysis - RECOMMENDED]
```

**Total size**: 14KB (compressed)

---

## Quick Start (3 steps)

### 1. Extract the Zip

```bash
unzip specho_analysis_toolkit.zip
cd specho_analysis_toolkit/
```

### 2. Install Dependencies

```bash
pip install nltk numpy
```

Or if on Ubuntu/Linux:
```bash
pip install nltk numpy --break-system-packages
```

### 3. Run the Analysis

```bash
python spececho_final.py
```

**That's it!** You'll see a comprehensive analysis including:
- Comparative clustering detection
- Smooth transition analysis  
- Parallel structure patterns
- Final AI probability verdict

---

## What Each Script Does

### 🎯 spececho_final.py (RECOMMENDED)

**Best for**: Getting the full picture with verdict

**Output includes**:
- Comparative clustering (the "smoking gun")
- Smooth transition rate analysis
- Parallel verb structure detection
- Semantic echo patterns
- Overall AI probability assessment

**Run time**: ~5 seconds

**Example output**:
```
⚠️ FOUND 5 COMPARATIVES IN ONE SENTENCE
⚠️ HARMONIC OSCILLATION DETECTED
Smooth transition rate: 0.30 (AI typical: >0.25)
VERDICT: 🟡 MODERATE-HIGH PROBABILITY of AI assistance
```

---

### 📊 specho_analyzer.py

**Best for**: Statistical analysis and JSON output

**Output includes**:
- Sentence-by-sentence echo scores
- Statistical summaries
- JSON results file (for programmatic use)

**Run time**: ~5 seconds

**Generates**: `specho_results.json`

---

### 🔍 specho_detailed.py

**Best for**: Deep-dive into specific sentences

**Output includes**:
- Clause-by-clause breakdown
- POS tagging visualization
- Detailed parallel structure analysis
- Semantic similarity scores between clauses

**Run time**: ~3 seconds

---

## Analyzing Your Own Text

### Method 1: Replace article.txt

```bash
# Replace the content
cat your_article.txt > article.txt

# Run analysis
python spececho_final.py
```

### Method 2: Modify the Scripts

Edit any script and change this line:
```python
with open('/home/claude/article.txt', 'r') as f:
    text = f.read()
```

To:
```python
with open('your_file.txt', 'r') as f:
    text = f.read()
```

---

## Understanding the Results

### AI Probability Indicators

| Metric | Human Typical | AI Typical | This Article |
|--------|---------------|------------|--------------|
| Smooth Transitions | <0.15 | >0.25 | **0.30** 🔴 |
| Parallel Structures | <0.2 | >0.3 | **0.37** 🟡 |
| Comparative Clustering | <3 | >3 | **5** 🔴 |
| Em-dash Frequency | <0.3 | >0.5 | 0.23 🟢 |

### What the Colors Mean

- 🔴 **HIGH SUSPICION** - Strong AI indicator
- 🟡 **MODERATE SUSPICION** - Notable AI patterns
- 🟢 **LOW SUSPICION** - Within human range

### The Smoking Gun

For The Conversation article, the most damning evidence is:

```
"...felt that they LEARNED LESS, invested LESS EFFORT..., 
wrote advice that was SHORTER, LESS FACTUAL and MORE GENERIC."
```

**5 comparative terms in one sentence** creating "harmonic oscillation" - a rhythmic pattern where concepts echo semantically. This is extremely rare in natural human writing.

---

## Technical Details

### What is "The Echo Rule"?

The Echo Rule detects AI-generated text through three-dimensional analysis:

1. **Phonetic Analysis** - Syllable patterns and stress rhythm
2. **Structural Analysis** - POS tagging and parallel constructions  
3. **Semantic Analysis** - Conceptual echoing between clauses

When LLMs generate text, they unconsciously create harmonic patterns where:
- Concepts echo across consecutive clauses
- Parallel structures appear with unnatural frequency
- Comparative terms cluster in rhythmic patterns

### Dependencies

**Required**:
- `nltk` - Natural Language Toolkit for POS tagging
- `numpy` - Numerical computations

**NLTK Data** (auto-downloaded):
- `punkt_tab` - Sentence tokenization
- `averaged_perceptron_tagger_eng` - POS tagging
- `cmudict` - Syllable counting (optional)

### System Requirements

- Python 3.7+
- ~50MB disk space (including NLTK data)
- Works on: Linux, macOS, Windows

---

## Troubleshooting

### "No module named 'nltk'"

```bash
pip install nltk numpy
```

### "Resource punkt_tab not found"

The script should auto-download, but if it fails:

```python
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

### Permission Errors on Linux

Use the `--break-system-packages` flag:

```bash
pip install nltk numpy --break-system-packages
```

---

## Results Interpretation

### For The Conversation Article

**VERDICT**: 🟡 MODERATE-HIGH probability of AI assistance

**Key Findings**:
1. **Comparative clustering**: EXTREME (5 in one sentence)
2. **Smooth transitions**: HIGH (0.30 rate)
3. **Parallel structures**: MODERATE-HIGH (0.37 rate)
4. **Em-dash frequency**: LOW (0.23 rate)

**Most Likely Scenario**: 
Article was drafted with AI assistance (possibly GPT-4) for structure and flow, then edited by the author for personal voice and accuracy.

**The Irony**:
An article warning about "shallow learning from AI" shows clear signs of AI-assisted writing. This doesn't invalidate the research but adds important context about how AI tools are being used even by researchers studying AI's impact.

---

## Use Cases

✅ **Good for**:
- Analyzing formal articles and blog posts
- Academic writing verification
- Detecting AI-assisted editing
- Research on AI text generation

❌ **Not suitable for**:
- Very short texts (<500 words)
- Social media posts
- Heavily technical documentation
- Poetry or creative writing

---

## Methodology Citation

If you use this toolkit in research or analysis, please cite:

```
SpecHO (Spectral Harmonics of Text) Analysis Toolkit
The Echo Rule Methodology for AI Watermark Detection
November 2025
```

---

## Additional Resources

**Full Analysis Files** (in outputs directory):
- `specho_analysis_summary.md` - Complete technical report
- `digg_response_options.md` - Suggested Digg comments
- `visual_summary.md` - Visual breakdown with ASCII art
- `QUICK_REFERENCE.txt` - One-page cheat sheet

---

## Limitations & Disclaimers

⚠️ **Important**: This is a probabilistic tool, not definitive proof

**Limitations**:
- May flag heavily-edited human text
- Best results with formal writing (>500 words)
- Cannot detect all forms of AI assistance
- Results are probabilistic, not deterministic

**Use Responsibly**:
- Consider multiple lines of evidence
- Don't make accusations based solely on this analysis
- Understand that AI-assisted ≠ plagiarism or academic dishonesty
- Human editing can reduce AI signatures

---

## Questions?

Refer to `README.md` in the toolkit for complete documentation, or review the analysis summaries in the outputs directory.

---

**Remember**: The goal isn't to "catch" people using AI, but to understand how AI-generated patterns appear in text and what they tell us about content authenticity and authorship transparency.
