# Reference Distribution Establishment: Best Practices Guide

**Purpose**: Guide for establishing robust baseline distributions for statistical validation in watermark detection systems

**Audience**: ML Engineers, Data Scientists, System Architects

**Version**: 1.0 (Tier 1 Implementation)

---

## EXECUTIVE SUMMARY

Statistical validation in watermark detection requires comparing a document's score against a **reference distribution** that represents non-watermarked text. This guide covers the theory, best practices, and implementation strategies for establishing robust baseline statistics.

**Key Insight**: The quality of your baseline distribution directly determines the accuracy of your watermark detection. A poor baseline leads to high false positive/negative rates, while a well-constructed baseline enables precise confidence estimates.

---

## THEORETICAL FOUNDATION

### Why Reference Distributions Matter

**The Problem**: Given a document score (e.g., 0.45), how do we determine if it's watermarked?

**Without Baseline**:
- 0.45 could be high, low, or average - no context
- No way to calculate confidence levels
- Arbitrary threshold selection (guesswork)

**With Baseline**:
- If baseline: mean=0.15, std=0.08
- z-score = (0.45 - 0.15) / 0.08 = 3.75
- Confidence = norm.cdf(3.75) = 99.99%
- **Interpretation**: 0.45 is 3.75 standard deviations above human mean → very likely watermarked

### Statistical Framework

```
Document Score (x) → Z-Score (z) → Confidence (p)

z = (x - μ_human) / σ_human
p = Φ(z)  [CDF of standard normal distribution]

Where:
  μ_human = baseline mean (expected score for non-watermarked text)
  σ_human = baseline std dev (variance in non-watermarked text)
  Φ(z) = cumulative distribution function (area under normal curve)
```

**Interpretation**:
- z < -2: Very likely human (p < 2.5%)
- -2 ≤ z ≤ 2: Uncertain region (2.5% ≤ p ≤ 97.5%)
- z > 2: Likely AI/watermarked (p > 97.5%)
- z > 3: Very likely watermarked (p > 99.7%)

---

## CORPUS SELECTION STRATEGIES

### Tier 1: Simple Baseline (Current Implementation)

**Strategy**: Single-source human corpus

**Requirements**:
- **Size**: 50-200 documents minimum
  - Fewer: High variance, unreliable statistics
  - More: Diminishing returns for Tier 1
- **Source**: High-quality human-written text
  - News articles (verified human authors)
  - Academic papers (pre-LLM era)
  - Professional writing samples
  - Book excerpts
- **Homogeneity**: Similar domain/style to target documents
- **Quality**: Clean, well-edited text

**Example Corpus Structure**:
```
data/corpus/human/
├── news_article_001.txt
├── news_article_002.txt
├── academic_paper_001.txt
├── ...
└── book_excerpt_050.txt
```

**Validation**:
- Mean should be low (0.1-0.3 for human text)
- Std should be reasonable (0.05-0.15)
- No extreme outliers (z-score > 3 within corpus)

### Tier 2: Multi-Source Baseline (Future)

**Strategy**: Stratified sampling across domains

**Benefits**:
- More robust to domain shifts
- Better generalization
- Confidence intervals for baseline statistics

**Sources**:
- News: 30% (diverse topics)
- Academic: 20% (various fields)
- Fiction: 15% (multiple authors)
- Technical: 20% (documentation, how-tos)
- Social: 15% (forums, blogs - verified human)

**Stratification**:
```python
baseline_stats = {
    'overall': {'mean': 0.15, 'std': 0.10},
    'by_domain': {
        'news': {'mean': 0.12, 'std': 0.08},
        'academic': {'mean': 0.18, 'std': 0.09},
        'fiction': {'mean': 0.14, 'std': 0.11}
    }
}
```

### Tier 3: Adaptive Baseline (Research)

**Strategy**: Continuous baseline updates with online learning

**Features**:
- Domain-specific baselines
- Temporal baseline updates (language evolution)
- Per-author baselines (writing style profiles)
- Automatic outlier detection and removal

---

## IMPLEMENTATION BEST PRACTICES

### 1. Corpus Quality Control

**Pre-Processing Checks**:
```python
def validate_corpus_file(text: str) -> bool:
    """Check if file is suitable for baseline corpus."""

    # Length check: Too short → unreliable
    if len(text.split()) < 100:
        return False

    # Language check: English only (for English watermark detection)
    if not is_english(text):
        return False

    # Content check: Avoid generated text contamination
    if has_known_ai_patterns(text):
        return False

    # Quality check: Avoid low-quality text
    if has_excessive_typos(text):
        return False

    return True
```

**Outlier Detection**:
```python
def detect_baseline_outliers(scores: List[float]) -> List[int]:
    """Identify outlier documents in baseline corpus."""

    mean = np.mean(scores)
    std = np.std(scores)

    outliers = []
    for i, score in enumerate(scores):
        z = (score - mean) / std
        if abs(z) > 3:  # 3-sigma rule
            outliers.append(i)

    return outliers
```

### 2. Statistical Robustness

**Sample Size Calculation**:
```python
def calculate_minimum_corpus_size(
    desired_margin_of_error: float = 0.02,
    confidence_level: float = 0.95,
    estimated_std: float = 0.10
) -> int:
    """Calculate minimum corpus size for reliable statistics.

    Formula: n = (z * σ / E)²
    Where:
        z = z-score for confidence level (1.96 for 95%)
        σ = estimated population std deviation
        E = desired margin of error
    """
    from scipy.stats import norm

    z = norm.ppf((1 + confidence_level) / 2)
    n = (z * estimated_std / desired_margin_of_error) ** 2

    return int(np.ceil(n))

# Example: For 95% confidence, ±0.02 margin, σ=0.10
# n = (1.96 * 0.10 / 0.02)² = 96.04 → need ~100 documents
```

**Confidence Intervals**:
```python
def calculate_baseline_confidence_intervals(
    scores: np.ndarray,
    confidence: float = 0.95
) -> dict:
    """Calculate confidence intervals for baseline statistics."""
    from scipy.stats import t, sem

    n = len(scores)
    mean = np.mean(scores)
    std_error = sem(scores)

    # t-distribution for small samples
    t_value = t.ppf((1 + confidence) / 2, df=n-1)

    margin = t_value * std_error

    return {
        'mean': mean,
        'mean_ci_lower': mean - margin,
        'mean_ci_upper': mean + margin,
        'std': np.std(scores, ddof=1),
        'n': n,
        'confidence': confidence
    }
```

### 3. Distribution Validation

**Normality Testing**:
```python
def test_baseline_normality(scores: np.ndarray) -> dict:
    """Test if baseline scores follow normal distribution."""
    from scipy.stats import shapiro, normaltest

    # Shapiro-Wilk test (good for small samples)
    shapiro_stat, shapiro_p = shapiro(scores)

    # D'Agostino's K² test (better for larger samples)
    k2_stat, k2_p = normaltest(scores)

    is_normal = shapiro_p > 0.05  # Accept normality if p > 0.05

    return {
        'is_normal': is_normal,
        'shapiro_p_value': shapiro_p,
        'k2_p_value': k2_p,
        'interpretation': 'normal' if is_normal else 'non-normal'
    }
```

**Visual Diagnostics**:
```python
def visualize_baseline_distribution(scores: np.ndarray):
    """Create diagnostic plots for baseline distribution."""
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Histogram with normal overlay
    axes[0, 0].hist(scores, bins=30, density=True, alpha=0.7)
    mean, std = np.mean(scores), np.std(scores)
    x = np.linspace(scores.min(), scores.max(), 100)
    axes[0, 0].plot(x, norm.pdf(x, mean, std), 'r-', label='Normal fit')
    axes[0, 0].set_title('Distribution vs Normal')

    # Q-Q plot (quantile-quantile)
    from scipy.stats import probplot
    probplot(scores, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')

    # Box plot (outlier detection)
    axes[1, 0].boxplot(scores)
    axes[1, 0].set_title('Box Plot (Outliers)')

    # ECDF (empirical cumulative distribution)
    sorted_scores = np.sort(scores)
    ecdf = np.arange(1, len(scores)+1) / len(scores)
    axes[1, 1].plot(sorted_scores, ecdf, label='ECDF')
    axes[1, 1].plot(x, norm.cdf(x, mean, std), 'r-', label='Normal CDF')
    axes[1, 1].set_title('ECDF vs Normal CDF')

    plt.tight_layout()
    return fig
```

---

## COMMON PITFALLS & SOLUTIONS

### Pitfall 1: Insufficient Sample Size

**Problem**:
```python
# Only 10 documents
stats = {'mean': 0.15, 'std': 0.08, 'n': 10}
# High variance in mean estimate → unreliable
```

**Solution**:
```python
# Minimum 50-100 documents for Tier 1
# Calculate required sample size based on desired precision
min_n = calculate_minimum_corpus_size(
    desired_margin_of_error=0.02,  # ±0.02 on mean
    confidence_level=0.95,
    estimated_std=0.10
)
# Result: ~96 documents needed
```

### Pitfall 2: Domain Mismatch

**Problem**:
```python
# Baseline: News articles (formal, structured)
# Target: Social media posts (casual, fragmented)
# → Baseline doesn't represent target distribution
```

**Solution**:
```python
# Match baseline domain to target domain
baseline_sources = {
    'news_detection': 'data/corpus/news/',
    'social_detection': 'data/corpus/social/',
    'academic_detection': 'data/corpus/academic/'
}
```

### Pitfall 3: AI Contamination

**Problem**:
```python
# Corpus includes AI-generated text labeled as "human"
# → Inflated baseline mean → false negatives
```

**Solution**:
```python
def verify_corpus_provenance(corpus_dir: Path) -> bool:
    """Verify corpus is genuinely human-written."""

    # Check creation date (pre-LLM era preferred)
    # Check source metadata
    # Run preliminary watermark detection
    # Manual spot-checking

    for file in corpus_dir.glob("*.txt"):
        metadata = load_metadata(file)
        if metadata['date'] > '2023-01-01':  # After ChatGPT
            if not metadata['verified_human']:
                warn(f"Suspicious file: {file}")

    return True
```

### Pitfall 4: Non-Normal Distribution

**Problem**:
```python
# Scores have heavy tails or multiple modes
# → Normal distribution assumption invalid
# → Confidence calculations incorrect
```

**Solution (Tier 2+)**:
```python
# Option 1: Transform to normality
scores_transformed = np.log(scores + 1)  # Log transform

# Option 2: Use non-parametric methods
from scipy.stats import percentileofscore
confidence = percentileofscore(baseline_scores, document_score) / 100

# Option 3: Fit better distribution
from scipy.stats import gamma, lognorm
best_dist = fit_distribution(scores)  # Returns best-fitting distribution
```

---

## BASELINE MAINTENANCE & UPDATES

### When to Rebuild Baseline

**Triggers**:
1. **Language Drift**: Language evolves over time
   - Rebuild annually or when significant changes detected

2. **Domain Shift**: Target documents change domain
   - Rebuild when switching from news to social media

3. **Watermark Evolution**: New watermarking techniques emerge
   - Rebuild if watermark detection algorithm changes

4. **Poor Performance**: High false positive/negative rates
   - Investigate baseline quality first

### Incremental Updates (Tier 2)

```python
class IncrementalBaseline:
    """Online baseline updates with exponential moving average."""

    def __init__(self, initial_stats: dict, alpha: float = 0.01):
        self.mean = initial_stats['mean']
        self.std = initial_stats['std']
        self.n = initial_stats['n']
        self.alpha = alpha  # Learning rate

    def update(self, new_score: float, verified_human: bool):
        """Update baseline with new verified human document."""
        if not verified_human:
            return  # Only update with verified human text

        # Exponential moving average
        self.mean = (1 - self.alpha) * self.mean + self.alpha * new_score

        # Variance update (Welford's algorithm)
        delta = new_score - self.mean
        self.std = np.sqrt(
            (1 - self.alpha) * self.std**2 +
            self.alpha * delta**2
        )

        self.n += 1
```

---

## VALIDATION & QUALITY ASSURANCE

### Baseline Quality Checklist

- [ ] **Sample Size**: n ≥ 50 (preferably 100+)
- [ ] **Mean Range**: 0.0 ≤ μ ≤ 0.3 (human text should score low)
- [ ] **Std Range**: 0.05 ≤ σ ≤ 0.20 (reasonable variance)
- [ ] **Normality**: p-value > 0.05 (Shapiro-Wilk test)
- [ ] **No Outliers**: All scores within 3σ of mean
- [ ] **Domain Match**: Corpus domain matches target domain
- [ ] **Provenance**: All documents verified human-written
- [ ] **Diversity**: Multiple authors, topics, styles represented
- [ ] **Quality**: Clean, well-edited text (no OCR errors, etc.)
- [ ] **Recency**: Created within last 2 years (for language drift)

### Cross-Validation

```python
def cross_validate_baseline(corpus_scores: np.ndarray, k: int = 5):
    """K-fold cross-validation for baseline stability."""
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_stats = []
    for train_idx, test_idx in kf.split(corpus_scores):
        train_scores = corpus_scores[train_idx]

        fold_mean = np.mean(train_scores)
        fold_std = np.std(train_scores, ddof=1)

        fold_stats.append({'mean': fold_mean, 'std': fold_std})

    # Check consistency across folds
    means = [s['mean'] for s in fold_stats]
    stds = [s['std'] for s in fold_stats]

    mean_variance = np.var(means)
    std_variance = np.var(stds)

    print(f"Mean stability: {mean_variance:.4f} (lower is better)")
    print(f"Std stability: {std_variance:.4f} (lower is better)")

    return fold_stats
```

---

## TIER PROGRESSION

### Tier 1: Simple Baseline (Current)
- Single corpus source
- Basic mean/std calculation
- Pickle persistence
- Manual corpus curation

### Tier 2: Robust Baseline
- Multi-source stratified corpus
- Confidence intervals
- JSON persistence with versioning
- Automated outlier detection
- Cross-validation

### Tier 3: Adaptive Baseline
- Domain-specific baselines
- Online learning / incremental updates
- Distribution fitting (beyond normal)
- Temporal baseline tracking
- Per-author profiling

---

## REFERENCES & RESOURCES

### Statistical Theory
- Casella & Berger (2002): *Statistical Inference* - Chapter 5 (Distribution Theory)
- Wasserman (2004): *All of Statistics* - Chapter 6 (Convergence)

### Machine Learning Context
- Murphy (2022): *Probabilistic Machine Learning* - Chapter 3.2 (Gaussian distribution)
- Bishop (2006): *Pattern Recognition and Machine Learning* - Chapter 2.3 (Bayesian inference)

### Watermarking Specific
- Kirchenbauer et al. (2023): "A Watermark for Large Language Models"
- Aaronson & Kirchner (2023): "Paraphrasing evades detectors of AI-generated text"

---

## CONCLUSION

**Key Takeaways**:

1. **Quality > Quantity**: 100 high-quality documents > 1000 low-quality documents
2. **Domain Matters**: Match baseline to target domain for accurate validation
3. **Validate Assumptions**: Test normality, check for outliers, verify provenance
4. **Iterate**: Start simple (Tier 1), measure performance, improve (Tier 2/3)
5. **Monitor**: Baseline quality directly impacts detection accuracy

**Success Metrics**:
- Low false positive rate (<5%)
- Low false negative rate (<5%)
- Stable performance across domains
- Confident predictions (high-confidence regions well-separated)

**Next Steps**:
1. Implement ZScoreCalculator (Task 6.2)
2. Implement ConfidenceConverter (Task 6.3)
3. Implement StatisticalValidator orchestrator (Task 6.4)
4. Test on real corpus
5. Measure detection performance
6. Iterate on baseline quality if needed

---

**Document Version**: 1.0
**Last Updated**: 2025-10-19
**Status**: Tier 1 Complete, Tier 2/3 Planning
