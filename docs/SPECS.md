# SPECS

**Format:** Machine-readable tier specifications  
**Purpose:** Detailed implementation specs for each component  
**Usage:** Reference when implementing component features

---

## TIER_SYSTEM_OVERVIEW

```yaml
tier_1_mvp:
  timeline: Weeks 1-12
  philosophy: Get it working
  algorithms: Simple only
  testing: Basic coverage
  goal: Complete functional detector
  
tier_2_production:
  timeline: Weeks 13-17
  philosophy: Make it reliable
  algorithms: Proven enhancements
  testing: Comprehensive coverage
  goal: Production-ready system
  
tier_3_research:
  timeline: Week 18+
  philosophy: Optimize performance
  algorithms: Advanced/experimental
  testing: Comparative benchmarks
  goal: Research-grade optimization
```

---

## COMPONENT_2_CLAUSE_IDENTIFIER

### BOUNDARY_DETECTOR

```yaml
tier_1:
  api: identify_clauses(doc: spacy.tokens.Doc) -> List[Clause]
  
  heuristics:
    - Split on finite verb heads (ROOT, conj)
    - Subordinate clauses (advcl, ccomp)
    - Simple punctuation rules (period, semicolon, em dash)
  
  output:
    - List[Clause] with start_idx, end_idx, clause_type, tokens
  
  edge_cases: None (defer to Tier 2)

tier_2:
  additions:
    - Relative clause detection (acl/relcl with mark)
    - Edge case handling (quotes/parentheses trimming)
    - Avoid zero-length spans
    - Fragment merging (clauses < 3 tokens into adjacent)
  
  config:
    min_clause_length: 3
    max_clause_length: 50
    punctuation: [";", "—", ":"]
    dependency_labels: ["ROOT", "conj", "advcl", "ccomp"]
    strict_mode: false

tier_3:
  additions:
    - Multi-sentence cross-boundary pairing
    - Sophisticated trimming (sentence-initial adverbials, list detection)
    - Tie-breakers for overlapping spans
```

### PAIR_RULES_ENGINE

```yaml
tier_1:
  api:
    - apply_rule_a(clauses: List[Clause]) -> List[ClausePair]
    - apply_rule_b(clauses: List[Clause]) -> List[ClausePair]
    - apply_rule_c(clauses: List[Clause]) -> List[ClausePair]
  
  rule_a_punctuation:
    triggers: [";", "—"]
    logic: Pair clauses split by these punctuation marks
  
  rule_b_conjunction:
    triggers: ["but", "and", "or"]
    logic: Pair clauses on either side of coordinating conjunction
  
  rule_c_transition:
    triggers: ["However,", "Therefore,", "Thus,"]
    logic: Pair clause with transitioning neighbor
  
  deduplication: By clause indices (simple)

tier_2:
  additions:
    - ClausePair fields: pair_type, rule_id, confidence, rationale
    - Rule A enhancement: weight semicolon > em dash > colon
    - Rule B enhancement: correlative pairs (either/or, not only/but also)
    - Rule C enhancement: expanded transition list + directionality
    - Confidence scoring based on signal strength
    - Negative guards: skip if either clause < min_tokens
  
  config:
    conjunctions: ["but", "yet", "however", "and", "or"]
    transitions: ["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,"]
    min_pair_confidence: 0.3
    use_confidence_weighting: false  # Enable in Tier 2

tier_3:
  additions:
    - Lightweight pre-scoring (zone content-word overlap)
    - False positive detection for parentheticals
    - List detection after colons (skip if no verb)
```

### ZONE_EXTRACTOR

```yaml
tier_1:
  api: extract_zones(clause_pair: ClausePair) -> Tuple[List[Token], List[Token]]
  
  logic:
    - Extract last 3 content words from clause_a
    - Extract first 3 content words from clause_b
    - Basic content word detection (nouns, verbs, adjectives)
  
  implementation: Simple iteration, no lemmatization

tier_2:
  additions:
    - Add lemmatization
    - Exclude discourse markers (however, therefore)
    - Handle short clauses (use all content words if < 3)
    - Leading/trailing quote/parenthesis trimming
    - Deterministic sorting by token index
  
  config:
    window_size: 3
    min_zone_length: 1
    exclude_discourse_markers: true
    discourse_markers: ["however", "therefore", "thus", "moreover"]

tier_3:
  additions:
    - Configurable window sizes (1-5 words)
    - Priority weighting (terminal vs initial)
    - Low-confidence marking for edge cases
```

---

## COMPONENT_3_ECHO_ENGINE

### PHONETIC_ANALYZER

```yaml
tier_1:
  api: analyze(zone_a: List[Token], zone_b: List[Token]) -> float
  
  algorithm: Levenshtein distance on ARPAbet strings
  
  implementation:
    - Use Token.phonetic field from preprocessor
    - Pairwise comparison between zones
    - Best match selection
    - Normalize to [0,1]: similarity = 1 - (distance / max_length)
  
  edge_cases:
    - Return 0 if either zone empty
    - Single-word zones: direct comparison
  
  library: python-Levenshtein

tier_2:
  additions:
    - Rime-based comparison (from last stressed vowel onward)
    - Multiple features (rhyme match, onset overlap, syllable similarity)
    - Aggregation: average top-k matches with length penalty
    - Handle OOV with fallback G2P
    - Strip punctuation/interjections
  
  config:
    algorithm: "rime"  # Changed from "levenshtein"
    top_k_matches: 2
    length_penalty: 0.1
    use_stress_patterns: false  # Tier 3

tier_3:
  additions:
    - Hungarian algorithm for optimal alignment
    - Phoneme-level feature extraction
    - Stress pattern matching
    - LRU cache for phonetic computations
  
  config:
    algorithm: "hungarian"
    use_stress_patterns: true
```

### STRUCTURAL_ANALYZER

```yaml
tier_1:
  api: analyze(zone_a: List[Token], zone_b: List[Token]) -> float
  
  features:
    - POS pattern comparison (exact match)
    - Syllable count similarity
  
  scoring: pattern_sim * 0.5 + syllable_sim * 0.5
  
  implementation: Direct comparison, no sophisticated algorithms

tier_2:
  additions:
    - Coarse POS tag mapping
    - LCS similarity for patterns: 2*LCS/(lenA+lenB)
    - Word property checks (content-word ratios, NER presence)
    - Weighted aggregation of features
    - Short zone fallback (length-sim only)
  
  config:
    features:
      pos_pattern: {enabled: true, weight: 0.4}
      syllable_count: {enabled: true, weight: 0.3}
      word_properties: {enabled: true, weight: 0.3}

tier_3:
  additions:
    - Abstract noun detection
    - Syntactic role similarity
    - Configurable feature weights per analysis
```

### SEMANTIC_ANALYZER

```yaml
tier_1:
  api: analyze(zone_a: List[Token], zone_b: List[Token]) -> float
  
  algorithm:
    - Mean-pooled word embeddings (Word2Vec or GloVe)
    - Cosine similarity between zone vectors
    - Map to [0,1]: (1 + cos) / 2
  
  fallback: Return 0.5 if embeddings unavailable (neutral)
  
  library: gensim

tier_2:
  additions:
    - Sentence-level embeddings (all-MiniLM-L6-v2)
    - OOV backoff: try static vectors, then return NaN
    - Antonym detection via polarity check
    - Batch processing for multiple pairs
    - Embedding cache (LRU + disk)
  
  config:
    model: "all-MiniLM-L6-v2"
    use_antonym_detection: false  # Enable in Tier 2
    batch_size: 32
    device: "cpu"
    cache_embeddings: true

tier_3:
  additions:
    - Transformer-based embeddings (BERT, RoBERTa)
    - WordNet antonym relationships
    - Negation handling
    - GPU acceleration
  
  config:
    model: "sentence-transformers/all-mpnet-base-v2"
    use_antonym_detection: true
    device: "cuda"
```

---

## COMPONENT_4_SCORING

### WEIGHTED_SCORER

```yaml
tier_1:
  api: calculate_pair_score(echo_score: EchoScore, weights: Dict[str, float]) -> float
  
  formula: w_p * phonetic + w_s * structural + w_sem * semantic
  
  weights: {phonetic: 0.33, structural: 0.33, semantic: 0.33}
  
  nan_handling: Treat as 0
  
  output: Clip to [0,1]

tier_2:
  additions:
    - Configurable weights with auto-normalization
    - Missing data strategies: 'zero' or 'renorm'
    - Pair confidence multipliers from ClausePair metadata
    - Per-analyzer on/off flags
  
  config:
    weights: {phonetic: 0.4, structural: 0.3, semantic: 0.3}
    missing_data_strategy: "renorm"  # Changed from "zero"
    use_pair_confidence: true

tier_3:
  additions:
    - Calibration curves (piecewise linear, sigmoid)
    - Zone length penalties
    - Rule-type boosts
  
  config:
    calibration:
      type: "sigmoid"
      params: {k: 10, x0: 0.5}
```

### DOCUMENT_AGGREGATOR

```yaml
tier_1:
  api: aggregate_scores(pair_scores: List[float]) -> float
  
  algorithm: Simple mean
  
  edge_cases:
    - Return 0.0 if no pairs
    - Emit warning for empty input

tier_2:
  additions:
    - Multiple strategies: mean, median, trimmed_mean
    - Outlier detection via IQR or z-score
    - Return statistics: n_pairs, mean, median, std, p25/p75
    - Optional pair weighting by confidence
  
  config:
    strategy: "trimmed_mean"  # Changed from "mean"
    trim_percent: 0.1
    outlier_removal: true
    return_statistics: true

tier_3:
  additions:
    - Winsorized mean
    - Weighted aggregation
    - Adaptive outlier removal
```

---

## COMPONENT_5_VALIDATOR

### STATISTICAL_VALIDATOR

```yaml
tier_1:
  api: validate(document_score: float) -> Tuple[float, float]
  
  baseline:
    - Pre-computed human_mean and human_std
    - Loaded from pickled file
  
  z_score: (document_score - human_mean) / human_std
  
  confidence: scipy.stats.norm.cdf(z_score)

tier_2:
  additions:
    - Baseline builder with progress tracking
    - Multiple corpus sources
    - Caching with version control
    - Robust error handling
  
  features:
    - Validate baseline freshness
    - Support multiple baseline profiles
    - Incremental baseline updates

tier_3:
  additions:
    - Online baseline updates
    - Corpus stratification (by genre, domain)
    - Distribution fitting (not just normal)
    - Adaptive baseline selection
```

---

## CONFIG_PROFILES

### SIMPLE_PROFILE

```yaml
name: simple
tier: 1
purpose: MVP implementation

scoring:
  weights:
    phonetic: 0.33
    structural: 0.33
    semantic: 0.33
  missing_data_strategy: "zero"

aggregation:
  strategy: "mean"

phonetic_analysis:
  algorithm: "levenshtein"

semantic_analysis:
  model: "static"  # Word2Vec/GloVe
  device: "cpu"

clause_detection:
  min_length: 3
  max_length: 50

pair_rules:
  conjunctions: ["but", "and", "or"]
  transitions: ["However,", "Therefore,", "Thus,"]

zone_extraction:
  window_size: 3
  min_zone_length: 1
```

### ROBUST_PROFILE

```yaml
name: robust
tier: 2
purpose: Production deployment

scoring:
  weights:
    phonetic: 0.4
    structural: 0.3
    semantic: 0.3
  missing_data_strategy: "renorm"
  use_pair_confidence: true

aggregation:
  strategy: "trimmed_mean"
  trim_percent: 0.1
  outlier_removal: true
  return_statistics: true

phonetic_analysis:
  algorithm: "rime"
  top_k_matches: 2
  length_penalty: 0.1

semantic_analysis:
  model: "all-MiniLM-L6-v2"
  batch_size: 32
  cache_embeddings: true
  device: "cpu"

clause_detection:
  min_length: 3
  max_length: 50
  strict_mode: false

pair_rules:
  conjunctions: ["but", "yet", "however", "and", "or"]
  transitions: ["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,"]
  min_pair_confidence: 0.3
  use_confidence_weighting: true

zone_extraction:
  window_size: 3
  min_zone_length: 1
  exclude_discourse_markers: true
  discourse_markers: ["however", "therefore", "thus", "moreover"]
```

### RESEARCH_PROFILE

```yaml
name: research
tier: 3
purpose: Experimental optimization

scoring:
  weights:
    phonetic: 0.4
    structural: 0.3
    semantic: 0.3
  missing_data_strategy: "renorm"
  use_pair_confidence: true
  calibration:
    type: "sigmoid"
    params:
      k: 10
      x0: 0.5

aggregation:
  strategy: "winsorized_mean"
  winsor_percent: 0.05
  outlier_removal: true
  return_statistics: true

phonetic_analysis:
  algorithm: "hungarian"
  use_stress_patterns: true
  cache_results: true

semantic_analysis:
  model: "sentence-transformers/all-mpnet-base-v2"
  use_antonym_detection: true
  batch_size: 64
  device: "cuda"
  cache_embeddings: true

clause_detection:
  min_length: 2
  max_length: 100
  strict_mode: false
  cross_sentence_pairing: true

pair_rules:
  conjunctions: ["but", "yet", "however", "and", "or", "nor", "so"]
  transitions: ["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,", "Nonetheless,", "Furthermore,"]
  min_pair_confidence: 0.2
  use_confidence_weighting: true

zone_extraction:
  window_size: 5
  min_zone_length: 1
  exclude_discourse_markers: true
  adaptive_window: true
```

---

## TIER_TRANSITION_CHECKLIST

### TIER_1_TO_TIER_2

```yaml
required_before_transition:
  tasks:
    - All 32 Tier 1 tasks complete (1.1 through 8.6)
    - All unit tests passing
    - All integration tests passing
  
  testing:
    - Code coverage > 80%
    - At least 5 integration tests with real text samples
    - At least 50 test documents analyzed
  
  validation:
    - Baseline corpus processed and statistics validated
    - End-to-end pipeline runs without errors
    - CLI functional on real documents
    - "simple" config profile stable
  
  measurement:
    - False positive/negative rate measured
    - Performance benchmarked (time per document, memory usage)
    - 2-3 real limitations identified through actual usage (not theoretical)

blockers:
  - Adding Tier 2 features "just in case" without proven need
  - Unfixed bugs in Tier 1 code
  - No real-world testing completed
  - Tests failing or skipped
  - No measurement of actual performance

critical_rule: |
  Tier 1 must be complete, functional, and validated before any Tier 2 work begins.
  Do not use Tier 2 as an excuse to skip Tier 1 validation.
```

### TIER_2_TO_TIER_3

```yaml
required_before_transition:
  deployment:
    - Tier 2 deployed in production environment
    - System running for 2+ weeks with real usage
    - User feedback collected (if applicable)
  
  measurement:
    - Performance bottlenecks documented with profiling data
    - False positive/negative rate < 5% on validation corpus
    - Specific Tier 3 features identified that address measured problems
    - ROI analysis shows Tier 3 features provide >2x improvement
  
  code_quality:
    - "robust" config profile tuned and validated
    - Code review completed
    - Technical debt addressed
    - Documentation updated

blockers:
  - Adding Tier 3 complexity because it sounds cool
  - No measurement of Tier 2 effectiveness
  - Trying to fix Tier 2 bugs by adding Tier 3 features
  - No production data justifying advanced algorithms
  - ROI doesn't justify complexity increase

critical_rule: |
  Tier 3 features must be justified by real production data showing
  they provide significant improvement over Tier 2. Never add Tier 3
  features speculatively.
```

---

## IMPLEMENTATION_GUIDELINES

### WHEN_IMPLEMENTING_TIER_1

```yaml
do:
  - Implement exactly what Tier 1 specifies, nothing more
  - Use simplest possible algorithms
  - Write tests before moving to next task
  - Document assumptions and limitations
  - Use "simple" config profile exclusively
  - Validate each component works before proceeding

do_not:
  - Add Tier 2 features even if they seem easy
  - Optimize prematurely
  - Skip tests to "go faster"
  - Jump ahead in task sequence
  - Implement features without specification
  - Use complex algorithms when simple ones suffice
```

### WHEN_IMPLEMENTING_TIER_2

```yaml
do:
  - Start with Tier 1 validation results
  - Address measured limitations only
  - Add features incrementally with testing
  - Use "robust" config profile
  - Maintain backward compatibility with Tier 1
  - Document all changes and rationale

do_not:
  - Add features without data justification
  - Break Tier 1 functionality
  - Skip comparative benchmarks
  - Implement Tier 3 features early
```

### WHEN_IMPLEMENTING_TIER_3

```yaml
do:
  - Have production data justifying each feature
  - Implement experimental features as optional
  - Maintain fallback to Tier 2 behavior
  - A/B test against Tier 2 baseline
  - Document performance characteristics
  - Measure actual improvement

do_not:
  - Assume more complexity = better results
  - Make Tier 3 features required
  - Remove Tier 2 implementations
  - Skip benchmarking
```

---

## TESTING_STRATEGY_BY_TIER

### TIER_1_TESTING

```yaml
unit_tests:
  coverage: Happy path + simple edge cases
  data: Known good inputs
  mocking: Mock external dependencies
  assertions: Output correctness, type safety

integration_tests:
  coverage: End-to-end pipeline with sample texts
  data: 5-10 representative documents
  assertions: Pipeline completes, reasonable outputs

performance_tests:
  coverage: Basic timing measurements
  data: Documents of varying lengths (100-10000 words)
  assertions: No crashes, completes in reasonable time
```

### TIER_2_TESTING

```yaml
unit_tests:
  coverage: All edge cases from spec
  data: Edge cases, malformed inputs, boundary conditions
  stress: Very long inputs, unusual characters
  assertions: Graceful error handling

integration_tests:
  coverage: Comprehensive end-to-end scenarios
  data: 50+ diverse documents
  assertions: Accuracy metrics, false positive/negative rates

performance_tests:
  coverage: Profiling and optimization
  data: Large corpus (1000+ documents)
  assertions: Performance targets met, memory usage acceptable
```

### TIER_3_TESTING

```yaml
unit_tests:
  coverage: Algorithm correctness proofs
  data: Adversarial inputs
  assertions: Mathematical correctness

integration_tests:
  coverage: Comparative benchmarks
  data: Same corpus as Tier 2
  assertions: Improvement over Tier 2 measured and significant

performance_tests:
  coverage: Ablation studies
  data: Stratified test corpus
  assertions: Each Tier 3 feature shows measurable benefit
```

---

END OF SPECS