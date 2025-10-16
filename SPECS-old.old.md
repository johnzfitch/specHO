# SpecHO Implementation Specifications

**Purpose:** Detailed tier specifications for implementing each component. Reference this file when Claude Code asks for component-specific details.

**Format:** YAML-structured data for machine parsing.

---

## Configuration Profiles

All three configuration profiles required for system initialization. Load using `config.load_config(profile_name)`.

```yaml
profiles:
  simple:
    name: "Simple (MVP/Tier 1)"
    description: "Minimum viable product. All features working, limited optimization."
    scoring:
      weights:
        phonetic: 0.33
        structural: 0.33
        semantic: 0.33
      missing_data_strategy: "zero"
    aggregation:
      strategy: "mean"
      trim_percent: null
      outlier_removal: false
      return_statistics: false
    phonetic_analysis:
      algorithm: "levenshtein"
      top_k_matches: null
      length_penalty: null
      use_stress_patterns: false
    structural_analysis:
      features:
        pos_pattern:
          enabled: true
          weight: 0.5
        syllable_count:
          enabled: true
          weight: 0.5
        word_properties:
          enabled: false
    semantic_analysis:
      model: "static"  # Word2Vec or GloVe
      use_antonym_detection: false
      batch_size: 1
      device: "cpu"
      cache_embeddings: false
    clause_detection:
      min_length: 3
      max_length: 50
      punctuation: [";"]
      dependency_labels: ["ROOT", "conj"]
      strict_mode: false
    pair_rules:
      conjunctions: ["but", "yet", "and", "or"]
      transitions: []
      min_pair_confidence: null
      use_confidence_weighting: false
    zone_extraction:
      window_size: 3
      min_zone_length: 1
      exclude_discourse_markers: false

  robust:
    name: "Robust (Tier 2)"
    description: "Production-ready. Enhanced algorithms, better error handling, performance optimized."
    scoring:
      weights:
        phonetic: 0.4
        structural: 0.3
        semantic: 0.3
      missing_data_strategy: "renorm"
    aggregation:
      strategy: "trimmed_mean"
      trim_percent: 0.1
      outlier_removal: true
      return_statistics: true
    phonetic_analysis:
      algorithm: "rime"
      top_k_matches: 2
      length_penalty: 0.1
      use_stress_patterns: false
    structural_analysis:
      features:
        pos_pattern:
          enabled: true
          weight: 0.4
        syllable_count:
          enabled: true
          weight: 0.3
        word_properties:
          enabled: true
          weight: 0.3
    semantic_analysis:
      model: "all-MiniLM-L6-v2"
      use_antonym_detection: true
      batch_size: 32
      device: "cpu"
      cache_embeddings: true
    clause_detection:
      min_length: 3
      max_length: 50
      punctuation: [";", ":", "—"]
      dependency_labels: ["ROOT", "conj", "advcl", "ccomp"]
      strict_mode: false
    pair_rules:
      conjunctions: ["but", "yet", "however", "and", "or", "nor"]
      transitions: ["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,"]
      min_pair_confidence: 0.3
      use_confidence_weighting: true
    zone_extraction:
      window_size: 3
      min_zone_length: 1
      exclude_discourse_markers: true
      discourse_markers: ["however", "therefore", "thus", "moreover", "furthermore"]

  research:
    name: "Research (Tier 3)"
    description: "Experimental. Advanced algorithms, maximum accuracy, research-grade."
    scoring:
      weights:
        phonetic: 0.4
        structural: 0.3
        semantic: 0.3
      missing_data_strategy: "renorm"
    aggregation:
      strategy: "winsorized_mean"
      trim_percent: 0.05
      outlier_removal: true
      return_statistics: true
    phonetic_analysis:
      algorithm: "hungarian"
      top_k_matches: 3
      length_penalty: 0.15
      use_stress_patterns: true
    structural_analysis:
      features:
        pos_pattern:
          enabled: true
          weight: 0.35
        syllable_count:
          enabled: true
          weight: 0.25
        word_properties:
          enabled: true
          weight: 0.4
    semantic_analysis:
      model: "all-mpnet-base-v2"
      use_antonym_detection: true
      batch_size: 64
      device: "cuda"  # GPU support
      cache_embeddings: true
    clause_detection:
      min_length: 2
      max_length: 100
      punctuation: [";", ":", "—", "."]
      dependency_labels: ["ROOT", "conj", "advcl", "ccomp", "acl", "relcl"]
      strict_mode: false
    pair_rules:
      conjunctions: ["but", "yet", "however", "and", "or", "nor", "either", "also"]
      transitions: ["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,", "Moreover,"]
      min_pair_confidence: 0.2
      use_confidence_weighting: true
    zone_extraction:
      window_size: 5
      min_zone_length: 1
      exclude_discourse_markers: true
      discourse_markers: ["however", "therefore", "thus", "moreover", "furthermore", "additionally"]
```

---

## Component 2: Clause Identifier Specifications

Detailed breakdown of boundary detection, pair rules, and zone extraction across all tiers.

### Boundary Detector

```yaml
component: "Clause Boundary Detector"
class_path: "preprocessor.boundary_detector.ClauseBoundaryDetector"
api:
  primary: "identify_clauses(doc: spacy.tokens.Doc) -> List[Clause]"

tier_1:
  description: "MVP clause boundary detection using basic dependency rules."
  algorithms:
    - name: "Finite verb heads"
      description: "Split on ROOT and conj dependency labels"
      coverage: "simple sentences"
    - name: "Punctuation rules"
      description: "Period, semicolon only"
      coverage: "end-of-clause markers"
  edge_cases_handled:
    - empty_input: "Return empty list"
    - single_clause: "Return one Clause object"
    - contractions: "spaCy handles natively"
  deliverables:
    - "List[Clause] with start_idx, end_idx, clause_type, tokens"
  testing:
    - "Test simple sentences (SVO structure)"
    - "Test compound sentences (with conjunctions)"
    - "Test empty/single word input"

tier_2:
  additions:
    - "Relative clause detection (acl/relcl with mark)"
    - "Fragment merging (merge clauses < 3 tokens)"
    - "Quote/parentheses trimming"
  config_params:
    - "min_clause_length: 3"
    - "max_clause_length: 50"
    - "strict_mode: False"
  edge_cases_added:
    - "quoted clauses"
    - "parenthetical asides"
    - "sentence fragments"
  testing:
    - "Test complex sentences (nested clauses)"
    - "Test quoted speech"
    - "Test fragments"

tier_3:
  additions:
    - "Multi-sentence cross-boundary pairing"
    - "Sophisticated trimming of sentence-initial adverbials"
    - "List detection after colons"
  advanced_features:
    - "Tie-breaker logic for overlapping spans"
    - "Context-aware minimum length"
  testing:
    - "Test multi-sentence documents"
    - "Test edge cases in corpus"
```

### Pair Rules Engine

```yaml
component: "Pair Rules Engine"
class_path: "clause_identifier.pair_rules.PairRulesEngine"
api:
  methods:
    - "apply_rule_a(clauses: List[Clause]) -> List[ClausePair]"
    - "apply_rule_b(clauses: List[Clause]) -> List[ClausePair]"
    - "apply_rule_c(clauses: List[Clause]) -> List[ClausePair]"

tier_1:
  rule_a:
    description: "Punctuation-based pairing"
    triggers: [";"]
    logic: "If two clauses separated by semicolon, pair them"
  rule_b:
    description: "Conjunction-based pairing"
    triggers: ["but", "yet", "and", "or"]
    logic: "If conjunction between clauses, pair them"
  rule_c:
    description: "Transition-based pairing"
    triggers: []  # Empty in Tier 1
    logic: "No transitions in Tier 1"
  deliverables:
    - "List[ClausePair] with clause_a, clause_b"
    - "Simple de-duplication by clause indices"
  testing:
    - "Test Rule A with semicolons"
    - "Test Rule B with basic conjunctions"
    - "Test no duplicate pairs"

tier_2:
  rule_a_enhanced:
    signals:
      - signal: ";"
        weight: 1.0
      - signal: "—"
        weight: 0.8
      - signal: ":"
        weight: 0.6
  rule_b_enhanced:
    conjunctions:
      basic: ["but", "yet", "and", "or", "nor"]
      correlative: ["either...or", "neither...nor", "not only...but also"]
  rule_c_transitions:
    triggers: ["However,", "Therefore,", "Thus,", "In contrast,", "Meanwhile,"]
  pair_metadata:
    - "pair_type: string (rule_a, rule_b, rule_c)"
    - "rule_id: int"
    - "confidence: float (0.0-1.0)"
    - "rationale: string"
  testing:
    - "Test confidence scoring"
    - "Test correlative conjunctions"
    - "Test transition phrases"

tier_3:
  additions:
    - "Lightweight pre-scoring based on zone content-word overlap"
    - "False positive detection for parentheticals"
    - "List detection after colons (skip if no verb)"
  testing:
    - "A/B test pre-scoring accuracy"
    - "Test on diverse corpus"
```

### Zone Extractor

```yaml
component: "Zone Extractor"
class_path: "clause_identifier.zone_extractor.ZoneExtractor"
api:
  primary: "extract_zones(clause_pair: ClausePair) -> Tuple[List[Token], List[Token]]"

tier_1:
  logic: |
    zone_a = last 3 content words from clause_a
    zone_b = first 3 content words from clause_b
    content_word = noun, verb, or adjective
  deliverables:
    - "Tuple[List[Token], List[Token]] (zone_a, zone_b)"
    - "Both zones containing Token objects with all fields"
  testing:
    - "Test with 3-word zones"
    - "Test with longer zones"

tier_2:
  enhancements:
    - "Lemmatization of tokens"
    - "Exclude discourse markers (however, therefore, etc.)"
    - "Handle short clauses: use all content words if < 3"
    - "Leading/trailing quote/parenthesis trimming"
    - "Deterministic sorting by token index"
  discourse_markers:
    - "however", "therefore", "thus", "moreover", "furthermore"
  testing:
    - "Test lemmatization accuracy"
    - "Test short clause handling"
    - "Test quote removal"

tier_3:
  additions:
    - "Configurable window sizes (1-5 words)"
    - "Priority weighting (terminal vs initial)"
    - "Low-confidence marking for edge cases"
  testing:
    - "Compare window size performance"
```

---

## Component 3: Echo Engine Specifications

Phonetic, structural, and semantic analysis with tier progression.

### Phonetic Analyzer

```yaml
component: "Phonetic Echo Analyzer"
class_path: "echo_engine.phonetic_analyzer.PhoneticEchoAnalyzer"
api:
  primary: "analyze(zone_a: List[Token], zone_b: List[Token]) -> float"

tier_1:
  algorithm: "levenshtein"
  logic: |
    1. Extract phonetic strings from zone_a and zone_b tokens
    2. Calculate Levenshtein distance between all pairs
    3. Find best matches between zones
    4. Normalize to 0.0-1.0 similarity score
    5. Return average of top matches
  edge_cases:
    - empty_zones: "Return 0.0"
    - oov_words: "Use fallback G2P"
  deliverables:
    - "float: 0.0-1.0 similarity score"
  testing:
    - "Test with known rhyming pairs"
    - "Test with non-rhyming pairs"
    - "Test edge cases"

tier_2:
  algorithm: "rime-based"
  enhancements:
    - "Rime comparison (last stressed vowel onward)"
    - "Multiple features: rhyme match, onset overlap, syllable similarity"
    - "Aggregation: average top-k matches with length penalty"
    - "Strip punctuation and interjections"
  parameters:
    top_k_matches: 2
    length_penalty: 0.1
  testing:
    - "Test rime-based scoring"
    - "Test length penalties"

tier_3:
  algorithm: "hungarian"
  features:
    - "Phoneme-level feature extraction"
    - "Stress pattern matching"
    - "LRU cache for phonetic computations"
  parameters:
    top_k_matches: 3
    length_penalty: 0.15
    use_stress_patterns: true
```

### Structural Analyzer

```yaml
component: "Structural Echo Analyzer"
class_path: "echo_engine.structural_analyzer.StructuralEchoAnalyzer"
api:
  primary: "analyze(zone_a: List[Token], zone_b: List[Token]) -> float"

tier_1:
  features:
    - "POS pattern comparison"
    - "Syllable count similarity"
  logic: |
    1. Extract POS tags from both zones
    2. Compare patterns (exact match or similarity)
    3. Calculate syllable counts for each zone
    4. Compare syllable counts
    5. Return: pattern_sim * 0.5 + syllable_sim * 0.5
  testing:
    - "Test with matching POS patterns"
    - "Test with different POS patterns"
    - "Test syllable counting"

tier_2:
  features:
    - pos_pattern:
        enabled: true
        algorithm: "LCS"  # Longest Common Subsequence
        description: "2*LCS/(lenA+lenB)"
        weight: 0.4
    - syllable_count:
        enabled: true
        algorithm: "euclidean_distance"
        description: "Normalize to 0-1"
        weight: 0.3
    - word_properties:
        enabled: true
        description: "Content-word ratios, NER presence"
        weight: 0.3
  coarse_pos_mapping:
    - "NOUN, PROPN -> NOUN"
    - "VERB, AUX -> VERB"
    - "ADJ, ADV -> MOD"
  short_zone_fallback: "Use length similarity only if zone < 2 words"
  testing:
    - "Test LCS algorithm"
    - "Test weighted aggregation"
    - "Test short zone handling"

tier_3:
  additions:
    - "Abstract noun detection"
    - "Syntactic role similarity"
    - "Configurable feature weights"
  testing:
    - "Compare performance with/without abstract noun detection"
```

### Semantic Analyzer

```yaml
component: "Semantic Echo Analyzer"
class_path: "echo_engine.semantic_analyzer.SemanticEchoAnalyzer"
api:
  primary: "analyze(zone_a: List[Token], zone_b: List[Token]) -> float"

tier_1:
  model: "static"  # Word2Vec or GloVe
  logic: |
    1. Load pre-trained word embeddings
    2. Mean-pool embeddings for each zone
    3. Calculate cosine similarity
    4. Map to 0-1 range: (1 + cos) / 2
    5. Return 0.5 if embeddings unavailable (neutral)
  oov_handling: "Return neutral (0.5)"
  testing:
    - "Test with known synonyms"
    - "Test with known antonyms"
    - "Test with OOV words"

tier_2:
  model: "all-MiniLM-L6-v2"  # Sentence-Transformers
  enhancements:
    - "Sentence-level embeddings (all words, not pooled)"
    - "OOV backoff: try static vectors, then neutral"
    - "Antonym detection via polarity check"
    - "Batch processing for multiple pairs"
    - "Embedding cache (LRU + optional disk)"
  parameters:
    batch_size: 32
    device: "cpu"
    cache_embeddings: true
    use_antonym_detection: true
  testing:
    - "Test sentence-level vs word-level embeddings"
    - "Test antonym detection"
    - "Test cache performance"

tier_3:
  model: "all-mpnet-base-v2"  # Larger transformer
  features:
    - "Transformer-based embeddings (BERT, RoBERTa)"
    - "WordNet antonym relationships"
    - "Negation handling"
    - "GPU acceleration"
  parameters:
    batch_size: 64
    device: "cuda"
    use_antonym_detection: true
  testing:
    - "Benchmark against Tier 2"
    - "Test on diverse corpus"
```

---

## Tier Transition Metrics

Quantitative criteria for advancing between tiers.

```yaml
tier_1_to_tier_2_requirements:
  code_completion:
    all_tasks_complete: true
    unit_test_coverage: ">80%"
    integration_tests: ">= 5 passing"
  functionality:
    end_to_end_works: true
    baseline_corpus_processed: true
    no_critical_errors: true
  usage:
    real_text_samples: ">= 10"
    identified_limitations: ">= 2"
  performance:
    documentation_complete: true
    code_reviewed: true

tier_2_to_tier_3_requirements:
  deployment:
    production_weeks: ">= 2"
    real_users: ">= 1"
  measurements:
    false_positive_rate: "< 10%"
    false_negative_rate: "< 10%"
    false_positive_rate_measured: true
  analysis:
    performance_bottleneck: "identified and measured"
    profiling_data: "available"
    roi_analysis: "shows >2x improvement needed"
  decision:
    tier_3_features_justified: true
    data_driven_decision: true
```

---

## Testing Strategy by Tier

How to test each tier's implementation.

```yaml
tier_1_testing:
  scope: "MVP functionality only"
  approach:
    - "Happy path testing (normal inputs)"
    - "Known good inputs only"
    - "Simple edge cases (empty, single item)"
  coverage_target: "80%"
  test_examples:
    - "tokenize('The cat sat.') -> List[Token] with 4 items"
    - "identify_clauses(doc) -> List[Clause] with expected boundaries"
    - "analyze(zone_a, zone_b) -> float between 0.0 and 1.0"
  regression_tests: "none"

tier_2_testing:
  scope: "Production reliability"
  additions:
    - "All edge cases from spec"
    - "Stress tests (very long inputs)"
    - "Malformed inputs (invalid UTF-8, etc.)"
    - "Performance benchmarks"
    - "Error handling verification"
  coverage_target: "90%"
  regression_testing: true
  performance_tests:
    - "Analyze 10,000 word document"
    - "Process 100 documents sequentially"
    - "Memory usage profiling"

tier_3_testing:
  scope: "Research-grade validation"
  additions:
    - "Comparative benchmarks (algorithm A vs B)"
    - "Ablation studies (with/without feature X)"
    - "Cross-validation on held-out corpus"
    - "Statistical significance testing"
  coverage_target: "95%"
  ablation_studies:
    - "Phonetic analysis impact"
    - "Structural analysis impact"
    - "Semantic analysis impact"
    - "Weight combinations"
```

---

## Reference

For task-specific implementation details, see TASKS.md. For architecture overview, see CLAUDE.md. For deployment guidance, see DEPLOYMENT.md.
