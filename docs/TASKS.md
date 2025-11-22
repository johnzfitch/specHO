# <img src="../icons/task-list.png" width="32" height="32"> TASKS

**Format:** Machine-readable task specifications  
**Total Tasks:** 32  
**Implementation:** Sequential (follow order)

---

## TASK_1.1

```yaml
id: 1.1
phase: Foundation
file: SpecHO/models.py
type: dataclasses
tier: 1
libraries: [dataclasses, typing]
dependencies: []

deliverables:
  - Token dataclass: (text, pos_tag, phonetic, is_content_word, syllable_count)
  - Clause dataclass: (tokens, start_idx, end_idx, clause_type)
  - ClausePair dataclass: (clause_a, clause_b, zone_a_tokens, zone_b_tokens, pair_type)
  - EchoScore dataclass: (phonetic_score, structural_score, semantic_score, combined_score)
  - DocumentAnalysis dataclass: (text, clause_pairs, echo_scores, final_score, z_score, confidence)

notes: No processing logic, data structures only
```

## TASK_1.2

```yaml
id: 1.2
phase: Foundation
file: SpecHO/config.py
type: configuration_system
tier: 1
libraries: [dataclasses, typing]
dependencies: [Task 1.1]

deliverables:
  - SpecHOConfig dataclass with component-level configs
  - PROFILES dict with three profiles: simple, robust, research
  - load_config() function with override support
  
specifications:
  - simple profile: Tier 1 MVP settings
  - robust profile: Tier 2 production settings  
  - research profile: Tier 3 experimental settings

notes: Start with simple profile only, add others as scaffolding
```

## TASK_7.3

```yaml
id: 7.3
phase: Utilities
file: SpecHO/utils.py
type: utility_functions
tier: 1
libraries: [logging, pathlib]
dependencies: [Task 1.1]

deliverables:
  - load_text_file(path: str) -> str
  - save_analysis_results(analysis: DocumentAnalysis, output_path: str)
  - setup_logging(level: str)
  - Error handling decorators

notes: Basic implementations, enhance in Tier 2
```

## TASK_2.1

```yaml
id: 2.1
phase: Preprocessor
file: SpecHO/preprocessor/tokenizer.py
class: Tokenizer
tier: 1
libraries: [spacy]
dependencies: [Task 1.1]

api:
  - tokenize(text: str) -> List[Token]

features:
  - Integrate with spaCy tokenizer
  - Handle contractions and hyphenated words
  - Return Token objects with text field populated

notes: Minimal implementation, other Token fields populated by later components
```

## TASK_2.2

```yaml
id: 2.2
phase: Preprocessor
file: SpecHO/preprocessor/pos_tagger.py
class: POSTagger
tier: 1
libraries: [spacy]
dependencies: [Task 1.1, Task 2.1]

api:
  - tag(tokens: List[Token]) -> List[Token]
  - is_content_word(token: Token) -> bool

features:
  - Enrich tokens with POS tags
  - Identify content words (nouns, verbs, adjectives)
  - Filter stopwords for zone extraction

notes: Use spaCy's POS tagger
```

## TASK_2.3

```yaml
id: 2.3
phase: Preprocessor
file: SpecHO/preprocessor/dependency_parser.py
class: DependencyParser
tier: 1
libraries: [spacy]
dependencies: [Task 1.1]

api:
  - parse(text: str) -> spacy.tokens.Doc
  - get_clause_boundaries(doc: spacy.tokens.Doc) -> List[Tuple[int, int]]

features:
  - Extract dependency tree structures
  - Identify clause boundaries using dependency relations

notes: Return spaCy Doc object for downstream use
```

## TASK_2.4

```yaml
id: 2.4
phase: Preprocessor
file: SpecHO/preprocessor/phonetic.py
class: PhoneticTranscriber
tier: 1
libraries: [pronouncing]
dependencies: [Task 1.1]

api:
  - transcribe(word: str) -> str
  - get_stressed_syllables(phonetic: str) -> List[str]

features:
  - Convert words to ARPAbet representation
  - Handle OOV words with fallback rules
  - Extract stressed syllables

notes: Use pronouncing library for dictionary lookup
```

## TASK_2.5

```yaml
id: 2.5
phase: Preprocessor
file: SpecHO/preprocessor/pipeline.py
class: LinguisticPreprocessor
tier: 1
libraries: []
dependencies: [Tasks 2.1, 2.2, 2.3, 2.4]

api:
  - process(text: str) -> Tuple[List[Token], spacy.tokens.Doc]

features:
  - Chain all preprocessor components
  - Return enriched tokens and dependency parse

notes: Orchestrator pattern, minimal logic
```

## TASK_8.1

```yaml
id: 8.1
phase: Testing
file: tests/test_preprocessor.py
type: unit_tests
tier: 1
libraries: [pytest, pytest-mock]
dependencies: [Tasks 2.1-2.5]

test_coverage:
  - Tokenization with various inputs
  - POS tagging accuracy
  - Phonetic transcription (known words)
  - Dependency parsing with sample sentences

notes: Use fixtures for sample data
```

## TASK_3.1

```yaml
id: 3.1
phase: Clause Identifier
file: SpecHO/clause_identifier/boundary_detector.py
class: ClauseBoundaryDetector
tier: 1
libraries: [spacy]
dependencies: [Task 1.1]

api:
  - identify_clauses(doc: spacy.tokens.Doc) -> List[Clause]

tier_1_features:
  - Basic ROOT/conj detection
  - Simple punctuation rules (period, semicolon, em dash)
  - Subordinate clauses (advcl, ccomp)

notes: See docs/SPECS.md for tier details
```

## TASK_3.2

```yaml
id: 3.2
phase: Clause Identifier
file: SpecHO/clause_identifier/pair_rules.py
class: PairRulesEngine
tier: 1
libraries: [re]
dependencies: [Task 1.1, Task 3.1]

api:
  - apply_rule_a(clauses: List[Clause]) -> List[ClausePair]  # Punctuation
  - apply_rule_b(clauses: List[Clause]) -> List[ClausePair]  # Conjunction
  - apply_rule_c(clauses: List[Clause]) -> List[ClausePair]  # Transition

tier_1_features:
  - Rule A: Semicolon and em dash only
  - Rule B: Basic conjunctions (but, and, or)
  - Rule C: Common transitions (However, Therefore, Thus)
  - Simple de-duplication by clause indices

notes: See docs/SPECS.md for rule specifications
```

## TASK_3.3

```yaml
id: 3.3
phase: Clause Identifier
file: SpecHO/clause_identifier/zone_extractor.py
class: ZoneExtractor
tier: 1
libraries: []
dependencies: [Task 1.1]

api:
  - extract_zones(clause_pair: ClausePair) -> Tuple[List[Token], List[Token]]
  - get_terminal_content_words(clause: Clause, n: int = 3) -> List[Token]
  - get_initial_content_words(clause: Clause, n: int = 3) -> List[Token]

tier_1_features:
  - Extract last 3 content words from clause_a
  - Extract first 3 content words from clause_b
  - Basic content word detection (nouns, verbs, adjectives)

notes: Simple implementation, enhance in Tier 2
```

## TASK_3.4

```yaml
id: 3.4
phase: Clause Identifier
file: SpecHO/clause_identifier/pipeline.py
class: ClauseIdentifier
tier: 1
libraries: []
dependencies: [Tasks 3.1, 3.2, 3.3]

api:
  - identify_pairs(tokens: List[Token], doc: spacy.tokens.Doc) -> List[ClausePair]

features:
  - Apply all rules
  - Extract zones for each pair
  - Return complete ClausePair objects

notes: Orchestrator pattern
```

## TASK_8.2

```yaml
id: 8.2
phase: Testing
file: tests/test_clause_identifier.py
type: unit_tests
tier: 1
libraries: [pytest]
dependencies: [Tasks 3.1-3.4]

test_coverage:
  - Clause boundary detection with known structures
  - Each pair rule (A, B, C) independently
  - Zone extraction with various clause lengths

notes: Create fixtures with sample sentences
```

## TASK_4.1

```yaml
id: 4.1
phase: Echo Engine
file: SpecHO/echo_engine/phonetic_analyzer.py
class: PhoneticEchoAnalyzer
tier: 1
libraries: [python-Levenshtein]
dependencies: [Task 1.1]

api:
  - analyze(zone_a: List[Token], zone_b: List[Token]) -> float
  - calculate_phonetic_distance(phoneme_a: str, phoneme_b: str) -> float

tier_1_features:
  - Simple Levenshtein distance on ARPAbet strings
  - Normalize to [0,1] similarity
  - Pairwise comparison, best match
  - Return 0 if either zone empty

notes: Use Token.phonetic field from preprocessor
```

## TASK_4.2

```yaml
id: 4.2
phase: Echo Engine
file: SpecHO/echo_engine/structural_analyzer.py
class: StructuralEchoAnalyzer
tier: 1
libraries: []
dependencies: [Task 1.1]

api:
  - analyze(zone_a: List[Token], zone_b: List[Token]) -> float
  - compare_pos_patterns(zone_a: List[Token], zone_b: List[Token]) -> float
  - compare_syllable_counts(zone_a: List[Token], zone_b: List[Token]) -> float
  - compare_word_properties(zone_a: List[Token], zone_b: List[Token]) -> float

tier_1_features:
  - Simple POS pattern comparison
  - Syllable count similarity
  - Basic scoring: pattern_sim * 0.5 + syllable_sim * 0.5

notes: Tier 1 is intentionally simple
```

## TASK_4.3

```yaml
id: 4.3
phase: Echo Engine
file: SpecHO/echo_engine/semantic_analyzer.py
class: SemanticEchoAnalyzer
tier: 1
libraries: [gensim, numpy]
dependencies: [Task 1.1]

api:
  - analyze(zone_a: List[Token], zone_b: List[Token]) -> float
  - get_word_vector(word: str) -> np.ndarray
  - calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float

tier_1_features:
  - Mean-pooled word embeddings (Word2Vec or GloVe)
  - Cosine similarity mapped to [0,1]: (1 + cos) / 2
  - Return 0.5 if embeddings unavailable

notes: Load pre-trained embeddings in __init__
```

## TASK_4.4

```yaml
id: 4.4
phase: Echo Engine
file: SpecHO/echo_engine/pipeline.py
class: EchoAnalysisEngine
tier: 1
libraries: []
dependencies: [Tasks 4.1, 4.2, 4.3]

api:
  - analyze_pair(clause_pair: ClausePair) -> EchoScore

features:
  - Run all three analyzers
  - Return consolidated EchoScore object

notes: Orchestrator pattern
```

## TASK_8.3

```yaml
id: 8.3
phase: Testing
file: tests/test_echo_analyzers.py
type: unit_tests
tier: 1
libraries: [pytest]
dependencies: [Tasks 4.1-4.4]

test_coverage:
  - Phonetic similarity with known phoneme pairs
  - Structural similarity with known POS patterns
  - Semantic similarity with synonym/antonym pairs
  - Score ranges (0.0-1.0)

notes: Mock embeddings for semantic tests
```

## TASK_5.1

```yaml
id: 5.1
phase: Scoring
file: SpecHO/scoring/weighted_scorer.py
class: WeightedScorer
tier: 1
libraries: [numpy]
dependencies: [Task 1.1, Task 1.2]

api:
  - calculate_pair_score(echo_score: EchoScore, weights: Dict[str, float]) -> float

tier_1_features:
  - Simple weighted sum: w_p * phonetic + w_s * structural + w_sem * semantic
  - Fixed weights from config: {phonetic: 0.33, structural: 0.33, semantic: 0.33}
  - NaN handling: treat as 0
  - Clip to [0,1]

notes: Load weights from config.simple profile
```

## TASK_5.2

```yaml
id: 5.2
phase: Scoring
file: SpecHO/scoring/aggregator.py
class: DocumentAggregator
tier: 1
libraries: [statistics]
dependencies: []

api:
  - aggregate_scores(pair_scores: List[float]) -> float

tier_1_features:
  - Simple mean of all pair scores
  - Return 0.0 if no pairs
  - Emit warning for empty input

notes: statistics.mean() is sufficient for Tier 1
```

## TASK_5.3

```yaml
id: 5.3
phase: Scoring
file: SpecHO/scoring/pipeline.py
class: ScoringModule
tier: 1
libraries: []
dependencies: [Tasks 5.1, 5.2]

api:
  - score_document(echo_scores: List[EchoScore]) -> float

features:
  - Orchestrate weighted scoring and aggregation
  - Return document-level score

notes: Orchestrator pattern
```

## TASK_8.4

```yaml
id: 8.4
phase: Testing
file: tests/test_scoring.py
type: unit_tests
tier: 1
libraries: [pytest]
dependencies: [Tasks 5.1-5.3]

test_coverage:
  - Weighted scoring with known weights and scores
  - Aggregation with various score distributions
  - Edge cases (empty lists, single pair)

notes: Use hardcoded test data
```

## TASK_6.1

```yaml
id: 6.1
phase: Validator
file: SpecHO/validator/baseline_builder.py
class: BaselineCorpusProcessor
tier: 1
libraries: [numpy, pickle, tqdm]
dependencies: [Task 1.1, all prior components]

api:
  - process_corpus(corpus_path: str) -> Dict[str, float]
  - save_baseline(baseline_stats: Dict, output_path: str)
  - load_baseline(baseline_path: str) -> Dict[str, float]

tier_1_features:
  - Run SpecHO pipeline on corpus texts
  - Calculate human_mean_score and human_std_dev
  - Save/load with pickle
  - Progress tracking with tqdm

notes: Requires complete pipeline to function
```

## TASK_6.2

```yaml
id: 6.2
phase: Validator
file: SpecHO/validator/z_score.py
class: ZScoreCalculator
tier: 1
libraries: []
dependencies: []

api:
  - calculate_z_score(document_score: float, human_mean: float, human_std: float) -> float

implementation:
  formula: (document_score - human_mean) / human_std

notes: Pure function, no dependencies
```

## TASK_6.3

```yaml
id: 6.3
phase: Validator
file: SpecHO/validator/confidence.py
class: ConfidenceConverter
tier: 1
libraries: [scipy.stats]
dependencies: []

api:
  - z_score_to_percentile(z_score: float) -> float
  - z_score_to_confidence(z_score: float) -> float

implementation:
  use: scipy.stats.norm.cdf for conversion

notes: Returns confidence in [0,1] range
```

## TASK_6.4

```yaml
id: 6.4
phase: Validator
file: SpecHO/validator/pipeline.py
class: StatisticalValidator
tier: 1
libraries: []
dependencies: [Tasks 6.1, 6.2, 6.3]

api:
  - validate(document_score: float) -> Tuple[float, float]  # (z_score, confidence)

features:
  - Load baseline statistics in __init__
  - Orchestrate z-score calculation and confidence conversion

notes: Orchestrator pattern
```

## TASK_8.5

```yaml
id: 8.5
phase: Testing
file: tests/test_validator.py
type: unit_tests
tier: 1
libraries: [pytest]
dependencies: [Tasks 6.1-6.4]

test_coverage:
  - Z-score calculation with known baseline
  - Confidence conversion
  - Baseline loading/saving

notes: Use mock baseline data for tests
```

## TASK_7.1

```yaml
id: 7.1
phase: Integration
file: SpecHO/detector.py
class: SpecHODetector
tier: 1
libraries: []
dependencies: [All prior tasks]

api:
  - analyze(text: str) -> DocumentAnalysis

features:
  - Chain all five components in sequence
  - Handle errors gracefully at each stage
  - Log intermediate results for debugging
  - Return complete DocumentAnalysis

notes: Main entry point for the system
```

## TASK_7.2

```yaml
id: 7.2
phase: Integration
file: scripts/cli.py
type: command_line_interface
tier: 1
libraries: [argparse, rich]
dependencies: [Task 7.1]

features:
  - Accept text file or string input
  - Display results with confidence score
  - Optional verbose mode for detailed breakdown
  - Optional JSON output format

example_usage:
  - python scripts/cli.py --file sample.txt
  - python scripts/cli.py --text "Sample text here"
  - python scripts/cli.py --file sample.txt --verbose --json

notes: Use Rich for formatted output
```

## TASK_7.4

```yaml
id: 7.4
phase: Integration
file: scripts/build_baseline.py
type: utility_script
tier: 1
libraries: [argparse, tqdm]
dependencies: [Task 6.1, Task 7.1]

features:
  - Process large corpus of human-written text
  - Save baseline statistics to disk
  - Progress tracking and error recovery
  - Configurable corpus path and output path

example_usage:
  - python scripts/build_baseline.py --corpus data/corpus/ --output data/baseline/baseline_stats.pkl

notes: Run this once before using detector
```

## TASK_8.6

```yaml
id: 8.6
phase: Testing
file: tests/test_integration.py
type: integration_tests
tier: 1
libraries: [pytest]
dependencies: [All prior tasks]

test_coverage:
  - Full pipeline with sample human-written text
  - Full pipeline with synthetic watermarked text
  - Performance with long documents
  - Error handling throughout pipeline

notes: Use real text samples, measure end-to-end behavior
```

---

## TASK_DEPENDENCIES_GRAPH

```
1.1 (models) → 1.2 (config), 7.3 (utils)
         ↓
    2.1 (tokenizer) → 2.2 (pos_tagger) → 2.5 (preprocessor pipeline)
         ↓                                         ↓
    2.3 (dependency_parser) ────────────────────→  ↓
         ↓                                         ↓
    2.4 (phonetic) ───────────────────────────────→ ↓
                                                    ↓
                                          8.1 (preprocessor tests)
                                                    ↓
    3.1 (boundary_detector) → 3.2 (pair_rules) → 3.3 (zone_extractor) → 3.4 (clause identifier pipeline)
                                                                                    ↓
                                                                          8.2 (clause tests)
                                                                                    ↓
    4.1 (phonetic_analyzer) → 4.2 (structural_analyzer) → 4.3 (semantic_analyzer) → 4.4 (echo engine pipeline)
                                                                                              ↓
                                                                                    8.3 (echo tests)
                                                                                              ↓
    5.1 (weighted_scorer) → 5.2 (aggregator) → 5.3 (scoring pipeline)
                                                          ↓
                                                 8.4 (scoring tests)
                                                          ↓
    6.1 (baseline_builder) → 6.2 (z_score) → 6.3 (confidence) → 6.4 (validator pipeline)
                                                                          ↓
                                                                 8.5 (validator tests)
                                                                          ↓
    7.1 (detector) → 7.2 (CLI) → 7.4 (baseline script)
           ↓
    8.6 (integration tests)
```

---

END OF TASKS