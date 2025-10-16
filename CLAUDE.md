# SpecHO Development Guide

**Version:** 2.0  
**Type:** Machine-readable project specification  
**Target:** Claude Code AI assistant  
**Purpose:** Watermark detection system for AI-generated text

---

## PROJECT_METADATA

```yaml
project_name: SpecHO
objective: Detect "Echo Rule" watermark in AI-generated text
approach: Three-tier implementation (MVP → Production → Research)
total_tasks: 32
estimated_duration: 12 weeks (Tier 1)
primary_language: Python 3.11+
architecture: Five-component sequential pipeline
```

---

## DOCUMENTATION_STRUCTURE

```yaml
navigation_files:
  setup:
    - file: docs/QUICKSTART.md
      purpose: Initial environment setup and Task 1.1 implementation
      read_when: First session
    
    - file: architecture.md
      purpose: Original watermark design specification
      read_when: Need context on Echo Rule algorithm
  
  implementation:
    - file: docs/TASKS.md
      purpose: All 32 task specifications (APIs, libraries, deliverables)
      read_when: Starting any new task
    
    - file: docs/SPECS.md
      purpose: Detailed tier specifications for each component
      read_when: Implementing component features
  
  post_tier1:
    - file: docs/DEPLOYMENT.md
      purpose: Web interface and Docker deployment
      read_when: Tier 1 complete and validated
    
    - file: docs/PHILOSOPHY.md
      purpose: Design rationale and tradeoffs
      read_when: User requests context on decisions
```

---

## TIER_SYSTEM

```yaml
tier_1_mvp:
  duration: Weeks 1-12
  scope: Tasks 1-32
  constraints:
    - Implement ONLY Tier 1 specifications
    - Use simple algorithms
    - No premature optimization
  deliverable: Working CLI-based detector
  
tier_2_production:
  duration: Weeks 13-17
  trigger: Tier 1 validation complete (see TIER_TRANSITION_CHECKLIST)
  scope: Selected enhancements based on measured limitations
  deliverable: Production-ready system
  
tier_3_research:
  duration: Week 18+
  trigger: Tier 2 deployed with 2+ weeks production data
  scope: Advanced algorithms with proven ROI
  deliverable: Optimized research-grade system
```

---

## TASK_SEQUENCE

```yaml
foundation:
  task_1.1: {file: SpecHO/models.py, dataclasses: [Token, Clause, ClausePair, EchoScore, DocumentAnalysis]}
  task_1.2: {file: SpecHO/config.py, implements: three-tier config system}
  task_7.3: {file: SpecHO/utils.py, functions: [load_text_file, save_analysis, setup_logging]}

preprocessor:
  task_2.1: {file: SpecHO/preprocessor/tokenizer.py, class: Tokenizer, library: spacy}
  task_2.2: {file: SpecHO/preprocessor/pos_tagger.py, class: POSTagger, library: spacy}
  task_2.3: {file: SpecHO/preprocessor/dependency_parser.py, class: DependencyParser, library: spacy}
  task_2.4: {file: SpecHO/preprocessor/phonetic.py, class: PhoneticTranscriber, library: pronouncing}
  task_2.5: {file: SpecHO/preprocessor/pipeline.py, class: LinguisticPreprocessor}
  task_8.1: {file: tests/test_preprocessor.py, framework: pytest}

clause_identifier:
  task_3.1: {file: SpecHO/clause_identifier/boundary_detector.py, class: ClauseBoundaryDetector}
  task_3.2: {file: SpecHO/clause_identifier/pair_rules.py, class: PairRulesEngine}
  task_3.3: {file: SpecHO/clause_identifier/zone_extractor.py, class: ZoneExtractor}
  task_3.4: {file: SpecHO/clause_identifier/pipeline.py, class: ClauseIdentifier}
  task_8.2: {file: tests/test_clause_identifier.py, framework: pytest}

echo_engine:
  task_4.1: {file: SpecHO/echo_engine/phonetic_analyzer.py, class: PhoneticEchoAnalyzer, library: python-Levenshtein}
  task_4.2: {file: SpecHO/echo_engine/structural_analyzer.py, class: StructuralEchoAnalyzer}
  task_4.3: {file: SpecHO/echo_engine/semantic_analyzer.py, class: SemanticEchoAnalyzer, library: gensim}
  task_4.4: {file: SpecHO/echo_engine/pipeline.py, class: EchoAnalysisEngine}
  task_8.3: {file: tests/test_echo_analyzers.py, framework: pytest}

scoring:
  task_5.1: {file: SpecHO/scoring/weighted_scorer.py, class: WeightedScorer, library: numpy}
  task_5.2: {file: SpecHO/scoring/aggregator.py, class: DocumentAggregator}
  task_5.3: {file: SpecHO/scoring/pipeline.py, class: ScoringModule}
  task_8.4: {file: tests/test_scoring.py, framework: pytest}

validator:
  task_6.1: {file: SpecHO/validator/baseline_builder.py, class: BaselineCorpusProcessor}
  task_6.2: {file: SpecHO/validator/z_score.py, class: ZScoreCalculator, library: scipy.stats}
  task_6.3: {file: SpecHO/validator/confidence.py, class: ConfidenceConverter, library: scipy.stats}
  task_6.4: {file: SpecHO/validator/pipeline.py, class: StatisticalValidator}
  task_8.5: {file: tests/test_validator.py, framework: pytest}

integration:
  task_7.1: {file: SpecHO/detector.py, class: SpecHODetector}
  task_7.2: {file: scripts/cli.py, implements: argparse CLI}
  task_7.4: {file: scripts/build_baseline.py, implements: corpus processor}
  task_8.6: {file: tests/test_integration.py, framework: pytest}
```

---

## DIRECTORY_STRUCTURE

```
SpecHO/
├── SpecHO/              # Implementation files
│   ├── models.py        # Task 1.1 - START HERE
│   ├── config.py        # Task 1.2
│   ├── utils.py         # Task 7.3
│   ├── detector.py      # Task 7.1
│   ├── preprocessor/
│   │   ├── tokenizer.py
│   │   ├── pos_tagger.py
│   │   ├── dependency_parser.py
│   │   ├── phonetic.py
│   │   └── pipeline.py
│   ├── clause_identifier/
│   │   ├── boundary_detector.py
│   │   ├── pair_rules.py
│   │   ├── zone_extractor.py
│   │   └── pipeline.py
│   ├── echo_engine/
│   │   ├── phonetic_analyzer.py
│   │   ├── structural_analyzer.py
│   │   ├── semantic_analyzer.py
│   │   └── pipeline.py
│   ├── scoring/
│   │   ├── weighted_scorer.py
│   │   ├── aggregator.py
│   │   └── pipeline.py
│   └── validator/
│       ├── baseline_builder.py
│       ├── z_score.py
│       ├── confidence.py
│       └── pipeline.py
├── tests/
│   └── [test files matching above structure]
├── scripts/
│   ├── cli.py
│   └── build_baseline.py
├── data/
│   ├── baseline/
│   ├── models/
│   └── corpus/
└── docs/
    ├── QUICKSTART.md
    ├── TASKS.md
    ├── SPECS.md
    ├── DEPLOYMENT.md
    └── PHILOSOPHY.md
```

---

## DATA_FLOW

```yaml
pipeline:
  input: "str (raw text)"
  
  stage_1_preprocessor:
    output: "List[Token] + spacy.Doc"
    
  stage_2_clause_identifier:
    input: "List[Token] + spacy.Doc"
    output: "List[ClausePair]"
    
  stage_3_echo_engine:
    input: "List[ClausePair]"
    output: "List[EchoScore]"
    
  stage_4_scoring:
    input: "List[EchoScore]"
    output: "float (document_score)"
    
  stage_5_validator:
    input: "float (document_score)"
    output: "Tuple[float (z_score), float (confidence)]"
    
  final_output: "DocumentAnalysis dataclass"
```

---

## TIER_TRANSITION_CHECKLIST

```yaml
tier_1_to_tier_2:
  required:
    - All 32 tasks complete
    - Unit tests passing with >80% coverage
    - 5+ integration tests passing
    - Baseline corpus processed and validated
    - CLI functional on real documents
    - 2-3 real limitations identified through actual usage
    - False positive/negative rate measured on 50+ documents
    - Performance benchmarked
    - "simple" config profile stable
  
  blockers:
    - Adding features without proven need
    - Unfixed bugs in Tier 1
    - No real-world testing
    - Failing or skipped tests

tier_2_to_tier_3:
  required:
    - Tier 2 deployed for 2+ weeks
    - Performance bottlenecks documented with profiling
    - False positive/negative rate < 5%
    - Specific Tier 3 features identified with ROI
    - "robust" config profile tuned
    - User feedback collected
    - Code review complete
  
  blockers:
    - Adding complexity for novelty
    - No measurement of Tier 2 effectiveness
    - Using Tier 3 to fix Tier 2 bugs
    - No production data justifying advanced algorithms
```

---

## IMPLEMENTATION_RULES

```yaml
during_tier_1:
  do:
    - Implement exactly what Tier 1 specifies
    - Write tests before moving to next task
    - Document assumptions and simplifications
    - Use "simple" config profile
    - Follow task sequence strictly
  
  do_not:
    - Add Tier 2 features even if easy
    - Optimize prematurely
    - Skip tests
    - Jump ahead in task sequence
    - Implement features without specification

when_user_requests_task:
  step_1: Read docs/TASKS.md for task specification
  step_2: Read docs/SPECS.md for tier-specific details
  step_3: Implement Tier 1 version only
  step_4: Create corresponding test file
  step_5: Validate implementation before proceeding

when_user_requests_clarification:
  step_1: Check docs/SPECS.md for detailed specification
  step_2: Check architecture.md for algorithm context
  step_3: Provide clear explanation referencing source document
  step_4: Offer code example if helpful
```

---

## DEPENDENCIES

```yaml
required:
  nlp:
    - spacy>=3.7.0
    - en-core-web-sm
  
  phonetic:
    - pronouncing>=0.2.0
    # OR g2p-en>=2.1.0
  
  similarity:
    - python-Levenshtein>=0.21.0
    - jellyfish>=1.0.0
  
  semantic:
    - gensim>=4.3.0
    - numpy>=1.24.0
    - scipy>=1.11.0
  
  config:
    - pydantic>=2.0.0
  
  testing:
    - pytest>=7.4.0
    - pytest-cov>=4.1.0
    - pytest-mock>=3.11.0
  
  cli:
    - rich>=13.0.0
    - tqdm>=4.66.0

optional_tier_2:
  - sentence-transformers>=2.2.0
```

---

## FIRST_SESSION_INSTRUCTIONS

When user starts first Claude Code session:

1. Read docs/QUICKSTART.md for environment setup
2. Implement Task 1.1 (SpecHO/models.py)
3. Create all 5 dataclasses with type hints and docstrings
4. Use Python 3.11+ features
5. No processing logic - data structures only

Example first prompt to expect:
"Read QUICKSTART.md and help me implement Task 1.1: Create Core Data Models"

---

## REFERENCE_DOCUMENT_USAGE

```yaml
TASKS.md:
  read_when: Starting any task
  contains: Task number, file path, class name, key libraries, specific deliverables
  
SPECS.md:
  read_when: Implementing component features
  contains: Tier 1/2/3 specifications, config examples, API signatures
  
architecture.md:
  read_when: Need algorithm context
  contains: Original Echo Rule design, component purposes, methodology
  
DEPLOYMENT.md:
  read_when: Tier 1 complete
  contains: Web interface, Docker setup, deployment guidelines
```

---

## RESPONSE_PATTERNS

When asked to implement a task:
1. Acknowledge task number
2. Reference relevant documentation
3. State tier level being implemented
4. Create file with complete implementation
5. Suggest next task or test creation

When asked for clarification:
1. Reference specific section in documentation
2. Provide concrete example
3. Explain in context of overall architecture

When user deviates from plan:
1. Acknowledge request
2. Note if it conflicts with tier system
3. Suggest alternative if premature
4. Implement if reasonable

---

END OF MACHINE-READABLE SPECIFICATION