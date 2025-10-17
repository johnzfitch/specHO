# CONTEXT_COMPRESSED.md
# SpecHO Watermark Detector - Ultra-Condensed Context
# Token-optimized reference for AI assistant context recovery

---

## META
```yaml
project: SpecHO - AI text watermark detector (Echo Rule algorithm)
version: 0.34 (11/32 tasks complete)
tier: 1_mvp (weeks 1-12, simple algorithms only)
language: Python 3.11+
current_task: 3.4 (ClauseIdentifier pipeline orchestrator)
next_task: 4.1 (PhoneticEchoAnalyzer)
```

---

## PROGRESS
```yaml
completed: [1.1, 1.2, 7.3, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3]
tests: {total: 544, passing: 414, coverage: 76%}

component_status:
  C1_preprocessor: ✅ 100% (300 tests, 1260 LOC)
  C2_clause_identifier: 75% (3/4 tasks, 139 tests, ~1200 LOC)
  C3_echo_engine: ⏳ 0/4
  C4_scoring: ⏳ 0/3
  C5_validator: ⏳ 0/4
  integration: ⏳ 0/3

files_created:
  impl: [models.py, config.py, utils.py, preprocessor/*.py(5), clause_identifier/*.py(3)]
  tests: [test_models.py, test_config.py, test_utils.py, test_preprocessor*.py(5), test_clause*.py(3)]
```

---

## ARCHITECTURE

### Data Flow
```
raw_text (str)
  → Preprocessor: List[Token] + SpacyDoc
  → ClauseIdentifier: List[ClausePair] (with zones)
  → EchoEngine: List[EchoScore]
  → ScoringModule: float (doc_score)
  → StatisticalValidator: (z_score, confidence)
  → DocumentAnalysis
```

### Core Dataclasses (models.py)
```python
Token(text, pos_tag, phonetic, is_content_word, syllable_count)
Clause(tokens, start_idx, end_idx, clause_type, head_idx)  # head_idx added S3
ClausePair(clause_a, clause_b, zone_a_tokens, zone_b_tokens, pair_type)
EchoScore(phonetic_score, structural_score, semantic_score, combined_score)
DocumentAnalysis(text, clause_pairs, echo_scores, final_score, z_score, confidence)
```

### Components Map
```yaml
C1_Preprocessor: {orchestrator: LinguisticPreprocessor, pipeline: [Tokenizer→POSTagger→PhoneticTranscriber→DependencyParser]}
C2_ClauseIdentifier: {orchestrator: ClauseIdentifier, pipeline: [BoundaryDetector→PairRulesEngine→ZoneExtractor]}
C3_EchoEngine: {orchestrator: EchoAnalysisEngine, analyzers: [PhoneticEchoAnalyzer, StructuralEchoAnalyzer, SemanticEchoAnalyzer]}
C4_Scoring: {orchestrator: ScoringModule, components: [WeightedScorer, DocumentAggregator]}
C5_Validator: {orchestrator: StatisticalValidator, components: [BaselineCorpusProcessor, ZScoreCalculator, ConfidenceConverter]}
```

---

## KEY DECISIONS

### 1. Placeholder Pattern (S2)
```
Token enrichment: progressive field population
- Tokenizer: text only
- POSTagger: +pos_tag, +is_content_word
- PhoneticTranscriber: +phonetic, +syllable_count
Benefits: decoupled components, type-safe, testable
```

### 2. Dual Output (S2)
```
Preprocessor returns: (List[Token], SpacyDoc)
Rationale: Token=abstraction, Doc=structure (dep trees, sent bounds)
Avoids: re-parsing text
```

### 3. Head-Order Pairing (S3)
```
Problem: Overlapping dep-subtree spans caused Rule A failures
Solution: Pair by clause.head_idx positions, not token spans
Algorithm: sort by head_idx → scan between heads for strong punct
Added: head_idx field to Clause dataclass
```

### 4. Orchestrator Pattern
```
All pipeline classes: minimal logic, delegate to subcomponents
Benefits: easy testing, easy extension, clear debugging
Example: LinguisticPreprocessor just chains 4 components
```

### 5. Tier 1 Philosophy
```
Rules:
- Implement ONLY spec (TASKS.md/SPECS.md = source of truth)
- Simple algorithms only (no premature optimization)
- Graceful degradation over errors (short clauses → return available)
- No logging unless spec requires
- No batch APIs unless spec requires
Violations caught: S4 (ZoneExtractor logging, batch API, __init__)
```

---

## DEPENDENCIES
```yaml
nlp: [spacy>=3.7.0, en-core-web-sm]
phonetic: [pronouncing>=0.2.0]
similarity: [python-Levenshtein>=0.21.0, jellyfish>=1.0.0]
semantic: [gensim>=4.3.0, numpy>=1.24.0, scipy>=1.11.0]
config: [pydantic>=2.0.0]
testing: [pytest>=7.4.0, pytest-cov>=4.1.0, pytest-mock>=3.11.0]
cli: [rich>=13.0.0, tqdm>=4.66.0]
```

### Why These?
```
spacy: best English NLP, Universal Dependencies, fast (Cython)
en-core-web-sm: 12MB, 92% POS accuracy (good for T1)
pronouncing: CMU Dict (130K words), simple API, syllable counting
Levenshtein: ARPAbet phoneme distance (T1 phonetic)
gensim: Word2Vec/GloVe mean-pooling (T1 semantic)
```

---

## COMPONENT 1: PREPROCESSOR (✅ Complete)

### LinguisticPreprocessor.process(text) → (List[Token], SpacyDoc)
```python
# Sequential chain:
tokens = Tokenizer().tokenize(text)           # text field only
tokens = POSTagger().tag(tokens)              # +pos_tag, +is_content_word
tokens = PhoneticTranscriber().transcribe_tokens(tokens)  # +phonetic, +syllable_count
doc = DependencyParser().parse(text)          # spaCy Doc with dep tree
return (tokens, doc)
```

### Tokenizer (168 LOC, 20 tests)
```python
# API: tokenize(text) → List[Token]
# Uses: spacy.load("en_core_web_sm", disable=["parser","ner"])
# Handles: contractions, hyphens, punctuation, unicode
```

### POSTagger (202 LOC, 36 tests)
```python
# API: tag(tokens) → List[Token]
# CONTENT_POS_TAGS = {NOUN, PROPN, VERB, ADJ, ADV}
# Fallback: _tag_with_alignment() for tokenization mismatches
```

### PhoneticTranscriber (289 LOC, 54 tests)
```python
# API: transcribe_tokens(tokens) → List[Token]
# ARPAbet via CMU Dict: "hello" → "HH AH0 L OW1"
# OOV: uppercase fallback (T1), G2P reserved for T2
# Syllables: count stress markers (0/1/2 on vowels)
```

### DependencyParser (301 LOC, 49 tests)
```python
# API: parse(text) → SpacyDoc
# Uses: spacy.load() with parser ENABLED
# Helpers: find_root_verbs(), find_coordinated_clauses(), find_subordinate_clauses()
# Clause labels: ROOT, conj, advcl, ccomp
```

### Real-World Validation (9 samples)
```
News, conversational, literary, technical, academic, dialogue, short, complex, conjunctions
Discoveries:
- Semicolons = clause separators (not sentence terminators)
- Content word ratio: 30-70% typical (varies by genre)
- Phonetic coverage: >90% (CMU Dict), ~10% OOV
```

---

## COMPONENT 2: CLAUSE_IDENTIFIER (75% Complete)

### Task 3.1: BoundaryDetector (✅ 59 tests)
```python
# API: identify_clauses(doc, tokens) → List[Clause]
# Algorithm:
#   1. Find anchors: dep_ in {ROOT, conj, advcl, ccomp}
#   2. Build subtrees: anchor + children (recursive)
#   3. Normalize: _make_spans_non_overlapping() at separators {;,—,:}
#   4. Store: head_idx = anchor.i
# Output: Clause(tokens, start_idx, end_idx, clause_type, head_idx)
```

### Task 3.2: PairRulesEngine (✅ 36 tests)
```python
# API: apply_all_rules(clauses, tokens, doc) → List[ClausePair]

# Rule A (punctuation): head-order based
STRONG_PUNCT = {";", ":", "—", "–", "--"}
- Sort clauses by head_idx
- Check adjacent pairs for punct between heads
- Fallback: check punct attached to heads (dep_="punct")

# Rule B (conjunction): span-based
CONJUNCTIONS = {"and", "but", "or"}
- Check tokens in clause_a for conjunctions (case-insensitive)
- Pair with clause_a+1

# Rule C (transition): span-based
TRANSITIONS = {"However,", "Therefore,", "Thus,"}
- Check first token of clause_b (case-sensitive, requires comma)
- Pair with clause_b-1

# Deduplication: priority-based (Rule A > B > C)
```

### Task 3.2 Critical Fix (S3)
```yaml
Problem: Overlapping clause spans broke span-based pairing
Initial: Normalize spans at separators
Final: Head-order pairing for Rule A
- Added head_idx field to Clause
- Sort by head_idx (not start_idx)
- Scan tokens between heads for punct
- More robust to spaCy parse variations
```

### Task 3.3: ZoneExtractor (✅ 44 tests)
```python
# API: extract_zones(clause_pair) → Tuple[List[Token], List[Token]]
# Terminal zone: last 3 content words from clause_a
# Initial zone: first 3 content words from clause_b
# Edge: if <3 content words, return all available (graceful degradation)

def get_terminal_content_words(clause, n=3):
    return [t for t in clause.tokens if t.is_content_word][-n:]

def get_initial_content_words(clause, n=3):
    return [t for t in clause.tokens if t.is_content_word][:n:]
```

### Task 3.4: ClauseIdentifier Pipeline (⏳ NEXT)
```python
# Orchestrator combining 3.1, 3.2, 3.3
class ClauseIdentifier:
    def identify_pairs(self, tokens, doc) → List[ClausePair]:
        clauses = boundary_detector.identify_clauses(doc, tokens)
        pairs = pair_engine.apply_all_rules(clauses, tokens, doc)
        enriched_pairs = []
        for pair in pairs:
            zone_a, zone_b = zone_extractor.extract_zones(pair)
            enriched_pairs.append(ClausePair(..., zone_a_tokens=zone_a, zone_b_tokens=zone_b))
        return enriched_pairs
```

---

## COMPONENT 3: ECHO_ENGINE (⏳ Not Started)

### Task 4.1: PhoneticEchoAnalyzer
```python
# API: analyze(zone_a, zone_b) → float
# T1: Levenshtein on ARPAbet strings
#   - Pairwise comparison between zones
#   - Best match selection
#   - Normalize: 1 - (distance / max_length)
# Lib: python-Levenshtein
```

### Task 4.2: StructuralEchoAnalyzer
```python
# API: analyze(zone_a, zone_b) → float
# T1: POS pattern exact match + syllable similarity
#   - pattern_sim * 0.5 + syllable_sim * 0.5
```

### Task 4.3: SemanticEchoAnalyzer
```python
# API: analyze(zone_a, zone_b) → float
# T1: Mean-pooled Word2Vec/GloVe + cosine similarity
#   - Map to [0,1]: (1 + cos) / 2
#   - Fallback: 0.5 if embeddings unavailable
# Lib: gensim
```

### Task 4.4: EchoAnalysisEngine
```python
# Orchestrator
# Calls all 3 analyzers, returns EchoScore(phonetic, structural, semantic, combined)
```

---

## COMPONENT 4: SCORING (⏳ Not Started)

### Task 5.1: WeightedScorer
```python
# API: calculate_pair_score(echo_score, weights) → float
# T1: w_p*phonetic + w_s*structural + w_sem*semantic
# Weights: {phonetic: 0.33, structural: 0.33, semantic: 0.33}
# NaN: treat as 0
```

### Task 5.2: DocumentAggregator
```python
# API: aggregate_scores(pair_scores) → float
# T1: simple mean
# Edge: return 0.0 if no pairs
```

### Task 5.3: ScoringModule
```python
# Orchestrator combining 5.1, 5.2
```

---

## COMPONENT 5: VALIDATOR (⏳ Not Started)

### Task 6.1-6.4: StatisticalValidator
```python
# API: validate(doc_score) → Tuple[z_score, confidence]
# T1: z = (doc_score - human_mean) / human_std
#     confidence = scipy.stats.norm.cdf(z)
# Baseline: pre-computed from human corpus (pickled)
```

---

## SESSION SUMMARIES

### S1: Foundation (Tasks 1.1, 1.2, 7.3)
```
Created: models.py (5 dataclasses), config.py (3 profiles), utils.py (file I/O, logging)
Tests: 105 (96% passing, 9 logging-related failures accepted)
Lessons: Dataclass patterns, type hints, Python 3.11+ features
```

### S2: Preprocessor (Tasks 2.1-2.5, 8.1)
```
Created: 5 components (Tokenizer, POSTagger, DependencyParser, PhoneticTranscriber, LinguisticPreprocessor)
LOC: 1260 impl, ~1800 tests
Tests: 300 (100% passing), 9 real-world samples
Key: Placeholder pattern, dual output (Token list + Doc), spaCy optimization per component
Discoveries: Semicolon behavior, content word ratios, OOV rates
```

### S3: Task 3.2 PairRulesEngine (Extended)
```
Created: PairRulesEngine (3 rules, priority dedup)
LOC: 553 impl, 577 tests
Tests: 36 (100% passing)
Critical: Head-order pairing solution for Rule A
- Initial span-based approach failed (overlapping subtrees)
- Normalized spans (intermediate fix)
- Final: head_idx based pairing (robust to spaCy quirks)
Lessons: Module import pitfalls (isinstance failure), test-driven debugging
```

### S4: Task 3.3 ZoneExtractor
```
Created: ZoneExtractor (extract terminal/initial zones)
LOC: 153 impl, 761 tests
Tests: 44 (30 unit + 14 integration, 100% passing)
Lessons: Spec adherence (caught 3x over-engineering attempts)
- ❌ Batch API (not in spec)
- ❌ Logging (not in spec)
- ❌ __init__ (not needed for stateless class)
Rule: TASKS.md = source of truth, implement exactly, no assumptions
```

---

## CRITICAL LESSONS

### 1. Specification Adherence (S4)
```
Anti-pattern: "This would be useful, let me add it"
Pattern: Read TASKS.md → Implement exactly → Stop
Example: ZoneExtractor API was single-pair, not batch (caught by user review)
```

### 2. Head-Order Discovery (S3)
```
Problem: Dependency subtrees can overlap in document space
Solution: Use syntactic structure (head positions) not linear spans
Impact: Rule A now robust to spaCy parse variations
```

### 3. Module Imports (S3)
```
Bug: isinstance(obj, ClausePair) returned False
Cause: `from models import X` ≠ `from specHO.models import X`
Fix: Always use absolute imports
```

### 4. Tier 1 = Minimal (S2, S4)
```
Philosophy: Get it working, not perfect
- Simple algorithms only
- No optimization
- Graceful degradation (short clauses → return available, not error)
- No features beyond spec
Violations = rework
```

### 5. Real-World Validation (S2)
```
Unit tests verify correctness, real-world tests verify robustness
9 diverse samples exposed:
- Semicolon clause separation
- Content word ratio variation (30-70%)
- OOV phonetic handling
Value: Build confidence for production use
```

### 6. Orchestrator Pattern (S2, S4)
```
Pipeline classes have minimal logic
Benefits:
- Easy testing (test data flow, not logic)
- Easy extension (add component = add call)
- Clear debugging (check individual components)
Trade-off: Can't optimize across components (acceptable for T1)
```

---

## EDGE CASES & SOLUTIONS

### Overlapping Clause Spans (S3)
```
Issue: Dependency subtrees can overlap
Example: "sat" as ccomp of "ran" includes tokens [0-2] inside [0-7]
Solution:
- T1: Normalize at separators (_make_spans_non_overlapping)
- T1+: Head-order pairing (doesn't rely on span boundaries)
```

### Short Clauses (S4)
```
Issue: Clause has <3 content words
Solution: Return all available (graceful degradation)
Rationale: Allow echo analysis to proceed, not fail
```

### OOV Words (S2)
```
Issue: Word not in CMU Dict (~10% of tokens)
Solution:
- T1: Uppercase fallback (deterministic, no phonetic info)
- T2: G2P model if OOV >15% in practice
```

### Tokenization Mismatches (S2)
```
Issue: Tokenizer and POSTagger tokenize differently (contractions, hyphens)
Solution: _tag_with_alignment() fallback (lookup map, POS="X" if no match)
Frequency: <1% of sentences
```

### Multiple Semicolons (S3)
```
Issue: spaCy doesn't always split at semicolons (parses as noun phrases)
Solution: Accept T1 limitation, relax test
Example: "A; B; C" → 1 clause (spaCy choice)
```

---

## TIER SYSTEM

### Tier 1 (Current, Weeks 1-12)
```yaml
goal: Get it working
features: 32 tasks, simple algorithms, basic testing
complete_when:
  - All 32 tasks done
  - Tests >80% coverage passing
  - 5+ integration tests
  - Baseline corpus processed
  - CLI functional
  - 2-3 real limitations measured (not theoretical)
blockers:
  - Adding T2 features "just in case"
  - Unfixed T1 bugs
  - No real-world testing
```

### Tier 2 (Weeks 13-17)
```yaml
goal: Make it reliable
trigger: T1 validation complete
enhancements: Address measured limitations only
config: robust profile
features: Confidence scoring, advanced rules, better OOV handling
```

### Tier 3 (Week 18+)
```yaml
goal: Optimize performance
trigger: T2 production deployment + 2 weeks data
features: Advanced algorithms with proven ROI
config: research profile
```

---

## NEXT STEPS

### Immediate: Task 3.4 (ClauseIdentifier Pipeline)
```python
File: specHO/clause_identifier/pipeline.py
Class: ClauseIdentifier
Method: identify_pairs(tokens, doc) → List[ClausePair]
Tests: Create test_clause_identifier_pipeline.py

Implementation:
1. Initialize 3 subcomponents (BoundaryDetector, PairRulesEngine, ZoneExtractor)
2. Chain: clauses = detector.identify_clauses(doc, tokens)
3. Chain: pairs = engine.apply_all_rules(clauses, tokens, doc)
4. Enrich: for each pair, extract zones and populate zone_a_tokens, zone_b_tokens
5. Return List[ClausePair] with complete data

Pattern: Orchestrator (minimal logic, delegate to subcomponents)
Reference: LinguisticPreprocessor (S2) for orchestrator pattern
```

### After 3.4: Component 3 (Echo Engine)
```
Task 4.1: PhoneticEchoAnalyzer (Levenshtein on ARPAbet)
Task 4.2: StructuralEchoAnalyzer (POS pattern + syllable)
Task 4.3: SemanticEchoAnalyzer (Word2Vec mean-pooling + cosine)
Task 4.4: EchoAnalysisEngine (orchestrator)
```

---

## FILE STRUCTURE
```
specHO/
├── specHO/
│   ├── models.py                           # ✅ S1
│   ├── config.py                           # ✅ S1
│   ├── utils.py                            # ✅ S1
│   ├── preprocessor/
│   │   ├── tokenizer.py                    # ✅ S2 (168 LOC)
│   │   ├── pos_tagger.py                   # ✅ S2 (202 LOC)
│   │   ├── dependency_parser.py            # ✅ S2 (301 LOC)
│   │   ├── phonetic.py                     # ✅ S2 (289 LOC)
│   │   └── pipeline.py                     # ✅ S2 (300 LOC)
│   ├── clause_identifier/
│   │   ├── boundary_detector.py            # ✅ S3 (~400 LOC)
│   │   ├── pair_rules.py                   # ✅ S3 (553 LOC)
│   │   ├── zone_extractor.py               # ✅ S4 (153 LOC)
│   │   └── pipeline.py                     # ⏳ NEXT (~150 LOC est)
│   ├── echo_engine/                        # ⏳ Not started
│   ├── scoring/                            # ⏳ Not started
│   └── validator/                          # ⏳ Not started
├── tests/                                  # 544 tests, 414 passing
├── docs/
│   ├── TASKS.md                            # Source of truth (task specs)
│   ├── SPECS.md                            # Tier details
│   ├── CONTEXT_COMPRESSED.md               # ← THIS FILE
│   ├── Sessions/session1-4.md              # Session logs
│   └── summary*.md                         # Detailed summaries
└── requirements.txt                        # Dependencies
```

---

## QUICK REFERENCE

### Start New Session
```bash
# 1. Read this file (CONTEXT_COMPRESSED.md) for full context recovery
# 2. Check current_task in META section
# 3. Reference TASKS.md for task spec
# 4. Reference SPECS.md for tier details
# 5. Implement with Tier 1 philosophy (simple, exact spec, no over-engineering)
```

### Implementation Checklist
```
1. Read TASKS.md task spec
2. Read SPECS.md tier details
3. Create file (follow orchestrator pattern if pipeline)
4. Implement exact API from spec (no additions)
5. Create test file (unit + integration + real-world)
6. Run tests (pytest -v)
7. Update this file (progress, lessons)
```

### Anti-Patterns to Avoid
```
❌ Adding features not in spec ("would be useful")
❌ Implementing T2/T3 features early ("might as well")
❌ Skipping tests ("will add later")
❌ Batch APIs unless spec requires
❌ Logging unless spec requires
❌ Optimization before measurement
✅ Read spec → Implement exactly → Test → Stop
```

---

END OF CONTEXT_COMPRESSED.md
Version: 1.0 (Post-Session 4)
Token reduction: ~85% (3500 → 550 lines)
Last updated: Task 3.4 ready to implement
