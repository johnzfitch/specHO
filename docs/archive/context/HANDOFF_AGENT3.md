# Agent3 Handoff: Session 6 Bug Hunt & Continuation

**Context**: Agent2 discovered critical preprocessing bug during full pipeline test. Agent3 must diagnose, fix, and continue project.

---

## CRITICAL BUG DISCOVERED

### Symptom
- **File**: `specHO/sample.txt` (AI-generated essay, 4,153 words)
- **Actual Score**: 0.0381
- **Actual Classification**: HUMAN/NATURAL
- **Expected Score**: 0.25-0.50 (unwatermarked AI)
- **Expected Classification**: UNWATERMARKED_AI

**Impact**: System cannot distinguish AI from human text → complete detection failure

### Evidence from Pipeline Run
```
Content words: 238 / 4,899 tokens = 4.9% (EXPECTED: 30-70%)
Average phonetic: 0.033 (EXPECTED: 0.25-0.40 for unwatermarked AI)
Average structural: 0.036 (EXPECTED: 0.25-0.40 for unwatermarked AI)
Average semantic: 0.045 (EXPECTED: 0.40-0.60 for coherent AI text)

Warnings:
- "Low field population rate: 9.1%"
- "Unusual content word rate: 4.9%"
- "Token count mismatch: input=4899, spaCy=4905"
```

### Root Cause Hypothesis
**POSTagger alignment failure** → Token list not properly enriched → Low content word count → Small zones → Low similarity scores → False negative

---

## CURRENT PROJECT STATUS

### Completed (17/32 tasks - 53%)
- ✅ **Foundation** (3/3): models.py, config.py, utils.py
- ✅ **Preprocessor** (6/6): Tasks 2.1-2.5, tests
- ✅ **Clause Identifier** (5/5): Tasks 3.1-3.4, tests
- ✅ **Echo Engine** (4/4): Tasks 4.1-4.4, tests
- ✅ **Scoring** (2/4): Tasks 5.1 (WeightedScorer), 5.2 (DocumentAggregator)

### In Progress
- 🔄 **Scoring** (Task 5.3 pending): ScoringModule orchestrator

### Not Started
- ⏳ **Validator** (5 tasks): 6.1-6.4, tests
- ⏳ **Integration** (4 tasks): 7.1-7.2, 7.4, tests

### Test Coverage
- **385 tests total** (100% passing before bug discovery)
- **Issue**: All tests use mock data with perfect enrichment
- **Missing**: Real-data integration tests

---

## APPROVED AGENT3 PLAN

### Phase 1: Diagnostics (30 min)
**Create**: `scripts/diagnose_preprocessing.py`

**Must Report**:
1. **Tokenizer**: Token count, samples
2. **POSTagger**: POS distribution, content word %, Token/Doc alignment
3. **PhoneticTranscriber**: Phonetic coverage, syllable counts
4. **Zone Extraction**: Average zone lengths, content word counts
5. **Semantic Analyzer**: Model loaded?, embeddings generated?

**Key Diagnostic Questions**:
- Why is content word rate 4.9% instead of 30-70%?
- Why are only 9.1% of fields populated?
- Is POSTagger creating misaligned spaCy doc?
- Are zones too small (1-2 tokens)?

### Phase 2: Fixes (2-3 hours)

#### Fix 1: POSTagger Alignment
**File**: `specHO/preprocessor/pos_tagger.py`
**Issue**: Creates new spaCy doc that doesn't align with Token list
**Solution**: Pass existing dependency doc from pipeline OR improve alignment logic

#### Fix 2: Field Population Validation
**File**: `specHO/preprocessor/pipeline.py`
**Issue**: Tokens not fully enriched (9.1% population)
**Solution**: Add assertions after each enrichment step

#### Fix 3: Semantic Analyzer Robustness
**File**: `specHO/echo_engine/semantic_analyzer.py`
**Issue**: Returns 0.5 default when vectors are None (masks bugs)
**Solution**: Add logging, validate model on init, check minimum zone size

#### Fix 4: Zone Size Validation
**File**: `specHO/clause_identifier/zone_extractor.py`
**Issue**: 1-token zones produce unreliable scores
**Solution**: Add warnings for small zones, validate minimum size

### Phase 3: Test Improvements (1 hour)
**Create**: `tests/test_integration_real_data.py`

**Must Include**:
- Test preprocessing on sample.txt (validate content word rate 30-70%)
- Test full pipeline on sample.txt (validate score 0.25-0.50 for AI)
- Test with realistic score distributions (not just mock perfect data)

### Phase 4: Validation (30 min)
1. Re-run `scripts/demo_full_pipeline.py` on sample.txt
2. Verify score: 0.25-0.50 (unwatermarked AI)
3. Verify classification: UNWATERMARKED_AI (not HUMAN)
4. Run all existing tests (should still pass)
5. Run new integration tests

### Phase 5: Documentation (30 min)
**Create**:
- `docs/agent-training3.md` - Session 6 lessons learned
- `docs/CONTEXT_SESSION6.md` - Ultra-compressed session context

---

## CODE LOCATIONS TO INVESTIGATE

### Primary Suspects

1. **POSTagger.tag()** - `specHO/preprocessor/pos_tagger.py:141-147`
   ```python
   # Creates new spaCy doc - may misalign with Token list
   doc = self.nlp(text)
   for i, spacy_token in enumerate(doc):
       # Assumes 1:1 alignment - may be wrong
   ```

2. **LinguisticPreprocessor.process()** - `specHO/preprocessor/pipeline.py:166-192`
   ```python
   # Sequential enrichment - validate each step
   tokens = self.tokenizer.tokenize(text)           # Step 1
   tagged_tokens = self.pos_tagger.tag(tokens)      # Step 2 - SUSPECT
   enriched_tokens = self.phonetic_transcriber.transcribe_tokens(tagged_tokens)  # Step 3
   dependency_doc = self.dependency_parser.parse(text)  # Step 4
   ```

3. **ZoneExtractor** - `specHO/clause_identifier/zone_extractor.py:104-105`
   ```python
   # Filters content words - if POSTagger failed, this fails too
   content_words = [t for t in clause.tokens if t.is_content_word]
   return content_words[-n:]  # May return 0-1 tokens
   ```

4. **SemanticEchoAnalyzer._get_zone_vector()** - `specHO/echo_engine/semantic_analyzer.py:128-159`
   ```python
   # Returns None if no embeddings → defaults to 0.5
   # Should log warnings when this happens
   ```

### Validation Code

5. **LinguisticPreprocessor._validate_output()** - `specHO/preprocessor/pipeline.py:194-227`
   ```python
   # Already has warnings - check logs!
   if content_rate < 0.2 or content_rate > 0.8:
       logging.warning(f"Unusual content word rate: {content_rate:.1%}")
   ```

---

## SUCCESS CRITERIA

### Must Achieve:
1. ✅ **Content word rate**: 30-70% on sample.txt
2. ✅ **Field population**: >80% on sample.txt
3. ✅ **Sample.txt score**: 0.25-0.50 (unwatermarked AI range)
4. ✅ **Sample.txt classification**: UNWATERMARKED_AI (not HUMAN)
5. ✅ **All existing tests**: Still passing
6. ✅ **Integration tests**: Created and passing
7. ✅ **Documentation**: agent-training3.md and CONTEXT_SESSION6.md complete

### Nice to Have:
- Task 5.3 (ScoringModule) implemented if time permits
- Additional test documents (watermarked AI, human text)
- Automated regression test suite

---

## REFERENCE DOCUMENTS

**Read First**:
1. `CONTEXT_SESSION5.md` - Session 5 context (Echo Engine completion)
2. `agent-training2.md` - Session 5 lessons learned
3. This file (HANDOFF_AGENT3.md) - Current mission

**Reference as Needed**:
- `TASKS.md` - Task specifications (lines 433-493 for Scoring tasks)
- `SPECS.md` - Tier specifications (lines 371-441 for Scoring)
- `architecture.md` - Original Echo Rule design
- `agent-training.md` - General compression techniques

**Test Output**:
- `scripts/demo_full_pipeline.py` - Full pipeline test that revealed bug
- Output showed: score 0.0381, classification HUMAN/NATURAL (WRONG)

---

## EXPECTED SCORE RANGES (For Validation)

### Document Types
- **Watermarked AI**: 0.75-0.95 (strong deliberate echo)
- **Unwatermarked AI**: 0.25-0.50 (coherent but no echo)
- **Human Text**: 0.10-0.30 (minimal accidental echo)

### Echo Score Dimensions (Unwatermarked AI)
- **Phonetic**: 0.25-0.40 (low - no deliberate sound patterns)
- **Structural**: 0.25-0.40 (low - no deliberate syntax patterns)
- **Semantic**: 0.40-0.60 (moderate - coherent related content)

### What We Got (WRONG)
- Phonetic: 0.033 (too low)
- Structural: 0.036 (too low)
- Semantic: 0.045 (way too low)
- **Document**: 0.0381 (misclassified as human)

---

## QUICK START COMMANDS

```bash
# 1. Check current status
git status && git branch

# 2. Run diagnostic script (create this first!)
python -m scripts.diagnose_preprocessing

# 3. After fixes - validate
python -m scripts.demo_full_pipeline

# 4. Run tests
python -m pytest tests/ -v

# 5. Run integration tests (create these!)
python -m pytest tests/test_integration_real_data.py -v
```

---

## ANTI-PATTERNS TO AVOID

❌ **Don't skip diagnostics** - Understand root cause before fixing
❌ **Don't assume tests validate everything** - Mock data hides real issues
❌ **Don't use default fallbacks silently** - Log when using defaults
❌ **Don't ignore warnings** - Pipeline already warned about 4.9% content rate
❌ **Don't commit without validation** - Re-run full pipeline before declaring victory

---

## TIME BUDGET

**Total: 4-5 hours**
- Diagnostics: 30 min
- Fixes: 2-3 hours
- Tests: 1 hour
- Documentation: 30 min
- Validation: 30 min

**Prioritize**: Fixes > Tests > Task 5.3

---

## SESSION 6 DELIVERABLES

1. ✅ Diagnostic script revealing root cause
2. ✅ Fixed preprocessing (correct content word identification)
3. ✅ Real-data integration tests
4. ✅ sample.txt scoring 0.25-0.50 (AI classification)
5. ✅ Documentation (agent-training3.md, CONTEXT_SESSION6.md)
6. ⭐ (Optional) Task 5.3 implementation

---

**Good luck, Agent3! The project's detection capability depends on you. 🎯**
