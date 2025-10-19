# Agent3 Session Checklist

Quick reference for tracking Session 6 Part 2 progress.

---

## PRE-FLIGHT CHECK

- [ ] Read `HANDOFF_AGENT3.md` (mission briefing)
- [ ] Read `SESSION6_PART1.md` (what Agent2 did)
- [ ] Read `CONTEXT_SESSION5.md` (previous session)
- [ ] Run `git status && git branch` (verify main branch, clean state)
- [ ] Understand the bug: AI text scored 0.0381 (should be 0.25-0.50)

---

## PHASE 1: DIAGNOSTICS (30 min)

- [ ] Create `scripts/diagnose_preprocessing.py`
- [ ] Run diagnostic on sample.txt
- [ ] Confirm content word rate: 4.9% (should be 30-70%)
- [ ] Confirm field population: 9.1% (should be >80%)
- [ ] Identify root cause (POSTagger alignment? Other?)
- [ ] Document findings in diagnostic output

**Expected Discovery**: POSTagger creates misaligned spaCy doc

---

## PHASE 2: FIXES (2-3 hours)

### Fix 1: POSTagger Alignment
- [ ] File: `specHO/preprocessor/pos_tagger.py`
- [ ] Problem understood: Creates new spaCy doc, misaligns with Token list
- [ ] Solution implemented: Pass existing doc OR improve alignment
- [ ] Quick test: Check content word rate on sample.txt (should be 30-70%)

### Fix 2: Field Population Validation
- [ ] File: `specHO/preprocessor/pipeline.py`
- [ ] Add assertion after POSTagger: `all(t.pos_tag != "" for t in tokens)`
- [ ] Add assertion after PhoneticTranscriber: `all(t.phonetic != "" for t in tokens)`
- [ ] Quick test: Run pipeline, verify no assertion errors

### Fix 3: Semantic Analyzer Robustness
- [ ] File: `specHO/echo_engine/semantic_analyzer.py`
- [ ] Add logging when model is None
- [ ] Add logging when vectors are None
- [ ] Validate model loaded on initialization
- [ ] Quick test: Check logs, verify model loaded and vectors generated

### Fix 4: Zone Size Validation
- [ ] File: `specHO/clause_identifier/zone_extractor.py`
- [ ] Add warning when zone < 2 tokens
- [ ] Add debug logging for zone sizes
- [ ] Quick test: Check average zone sizes (should be 3-5 tokens)

---

## PHASE 3: INTEGRATION TESTS (1 hour)

- [ ] Create `tests/test_integration_real_data.py`
- [ ] Test 1: Preprocessing on sample.txt (content word rate 30-70%)
- [ ] Test 2: Field population on sample.txt (>80%)
- [ ] Test 3: Full pipeline on sample.txt (score 0.25-0.50)
- [ ] Test 4: Realistic score distributions (not just perfect mock data)
- [ ] Run: `python -m pytest tests/test_integration_real_data.py -v`
- [ ] All integration tests passing: YES / NO

---

## PHASE 4: VALIDATION (30 min)

- [ ] Run existing test suite: `python -m pytest tests/ -q`
- [ ] All existing tests still pass: YES / NO
- [ ] Run full pipeline: `python -m scripts.demo_full_pipeline`
- [ ] Sample.txt score: _______ (should be 0.25-0.50)
- [ ] Sample.txt classification: _______ (should be UNWATERMARKED_AI)
- [ ] Content word rate: _______ (should be 30-70%)
- [ ] Field population: _______ (should be >80%)
- [ ] Average semantic score: _______ (should be 0.40-0.60)

**SUCCESS CRITERIA**:
- ✅ Score: 0.25-0.50
- ✅ Classification: UNWATERMARKED_AI
- ✅ Content words: 30-70%
- ✅ Field population: >80%

---

## PHASE 5: DOCUMENTATION (30 min)

### Create `docs/agent-training3.md`
- [ ] Document the bug (false negative on AI text)
- [ ] Document root cause (POSTagger alignment, etc.)
- [ ] Document lessons learned (L1-L5)
- [ ] Document anti-patterns identified
- [ ] Document fixes applied

### Create `docs/CONTEXT_SESSION6.md`
- [ ] Session metadata (session 6, progress, status)
- [ ] Critical bug summary
- [ ] Fixes implemented
- [ ] Validation results
- [ ] Next steps

---

## OPTIONAL: TASK 5.3 (if time permits)

- [ ] Read `TASKS.md` Task 5.3 spec (ScoringModule)
- [ ] Create `specHO/scoring/pipeline.py`
- [ ] Create `tests/test_scoring_pipeline.py`
- [ ] Update progress to 18/32 tasks

---

## FINAL CHECKS

- [ ] All tests passing: `python -m pytest tests/ -v`
- [ ] Integration tests passing: `python -m pytest tests/test_integration_real_data.py -v`
- [ ] Full pipeline test passing: `python -m scripts.demo_full_pipeline`
- [ ] Sample.txt scores correctly (0.25-0.50, UNWATERMARKED_AI)
- [ ] Documentation complete (`agent-training3.md`, `CONTEXT_SESSION6.md`)
- [ ] Git status clean (or ready to commit)

---

## TIME TRACKING

| Phase | Estimated | Actual | Notes |
|-------|-----------|--------|-------|
| Diagnostics | 30 min | | |
| Fixes | 2-3 hours | | |
| Integration Tests | 1 hour | | |
| Validation | 30 min | | |
| Documentation | 30 min | | |
| **TOTAL** | **4-5 hours** | | |

---

## DELIVERABLES CHECKLIST

- [ ] ✅ Diagnostic script (`scripts/diagnose_preprocessing.py`)
- [ ] ✅ Fixed preprocessing (correct content word identification)
- [ ] ✅ Field population validation (assertions added)
- [ ] ✅ Semantic analyzer improvements (logging, validation)
- [ ] ✅ Zone size validation (warnings added)
- [ ] ✅ Integration tests (`tests/test_integration_real_data.py`)
- [ ] ✅ Validation report (sample.txt scores correctly)
- [ ] ✅ Documentation (`docs/agent-training3.md`)
- [ ] ✅ Session context (`docs/CONTEXT_SESSION6.md`)
- [ ] ⭐ (Optional) Task 5.3 ScoringModule

---

## QUICK COMMANDS

```bash
# Check status
git status && git branch

# Run diagnostics (after creating)
python -m scripts.diagnose_preprocessing

# Run full pipeline
python -m scripts.demo_full_pipeline

# Run all tests
python -m pytest tests/ -v

# Run integration tests
python -m pytest tests/test_integration_real_data.py -v

# Quick test count
python -m pytest tests/ -q
```

---

## EMERGENCY CONTACTS

**If stuck**:
- Read `HANDOFF_AGENT3.md` section "Code Locations to Investigate"
- Check `agent-training2.md` for Session 5 lessons
- Review `SPECS.md` for tier specifications
- Look at existing test files for patterns

**If unsure about fix**:
- Run diagnostic first
- Validate hypothesis before fixing
- Test incrementally (one fix at a time)
- Document what you tried

---

**Good luck! The project depends on you getting preprocessing right. 🎯**
