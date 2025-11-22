# <img src="../icons/wrench.png" width="32" height="32"> Implementation Notes

Extracted learnings, gotchas, and insights from SpecHO development sessions.

---

## <img src="../icons/filter.png" width="24" height="24"> Preprocessor

### <img src="../icons/star.png" width="20" height="20"> Learnings

**Dual API Strategy**: Both `process()` (full analysis) and `tokenize()` (quick tokenization) methods provide flexibility for different use cases. Most downstream components need full analysis, but quick tokenization is useful for validation and debugging.

**Content Word Filtering**: Content words (nouns, verbs, adjectives, adverbs) typically comprise 30-50% of total tokens. The `is_content_word` flag enables efficient filtering without re-parsing.

**spaCy Integration**: Using spaCy's pre-trained `en_core_web_sm` model provides good balance of speed and accuracy. The model handles most edge cases well, but has known limitations with semicolon-separated independent clauses.

### <img src="../icons/warning.png" width="20" height="20"> Gotchas

- **Semicolons**: NOT always clause boundaries - context-dependent. spaCy sometimes treats semicolon-separated clauses as single syntactic units.
- **Unicode**: Normalize BEFORE tokenization. Unicode variants of punctuation (em dash variants: `—`, `–`, `--`) need consistent handling.
- **Empty input**: Returns empty list `[]`, not error. Downstream components must handle empty results gracefully.
- **Phonetic transcription**: Out-of-vocabulary words fall back to grapheme-to-phoneme rules. Names and technical terms may have inaccurate transcriptions.

### <img src="../icons/tick.png" width="20" height="20"> Validation Results

```
Test Coverage: 275/279 passing (98.6%)
Real-World Validation: 9 text types tested
- Academic, creative, technical, conversational
- Legal, medical, news, social, mixed
Accuracy varies 15-20% across text types.
Creative text is hardest (intentional style echoes).
```

---

## <img src="../icons/connect.png" width="24" height="24"> Clause Identifier

### <img src="../icons/star.png" width="20" height="20"> Learnings

**Head-Order Pairing**: Using clause head positions instead of token spans aligns pairing logic with syntactic structure. More robust to spaCy's parse variations than span-based approaches.

**Priority-Based Deduplication**: When multiple rules match same clause pair, strongest signal wins:
- Rule A (punctuation) > Rule B (conjunction) > Rule C (transition)

**Document-Order Normalization**: Always ensure `clause_a` precedes `clause_b` in document position. Tests and downstream components expect this ordering.

### <img src="../icons/warning.png" width="20" height="20"> Gotchas

- **Single-word sentences**: Valid clauses - don't filter them out.
- **Nested quotes**: Require special handling for boundary detection.
- **Module imports**: Use absolute imports (`from specHO.models import ...`) not relative. Python creates distinct class objects for different import paths, causing `isinstance()` failures.
- **spaCy parse variations**: Some sentences don't split at semicolons correctly. Accepted as Tier 1 limitation.

### <img src="../icons/tick.png" width="20" height="20"> Validation Results

```
PairRulesEngine: 36/36 tests passing (100%)
ClauseBoundaryDetector: Integration validated
Head-order approach handles overlapping dependency subtrees.
```

---

## <img src="../icons/sound.png" width="24" height="24"> Echo Engine

### <img src="../icons/star.png" width="20" height="20"> Learnings

**Three Separate Analyzers**: Phonetic, structural, and semantic similarity are fundamentally different measurements. Keeping them separate preserves diagnostic information about which type of echoing is present.

**Levenshtein Distance**: Well-understood algorithm for phonetic similarity. Fast computation for short strings (phonetic transcriptions typically <20 phonemes).

**Zone Size**: Default window of 3 content words balances capturing phrase-level patterns without including irrelevant material.

### <img src="../icons/warning.png" width="20" height="20"> Gotchas

- **Phonetic matching**: Metaphone may fail on non-English names.
- **Threshold tuning**: 0.6 similarity threshold works well for Tier 1; make configurable for tuning.
- **Empty zones**: Handle gracefully - some clause pairs may have insufficient content words.

---

## <img src="../icons/gear.png" width="24" height="24"> Cross-Cutting Concerns

### Configuration Interactions

The three configuration profiles interact with component behavior:

| Profile | sentence_min_length | clause_min_tokens | phonetic_threshold |
|---------|--------------------|--------------------|-------------------|
| simple | 3 | 2 | 0.5 |
| robust | 5 | 3 | 0.6 |
| research | 7 | 4 | 0.7 |

**Note**: `robust` profile's `sentence_min_length=3` interacts with clause extraction - very short sentences may not produce clause pairs.

### Testing Patterns

**Placeholder Pattern**: Create working placeholder implementations that pass basic tests, then replace with full logic. Validates API contracts early.

**Orchestrator Pattern**: Pipeline orchestrators should validate inputs, coordinate components, and aggregate outputs without duplicating component logic.

**Test Relaxation**: Acceptable to relax tests when encountering known limitations of underlying libraries. Document the limitation and accept graceful degradation.

### Performance Observations

```
Preprocessor: ~1000 words/second (spaCy tokenization)
Clause Identifier: ~50ms for 100 clauses
Echo Engine: Semantic analyzer is potential bottleneck with large embeddings
```

---

## <img src="../icons/lightbulb.png" width="24" height="24"> Future Ideas

Extracted from session handoffs and context documents:

- [ ] N-gram patterns across clause boundaries
- [ ] Semantic similarity thresholds for echo detection
- [ ] Ensemble scoring approaches
- [ ] Corpus harvesting from public domain texts (see archive/legacy/CORPUS_HARVESTING_ARCHITECTURE.md)
- [ ] Language detection before phonetic analysis
- [ ] Real-time streaming analysis mode
- [ ] Confidence calibration against human judgment
- [ ] Cross-language watermark detection

---

## <img src="../icons/clock.png" width="24" height="24"> Decision Log

Key implementation decisions and their rationale:

| Decision | Rationale | Revisit When |
|----------|-----------|--------------|
| spaCy for NLP | Balance of speed/accuracy, unified API | POS accuracy <90% |
| Levenshtein for phonetic | Well-understood, no training data needed | False positive rate >20% |
| Word2Vec for semantic | Fast, pre-trained available | Accuracy <75% and semantic is key signal |
| CLI-first interface | Simplest deployment for validation | Tier 1 complete |
| Z-score validation | No labeled data required | >500 labeled documents available |

---

*Last Updated: 2025-10-25*
*Source: Extracted from archived session documents*
