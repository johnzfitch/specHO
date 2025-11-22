# Session 8 Entry Point: Statistical Validator (Tasks 6.2-6.4)

**Status**: Ready to implement
**Progress**: 19/32 tasks (59.4%)
**Component**: Statistical Validator (Tasks 6.2, 6.3, 6.4)
**Prerequisites**: ✅ Complete (Task 6.1 BaselineCorpusProcessor implemented)

---

## IMMEDIATE CONTEXT (30 SECONDS)

**YOU ARE HERE**: Beginning Session 8, implementing the statistical validation pipeline

**WHAT WAS DONE** (Session 7):
- ✅ Task 5.3: ScoringModule verified (all tests passing)
- ✅ Task 6.1: BaselineCorpusProcessor implemented
  - Process corpus through complete pipeline
  - Calculate mean and std deviation
  - Save/load baseline statistics
  - 22 comprehensive tests (all passing)

**YOUR MISSION** (Session 8):
Implement the remaining 3 components of the Statistical Validator:
1. **Task 6.2**: ZScoreCalculator - Convert document scores to z-scores
2. **Task 6.3**: ConfidenceConverter - Convert z-scores to confidence levels
3. **Task 6.4**: StatisticalValidator - Orchestrate the validation pipeline

---

## CONTEXT RESTORATION PROTOCOL

### Level 1: Essential (Read First - 5 min)

1. **This file** (HANDOFF_SESSION8.md) - Current task context
2. **docs/TASKS.md** (lines 540-580) - Tasks 6.2, 6.3, 6.4 specifications
3. **docs/SPECS.md** (lines 445-480) - Statistical Validator specifications
4. **docs/REFERENCE_DISTRIBUTION_GUIDE.md** - Theory and best practices

### Level 2: Implementation Context (Read Second - 10 min)

5. **specHO/validator/baseline_builder.py** - Task 6.1 implementation (reference pattern)
6. **specHO/scoring/pipeline.py** - ScoringModule orchestrator pattern
7. **docs/CONTEXT_SESSION7.md** - Recent session context

### Level 3: Testing & Validation (Reference as Needed)

8. **tests/test_baseline_builder.py** - Task 6.1 test patterns
9. **tests/test_scoring_pipeline.py** - Orchestrator test patterns
10. **docs/agent-training3.md** - Lessons learned from bugs

---

## TASK SPECIFICATIONS

### TASK 6.2: ZScoreCalculator

```yaml
id: 6.2
file: specHO/validator/z_score.py
class: ZScoreCalculator
tier: 1
libraries: []  # Pure Python, no dependencies
dependencies: None (stateless math operations)

api:
  - calculate_z_score(document_score: float, human_mean: float,
                      human_std: float) -> float

algorithm:
  formula: z = (document_score - human_mean) / human_std

tier_1_features:
  - Simple z-score calculation
  - Input validation (non-zero std)
  - Returns float (can be negative, zero, or positive)

notes:
  - Stateless (no initialization required)
  - Can be static method or instance method
  - Tier 1: No caching, no error recovery
```

**Implementation Example**:
```python
class ZScoreCalculator:
    """Calculate z-scores for statistical validation."""

    def calculate_z_score(self, document_score: float,
                         human_mean: float, human_std: float) -> float:
        """Calculate z-score: (x - μ) / σ

        Args:
            document_score: Score from ScoringModule
            human_mean: Baseline mean from BaselineCorpusProcessor
            human_std: Baseline std from BaselineCorpusProcessor

        Returns:
            Z-score (can be negative, zero, or positive)
            - Negative: Below human average
            - Zero: Exactly at human average
            - Positive: Above human average

        Raises:
            ValueError: If human_std is zero or negative
        """
        if human_std <= 0:
            raise ValueError(f"Standard deviation must be positive, got {human_std}")

        z_score = (document_score - human_mean) / human_std
        return z_score
```

**Test Categories** (aim for ~15 tests):
- Basic calculation (positive, negative, zero z-scores)
- Edge cases (score = mean, very high/low scores)
- Input validation (zero std, negative std)
- Numerical precision (float accuracy)

---

### TASK 6.3: ConfidenceConverter

```yaml
id: 6.3
file: specHO/validator/confidence.py
class: ConfidenceConverter
tier: 1
libraries: [scipy.stats]
dependencies: Task 6.2 (uses z-scores)

api:
  - convert_to_confidence(z_score: float) -> float

algorithm:
  formula: confidence = Φ(z)  # CDF of standard normal
  implementation: scipy.stats.norm.cdf(z_score)

tier_1_features:
  - Convert z-score to confidence level using normal CDF
  - Returns probability in [0, 1] range
  - No caching (recalculate each time)

notes:
  - Confidence represents P(score ≤ document_score | human distribution)
  - High confidence (>0.95) suggests watermark
  - Low confidence (<0.05) suggests human text
  - Medium confidence (0.05-0.95) is uncertain
```

**Implementation Example**:
```python
from scipy.stats import norm

class ConfidenceConverter:
    """Convert z-scores to confidence levels."""

    def convert_to_confidence(self, z_score: float) -> float:
        """Convert z-score to confidence using normal CDF.

        Args:
            z_score: Z-score from ZScoreCalculator

        Returns:
            Confidence level in [0, 1]:
            - 0.0: Extremely unlikely (far below human mean)
            - 0.5: Exactly at human mean
            - 1.0: Extremely likely (far above human mean)

        Interpretation:
            confidence > 0.95: Likely watermarked (95th percentile+)
            0.05 < confidence < 0.95: Uncertain region
            confidence < 0.05: Likely human (below 5th percentile)
        """
        confidence = norm.cdf(z_score)
        return float(confidence)
```

**Test Categories** (aim for ~15 tests):
- Basic conversion (z=0, z=1, z=2, z=3)
- Symmetry (z vs -z should sum to 1.0)
- Extreme values (z=10, z=-10)
- Known values (z=1.96 → 0.975 for 95% confidence)
- Output range validation (always in [0, 1])

---

### TASK 6.4: StatisticalValidator

```yaml
id: 6.4
file: specHO/validator/pipeline.py
class: StatisticalValidator
tier: 1
libraries: []
dependencies: [Tasks 6.1, 6.2, 6.3]

api:
  - validate(document_score: float) -> Tuple[float, float]
    returns: (z_score, confidence)

tier_1_features:
  - Load baseline statistics from pickle file
  - Orchestrate ZScoreCalculator and ConfidenceConverter
  - Return both z-score and confidence
  - Simple error handling (missing baseline file)

notes:
  - Orchestrator pattern (like ScoringModule)
  - Stateful (loads baseline at initialization)
  - Tier 1: Single baseline file, no caching
```

**Implementation Example**:
```python
class StatisticalValidator:
    """Orchestrate statistical validation pipeline."""

    def __init__(self, baseline_path: str = "data/baseline/human_stats.pkl"):
        """Initialize validator with baseline statistics.

        Args:
            baseline_path: Path to baseline pickle file

        Raises:
            FileNotFoundError: If baseline file doesn't exist
        """
        # Load baseline statistics
        from .baseline_builder import BaselineCorpusProcessor
        processor = BaselineCorpusProcessor()
        self.baseline_stats = processor.load_baseline(baseline_path)

        # Initialize components
        self.z_score_calculator = ZScoreCalculator()
        self.confidence_converter = ConfidenceConverter()

    def validate(self, document_score: float) -> Tuple[float, float]:
        """Validate document score against baseline distribution.

        Args:
            document_score: Score from ScoringModule (in [0,1] range)

        Returns:
            Tuple of (z_score, confidence):
            - z_score: Standard deviations from human mean
            - confidence: Probability score came from non-human distribution
        """
        # Step 1: Calculate z-score
        z_score = self.z_score_calculator.calculate_z_score(
            document_score,
            self.baseline_stats['human_mean'],
            self.baseline_stats['human_std']
        )

        # Step 2: Convert to confidence
        confidence = self.confidence_converter.convert_to_confidence(z_score)

        return z_score, confidence
```

**Test Categories** (aim for ~20 tests):
- Initialization (successful load, missing file)
- Basic validation (known baseline + score → expected z-score and confidence)
- Edge cases (score = mean, very high/low scores)
- Orchestration verification (calls both components)
- Integration scenarios (realistic baseline + document scores)

---

## IMPLEMENTATION CHECKLIST

### Phase 1: ZScoreCalculator (20 min)

- [ ] Create `specHO/validator/z_score.py`
  - [ ] `ZScoreCalculator` class
  - [ ] `calculate_z_score()` method
  - [ ] Input validation (positive std)
  - [ ] Comprehensive docstrings with examples

- [ ] Create `tests/test_z_score.py`
  - [ ] Basic calculation tests (5)
  - [ ] Edge case tests (5)
  - [ ] Input validation tests (5)
  - [ ] Run: `pytest tests/test_z_score.py -v`

- [ ] Update `specHO/validator/__init__.py`
  - [ ] Export `ZScoreCalculator`

### Phase 2: ConfidenceConverter (20 min)

- [ ] Create `specHO/validator/confidence.py`
  - [ ] `ConfidenceConverter` class
  - [ ] `convert_to_confidence()` method
  - [ ] Uses `scipy.stats.norm.cdf()`
  - [ ] Comprehensive docstrings with examples

- [ ] Create `tests/test_confidence.py`
  - [ ] Basic conversion tests (5)
  - [ ] Symmetry tests (3)
  - [ ] Extreme value tests (3)
  - [ ] Known value tests (4)
  - [ ] Run: `pytest tests/test_confidence.py -v`

- [ ] Update `specHO/validator/__init__.py`
  - [ ] Export `ConfidenceConverter`

### Phase 3: StatisticalValidator (30 min)

- [ ] Create `specHO/validator/pipeline.py`
  - [ ] `StatisticalValidator` class
  - [ ] `__init__()` with baseline loading
  - [ ] `validate()` method returning (z_score, confidence)
  - [ ] Orchestrates ZScoreCalculator + ConfidenceConverter
  - [ ] Comprehensive docstrings with examples

- [ ] Create `tests/test_statistical_validator.py`
  - [ ] Initialization tests (3)
  - [ ] Basic validation tests (7)
  - [ ] Orchestration tests (5)
  - [ ] Integration tests (5)
  - [ ] Run: `pytest tests/test_statistical_validator.py -v`

- [ ] Update `specHO/validator/__init__.py`
  - [ ] Export `StatisticalValidator`

### Phase 4: Integration Testing (20 min)

- [ ] Create sample baseline file
  - [ ] `data/baseline/test_baseline.pkl`
  - [ ] Known statistics for testing

- [ ] Create demo script
  - [ ] `scripts/demo_validator.py`
  - [ ] Show complete validation workflow
  - [ ] Test with known scores

- [ ] Run full test suite
  - [ ] `pytest tests/ -x -q`
  - [ ] Verify no regressions
  - [ ] Should have ~650+ tests total

### Phase 5: Documentation (10 min)

- [ ] Update session docs
  - [ ] Create `docs/CONTEXT_SESSION8.md`
  - [ ] Document what was implemented
  - [ ] Record any issues encountered

- [ ] Mark tasks complete
  - [ ] Update progress tracking
  - [ ] Verify all tests passing

---

## CRITICAL PATTERNS FROM PREVIOUS SESSIONS

### ✅ DO (Validated Patterns)

**1. Orchestrator Pattern** (from ScoringModule):
```python
class StatisticalValidator:
    def __init__(self):
        # Create component instances
        self.z_score_calculator = ZScoreCalculator()
        self.confidence_converter = ConfidenceConverter()

    def validate(self, document_score: float):
        # Step 1: Component A
        z_score = self.z_score_calculator.calculate_z_score(...)

        # Step 2: Component B
        confidence = self.confidence_converter.convert_to_confidence(z_score)

        # Step 3: Return combined result
        return z_score, confidence
```

**2. Comprehensive Testing** (from BaselineCorpusProcessor):
- Test initialization
- Test basic functionality
- Test edge cases
- Test error handling
- Test integration scenarios

**3. Docstring Examples** (from all components):
```python
def calculate_z_score(self, score, mean, std):
    """Calculate z-score.

    Examples:
        >>> calc = ZScoreCalculator()
        >>> z = calc.calculate_z_score(0.45, 0.15, 0.10)
        >>> print(f"{z:.2f}")
        3.00
    """
```

### ❌ DON'T (Anti-Patterns)

**1. Don't Skip Input Validation**:
```python
# Bad: No validation
def calculate_z_score(self, score, mean, std):
    return (score - mean) / std  # Crashes if std=0!

# Good: Validate inputs
def calculate_z_score(self, score, mean, std):
    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    return (score - mean) / std
```

**2. Don't Ignore Edge Cases**:
```python
# Test extreme values, not just typical cases
def test_extreme_z_score():
    calc = ZScoreCalculator()
    # What happens with very large scores?
    z = calc.calculate_z_score(10.0, 0.15, 0.10)
    assert z > 0  # Should still work
```

**3. Don't Forget Type Hints**:
```python
# Good: Clear type hints
def validate(self, document_score: float) -> Tuple[float, float]:
    ...
```

---

## EXPECTED OUTCOMES

### Files to Create (7 files)

**Implementation**:
1. `specHO/validator/z_score.py` (~100 LOC)
2. `specHO/validator/confidence.py` (~90 LOC)
3. `specHO/validator/pipeline.py` (~150 LOC)

**Tests**:
4. `tests/test_z_score.py` (~200 LOC, 15 tests)
5. `tests/test_confidence.py` (~200 LOC, 15 tests)
6. `tests/test_statistical_validator.py` (~300 LOC, 20 tests)

**Demo**:
7. `scripts/demo_validator.py` (~100 LOC)

### Test Count Progression

- Session 7 end: 625 tests
- Session 8 target: **675+ tests** (+50 new tests)
- Pass rate target: >99.5%

### Time Estimates

- ZScoreCalculator: 20 min (simple math)
- ConfidenceConverter: 20 min (scipy wrapper)
- StatisticalValidator: 30 min (orchestrator)
- Integration testing: 20 min
- Documentation: 10 min
- **Total**: ~100 minutes (~1.5 hours)

---

## VALIDATION CRITERIA

### Functional Requirements

✅ **ZScoreCalculator**:
- Correctly calculates z-score: (x - μ) / σ
- Validates input (positive std)
- Returns float (can be negative, zero, or positive)

✅ **ConfidenceConverter**:
- Correctly uses scipy.stats.norm.cdf()
- Returns confidence in [0, 1] range
- Handles extreme z-scores gracefully

✅ **StatisticalValidator**:
- Loads baseline statistics from pickle
- Orchestrates both components correctly
- Returns (z_score, confidence) tuple
- Handles missing baseline file with clear error

### Quality Gates

✅ **Unit Tests**:
- ZScoreCalculator: 15 tests, 100% coverage
- ConfidenceConverter: 15 tests, 100% coverage
- StatisticalValidator: 20 tests, >90% coverage

✅ **Integration**:
- Demo script runs successfully
- Real baseline file works correctly
- Full test suite passes (no regressions)

✅ **Documentation**:
- Comprehensive docstrings with examples
- Session context documented
- Progress tracking updated

---

## QUICK REFERENCE

### Command Palette

```bash
# Create validator directory (if needed)
mkdir -p specHO/validator

# Run specific component tests
pytest tests/test_z_score.py -v
pytest tests/test_confidence.py -v
pytest tests/test_statistical_validator.py -v

# Run all validator tests
pytest tests/test_*score*.py tests/test_*confidence*.py tests/test_*validator*.py -v

# Run full test suite
pytest tests/ -x -q

# Create demo baseline for testing
python scripts/demo_validator.py
```

### File Locations

**Implementation**:
- `specHO/validator/z_score.py`
- `specHO/validator/confidence.py`
- `specHO/validator/pipeline.py`
- `specHO/validator/__init__.py` (update exports)

**Tests**:
- `tests/test_z_score.py`
- `tests/test_confidence.py`
- `tests/test_statistical_validator.py`

**Documentation**:
- `docs/TASKS.md` (lines 540-580)
- `docs/SPECS.md` (lines 445-480)
- `docs/REFERENCE_DISTRIBUTION_GUIDE.md`

### Key Formulas

**Z-Score**:
```
z = (x - μ) / σ

Where:
  x = document_score (from ScoringModule)
  μ = human_mean (from baseline)
  σ = human_std (from baseline)
```

**Confidence**:
```
confidence = Φ(z) = ∫[-∞, z] φ(t) dt

Where:
  Φ = CDF of standard normal distribution
  φ = PDF of standard normal distribution

Implementation:
  from scipy.stats import norm
  confidence = norm.cdf(z_score)
```

**Interpretation**:
```
z < -2:  Very likely human (confidence < 0.025)
-2 ≤ z ≤ 2:  Uncertain (0.025 ≤ confidence ≤ 0.975)
z > 2:  Likely watermarked (confidence > 0.975)
z > 3:  Very likely watermarked (confidence > 0.999)
```

---

## SUCCESS CRITERIA

### Definition of Done

- [ ] All 3 components implemented (ZScoreCalculator, ConfidenceConverter, StatisticalValidator)
- [ ] All 50 tests passing (15 + 15 + 20)
- [ ] Full test suite passes with no regressions (675+ tests)
- [ ] Demo script demonstrates complete validation workflow
- [ ] Documentation updated (CONTEXT_SESSION8.md created)
- [ ] Progress tracking shows 22/32 tasks complete (68.8%)

### Quality Metrics

- **Test Coverage**: >95% on new components
- **Code Quality**: All docstrings complete, type hints present
- **Integration**: Works with existing pipeline (preprocessor → ... → validator)
- **Performance**: Fast (<1ms per validation)

---

## EMERGENCY CONTACTS

### If Tests Fail

1. **Check API compatibility**: Ensure z_score calculation matches specification
2. **Verify scipy version**: `pip show scipy` (should be >=1.11.0)
3. **Check baseline file**: Ensure test baseline has correct structure
4. **Review Session 6 bug**: Check docs/agent-training3.md for alignment patterns

### If Integration Fails

1. **Check baseline file format**: Should be pickle with {'human_mean', 'human_std', 'n_documents'}
2. **Verify component chaining**: StatisticalValidator → ZScoreCalculator → ConfidenceConverter
3. **Check return types**: validate() should return Tuple[float, float]

### If Context Insufficient

- **Full project context**: docs/CLAUDE.md
- **Architecture overview**: docs/architecture.md
- **Task specifications**: docs/TASKS.md
- **Component specs**: docs/SPECS.md
- **Previous sessions**: docs/CONTEXT_SESSION6.md, docs/CONTEXT_SESSION7.md

---

## NEXT AGENT PROMPT

```
You are beginning Session 8 of the SpecHO watermark detection project.

CONTEXT:
- Session 7 completed Task 6.1 (BaselineCorpusProcessor)
- 625 tests currently passing (99.8% pass rate)
- You are implementing Tasks 6.2, 6.3, 6.4 (Statistical Validator components)

YOUR TASKS:
1. Implement ZScoreCalculator (Task 6.2)
2. Implement ConfidenceConverter (Task 6.3)
3. Implement StatisticalValidator orchestrator (Task 6.4)
4. Create comprehensive tests (~50 new tests)
5. Validate with demo script

READ FIRST:
1. This file (docs/HANDOFF_SESSION8.md)
2. docs/TASKS.md (lines 540-580) for task specifications
3. docs/REFERENCE_DISTRIBUTION_GUIDE.md for statistical theory

PATTERN TO FOLLOW:
- Use orchestrator pattern (see ScoringModule in specHO/scoring/pipeline.py)
- Create comprehensive tests (see tests/test_baseline_builder.py)
- Validate inputs, handle edge cases
- Document with examples in docstrings

START WITH:
Read the handoff document thoroughly, then implement Task 6.2 (ZScoreCalculator).
This is the simplest component and will establish the pattern for the others.

EXPECTED TIME: ~1.5 hours for all 3 tasks + tests + validation

Good luck! The validator is the final major component before integration.
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-19 (Session 7 end)
**Status**: Ready for Session 8 implementation
**Progress**: 19/32 tasks (59.4%) → Target: 22/32 tasks (68.8%)
