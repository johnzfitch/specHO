# Tier 2 Enhancement Ideas

**Created**: 2025-10-24
**Status**: Planning Phase (Tier 1 must be validated first)
**Purpose**: Track proposed enhancements for Tier 2 implementation

---

## ⚠️ Pre-Tier 2 Requirements

Before implementing ANY Tier 2 features, the following MUST be complete:

- [ ] All 32 Tier 1 tasks complete and tested
- [ ] 830 tests passing at >95%
- [ ] End-to-end validation on 50+ real documents
- [ ] False positive/negative rates measured
- [ ] 2-3 specific limitations identified through actual usage
- [ ] Performance benchmarked
- [ ] Baseline corpus processed
- [ ] CLI functional and tested
- [ ] 2+ weeks of Tier 1 validation

**Do not implement Tier 2 features prematurely!**

---

## 🎯 Enhancement #1: Granular Clause Classification

**Proposed By**: User (2025-10-24)
**Category**: Clause Identifier Enhancement
**Complexity**: Medium
**Expected ROI**: High (better pairing accuracy)

### Current State (Tier 1)

The `ClauseBoundaryDetector` identifies 3 broad clause types:
- **Main clauses**: ROOT verbs
- **Coordinate clauses**: conj relations (and/but/or)
- **Subordinate clauses**: advcl, ccomp (all lumped together)

### Proposed Enhancement

Break subordinate clauses into 3 linguistic subtypes:

1. **Relative Clauses** (acl, relcl dependency labels)
   - Modify nouns
   - Example: "the cat **that sat on the mat**"
   - Function: Provides additional information about antecedent

2. **Adverbial Clauses** (advcl, mark dependency labels)
   - Modify verbs or entire clauses
   - Example: "he left **when the bell rang**"
   - Function: Express time, cause, condition, concession

3. **Noun Clauses** (ccomp, xcomp dependency labels)
   - Function as noun phrases
   - Example: "I believe **that he's right**"
   - Function: Subject, object, or complement of sentence

### Linguistic Justification

Different clause types exhibit different echo patterns:

**Pattern 1: Relative Clauses Echo Antecedents**
```
Zone A: "The dog that barked loudly"
        └─ relative clause modifies "dog"

Zone B: "the cat that meowed softly"
        └─ relative clause modifies "cat"

Echo Pattern: Parallel relative clause structures
- Noun + relative pronoun + verb + adverb
- "dog...barked...loudly" ↔ "cat...meowed...softly"
```

**Pattern 2: Adverbial Clauses Create Temporal/Causal Parallels**
```
Zone A: "When night fell, silence reigned"
        └─ temporal adverbial clause

Zone B: "As dawn broke, noise erupted"
        └─ temporal adverbial clause

Echo Pattern: Parallel time-based clause structures
- Temporal marker + event + consequence
```

**Pattern 3: Noun Clauses Carry Semantic Content**
```
Zone A: "I believe that hope endures"
        └─ noun clause as object of "believe"

Zone B: "I think that faith persists"
        └─ noun clause as object of "think"

Echo Pattern: Parallel belief statements with embedded clauses
- Mental verb + that + abstract noun + persists
```

### Implementation Sketch

```python
# specHO/clause_identifier/boundary_detector.py (Tier 2 version)

class ClauseBoundaryDetector:
    """Enhanced clause detection with granular classification (Tier 2)."""

    def identify_clauses(self, doc, tokens, granular=True):
        """
        Identify clauses with optional granular classification.

        Args:
            doc: spaCy Doc object
            tokens: List[Token] from preprocessor
            granular: If True, classify subordinate clauses into subtypes

        Returns:
            List[Clause] with enhanced clause_type field
        """
        clauses = []

        # Main clauses (unchanged from Tier 1)
        for token in doc:
            if token.dep_ == "ROOT":
                clause = self._build_clause_from_anchor(token, tokens, "main")
                clauses.append(clause)

        # Coordinate clauses (unchanged from Tier 1)
        for token in doc:
            if token.dep_ == "conj":
                clause = self._build_clause_from_anchor(token, tokens, "coordinate")
                clauses.append(clause)

        if granular:
            # Relative clauses (NEW for Tier 2)
            for token in doc:
                if token.dep_ in ["acl", "relcl"]:
                    clause = self._build_clause_from_anchor(token, tokens, "relative")
                    clause.antecedent = self._find_antecedent(token)  # Track what it modifies
                    clauses.append(clause)

            # Adverbial clauses (NEW for Tier 2)
            for token in doc:
                if token.dep_ == "advcl":
                    clause = self._build_clause_from_anchor(token, tokens, "adverbial")
                    clause.adverbial_type = self._classify_adverbial(token)  # time/cause/condition/etc
                    clauses.append(clause)

            # Noun clauses (NEW for Tier 2)
            for token in doc:
                if token.dep_ in ["ccomp", "xcomp"]:
                    clause = self._build_clause_from_anchor(token, tokens, "noun")
                    clause.clause_function = self._get_clause_function(token)  # subject/object/complement
                    clauses.append(clause)
        else:
            # Tier 1 fallback: lump all subordinate together
            for token in doc:
                if token.dep_ in ["advcl", "ccomp", "xcomp", "acl", "relcl"]:
                    clause = self._build_clause_from_anchor(token, tokens, "subordinate")
                    clauses.append(clause)

        return sorted(clauses, key=lambda c: c.start_idx)

    def _classify_adverbial(self, token):
        """Classify adverbial clause type (Tier 2 only)."""
        # Check subordinating conjunction
        for child in token.children:
            if child.dep_ == "mark":
                marker = child.text.lower()

                # Temporal
                if marker in ["when", "while", "after", "before", "until", "since"]:
                    return "temporal"

                # Causal
                if marker in ["because", "since", "as"]:
                    return "causal"

                # Conditional
                if marker in ["if", "unless", "provided"]:
                    return "conditional"

                # Concessive
                if marker in ["although", "though", "even though", "while"]:
                    return "concessive"

                # Purpose
                if marker in ["so that", "in order that"]:
                    return "purpose"

        return "other"

    def _find_antecedent(self, token):
        """Find noun that relative clause modifies (Tier 2 only)."""
        # Relative clauses attach to their head noun
        if token.head and token.head.pos_ in ["NOUN", "PROPN", "PRON"]:
            return token.head.text
        return None

    def _get_clause_function(self, token):
        """Determine grammatical function of noun clause (Tier 2 only)."""
        # Check what role the clause plays in parent clause
        parent_dep = token.dep_

        if parent_dep == "ccomp":
            # Usually object of verb
            return "object"
        elif parent_dep == "xcomp":
            # Open clausal complement
            return "complement"
        elif parent_dep == "nsubj":
            # Nominal subject (rare)
            return "subject"

        return "other"
```

### Enhanced Pairing Rules (Tier 2)

```python
# specHO/clause_identifier/pair_rules.py (Tier 2 additions)

class PairRulesEngine:
    """Enhanced pairing with clause-type awareness (Tier 2)."""

    def apply_all_rules(self, clauses, tokens, doc):
        """Apply all pairing rules with clause-type filtering."""
        pairs = []

        # Existing Tier 1 rules
        pairs.extend(self._apply_rule_a_punctuation(clauses, tokens, doc))
        pairs.extend(self._apply_rule_b_conjunction(clauses, tokens, doc))
        pairs.extend(self._apply_rule_c_transition(clauses, tokens, doc))

        # NEW Tier 2 rule: Parallel relative clauses
        pairs.extend(self._apply_rule_d_parallel_relatives(clauses))

        # NEW Tier 2 rule: Matching adverbial types
        pairs.extend(self._apply_rule_e_adverbial_match(clauses))

        return self._deduplicate_pairs(pairs)

    def _apply_rule_d_parallel_relatives(self, clauses):
        """
        Rule D (Tier 2): Pair relative clauses with same antecedent type.

        Logic: If two relative clauses modify the same type of noun
        (e.g., both modify animate nouns), pair them.
        """
        pairs = []
        relative_clauses = [c for c in clauses if c.clause_type == "relative"]

        for i, clause_a in enumerate(relative_clauses):
            for clause_b in relative_clauses[i+1:]:
                # Check if antecedents are semantically related
                if self._are_related_nouns(clause_a.antecedent, clause_b.antecedent):
                    pair = ClausePair(
                        clause_a=clause_a,
                        clause_b=clause_b,
                        pair_type="parallel_relative",
                        zone_a_tokens=[],  # Will be filled by ZoneExtractor
                        zone_b_tokens=[]
                    )
                    pairs.append(pair)

        return pairs

    def _apply_rule_e_adverbial_match(self, clauses):
        """
        Rule E (Tier 2): Pair adverbial clauses of same type.

        Logic: Temporal pairs with temporal, causal with causal, etc.
        """
        pairs = []
        adverbial_clauses = [c for c in clauses if c.clause_type == "adverbial"]

        # Group by adverbial type
        by_type = {}
        for clause in adverbial_clauses:
            adv_type = clause.adverbial_type
            if adv_type not in by_type:
                by_type[adv_type] = []
            by_type[adv_type].append(clause)

        # Pair within each type
        for adv_type, type_clauses in by_type.items():
            for i, clause_a in enumerate(type_clauses):
                for clause_b in type_clauses[i+1:]:
                    pair = ClausePair(
                        clause_a=clause_a,
                        clause_b=clause_b,
                        pair_type=f"adverbial_{adv_type}",
                        zone_a_tokens=[],
                        zone_b_tokens=[]
                    )
                    pairs.append(pair)

        return pairs

    def _are_related_nouns(self, noun1, noun2):
        """Check if two nouns are semantically related (simple Tier 2)."""
        if noun1 is None or noun2 is None:
            return False

        # Simple heuristic: same word or both animate/inanimate
        if noun1.lower() == noun2.lower():
            return True

        # Use WordNet or semantic model for better matching (Tier 3)
        # For Tier 2, use simple category matching
        animate = ["person", "dog", "cat", "man", "woman", "child", "animal"]
        abstract = ["hope", "faith", "belief", "idea", "thought", "dream"]

        n1_lower = noun1.lower()
        n2_lower = noun2.lower()

        # Both animate?
        if any(a in n1_lower for a in animate) and any(a in n2_lower for a in animate):
            return True

        # Both abstract?
        if any(a in n1_lower for a in abstract) and any(a in n2_lower for a in abstract):
            return True

        return False
```

### Testing Strategy

1. **Unit Tests**: Test each clause type identification
   ```python
   def test_identify_relative_clauses():
       text = "The dog that barked sat on the mat that was soft."
       # Should identify 2 relative clauses

   def test_classify_adverbial_temporal():
       text = "When the sun rose, the birds sang."
       # Should classify as "temporal"

   def test_classify_adverbial_causal():
       text = "Because it rained, the game was cancelled."
       # Should classify as "causal"
   ```

2. **Integration Tests**: Test enhanced pairing
   ```python
   def test_pair_parallel_relatives():
       text = "The cat that slept here and the dog that ran there."
       # Should pair the two relative clauses

   def test_pair_matching_adverbials():
       text = "When night fell, darkness came. When dawn broke, light returned."
       # Should pair the two temporal adverbial clauses
   ```

3. **Real-World Validation**: Compare Tier 1 vs Tier 2 accuracy
   - Run 100 watermarked texts through both versions
   - Measure: Does granular classification improve detection?
   - Metric: Precision, recall, F1-score

### Data Model Changes

Add fields to `Clause` dataclass:

```python
# specHO/models.py (Tier 2 additions)

@dataclass
class Clause:
    """Enhanced clause representation (Tier 2)."""
    tokens: List[Token]
    start_idx: int
    end_idx: int
    clause_type: str  # "main", "coordinate", "relative", "adverbial", "noun"
    head_idx: int

    # NEW Tier 2 fields (optional, only for certain types)
    antecedent: Optional[str] = None       # For relative clauses
    adverbial_type: Optional[str] = None   # For adverbial clauses
    clause_function: Optional[str] = None  # For noun clauses
```

### Config Changes

Add Tier 2 config option:

```python
# specHO/config.py (Tier 2 additions)

@dataclass
class ClauseDetectionConfig:
    """Clause detection configuration."""
    # Tier 1 options
    simple_detection: bool = True

    # NEW Tier 2 options
    granular_classification: bool = False  # Enable subordinate clause subtypes
    use_wordnet_similarity: bool = False   # For _are_related_nouns()

# In ROBUST_PROFILE (Tier 2):
ROBUST_PROFILE = SpecHOConfig(
    clause_detection=ClauseDetectionConfig(
        simple_detection=False,
        granular_classification=True,  # ← Enable Tier 2 feature
        use_wordnet_similarity=False    # Still simple in Tier 2
    ),
    # ... rest of config
)
```

### Expected Benefits

1. **Higher Precision**: Fewer false positives by matching clause types
2. **Better Semantic Matching**: Relative clauses pair with relatives, etc.
3. **Richer Features**: More signal for echo detection
4. **Linguistic Accuracy**: Aligns with actual linguistic structure

### Risks & Challenges

1. **Complexity**: More clause types = more edge cases
2. **spaCy Accuracy**: Dependency labels not always perfect
3. **Overfitting**: May reduce recall if too restrictive
4. **Testing Burden**: More test cases needed

### Validation Criteria

Before merging to Tier 2:
- [ ] F1-score improves by ≥5% on validation set
- [ ] No decrease in recall >2%
- [ ] All new tests passing
- [ ] Real-world validation on 50+ documents
- [ ] Performance impact <10% slower

---

## 🎯 Enhancement #2: [Reserved for Next Idea]

Add your next Tier 2 enhancement idea here!

---

## 📋 Enhancement Prioritization

When Tier 1 is complete, evaluate enhancements based on:

1. **Measured Impact**: Does it address a real limitation found in Tier 1 validation?
2. **Implementation Cost**: How complex is it to implement and test?
3. **Risk**: How likely is it to introduce new bugs or reduce performance?
4. **Linguistic Validity**: Is it grounded in established linguistic theory?

**Selection Process**:
1. Run Tier 1 for 2+ weeks on real data
2. Measure false positives/negatives
3. Identify top 3 limitations
4. Select enhancements that address those limitations
5. Implement in order of ROI (impact / cost)

---

**Note**: This document is for planning only. Do NOT implement any Tier 2 features until Tier 1 is fully validated!
