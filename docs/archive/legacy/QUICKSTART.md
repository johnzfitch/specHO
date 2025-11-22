# QUICKSTART

**Target:** Claude Code AI  
**Purpose:** Initial project setup and Task 1.1 implementation  
**Duration:** 5 minutes

---

## SETUP_SEQUENCE

```bash
# Step 1: Create project structure
mkdir -p SpecHO/SpecHO SpecHO/tests SpecHO/scripts SpecHO/data/{baseline,models,corpus} SpecHO/docs
cd SpecHO

# Step 2: Initialize git
git init

# Step 3: Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.Python
venv/
env/
*.egg-info/
.pytest_cache/
.coverage
data/baseline/*.pkl
data/models/*.bin
data/corpus/
.DS_Store
EOF

# Step 4: Create requirements.txt
cat > requirements.txt << 'EOF'
spacy>=3.7.0
pronouncing>=0.2.0
python-Levenshtein>=0.21.0
jellyfish>=1.0.0
gensim>=4.3.0
numpy>=1.24.0
scipy>=1.11.0
pydantic>=2.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-mock>=3.11.0
rich>=13.0.0
tqdm>=4.66.0
EOF

# Step 5: Create Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Step 6: Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Step 7: Start Claude Code
claude code
```

---

## TASK_1.1_SPECIFICATION

**File:** `SpecHO/models.py`  
**Tier:** 1 (MVP)  
**Dependencies:** None (foundation task)

### DELIVERABLES

Create 5 dataclasses with complete type hints:

```python
@dataclass
class Token:
    """Single token with linguistic annotations."""
    text: str
    pos_tag: str
    phonetic: str
    is_content_word: bool
    syllable_count: int

@dataclass  
class Clause:
    """Clause boundary with tokens."""
    tokens: List[Token]
    start_idx: int
    end_idx: int
    clause_type: str

@dataclass
class ClausePair:
    """Pair of clauses to analyze for echoes."""
    clause_a: Clause
    clause_b: Clause
    zone_a_tokens: List[Token]
    zone_b_tokens: List[Token]
    pair_type: str

@dataclass
class EchoScore:
    """Scores from three analyzers for a clause pair."""
    phonetic_score: float  # 0.0-1.0
    structural_score: float  # 0.0-1.0
    semantic_score: float  # 0.0-1.0
    combined_score: float  # 0.0-1.0

@dataclass
class DocumentAnalysis:
    """Complete analysis results for a document."""
    text: str
    clause_pairs: List[ClausePair]
    echo_scores: List[EchoScore]
    final_score: float
    z_score: float
    confidence: float
```

### REQUIREMENTS

- Python 3.11+ type hints (use `from typing import List`)
- Docstrings for each class
- No methods or logic (data structures only)
- All fields must have type annotations

### EXPECTED_FILE_STRUCTURE

```python
"""Core data models for SpecHO watermark detection pipeline."""

from dataclasses import dataclass
from typing import List

@dataclass
class Token:
    """..."""
    # fields

@dataclass
class Clause:
    """..."""
    # fields

@dataclass
class ClausePair:
    """..."""
    # fields

@dataclass
class EchoScore:
    """..."""
    # fields

@dataclass
class DocumentAnalysis:
    """..."""
    # fields
```

### VALIDATION

After implementation, verify:
- [ ] File created at `SpecHO/models.py`
- [ ] All 5 dataclasses defined
- [ ] All fields have type hints
- [ ] All classes have docstrings
- [ ] File imports successfully: `python -c "from SpecHO.models import Token"`

---

## NEXT_TASK

After Task 1.1 complete:
- Proceed to Task 1.2 (SpecHO/config.py)
- Reference docs/TASKS.md for specification
- Create three config profiles: simple, robust, research

---

## TYPICAL_FIRST_PROMPT

```
I'm starting the SpecHO project. I've completed the setup steps in QUICKSTART.md.

Please implement Task 1.1: Create Core Data Models (SpecHO/models.py)

Requirements:
- 5 dataclasses: Token, Clause, ClausePair, EchoScore, DocumentAnalysis
- Full type hints
- Docstrings for each class
- Python 3.11+ syntax
- No processing logic, data structures only

Create the complete file.
```

---

END OF QUICKSTART