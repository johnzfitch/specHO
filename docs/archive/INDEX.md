# <img src="../../icons/folder.png" width="32" height="32"> Documentation Archive Index

This archive contains historical documentation from the SpecHO project consolidation (2025-10-25).

## Archive Structure

```
archive/
├── sessions/      # Development session narratives
├── context/       # AI context and handoff documents
└── legacy/        # Pre-consolidation documentation
```

## Why Archived

These documents were archived during documentation consolidation to:
- Reduce active documentation from 49 files to 6
- Preserve historical record for future reference
- Extract unique insights into consolidated active docs

**Nothing was deleted** - all content is searchable here.

---

## Legacy Reference Map

When old documents referenced these files, use this mapping:

| Old Reference | New Location |
|---------------|--------------|
| `session1.md` | archive/sessions/session1.md OR [IMPLEMENTATION.md#foundation] |
| `session2.md` | archive/sessions/session2.md OR [IMPLEMENTATION.md#preprocessor] |
| `session3.md` | archive/sessions/session3.md OR [IMPLEMENTATION.md#clause-identifier] |
| `insights.md` | archive/sessions/insights.md OR [IMPLEMENTATION.md] |
| `CONTEXT_SESSION*.md` | archive/context/ (historical context snapshots) |
| `HANDOFF_*.md` | archive/context/ (session handoffs) |
| `summary*.md` | archive/legacy/ OR [STATUS.md] |
| `DOCUMENTATION_MAP.md` | archive/legacy/ OR README.md |

---

## Archive Contents

### sessions/ (Development Narratives)

| File | Content | Key Insights Extracted |
|------|---------|------------------------|
| session1.md | Foundation tasks 1.1, 1.2, 7.3 | Dataclass design, config profiles |
| session2.md | Preprocessor tasks 2.1-2.5 | Semicolon handling, content word ratios |
| session3.md | Clause identifier task 3.2 | Head-order pairing, module imports |
| session4.md | Task 4.0+ development | Component progress |
| session5_task4.1_phonetic_analyzer.md | Phonetic analyzer notes | Algorithm selection |
| insights.md | Cross-session implementation notes | Patterns, gotchas |

### context/ (AI Context Documents)

| File | Purpose |
|------|---------|
| CONTEXT_SESSION5-9.md | Session-specific context snapshots |
| CONTEXT_COMPRESSED.md | Compressed context for /clear recovery |
| HANDOFF_SESSION8.md | Detailed handoff documentation |
| HANDOFF_AGENT3.md | Agent-specific handoff |
| REINIT_PROMPT.md | Re-initialization template |

### legacy/ (Pre-Consolidation Docs)

| File | Original Purpose | Disposition |
|------|------------------|-------------|
| agent-training*.md | AI collaboration patterns | Insights in IMPLEMENTATION.md |
| summary*.md | Progress summaries | Consolidated to STATUS.md |
| DOCUMENTATION_MAP.md | Navigation guide | Simplified to README.md |
| CORPUS_HARVESTING_ARCHITECTURE.md | Future corpus strategy | Preserved for later |
| zone_extractor_validation.md | Validation results | In IMPLEMENTATION.md |

---

## How to Use This Archive

**Finding historical context:**
```bash
grep -r "keyword" docs/archive/
```

**Reading specific session:**
```bash
cat docs/archive/sessions/session2.md
```

**Tracing a decision:**
1. Check IMPLEMENTATION.md for extracted insight
2. If need more detail, find original session doc
3. Search archive for related context

---

## Archive Policy

- **Read-only**: Do not modify archived files
- **Searchable**: Use grep/search freely
- **Reference**: Link to archive when citing historical context
- **No new files**: Archive is closed; new work goes in active docs

---

*Archived: 2025-10-25*
*Reason: Documentation consolidation (49 → 6 active docs)*
