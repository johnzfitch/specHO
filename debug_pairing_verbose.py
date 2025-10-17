import logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')

from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector
from specHO.clause_identifier.pair_rules import PairRulesEngine

p = LinguisticPreprocessor()
d = ClauseBoundaryDetector()
e = PairRulesEngine()

# Test Rule A (semicolon)
text = "The cat sat; the dog ran."
print(f"\n{'='*60}")
print(f"Processing: {text}")
print(f"{'='*60}\n")

tokens, doc = p.process(text)
clauses = d.identify_clauses(doc, tokens)

print(f"\nFinal clauses: {len(clauses)}")
for i, clause in enumerate(clauses):
    clause_text = ' '.join(t.text for t in clause.tokens)
    print(f"  Clause {i} ({clause.clause_type}): [{clause.start_idx}, {clause.end_idx}] = '{clause_text}'")
