import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')

from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector

p = LinguisticPreprocessor()
d = ClauseBoundaryDetector()

text = "The cat sat; the dog ran."
tokens, doc = p.process(text)

print(f"\nText: {text}")
print("="*60)
clauses = d.identify_clauses(doc, tokens)
print("="*60)

print(f"\nFinal result: {len(clauses)} clauses")
for i, c in enumerate(clauses):
    clause_text = ' '.join(t.text for t in c.tokens)
    print(f"  Clause {i} ({c.clause_type}): [{c.start_idx}, {c.end_idx}] = '{clause_text}'")
