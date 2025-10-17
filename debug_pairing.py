from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector
from specHO.clause_identifier.pair_rules import PairRulesEngine

p = LinguisticPreprocessor()
d = ClauseBoundaryDetector()
e = PairRulesEngine()

# Test Rule A (semicolon)
text = "The cat sat; the dog ran."
tokens, doc = p.process(text)
clauses = d.identify_clauses(doc, tokens)

print(f"Text: {text}")
print(f"Clauses detected: {len(clauses)}\n")

for i, clause in enumerate(clauses):
    clause_text = ' '.join(t.text for t in clause.tokens)
    print(f"Clause {i}: {clause_text}")
    print(f"  Last token: '{clause.tokens[-1].text}'" if clause.tokens else "  (empty)")

print()
pairs = e.apply_rule_a(clauses)
print(f"Rule A pairs: {len(pairs)}")

# Test Rule B (conjunction)
text2 = "The cat sat, and the dog ran."
tokens2, doc2 = p.process(text2)
clauses2 = d.identify_clauses(doc2, tokens2)

print(f"\nText: {text2}")
print(f"Clauses detected: {len(clauses2)}\n")

for i, clause in enumerate(clauses2):
    clause_text = ' '.join(t.text for t in clause.tokens)
    print(f"Clause {i}: {clause_text}")

pairs2 = e.apply_rule_b(clauses2)
print(f"\nRule B pairs: {len(pairs2)}")
