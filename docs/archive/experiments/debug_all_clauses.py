from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector

p = LinguisticPreprocessor()
d = ClauseBoundaryDetector()

text = "The cat sat; the dog ran."
tokens, doc = p.process(text)

print(f"Text: {text}\n")
print("Dependency structure:")
for token in doc:
    print(f"  {token.i}: '{token.text}' dep={token.dep_} head={token.head.i}")

print("\nClause anchors found:")
for token in doc:
    if token.dep_ in {"ROOT", "conj", "advcl", "ccomp"}:
        print(f"  {token.i}: '{token.text}' ({token.dep_})")

clauses = d.identify_clauses(doc, tokens)

print(f"\nFinal clauses ({len(clauses)}):")
for i, c in enumerate(clauses):
    clause_text = ' '.join(t.text for t in c.tokens)
    print(f"  Clause {i} ({c.clause_type}): [{c.start_idx}, {c.end_idx}] = '{clause_text}'")
