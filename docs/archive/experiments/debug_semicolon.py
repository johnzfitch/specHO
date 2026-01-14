from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.clause_identifier.boundary_detector import ClauseBoundaryDetector

p = LinguisticPreprocessor()
d = ClauseBoundaryDetector()

text = "The cat sat; the dog ran."
tokens, doc = p.process(text)
clauses = d.identify_clauses(doc, tokens)

print(f"Text: {text}")
print(f"\nAll tokens ({len(tokens)}):")
for i, t in enumerate(tokens):
    print(f"  {i}: '{t.text}'")

print(f"\nClauses ({len(clauses)}):")
for i, c in enumerate(clauses):
    print(f"  Clause {i}: start={c.start_idx}, end={c.end_idx}")
    print(f"    Tokens: {[t.text for t in c.tokens]}")

print("\nChecking between clauses:")
if len(clauses) >= 2:
    c0, c1 = clauses[0], clauses[1]
    print(f"  Clause 0 end: {c0.end_idx}")
    print(f"  Clause 1 start: {c1.start_idx}")
    print(f"  Gap: {c1.start_idx - c0.end_idx - 1} tokens")

    if c1.start_idx > c0.end_idx + 1:
        print(f"  Tokens between: {[tokens[i].text for i in range(c0.end_idx + 1, c1.start_idx)]}")
