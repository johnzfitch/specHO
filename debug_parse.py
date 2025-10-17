from specHO.preprocessor.pipeline import LinguisticPreprocessor

p = LinguisticPreprocessor()

text = "The cat sat; the dog ran."
tokens, doc = p.process(text)

print(f"Text: {text}\n")
print("Token analysis:")
for token in doc:
    print(f"  {token.i:2d}. '{token.text:10s}' pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.i} head_text='{token.head.text}'")

print("\nClause anchors (ROOT, conj, advcl, ccomp):")
for token in doc:
    if token.dep_ in {"ROOT", "conj", "advcl", "ccomp"}:
        print(f"  {token.text} ({token.dep_})")
