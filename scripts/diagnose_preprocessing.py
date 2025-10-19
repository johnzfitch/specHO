#!/usr/bin/env python3
"""
Diagnostic script for investigating preprocessing bugs.

Purpose:
    Investigate why sample.txt (AI-generated text) scores 0.0381 instead of 0.25-0.50.

Expected Issues:
    - Content word rate: 4.9% (should be 30-70%)
    - Field population: 9.1% (should be >80%)
    - POSTagger alignment problems
    - Zone extraction producing tiny zones

Usage:
    python -m scripts.diagnose_preprocessing
"""

import sys
from pathlib import Path
from collections import Counter
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from specHO.preprocessor.pipeline import LinguisticPreprocessor
from specHO.preprocessor.tokenizer import Tokenizer
from specHO.preprocessor.pos_tagger import POSTagger
from specHO.preprocessor.phonetic import PhoneticTranscriber
from specHO.clause_identifier.pipeline import ClauseIdentifier
from specHO.config import load_config
from specHO.models import Token


def diagnose_preprocessing(text: str):
    """Run comprehensive diagnostics on preprocessing pipeline."""

    print("=" * 80)
    print("PREPROCESSING DIAGNOSTICS")
    print("=" * 80)
    print()

    # Initialize components
    config = load_config("simple")
    preprocessor = LinguisticPreprocessor()
    clause_identifier = ClauseIdentifier()

    # Stage 1: Tokenization
    print("STAGE 1: TOKENIZATION")
    print("-" * 80)
    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(text)

    print(f"Total tokens: {len(tokens)}")
    print(f"First 10 tokens: {[t.text for t in tokens[:10]]}")
    print(f"Sample tokens (every 500th):")
    for i in range(0, len(tokens), 500):
        t = tokens[i]
        print(f"  [{i}] '{t.text}' | pos={t.pos_tag} | is_content={t.is_content_word} | phonetic={t.phonetic}")
    print()

    # Stage 2: POS Tagging
    print("STAGE 2: POS TAGGING")
    print("-" * 80)
    pos_tagger = POSTagger()
    tagged_tokens = pos_tagger.tag(tokens)

    # Check alignment
    if len(tokens) != len(tagged_tokens):
        print(f"⚠️  WARNING: Token count mismatch!")
        print(f"   Input tokens: {len(tokens)}")
        print(f"   Output tokens: {len(tagged_tokens)}")

    # Check POS distribution
    pos_counts = Counter(t.pos_tag for t in tagged_tokens)
    print(f"POS tag distribution (top 10):")
    for pos, count in pos_counts.most_common(10):
        print(f"  {pos}: {count} ({count/len(tagged_tokens)*100:.1f}%)")

    # Check content word identification
    content_words = [t for t in tagged_tokens if t.is_content_word]
    content_rate = len(content_words) / len(tagged_tokens) if tagged_tokens else 0

    print(f"\nContent word analysis:")
    print(f"  Total content words: {len(content_words)}")
    print(f"  Content word rate: {content_rate*100:.1f}%")
    print(f"  Expected: 30-70%")

    if content_rate < 0.30:
        print(f"  [X] CRITICAL: Content word rate too low! ({content_rate*100:.1f}%)")
    elif content_rate > 0.70:
        print(f"  [!] WARNING: Content word rate too high! ({content_rate*100:.1f}%)")
    else:
        print(f"  [OK] Content word rate OK")

    # Check specific content words
    print(f"\nSample content words:")
    for i, cw in enumerate(content_words[:20]):
        print(f"  [{i}] '{cw.text}' (pos={cw.pos_tag})")

    # Check field population
    populated_tokens = [
        t for t in tagged_tokens
        if t.pos_tag and t.pos_tag != "" and t.is_content_word is not None
    ]
    population_rate = len(populated_tokens) / len(tagged_tokens) if tagged_tokens else 0

    print(f"\nField population:")
    print(f"  Populated tokens: {len(populated_tokens)} / {len(tagged_tokens)}")
    print(f"  Population rate: {population_rate*100:.1f}%")
    print(f"  Expected: >80%")

    if population_rate < 0.80:
        print(f"  [X] CRITICAL: Field population too low! ({population_rate*100:.1f}%)")
    else:
        print(f"  [OK] Field population OK")

    print()

    # Stage 3: Phonetic Transcription
    print("STAGE 3: PHONETIC TRANSCRIPTION")
    print("-" * 80)
    phonetic_transcriber = PhoneticTranscriber()
    phonetic_tokens = phonetic_transcriber.transcribe_tokens(tagged_tokens)

    # Check phonetic coverage
    with_phonetic = [t for t in phonetic_tokens if t.phonetic and t.phonetic != ""]
    phonetic_coverage = len(with_phonetic) / len(phonetic_tokens) if phonetic_tokens else 0

    print(f"Phonetic coverage: {len(with_phonetic)} / {len(phonetic_tokens)} ({phonetic_coverage*100:.1f}%)")
    print(f"Expected: >50% (content words should have phonetic)")

    # Sample phonetic transcriptions
    print(f"\nSample phonetic transcriptions:")
    for i in range(0, min(20, len(phonetic_tokens))):
        t = phonetic_tokens[i]
        if t.phonetic and t.phonetic != "":
            print(f"  '{t.text}' -> {t.phonetic} (syllables: {t.syllable_count})")

    print()

    # Stage 4: Full Pipeline
    print("STAGE 4: FULL PIPELINE")
    print("-" * 80)
    full_tokens, spacy_doc = preprocessor.process(text)

    # Check spaCy doc alignment
    print(f"Token alignment check:")
    print(f"  Custom tokens: {len(full_tokens)}")
    print(f"  spaCy tokens: {len(spacy_doc)}")
    print(f"  Difference: {abs(len(full_tokens) - len(spacy_doc))}")

    if len(full_tokens) != len(spacy_doc):
        print(f"  [!] WARNING: Token count mismatch detected!")
        print(f"  This may indicate POSTagger alignment issues")

    # Field population after full pipeline
    fully_populated = [
        t for t in full_tokens
        if t.pos_tag and t.phonetic and t.is_content_word is not None
    ]
    full_population = len(fully_populated) / len(full_tokens) if full_tokens else 0

    print(f"\nFull enrichment status:")
    print(f"  Fully populated: {len(fully_populated)} / {len(full_tokens)} ({full_population*100:.1f}%)")
    print(f"  Expected: >80%")

    if full_population < 0.80:
        print(f"  [X] CRITICAL: Full population too low! ({full_population*100:.1f}%)")

    # Content words after full pipeline
    final_content_words = [t for t in full_tokens if t.is_content_word]
    final_content_rate = len(final_content_words) / len(full_tokens) if full_tokens else 0

    print(f"\nFinal content word analysis:")
    print(f"  Content words: {len(final_content_words)} / {len(full_tokens)} ({final_content_rate*100:.1f}%)")
    print(f"  Expected: 30-70%")

    if final_content_rate < 0.30:
        print(f"  [X] CRITICAL: Content word rate too low! ({final_content_rate*100:.1f}%)")

    print()

    # Stage 5: Clause Identification and Zones
    print("STAGE 5: CLAUSE IDENTIFICATION & ZONES")
    print("-" * 80)
    clause_pairs = clause_identifier.identify_clause_pairs(full_tokens, spacy_doc)

    print(f"Total clause pairs: {len(clause_pairs)}")

    # Analyze zone sizes
    zone_sizes_first = []
    zone_sizes_second = []
    zone_content_first = []
    zone_content_second = []

    for pair in clause_pairs:
        # First clause zones
        if pair.first_clause_zone:
            zone_sizes_first.append(len(pair.first_clause_zone))
            content_in_zone = [t for t in pair.first_clause_zone if t.is_content_word]
            zone_content_first.append(len(content_in_zone))

        # Second clause zones
        if pair.second_clause_zone:
            zone_sizes_second.append(len(pair.second_clause_zone))
            content_in_zone = [t for t in pair.second_clause_zone if t.is_content_word]
            zone_content_second.append(len(content_in_zone))

    print(f"\nZone size analysis:")
    if zone_sizes_first:
        avg_first = sum(zone_sizes_first) / len(zone_sizes_first)
        avg_content_first = sum(zone_content_first) / len(zone_content_first)
        print(f"  First clause zones:")
        print(f"    Average size: {avg_first:.1f} tokens")
        print(f"    Average content words: {avg_content_first:.1f}")
        print(f"    Min/Max: {min(zone_sizes_first)} / {max(zone_sizes_first)}")

    if zone_sizes_second:
        avg_second = sum(zone_sizes_second) / len(zone_sizes_second)
        avg_content_second = sum(zone_content_second) / len(zone_content_second)
        print(f"  Second clause zones:")
        print(f"    Average size: {avg_second:.1f} tokens")
        print(f"    Average content words: {avg_content_second:.1f}")
        print(f"    Min/Max: {min(zone_sizes_second)} / {max(zone_sizes_second)}")

    # Check for tiny zones
    tiny_zones_first = [s for s in zone_sizes_first if s < 2]
    tiny_zones_second = [s for s in zone_sizes_second if s < 2]

    if tiny_zones_first or tiny_zones_second:
        print(f"\n  [!] WARNING: Tiny zones detected!")
        print(f"    First clause: {len(tiny_zones_first)} zones with <2 tokens")
        print(f"    Second clause: {len(tiny_zones_second)} zones with <2 tokens")
        print(f"    Tiny zones produce unreliable similarity scores")

    # Sample clause pairs
    print(f"\nSample clause pairs:")
    for i, pair in enumerate(clause_pairs[:5]):
        print(f"\n  Pair {i+1}:")
        print(f"    First clause: {pair.first_clause.boundary_type} | {len(pair.first_clause.tokens)} tokens")
        if pair.first_clause_zone:
            zone_text = " ".join(t.text for t in pair.first_clause_zone[:5])
            print(f"    First zone: [{len(pair.first_clause_zone)} tokens] {zone_text}...")
        print(f"    Second clause: {pair.second_clause.boundary_type} | {len(pair.second_clause.tokens)} tokens")
        if pair.second_clause_zone:
            zone_text = " ".join(t.text for t in pair.second_clause_zone[:5])
            print(f"    Second zone: [{len(pair.second_clause_zone)} tokens] {zone_text}...")

    print()

    # Summary
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print()

    issues = []

    if content_rate < 0.30 or content_rate > 0.70:
        issues.append(f"Content word rate: {content_rate*100:.1f}% (expected 30-70%)")

    if population_rate < 0.80:
        issues.append(f"Field population: {population_rate*100:.1f}% (expected >80%)")

    if len(full_tokens) != len(spacy_doc):
        issues.append(f"Token alignment mismatch: {len(full_tokens)} vs {len(spacy_doc)}")

    if tiny_zones_first or tiny_zones_second:
        total_tiny = len(tiny_zones_first) + len(tiny_zones_second)
        issues.append(f"Tiny zones: {total_tiny} zones with <2 tokens")

    if issues:
        print("[!!!] CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  [X] {issue}")
    else:
        print("[OK] All diagnostics passed!")

    print()
    print("=" * 80)


def main():
    """Main entry point."""

    # Load sample.txt
    sample_path = Path(__file__).parent.parent / "specHO" / "sample.txt"

    if not sample_path.exists():
        print(f"Error: sample.txt not found at {sample_path}")
        sys.exit(1)

    print(f"Loading: {sample_path}")
    text = sample_path.read_text(encoding='utf-8')

    print(f"Text length: {len(text)} characters")
    print(f"Text preview: {text[:200]}...")
    print()

    # Run diagnostics
    diagnose_preprocessing(text)


if __name__ == "__main__":
    main()
