#!/usr/bin/env python3
"""Manual Echo Rule Watermark Analysis Script.

This script performs a manual analysis of text for Echo Rule watermark patterns
when the full spaCy model is not available. It implements the core concepts
from the SpecHO pipeline using available tools.

The Echo Rule watermark detection looks for three types of echoes between
clause boundaries:
1. Phonetic echoes - similar sounds at clause joints
2. Structural echoes - parallel grammatical structures
3. Semantic echoes - thematically related word choices
"""

import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import json

# Try to import available libraries
try:
    import cmudict
    CMU_DICT = cmudict.dict()
    HAS_CMU = True
except ImportError:
    HAS_CMU = False
    CMU_DICT = {}

try:
    from Levenshtein import ratio as levenshtein_ratio
    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class ClausePair:
    """Represents a pair of adjacent clauses for analysis."""
    clause_a: str
    clause_b: str
    separator: str  # What separates them (punctuation, conjunction, etc.)
    zone_a: List[str]  # Terminal words from clause_a
    zone_b: List[str]  # Initial words from clause_b


@dataclass
class EchoScore:
    """Scores for a single clause pair."""
    phonetic: float = 0.0
    structural: float = 0.0
    semantic: float = 0.0
    combined: float = 0.0
    details: Dict = field(default_factory=dict)


@dataclass
class AnalysisReport:
    """Complete analysis report for a document."""
    text_length: int
    word_count: int
    sentence_count: int
    clause_pairs_found: int
    echo_scores: List[EchoScore] = field(default_factory=list)
    average_phonetic: float = 0.0
    average_structural: float = 0.0
    average_semantic: float = 0.0
    final_score: float = 0.0
    verdict: str = "UNKNOWN"
    confidence: float = 0.0
    reasoning: List[str] = field(default_factory=list)


# ============================================================================
# CONTENT WORD DETECTION
# ============================================================================

# Common function words to exclude from echo analysis
FUNCTION_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'when', 'where', 'who',
    'what', 'which', 'how', 'why', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'must', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'over', 'out', 'up', 'down',
    'off', 'about', 'than', 'so', 'that', 'this', 'these', 'those', 'it', 'its',
    'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'i', 'me', 'my', 'not', 'no', 'yes', 'all', 'each', 'every',
    'both', 'few', 'more', 'most', 'other', 'some', 'such', 'any', 'only', 'same',
    'also', 'just', 'even', 'very', 'too', 'however', 'therefore', 'thus',
    'meanwhile', 'moreover', 'furthermore', 'nonetheless', 'nevertheless',
    'indeed', 'instead', 'rather', 'yet', 'still', 'already', 'always', 'never',
    'often', 'sometimes', 'usually', 'perhaps', 'probably', 'certainly'
}

TRANSITION_PHRASES = [
    'however', 'therefore', 'thus', 'meanwhile', 'moreover', 'furthermore',
    'nonetheless', 'nevertheless', 'in turn', 'rather', 'indeed', 'instead',
    'as a result', 'on the other hand', 'in contrast', 'similarly', 'likewise',
    'to be clear', 'in other words', 'for example', 'for instance'
]


def is_content_word(word: str) -> bool:
    """Check if a word is a content word (not a function word)."""
    return word.lower() not in FUNCTION_WORDS and len(word) > 2


def get_content_words(text: str) -> List[str]:
    """Extract content words from text."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return [w for w in words if is_content_word(w)]


# ============================================================================
# CLAUSE IDENTIFICATION
# ============================================================================

def identify_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr)\.\s', r'\1_DOT_ ', text)
    text = re.sub(r'\b(e\.g|i\.e)\.\s', r'\1_DOT_ ', text)

    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Restore dots
    sentences = [s.replace('_DOT_', '.') for s in sentences]

    return [s.strip() for s in sentences if s.strip()]


def identify_clause_pairs(text: str) -> List[ClausePair]:
    """Identify clause pairs using punctuation and conjunction rules."""
    pairs = []
    sentences = identify_sentences(text)

    for sentence in sentences:
        # Rule A: Punctuation-linked clauses (semicolon, em-dash, colon)
        for sep in [';', ' – ', ': ', ' — ']:
            if sep in sentence:
                parts = sentence.split(sep)
                for i in range(len(parts) - 1):
                    if len(parts[i].strip()) > 10 and len(parts[i+1].strip()) > 10:
                        pairs.append(create_clause_pair(parts[i], parts[i+1], sep))

        # Rule B: Conjunction-linked clauses
        conj_pattern = r',?\s*(but|and|or|yet)\s+'
        matches = list(re.finditer(conj_pattern, sentence, re.IGNORECASE))
        for match in matches:
            before = sentence[:match.start()]
            after = sentence[match.end():]
            if len(before.strip()) > 10 and len(after.strip()) > 10:
                pairs.append(create_clause_pair(before, after, match.group(1)))

    # Rule C: Transition-linked sentences (check consecutive sentences)
    for i in range(len(sentences) - 1):
        sentence_b = sentences[i + 1]
        for transition in TRANSITION_PHRASES:
            if sentence_b.lower().startswith(transition.lower()):
                pairs.append(create_clause_pair(sentences[i], sentence_b, transition))
                break

    return pairs


def create_clause_pair(clause_a: str, clause_b: str, separator: str) -> ClausePair:
    """Create a ClausePair with extracted zones."""
    words_a = get_content_words(clause_a)
    words_b = get_content_words(clause_b)

    # Zone A: last 3 content words
    zone_a = words_a[-3:] if len(words_a) >= 3 else words_a

    # Zone B: first 3 content words
    zone_b = words_b[:3] if len(words_b) >= 3 else words_b

    return ClausePair(
        clause_a=clause_a.strip(),
        clause_b=clause_b.strip(),
        separator=separator.strip(),
        zone_a=zone_a,
        zone_b=zone_b
    )


# ============================================================================
# PHONETIC ANALYSIS
# ============================================================================

def get_phonetic(word: str) -> Optional[str]:
    """Get ARPAbet phonetic transcription for a word."""
    if not HAS_CMU:
        return None

    word_lower = word.lower()
    if word_lower in CMU_DICT:
        # Return first pronunciation, joined
        return ' '.join(CMU_DICT[word_lower][0])
    return None


def analyze_phonetic_echo(zone_a: List[str], zone_b: List[str]) -> Tuple[float, Dict]:
    """Analyze phonetic similarity between zones."""
    if not HAS_CMU or not HAS_LEVENSHTEIN:
        return 0.0, {'error': 'Required libraries not available'}

    # Get phonetics for all words
    phonetics_a = [(w, get_phonetic(w)) for w in zone_a]
    phonetics_b = [(w, get_phonetic(w)) for w in zone_b]

    # Filter to words with phonetics
    phonetics_a = [(w, p) for w, p in phonetics_a if p]
    phonetics_b = [(w, p) for w, p in phonetics_b if p]

    if not phonetics_a or not phonetics_b:
        return 0.0, {'note': 'No phonetic data available'}

    # Calculate pairwise similarities
    similarities = []
    best_matches = []
    for word_a, phon_a in phonetics_a:
        for word_b, phon_b in phonetics_b:
            sim = levenshtein_ratio(phon_a, phon_b)
            similarities.append(sim)
            if sim > 0.5:
                best_matches.append((word_a, word_b, phon_a, phon_b, sim))

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    # Check for specific echo patterns
    # Initial consonant clusters
    initial_echo = check_initial_consonants(phonetics_a, phonetics_b)

    # Final sounds (rhyming)
    final_echo = check_final_sounds(phonetics_a, phonetics_b)

    # Combine scores (weighted)
    combined = 0.4 * avg_sim + 0.3 * initial_echo + 0.3 * final_echo

    return combined, {
        'avg_similarity': avg_sim,
        'initial_echo': initial_echo,
        'final_echo': final_echo,
        'best_matches': best_matches[:3],
        'zone_a_phonetics': phonetics_a,
        'zone_b_phonetics': phonetics_b
    }


def check_initial_consonants(phonetics_a: List, phonetics_b: List) -> float:
    """Check for alliterative patterns (matching initial consonants)."""
    initials_a = set()
    initials_b = set()

    for _, phon in phonetics_a:
        if phon:
            first_phone = phon.split()[0] if phon else ''
            # Strip stress markers from vowels
            first_phone = re.sub(r'[0-9]', '', first_phone)
            if first_phone and first_phone[0] not in 'AEIOU':
                initials_a.add(first_phone)

    for _, phon in phonetics_b:
        if phon:
            first_phone = phon.split()[0] if phon else ''
            first_phone = re.sub(r'[0-9]', '', first_phone)
            if first_phone and first_phone[0] not in 'AEIOU':
                initials_b.add(first_phone)

    if not initials_a or not initials_b:
        return 0.0

    # Jaccard similarity of initial consonants
    intersection = initials_a & initials_b
    union = initials_a | initials_b

    return len(intersection) / len(union) if union else 0.0


def check_final_sounds(phonetics_a: List, phonetics_b: List) -> float:
    """Check for rhyming patterns (matching final sounds)."""
    finals_a = set()
    finals_b = set()

    for _, phon in phonetics_a:
        if phon:
            phones = phon.split()
            if len(phones) >= 2:
                # Last two phonemes (stripped of stress)
                final = ' '.join(re.sub(r'[0-9]', '', p) for p in phones[-2:])
                finals_a.add(final)

    for _, phon in phonetics_b:
        if phon:
            phones = phon.split()
            if len(phones) >= 2:
                final = ' '.join(re.sub(r'[0-9]', '', p) for p in phones[-2:])
                finals_b.add(final)

    if not finals_a or not finals_b:
        return 0.0

    intersection = finals_a & finals_b
    union = finals_a | finals_b

    return len(intersection) / len(union) if union else 0.0


# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================

def get_simple_pos(word: str) -> str:
    """Simple rule-based POS approximation."""
    word_lower = word.lower()

    # Common verb endings
    if word_lower.endswith(('ing', 'ed', 'ize', 'ise', 'ate')):
        return 'VERB'
    # Common noun endings
    if word_lower.endswith(('tion', 'ment', 'ness', 'ity', 'ism', 'ist', 'er', 'or')):
        return 'NOUN'
    # Common adjective endings
    if word_lower.endswith(('ive', 'ous', 'ful', 'less', 'able', 'ible', 'al', 'ic')):
        return 'ADJ'
    # Common adverb endings
    if word_lower.endswith('ly'):
        return 'ADV'

    return 'NOUN'  # Default


def analyze_structural_echo(zone_a: List[str], zone_b: List[str]) -> Tuple[float, Dict]:
    """Analyze structural (POS pattern) similarity between zones."""
    if not zone_a or not zone_b:
        return 0.0, {'error': 'Empty zones'}

    # Get POS tags
    pos_a = [get_simple_pos(w) for w in zone_a]
    pos_b = [get_simple_pos(w) for w in zone_b]

    # Create bigrams
    bigrams_a = set()
    bigrams_b = set()

    for i in range(len(pos_a) - 1):
        bigrams_a.add((pos_a[i], pos_a[i+1]))
    for i in range(len(pos_b) - 1):
        bigrams_b.add((pos_b[i], pos_b[i+1]))

    # Jaccard similarity of bigrams
    if not bigrams_a or not bigrams_b:
        # Fall back to unigram comparison
        pos_set_a = set(pos_a)
        pos_set_b = set(pos_b)
        intersection = pos_set_a & pos_set_b
        union = pos_set_a | pos_set_b
        unigram_sim = len(intersection) / len(union) if union else 0.0
        return unigram_sim, {
            'pos_a': pos_a,
            'pos_b': pos_b,
            'method': 'unigram',
            'similarity': unigram_sim
        }

    intersection = bigrams_a & bigrams_b
    union = bigrams_a | bigrams_b
    bigram_sim = len(intersection) / len(union) if union else 0.0

    return bigram_sim, {
        'pos_a': pos_a,
        'pos_b': pos_b,
        'bigrams_a': list(bigrams_a),
        'bigrams_b': list(bigrams_b),
        'matching_bigrams': list(intersection),
        'method': 'bigram',
        'similarity': bigram_sim
    }


# ============================================================================
# SEMANTIC ANALYSIS
# ============================================================================

# Word categories for basic semantic grouping
SEMANTIC_CATEGORIES = {
    'learning': ['learn', 'learning', 'knowledge', 'understand', 'education', 'skill', 'study', 'information', 'insight', 'comprehension'],
    'technology': ['llm', 'chatgpt', 'ai', 'model', 'google', 'search', 'tool', 'technology', 'digital', 'computer', 'algorithm'],
    'research': ['study', 'research', 'experiment', 'paper', 'finding', 'data', 'evidence', 'participant', 'result', 'analysis'],
    'cognitive': ['think', 'thought', 'cognitive', 'mental', 'process', 'active', 'passive', 'engage', 'effort', 'friction'],
    'comparison': ['compare', 'versus', 'difference', 'similar', 'contrast', 'better', 'worse', 'more', 'less', 'shallow', 'deep'],
    'communication': ['write', 'read', 'advice', 'information', 'message', 'response', 'summary', 'synthesis', 'source'],
}


def get_semantic_category(word: str) -> Optional[str]:
    """Get semantic category for a word."""
    word_lower = word.lower()
    for category, words in SEMANTIC_CATEGORIES.items():
        if word_lower in words or any(word_lower.startswith(w) for w in words):
            return category
    return None


def analyze_semantic_echo(zone_a: List[str], zone_b: List[str]) -> Tuple[float, Dict]:
    """Analyze semantic relatedness between zones."""
    if not zone_a or not zone_b:
        return 0.0, {'error': 'Empty zones'}

    # Get categories
    cats_a = [get_semantic_category(w) for w in zone_a]
    cats_b = [get_semantic_category(w) for w in zone_b]

    # Filter None values
    cats_a = [c for c in cats_a if c]
    cats_b = [c for c in cats_b if c]

    # Check for overlapping categories
    if not cats_a and not cats_b:
        # No categorized words - use simple word overlap
        words_a = set(w.lower() for w in zone_a)
        words_b = set(w.lower() for w in zone_b)
        overlap = words_a & words_b
        if overlap:
            return 0.8, {'method': 'exact_overlap', 'overlapping': list(overlap)}
        return 0.2, {'method': 'no_overlap', 'note': 'No semantic data'}

    cats_set_a = set(cats_a)
    cats_set_b = set(cats_b)

    intersection = cats_set_a & cats_set_b
    union = cats_set_a | cats_set_b

    category_sim = len(intersection) / len(union) if union else 0.0

    return category_sim, {
        'zone_a_categories': cats_a,
        'zone_b_categories': cats_b,
        'matching_categories': list(intersection),
        'method': 'category',
        'similarity': category_sim
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_text(text: str) -> AnalysisReport:
    """Perform complete Echo Rule watermark analysis on text."""
    # Basic stats
    sentences = identify_sentences(text)
    words = re.findall(r'\b[a-zA-Z]+\b', text)

    report = AnalysisReport(
        text_length=len(text),
        word_count=len(words),
        sentence_count=len(sentences),
        clause_pairs_found=0
    )

    # Identify clause pairs
    pairs = identify_clause_pairs(text)
    report.clause_pairs_found = len(pairs)

    if not pairs:
        report.verdict = "INSUFFICIENT_DATA"
        report.reasoning.append("No suitable clause pairs found for echo analysis")
        return report

    # Analyze each pair
    phonetic_scores = []
    structural_scores = []
    semantic_scores = []

    for pair in pairs:
        # Phonetic analysis
        phon_score, phon_details = analyze_phonetic_echo(pair.zone_a, pair.zone_b)

        # Structural analysis
        struct_score, struct_details = analyze_structural_echo(pair.zone_a, pair.zone_b)

        # Semantic analysis
        sem_score, sem_details = analyze_semantic_echo(pair.zone_a, pair.zone_b)

        # Combined score (using Tier 1 weights: 40% phonetic, 30% structural, 30% semantic)
        combined = 0.4 * phon_score + 0.3 * struct_score + 0.3 * sem_score

        echo_score = EchoScore(
            phonetic=phon_score,
            structural=struct_score,
            semantic=sem_score,
            combined=combined,
            details={
                'clause_a': pair.clause_a[:50] + '...' if len(pair.clause_a) > 50 else pair.clause_a,
                'clause_b': pair.clause_b[:50] + '...' if len(pair.clause_b) > 50 else pair.clause_b,
                'separator': pair.separator,
                'zone_a': pair.zone_a,
                'zone_b': pair.zone_b,
                'phonetic': phon_details,
                'structural': struct_details,
                'semantic': sem_details
            }
        )

        report.echo_scores.append(echo_score)
        phonetic_scores.append(phon_score)
        structural_scores.append(struct_score)
        semantic_scores.append(sem_score)

    # Calculate averages
    report.average_phonetic = sum(phonetic_scores) / len(phonetic_scores)
    report.average_structural = sum(structural_scores) / len(structural_scores)
    report.average_semantic = sum(semantic_scores) / len(semantic_scores)

    # Final document score (Tier 1: simple average of combined scores)
    combined_scores = [es.combined for es in report.echo_scores]
    report.final_score = sum(combined_scores) / len(combined_scores)

    # Generate verdict
    # These thresholds are based on the Echo Rule watermark detection spec
    # Human text typically scores 0.15-0.30, AI with watermark scores 0.45+
    if report.final_score >= 0.45:
        report.verdict = "HIGH_PROBABILITY_AI"
        report.confidence = min(0.95, 0.5 + report.final_score)
        report.reasoning.append(f"High echo score ({report.final_score:.3f}) suggests Echo Rule watermark presence")
    elif report.final_score >= 0.35:
        report.verdict = "MODERATE_PROBABILITY_AI"
        report.confidence = 0.3 + report.final_score
        report.reasoning.append(f"Moderate echo score ({report.final_score:.3f}) - possible watermark presence")
    elif report.final_score >= 0.25:
        report.verdict = "LOW_PROBABILITY_AI"
        report.confidence = 0.2 + report.final_score * 0.5
        report.reasoning.append(f"Low echo score ({report.final_score:.3f}) - unlikely watermark presence")
    else:
        report.verdict = "LIKELY_HUMAN"
        report.confidence = max(0.1, 0.6 - report.final_score)
        report.reasoning.append(f"Very low echo score ({report.final_score:.3f}) - consistent with human writing")

    # Add specific observations
    if report.average_phonetic > 0.4:
        report.reasoning.append(f"Elevated phonetic echoes ({report.average_phonetic:.3f}) - sound patterns at clause boundaries")
    if report.average_structural > 0.5:
        report.reasoning.append(f"Strong structural parallelism ({report.average_structural:.3f}) - similar grammatical patterns")
    if report.average_semantic > 0.4:
        report.reasoning.append(f"Semantic coherence ({report.average_semantic:.3f}) - related concepts across boundaries")

    return report


def print_report(report: AnalysisReport, verbose: bool = False):
    """Print formatted analysis report."""
    print("\n" + "=" * 70)
    print("   SPECULATIVE ECHO RULE WATERMARK ANALYSIS REPORT")
    print("=" * 70)

    print(f"\n{'Document Statistics':=^50}")
    print(f"  Text length: {report.text_length:,} characters")
    print(f"  Word count: {report.word_count:,} words")
    print(f"  Sentences: {report.sentence_count}")
    print(f"  Clause pairs analyzed: {report.clause_pairs_found}")

    print(f"\n{'Echo Scores':=^50}")
    print(f"  Phonetic echo (avg):    {report.average_phonetic:.4f}")
    print(f"  Structural echo (avg):  {report.average_structural:.4f}")
    print(f"  Semantic echo (avg):    {report.average_semantic:.4f}")
    print(f"  {'─' * 40}")
    print(f"  FINAL SCORE:            {report.final_score:.4f}")

    print(f"\n{'Verdict':=^50}")
    print(f"  Status: {report.verdict}")
    print(f"  Confidence: {report.confidence:.1%}")

    print(f"\n{'Chain of Reasoning':=^50}")
    for i, reason in enumerate(report.reasoning, 1):
        print(f"  {i}. {reason}")

    if verbose and report.echo_scores:
        print(f"\n{'Detailed Clause Pair Analysis':=^50}")
        for i, score in enumerate(report.echo_scores[:10], 1):
            print(f"\n  Pair {i}:")
            print(f"    Separator: '{score.details.get('separator', 'N/A')}'")
            print(f"    Zone A (end): {score.details.get('zone_a', [])}")
            print(f"    Zone B (start): {score.details.get('zone_b', [])}")
            print(f"    Phonetic: {score.phonetic:.3f}, Structural: {score.structural:.3f}, Semantic: {score.semantic:.3f}")
            print(f"    Combined: {score.combined:.3f}")

    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    # Read input file
    input_file = Path('/home/user/specHO/data/analysis_input.txt')

    if not input_file.exists():
        print("Error: Input file not found")
        sys.exit(1)

    text = input_file.read_text()
    print(f"Analyzing text ({len(text):,} characters)...")

    # Run analysis
    report = analyze_text(text)

    # Print report
    print_report(report, verbose=True)

    # Save JSON report
    output_file = Path('/home/user/specHO/data/analysis_output.json')
    report_dict = {
        'text_length': report.text_length,
        'word_count': report.word_count,
        'sentence_count': report.sentence_count,
        'clause_pairs_found': report.clause_pairs_found,
        'average_phonetic': report.average_phonetic,
        'average_structural': report.average_structural,
        'average_semantic': report.average_semantic,
        'final_score': report.final_score,
        'verdict': report.verdict,
        'confidence': report.confidence,
        'reasoning': report.reasoning,
        'echo_scores': [
            {
                'phonetic': es.phonetic,
                'structural': es.structural,
                'semantic': es.semantic,
                'combined': es.combined,
                'details': es.details
            }
            for es in report.echo_scores
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(report_dict, f, indent=2, default=str)

    print(f"\nJSON report saved to: {output_file}")

    return report


if __name__ == '__main__':
    main()
