"""
Enhanced SpecHO analysis with detailed clause-level reporting
"""

import re
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

# Read the article
with open('/home/claude/article.txt', 'r') as f:
    text = f.read()

# Parse into sentences
sentences = sent_tokenize(text)

print("="*80)
print("DETAILED SpecHO ANALYSIS - CLAUSE-LEVEL BREAKDOWN")
print("="*80)

# Focus on the most suspicious patterns we discussed
suspicious_sentences = [
    "And it's easy to understand their appeal: Ask a question, get a polished synthesis and move on – it feels like effortless learning.",
    "People who learned about a topic through an LLM versus web search felt that they learned less, invested less effort in subsequently writing their advice, and ultimately wrote advice that was shorter, less factual and more generic.",
    "When we learn about a topic through Google search, we face much more \"friction\": We must navigate different web links, read informational sources, and interpret and synthesize them ourselves.",
    "To be clear, we do not believe the solution to these issues is to avoid using LLMs, especially given the undeniable benefits they offer in many contexts.",
    "Rather, our message is that people simply need to become smarter or more strategic users of LLMs – which starts by understanding the domains wherein LLMs are beneficial versus harmful to their goals."
]

def analyze_sentence_deeply(sentence):
    """Deep analysis of a single sentence"""
    print(f"\n{'-'*80}")
    print(f"SENTENCE: {sentence}")
    print(f"{'-'*80}")
    
    # Split on major clause boundaries
    clauses = re.split(r'[;:]|\s+–\s+|\s+—\s+', sentence)
    clauses = [c.strip() for c in clauses if c.strip()]
    
    print(f"\nCLAUSE COUNT: {len(clauses)}")
    
    # Further split on commas for parallel structure detection
    sub_clauses = []
    for clause in clauses:
        parts = [p.strip() for p in clause.split(',') if p.strip()]
        sub_clauses.append(parts)
        
    print(f"\nCLAUSE BREAKDOWN:")
    for i, clause in enumerate(clauses, 1):
        print(f"  [{i}] {clause}")
        
        # Check for comma-separated parallel elements
        if ',' in clause:
            parts = [p.strip() for p in clause.split(',')]
            if len(parts) > 2:
                print(f"      → Contains {len(parts)} parallel elements:")
                for j, part in enumerate(parts, 1):
                    print(f"        {j}. {part}")
    
    # Analyze parallel structure
    if len(clauses) > 1 or any(len(sc) > 2 for sc in sub_clauses):
        print(f"\n  PARALLEL STRUCTURE ANALYSIS:")
        
        # Check for repeated phrase patterns
        for i, parts in enumerate(sub_clauses):
            if len(parts) >= 3:
                print(f"\n    Clause {i+1} has {len(parts)} parallel elements:")
                
                # Get POS tags for each part
                for j, part in enumerate(parts):
                    words = word_tokenize(part)
                    pos = pos_tag(words)
                    # Get just the first POS tag (sentence structure)
                    if pos:
                        first_pos = pos[0][1]
                        print(f"      [{j+1}] {part[:40]:<40} starts with: {first_pos}")
                
                # Check for repetitive starts
                starts = [word_tokenize(p)[0].lower() if word_tokenize(p) else '' for p in parts]
                if len(set(starts)) < len(starts):
                    print(f"    ⚠️  REPETITIVE STARTS DETECTED: {starts}")
    
    # Semantic similarity check
    print(f"\n  SEMANTIC ANALYSIS:")
    if len(clauses) >= 2:
        for i in range(len(clauses)-1):
            words1 = set(word_tokenize(clauses[i].lower())) 
            words2 = set(word_tokenize(clauses[i+1].lower()))
            
            stop_words = {'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from','as','is','was','are','were','be','been','being','that','this','these','those'}
            words1 = words1 - stop_words
            words2 = words2 - stop_words
            
            if words1 and words2:
                overlap = words1 & words2
                jaccard = len(overlap) / len(words1 | words2)
                print(f"    Clauses {i+1}↔{i+2}: {jaccard:.2f} similarity")
                if overlap:
                    print(f"      Shared words: {', '.join(sorted(overlap))}")
    
    # Count em-dashes
    em_count = sentence.count('–') + sentence.count('—')
    if em_count > 0:
        print(f"\n  ⚠️  EM-DASH COUNT: {em_count}")

# Analyze suspicious sentences
for sent in suspicious_sentences:
    analyze_sentence_deeply(sent)

# Overall statistics
print("\n" + "="*80)
print("OVERALL ARTICLE STATISTICS")
print("="*80)

em_dash_count = text.count('–') + text.count('—')
print(f"Total sentences: {len(sentences)}")
print(f"Total em-dashes: {em_dash_count}")
print(f"Em-dash rate: {em_dash_count/len(sentences):.2f} per sentence")
print(f"\nBenchmarks:")
print(f"  Human writing: typically <0.3 em-dashes per sentence")
print(f"  AI writing (GPT-4): typically 0.5-1.0 em-dashes per sentence")
print(f"  This article: {em_dash_count/len(sentences):.2f}")

# Check for other AI tells
parallel_count = 0
for sent in sentences:
    clauses = re.split(r'[;:]|\s+–\s+|\s+—\s+', sent)
    for clause in clauses:
        parts = [p.strip() for p in clause.split(',') if p.strip()]
        if len(parts) >= 3:
            parallel_count += 1

print(f"\nSentences with 3+ parallel elements: {parallel_count}")
print(f"Parallel structure rate: {parallel_count/len(sentences):.2f} per sentence")
print(f"  (AI typical: >0.3, Human typical: <0.2)")

