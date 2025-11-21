"""
SpecHO (Spectral Harmonics of Text) Analyzer
Implements "The Echo Rule" methodology for detecting AI-generated text
through phonetic, structural, and semantic analysis of clause pairs.
"""

import re
import numpy as np
from collections import defaultdict
import json

# We'll use basic NLP without heavy dependencies first
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('cmudict', quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag

class SpecHOAnalyzer:
    def __init__(self):
        self.cmu_dict = None
        try:
            self.cmu_dict = nltk.corpus.cmudict.dict()
        except:
            pass
    
    def parse_into_clauses(self, text):
        """Parse text into sentences and clauses"""
        sentences = sent_tokenize(text)
        
        results = []
        for sent in sentences:
            # Split on common clause boundaries
            clauses = re.split(r'[;:,]|\s+–\s+|\s+—\s+', sent)
            clauses = [c.strip() for c in clauses if c.strip()]
            
            if len(clauses) > 1:
                results.append({
                    'sentence': sent,
                    'clauses': clauses,
                    'clause_count': len(clauses)
                })
        
        return results
    
    def count_syllables(self, word):
        """Count syllables using CMU dict or fallback heuristic"""
        word = word.lower()
        
        if self.cmu_dict and word in self.cmu_dict:
            # CMU dict entries have stress markers (0,1,2)
            phonemes = self.cmu_dict[word][0]
            return sum(1 for p in phonemes if p[-1].isdigit())
        
        # Fallback: count vowel groups
        word = re.sub(r'[^aeiouy]+', ' ', word.lower())
        syllables = len(word.split())
        return max(1, syllables)
    
    def get_stress_pattern(self, text):
        """Extract stress pattern from text"""
        words = word_tokenize(text)
        pattern = []
        
        for word in words:
            if word.isalpha():
                syllables = self.count_syllables(word)
                pattern.append(syllables)
        
        return pattern
    
    def analyze_phonetic_rhythm(self, clauses):
        """Analyze phonetic patterns across clause pairs"""
        if len(clauses) < 2:
            return None
        
        patterns = [self.get_stress_pattern(c) for c in clauses]
        
        # Compare consecutive clause pairs
        similarities = []
        for i in range(len(patterns) - 1):
            p1, p2 = patterns[i], patterns[i+1]
            
            # Calculate rhythm similarity (total syllables, average per word)
            total_sim = abs(sum(p1) - sum(p2))
            avg_sim = abs(np.mean(p1) - np.mean(p2)) if p1 and p2 else 0
            
            similarities.append({
                'clause_pair': (i, i+1),
                'syllable_diff': total_sim,
                'avg_syllable_diff': avg_sim,
                'pattern_1': p1,
                'pattern_2': p2
            })
        
        return similarities
    
    def analyze_structural_parallelism(self, clauses):
        """Detect parallel syntactic structures"""
        if len(clauses) < 2:
            return None
        
        pos_patterns = []
        for clause in clauses:
            words = word_tokenize(clause)
            tags = pos_tag(words)
            # Simplify POS tags to basic categories
            simplified = [tag[:2] for word, tag in tags]
            pos_patterns.append(simplified)
        
        # Compare consecutive pairs
        parallels = []
        for i in range(len(pos_patterns) - 1):
            p1, p2 = pos_patterns[i], pos_patterns[i+1]
            
            # Calculate structural similarity
            # Check if they start with same POS pattern
            min_len = min(len(p1), len(p2))
            matches = sum(1 for j in range(min_len) if p1[j] == p2[j])
            
            similarity = matches / max(len(p1), len(p2)) if p1 or p2 else 0
            
            parallels.append({
                'clause_pair': (i, i+1),
                'pos_pattern_1': p1,
                'pos_pattern_2': p2,
                'structural_similarity': similarity,
                'starts_same': p1[0] == p2[0] if p1 and p2 else False
            })
        
        return parallels
    
    def simple_semantic_similarity(self, text1, text2):
        """Calculate simple semantic similarity based on word overlap"""
        words1 = set(word_tokenize(text1.lower()))
        words2 = set(word_tokenize(text2.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'being'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_semantic_similarity(self, clauses):
        """Analyze semantic similarity between clause pairs"""
        if len(clauses) < 2:
            return None
        
        similarities = []
        for i in range(len(clauses) - 1):
            c1, c2 = clauses[i], clauses[i+1]
            
            sim = self.simple_semantic_similarity(c1, c2)
            
            similarities.append({
                'clause_pair': (i, i+1),
                'semantic_similarity': sim,
                'clause_1': c1[:50] + '...' if len(c1) > 50 else c1,
                'clause_2': c2[:50] + '...' if len(c2) > 50 else c2
            })
        
        return similarities
    
    def detect_echo_patterns(self, parsed_data):
        """Main analysis function combining all three methods"""
        echo_scores = []
        
        for item in parsed_data:
            if item['clause_count'] < 2:
                continue
            
            clauses = item['clauses']
            
            # Run all three analyses
            phonetic = self.analyze_phonetic_rhythm(clauses)
            structural = self.analyze_structural_parallelism(clauses)
            semantic = self.analyze_semantic_similarity(clauses)
            
            # Calculate composite echo score
            echo_indicators = []
            
            if phonetic:
                # Low syllable differences suggest rhythmic echoing
                avg_syl_diff = np.mean([p['avg_syllable_diff'] for p in phonetic])
                echo_indicators.append(('phonetic', 1.0 - min(avg_syl_diff, 1.0)))
            
            if structural:
                # High structural similarity suggests parallel construction
                avg_struct_sim = np.mean([s['structural_similarity'] for s in structural])
                echo_indicators.append(('structural', avg_struct_sim))
            
            if semantic:
                # Moderate semantic similarity is the AI "sweet spot"
                avg_sem_sim = np.mean([s['semantic_similarity'] for s in semantic])
                # Peak suspicion around 0.3-0.5 similarity
                if 0.3 <= avg_sem_sim <= 0.5:
                    echo_indicators.append(('semantic', avg_sem_sim * 2))
                else:
                    echo_indicators.append(('semantic', avg_sem_sim))
            
            if echo_indicators:
                composite_score = np.mean([score for _, score in echo_indicators])
                
                echo_scores.append({
                    'sentence': item['sentence'][:100] + '...' if len(item['sentence']) > 100 else item['sentence'],
                    'clause_count': item['clause_count'],
                    'phonetic_analysis': phonetic,
                    'structural_analysis': structural,
                    'semantic_analysis': semantic,
                    'echo_indicators': echo_indicators,
                    'composite_echo_score': composite_score,
                    'high_suspicion': composite_score > 0.6
                })
        
        return echo_scores
    
    def analyze_text(self, text):
        """Full SpecHO analysis pipeline"""
        parsed = self.parse_into_clauses(text)
        echo_results = self.detect_echo_patterns(parsed)
        
        # Calculate overall statistics
        if echo_results:
            avg_score = np.mean([r['composite_echo_score'] for r in echo_results])
            high_suspicion_count = sum(1 for r in echo_results if r['high_suspicion'])
            
            return {
                'parsed_sentences': len(parsed),
                'analyzed_sentences': len(echo_results),
                'average_echo_score': avg_score,
                'high_suspicion_sentences': high_suspicion_count,
                'suspicion_rate': high_suspicion_count / len(echo_results) if echo_results else 0,
                'detailed_results': echo_results
            }
        
        return None

def main():
    # Read the article
    with open('/home/claude/article.txt', 'r') as f:
        text = f.read()
    
    analyzer = SpecHOAnalyzer()
    results = analyzer.analyze_text(text)
    
    if results:
        print("="*80)
        print("SpecHO ANALYSIS RESULTS")
        print("="*80)
        print(f"\nOverall Statistics:")
        print(f"  Sentences analyzed: {results['analyzed_sentences']}")
        print(f"  Average Echo Score: {results['average_echo_score']:.3f}")
        print(f"  High Suspicion Sentences: {results['high_suspicion_sentences']} ({results['suspicion_rate']*100:.1f}%)")
        print("\n" + "="*80)
        
        print("\nDETAILED ANALYSIS OF HIGH-SUSPICION SENTENCES:\n")
        
        for i, result in enumerate(results['detailed_results'], 1):
            if result['high_suspicion']:
                print(f"\n[SENTENCE {i}] Echo Score: {result['composite_echo_score']:.3f}")
                print(f"Sentence: {result['sentence']}")
                print(f"Clauses: {result['clause_count']}")
                
                print("\n  Echo Indicators:")
                for indicator_type, score in result['echo_indicators']:
                    print(f"    {indicator_type.capitalize()}: {score:.3f}")
                
                if result['structural_analysis']:
                    print("\n  Structural Parallelism:")
                    for s in result['structural_analysis']:
                        if s['structural_similarity'] > 0.5:
                            print(f"    Clauses {s['clause_pair']}: {s['structural_similarity']:.2f} similarity")
                            print(f"    Starts with same pattern: {s['starts_same']}")
                
                if result['semantic_analysis']:
                    print("\n  Semantic Similarity:")
                    for s in result['semantic_analysis']:
                        if s['semantic_similarity'] > 0.2:
                            print(f"    Clauses {s['clause_pair']}: {s['semantic_similarity']:.2f}")
                
                print("\n" + "-"*80)
        
        # Em-dash analysis
        em_dash_count = text.count('–') + text.count('—')
        sentence_count = len(sent_tokenize(text))
        
        print("\n" + "="*80)
        print("ADDITIONAL INDICATORS:")
        print("="*80)
        print(f"Em-dash frequency: {em_dash_count} em-dashes in {sentence_count} sentences")
        print(f"Em-dash rate: {em_dash_count/sentence_count:.2f} per sentence")
        print(f"  (Human typical: <0.3, AI typical: >0.5)\n")
        
        # Save detailed results to JSON
        with open('/home/claude/specho_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Full results saved to: /home/claude/specho_results.json")

if __name__ == "__main__":
    main()
