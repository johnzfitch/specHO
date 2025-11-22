"""
Transition Smoothness Analyzer for SpecHO Watermark Detector.

Analyzes the frequency of smooth transition words/phrases as an AI watermark
indicator. AI models (especially GPT-4) tend to overuse smooth, formal transitions
to create cohesive-sounding text.

Part of Component 4: Scoring Module (Supplementary analyzer).
"""

from typing import List, Set, Tuple


class TransitionSmoothnessAnalyzer:
    """
    Analyzes transition word/phrase frequency as an AI tell.

    AI-generated text often contains an unnaturally high rate of smooth,
    formal transition words and phrases. This creates overly polished,
    academic-sounding prose that lacks the natural roughness of human writing.

    Key AI Tell:
    - Transition rate >0.25 per sentence = HIGH suspicion (AI-typical)
    - Transition rate 0.15-0.25 per sentence = MODERATE suspicion
    - Transition rate <0.15 per sentence = LOW suspicion (human-typical)

    Based on toolkit analysis of GPT-4 generated academic text.
    """

    # Comprehensive list of AI-typical smooth transitions
    SMOOTH_TRANSITIONS: Set[str] = {
        # Contrast transitions
        'however', 'nevertheless', 'nonetheless', 'conversely',
        'on the other hand', 'in contrast', 'rather', 'alternatively',
        
        # Addition transitions
        'moreover', 'furthermore', 'additionally', 'likewise',
        'similarly', 'in addition', 'also', 'besides',
        
        # Clarification transitions
        'to be clear', 'in other words', 'specifically', 'namely',
        'that is', 'in particular', 'more precisely',
        
        # Sequential transitions
        'first', 'second', 'third', 'next', 'then', 'finally',
        'subsequently', 'meanwhile', 'in turn',
        
        # Causal transitions
        'therefore', 'thus', 'consequently', 'accordingly',
        'as a result', 'hence', 'for this reason',
        
        # Elaboration transitions
        'indeed', 'in fact', 'notably', 'significantly',
        'importantly', 'as part of', 'building on this',
        
        # Experimentation/research transitions
        'in another experiment', 'in a follow-up study',
        'to test this', 'to examine this further',
        
        # Summary transitions
        'in summary', 'in conclusion', 'overall', 'ultimately',
        'in essence', 'to sum up'
    }

    def analyze_text(self, text: str) -> Tuple[int, int, float, float]:
        """
        Analyze transition smoothness for entire document.

        Args:
            text: Full document text to analyze

        Returns:
            Tuple of (transition_count, sentence_count, rate, score):
            - transition_count: Number of smooth transitions found
            - sentence_count: Number of sentences in text
            - rate: Transitions per sentence (float)
            - score: Float in [0, 1] representing AI suspicion based on rate
                0.0-0.3: Low suspicion (human-typical)
                0.3-0.6: Moderate suspicion
                0.6-1.0: High suspicion (AI-typical)

        Algorithm:
            1. Split text into sentences (simple split on .!?)
            2. Search for transition words/phrases at sentence start or after commas
            3. Calculate rate = transition_count / sentence_count
            4. Map rate to [0,1] suspicion score
        """
        if not text or not text.strip():
            return 0, 0, 0.0, 0.0

        # Split into sentences (simple approach for Tier 1)
        # Split on periods, exclamation marks, question marks
        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        sentence_count = len(sentences)

        if sentence_count == 0:
            return 0, 0, 0.0, 0.0

        # Count transitions
        transition_count = 0
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if sentence starts with a transition
            for transition in self.SMOOTH_TRANSITIONS:
                # Check at start of sentence
                if sentence_lower.startswith(transition):
                    transition_count += 1
                    break  # Count only one transition per sentence
                # Check after comma (common AI pattern: ", however, ...")
                elif f', {transition}' in sentence_lower:
                    transition_count += 1
                    break

        # Calculate rate
        rate = transition_count / sentence_count

        # Map rate to suspicion score based on toolkit thresholds
        score = self._rate_to_score(rate)

        return transition_count, sentence_count, rate, score

    def _rate_to_score(self, rate: float) -> float:
        """
        Convert transition rate to normalized suspicion score.

        Scoring function based on toolkit analysis:
        - rate < 0.15: 0.0-0.3 (low suspicion, human-typical)
        - rate 0.15-0.25: 0.3-0.6 (moderate suspicion)
        - rate > 0.25: 0.6-1.0 (high suspicion, AI-typical)

        Args:
            rate: Transitions per sentence (float)

        Returns:
            Float in [0, 1] representing AI suspicion level
        """
        if rate < 0.15:
            # Linear mapping: 0.0 -> 0.0, 0.15 -> 0.3
            return rate * 2.0
        elif rate < 0.25:
            # Linear mapping: 0.15 -> 0.3, 0.25 -> 0.6
            return 0.3 + ((rate - 0.15) * 3.0)
        else:
            # Asymptotic approach to 1.0 for high rates
            # 0.25 -> 0.6, 0.5 -> 0.85, 1.0 -> 0.95
            normalized = min((rate - 0.25) / 0.75, 1.0)
            return 0.6 + (normalized * 0.4)

    def get_transitions_in_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Get list of transitions found in text (for debugging/display).

        Args:
            text: Text to analyze

        Returns:
            List of (transition_phrase, sentence_snippet) tuples

        Example:
            >>> analyzer = TransitionSmoothnessAnalyzer()
            >>> transitions = analyzer.get_transitions_in_text(text)
            >>> for phrase, snippet in transitions:
            ...     print(f"'{phrase}' in: {snippet}")
            'however' in: However, the results showed...
            'moreover' in: Moreover, we found that...
        """
        if not text or not text.strip():
            return []

        import re
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        found_transitions = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            for transition in self.SMOOTH_TRANSITIONS:
                if sentence_lower.startswith(transition):
                    snippet = sentence[:80] + ('...' if len(sentence) > 80 else '')
                    found_transitions.append((transition, snippet))
                    break
                elif f', {transition}' in sentence_lower:
                    # Find the context around the transition
                    idx = sentence_lower.find(f', {transition}')
                    start = max(0, idx - 20)
                    end = min(len(sentence), idx + 60)
                    snippet = ('...' if start > 0 else '') + sentence[start:end] + \
                             ('...' if end < len(sentence) else '')
                    found_transitions.append((transition, snippet))
                    break

        return found_transitions


def quick_transition_analysis(text: str) -> float:
    """
    Convenience function for quick transition smoothness analysis.

    Args:
        text: Document text

    Returns:
        Transition smoothness score in [0, 1]
    """
    analyzer = TransitionSmoothnessAnalyzer()
    _, _, _, score = analyzer.analyze_text(text)
    return score
