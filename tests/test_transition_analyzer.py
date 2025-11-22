"""
Tests for TransitionSmoothnessAnalyzer.

Tests the transition word frequency detection feature from toolkit analysis.
"""

import pytest
from specHO.scoring.transition_analyzer import (
    TransitionSmoothnessAnalyzer,
    quick_transition_analysis
)


@pytest.fixture
def analyzer():
    """Create a TransitionSmoothnessAnalyzer instance."""
    return TransitionSmoothnessAnalyzer()


class TestTransitionSmoothnessAnalyzer:
    """Test suite for TransitionSmoothnessAnalyzer."""

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer is not None
        assert isinstance(analyzer.SMOOTH_TRANSITIONS, set)
        assert 'however' in analyzer.SMOOTH_TRANSITIONS
        assert 'moreover' in analyzer.SMOOTH_TRANSITIONS
        assert 'furthermore' in analyzer.SMOOTH_TRANSITIONS

    def test_no_transitions(self, analyzer):
        """Test with text containing no smooth transitions."""
        text = "The cat sat on the mat. The dog ran in the park."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 0
        assert sents == 2
        assert rate == 0.0
        assert score < 0.3

    def test_one_transition(self, analyzer):
        """Test with one transition word."""
        text = "The study was important. However, results were mixed."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 1
        assert sents == 2
        assert rate == 0.5
        assert 0.3 < score < 0.8

    def test_multiple_transitions(self, analyzer):
        """Test with multiple transition words."""
        text = "First, we examined the data. Moreover, we conducted interviews. Finally, we synthesized the findings."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 3
        assert sents == 3
        assert rate == 1.0
        assert score > 0.8

    def test_high_transition_rate(self, analyzer):
        """Test with high transition rate (AI-typical)."""
        text = """
        However, the results were surprising. Moreover, the data supported this.
        Furthermore, additional studies confirmed it. Nevertheless, questions remained.
        """
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count >= 4
        assert rate > 0.25
        assert score > 0.6

    def test_comma_separated_transition(self, analyzer):
        """Test detection of transitions after commas."""
        text = "The experiment succeeded, however, with some caveats."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 1
        assert sents == 1
        assert rate == 1.0

    def test_case_insensitive(self, analyzer):
        """Test that detection is case-insensitive."""
        text = "HOWEVER, the results differed. Moreover, this was unexpected."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 2

    def test_empty_text(self, analyzer):
        """Test with empty text."""
        count, sents, rate, score = analyzer.analyze_text("")
        
        assert count == 0
        assert sents == 0
        assert rate == 0.0
        assert score == 0.0

    def test_whitespace_only(self, analyzer):
        """Test with whitespace-only text."""
        count, sents, rate, score = analyzer.analyze_text("   \n\n  ")
        
        assert count == 0
        assert sents == 0
        assert rate == 0.0
        assert score == 0.0

    def test_rate_to_score_thresholds(self, analyzer):
        """Test rate to score mapping thresholds."""
        # Low rate (human-typical)
        score_low = analyzer._rate_to_score(0.1)
        assert score_low < 0.3
        
        # Moderate rate
        score_mid = analyzer._rate_to_score(0.2)
        assert 0.3 <= score_mid <= 0.6
        
        # High rate (AI-typical)
        score_high = analyzer._rate_to_score(0.3)
        assert score_high > 0.6

    def test_get_transitions_in_text(self, analyzer):
        """Test getting list of transitions found."""
        text = "However, the study succeeded. Moreover, results were clear."
        transitions = analyzer.get_transitions_in_text(text)
        
        assert len(transitions) == 2
        assert any('however' in t[0] for t in transitions)
        assert any('moreover' in t[0] for t in transitions)

    def test_various_sentence_endings(self, analyzer):
        """Test with different sentence ending punctuation."""
        text = "What happened? However, we continued! Moreover, we succeeded."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 2
        assert sents == 3

    def test_contrast_transitions(self, analyzer):
        """Test contrast transition detection."""
        text = "The plan was solid. Nevertheless, issues emerged. Conversely, success followed."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 2
        assert 'nevertheless' in analyzer.SMOOTH_TRANSITIONS
        assert 'conversely' in analyzer.SMOOTH_TRANSITIONS

    def test_addition_transitions(self, analyzer):
        """Test addition transition detection."""
        text = "The method worked. Additionally, it was efficient. Furthermore, it was scalable."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 2
        assert 'additionally' in analyzer.SMOOTH_TRANSITIONS
        assert 'furthermore' in analyzer.SMOOTH_TRANSITIONS

    def test_causal_transitions(self, analyzer):
        """Test causal transition detection."""
        text = "The evidence was clear. Therefore, we proceeded. Consequently, results improved."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 2
        assert 'therefore' in analyzer.SMOOTH_TRANSITIONS
        assert 'consequently' in analyzer.SMOOTH_TRANSITIONS

    def test_multi_word_transitions(self, analyzer):
        """Test multi-word transition phrases."""
        text = "The data was analyzed. On the other hand, concerns remained. In addition, validation was needed."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert count == 2
        assert 'on the other hand' in analyzer.SMOOTH_TRANSITIONS
        assert 'in addition' in analyzer.SMOOTH_TRANSITIONS


class TestQuickTransitionAnalysis:
    """Test convenience function."""

    def test_quick_analysis(self):
        """Test quick_transition_analysis convenience function."""
        text = "However, the study succeeded. Moreover, results were clear."
        score = quick_transition_analysis(text)
        
        assert score > 0.6


class TestTransitionCoverage:
    """Test that comprehensive set of transition words is covered."""

    def test_contrast_transitions_present(self, analyzer):
        """Test contrast transitions are in the set."""
        contrast = ['however', 'nevertheless', 'nonetheless', 'conversely', 
                   'on the other hand', 'in contrast', 'rather']
        for term in contrast:
            assert term in analyzer.SMOOTH_TRANSITIONS

    def test_addition_transitions_present(self, analyzer):
        """Test addition transitions are in the set."""
        addition = ['moreover', 'furthermore', 'additionally', 'likewise',
                   'similarly', 'in addition', 'also']
        for term in addition:
            assert term in analyzer.SMOOTH_TRANSITIONS

    def test_clarification_transitions_present(self, analyzer):
        """Test clarification transitions are in the set."""
        clarification = ['to be clear', 'in other words', 'specifically', 
                        'namely', 'that is', 'in particular']
        for term in clarification:
            assert term in analyzer.SMOOTH_TRANSITIONS

    def test_sequential_transitions_present(self, analyzer):
        """Test sequential transitions are in the set."""
        sequential = ['first', 'second', 'third', 'next', 'then', 
                     'finally', 'subsequently', 'in turn']
        for term in sequential:
            assert term in analyzer.SMOOTH_TRANSITIONS

    def test_causal_transitions_present(self, analyzer):
        """Test causal transitions are in the set."""
        causal = ['therefore', 'thus', 'consequently', 'accordingly',
                 'as a result', 'hence', 'for this reason']
        for term in causal:
            assert term in analyzer.SMOOTH_TRANSITIONS


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_transition_at_start_of_text(self, analyzer):
        """Test transition word at very start of text."""
        text = "However, the experiment succeeded."
        count, _, _, _ = analyzer.analyze_text(text)
        assert count == 1

    def test_transition_only_sentence(self, analyzer):
        """Test sentence that is only a transition word."""
        text = "The plan worked. However."
        count, sents, rate, score = analyzer.analyze_text(text)
        assert count == 1
        assert sents == 2

    def test_very_long_text(self, analyzer):
        """Test with longer text to ensure performance."""
        sentences = []
        for i in range(100):
            if i % 3 == 0:
                sentences.append("However, the process continued.")
            else:
                sentences.append("The work was important.")
        
        text = " ".join(sentences)
        count, sents, rate, score = analyzer.analyze_text(text)
        
        assert sents == 100
        assert count > 0
        assert 0.0 <= rate <= 1.0
        assert 0.0 <= score <= 1.0

    def test_multiple_transitions_same_sentence(self, analyzer):
        """Test sentence with multiple transitions (should count once)."""
        text = "However, the results were clear; moreover, they were significant."
        count, sents, rate, score = analyzer.analyze_text(text)
        
        # Should only count one transition per sentence
        assert count == 1
        assert sents == 1
