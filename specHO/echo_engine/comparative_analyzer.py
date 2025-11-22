"""
Comparative Clustering Analyzer for SpecHO Watermark Detector.

Analyzes clustering of comparative terms within clause zones as an AI watermark
indicator. Based on toolkit analysis showing that AI (especially GPT-4) tends to
cluster multiple comparative terms in single sentences, creating a "harmonic
oscillation" pattern.

Part of Component 3: Echo Engine (Supplementary analyzer).
"""

from typing import List, Set
from specHO.models import Token


class ComparativeClusterAnalyzer:
    """
    Analyzes comparative term clustering between clause zones.

    AI-generated text often contains clusters of comparative and superlative
    terms (less/more/shorter/longer/better/worse) that create a rhythmic pattern.
    This analyzer detects such clustering as a watermark indicator.

    Key AI Tell:
    - 5+ comparatives in a sentence pair = EXTREME suspicion
    - 3-4 comparatives = HIGH suspicion
    - 2 comparatives = MODERATE suspicion
    - 0-1 comparatives = LOW suspicion

    Examples from toolkit analysis:
        "learned less, invested less effort, wrote advice that was shorter,
        less factual and more generic" → 5 comparatives (strong AI tell)
    """

    # Comprehensive list of comparative terms typical in AI writing
    COMPARATIVE_TERMS: Set[str] = {
        # Basic comparatives
        'less', 'more', 'fewer', 'greater',
        'smaller', 'larger', 'shorter', 'longer',
        'better', 'worse', 'deeper', 'shallower',
        'higher', 'lower', 'stronger', 'weaker',
        'faster', 'slower', 'easier', 'harder',
        
        # Superlatives
        'least', 'most', 'fewest', 'greatest',
        'smallest', 'largest', 'shortest', 'longest',
        'best', 'worst', 'deepest', 'shallowest',
        'highest', 'lowest', 'strongest', 'weakest',
        'fastest', 'slowest', 'easiest', 'hardest',
        
        # Adjective comparatives (common in AI)
        'simpler', 'clearer', 'broader', 'narrower',
        'richer', 'poorer', 'newer', 'older',
        'younger', 'earlier', 'later', 'nearer',
        'farther', 'further', 'closer', 'tighter',
        'looser', 'wider', 'thinner', 'thicker'
    }

    def analyze(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Calculate comparative clustering score for a clause pair.

        Args:
            zone_a: First zone (terminal content words from clause A)
            zone_b: Second zone (initial content words from clause B)

        Returns:
            Float in [0, 1] representing comparative clustering intensity.
            - 0.0-0.2: No/minimal clustering (0-1 comparatives)
            - 0.2-0.4: Mild clustering (2 comparatives)
            - 0.4-0.7: Moderate clustering (3 comparatives)
            - 0.7-0.9: High clustering (4 comparatives)
            - 0.9-1.0: Extreme clustering (5+ comparatives)

        Algorithm:
            1. Extract all tokens from both zones
            2. Count comparative terms (case-insensitive)
            3. Map count to [0,1] score using threshold function
            4. Return clustering score
        """
        if not zone_a and not zone_b:
            return 0.0

        # Extract all text tokens from both zones
        all_tokens = zone_a + zone_b
        token_texts = [token.text.lower() for token in all_tokens if token.text]

        # Count comparative terms
        comparative_count = sum(
            1 for text in token_texts if text in self.COMPARATIVE_TERMS
        )

        # Map count to [0,1] score using threshold function
        # Based on toolkit analysis thresholds
        score = self._count_to_score(comparative_count)

        return score

    def _count_to_score(self, count: int) -> float:
        """
        Convert comparative count to normalized score.

        Scoring function based on toolkit analysis:
        - 0-1 comparatives: 0.0-0.2 (minimal)
        - 2 comparatives: 0.3 (mild)
        - 3 comparatives: 0.5 (moderate)
        - 4 comparatives: 0.8 (high)
        - 5+ comparatives: 0.95-1.0 (extreme)

        Args:
            count: Number of comparative terms found

        Returns:
            Float in [0, 1]
        """
        if count == 0:
            return 0.0
        elif count == 1:
            return 0.15
        elif count == 2:
            return 0.3
        elif count == 3:
            return 0.5
        elif count == 4:
            return 0.8
        elif count >= 5:
            # Scale beyond 5: 0.95 for 5, approach 1.0 asymptotically
            return min(1.0, 0.9 + (count - 5) * 0.02)
        
        return 0.0

    def get_comparatives_in_zones(self, zone_a: List[Token], zone_b: List[Token]) -> List[str]:
        """
        Get list of comparative terms found in the zones (for debugging/display).

        Args:
            zone_a: First zone
            zone_b: Second zone

        Returns:
            List of comparative terms found (lowercase)

        Example:
            >>> analyzer = ComparativeClusterAnalyzer()
            >>> comparatives = analyzer.get_comparatives_in_zones(zone_a, zone_b)
            >>> print(comparatives)
            ['less', 'more', 'shorter', 'better']
        """
        all_tokens = zone_a + zone_b
        token_texts = [token.text.lower() for token in all_tokens if token.text]
        
        found_comparatives = [
            text for text in token_texts if text in self.COMPARATIVE_TERMS
        ]
        
        return found_comparatives


def quick_comparative_analysis(zone_a: List[Token], zone_b: List[Token]) -> float:
    """
    Convenience function for quick comparative clustering analysis.

    Args:
        zone_a: First zone
        zone_b: Second zone

    Returns:
        Comparative clustering score in [0, 1]
    """
    analyzer = ComparativeClusterAnalyzer()
    return analyzer.analyze(zone_a, zone_b)
