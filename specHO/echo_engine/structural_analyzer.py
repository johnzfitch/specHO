"""
Structural Echo Analyzer for SpecHO Watermark Detector.

Analyzes structural similarity between clause zones based on POS patterns and syllable counts.
Also tracks em-dash frequency as an AI watermark indicator.
Part of Component 3: Echo Engine (Task 4.2).

Tier 1: Simple exact matching of POS patterns and syllable count comparison.
"""

from typing import List, Tuple
from specHO.models import Token


class StructuralEchoAnalyzer:
    """
    Analyzes structural echoes between clause zones.

    Tier 1 algorithm:
    - POS pattern comparison: exact match of POS tag sequences
    - Syllable count similarity: normalized absolute difference
    - Combined score: pattern_sim * 0.5 + syllable_sim * 0.5
    """

    def analyze(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Calculate structural similarity between two zones.

        Args:
            zone_a: First zone (terminal content words from clause A)
            zone_b: Second zone (initial content words from clause B)

        Returns:
            Float in [0, 1] representing structural similarity.
            Returns 0.0 if either zone is empty.

        Algorithm:
            1. Extract POS patterns from both zones
            2. Calculate POS pattern similarity (exact match)
            3. Extract syllable counts from both zones
            4. Calculate syllable count similarity
            5. Combine: 0.5 * pattern_sim + 0.5 * syllable_sim
        """
        if not zone_a or not zone_b:
            return 0.0

        # Calculate individual similarity scores
        pos_similarity = self.compare_pos_patterns(zone_a, zone_b)
        syllable_similarity = self.compare_syllable_counts(zone_a, zone_b)

        # Combine with equal weighting (Tier 1)
        combined_score = (pos_similarity * 0.5) + (syllable_similarity * 0.5)

        # Ensure output is in [0, 1] range
        return max(0.0, min(1.0, combined_score))

    def compare_pos_patterns(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Compare POS tag patterns between two zones.

        Args:
            zone_a: First zone
            zone_b: Second zone

        Returns:
            Float in [0, 1]. Returns 1.0 for exact match, 0.0 for no match.

        Tier 1 algorithm:
            - Extract POS tag sequences
            - Exact match comparison (all-or-nothing)
            - Returns 1.0 if patterns match exactly, 0.0 otherwise
        """
        if not zone_a or not zone_b:
            return 0.0

        # Extract POS tag sequences
        pos_pattern_a = [token.pos_tag for token in zone_a if token.pos_tag is not None]
        pos_pattern_b = [token.pos_tag for token in zone_b if token.pos_tag is not None]

        # Handle case where no valid POS tags
        if not pos_pattern_a or not pos_pattern_b:
            return 0.0

        # Tier 1: Exact match only
        # Convert to tuples for comparison
        if tuple(pos_pattern_a) == tuple(pos_pattern_b):
            return 1.0
        else:
            return 0.0

    def compare_syllable_counts(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """
        Compare syllable count similarity between two zones.

        Args:
            zone_a: First zone
            zone_b: Second zone

        Returns:
            Float in [0, 1] representing syllable count similarity.

        Tier 1 algorithm:
            - Sum total syllables in each zone
            - Calculate normalized similarity: 1 - (abs_diff / max_count)
            - Returns 0.0 if either zone has no syllable data
        """
        if not zone_a or not zone_b:
            return 0.0

        # Extract syllable counts
        syllables_a = [token.syllable_count for token in zone_a
                      if token.syllable_count is not None]
        syllables_b = [token.syllable_count for token in zone_b
                      if token.syllable_count is not None]

        # Handle case where no valid syllable counts
        if not syllables_a or not syllables_b:
            return 0.0

        # Calculate total syllables in each zone
        total_a = sum(syllables_a)
        total_b = sum(syllables_b)

        # Edge case: both zones have 0 syllables (shouldn't happen, but handle gracefully)
        if total_a == 0 and total_b == 0:
            return 1.0

        # Calculate similarity: 1 - (absolute_difference / max_count)
        max_count = max(total_a, total_b)
        if max_count == 0:
            return 0.0

        abs_diff = abs(total_a - total_b)
        similarity = 1.0 - (abs_diff / max_count)

        # Ensure output is in [0, 1] range
        return max(0.0, min(1.0, similarity))

    def detect_em_dashes(self, zone_a: List[Token], zone_b: List[Token]) -> Tuple[int, float]:
        """
        Detect em-dash usage in zones as an AI watermark indicator.

        AI models (especially GPT-4) overuse em-dashes in their writing.
        This method counts em-dashes and calculates a suspicion score.

        Args:
            zone_a: First zone
            zone_b: Second zone

        Returns:
            Tuple of (count, score):
            - count: Number of em-dashes found
            - score: Float in [0, 1] representing em-dash suspicion
                0.0-0.2: Low (0 em-dashes, human-typical)
                0.3-0.5: Moderate (1 em-dash)
                0.6-1.0: High (2+ em-dashes, AI-typical)

        Based on toolkit analysis:
        - Human writing: typically <0.3 em-dashes per sentence
        - AI writing (GPT-4): typically 0.5-1.0+ em-dashes per sentence

        Note: Em-dashes include both – (en-dash) and — (em-dash) Unicode chars
        """
        if not zone_a and not zone_b:
            return 0, 0.0

        # Combine zones for counting
        all_tokens = zone_a + zone_b
        
        # Count em-dashes (both en-dash – and em-dash —)
        em_dash_count = 0
        for token in all_tokens:
            if token.text in ('–', '—', '--'):
                em_dash_count += 1
        
        # Map count to suspicion score
        # 0 em-dashes: 0.0 (typical)
        # 1 em-dash: 0.4 (moderate)
        # 2+ em-dashes: 0.7-1.0 (high suspicion)
        if em_dash_count == 0:
            score = 0.0
        elif em_dash_count == 1:
            score = 0.4
        elif em_dash_count == 2:
            score = 0.7
        else:  # 3+
            score = min(1.0, 0.7 + (em_dash_count - 2) * 0.1)
        
        return em_dash_count, score


def quick_structural_analysis(zone_a: List[Token], zone_b: List[Token]) -> float:
    """
    Convenience function for quick structural analysis.

    Args:
        zone_a: First zone
        zone_b: Second zone

    Returns:
        Structural similarity score in [0, 1]
    """
    analyzer = StructuralEchoAnalyzer()
    return analyzer.analyze(zone_a, zone_b)
