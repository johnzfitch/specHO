"""
Pair Rules Engine

Identifies thematic clause pairs using three rule types for echo analysis.
This is Task 3.2 of the SpecHO watermark detection system.

The PairRulesEngine applies three distinct pairing rules to identify "thematically related"
clause pairs that should be analyzed for echoes:

- **Rule A (Punctuation)**: Pairs separated by semicolon or em dash
- **Rule B (Conjunction)**: Pairs connected by coordinating conjunctions
- **Rule C (Transition)**: Pairs where second clause begins with transitional phrase

Tier 1 Implementation (Simple Rules):
- Rule A: Only semicolon (;) and em dash (—)
- Rule B: Only "but", "and", "or"
- Rule C: Only "However,", "Therefore,", "Thus," (case-sensitive, with comma)
- Simple deduplication by clause indices
- No confidence scoring or weighting
"""

import logging
import re
from typing import List, Set, Tuple

# Import our data models
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import Clause, ClausePair


class PairRulesEngine:
    """Identifies thematic clause pairs using punctuation, conjunction, and transition rules.

    The PairRulesEngine implements the three pairing rules defined in the Echo Rule
    watermark specification. These rules identify clause pairs that are likely to
    contain phonetic, structural, or semantic echoes.

    **Rule A - Punctuation Pairing**:
    Clauses separated by semicolon or em dash are thematically related. These
    punctuation marks signal a close relationship between clauses.
    - Trigger: `;` or `—` between clauses
    - Example: "The cat sat; the dog ran." → pair("cat sat", "dog ran")

    **Rule B - Conjunction Pairing**:
    Clauses joined by coordinating conjunctions show contrast or addition.
    - Trigger: "but", "and", "or" between clauses
    - Example: "The cat sat, and the dog ran." → pair("cat sat", "dog ran")

    **Rule C - Transition Pairing**:
    Clauses where the second starts with a transition word signal logical connection.
    - Trigger: "However,", "Therefore,", "Thus," at start of clause
    - Example: "The cat sat. However, the dog ran." → pair("cat sat", "dog ran")

    Tier 1 Limitations (deferred to Tier 2):
    - No confidence scoring
    - No rule weighting (all pairs treated equally)
    - No minimum clause length filtering
    - Basic deduplication only

    Attributes:
        None (stateless engine in Tier 1)

    Examples:
        >>> engine = PairRulesEngine()
        >>> # Assume we have clauses from BoundaryDetector
        >>> pairs = engine.apply_all_rules(clauses)
        >>> # Returns List[ClausePair]
    """

    # Tier 1 rule triggers (simple, no configuration)
    RULE_A_PUNCTUATION = {";", "—"}  # Semicolon and em dash
    RULE_B_CONJUNCTIONS = {"but", "and", "or"}  # Case-insensitive
    RULE_C_TRANSITIONS = {"However,", "Therefore,", "Thus,"}  # Case-sensitive with comma

    def __init__(self):
        """Initialize the PairRulesEngine.

        Tier 1 implementation requires no configuration.
        """
        logging.info("Initializing PairRulesEngine (Tier 1)")

    def apply_all_rules(self, clauses: List[Clause]) -> List[ClausePair]:
        """Apply all three pairing rules and return deduplicated pairs.

        This is the main entry point. It applies Rule A, Rule B, and Rule C
        in sequence, then deduplicates the results.

        Args:
            clauses: List of Clause objects from ClauseBoundaryDetector

        Returns:
            Deduplicated list of ClausePair objects

        Examples:
            >>> engine = PairRulesEngine()
            >>> clauses = detector.identify_clauses(doc, tokens)
            >>> pairs = engine.apply_all_rules(clauses)
            >>> len(pairs)  # Number of unique pairs
        """
        if not clauses or len(clauses) < 2:
            logging.warning("Insufficient clauses for pairing (need at least 2)")
            return []

        logging.debug(f"Applying pairing rules to {len(clauses)} clauses")

        # Apply each rule
        rule_a_pairs = self.apply_rule_a(clauses)
        rule_b_pairs = self.apply_rule_b(clauses)
        rule_c_pairs = self.apply_rule_c(clauses)

        # Combine and deduplicate
        all_pairs = rule_a_pairs + rule_b_pairs + rule_c_pairs
        deduplicated_pairs = self._deduplicate_pairs(all_pairs)

        logging.info(f"Generated {len(deduplicated_pairs)} unique clause pairs "
                    f"(Rule A: {len(rule_a_pairs)}, Rule B: {len(rule_b_pairs)}, Rule C: {len(rule_c_pairs)})")

        return deduplicated_pairs

    def apply_rule_a(self, clauses: List[Clause]) -> List[ClausePair]:
        """Apply Rule A: Pair clauses separated by semicolon or em dash.

        Rule A identifies clause pairs separated by strong punctuation marks
        that indicate thematic relationship.

        Tier 1 triggers: `;` (semicolon) and `—` (em dash)

        Algorithm:
        1. For each adjacent clause pair (i, i+1)
        2. Check if punctuation between them contains trigger
        3. If yes, create ClausePair

        Args:
            clauses: List of Clause objects sorted by position

        Returns:
            List of ClausePair objects from Rule A

        Examples:
            >>> # "The cat sat; the dog ran."
            >>> # Two clauses separated by semicolon
            >>> pairs = engine.apply_rule_a(clauses)
            >>> len(pairs)
            1
        """
        pairs = []

        for i in range(len(clauses) - 1):
            clause_a = clauses[i]
            clause_b = clauses[i + 1]

            # Check punctuation between clauses
            if self._has_punctuation_trigger(clause_a, clause_b):
                pair = ClausePair(
                    clause_a=clause_a,
                    clause_b=clause_b,
                    zone_a_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    zone_b_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    pair_type="punctuation"
                )
                pairs.append(pair)
                logging.debug(f"Rule A match: clause {i} <--> clause {i+1}")

        logging.debug(f"Rule A generated {len(pairs)} pairs")
        return pairs

    def apply_rule_b(self, clauses: List[Clause]) -> List[ClausePair]:
        """Apply Rule B: Pair clauses connected by coordinating conjunction.

        Rule B identifies clause pairs joined by coordinating conjunctions
        (and, but, or) that signal addition, contrast, or choice.

        Tier 1 triggers: "but", "and", "or" (case-insensitive)

        Algorithm:
        1. For each adjacent clause pair (i, i+1)
        2. Check if conjunction word appears between them
        3. If yes, create ClausePair

        Args:
            clauses: List of Clause objects sorted by position

        Returns:
            List of ClausePair objects from Rule B

        Examples:
            >>> # "The cat sat, and the dog ran."
            >>> # Two clauses connected by "and"
            >>> pairs = engine.apply_rule_b(clauses)
            >>> len(pairs)
            1
        """
        pairs = []

        for i in range(len(clauses) - 1):
            clause_a = clauses[i]
            clause_b = clauses[i + 1]

            # Check for conjunction between clauses
            if self._has_conjunction_trigger(clause_a, clause_b):
                pair = ClausePair(
                    clause_a=clause_a,
                    clause_b=clause_b,
                    zone_a_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    zone_b_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    pair_type="conjunction"
                )
                pairs.append(pair)
                logging.debug(f"Rule B match: clause {i} <--> clause {i+1}")

        logging.debug(f"Rule B generated {len(pairs)} pairs")
        return pairs

    def apply_rule_c(self, clauses: List[Clause]) -> List[ClausePair]:
        """Apply Rule C: Pair clauses where second begins with transition word.

        Rule C identifies clause pairs where the second clause begins with
        a transitional phrase that signals logical relationship.

        Tier 1 triggers: "However,", "Therefore,", "Thus," (exact match with comma)

        Algorithm:
        1. For each clause starting at position 1
        2. Check if it begins with a transition trigger
        3. If yes, pair with previous clause

        Args:
            clauses: List of Clause objects sorted by position

        Returns:
            List of ClausePair objects from Rule C

        Examples:
            >>> # "The cat sat. However, the dog ran."
            >>> # Second clause starts with "However,"
            >>> pairs = engine.apply_rule_c(clauses)
            >>> len(pairs)
            1
        """
        pairs = []

        for i in range(1, len(clauses)):
            clause_a = clauses[i - 1]
            clause_b = clauses[i]

            # Check if clause_b starts with transition word
            if self._has_transition_trigger(clause_b):
                pair = ClausePair(
                    clause_a=clause_a,
                    clause_b=clause_b,
                    zone_a_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    zone_b_tokens=[],  # Will be populated by ZoneExtractor (Task 3.3)
                    pair_type="transition"
                )
                pairs.append(pair)
                logging.debug(f"Rule C match: clause {i-1} <--> clause {i}")

        logging.debug(f"Rule C generated {len(pairs)} pairs")
        return pairs

    def _has_punctuation_trigger(self, clause_a: Clause, clause_b: Clause) -> bool:
        """Check if semicolon or em dash appears between two clauses.

        Looks at the tokens between clause_a's end and clause_b's start
        for punctuation triggers.

        Args:
            clause_a: First clause
            clause_b: Second clause

        Returns:
            True if punctuation trigger found between clauses
        """
        # Check last token of clause_a for semicolon or em dash
        if clause_a.tokens:
            last_token_text = clause_a.tokens[-1].text
            if last_token_text in self.RULE_A_PUNCTUATION:
                return True

        # Check first token of clause_b for em dash (less common but possible)
        if clause_b.tokens:
            first_token_text = clause_b.tokens[0].text
            if first_token_text in self.RULE_A_PUNCTUATION:
                return True

        return False

    def _has_conjunction_trigger(self, clause_a: Clause, clause_b: Clause) -> bool:
        """Check if coordinating conjunction appears between two clauses.

        Looks for "but", "and", "or" (case-insensitive) in the boundary
        region between clauses.

        Args:
            clause_a: First clause
            clause_b: Second clause

        Returns:
            True if conjunction trigger found between clauses
        """
        # Check last few tokens of clause_a for conjunction
        # Typically: "clause , and other-clause"
        if len(clause_a.tokens) >= 2:
            # Check last 2 tokens
            for i in range(max(0, len(clause_a.tokens) - 2), len(clause_a.tokens)):
                token_text = clause_a.tokens[i].text.lower()
                if token_text in self.RULE_B_CONJUNCTIONS:
                    return True

        # Check first few tokens of clause_b for conjunction
        if len(clause_b.tokens) >= 2:
            # Check first 2 tokens
            for i in range(min(2, len(clause_b.tokens))):
                token_text = clause_b.tokens[i].text.lower()
                if token_text in self.RULE_B_CONJUNCTIONS:
                    return True

        return False

    def _has_transition_trigger(self, clause: Clause) -> bool:
        """Check if clause begins with a transition word.

        Looks for "However,", "Therefore,", "Thus," (case-sensitive with comma)
        at the start of the clause.

        Args:
            clause: Clause to check

        Returns:
            True if clause starts with transition trigger
        """
        if not clause.tokens or len(clause.tokens) < 2:
            return False

        # Check first token + second token for "However," pattern
        # First token should be transition word, second should be comma
        first_token = clause.tokens[0].text

        # Option 1: "However," as single token
        if first_token in self.RULE_C_TRANSITIONS:
            return True

        # Option 2: "However" + "," as two tokens
        if len(clause.tokens) >= 2:
            first_word = first_token
            second_token = clause.tokens[1].text
            combined = f"{first_word}{second_token}"
            if combined in self.RULE_C_TRANSITIONS:
                return True

        return False

    def _deduplicate_pairs(self, pairs: List[ClausePair]) -> List[ClausePair]:
        """Remove duplicate clause pairs based on clause indices.

        Tier 1 deduplication: Simple set-based deduplication using
        (start_idx_a, end_idx_a, start_idx_b, end_idx_b) as key.

        Args:
            pairs: List of ClausePair objects (may contain duplicates)

        Returns:
            Deduplicated list of ClausePair objects

        Examples:
            >>> # If Rule A and Rule B both identify same pair
            >>> # Only one copy is returned
        """
        seen = set()
        deduplicated = []

        for pair in pairs:
            # Create key from clause indices
            key = (
                pair.clause_a.start_idx,
                pair.clause_a.end_idx,
                pair.clause_b.start_idx,
                pair.clause_b.end_idx
            )

            if key not in seen:
                seen.add(key)
                deduplicated.append(pair)
            else:
                logging.debug(f"Duplicate pair removed: {key}")

        return deduplicated


# Convenience function for quick pairing
def quick_apply_rules(clauses: List[Clause]) -> List[ClausePair]:
    """Convenience function for one-off clause pairing.

    Args:
        clauses: List of Clause objects

    Returns:
        List of ClausePair objects

    Examples:
        >>> pairs = quick_apply_rules(clauses)
    """
    engine = PairRulesEngine()
    return engine.apply_all_rules(clauses)
