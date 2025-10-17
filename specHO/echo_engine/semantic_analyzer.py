"""Semantic echo analysis using word embeddings.

This module implements the semantic dimension of the Echo Rule watermark
detection. It measures semantic similarity between clause zones using
word embeddings (Word2Vec or GloVe via gensim).

Tier: 1 (MVP)
Task: 4.3
Dependencies: Task 1.1 (Token dataclass)
"""

from typing import List
import numpy as np
from SpecHO.models import Token


class SemanticEchoAnalyzer:
    """Analyzes semantic similarity between clause zones using word embeddings.

    Uses mean-pooled word embeddings (Word2Vec or GloVe) to represent each zone
    as a single vector, then calculates cosine similarity between the zone vectors.
    This captures whether the zones express semantically related concepts, even
    when phonetic and structural similarities are low.

    Tier 1 Implementation:
    - Pre-trained Word2Vec or GloVe embeddings via gensim
    - Mean pooling across zone tokens
    - Cosine similarity mapped to [0,1] range
    - Fallback to 0.5 (neutral) if embeddings unavailable

    Attributes:
        model: Gensim KeyedVectors model (Word2Vec or GloVe)
               None if embeddings could not be loaded
    """

    def __init__(self, model_path: str = None):
        """Initialize semantic analyzer with word embeddings.

        Args:
            model_path: Path to gensim-compatible embeddings file
                       If None, operates in fallback mode (returns 0.5)
        """
        self.model = None

        if model_path:
            try:
                from gensim.models import KeyedVectors
                self.model = KeyedVectors.load_word2vec_format(
                    model_path,
                    binary=False
                )
            except Exception:
                # Fallback mode: embeddings unavailable
                self.model = None

    def analyze(self, zone_a: List[Token], zone_b: List[Token]) -> float:
        """Calculate semantic similarity between two clause zones.

        Computes mean-pooled embeddings for each zone and returns cosine
        similarity mapped to [0,1] range. Falls back to 0.5 if embeddings
        are unavailable or zones are empty.

        Args:
            zone_a: List of tokens from terminal zone (clause A)
            zone_b: List of tokens from initial zone (clause B)

        Returns:
            Float in [0.0, 1.0] representing semantic similarity:
            - 0.0: semantically opposite or completely unrelated
            - 0.5: neutral (no clear semantic relationship) or fallback
            - 1.0: semantically identical or highly related

        Edge Cases:
            - Empty zones: returns 0.0
            - No embeddings available: returns 0.5 (neutral)
            - Tokens not in vocabulary: skipped (uses available tokens only)
            - All tokens OOV: returns 0.5 (neutral)
        """
        # Edge case: empty zones
        if not zone_a or not zone_b:
            return 0.0

        # Fallback: no embeddings available
        if self.model is None:
            return 0.5

        # Get zone vectors
        vec_a = self._get_zone_vector(zone_a)
        vec_b = self._get_zone_vector(zone_b)

        # Handle cases where no embeddings found
        if vec_a is None or vec_b is None:
            return 0.5

        # Calculate cosine similarity
        similarity = self._calculate_cosine_similarity(vec_a, vec_b)

        return similarity

    def _get_zone_vector(self, zone: List[Token]) -> np.ndarray:
        """Compute mean-pooled embedding vector for a zone.

        Args:
            zone: List of tokens to embed

        Returns:
            Mean-pooled embedding vector, or None if no tokens have embeddings
        """
        vectors = []

        for token in zone:
            # Get lowercase version for embedding lookup
            word = token.text.lower()

            try:
                if word in self.model:
                    vectors.append(self.model[word])
            except Exception:
                # Token not in vocabulary, skip it
                continue

        # No embeddings found
        if not vectors:
            return None

        # Mean pooling
        return np.mean(vectors, axis=0)

    def _calculate_cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray
    ) -> float:
        """Calculate cosine similarity and map to [0,1] range.

        Args:
            vec_a: Embedding vector for zone A
            vec_b: Embedding vector for zone B

        Returns:
            Similarity score in [0,1] range
        """
        # Cosine similarity formula: dot(a,b) / (||a|| * ||b||)
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.5

        cosine_sim = dot_product / (norm_a * norm_b)

        # Map from [-1, 1] to [0, 1]: (1 + cos) / 2
        similarity = (1.0 + cosine_sim) / 2.0

        # Clip to [0,1] range (handle floating point errors)
        return np.clip(similarity, 0.0, 1.0)
