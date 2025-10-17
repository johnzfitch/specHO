"""Semantic echo analysis using word embeddings.

This module implements the semantic dimension of the Echo Rule watermark
detection. It measures semantic similarity between clause zones using
word embeddings (Word2Vec, GloVe via gensim, or Sentence Transformers).

Tier: 1 (MVP)
Task: 4.3
Dependencies: Task 1.1 (Token dataclass)
"""

from typing import List, Union
import numpy as np
from specHO.models import Token


class SemanticEchoAnalyzer:
    """Analyzes semantic similarity between clause zones using word embeddings.

    Uses embeddings to represent each zone, then calculates cosine similarity.
    This captures whether the zones express semantically related concepts, even
    when phonetic and structural similarities are low.

    Tier 1 Implementation:
    - Pre-trained embeddings via gensim (Word2Vec/GloVe) or Sentence Transformers
    - Mean pooling across zone tokens (for gensim models)
    - Direct sentence encoding (for Sentence Transformers)
    - Cosine similarity mapped to [0,1] range
    - Fallback to 0.5 (neutral) if embeddings unavailable

    Attributes:
        model: Embedding model (gensim KeyedVectors or SentenceTransformer)
               None if embeddings could not be loaded
        model_type: Type of model ('gensim' or 'sentence_transformer')
    """

    def __init__(self, model_path: str = None):
        """Initialize semantic analyzer with word embeddings.

        Args:
            model_path: Path to gensim-compatible embeddings file, OR
                       name of Sentence Transformer model (e.g., 'all-MiniLM-L6-v2')
                       If None, operates in fallback mode (returns 0.5)
        """
        self.model = None
        self.model_type = None

        if model_path:
            # Try Sentence Transformers first (if model_path looks like a model name)
            if '/' not in model_path and '\\' not in model_path and not model_path.endswith('.txt'):
                try:
                    from sentence_transformers import SentenceTransformer
                    self.model = SentenceTransformer(model_path)
                    self.model_type = 'sentence_transformer'
                    return
                except Exception:
                    pass  # Fall through to try gensim

            # Try gensim KeyedVectors (for file paths)
            try:
                from gensim.models import KeyedVectors
                self.model = KeyedVectors.load_word2vec_format(
                    model_path,
                    binary=False
                )
                self.model_type = 'gensim'
            except Exception:
                # Fallback mode: embeddings unavailable
                self.model = None
                self.model_type = None

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
        """Compute embedding vector for a zone.

        For gensim models: Uses mean-pooled word embeddings.
        For Sentence Transformers: Encodes the full text sequence.

        Args:
            zone: List of tokens to embed

        Returns:
            Embedding vector, or None if no tokens have embeddings
        """
        if self.model_type == 'sentence_transformer':
            # Use Sentence Transformer to encode the full text
            text = ' '.join(token.text for token in zone)
            if not text.strip():
                return None
            return self.model.encode(text, convert_to_numpy=True)

        elif self.model_type == 'gensim':
            # Use gensim word embeddings with mean pooling
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

        else:
            # Unknown model type
            return None

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
