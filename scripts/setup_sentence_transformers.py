"""Download and setup Sentence Transformers for semantic analysis.

This script installs sentence-transformers and downloads the recommended model.
"""

import sys
import subprocess
from pathlib import Path


def install_sentence_transformers():
    """Install sentence-transformers package."""
    print("=" * 80)
    print("INSTALLING SENTENCE TRANSFORMERS")
    print("=" * 80)
    print()

    print("[*] Installing sentence-transformers package...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "sentence-transformers", "-q"
        ])
        print("[OK] Package installed successfully")
        print()
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Installation failed: {e}")
        sys.exit(1)


def download_model():
    """Download the sentence transformer model."""
    print("=" * 80)
    print("DOWNLOADING MODEL: all-MiniLM-L6-v2")
    print("=" * 80)
    print()
    print("Model details:")
    print("  - Size: ~80 MB")
    print("  - Dimensions: 384")
    print("  - Quality: Excellent (2023 SOTA)")
    print("  - Speed: Very fast")
    print("  - Context-aware: Yes")
    print()

    try:
        from sentence_transformers import SentenceTransformer

        print("[*] Downloading model (first time only)...")
        print("    Cache location: ~/.cache/torch/sentence_transformers/")
        print()

        model = SentenceTransformer('all-MiniLM-L6-v2')

        print("[OK] Model downloaded and loaded successfully")
        print()

        # Test the model
        print("[*] Testing model...")
        test_sentences = [
            "The cat sat on the mat",
            "A feline rested on the rug",
            "The dog ran in the park"
        ]

        embeddings = model.encode(test_sentences)
        print(f"[OK] Generated embeddings: {embeddings.shape}")
        print()

        # Show similarity
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        sim_matrix = cosine_similarity(embeddings)

        print("Similarity matrix (0=unrelated, 1=identical):")
        print()
        for i, sent1 in enumerate(test_sentences):
            for j, sent2 in enumerate(test_sentences):
                if i < j:
                    sim = sim_matrix[i][j]
                    print(f"  '{sent1[:30]}...'")
                    print(f"  '{sent2[:30]}...'")
                    print(f"  Similarity: {sim:.3f}")
                    print()

        print("=" * 80)
        print("[OK] SETUP COMPLETE")
        print("=" * 80)
        print()
        print("The model is ready to use!")
        print()
        print("Usage in SemanticEchoAnalyzer:")
        print()
        print("  from sentence_transformers import SentenceTransformer")
        print("  from specHO.echo_engine.semantic_analyzer import SemanticEchoAnalyzer")
        print()
        print("  # Load model")
        print("  st_model = SentenceTransformer('all-MiniLM-L6-v2')")
        print()
        print("  # For now, we'd need to modify SemanticEchoAnalyzer to accept")
        print("  # SentenceTransformer models (currently only supports gensim)")
        print()
        print("  # Alternative: Use for testing/comparison")
        print("  embeddings = st_model.encode(['word1', 'word2'])")
        print()

    except Exception as e:
        print(f"[ERROR] Model download failed: {e}")
        print()
        print("Troubleshooting:")
        print("  - Check internet connection")
        print("  - Ensure ~100MB free disk space")
        print("  - Try manual download from: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2")
        sys.exit(1)


def main():
    """Main setup function."""
    print()
    install_sentence_transformers()
    download_model()


if __name__ == "__main__":
    main()
