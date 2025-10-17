"""Download GloVe pre-trained embeddings for semantic analysis.

Downloads the GloVe 6B (Wikipedia + Gigaword) embeddings.
The 50d version is ~170MB, suitable for Tier 1 testing.
"""

import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for urllib downloads."""

    def update_to(self, b=1, bsize=1, tsize=None):
        """Update progress bar."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_with_progress(url: str, output_path: Path):
    """Download file with progress bar."""
    with DownloadProgressBar(
        unit='B',
        unit_scale=True,
        miniters=1,
        desc=output_path.name
    ) as t:
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to
        )


def main():
    """Download GloVe embeddings."""

    # Setup paths
    project_root = Path(__file__).parent.parent
    embeddings_dir = project_root / "data" / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    # GloVe download URL
    glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
    zip_path = embeddings_dir / "glove.6B.zip"

    print("=" * 80)
    print("DOWNLOADING GLOVE EMBEDDINGS")
    print("=" * 80)
    print()
    print(f"Source: {glove_url}")
    print(f"Destination: {zip_path}")
    print(f"Size: ~862 MB (compressed)")
    print()
    print("This contains 4 embedding sizes:")
    print("  - glove.6B.50d.txt  (~170 MB) - 50 dimensions")
    print("  - glove.6B.100d.txt (~350 MB) - 100 dimensions")
    print("  - glove.6B.200d.txt (~700 MB) - 200 dimensions")
    print("  - glove.6B.300d.txt (~1.0 GB) - 300 dimensions")
    print()
    print("We'll use the 50d version for Tier 1 testing.")
    print()

    # Check if already downloaded
    if zip_path.exists():
        print(f"[!] File already exists: {zip_path}")
        print(f"    Size: {zip_path.stat().st_size:,} bytes")
        print()
        response = input("Re-download? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping download.")
            return
        zip_path.unlink()

    # Download
    print("[*] Downloading... (this may take several minutes)")
    print()

    try:
        download_with_progress(glove_url, zip_path)
        print()
        print(f"[OK] Downloaded successfully: {zip_path}")
        print(f"     Size: {zip_path.stat().st_size:,} bytes")
        print()

        # Extract
        print("[*] Extracting embeddings...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(embeddings_dir)
        print("[OK] Extracted successfully")
        print()

        # List files
        print("Available embedding files:")
        for file in sorted(embeddings_dir.glob("*.txt")):
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name} ({size_mb:.1f} MB)")
        print()

        print("=" * 80)
        print("[OK] SETUP COMPLETE")
        print("=" * 80)
        print()
        print("To use with SemanticEchoAnalyzer:")
        print()
        print("  from specHO.echo_engine.semantic_analyzer import SemanticEchoAnalyzer")
        print()
        print("  # Use 50d for Tier 1 (fast, good for testing)")
        print("  analyzer = SemanticEchoAnalyzer(")
        print("      model_path='data/embeddings/glove.6B.50d.txt'")
        print("  )")
        print()

    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print()
        print("Alternative: Download manually from:")
        print(f"  {glove_url}")
        print(f"Extract to: {embeddings_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
