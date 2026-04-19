"""
Stage 1 corpus prep — downloads WikiText-103 and extracts a clean
50MB subset of factual English text for pretraining.

Run: python factory/training/prepare_stage1_corpus.py

Output: factory/corpus/stage1_english/wiki_clean.txt
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = ROOT / "factory" / "corpus" / "stage1_english"
OUTPUT_FILE = OUTPUT_DIR / "wiki_clean.txt"
TARGET_BYTES = 50 * 1024 * 1024  # 50MB


def clean_wiki_text(text: str) -> str:
    text = re.sub(r"= = =.*?= = =", "", text)   # section headers
    text = re.sub(r"= =.*?= =", "", text)
    text = re.sub(r"=.*?=", "", text)
    text = re.sub(r"<unk>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading WikiText-103 from HuggingFace datasets...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets library...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", trust_remote_code=True)

    print(f"Dataset loaded: {len(ds):,} rows")
    print(f"Writing clean text to {OUTPUT_FILE} (target: 50MB)...")

    written = 0
    skipped = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if not text or len(text) < 100:
                skipped += 1
                continue
            cleaned = clean_wiki_text(text)
            if len(cleaned) < 80:
                skipped += 1
                continue
            f.write(cleaned + "\n")
            written += len(cleaned.encode("utf-8"))
            if written >= TARGET_BYTES:
                break

    size_mb = written / (1024 * 1024)
    print(f"\nDone.")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Size   : {size_mb:.1f} MB")
    print(f"  Rows   : {written // 200:,} approx paragraphs")
    print(f"  Skipped: {skipped:,} short/empty rows")


if __name__ == "__main__":
    main()
