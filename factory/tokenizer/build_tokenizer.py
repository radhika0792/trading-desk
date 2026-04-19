"""
Build a BPE tokenizer for Born Traders.

Sources (in priority order):
  1. factory/corpus/stage1_english/  — WikiText-103 general English (50MB)
  2. factory/corpus/stage2_financial/ — financial news corpus (when available)
  3. factory/tokenizer/corpus/        — our hand-curated financial corpus (11 files)

Combined corpus gives the tokenizer broad English coverage + financial domain terms.
Target vocab: 16,000 tokens (up from 9,000 — more headroom for general English).

Usage:
    python factory/tokenizer/build_tokenizer.py

Output:
    factory/tokenizer/tokenizer.json
"""

import glob
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

ROOT = Path(__file__).parent.parent.parent
CURATED_CORPUS_DIR = Path(__file__).parent / "corpus"
STAGE1_CORPUS_DIR = ROOT / "factory" / "corpus" / "stage1_english"
STAGE2_CORPUS_DIR = ROOT / "factory" / "corpus" / "stage2_financial"
OUTPUT_PATH = Path(__file__).parent / "tokenizer.json"

VOCAB_SIZE = 16000

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sys>", "<user>", "<assistant>"]

VALIDATION_SENTENCES = [
    "Morning data download complete. All 50 stocks clean, no gaps.",
    "RELIANCE intraday ready. 3 days of 15-minute candles. 4,500 data points.",
    "Fyers API is not responding. Tried twice. 12 stocks missing.",
    "The stock market closed higher today because investors were optimistic.",
    "There are three reasons why oil prices fell this quarter.",
    "FII sold 2,000 crore yesterday, which means institutional sentiment is bearish.",
    "NIFTY closed at 22,450. Bank NIFTY at 48,200.",
    "Brent crude at $82.40. Gold MCX at 72,400. USDINR at 83.45.",
    "Q1 FY26 earnings for INFY, TCS, WIPRO stored.",
    "Data Manager ready. All systems operational.",
]

SINGLE_TOKEN_CHECKS = [
    "TATASTEEL", "HDFCBANK", "NIFTY", "BANKNIFTY", "RELIANCE",
    "BAJFINANCE", "GIFTNIFTY", "USDINR", "OHLCV", "FY26",
    "the", "and", "that", "market", "stock", "price",
]


def collect_corpus_files():
    files = []

    # Stage 1 — general English (largest, first for BPE priority)
    if STAGE1_CORPUS_DIR.exists():
        s1_files = sorted(glob.glob(str(STAGE1_CORPUS_DIR / "*.txt")))
        files.extend(s1_files)
        print(f"Stage 1 English corpus : {len(s1_files)} file(s) — {STAGE1_CORPUS_DIR}")
    else:
        print(f"Stage 1 corpus NOT FOUND at {STAGE1_CORPUS_DIR} — skipping")

    # Stage 2 — financial news (optional, include if available)
    if STAGE2_CORPUS_DIR.exists():
        s2_files = sorted(glob.glob(str(STAGE2_CORPUS_DIR / "*.txt")))
        files.extend(s2_files)
        print(f"Stage 2 Financial corpus: {len(s2_files)} file(s) — {STAGE2_CORPUS_DIR}")

    # Curated financial corpus — our 11 hand-built files
    curated_files = sorted(glob.glob(str(CURATED_CORPUS_DIR / "*.txt")))
    files.extend(curated_files)
    print(f"Curated corpus         : {len(curated_files)} file(s) — {CURATED_CORPUS_DIR}")

    if not files:
        raise FileNotFoundError("No corpus files found. Run prepare_stage1_corpus.py first.")

    print(f"\nTotal corpus files: {len(files)}")
    return files


def build(vocab_size: int = VOCAB_SIZE):
    corpus_files = collect_corpus_files()

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    print(f"\nTraining BPE tokenizer — target vocab size: {vocab_size}")
    tokenizer.train(corpus_files, trainer)

    tokenizer.save(str(OUTPUT_PATH))
    print(f"\nTokenizer saved to {OUTPUT_PATH}")
    return tokenizer


def validate(tokenizer: Tokenizer):
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    actual_vocab = tokenizer.get_vocab_size()
    print(f"\nVocabulary size: {actual_vocab}")

    print("\n--- Token checks ---")
    for symbol in SINGLE_TOKEN_CHECKS:
        enc = tokenizer.encode(symbol)
        status = "PASS" if len(enc.ids) == 1 else "SPLIT"
        print(f"  [{status}] '{symbol}' → {enc.tokens}")

    print("\n--- Sample tokenizations ---")
    for sentence in VALIDATION_SENTENCES:
        enc = tokenizer.encode(sentence)
        print(f"\n  Input : {sentence}")
        print(f"  Tokens: {enc.tokens[:12]}{'...' if len(enc.tokens) > 12 else ''}")
        print(f"  Count : {len(enc.ids)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    tokenizer = build()
    validate(tokenizer)
