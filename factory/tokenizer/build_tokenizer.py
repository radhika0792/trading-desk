"""
Build a BPE tokenizer for the Data Manager Born Trader.
Target vocab size: 8,000 - 10,000 tokens.

Usage:
    python factory/tokenizer/build_tokenizer.py

Output:
    factory/tokenizer/tokenizer.json
"""

import os
import glob
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

CORPUS_DIR = Path(__file__).parent / "corpus"
OUTPUT_PATH = Path(__file__).parent / "tokenizer.json"
VOCAB_SIZE = 9000

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sys>", "<user>", "<assistant>"]

VALIDATION_SENTENCES = [
    "Morning data download complete. All 50 stocks clean, no gaps.",
    "RELIANCE intraday ready. 3 days of 15-minute candles. 4,500 data points.",
    "Fyers API is not responding. Tried twice. 12 stocks missing.",
    "That timeframe is not available through our broker. Daily and above only.",
    "Market was closed yesterday — Diwali holiday. No new data.",
    "TATASTEEL 52-week high is 912. HDFCBANK delivery percentage 52%.",
    "NIFTY closed at 22,450. Bank NIFTY at 48,200.",
    "Brent crude at $82.40. Gold MCX at 72,400. USDINR at 83.45.",
    "Q1 FY26 earnings for INFY, TCS, WIPRO stored.",
    "Data Manager ready. All systems operational.",
]

SINGLE_TOKEN_CHECKS = [
    "TATASTEEL", "HDFCBANK", "NIFTY", "BANKNIFTY", "RELIANCE",
    "BAJFINANCE", "GIFTNIFTY", "USDINR", "OHLCV", "FY26",
]


def collect_corpus_files():
    files = sorted(glob.glob(str(CORPUS_DIR / "*.txt")))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {CORPUS_DIR}")
    print(f"Found {len(files)} corpus files:")
    for f in files:
        print(f"  {Path(f).name}")
    return files


def build(vocab_size: int = VOCAB_SIZE):
    corpus_files = collect_corpus_files()

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=1,
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

    print("\n--- Single-token checks (should be one token each) ---")
    all_pass = True
    for symbol in SINGLE_TOKEN_CHECKS:
        enc = tokenizer.encode(symbol)
        token_count = len(enc.ids)
        status = "PASS" if token_count == 1 else "WARN"
        if token_count != 1:
            all_pass = False
        tokens = enc.tokens
        print(f"  [{status}] '{symbol}' → {tokens} ({token_count} token(s))")

    if not all_pass:
        print("\n  NOTE: Some symbols split into sub-tokens. Consider increasing vocab_size.")

    print("\n--- Sample tokenizations ---")
    for sentence in VALIDATION_SENTENCES:
        enc = tokenizer.encode(sentence)
        print(f"\n  Input : {sentence}")
        print(f"  Tokens: {enc.tokens}")
        print(f"  Count : {len(enc.ids)}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    tokenizer = build()
    validate(tokenizer)
