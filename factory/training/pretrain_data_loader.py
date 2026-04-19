"""
Data loader for Stage 1 and Stage 2 pretraining.
Reads plain .txt files, chunks into context_window token sequences,
returns batches for next-token prediction over all tokens (no masking).
"""

import glob
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

PAD_ID = 0


class PlainTextDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        context_window: int = 2048,
        max_tokens: int = None,
    ):
        self.context_window = context_window
        tokenizer = Tokenizer.from_file(tokenizer_path)

        txt_files = sorted(glob.glob(str(Path(data_dir) / "*.txt")))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files in {data_dir}")

        all_ids = []
        for path in txt_files:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
            ids = tokenizer.encode(text).ids
            all_ids.extend(ids)
            if max_tokens and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

        # Chunk into non-overlapping context_window blocks
        self.chunks = []
        for i in range(0, len(all_ids) - context_window, context_window):
            self.chunks.append(all_ids[i : i + context_window + 1])

        random.shuffle(self.chunks)
        print(
            f"Pretrain dataset: {len(all_ids):,} tokens → {len(self.chunks):,} chunks "
            f"of {context_window} tokens from {len(txt_files)} file(s)."
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        ids = self.chunks[idx]
        return {
            "input_ids": torch.tensor(ids[:-1], dtype=torch.long),
            "labels": torch.tensor(ids[1:], dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        n = item["input_ids"].shape[0]
        input_ids[i, :n] = item["input_ids"]
        labels[i, :n] = item["labels"]
    return {"input_ids": input_ids, "labels": labels}


def make_pretrain_dataloader(
    data_dir: str,
    tokenizer_path: str,
    context_window: int,
    batch_size: int,
    max_tokens: int = None,
) -> DataLoader:
    dataset = PlainTextDataset(
        data_dir=data_dir,
        tokenizer_path=tokenizer_path,
        context_window=context_window,
        max_tokens=max_tokens,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
