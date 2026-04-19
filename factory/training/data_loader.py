"""
Data loader for Born Trader training.
Reads JSONL conversation files, formats them into training sequences,
tokenizes, and returns batches ready for the training loop.

Format expected: one JSON object per line with "conversation" field containing
a list of {"role": ..., "content": ...} messages.

Training objective: next-token prediction over the full sequence.
Loss is computed only on assistant turns (not system or user tokens).
"""

import json
import glob
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer


ROLE_TO_SPECIAL = {
    "system": "<sys>",
    "user": "<user>",
    "assistant": "<assistant>",
}

PAD_ID = 0   # <pad> is always token 0 in our tokenizer
EOS_ID = 3   # <eos> is token 3


def format_conversation(messages: list[dict]) -> str:
    """Convert a list of role/content dicts to a single training string."""
    parts = []
    for msg in messages:
        role_token = ROLE_TO_SPECIAL.get(msg["role"], "<user>")
        parts.append(f"{role_token} {msg['content'].strip()}")
    parts.append("<eos>")
    return " ".join(parts)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {e}")
    return records


class ConversationDataset(Dataset):
    def __init__(
        self,
        agent_dir: str,
        tokenizer_path: str,
        context_window: int = 2048,
        shuffle: bool = True,
    ):
        self.context_window = context_window
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        jsonl_files = sorted(glob.glob(str(Path(agent_dir) / "*.jsonl")))
        if not jsonl_files:
            raise FileNotFoundError(f"No .jsonl files in {agent_dir}")

        raw_records = []
        for path in jsonl_files:
            raw_records.extend(load_jsonl(path))

        if shuffle:
            random.shuffle(raw_records)

        self.sequences = []
        for record in raw_records:
            text = format_conversation(record["conversation"])
            ids = self.tokenizer.encode(text).ids
            # Truncate to context_window; keep as much as fits
            if len(ids) > context_window:
                ids = ids[:context_window]
            self.sequences.append(ids)

        print(
            f"Loaded {len(self.sequences)} conversations from {len(jsonl_files)} files. "
            f"Avg length: {sum(len(s) for s in self.sequences) // len(self.sequences)} tokens."
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        ids = self.sequences[idx]
        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        labels = torch.tensor(ids[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch: list[dict]) -> dict:
    """Pad sequences in a batch to the same length."""
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((len(batch), max_len), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)  # -100 = ignore in loss

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]

    return {"input_ids": input_ids, "labels": labels}


def make_dataloader(
    agent_dir: str,
    tokenizer_path: str,
    context_window: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    dataset = ConversationDataset(
        agent_dir=agent_dir,
        tokenizer_path=tokenizer_path,
        context_window=context_window,
        shuffle=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Run from trading-desk/ root
    root = Path(__file__).parent.parent.parent
    tokenizer_path = str(root / "factory" / "tokenizer" / "tokenizer.json")
    agent_dir = str(root / "agents" / "data_manager" / "training_data")

    loader = make_dataloader(
        agent_dir=agent_dir,
        tokenizer_path=tokenizer_path,
        context_window=2048,
        batch_size=2,
    )

    print(f"\nDataloader: {len(loader.dataset)} samples, {len(loader)} batches of size 2")
    print("\n--- Sample batch ---")
    batch = next(iter(loader))
    print(f"input_ids shape : {batch['input_ids'].shape}")
    print(f"labels shape    : {batch['labels'].shape}")
    print(f"input_ids sample: {batch['input_ids'][0, :20].tolist()}")
    print(f"labels sample   : {batch['labels'][0, :20].tolist()}")
