"""
Main training script for Born Traders.

Usage:
    python factory/training/train.py --agent data_manager

Reads:
    agents/<agent>/config.yaml          (agent-specific overrides)
    factory/model/config.yaml           (model architecture)
    factory/training/config.yaml        (training hyperparameters)
    factory/tokenizer/tokenizer.json    (tokenizer)
    agents/<agent>/training_data/*.jsonl (training conversations)

Writes:
    agents/<agent>/weights/epoch_N.pt   (checkpoint per epoch)
    agents/<agent>/weights/best.pt      (best checkpoint by loss)
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parent.parent.parent  # trading-desk/


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(step: int, base_lr: float, warmup_steps: int) -> float:
    """Linear warmup then constant."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr


def train(agent: str):
    # ── Paths ──────────────────────────────────────────────────────────────
    agent_dir = ROOT / "agents" / agent
    weights_dir = agent_dir / "weights"
    weights_dir.mkdir(exist_ok=True)

    model_cfg_path = ROOT / "factory" / "model" / "config.yaml"
    train_cfg_path = ROOT / "factory" / "training" / "config.yaml"
    tokenizer_path = ROOT / "factory" / "tokenizer" / "tokenizer.json"
    training_data_dir = agent_dir / "training_data"

    # Agent-level config can override training hyperparameters
    agent_cfg_path = agent_dir / "config.yaml"

    # ── Load configs ───────────────────────────────────────────────────────
    train_cfg = load_config(str(train_cfg_path))
    if agent_cfg_path.exists():
        agent_overrides = load_config(str(agent_cfg_path))
        train_cfg.update(agent_overrides.get("training", {}))

    epochs = train_cfg["epochs"]
    batch_size = train_cfg["batch_size"]
    lr = train_cfg["learning_rate"]
    warmup_steps = train_cfg["warmup_steps"]
    weight_decay = train_cfg["weight_decay"]
    grad_clip = train_cfg["gradient_clipping"]
    save_every = train_cfg["save_every_n_epochs"]
    log_every = train_cfg["log_every_n_steps"]
    mixed_precision = train_cfg["mixed_precision"]

    # ── Import model and data loader ───────────────────────────────────────
    import sys
    sys.path.insert(0, str(ROOT))
    from factory.model.model import BornTrader, ModelConfig
    from factory.training.data_loader import make_dataloader

    # ── Device ─────────────────────────────────────────────────────────────
    device = get_device()
    print(f"Device: {device}")

    # ── Model ──────────────────────────────────────────────────────────────
    model_cfg = ModelConfig.from_yaml(str(model_cfg_path))
    model = BornTrader(model_cfg).to(device)
    param_count = model.count_parameters()
    print(f"Model: {param_count:,} parameters")

    # ── Data ───────────────────────────────────────────────────────────────
    loader = make_dataloader(
        agent_dir=str(training_data_dir),
        tokenizer_path=str(tokenizer_path),
        context_window=model_cfg.context_window,
        batch_size=batch_size,
        shuffle=True,
    )
    print(f"Data: {len(loader.dataset)} conversations, {len(loader)} batches/epoch")

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    # ── Mixed precision ────────────────────────────────────────────────────
    dtype = torch.bfloat16 if mixed_precision and device.type in ("mps", "cuda") else torch.float32
    use_amp = mixed_precision and device.type in ("mps", "cuda")

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\nStarting training — {epochs} epochs, batch_size={batch_size}, lr={lr}")
    print("=" * 60)

    global_step = 0
    best_loss = float("inf")
    loss_history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(loader, 1):
            global_step += 1

            # LR warmup
            current_lr = get_lr(global_step, lr, warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = current_lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type=device.type, dtype=dtype):
                    logits = model(input_ids)
                    loss = nn.functional.cross_entropy(
                        logits.reshape(-1, model_cfg.vocab_size),
                        labels.reshape(-1),
                        ignore_index=-100,
                    )
            else:
                logits = model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, model_cfg.vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()

            if step % log_every == 0:
                avg = epoch_loss / step
                print(
                    f"  Epoch {epoch}/{epochs} | Step {step}/{len(loader)} | "
                    f"Loss {loss.item():.4f} | Avg {avg:.4f} | LR {current_lr:.2e}"
                )

        epoch_loss /= len(loader)
        elapsed = time.time() - epoch_start
        loss_history.append(epoch_loss)
        print(f"Epoch {epoch}/{epochs} complete — Loss: {epoch_loss:.4f} — Time: {elapsed:.1f}s")

        # Save checkpoint
        if epoch % save_every == 0:
            ckpt_path = weights_dir / f"epoch_{epoch}.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "loss": epoch_loss}, ckpt_path)
            print(f"  Saved: {ckpt_path}")

        # Save best
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = weights_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "loss": epoch_loss}, best_path)
            print(f"  New best loss: {best_loss:.4f} — saved to best.pt")

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Agent       : {agent}")
    print(f"  Epochs      : {epochs}")
    print(f"  Final loss  : {loss_history[-1]:.4f}")
    print(f"  Best loss   : {best_loss:.4f}")
    print(f"  Loss curve  : {[round(l, 4) for l in loss_history]}")
    print(f"  Weights     : {weights_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True, help="Agent name, e.g. data_manager")
    args = parser.parse_args()
    train(args.agent)
