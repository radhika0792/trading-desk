"""
Main training script for Born Traders.

Modes:
  pretrain  — plain text next-token prediction (Stage 1, Stage 2)
  finetune  — conversation JSONL fine-tuning (Stage 3 and beyond)

Usage:
  Stage 1: python factory/training/train.py --mode pretrain --data factory/corpus/stage1_english --epochs 5 --lr 1e-4 --save stage1_english.pt
  Stage 2: python factory/training/train.py --mode pretrain --data factory/corpus/stage2_financial --epochs 5 --lr 5e-5 --resume factory/checkpoints/stage1_english.pt --save stage2_financial.pt
  Stage 3: python factory/training/train.py --mode finetune --agent data_manager --epochs 15 --lr 2e-5 --resume factory/checkpoints/stage2_financial.pt --save stage3_datamanager.pt
"""

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).parent.parent.parent  # trading-desk/
CHECKPOINTS_DIR = ROOT / "factory" / "checkpoints"


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
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    return base_lr


def train(args):
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    model_cfg_path = ROOT / "factory" / "model" / "config.yaml"
    tokenizer_path = ROOT / "factory" / "tokenizer" / "tokenizer.json"
    train_cfg_path = ROOT / "factory" / "training" / "config.yaml"

    train_cfg = load_config(str(train_cfg_path))

    # CLI args override config file
    epochs = args.epochs if args.epochs else train_cfg["epochs"]
    batch_size = args.batch_size if args.batch_size else train_cfg["batch_size"]
    lr = args.lr if args.lr else train_cfg["learning_rate"]
    warmup_steps = train_cfg["warmup_steps"]
    weight_decay = train_cfg["weight_decay"]
    grad_clip = train_cfg["gradient_clipping"]
    log_every = train_cfg["log_every_n_steps"]

    import sys
    sys.path.insert(0, str(ROOT))
    from factory.model.model import BornTrader, ModelConfig

    device = get_device()
    print(f"Device: {device}")

    model_cfg = ModelConfig.from_yaml(str(model_cfg_path))
    model = BornTrader(model_cfg, gradient_checkpointing=args.grad_checkpoint).to(device)
    print(f"Model: {model.count_parameters():,} parameters" + (" [gradient checkpointing ON]" if args.grad_checkpoint else ""))

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = CHECKPOINTS_DIR / args.resume
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
            print(f"Resumed from {resume_path} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

    # Data loader
    if args.mode == "pretrain":
        from factory.training.pretrain_data_loader import make_pretrain_dataloader
        data_dir = args.data if args.data else str(ROOT / "factory" / "corpus" / "stage1_english")
        loader = make_pretrain_dataloader(
            data_dir=data_dir,
            tokenizer_path=str(tokenizer_path),
            context_window=model_cfg.context_window,
            batch_size=batch_size,
            max_tokens=args.max_tokens,
        )
    else:  # finetune
        if not args.agent:
            raise ValueError("--agent required for finetune mode")
        from factory.training.data_loader import make_dataloader
        agent_dir = ROOT / "agents" / args.agent
        training_data_dir = agent_dir / "training_data"

        agent_cfg_path = agent_dir / "config.yaml"
        if agent_cfg_path.exists():
            agent_overrides = load_config(str(agent_cfg_path))
            train_cfg.update(agent_overrides.get("training", {}))
            if not args.epochs:
                epochs = train_cfg["epochs"]

        loader = make_dataloader(
            agent_dir=str(training_data_dir),
            tokenizer_path=str(tokenizer_path),
            context_window=model_cfg.context_window,
            batch_size=batch_size,
            shuffle=True,
        )

    print(f"Data: {len(loader.dataset)} samples, {len(loader)} batches/epoch")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )

    use_amp = device.type in ("cuda", "mps")
    dtype = torch.bfloat16 if use_amp else torch.float32

    print(f"\nStarting {args.mode} — {epochs} epochs, batch_size={batch_size}, lr={lr}")
    if args.resume:
        print(f"  Loaded from : {args.resume}")
    print(f"  Save to     : {args.save}")
    print("=" * 60)

    global_step = 0
    best_loss = float("inf")
    loss_history = []

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, batch in enumerate(loader, 1):
            global_step += 1

            current_lr = get_lr(global_step, lr, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

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

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = CHECKPOINTS_DIR / args.save
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "loss": epoch_loss}, save_path)
            print(f"  New best — saved to {save_path}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Mode        : {args.mode}")
    print(f"  Epochs      : {epochs}")
    print(f"  Final loss  : {loss_history[-1]:.4f}")
    print(f"  Best loss   : {best_loss:.4f}")
    print(f"  Loss curve  : {[round(l, 4) for l in loss_history]}")
    print(f"  Checkpoint  : {CHECKPOINTS_DIR / args.save}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--agent", help="Agent name (finetune mode only)")
    parser.add_argument("--data", help="Path to plain text corpus dir (pretrain mode only)")
    parser.add_argument("--resume", help="Checkpoint filename to load from factory/checkpoints/")
    parser.add_argument("--save", required=True, help="Checkpoint filename to save e.g. stage1_english.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    parser.add_argument("--max-tokens", type=int, default=None, dest="max_tokens",
                        help="Cap tokens loaded (pretrain) e.g. 10000000 for 10M")
    parser.add_argument("--grad-checkpoint", action="store_true", dest="grad_checkpoint",
                        help="Enable gradient checkpointing to reduce VRAM at cost of speed")
    args = parser.parse_args()
    train(args)
