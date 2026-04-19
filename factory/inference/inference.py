"""
Inference pipeline for Born Traders.
Load a trained model, accept text input, produce text output.

Usage:
    from factory.inference.inference import BornTraderLLM
    model = BornTraderLLM("agents/data_manager/weights/", "factory/tokenizer/tokenizer.json")
    response = model.generate("Morning. What's the data status?")
    print(response)
"""

from pathlib import Path
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).parent.parent.parent  # trading-desk/


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BornTraderLLM:
    """
    Load a trained Born Trader checkpoint and generate responses.
    Plug this into any agent's Python code as the language layer.
    """

    def __init__(
        self,
        weights_dir: str,
        tokenizer_path: str,
        model_config_path: str = None,
        device: str = None,
        temperature: float = 0.7,
        max_new_tokens: int = 256,
    ):
        import sys
        sys.path.insert(0, str(ROOT))
        from factory.model.model import BornTrader, ModelConfig

        self.device = torch.device(device) if device else _get_device()
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Tokenizer
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.sys_id = self.tokenizer.token_to_id("<sys>")
        self.user_id = self.tokenizer.token_to_id("<user>")
        self.asst_id = self.tokenizer.token_to_id("<assistant>")

        # Model config
        cfg_path = model_config_path or str(ROOT / "factory" / "model" / "config.yaml")
        model_cfg = ModelConfig.from_yaml(cfg_path)

        # Load weights
        weights_dir = Path(weights_dir)
        ckpt_path = weights_dir / "best.pt"
        if not ckpt_path.exists():
            # Fall back to latest epoch checkpoint
            ckpts = sorted(weights_dir.glob("epoch_*.pt"))
            if not ckpts:
                raise FileNotFoundError(f"No weights found in {weights_dir}")
            ckpt_path = ckpts[-1]

        self.model = BornTrader(model_cfg).to(self.device)
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        print(f"Loaded weights from {ckpt_path} (epoch {checkpoint['epoch']}, loss {checkpoint['loss']:.4f})")

    def _build_prompt(self, user_message: str, system_prompt: str = None) -> list[int]:
        """Build token ids for the prompt ending just before the assistant turn."""
        parts = []
        if system_prompt:
            sys_enc = self.tokenizer.encode(f"<sys> {system_prompt.strip()}")
            parts.extend(sys_enc.ids)
        user_enc = self.tokenizer.encode(f"<user> {user_message.strip()}")
        parts.extend(user_enc.ids)
        # Start the assistant turn — model completes from here
        asst_enc = self.tokenizer.encode("<assistant>")
        parts.extend(asst_enc.ids)
        return parts

    @torch.no_grad()
    def generate(
        self,
        user_message: str,
        system_prompt: str = None,
        temperature: float = None,
        max_new_tokens: int = None,
    ) -> str:
        """
        Generate a response to user_message.

        Args:
            user_message: The incoming message text.
            system_prompt: Optional system context override.
            temperature: Sampling temperature (0 = greedy).
            max_new_tokens: Max tokens to generate.

        Returns:
            The model's response as a plain string.
        """
        temperature = temperature if temperature is not None else self.temperature
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        prompt_ids = self._build_prompt(user_message, system_prompt)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        generated = []
        for _ in range(max_new_tokens):
            # Truncate context if needed
            ctx = input_ids[:, -self.model.cfg.context_window:]
            logits = self.model(ctx)  # (1, seq_len, vocab_size)
            next_logits = logits[0, -1, :]  # (vocab_size,)

            if temperature == 0:
                next_id = next_logits.argmax().item()
            else:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()

            if next_id == self.eos_id:
                break

            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=self.device)], dim=1
            )

        # Decode and clean up
        text = self.tokenizer.decode(generated)
        # Strip any role tokens that leaked into output
        for token in ["<sys>", "<user>", "<assistant>", "<eos>", "<bos>", "<pad>"]:
            text = text.replace(token, "").strip()
        return text


if __name__ == "__main__":
    import sys
    weights_dir = str(ROOT / "agents" / "data_manager" / "weights")
    tokenizer_path = str(ROOT / "factory" / "tokenizer" / "tokenizer.json")

    weights_exist = list(Path(weights_dir).glob("*.pt"))
    if not weights_exist:
        print("No trained weights found. Run training first:")
        print("  python factory/training/train.py --agent data_manager")
        sys.exit(0)

    model = BornTraderLLM(weights_dir=weights_dir, tokenizer_path=tokenizer_path)

    test_messages = [
        "Morning. What's the data status?",
        "RELIANCE data please.",
        "Any issues with Fyers today?",
    ]

    print("\n--- Inference test ---")
    for msg in test_messages:
        print(f"\nUser     : {msg}")
        response = model.generate(msg)
        print(f"Response : {response}")
