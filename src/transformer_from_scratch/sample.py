import argparse
import sys

import torch

from .train_shakespeare import DEFAULT_CKPT, load_checkpoint, pick_device


def encode_prompt(tok, text, device):
    """Prompt characters must exist in the corpus vocabulary."""
    unknown = sorted(set(text) - set(tok.chars))
    if unknown:
        raise ValueError(
            f"characters not in the model's vocabulary: {unknown!r}. "
            "This model only knows the characters in tiny shakespeare."
        )
    if not text:
        # A newline is a sensible neutral start: the corpus uses it between speeches.
        text = "\n"
    return tok.encode(text).unsqueeze(0).to(device)


def sample(
    ckpt_path=DEFAULT_CKPT,
    prompt="",
    tokens=500,
    temperature=0.8,
    top_k=40,
    seed=None,
    device=None,
):
    """Load a checkpoint and return generated text."""
    device = device or pick_device()
    if seed is not None:
        torch.manual_seed(seed)

    model, tok, ckpt = load_checkpoint(ckpt_path, device=device)
    ids = encode_prompt(tok, prompt, device)
    out = model.generate(ids, tokens, temperature=temperature, top_k=top_k)
    return tok.decode(out[0]), ckpt


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a trained checkpoint")
    p.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint path")
    p.add_argument("--prompt", default="", help="text to continue from")
    p.add_argument("--tokens", type=int, default=500, help="characters to generate")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="lower is more predictable, higher more chaotic",
    )
    p.add_argument(
        "--top-k", type=int, default=40, help="sample from the k likeliest characters"
    )
    p.add_argument("--seed", type=int, default=None, help="fix for reproducible output")
    p.add_argument("--device", default=None, help="cpu, mps or cuda")
    args = p.parse_args()

    text, ckpt = sample(
        args.ckpt,
        args.prompt,
        args.tokens,
        args.temperature,
        args.top_k,
        args.seed,
        args.device,
    )
    # Metadata on stderr so `> out.txt` captures only the generated text.
    print(
        f"[{args.ckpt} | step {ckpt['step']} | val {ckpt['val_loss']:.4f}]\n",
        file=sys.stderr,
    )
    print(text)


if __name__ == "__main__":
    main()
