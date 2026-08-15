import math
import time

import torch

from .data import CharTokenizer, get_batch, load_text, split_data
from .model import DecoderOnlyLM


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def lr_at(step, total, base_lr, min_lr, warmup):
    """Linear warmup then cosine decay -- the schedule nanoGPT uses."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, iters, device):
    """Average loss over a few random batches per split."""
    was_training = model.training
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for i in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            logits = model(x)
            losses[i] = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1)
            ).item()
        out[name] = losses.mean().item()
    model.train(was_training)
    return out


def train(
    steps=5000,
    n_layer=4,
    d_model=256,
    n_head=4,
    block_size=128,
    batch_size=32,
    dropout=0.2,
    lr=1e-3,
    min_lr=1e-4,
    warmup=100,
    weight_decay=0.1,
    eval_every=250,
    eval_iters=20,
    seed=0,
    device=None,
    text_path=None,
    sample_tokens=500,
):
    device = device or pick_device()
    torch.manual_seed(seed)

    text = load_text(text_path, download=True) if text_path else load_text(download=True)
    tok = CharTokenizer(text)
    train_ids, val_ids = split_data(tok.encode(text))
    print(
        f"device {device} | vocab {tok.vocab_size} | "
        f"train {len(train_ids):,} val {len(val_ids):,} chars"
    )

    # pad_id=None: language-model batches are dense, and reserving id 0 as pad
    # would silently delete whichever character maps to it.
    model = DecoderOnlyLM(
        tok.vocab_size,
        n_layer,
        d_model,
        n_head,
        4 * d_model,
        dropout,
        max_len=block_size,
        pad_id=None,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params / 1e6:.2f}M | {steps} steps")

    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=weight_decay
    )
    splits = {"train": train_ids, "val": val_ids}
    history = []
    model.train()
    t0 = time.perf_counter()

    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, steps, lr, min_lr, warmup)

        x, y = get_batch(train_ids, block_size, batch_size, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if eval_every and ((step + 1) % eval_every == 0 or step == 0):
            ev = estimate_loss(
                model, splits, block_size, batch_size, eval_iters, device
            )
            ev["step"] = step + 1
            ev["elapsed"] = time.perf_counter() - t0
            history.append(ev)
            print(
                f"step {step + 1:5d}  train {ev['train']:.4f}  "
                f"val {ev['val']:.4f}  {ev['elapsed'] / 60:5.1f} min"
            )

    print(f"\ntotal {(time.perf_counter() - t0) / 60:.1f} min")

    if sample_tokens:
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        ids = model.generate(start, sample_tokens, temperature=0.8, top_k=40)
        print("\n--- sample ---")
        print(tok.decode(ids[0]))

    return model, tok, history


def main() -> None:
    train()


if __name__ == "__main__":
    main()
