import torch
from torch import nn

from .model import Transformer
from .tasks import BOS, EOS, PAD, make_batch


def loss_fn(logits, tgt):
    """Teacher forcing: logits predict tgt shifted one step left."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD
    )


def step_loss(model, src, tgt):
    """Feed tgt[:, :-1], score against tgt[:, 1:]."""
    logits = model(src, tgt[:, :-1])
    return loss_fn(logits, tgt[:, 1:])


@torch.no_grad()
def exact_match(model, src, tgt, max_len=None):
    """Fraction of sequences whose greedy decode equals tgt up to EOS."""
    max_len = max_len or tgt.size(1)
    pred = model.greedy_decode(src, max_len, BOS, EOS)

    hits = 0
    for b in range(src.size(0)):
        want = tgt[b][tgt[b] != PAD].tolist()
        got = pred[b][pred[b] != PAD].tolist()
        if EOS in got:
            got = got[: got.index(EOS) + 1]
        hits += got == want
    return hits / src.size(0)


def build_model(vocab_size=20, norm_first=True, **kwargs):
    """Small model sized for the synthetic tasks, not the paper's base config."""
    defaults = dict(
        n_layer=2, d_model=64, n_head=4, d_ff=128, dropout=0.1, max_len=64
    )
    defaults.update(kwargs)
    return Transformer(
        vocab_size, vocab_size, norm_first=norm_first, pad_id=PAD, **defaults
    )


def train_copy(
    steps=400,
    batch_size=32,
    vocab_size=20,
    task="copy",
    norm_first=True,
    lr=1e-3,
    seed=0,
    log_every=0,
    model=None,
    fixed_batch=None,
    device=None,
    **model_kwargs,
):
    """Train on synthetic copy/reverse data. Returns (model, history).

    fixed_batch: reuse one batch every step (the overfit sanity check).
    """
    torch.manual_seed(seed)
    if model is None:
        model = build_model(vocab_size, norm_first, **model_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    history = []
    model.train()
    for i in range(steps):
        if fixed_batch is not None:
            src, tgt = fixed_batch
        else:
            src, tgt = make_batch(
                batch_size, vocab_size, task=task, device=device
            )
        loss = step_loss(model, src, tgt)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        history.append(loss.item())
        if log_every and (i + 1) % log_every == 0:
            print(f"step {i + 1:5d}  loss {loss.item():.4f}")

    return model, history


def main() -> None:
    for task in ("copy", "reverse"):
        model, history = train_copy(steps=600, task=task, log_every=200)
        src, tgt = make_batch(64, task=task)
        acc = exact_match(model, src, tgt)
        print(f"{task}: final loss {history[-1]:.4f}  exact match {acc:.1%}\n")


if __name__ == "__main__":
    main()
