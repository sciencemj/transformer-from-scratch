import torch

PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3


def make_batch(
    batch_size=32,
    vocab_size=20,
    min_len=4,
    max_len=10,
    task="copy",
    device=None,
    generator=None,
):
    """Synthetic sequence-to-sequence batch.

    src = [BOS, body, EOS, PAD...]
    tgt = [BOS, body (or reversed body), EOS, PAD...]

    Lengths vary within the batch, so the padding masks get exercised too.
    Returns (src, tgt), both (B, T) int64.
    """
    if task not in ("copy", "reverse"):
        raise ValueError(f"unknown task {task!r}")
    if vocab_size <= N_SPECIAL:
        raise ValueError(f"vocab_size must exceed {N_SPECIAL}, got {vocab_size}")

    lens = torch.randint(min_len, max_len + 1, (batch_size,), generator=generator)
    T = int(lens.max()) + 2  # room for BOS and EOS

    src = torch.full((batch_size, T), PAD, dtype=torch.long)
    tgt = torch.full((batch_size, T), PAD, dtype=torch.long)

    for b in range(batch_size):
        n = int(lens[b])
        body = torch.randint(N_SPECIAL, vocab_size, (n,), generator=generator)
        out = body if task == "copy" else body.flip(0)
        src[b, 0] = BOS
        src[b, 1 : n + 1] = body
        src[b, n + 1] = EOS
        tgt[b, 0] = BOS
        tgt[b, 1 : n + 1] = out
        tgt[b, n + 1] = EOS

    return src.to(device), tgt.to(device)
