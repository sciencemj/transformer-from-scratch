import torch

# Convention: masks are bool and True means "attention allowed". Every mask
# broadcasts against (B, n_head, T_q, T_k) so encoder / decoder / cross masks
# can be combined with a plain &.


def causal_mask(T, device=None):
    """(1, 1, T, T) -- position i may attend to j <= i."""
    return torch.ones(T, T, dtype=torch.bool, device=device).tril().view(1, 1, T, T)


def padding_mask(ids, pad_id=0):
    """ids: (B, T) -> (B, 1, 1, T), False on pad positions."""
    B, T = ids.shape
    return (ids != pad_id).view(B, 1, 1, T)


def combine(*masks):
    """AND together the masks that are not None; None if all are None."""
    out = None
    for m in masks:
        if m is None:
            continue
        out = m if out is None else out & m
    return out
