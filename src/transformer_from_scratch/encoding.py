import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al. 2017, sec 3.5).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Dimensions 2i and 2i+1 share one frequency, so [sin, cos] is a point on the
    unit circle and a shift of k positions is a fixed rotation independent of
    pos -- that is what lets the model learn relative offsets linearly.
    """

    # nn.Module.__getattr__ returns a union; pin the buffer's type for checkers.
    pe: torch.Tensor

    def __init__(self, d_model=512, max_len=5000, dropout=0.1) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even, got {d_model}")
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        # 10000^(-2i/d_model), computed in log space to avoid overflow.
        div = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        # Not a parameter: fixed table, but should follow .to(device) and land
        # in state_dict.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, T, d_model) -> same shape, positions added."""
        T = x.size(1)
        if T > self.pe.size(1):
            raise ValueError(f"sequence length {T} exceeds max_len {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, :T])


class TokenEmbedding(nn.Module):
    """Token ids -> vectors, scaled by sqrt(d_model) as in sec 3.4.

    The scaling only balances against the positional signal if the weights start
    with variance 1/d_model (norm ~ 1), so the multiply lands the embedding norm
    at ~sqrt(d_model) next to the encoding's sqrt(d_model/2). torch's default
    N(0, 1) init would instead make the token signal d_model times too loud.
    """

    def __init__(self, vocab_size, d_model=512, padding_idx=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        nn.init.normal_(self.emb.weight, mean=0.0, std=d_model**-0.5)
        if padding_idx is not None:
            with torch.no_grad():
                self.emb.weight[padding_idx].fill_(0)

    def forward(self, ids):
        """ids: (B, T) int64 -> (B, T, d_model)"""
        return self.emb(ids) * math.sqrt(self.d_model)
