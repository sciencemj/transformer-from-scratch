from torch import nn

from .attention import MultiHeadAttention


class PositionwiseFFN(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2  (sec 3.3), applied per position."""

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    """Residual connection around one sub-layer, with the LayerNorm placement.

    Keeping the norm_first branch here means it exists in exactly one place
    instead of once per sub-layer. Dropout sits on the sub-layer output before
    the add, per sec 5.4.

    norm_first=False is the paper's Post-LN: LN(x + drop(sublayer(x))).
    norm_first=True is Pre-LN: x + drop(sublayer(LN(x))), which trains without
    a warmup schedule.
    """

    def __init__(self, d_model=512, dropout=0.1, norm_first=True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, sublayer):
        if self.norm_first:
            return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    """Bidirectional self-attention + FFN."""

    def __init__(
        self, d_model=512, n_head=8, d_ff=2048, dropout=0.1, norm_first=True
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.res_self = Residual(d_model, dropout, norm_first)
        self.res_ffn = Residual(d_model, dropout, norm_first)

    def forward(self, x, src_mask=None):
        """x: (B, S, d_model) -> (B, S, d_model)"""
        x = self.res_self(x, lambda t: self.self_attn(t, t, t, src_mask))
        return self.res_ffn(x, self.ffn)


class DecoderLayer(nn.Module):
    """Causal self-attention + optional cross-attention + FFN.

    cross_attn=False drops the middle sub-layer, which is what turns the same
    class into a decoder-only (GPT-style) block.
    """

    def __init__(
        self,
        d_model=512,
        n_head=8,
        d_ff=2048,
        dropout=0.1,
        norm_first=True,
        cross_attn=True,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.res_self = Residual(d_model, dropout, norm_first)

        if cross_attn:
            self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
            self.res_cross = Residual(d_model, dropout, norm_first)
        else:
            self.cross_attn = None

        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.res_ffn = Residual(d_model, dropout, norm_first)

    def forward(self, x, memory=None, tgt_mask=None, src_mask=None):
        """x: (B, T, d_model), memory: (B, S, d_model) -> (B, T, d_model)"""
        x = self.res_self(x, lambda t: self.self_attn(t, t, t, tgt_mask))

        # Bound to a local so the None-check narrows inside the lambda.
        cross = self.cross_attn
        if cross is not None:
            if memory is None:
                raise ValueError("cross-attention layer requires memory")
            x = self.res_cross(x, lambda t: cross(t, memory, memory, src_mask))

        return self.res_ffn(x, self.ffn)
