import math

from torch import nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    q: (..., T_q, d_k)
    k: (..., T_k, d_k)
    v: (..., T_k, d_v)
    mask: broadcastable to (..., T_q, T_k), 0/False where attention is forbidden
    returns: (..., T_q, d_v)
    """
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn = scores.softmax(dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return attn @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.1) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_head={n_head}")
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # One big projection per role; heads are carved out by a reshape below.
        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        # (B, T, d_model) -> (B, n_head, T, d_head)
        B, T, _ = x.shape
        return x.view(B, T, self.n_head, self.d_head).transpose(1, 2)

    def forward(self, Q, K, V, mask=None):
        """Q: (B, T_q, d_model), K/V: (B, T_k, d_model) -> (B, T_q, d_model)

        mask must broadcast against (B, n_head, T_q, T_k), e.g. a causal mask
        of shape (1, 1, T, T) or a padding mask of shape (B, 1, 1, T_k).
        """
        B, T_q, _ = Q.shape
        q = self._split_heads(self.Wq(Q))
        k = self._split_heads(self.Wk(K))
        v = self._split_heads(self.Wv(V))

        out = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        # (B, n_head, T_q, d_head) -> (B, T_q, d_model)
        out = out.transpose(1, 2).reshape(B, T_q, self.d_model)
        return self.Wo(out)
