import torch
from torch import nn

from .encoding import PositionalEncoding, TokenEmbedding
from .layers import DecoderLayer, EncoderLayer
from .masks import causal_mask, combine, padding_mask


class Encoder(nn.Module):
    """Stack of N encoder layers."""

    def __init__(
        self,
        n_layer=6,
        d_model=512,
        n_head=8,
        d_ff=2048,
        dropout=0.1,
        norm_first=True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_head, d_ff, dropout, norm_first)
            for _ in range(n_layer)
        )
        # Pre-LN leaves the stack output un-normalised, so it needs a final LN.
        # Post-LN already ends each layer with one.
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N decoder layers. cross_attn=False gives a decoder-only stack."""

    def __init__(
        self,
        n_layer=6,
        d_model=512,
        n_head=8,
        d_ff=2048,
        dropout=0.1,
        norm_first=True,
        cross_attn=True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_head, d_ff, dropout, norm_first, cross_attn)
            for _ in range(n_layer)
        )
        self.norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()

    def forward(self, x, memory=None, tgt_mask=None, src_mask=None):
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.norm(x)


def _init_linears(module):
    """Xavier init on Linear weights (sec 5.3-ish); embeddings keep their own."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class Transformer(nn.Module):
    """Encoder-decoder transformer (Vaswani et al. 2017)."""

    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        n_layer=6,
        d_model=512,
        n_head=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        norm_first=True,
        pad_id=0,
        tie_weights=False,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.src_emb = TokenEmbedding(src_vocab, d_model, padding_idx=pad_id)
        self.tgt_emb = TokenEmbedding(tgt_vocab, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)

        self.encoder = Encoder(n_layer, d_model, n_head, d_ff, dropout, norm_first)
        self.decoder = Decoder(
            n_layer, d_model, n_head, d_ff, dropout, norm_first, cross_attn=True
        )
        self.lm_head = nn.Linear(d_model, tgt_vocab, bias=False)

        _init_linears(self)
        if tie_weights:
            # Share one matrix between the output projection and the target
            # embedding (sec 3.4).
            self.lm_head.weight = self.tgt_emb.emb.weight

    def _pad_mask(self, ids):
        """None when the model was built without a pad id (dense batches)."""
        if self.pad_id is None:
            return None
        return padding_mask(ids, self.pad_id)

    def make_masks(self, src, tgt):
        src_mask = self._pad_mask(src)
        tgt_mask = combine(self._pad_mask(tgt), causal_mask(tgt.size(1), tgt.device))
        return src_mask, tgt_mask

    def encode(self, src, src_mask=None):
        """src: (B, S) -> memory (B, S, d_model)"""
        if src_mask is None:
            src_mask = self._pad_mask(src)
        return self.encoder(self.pos(self.src_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask=None, src_mask=None):
        """tgt: (B, T) -> logits (B, T, tgt_vocab)"""
        if tgt_mask is None:
            tgt_mask = combine(
                self._pad_mask(tgt), causal_mask(tgt.size(1), tgt.device)
            )
        h = self.decoder(self.pos(self.tgt_emb(tgt)), memory, tgt_mask, src_mask)
        return self.lm_head(h)

    def forward(self, src, tgt):
        """src: (B, S), tgt: (B, T) -> logits (B, T, tgt_vocab)"""
        src_mask, tgt_mask = self.make_masks(src, tgt)
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, tgt_mask, src_mask)

    @torch.no_grad()
    def greedy_decode(self, src, max_len, bos_id, eos_id):
        """src: (B, S) -> generated ids (B, <=max_len), starting with bos."""
        was_training = self.training
        self.eval()
        src_mask = self._pad_mask(src)
        memory = self.encode(src, src_mask)

        B = src.size(0)
        out = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len - 1):
            logits = self.decode(out, memory, src_mask=src_mask)
            nxt = logits[:, -1].argmax(-1)
            # Once a sequence has emitted eos, keep padding it.
            fill = self.pad_id if self.pad_id is not None else eos_id
            nxt = torch.where(done, torch.full_like(nxt, fill), nxt)
            out = torch.cat([out, nxt.unsqueeze(1)], dim=1)
            done |= nxt == eos_id
            if bool(done.all()):
                break

        self.train(was_training)
        return out


class DecoderOnlyLM(nn.Module):
    """GPT-style stack: the same Decoder with its cross-attention removed."""

    def __init__(
        self,
        vocab_size,
        n_layer=6,
        d_model=512,
        n_head=8,
        d_ff=2048,
        dropout=0.1,
        max_len=5000,
        norm_first=True,
        pad_id=0,
        tie_weights=False,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.emb = TokenEmbedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.decoder = Decoder(
            n_layer, d_model, n_head, d_ff, dropout, norm_first, cross_attn=False
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        _init_linears(self)
        if tie_weights:
            self.lm_head.weight = self.emb.emb.weight

    def forward(self, ids, mask=None):
        """ids: (B, T) -> logits (B, T, vocab_size)"""
        if mask is None:
            mask = causal_mask(ids.size(1), ids.device)
            if self.pad_id is not None:
                mask = combine(padding_mask(ids, self.pad_id), mask)
        h = self.decoder(self.pos(self.emb(ids)), None, mask)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None):
        """ids: (B, T) prompt -> (B, T + max_new_tokens)"""
        was_training = self.training
        self.eval()
        max_len = self.pos.pe.size(1)

        for _ in range(max_new_tokens):
            window = ids[:, -max_len:]
            logits = self(window)[:, -1] / temperature
            if top_k is not None:
                kth = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            nxt = torch.multinomial(logits.softmax(-1), num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)

        self.train(was_training)
        return ids
