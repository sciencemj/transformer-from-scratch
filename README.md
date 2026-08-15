# transformer-from-scratch

A transformer built up from the individual sub-layers, following
[Attention Is All You Need](https://arxiv.org/abs/1706.03762), with tests that
check the properties the paper relies on rather than just the tensor shapes.

Everything is written directly against `torch.nn` primitives — attention,
positional encoding, the residual/LayerNorm wiring, and both stacks. Nothing
uses `nn.MultiheadAttention` or `nn.Transformer`.

## What's here

| Module | Contents |
| --- | --- |
| `attention.py` | `scaled_dot_product_attention`, `MultiHeadAttention` |
| `encoding.py` | `PositionalEncoding` (sinusoidal), `TokenEmbedding` |
| `masks.py` | `causal_mask`, `padding_mask`, `combine` |
| `layers.py` | `PositionwiseFFN`, `Residual`, `EncoderLayer`, `DecoderLayer` |
| `model.py` | `Encoder`, `Decoder`, `Transformer`, `DecoderOnlyLM` |
| `tasks.py` | Synthetic copy / reverse batches |
| `data.py` | Character tokenizer and batching for tiny shakespeare |
| `train.py` | Training loop for the synthetic tasks |
| `train_shakespeare.py` | Character-level language model training |

One `DecoderLayer` covers both architectures: `cross_attn=False` drops the
middle sub-layer, which is the only structural difference between a
sequence-to-sequence decoder and a GPT-style block. `norm_first` selects
between the paper's Post-LN and the now-standard Pre-LN.

Mask convention: every mask is boolean with `True` meaning *allowed*, shaped to
broadcast against `(batch, heads, query_len, key_len)`, so masks compose with a
plain `&`.

## Usage

```bash
uv sync
uv run pytest                                          # 43 tests
uv run python -m transformer_from_scratch.train        # copy / reverse tasks
uv run python -m transformer_from_scratch.train_shakespeare
```

The tiny shakespeare corpus downloads itself on first run.

```python
from transformer_from_scratch import Transformer, DecoderOnlyLM

seq2seq = Transformer(src_vocab=32000, tgt_vocab=32000)      # paper base config
logits = seq2seq(src_ids, tgt_ids)                           # (B, T, tgt_vocab)

lm = DecoderOnlyLM(vocab_size=65, n_layer=6, d_model=384, pad_id=None)
text = lm.generate(prompt_ids, max_new_tokens=500, temperature=0.8, top_k=40)
```

Use `pad_id=None` for dense language-model batches — otherwise whichever token
maps to the pad id gets masked out of the corpus.

## Results

Synthetic tasks (2 layers, `d_model=64`, CPU):

| Task | Steps | Exact match |
| --- | --- | --- |
| copy | 600 | 100% |
| reverse | 2000 | 96.9% |

Reverse needs the attention to learn an `i -> n-i` mapping, so it takes
noticeably longer than copy's identity mapping.

Tiny shakespeare, character level (4 layers, `d_model=256`, block 128, 3.19M
params, 5000 steps, ~5 min on an M-series GPU):

| Step | Train | Val |
| --- | --- | --- |
| 1 | 5.12 | 5.12 |
| 1000 | 1.67 | 1.84 |
| 2500 | 1.45 | 1.65 |
| 5000 | 1.36 | 1.57 |

Validation loss was still falling at the end, so this is undertrained rather
than converged.

## Testing

The tests assert behaviour the shapes can't catch:

- perturbing future target tokens leaves earlier outputs bit-identical
- masked source positions cannot influence any output
- the model can overfit ten fixed sequences to near-zero loss
- weight tying actually shares storage (`data_ptr`)
- every parameter receives a gradient

## License

MIT
