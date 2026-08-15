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
| `train_shakespeare.py` | Character-level language model training, checkpointing |
| `sample.py` | Generate text from a saved checkpoint |

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
uv run pytest                                          # 54 tests
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

Tiny shakespeare, character level. Two runs, same code:

| Config | Params | Hardware | Time | Best val |
| --- | --- | --- | --- | --- |
| 4 layers, `d_model=256`, block 128 | 3.19M | M-series GPU | 5 min | 1.574 |
| 6 layers, `d_model=384`, block 256 | 10.7M | RTX 3060 Ti | 19 min | **1.444** |

The larger run matches nanoGPT's configuration for this dataset, whose
published validation loss is about 1.48.

| Step | Train | Val |
| --- | --- | --- |
| 1 | 4.654 | 4.659 |
| 1000 | 1.356 | 1.579 |
| 2000 | 1.183 | 1.490 |
| 3500 | 1.046 | **1.444** |
| 5000 | 0.984 | 1.461 |

Validation loss bottoms out around step 3500 and rises after it while training
loss keeps falling, so the checkpoint kept is the one from 3500, not the last.

## Generating text

```bash
uv run python -m transformer_from_scratch.sample --prompt "ROMEO:" --tokens 300
uv run python -m transformer_from_scratch.sample --temperature 0.5 --top-k 20 --seed 0
```

`--top-k 0` disables truncation. Metadata goes to stderr, so redirecting stdout
captures only the generated text. Prompts may only use characters that appear
in the training corpus.

From the 10.7M model:

```
MENENIUS:
Beseech you, as far off more than he enter to get the
by-good man that he could speak upon him, whose advance
between the warrant of his followers.

COMINIUS:
Why?

MENENIUS:
He's sentenced; I'll see him here in the city.
```

## Testing

The tests assert behaviour the shapes can't catch:

- perturbing future target tokens leaves earlier outputs bit-identical
- masked source positions cannot influence any output
- the model can overfit ten fixed sequences to near-zero loss
- weight tying actually shares storage (`data_ptr`)
- every parameter receives a gradient

## License

MIT
