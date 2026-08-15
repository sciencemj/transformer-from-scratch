import torch

from transformer_from_scratch.data import CharTokenizer
from transformer_from_scratch.model import DecoderOnlyLM
from transformer_from_scratch.train_shakespeare import (
    load_checkpoint,
    save_checkpoint,
)


def test_checkpoint_roundtrip_preserves_outputs(tmp_path):
    torch.manual_seed(0)
    tok = CharTokenizer("hello shakespeare world")
    config = dict(
        vocab_size=tok.vocab_size,
        n_layer=2,
        d_model=32,
        n_head=4,
        d_ff=64,
        dropout=0.0,
        max_len=16,
        pad_id=None,
    )
    model = DecoderOnlyLM(**config).eval()

    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, tok, config, step=7, val_loss=1.23)

    restored, restored_tok, ckpt = load_checkpoint(path, device="cpu")
    assert ckpt["step"] == 7
    assert ckpt["val_loss"] == 1.23
    assert restored_tok.chars == tok.chars

    ids = torch.randint(0, tok.vocab_size, (2, 5))
    assert torch.allclose(model(ids), restored(ids), atol=1e-6)


def test_restored_tokenizer_round_trips_text(tmp_path):
    tok = CharTokenizer("to be or not to be\n")
    config = dict(
        vocab_size=tok.vocab_size, n_layer=1, d_model=16, n_head=2, d_ff=32,
        dropout=0.0, max_len=8, pad_id=None,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, DecoderOnlyLM(**config), tok, config, 1, 9.9)

    _, restored_tok, _ = load_checkpoint(path, device="cpu")
    text = "to be or not"
    assert restored_tok.decode(restored_tok.encode(text)) == text
