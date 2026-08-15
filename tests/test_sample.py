import pytest
import torch

from transformer_from_scratch.data import CharTokenizer
from transformer_from_scratch.model import DecoderOnlyLM
from transformer_from_scratch.sample import encode_prompt, sample
from transformer_from_scratch.train_shakespeare import save_checkpoint

TEXT = "to be or not to be, that is the question\n"


@pytest.fixture
def ckpt(tmp_path):
    torch.manual_seed(0)
    tok = CharTokenizer(TEXT)
    config = dict(
        vocab_size=tok.vocab_size, n_layer=1, d_model=32, n_head=4, d_ff=64,
        dropout=0.0, max_len=32, pad_id=None,
    )
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, DecoderOnlyLM(**config), tok, config, 42, 1.5)
    return path


def test_sample_generates_requested_length(ckpt):
    text, meta = sample(ckpt, prompt="to be", tokens=20, seed=0, device="cpu")
    assert len(text) == len("to be") + 20
    assert text.startswith("to be")
    assert meta["step"] == 42


def test_sample_is_reproducible_with_a_seed(ckpt):
    a, _ = sample(ckpt, prompt="to", tokens=15, seed=7, device="cpu")
    b, _ = sample(ckpt, prompt="to", tokens=15, seed=7, device="cpu")
    assert a == b


def test_empty_prompt_starts_from_newline(ckpt):
    text, _ = sample(ckpt, prompt="", tokens=10, seed=0, device="cpu")
    assert text.startswith("\n")


@pytest.mark.parametrize("top_k", [0, None, 1, 5, 10_000])
def test_top_k_edge_values_still_produce_text(ckpt, top_k):
    """top_k=0 used to mask every logit and yield nothing."""
    text, _ = sample(ckpt, prompt="to", tokens=10, top_k=top_k, seed=0, device="cpu")
    assert len(text) == len("to") + 10


def test_unknown_prompt_characters_are_reported():
    tok = CharTokenizer(TEXT)
    with pytest.raises(ValueError, match="not in the model's vocabulary"):
        encode_prompt(tok, "안녕", "cpu")
