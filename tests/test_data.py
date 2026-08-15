import pytest
import torch

from transformer_from_scratch.data import (
    CharTokenizer,
    get_batch,
    load_text,
    split_data,
)

TEXT = "hello shakespeare\nsecond line!"


def test_tokenizer_roundtrip_and_vocab():
    tok = CharTokenizer(TEXT)
    assert tok.vocab_size == len(set(TEXT))
    assert tok.chars == sorted(set(TEXT))
    ids = tok.encode(TEXT)
    assert ids.dtype == torch.long
    assert tok.decode(ids) == TEXT


def test_tokenizer_ids_start_at_zero():
    """Char ids are dense from 0, which is why the LM must run with pad_id=None."""
    tok = CharTokenizer(TEXT)
    assert min(tok.stoi.values()) == 0
    assert max(tok.stoi.values()) == tok.vocab_size - 1


def test_split_is_contiguous_and_disjoint():
    ids = torch.arange(100)
    train, val = split_data(ids, val_frac=0.1)
    assert len(train) == 90 and len(val) == 10
    assert torch.equal(train, torch.arange(90))
    assert torch.equal(val, torch.arange(90, 100))


def test_get_batch_shapes_and_shift():
    data = torch.arange(200)
    x, y = get_batch(data, block_size=8, batch_size=4)
    assert x.shape == y.shape == (4, 8)
    # y is x shifted one position left
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)  # holds because data is arange


def test_get_batch_rejects_short_data():
    with pytest.raises(ValueError, match="too short"):
        get_batch(torch.arange(5), block_size=8, batch_size=2)


def test_load_text_reports_missing_file():
    with pytest.raises(FileNotFoundError, match="download"):
        load_text("/nonexistent/tinyshakespeare.txt")
