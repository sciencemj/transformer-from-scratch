import torch

from transformer_from_scratch.masks import causal_mask, combine, padding_mask


def test_causal_mask_shape_and_triangle():
    m = causal_mask(4)
    assert m.shape == (1, 1, 4, 4)
    assert m.dtype == torch.bool
    # row i allows exactly the first i+1 columns
    assert m[0, 0].sum(-1).tolist() == [1, 2, 3, 4]
    assert not m[0, 0, 0, 1]  # position 0 cannot see position 1
    assert m[0, 0, 3, 0]  # position 3 can see position 0


def test_padding_mask():
    ids = torch.tensor([[5, 6, 0, 0], [7, 0, 0, 0]])
    m = padding_mask(ids, pad_id=0)
    assert m.shape == (2, 1, 1, 4)
    assert m[0, 0, 0].tolist() == [True, True, False, False]
    assert m[1, 0, 0].tolist() == [True, False, False, False]


def test_combine_broadcasts_and_skips_none():
    ids = torch.tensor([[5, 6, 0]])
    both = combine(padding_mask(ids), causal_mask(3))
    assert both.shape == (1, 1, 3, 3)
    # last column is padded, so it is blocked on every row
    assert not both[0, 0, :, 2].any()
    # causal structure survives
    assert both[0, 0, 1].tolist() == [True, True, False]

    assert combine(None, None) is None
    only = combine(None, causal_mask(3))
    assert torch.equal(only, causal_mask(3))
