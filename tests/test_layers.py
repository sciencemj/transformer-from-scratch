import pytest
import torch
from torch import nn

from transformer_from_scratch.layers import (
    DecoderLayer,
    EncoderLayer,
    PositionwiseFFN,
    Residual,
)

D, H, FF = 32, 4, 64


def test_ffn_shape_and_is_per_position():
    ffn = PositionwiseFFN(D, FF, dropout=0.0).eval()
    x = torch.randn(2, 5, D)
    assert ffn(x).shape == (2, 5, D)
    # Same vector at two positions must give the same output: FFN sees no context.
    x[0, 1] = x[0, 3]
    out = ffn(x)
    assert torch.allclose(out[0, 1], out[0, 3], atol=1e-6)


@pytest.mark.parametrize("norm_first", [True, False])
def test_residual_placement(norm_first):
    res = Residual(D, dropout=0.0, norm_first=norm_first).eval()
    x = torch.randn(2, 5, D)
    sub = nn.Linear(D, D)
    got = res(x, sub)

    want = x + sub(res.norm(x)) if norm_first else res.norm(x + sub(x))
    assert torch.allclose(got, want, atol=1e-6)


@pytest.mark.parametrize("norm_first", [True, False])
def test_encoder_layer_shape(norm_first):
    layer = EncoderLayer(D, H, FF, 0.0, norm_first).eval()
    x = torch.randn(2, 7, D)
    assert layer(x).shape == (2, 7, D)


@pytest.mark.parametrize("norm_first", [True, False])
def test_decoder_layer_with_cross(norm_first):
    layer = DecoderLayer(D, H, FF, 0.0, norm_first, cross_attn=True).eval()
    x = torch.randn(2, 7, D)
    memory = torch.randn(2, 11, D)  # source length differs from target length
    assert layer(x, memory).shape == (2, 7, D)


def test_decoder_layer_without_cross_has_no_cross_module():
    layer = DecoderLayer(D, H, FF, 0.0, cross_attn=False).eval()
    assert layer.cross_attn is None
    x = torch.randn(2, 7, D)
    assert layer(x).shape == (2, 7, D)  # works with no memory at all


def test_cross_attn_layer_requires_memory():
    layer = DecoderLayer(D, H, FF, 0.0, cross_attn=True).eval()
    with pytest.raises(ValueError, match="requires memory"):
        layer(torch.randn(2, 7, D))


def test_cross_attn_adds_one_sublayer_of_params():
    with_cross = DecoderLayer(D, H, FF, cross_attn=True)
    without = DecoderLayer(D, H, FF, cross_attn=False)
    n_with = sum(p.numel() for p in with_cross.parameters())
    n_without = sum(p.numel() for p in without.parameters())
    # one extra MultiHeadAttention (4 * D * D) plus one extra LayerNorm (2 * D)
    assert n_with - n_without == 4 * D * D + 2 * D
