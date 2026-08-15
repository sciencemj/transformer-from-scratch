import pytest
import torch

from transformer_from_scratch.masks import causal_mask, padding_mask
from transformer_from_scratch.model import DecoderOnlyLM, Transformer
from transformer_from_scratch.tasks import BOS, EOS, PAD

V, D, H, FF, N = 20, 32, 4, 64, 2


def build(norm_first=True, dropout=0.0, **kw):
    return Transformer(
        V, V, N, D, H, FF, dropout, max_len=64, norm_first=norm_first, **kw
    ).eval()


@pytest.mark.parametrize("norm_first", [True, False])
def test_forward_shape_with_different_lengths(norm_first):
    model = build(norm_first)
    src = torch.randint(3, V, (2, 11))
    tgt = torch.randint(3, V, (2, 7))
    assert model(src, tgt).shape == (2, 7, V)


def test_decoder_cannot_see_future_target_tokens():
    model = build()
    src = torch.randint(3, V, (2, 6))
    tgt = torch.randint(3, V, (2, 8))
    memory = model.encode(src)

    base = model.decode(tgt, memory)
    tgt2 = tgt.clone()
    tgt2[:, 5:] = torch.randint(3, V, (2, 3))
    perturbed = model.decode(tgt2, memory)

    assert torch.allclose(base[:, :5], perturbed[:, :5], atol=1e-5)
    assert not torch.allclose(base[:, 5:], perturbed[:, 5:], atol=1e-5)


def test_masked_source_positions_do_not_affect_output():
    model = build()
    src = torch.tensor([[BOS, 5, 6, EOS, PAD, PAD]])
    tgt = torch.tensor([[BOS, 5, 6, EOS]])
    src_mask = padding_mask(src, PAD)

    base = model.decode(tgt, model.encode(src, src_mask), src_mask=src_mask)

    # Overwrite the padded slots with junk but keep the original mask.
    noisy = src.clone()
    noisy[0, 4:] = torch.tensor([9, 11])
    got = model.decode(tgt, model.encode(noisy, src_mask), src_mask=src_mask)

    assert torch.allclose(base, got, atol=1e-5)


def test_decoder_only_is_causal_and_has_no_cross_attention():
    lm = DecoderOnlyLM(V, N, D, H, FF, dropout=0.0, max_len=64).eval()
    assert all(layer.cross_attn is None for layer in lm.decoder.layers)

    ids = torch.randint(3, V, (2, 9))
    base = lm(ids)
    ids2 = ids.clone()
    ids2[:, 6:] = torch.randint(3, V, (2, 3))
    assert torch.allclose(base[:, :6], lm(ids2)[:, :6], atol=1e-5)


def test_parameters_registered_and_gradients_reach_everything():
    model = build(dropout=0.1).train()
    assert sum(p.numel() for p in model.parameters()) > 0

    src = torch.randint(3, V, (2, 6))
    tgt = torch.randint(3, V, (2, 6))
    model(src, tgt).sum().backward()

    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert missing == [], f"no gradient reached: {missing}"


def test_weight_tying_shares_storage():
    tied = build(tie_weights=True)
    assert tied.lm_head.weight.data_ptr() == tied.tgt_emb.emb.weight.data_ptr()

    untied = build(tie_weights=False)
    assert untied.lm_head.weight.data_ptr() != untied.tgt_emb.emb.weight.data_ptr()
    assert sum(p.numel() for p in tied.parameters()) < sum(
        p.numel() for p in untied.parameters()
    )


def test_pre_ln_adds_final_norm_post_ln_does_not():
    assert isinstance(build(norm_first=True).encoder.norm, torch.nn.LayerNorm)
    assert isinstance(build(norm_first=False).encoder.norm, torch.nn.Identity)


def test_greedy_decode_shape_and_starts_with_bos():
    model = build()
    src = torch.randint(3, V, (3, 6))
    out = model.greedy_decode(src, max_len=10, bos_id=BOS, eos_id=EOS)

    assert out.dim() == 2 and out.size(0) == 3
    assert out.size(1) <= 10
    assert (out[:, 0] == BOS).all()


def test_greedy_decode_restores_training_mode():
    model = build().train()
    model.greedy_decode(torch.randint(3, V, (1, 5)), 6, BOS, EOS)
    assert model.training


def test_make_masks_blocks_pad_and_future():
    model = build()
    src = torch.tensor([[BOS, 5, EOS, PAD]])
    tgt = torch.tensor([[BOS, 5, EOS, PAD]])
    src_mask, tgt_mask = model.make_masks(src, tgt)

    assert src_mask.shape == (1, 1, 1, 4)
    assert not src_mask[0, 0, 0, 3]
    assert tgt_mask.shape == (1, 1, 4, 4)
    assert not tgt_mask[..., 3].any()  # pad column blocked everywhere
    assert torch.equal(tgt_mask, tgt_mask & causal_mask(4))  # still causal
