import pytest
import torch

from transformer_from_scratch.tasks import BOS, EOS, N_SPECIAL, PAD, make_batch


@pytest.mark.parametrize("task", ["copy", "reverse"])
def test_batch_shape_and_specials(task):
    torch.manual_seed(0)
    src, tgt = make_batch(8, vocab_size=20, min_len=4, max_len=10, task=task)

    assert src.shape == tgt.shape
    assert src.size(0) == 8
    assert src.dtype == torch.long
    assert (src[:, 0] == BOS).all()
    assert (tgt[:, 0] == BOS).all()
    assert (src == EOS).sum(1).eq(1).all()  # exactly one EOS per row


@pytest.mark.parametrize("task", ["copy", "reverse"])
def test_target_body_matches_task(task):
    torch.manual_seed(1)
    src, tgt = make_batch(16, task=task)

    for b in range(src.size(0)):
        n = int((src[b] == EOS).nonzero()[0]) - 1
        body = src[b, 1 : n + 1]
        want = body if task == "copy" else body.flip(0)
        assert torch.equal(tgt[b, 1 : n + 1], want)


def test_body_tokens_avoid_special_ids():
    torch.manual_seed(2)
    src, _ = make_batch(16, vocab_size=20)
    for b in range(src.size(0)):
        n = int((src[b] == EOS).nonzero()[0]) - 1
        assert (src[b, 1 : n + 1] >= N_SPECIAL).all()


def test_lengths_vary_so_padding_is_exercised():
    torch.manual_seed(3)
    src, _ = make_batch(32, min_len=4, max_len=10)
    assert (src == PAD).any(), "no padding produced; masks would go untested"


def test_rejects_bad_arguments():
    with pytest.raises(ValueError, match="unknown task"):
        make_batch(2, task="sort")
    with pytest.raises(ValueError, match="vocab_size"):
        make_batch(2, vocab_size=3)
