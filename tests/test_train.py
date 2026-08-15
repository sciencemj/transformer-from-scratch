import pytest
import torch

from transformer_from_scratch.tasks import make_batch
from transformer_from_scratch.train import build_model, exact_match, train_copy


def test_overfits_a_single_fixed_batch():
    """The sharpest learning check: if the model cannot memorise 10 sequences,
    something in the wiring is broken regardless of what the shapes say."""
    torch.manual_seed(0)
    batch = make_batch(10, vocab_size=20, min_len=4, max_len=8)
    _, history = train_copy(steps=300, fixed_batch=batch, dropout=0.0, seed=0)

    assert history[-1] < 0.05, f"failed to overfit; final loss {history[-1]:.4f}"
    assert history[-1] < history[0] / 10


@pytest.mark.parametrize("norm_first", [True, False])
def test_both_layernorm_placements_train(norm_first):
    _, history = train_copy(steps=60, norm_first=norm_first, seed=0)
    assert torch.isfinite(torch.tensor(history)).all()
    assert history[-1] < history[0]


# reverse needs the attention to map position i to n-i, which takes noticeably
# longer to learn than the identity mapping copy needs.
@pytest.mark.parametrize("task,steps", [("copy", 600), ("reverse", 2000)])
def test_task_converges_to_exact_match(task, steps):
    model, history = train_copy(steps=steps, task=task, seed=0)
    src, tgt = make_batch(64, task=task, min_len=4, max_len=10)
    acc = exact_match(model, src, tgt)

    assert acc > 0.95, f"{task}: exact match {acc:.1%}, final loss {history[-1]:.4f}"


def test_exact_match_is_zero_for_an_untrained_model():
    torch.manual_seed(0)
    model = build_model(dropout=0.0)
    src, tgt = make_batch(16)
    assert exact_match(model, src, tgt) < 0.2
