"""ModernBERT backbone (#33 SOTA lever) — registry + a guarded end-to-end smoke.

The end-to-end test loads a 149M pretrained encoder (network + heavy), so it's
opt-in: set POLYMORPH_TEST_MODERNBERT=1 to run it. The cheap registry/guard
checks always run.
"""

import os

import pytest

from polymorph_lamr.model.lamr import _BACKBONES, LaMRConfig, LaMRModel


def test_modernbert_is_registered():
    assert "modernbert" in _BACKBONES
    # Config carries the encoder name knob (only used by this backbone).
    assert LaMRConfig().encoder_name


def test_other_backbones_tolerate_encoder_name_kwarg():
    # The factory passes encoder_name to every backbone; the from-scratch ones
    # must swallow it (regression guard for the **kwargs change).
    cfg = LaMRConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, ff_mult=2,
                     dropout=0.0, backbone="transformer")
    model = LaMRModel(cfg)
    import torch

    ids = torch.randint(0, 64, (1, 8))
    mask = torch.ones((1, 8), dtype=torch.bool)
    assert model(ids, mask).shape == (1, 8)


@pytest.mark.skipif(
    os.environ.get("POLYMORPH_TEST_MODERNBERT") != "1",
    reason="set POLYMORPH_TEST_MODERNBERT=1 to run the heavy ModernBERT load",
)
def test_modernbert_trains_end_to_end():
    import torch

    cfg = LaMRConfig(vocab_size=50368, d_model=768, backbone="modernbert")
    model = LaMRModel(cfg)
    b, t = 2, 16
    ids = torch.randint(0, 50368, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    logits = model(ids, mask)
    assert logits.shape == (b, t)
    p = model.drop_prob(ids, mask)
    assert (p >= 0).all() and (p <= 1).all()
    tags = torch.randint(0, 2, (b, t))
    out = model.loss(ids, mask, tags, drop_class_weight=2.5)
    out["loss"].backward()
    assert any(pp.grad is not None and pp.grad.abs().sum() > 0 for pp in model.export_core.head.parameters())
    assert any(pp.grad is not None and pp.grad.abs().sum() > 0 for pp in model.export_core.backbone.parameters())


def test_modernbert_d_model_guard():
    # A d_model that doesn't match the encoder hidden size must be rejected.
    if os.environ.get("POLYMORPH_TEST_MODERNBERT") != "1":
        pytest.skip("set POLYMORPH_TEST_MODERNBERT=1 to run the heavy ModernBERT load")
    cfg = LaMRConfig(vocab_size=50368, d_model=256, backbone="modernbert")  # wrong d_model
    with pytest.raises(ValueError, match="hidden_size"):
        LaMRModel(cfg)
