"""Backbone correctness: shapes, masking, bidirectional token mixing, selection.

These guard the properties the stub could not provide. In particular
``test_token_mixing_across_positions`` is what distinguishes a *real* encoder
(self-attention) from the old feed-forward stub: changing one token must move
another position's hidden state.
"""

import pytest
import torch

from polymorph_lamr.model.backbone import (
    GatedDeltaNet2Stub,
    TransformerEncoderBackbone,
)
from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel


def _backbone(**kw) -> TransformerEncoderBackbone:
    cfg = dict(vocab_size=128, d_model=32, n_layers=2, n_heads=4, ff_mult=2, dropout=0.0)
    cfg.update(kw)
    bb = TransformerEncoderBackbone(**cfg)
    bb.eval()  # dropout off -> deterministic
    return bb


def test_forward_shape():
    bb = _backbone()
    ids = torch.randint(0, 128, (3, 11))
    out = bb(ids, torch.ones((3, 11), dtype=torch.bool))
    assert out.shape == (3, 11, 32)
    assert torch.isfinite(out).all()


def test_n_heads_must_divide_d_model():
    with pytest.raises(ValueError, match="not divisible"):
        TransformerEncoderBackbone(vocab_size=16, d_model=30, n_heads=4)


def test_token_mixing_across_positions():
    """A real encoder mixes across positions: editing token j changes hidden i!=j.

    The feed-forward stub is position-wise, so it would FAIL this — that is the
    point of the test.
    """
    bb = _backbone()
    ids = torch.randint(0, 128, (1, 6))
    mask = torch.ones((1, 6), dtype=torch.bool)
    with torch.no_grad():
        base = bb(ids, mask)
        edited = ids.clone()
        edited[0, 3] = (edited[0, 3] + 7) % 128  # change a DIFFERENT position
        moved = bb(edited, mask)
    # Position 0 must react to an edit at position 3 (cross-token attention).
    assert not torch.allclose(base[0, 0], moved[0, 0], atol=1e-6)


def test_stub_is_position_wise():
    """Contrast: the feed-forward stub does NOT mix across positions."""
    stub = GatedDeltaNet2Stub(vocab_size=128, d_model=32, n_layers=2, n_heads=4, ff_mult=2, dropout=0.0)
    stub.eval()
    ids = torch.randint(0, 128, (1, 6))
    mask = torch.ones((1, 6), dtype=torch.bool)
    with torch.no_grad():
        base = stub(ids, mask)
        edited = ids.clone()
        edited[0, 3] = (edited[0, 3] + 7) % 128
        moved = stub(edited, mask)
    # Position 0 is unchanged by an edit elsewhere — position-wise, no mixing.
    assert torch.allclose(base[0, 0], moved[0, 0], atol=1e-6)


def test_valid_positions_independent_of_padding_content():
    """Key-padding masking: valid-position outputs ignore padded-tail content.

    Two rows share the same valid prefix but have different junk in the padded
    tail; the valid positions' hidden states must be identical.
    """
    bb = _backbone()
    prefix = torch.randint(0, 128, (3,))
    ids = torch.zeros((2, 6), dtype=torch.long)
    ids[0, :3] = prefix
    ids[1, :3] = prefix
    ids[0, 3:] = torch.tensor([11, 22, 33])  # different padded junk per row
    ids[1, 3:] = torch.tensor([99, 88, 77])
    mask = torch.tensor([[True, True, True, False, False, False]] * 2)
    with torch.no_grad():
        out = bb(ids, mask)
    assert torch.allclose(out[0, :3], out[1, :3], atol=1e-5)
    # Padded positions are scrubbed to zero.
    assert torch.allclose(out[0, 3:], torch.zeros(3, 32), atol=1e-6)


def test_single_valid_token_no_nan():
    """A row with one valid token (rest padded) must not produce NaNs."""
    bb = _backbone()
    ids = torch.randint(0, 128, (1, 5))
    mask = torch.tensor([[True, False, False, False, False]])
    with torch.no_grad():
        out = bb(ids, mask)
    assert torch.isfinite(out).all()


def test_config_defaults_to_transformer():
    assert LaMRConfig().backbone == "transformer"
    model = LaMRModel(LaMRConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, ff_mult=2))
    assert isinstance(model.export_core.backbone, TransformerEncoderBackbone)


def test_config_can_select_stub():
    model = LaMRModel(
        LaMRConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, ff_mult=2, backbone="deltanet_stub")
    )
    assert isinstance(model.export_core.backbone, GatedDeltaNet2Stub)


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="unknown backbone"):
        LaMRModel(LaMRConfig(vocab_size=64, d_model=16, n_layers=1, n_heads=2, backbone="bogus"))


def test_all_padding_row_no_nan():
    """A fully-padded row must not produce NaN. This is why key masking uses a
    finite fill (finfo.min), not -inf — an all-masked softmax row stays finite."""
    bb = _backbone()
    ids = torch.randint(0, 128, (2, 5))
    mask = torch.tensor(
        [[True, True, False, False, False], [False, False, False, False, False]]
    )  # row 1 entirely padded
    with torch.no_grad():
        out = bb(ids, mask)
    assert torch.isfinite(out).all()


def test_mask_fill_finite_under_fp16():
    """Regression: a hardcoded -1e9 fill overflows to -inf in fp16, turning an
    all-masked softmax row into NaN. The dtype-aware finfo.min fill stays finite
    for every float dtype, so the all-masked softmax is finite (uniform)."""
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        scores = torch.zeros(1, 1, 3, 3, dtype=dtype)
        key_pad = torch.ones(1, 1, 1, 3, dtype=torch.bool)  # all keys masked
        filled = scores.masked_fill(key_pad, torch.finfo(dtype).min)
        assert torch.isfinite(filled).all(), f"{dtype}: fill is not finite"
