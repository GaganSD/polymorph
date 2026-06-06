"""Forward shape + gradient flow + objective checks on a tiny LaMR model.

The pruner is a per-token binary classifier: forward returns one drop logit per
token (B, T); ``sigmoid`` gives P(drop). No CRF.
"""

import torch
import torch.nn.functional as F

from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel


def _tiny_cfg() -> LaMRConfig:
    return LaMRConfig(
        vocab_size=128,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
    )


def test_forward_shape():
    cfg = _tiny_cfg()
    model = LaMRModel(cfg)
    b, t = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    logits = model(ids, mask)
    assert logits.shape == (b, t)
    assert torch.isfinite(logits).all()


def test_drop_prob_in_unit_interval():
    cfg = _tiny_cfg()
    model = LaMRModel(cfg)
    b, t = 2, 9
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    p = model.drop_prob(ids, mask)
    assert p.shape == (b, t)
    assert (p >= 0).all() and (p <= 1).all()


def test_loss_flows_gradients():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    b, t = 2, 8
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))

    out = model.loss(ids, mask, tags, drop_class_weight=2.5)
    loss = out["loss"]
    assert torch.isfinite(loss)
    assert torch.allclose(out["bce"], loss)  # bce IS the loss (no other terms)
    loss.backward()

    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert _has_grad(model.export_core.backbone)
    assert _has_grad(model.export_core.head)


def test_loss_is_token_mean_weighted_bce():
    """The objective is the mean over valid tokens of class-weighted BCE-with-logits
    on the drop logits (pos_weight = drop_class_weight)."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 2, 7
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))
    with torch.no_grad():
        out = model.loss(ids, mask, tags, drop_class_weight=2.5)
        logits = model(ids, mask).float()
        bce = F.binary_cross_entropy_with_logits(
            logits, tags.float(), pos_weight=torch.tensor(2.5), reduction="none"
        )
        valid = mask.float()
        expected = (bce * valid).sum() / valid.sum()
    assert torch.allclose(out["loss"], expected, atol=1e-5)


def test_loss_ignores_padding():
    """Padded (mask==False) positions never contribute to the loss, regardless of
    their tag/logit."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 1, 6
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    full_mask = torch.ones((b, t), dtype=torch.bool)
    part_mask = full_mask.clone()
    part_mask[0, 4:] = False  # pad the tail
    tags = torch.ones((b, t), dtype=torch.long)
    with torch.no_grad():
        # Flip the padded tags arbitrarily; loss over the valid prefix is unchanged.
        tags_a = tags.clone()
        tags_b = tags.clone()
        tags_b[0, 4:] = 0
        la = model.loss(ids, part_mask, tags_a, drop_class_weight=2.0)["loss"]
        lb = model.loss(ids, part_mask, tags_b, drop_class_weight=2.0)["loss"]
    assert torch.allclose(la, lb, atol=1e-6)


def test_drop_class_weight_scales_drop_token_loss():
    """Up-weighting the drop class raises the BCE when the gold tokens are drop —
    the lever that separates the drop ranking."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 2, 6
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.ones((b, t), dtype=torch.long)  # every token is 'drop'
    with torch.no_grad():
        lo = model.loss(ids, mask, tags, drop_class_weight=1.0)["loss"]
        hi = model.loss(ids, mask, tags, drop_class_weight=3.0)["loss"]
    assert hi > lo
