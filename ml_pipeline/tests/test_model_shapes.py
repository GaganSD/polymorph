"""Forward shape + gradient flow + objective checks on a tiny LaMR model."""

import torch

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
    emissions = model(ids, mask)
    assert emissions.shape == (b, t, 2)
    assert torch.isfinite(emissions).all()


def test_joint_loss_flows_gradients():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    b, t = 2, 8
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))

    out = model.joint_nll(ids, mask, tags)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()

    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert _has_grad(model.export_core.backbone)
    assert _has_grad(model.export_core.head)
    assert _has_grad(model.crf)


def test_joint_nll_is_single_crf_token_mean():
    """With aux_ce_weight=0 (default) the optimized loss is the per-token NLL of
    the SAME single CRF that inference decodes (train == infer), and no aux term
    leaks in."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 2, 7
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))
    with torch.no_grad():
        out = model.joint_nll(ids, mask, tags)
        emissions = model(ids, mask)
        expected = model.crf.nll(emissions, tags, mask, reduction="token_mean")
    assert torch.allclose(out["loss"], expected, atol=1e-5)
    assert torch.allclose(out["crf_nll"], expected, atol=1e-5)
    assert "aux_ce" not in out  # disabled by default


def test_joint_nll_adds_class_weighted_ce():
    """With aux_ce_weight>0 the objective is crf_nll + w * class_weighted_token_CE,
    computed on the same emissions the CRF consumes."""
    import torch.nn.functional as F

    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 2, 7
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))
    with torch.no_grad():
        out = model.joint_nll(ids, mask, tags, aux_ce_weight=0.5, drop_class_weight=2.5)
        emissions = model(ids, mask)
        crf_nll = model.crf.nll(emissions, tags, mask, reduction="token_mean")
        w = torch.tensor([1.0, 2.5])
        ce = F.cross_entropy(emissions.float().reshape(-1, 2), tags.reshape(-1), weight=w, reduction="none")
        valid = mask.reshape(-1).float()
        ce = (ce * valid).sum() / valid.sum().clamp(min=1.0)
        expected = crf_nll + 0.5 * ce
    assert torch.allclose(out["aux_ce"], ce, atol=1e-5)
    assert torch.allclose(out["loss"], expected, atol=1e-5)


def test_crf_nll_weight_scales_and_zeroes_crf_term():
    """crf_nll_weight scales the CRF NLL in the total; at 0 the objective is the
    pure class-weighted CE (transitions get no gradient, stay flat)."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 2, 7
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))
    with torch.no_grad():
        out = model.joint_nll(ids, mask, tags, aux_ce_weight=1.0, drop_class_weight=2.5, crf_nll_weight=0.0)
        # crf_nll still reported (for logging) but contributes 0 to the loss.
        assert torch.allclose(out["loss"], out["aux_ce"], atol=1e-5)
        # A half-weighted CRF lands halfway between pure-CE and full.
        half = model.joint_nll(ids, mask, tags, aux_ce_weight=1.0, drop_class_weight=2.5, crf_nll_weight=0.5)
        expected_half = 0.5 * out["crf_nll"] + out["aux_ce"]
        assert torch.allclose(half["loss"], expected_half, atol=1e-5)


def test_crf_nll_weight_zero_freezes_transitions():
    """With crf_nll_weight=0 the CRF transitions receive no gradient (only the
    emissions learn), so a flat-init CRF stays flat -> Viterbi = argmax."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    b, t = 2, 6
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))
    out = model.joint_nll(ids, mask, tags, aux_ce_weight=1.0, drop_class_weight=2.5, crf_nll_weight=0.0)
    out["loss"].backward()
    # Emissions head learns; CRF transition params get no gradient.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.export_core.head.parameters())
    for p in model.crf.parameters():
        assert p.grad is None or p.grad.abs().sum() == 0


def test_drop_class_weight_scales_drop_token_loss():
    """Up-weighting the drop class raises the aux CE when the gold tokens are drop —
    the lever that pulls the model off the keep-all collapse."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 2, 6
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.ones((b, t), dtype=torch.long)  # every token is 'drop'
    with torch.no_grad():
        lo = model.joint_nll(ids, mask, tags, aux_ce_weight=1.0, drop_class_weight=1.0)["aux_ce"]
        hi = model.joint_nll(ids, mask, tags, aux_ce_weight=1.0, drop_class_weight=3.0)["aux_ce"]
    assert hi > lo


def test_joint_loss_with_aux_ce_flows_gradients():
    """The composite (CRF NLL + class-weighted CE) backprops into backbone, head,
    and CRF."""
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    b, t = 2, 8
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))

    out = model.joint_nll(ids, mask, tags, aux_ce_weight=1.0, drop_class_weight=2.5)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()

    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert _has_grad(model.export_core.backbone)
    assert _has_grad(model.export_core.head)
    assert _has_grad(model.crf)
