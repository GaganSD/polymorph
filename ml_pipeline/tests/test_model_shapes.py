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
    """The optimized loss is the per-token NLL of the SAME single CRF that
    inference decodes (train == infer)."""
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
