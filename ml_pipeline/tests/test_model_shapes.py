"""Forward pass shape + gradient flow checks on a tiny LaMR model."""

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
        n_experts=3,
        top_k=2,
        expert_hidden_mult=2,
    )


def test_forward_shapes():
    cfg = _tiny_cfg()
    model = LaMRModel(cfg)
    b, t = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    sem, dep = model(ids, mask)
    assert sem.shape == (b, t, 2)
    assert dep.shape == (b, t, 2)


def test_joint_loss_flows_gradients():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    b, t = 2, 8
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.randint(0, 2, (b, t))
    w_s = torch.full((b, t), 0.6)
    w_d = torch.full((b, t), 0.4)

    out = model.joint_nll(ids, mask, tags, w_s, w_d)
    loss = out["loss"]
    assert torch.isfinite(loss)
    loss.backward()

    # At least one parameter in each major sub-module must receive a non-zero gradient.
    def _has_grad(module):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())

    assert _has_grad(model.export_core.backbone)
    assert _has_grad(model.export_core.moe)
    assert _has_grad(model.export_core.head_semantic)
    assert _has_grad(model.export_core.head_dependency)
    assert _has_grad(model.crf_semantic)
    assert _has_grad(model.crf_dependency)
