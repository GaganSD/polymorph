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
    )


def test_forward_shapes():
    cfg = _tiny_cfg()
    model = LaMRModel(cfg)
    b, t = 2, 16
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    sem, dep, head_weights = model(ids, mask)
    assert sem.shape == (b, t, 2)
    assert dep.shape == (b, t, 2)
    assert head_weights.shape == (b, 2)
    assert torch.allclose(head_weights.sum(dim=-1), torch.ones(b), atol=1e-6)


def test_head_gate_ignores_padding_values():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 6))
    ids[1, 3:] = torch.randint(0, cfg.vocab_size, (3,))
    mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, False, False, False]])
    ids[1, :3] = ids[0, :3]
    with torch.no_grad():
        _, _, weights = model(ids, mask)
    assert torch.allclose(weights[0], weights[1], atol=1e-6)


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
    assert _has_grad(model.export_core.head_gate)
    assert _has_grad(model.export_core.head_semantic)
    assert _has_grad(model.export_core.head_dependency)
    assert _has_grad(model.crf_semantic)
    assert _has_grad(model.crf_dependency)


def test_weighted_crf_parameters_shapes_and_decode_parity():
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    b, t = 2, 5
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    sem, dep, head_weights = model(ids, mask)
    params = model.weighted_crf_parameters(sem, dep, head_weights)
    assert params["emissions"].shape == (b, t, 2)
    assert params["transitions"].shape == (b, 2, 2)
    assert params["start_transitions"].shape == (b, 2)
    assert params["end_transitions"].shape == (b, 2)

    decoded = model.crf_semantic.decode_with_params(mask=mask, **params)
    for i in range(b):
        single = model.crf_semantic.decode_with_params(
            emissions=params["emissions"][i : i + 1],
            mask=mask[i : i + 1],
            transitions=params["transitions"][i : i + 1],
            start_transitions=params["start_transitions"][i : i + 1],
            end_transitions=params["end_transitions"][i : i + 1],
        )
        assert decoded[i] == single[0]
