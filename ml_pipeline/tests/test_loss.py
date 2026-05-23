"""Joint loss == sum of per-head NLLs at λ=1."""

import torch

from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel


def test_joint_equals_sum_at_lambda_one():
    cfg = LaMRConfig(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
        n_experts=2,
        top_k=2,
    )
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    b, t = 1, 5
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.zeros((b, t), dtype=torch.long)
    w_s = torch.ones((b, t))
    w_d = torch.ones((b, t))

    out = model.joint_nll(ids, mask, tags, w_s, w_d, lambda_sem=1.0, lambda_dep=1.0)
    assert torch.isclose(out["loss"], out["nll_sem"] + out["nll_dep"], atol=1e-5)
