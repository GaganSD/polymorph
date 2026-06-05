"""joint_loss returns the single-CRF per-token objective the model trains."""

import torch

from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel
from polymorph_lamr.train.loss import joint_loss


def test_joint_loss_returns_single_crf_nll():
    """The wrapper returns the model's trained objective: the per-token NLL of the
    single linear-chain CRF (the same CRF inference decodes)."""
    cfg = LaMRConfig(
        vocab_size=64,
        d_model=16,
        n_layers=1,
        n_heads=2,
        ff_mult=2,
        dropout=0.0,
    )
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()
    b, t = 1, 5
    ids = torch.randint(0, cfg.vocab_size, (b, t))
    mask = torch.ones((b, t), dtype=torch.bool)
    tags = torch.zeros((b, t), dtype=torch.long)

    out = model.joint_nll(ids, mask, tags)
    assert torch.equal(joint_loss(out), out["loss"])

    emissions = model(ids, mask)
    expected = model.crf.nll(emissions, tags, mask, reduction="token_mean")
    assert torch.isclose(out["loss"], expected, atol=1e-5)
