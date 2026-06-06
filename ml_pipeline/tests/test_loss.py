"""joint_loss returns the per-token BCE drop objective the model trains."""

import torch
import torch.nn.functional as F

from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel
from polymorph_lamr.train.loss import joint_loss


def test_joint_loss_returns_model_loss():
    """The wrapper returns the model's trained objective: the class-weighted
    per-token BCE on the drop logits."""
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

    out = model.loss(ids, mask, tags, drop_class_weight=2.5)
    assert torch.equal(joint_loss(out), out["loss"])

    logits = model(ids, mask).float()
    bce = F.binary_cross_entropy_with_logits(
        logits, tags.float(), pos_weight=torch.tensor(2.5), reduction="none"
    )
    valid = mask.float()
    expected = (bce * valid).sum() / valid.sum()
    assert torch.isclose(out["loss"], expected, atol=1e-5)


def test_joint_loss_lambdas_are_inert():
    """lambda_sem/lambda_dep are vestigial and must not change the returned loss."""
    out = {"loss": torch.tensor(2.5), "bce": torch.tensor(2.5)}
    assert torch.equal(joint_loss(out, lambda_sem=1.0, lambda_dep=1.0), out["loss"])
    assert torch.equal(joint_loss(out, lambda_sem=2.0, lambda_dep=0.5), out["loss"])
