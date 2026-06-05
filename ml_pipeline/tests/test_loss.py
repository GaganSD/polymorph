"""joint_loss returns the single blended-CRF objective the model trains (post-C1)."""

import torch

from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel
from polymorph_lamr.train.loss import joint_loss


def test_joint_loss_returns_blended_crf_nll():
    """The trained objective is the NLL of the blended CRF that inference decodes
    — NOT a sum of the two per-head NLLs. Blending happens inside one CRF
    partition (non-linear in the per-head NLLs), so the two are different
    quantities; conflating them was the C1 train/infer mismatch.
    """
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

    # The wrapper returns the model's single trained objective.
    assert torch.equal(joint_loss(out), out["loss"])

    # That objective is exactly the blended-CRF NLL (train == infer).
    sem, dep, hw = model(ids, mask)
    params = model.weighted_crf_parameters(sem, dep, hw)
    expected = model.crf_semantic.nll_with_params(
        params["emissions"],
        tags,
        mask,
        params["transitions"],
        params["start_transitions"],
        params["end_transitions"],
        reduction="token_mean",
    )
    assert torch.isclose(out["loss"], expected, atol=1e-5)
