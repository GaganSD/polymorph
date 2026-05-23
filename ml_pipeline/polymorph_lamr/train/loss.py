"""Joint CRF NLL loss thin wrapper.

The actual NLL computation lives in `model.lamr.LaMRModel.joint_nll` — this
module exists so trainers and tests have a single place to compose loss
weights and reductions without poking at the model internals.
"""

from __future__ import annotations

import torch


def joint_loss(
    model_outputs: dict[str, torch.Tensor],
    lambda_sem: float = 1.0,
    lambda_dep: float = 1.0,
) -> torch.Tensor:
    """Re-weight the per-head NLLs into a single scalar.

    `model_outputs` is expected to be the dict from LaMRModel.joint_nll. We
    return a fresh sum so callers can mix this loss with auxiliary terms
    (e.g. router load-balance regularisers) without aliasing `loss`.
    """
    return lambda_sem * model_outputs["nll_sem"] + lambda_dep * model_outputs["nll_dep"]
