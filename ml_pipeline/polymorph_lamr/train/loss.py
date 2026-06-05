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
    """Return the trained scalar objective from ``LaMRModel.joint_nll``.

    LaMR optimises a single linear-chain CRF (the exact CRF inference decodes),
    optionally plus a class-weighted token-CE auxiliary on the same emissions to
    counter keep/drop majority-class collapse. ``joint_nll`` already composes that
    sum into ``model_outputs["loss"]`` (and exposes the ``crf_nll`` / ``aux_ce``
    components for logging), so this wrapper just returns it. ``lambda_sem`` /
    ``lambda_dep`` are retained for call-site/config compatibility but are inert.
    This stays the single place to graft further auxiliary terms without aliasing
    ``loss``.
    """
    return model_outputs["loss"]
