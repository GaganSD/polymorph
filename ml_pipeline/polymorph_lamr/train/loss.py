"""Drop-loss thin wrapper.

The actual loss computation lives in ``model.lamr.LaMRModel.loss`` — this module
exists so trainers and tests have a single place to compose loss weights and
reductions without poking at the model internals.
"""

from __future__ import annotations

import torch


def joint_loss(
    model_outputs: dict[str, torch.Tensor],
    lambda_sem: float = 1.0,
    lambda_dep: float = 1.0,
) -> torch.Tensor:
    """Return the trained scalar objective from ``LaMRModel.loss``.

    LaMR optimises a class-weighted per-token binary cross-entropy on the drop
    logits (``sigmoid(logit) = P(drop)``); there is no CRF. ``loss`` already
    composes that into ``model_outputs["loss"]`` (and exposes the ``bce``
    component for logging), so this wrapper just returns it. ``lambda_sem`` /
    ``lambda_dep`` are retained for call-site/config compatibility but are inert.
    This stays the single place to graft further auxiliary terms without aliasing
    ``loss``.
    """
    return model_outputs["loss"]
