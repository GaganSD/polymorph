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
    """Return the trained scalar objective: the blended-CRF NLL.

    `model_outputs` is the dict from ``LaMRModel.joint_nll``. Since the C1 fix,
    LaMR optimises a single blended CRF (the head gate routes the blend of both
    heads' emissions + transitions; see ``LaMRModel.joint_nll``), so there is one
    ``loss`` to return — it is NOT a re-weighted sum of the two per-head NLLs
    (blending happens inside one CRF partition, which is non-linear in those
    NLLs). ``lambda_sem``/``lambda_dep`` are retained for call-site/config
    compatibility but no longer reweight per-head losses; the gate, learned
    end-to-end, subsumes head weighting. This wrapper stays as the single place
    to graft auxiliary terms (e.g. a router load-balance regulariser) without
    aliasing ``loss``.
    """
    return model_outputs["loss"]
