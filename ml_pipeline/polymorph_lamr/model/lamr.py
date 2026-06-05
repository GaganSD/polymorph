"""Full LaMR model: backbone + head gate + two emission heads + two CRFs.

The forward pass returns emissions and head weights. The CRFs live alongside so
they can be trained jointly, but ONNX export will only graph the backbone, gate,
and emission heads. Transitions ship as a side-car .npz at export time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import GatedDeltaNet2Stub, TransformerEncoderBackbone
from .crf import NUM_TAGS, LinearChainCRF
from .head_gate import HeadGate

# Selectable backbones. "transformer" is the real bidirectional encoder (default);
# "deltanet_stub" is the feed-forward placeholder kept for fallback/ablation.
_BACKBONES = {
    "transformer": TransformerEncoderBackbone,
    "deltanet_stub": GatedDeltaNet2Stub,
}


@dataclass
class LaMRConfig:
    vocab_size: int = 100352
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    backbone: str = "transformer"


class LaMRBackboneForExport(nn.Module):
    """Wraps backbone + gate + emission heads only — no CRFs. This is the
    sub-module we export to ONNX.
    """

    def __init__(self, cfg: LaMRConfig):
        super().__init__()
        self.cfg = cfg
        backbone_cls = _BACKBONES.get(cfg.backbone)
        if backbone_cls is None:
            raise ValueError(
                f"unknown backbone {cfg.backbone!r}; choose from {sorted(_BACKBONES)}"
            )
        self.backbone = backbone_cls(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            ff_mult=cfg.ff_mult,
            dropout=cfg.dropout,
        )
        self.head_gate = HeadGate(d_model=cfg.d_model)
        self.head_semantic = nn.Linear(cfg.d_model, NUM_TAGS)
        self.head_dependency = nn.Linear(cfg.d_model, NUM_TAGS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.backbone(input_ids, attention_mask)
        head_weights = self.head_gate(hidden, attention_mask)
        return self.head_semantic(hidden), self.head_dependency(hidden), head_weights


class LaMRModel(nn.Module):
    """Training-time module. Adds two CRF heads on top of the export backbone."""

    def __init__(self, cfg: LaMRConfig):
        super().__init__()
        self.cfg = cfg
        self.export_core = LaMRBackboneForExport(cfg)
        self.crf_semantic = LinearChainCRF()
        self.crf_dependency = LinearChainCRF()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.export_core(input_ids, attention_mask)

    def weighted_crf_parameters(
        self,
        emi_sem: torch.Tensor,
        emi_dep: torch.Tensor,
        head_weights: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Combine both CRF heads into one per-sequence weighted CRF.

        Emissions and head_weights are upcast to fp32 before the blend so that
        training (which may run under bf16/fp16 autocast) blends at the same
        precision the Rust/ONNX inference path uses (fp32), keeping the optimized
        objective numerically aligned with what Viterbi decodes.
        """
        emi_sem = emi_sem.float()
        emi_dep = emi_dep.float()
        head_weights = head_weights.float()
        w_sem = head_weights[:, 0].view(-1, 1, 1)
        w_dep = head_weights[:, 1].view(-1, 1, 1)
        return {
            "emissions": w_sem * emi_sem + w_dep * emi_dep,
            "transitions": w_sem * self.crf_semantic.transitions + w_dep * self.crf_dependency.transitions,
            "start_transitions": head_weights[:, 0:1] * self.crf_semantic.start_transitions
            + head_weights[:, 1:2] * self.crf_dependency.start_transitions,
            "end_transitions": head_weights[:, 0:1] * self.crf_semantic.end_transitions
            + head_weights[:, 1:2] * self.crf_dependency.end_transitions,
        }

    def joint_nll(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: torch.Tensor,                         # (B, T) gold labels (0=keep, 1=drop)
        w_semantic: torch.Tensor | None = None,     # (B, T) reserved — see note
        w_dependency: torch.Tensor | None = None,   # (B, T) reserved — see note
        lambda_sem: float = 1.0,                    # reserved; kept for call-site compatibility
        lambda_dep: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Training objective: NLL of the *blended* CRF that inference decodes.

        The head gate emits per-sequence weights that blend the two CRF heads
        (emissions + transitions) into a single per-sequence CRF — exactly the CRF
        the Rust/ONNX path runs Viterbi over (see ``weighted_crf_parameters``). We
        optimise the NLL of that same blended CRF, so training and inference share
        one model. (Previously training summed two independent CRF NLLs while
        inference decoded one blended CRF — a silent train/infer mismatch.)

        ``w_semantic`` / ``w_dependency`` (AST hop-decay soft labels) are NOT
        applied to the decoded CRF: inference has no access to them, so folding
        them into the loss would re-introduce a train/infer gap. They stay in the
        signature, reserved for a future token-level auxiliary loss. ``nll_sem`` /
        ``nll_dep`` are reported for diagnostics only (do the heads diverge?).
        """
        emi_sem, emi_dep, head_weights = self.forward(input_ids, attention_mask)
        params = self.weighted_crf_parameters(emi_sem, emi_dep, head_weights)
        loss = self.crf_semantic.nll_with_params(
            params["emissions"],
            tags,
            attention_mask,
            params["transitions"],
            params["start_transitions"],
            params["end_transitions"],
            reduction="mean",
        )
        with torch.no_grad():
            nll_sem = self.crf_semantic.nll(emi_sem, tags, attention_mask, reduction="mean")
            nll_dep = self.crf_dependency.nll(emi_dep, tags, attention_mask, reduction="mean")
        return {
            "loss": loss,
            "nll_sem": nll_sem,
            "nll_dep": nll_dep,
            "head_weights": head_weights,
        }
