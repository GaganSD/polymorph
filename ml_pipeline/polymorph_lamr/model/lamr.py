"""Full LaMR model: backbone + head gate + two emission heads + two CRFs.

The forward pass returns emissions and head weights. The CRFs live alongside so
they can be trained jointly, but ONNX export will only graph the backbone, gate,
and emission heads. Transitions ship as a side-car .npz at export time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import GatedDeltaNet2Stub
from .crf import NUM_TAGS, LinearChainCRF
from .head_gate import HeadGate


@dataclass
class LaMRConfig:
    vocab_size: int = 100352
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1


class LaMRBackboneForExport(nn.Module):
    """Wraps backbone + gate + emission heads only — no CRFs. This is the
    sub-module we export to ONNX.
    """

    def __init__(self, cfg: LaMRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = GatedDeltaNet2Stub(
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
        """Combine both CRF heads into one per-sequence weighted CRF."""
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
        tags: torch.Tensor,          # (B, T) — same gold labels for both heads
        w_semantic: torch.Tensor,    # (B, T)
        w_dependency: torch.Tensor,  # (B, T)
        lambda_sem: float = 1.0,
        lambda_dep: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        emi_sem, emi_dep, head_weights = self.forward(input_ids, attention_mask)
        nll_sem_vec = self.crf_semantic.nll(emi_sem, tags, attention_mask, w_semantic, reduction="none")
        nll_dep_vec = self.crf_dependency.nll(emi_dep, tags, attention_mask, w_dependency, reduction="none")
        weighted = lambda_sem * head_weights[:, 0] * nll_sem_vec + lambda_dep * head_weights[:, 1] * nll_dep_vec
        valid = (attention_mask.long().sum(dim=1) > 0).to(weighted.dtype)
        loss = (weighted * valid).sum() / valid.sum().clamp(min=1.0)
        nll_sem = (nll_sem_vec * valid).sum() / valid.sum().clamp(min=1.0)
        nll_dep = (nll_dep_vec * valid).sum() / valid.sum().clamp(min=1.0)
        return {
            "loss": loss,
            "nll_sem": nll_sem,
            "nll_dep": nll_dep,
            "head_weights": head_weights,
        }
