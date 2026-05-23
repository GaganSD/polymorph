"""Full LaMR model: backbone + MoE gate + two emission heads + two CRFs.

The forward pass returns emissions only — the CRFs live alongside so they can
be trained jointly, but ONNX export will only graph the backbone, gate, and
emission heads. Transitions ship as a side-car .npz at export time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import GatedDeltaNet2Stub
from .crf import NUM_TAGS, LinearChainCRF
from .moe_gate import MoEGate


@dataclass
class LaMRConfig:
    vocab_size: int = 100352
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ff_mult: int = 4
    dropout: float = 0.1
    n_experts: int = 4
    top_k: int = 2
    expert_hidden_mult: int = 2


class LaMRBackboneForExport(nn.Module):
    """Wraps backbone + MoE + emission heads only — no CRFs. This is the
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
        self.moe = MoEGate(
            d_model=cfg.d_model,
            n_experts=cfg.n_experts,
            top_k=cfg.top_k,
            expert_hidden_mult=cfg.expert_hidden_mult,
        )
        self.head_semantic = nn.Linear(cfg.d_model, NUM_TAGS)
        self.head_dependency = nn.Linear(cfg.d_model, NUM_TAGS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(input_ids, attention_mask)
        mixed = self.moe(hidden)
        return self.head_semantic(mixed), self.head_dependency(mixed)


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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.export_core(input_ids, attention_mask)

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
        emi_sem, emi_dep = self.forward(input_ids, attention_mask)
        nll_sem = self.crf_semantic.nll(emi_sem, tags, attention_mask, w_semantic)
        nll_dep = self.crf_dependency.nll(emi_dep, tags, attention_mask, w_dependency)
        total = lambda_sem * nll_sem + lambda_dep * nll_dep
        return {"loss": total, "nll_sem": nll_sem.detach(), "nll_dep": nll_dep.detach()}
