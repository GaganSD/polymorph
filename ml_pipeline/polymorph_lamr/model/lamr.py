"""Full LaMR model: backbone + one emission head + one linear-chain CRF.

The forward pass returns per-token tag emissions; ONNX export graphs the backbone
+ head, and the CRF transitions ship as a side-car at export time.

An earlier design had two emission heads + a gate that blended them into one CRF.
It was removed: the signal that would differentiate the two heads — the AST
per-token weights — was never wired into the loss (reserved), so the gate added
training instability (it oscillated between keep-all and over-drop) for no
benefit. Re-introduce a dual head as v1 if/when those weights drive the loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import GatedDeltaNet2Stub, TransformerEncoderBackbone
from .crf import NUM_TAGS, LinearChainCRF

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
    """Backbone + emission head — the sub-module exported to ONNX (no CRF)."""

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
        self.head = nn.Linear(cfg.d_model, NUM_TAGS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids, attention_mask)
        return self.head(hidden)  # (B, T, NUM_TAGS)


class LaMRModel(nn.Module):
    """Training-time module: the export core + one linear-chain CRF."""

    def __init__(self, cfg: LaMRConfig):
        super().__init__()
        self.cfg = cfg
        self.export_core = LaMRBackboneForExport(cfg)
        self.crf = LinearChainCRF()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.export_core(input_ids, attention_mask)  # emissions (B, T, NUM_TAGS)

    def joint_nll(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: torch.Tensor,                         # (B, T) gold labels (0=keep, 1=drop)
        w_semantic: torch.Tensor | None = None,     # reserved (unused) — see note
        w_dependency: torch.Tensor | None = None,   # reserved (unused) — see note
        lambda_sem: float = 1.0,                    # reserved; kept for call-site compatibility
        lambda_dep: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Training objective: per-token NLL of the single linear-chain CRF — the
        exact CRF the Rust/ONNX path decodes with Viterbi (so train == infer).

        ``w_semantic`` / ``w_dependency`` (AST hop-decay soft labels) and
        ``lambda_*`` are reserved and unused; they stay in the signature for
        call-site compatibility and a possible future token-level auxiliary loss.
        """
        emissions = self.forward(input_ids, attention_mask)
        loss = self.crf.nll(emissions, tags, attention_mask, reduction="token_mean")
        return {"loss": loss}
