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
        aux_ce_weight: float = 0.0,
        drop_class_weight: float = 1.0,
        crf_nll_weight: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Training objective: a weighted sum of the single linear-chain CRF's
        per-token NLL and a class-weighted token cross-entropy, both on the same
        emissions the Rust/ONNX path decodes with Viterbi (train == infer):

            loss = crf_nll_weight * crf_nll + aux_ce_weight * weighted_ce

        Why the aux CE: the keep/drop split is ~71/29, and the CRF NLL alone
        collapses to keep-all (high drop precision, ~0 recall, acc pinned at the
        baseline). The class-weighted CE (drop up-weighted by ``drop_class_weight``
        ~ the keep/drop frequency ratio) pushes balanced per-token supervision into
        the *emissions*.

        Why ``crf_nll_weight``: the CE only shapes emissions; the CRF *transitions*
        are trained solely by the CRF NLL, which on the imbalanced corpus drives a
        dominant keep->keep self-loop that suppresses drops at Viterbi decode — a
        transition-side collapse no emission-side knob can fix (observed: every run
        collapses at ~2 epochs regardless of CE/LR). Down-weighting the CRF NLL
        keeps the transitions near their flat init (flat transitions => Viterbi =
        per-token argmax of the balanced emissions), so the decode can't collapse.
        ``crf_nll_weight=0`` trains a pure class-weighted token classifier with the
        CRF acting as an identity smoother; ``=1`` (default) recovers prior
        behavior. With ``aux_ce_weight == 0`` the dict carries no ``aux_ce``.

        ``w_semantic`` / ``w_dependency`` (AST hop-decay soft labels) and
        ``lambda_*`` are reserved and unused; they stay in the signature for
        call-site compatibility and a possible future token-level auxiliary loss.
        """
        emissions = self.forward(input_ids, attention_mask)
        crf_nll = self.crf.nll(emissions, tags, attention_mask, reduction="token_mean")
        loss = crf_nll_weight * crf_nll
        out = {"crf_nll": crf_nll}
        if aux_ce_weight > 0.0:
            aux_ce = self._weighted_token_ce(emissions, tags, attention_mask, drop_class_weight)
            out["aux_ce"] = aux_ce
            loss = loss + aux_ce_weight * aux_ce
        out["loss"] = loss
        return out

    @staticmethod
    def _weighted_token_ce(
        emissions: torch.Tensor,        # (B, T, NUM_TAGS)
        tags: torch.Tensor,             # (B, T) int64 in [0, NUM_TAGS)
        attention_mask: torch.Tensor,   # (B, T) bool/0-1 — True = valid
        drop_class_weight: float,
    ) -> torch.Tensor:
        """Class-weighted per-token cross-entropy over valid (non-pad) positions.

        fp32 like the CRF forward (logits upcast) so bf16/fp16 autocast can't bias
        the softmax. Class weights are ``[keep=1.0, drop=drop_class_weight]`` (tag
        set is fixed: 0=keep, 1=drop). Returns weighted nats/token: the sum of
        per-token weighted CE over valid positions divided by the valid-token
        count — length-invariant and on the same per-token scale as the CRF's
        ``token_mean`` NLL, so ``aux_ce_weight`` is a clean relative knob.
        """
        k = emissions.shape[-1]
        logits = emissions.float().reshape(-1, k)   # (B*T, K)
        gold = tags.reshape(-1)                      # (B*T,)
        class_weight = torch.tensor(
            [1.0, float(drop_class_weight)], device=logits.device, dtype=logits.dtype
        )
        ce = torch.nn.functional.cross_entropy(
            logits, gold, weight=class_weight, reduction="none"
        )  # (B*T,) — each element already scaled by its target's class weight
        valid = attention_mask.reshape(-1).to(ce.dtype)
        return (ce * valid).sum() / valid.sum().clamp(min=1.0)
