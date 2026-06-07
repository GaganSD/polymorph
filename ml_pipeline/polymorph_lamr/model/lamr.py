"""Full LaMR model: backbone + one per-token drop head (sigmoid).

The pruner is a binary per-token classifier: for each token it emits a single
*drop logit*; ``sigmoid(logit)`` is the probability that the token can be
dropped. There is no CRF and no Viterbi — decode is a threshold on the
per-token drop probability, calibrated to a target compression rate (see
``eval.evaluate`` and ``src/lamr.rs``).

Why no CRF: the earlier design wrapped the emissions in a 2-tag linear-chain CRF
and decoded with Viterbi. Probing trained checkpoints showed the CRF was
degenerate — its ranking of which tokens to drop was stable (ROC-AUC ~0.78
across checkpoints) while only the *global keep/drop bias* oscillated, so the
"keep-all collapse" we chased was a calibration artifact, not a ranking failure.
A threshold calibrated to a target drop-rate neutralizes that bias and makes the
compression ratio a runtime knob. The CRF transitions, the transitions side-car,
and the Rust Viterbi all went away with it.

An even earlier design had two emission heads + a gate. It was removed first
(the AST per-token weights that would differentiate the heads were never wired
into the loss). Re-introduce a dual head as v1 if/when those weights drive the
loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .backbone import GatedDeltaNet2Stub, ModernBertBackbone, TransformerEncoderBackbone

# Selectable backbones. "transformer" is the real from-scratch bidirectional
# encoder (default); "modernbert" is a pretrained ModernBERT encoder (the #33
# SOTA lever — needs `transformers` + its own tokenizer); "deltanet_stub" is the
# feed-forward placeholder kept for fallback/ablation.
_BACKBONES = {
    "transformer": TransformerEncoderBackbone,
    "modernbert": ModernBertBackbone,
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
    # Only used when backbone == "modernbert": the HF encoder to load. d_model
    # must equal this encoder's hidden_size (768 for ModernBERT-base).
    encoder_name: str = "answerdotai/ModernBERT-base"


class LaMRBackboneForExport(nn.Module):
    """Backbone + drop head — the sub-module exported to ONNX.

    Forward returns one drop *logit* per token, shape ``(B, T)``. The ONNX graph
    exports exactly this; the runtime applies ``sigmoid`` + a calibrated
    threshold. No CRF / transitions are involved.
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
            encoder_name=getattr(cfg, "encoder_name", "answerdotai/ModernBERT-base"),
        )
        # Single drop logit per token (binary keep/drop). sigmoid(logit) = P(drop).
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.backbone(input_ids, attention_mask)
        return self.head(hidden).squeeze(-1)  # (B, T) drop logits


class LaMRModel(nn.Module):
    """Training-time module: the export core + the binary drop objective."""

    def __init__(self, cfg: LaMRConfig):
        super().__init__()
        self.cfg = cfg
        self.export_core = LaMRBackboneForExport(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.export_core(input_ids, attention_mask)  # drop logits (B, T)

    @torch.no_grad()
    def drop_prob(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """P(drop) per token, ``(B, T)`` in [0, 1] — the value the decode thresholds."""
        return torch.sigmoid(self.forward(input_ids, attention_mask).float())

    def loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tags: torch.Tensor,                         # (B, T) gold labels (0=keep, 1=drop)
        drop_class_weight: float = 1.0,
        # ``w_semantic`` is repurposed (Phase 0e) as a per-token KEEP-SALIENCE
        # weight: tokens inside answer-bearing spans (free-text root_cause / msg /
        # resolution values, structural atoms) carry weight > 1 so the loss
        # punishes dropping them harder, directly targeting answer-needle survival.
        # ``w_dependency`` / ``lambda_*`` remain reserved/inert.
        w_semantic: torch.Tensor | None = None,
        w_dependency: torch.Tensor | None = None,
        lambda_sem: float = 1.0,
        lambda_dep: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Class-weighted, salience-weighted per-token BCE over valid tokens.

            loss = mean_over_valid( w_sal · BCEWithLogits(logit, gold; pos_weight=drop_w) )

        The keep/drop split is ~71/29, so an unweighted objective drifts toward
        keep-all. ``drop_class_weight`` (``pos_weight``) up-weights the drop class
        so the *ranking* separates; absolute calibration is a decode-time threshold.
        ``w_semantic`` (per-token salience, default 1) additionally up-weights the
        tokens whose survival the benchmark measures, so the ranker learns to rank
        answer-needle tokens as keep even under a tight budget.

        bf16/fp16 autocast is bypassed (logits upcast to fp32) so the sigmoid
        isn't biased by low precision over long T.
        """
        logits = self.forward(input_ids, attention_mask)
        bce = self._weighted_token_bce(
            logits, tags, attention_mask, drop_class_weight, salience=w_semantic
        )
        return {"loss": bce, "bce": bce}

    @staticmethod
    def _weighted_token_bce(
        logits: torch.Tensor,           # (B, T) drop logits
        tags: torch.Tensor,             # (B, T) int in {0, 1}
        attention_mask: torch.Tensor,   # (B, T) bool/0-1 — True = valid
        drop_class_weight: float,
        salience: torch.Tensor | None = None,  # (B, T) per-token keep-salience weight
    ) -> torch.Tensor:
        """Class- and salience-weighted per-token BCE-with-logits over valid tokens.

        ``pos_weight`` scales the loss contribution of drop (gold==1) tokens.
        ``salience`` (default all-ones) is a per-token multiplier that up-weights
        answer-bearing tokens so the ranker protects them under budget. Returns a
        weighted mean (sum of weighted BCE / sum of effective weights) so it stays
        length-invariant and ``drop_class_weight`` remains a clean relative knob.
        """
        logits = logits.float()
        gold = tags.float()
        pos_weight = torch.tensor(float(drop_class_weight), device=logits.device, dtype=logits.dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, gold, pos_weight=pos_weight, reduction="none"
        )  # (B, T)
        valid = attention_mask.to(bce.dtype)
        if salience is not None:
            # Default salience is 1.0; only answer-bearing tokens exceed it. Clamp
            # to >=0 and floor the per-token weight at 1 so a malformed/zero weight
            # never silently zeroes a valid token's gradient.
            w = salience.to(bce.dtype).clamp(min=0.0)
            w = torch.where(w < 1.0, torch.ones_like(w), w) * valid
            return (bce * w).sum() / w.sum().clamp(min=1.0)
        return (bce * valid).sum() / valid.sum().clamp(min=1.0)
