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
        # Reserved/inert kwargs kept for call-site + config compatibility. The AST
        # hop-decay soft labels (w_semantic / w_dependency) and lambda_* are not
        # wired into the loss; they stay in the signature for a possible future
        # token-level auxiliary and so existing call sites don't break.
        w_semantic: torch.Tensor | None = None,
        w_dependency: torch.Tensor | None = None,
        lambda_sem: float = 1.0,
        lambda_dep: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Class-weighted binary cross-entropy over valid (non-pad) tokens.

            loss = mean_over_valid( BCEWithLogits(logit, gold; pos_weight=drop_w) )

        The keep/drop split is ~71/29, so an unweighted objective drifts toward
        keep-all. ``drop_class_weight`` (``pos_weight``, ~ the keep/drop frequency
        ratio) up-weights the drop (positive) class so the *ranking* of which
        tokens to drop is well separated. Absolute calibration is handled at
        decode time by the target-rate threshold, NOT by this weight — the weight
        only shapes the score ordering the threshold then cuts.

        bf16/fp16 autocast is bypassed (logits upcast to fp32) so the sigmoid
        isn't biased by low precision over long T.
        """
        logits = self.forward(input_ids, attention_mask)
        bce = self._weighted_token_bce(logits, tags, attention_mask, drop_class_weight)
        return {"loss": bce, "bce": bce}

    @staticmethod
    def _weighted_token_bce(
        logits: torch.Tensor,           # (B, T) drop logits
        tags: torch.Tensor,             # (B, T) int in {0, 1}
        attention_mask: torch.Tensor,   # (B, T) bool/0-1 — True = valid
        drop_class_weight: float,
    ) -> torch.Tensor:
        """Class-weighted per-token BCE-with-logits over valid positions.

        ``pos_weight`` scales the loss contribution of drop (gold==1) tokens.
        Returns nats/token: summed weighted BCE over valid positions divided by
        the valid-token count — length-invariant, so ``drop_class_weight`` is a
        clean relative knob and long sequences don't dominate.
        """
        logits = logits.float()
        gold = tags.float()
        pos_weight = torch.tensor(float(drop_class_weight), device=logits.device, dtype=logits.dtype)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, gold, pos_weight=pos_weight, reduction="none"
        )  # (B, T)
        valid = attention_mask.to(bce.dtype)
        return (bce * valid).sum() / valid.sum().clamp(min=1.0)
