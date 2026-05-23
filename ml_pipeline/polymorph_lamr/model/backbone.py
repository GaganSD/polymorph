"""Stubbed Gated DeltaNet-2 backbone.

This is a *placeholder* — a small ONNX-friendly feed-forward encoder that
exposes the same `(B, T) -> (B, T, D)` interface a real Gated DeltaNet-2 will.
When the real kernel is available, swap this class without touching the rest of
the pipeline.

Search marker for the swap point: `_TODO_REAL_DELTANET`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SinusoidalPosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class _FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, ff_mult: int, dropout: float):
        super().__init__()
        hidden = d_model * ff_mult
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))
        return x + self.dropout(y)


class GatedDeltaNet2Stub(nn.Module):
    """_TODO_REAL_DELTANET — replace with the actual Gated DeltaNet-2 kernel.

    The replacement must:
      - accept (B, T) token ids,
      - return (B, T, D) hidden states,
      - support attention masking via a (B, T) bool mask,
      - be ONNX-exportable with a dynamic T axis.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = _SinusoidalPosEnc(d_model)
        self.blocks = nn.ModuleList([_FeedForwardBlock(d_model, ff_mult, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.pos(x)
        if attention_mask is not None:
            x = x * attention_mask.to(x.dtype).unsqueeze(-1)
        for block in self.blocks:
            x = block(x)
            if attention_mask is not None:
                x = x * attention_mask.to(x.dtype).unsqueeze(-1)
        return self.norm(x)
