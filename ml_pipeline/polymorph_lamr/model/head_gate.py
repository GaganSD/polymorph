"""Two-head gate for balancing semantic and dependency CRFs.

The gate produces one `(semantic, dependency)` weight pair per sequence. It is
mask-aware so padding never changes the routing decision.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HeadGate(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.router = nn.Linear(d_model, 2)

    def forward(
        self,
        hidden: torch.Tensor,  # (B, T, D)
        attention_mask: torch.Tensor | None = None,  # (B, T), True = valid
    ) -> torch.Tensor:
        if attention_mask is None:
            pooled = hidden.mean(dim=1)
        else:
            mask = attention_mask.to(hidden.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = (hidden * mask).sum(dim=1) / denom
        return F.softmax(self.router(pooled), dim=-1)
