"""Query-adaptive top-k Mixture-of-Experts gate.

Given a sequence of hidden states `(B, T, D)` and an optional query embedding
`(B, D)` (defaulting to mean-pooled hidden states), route each *token* through
the top-k experts selected per-sequence. Output: a per-token mix of expert
MLP outputs.

Designed for ONNX export: no dynamic-shape Python control flow, no
DeepSpeed-style dispatch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Expert(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * hidden_mult)
        self.fc2 = nn.Linear(d_model * hidden_mult, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class MoEGate(nn.Module):
    def __init__(self, d_model: int, n_experts: int = 4, top_k: int = 2, expert_hidden_mult: int = 2):
        super().__init__()
        if top_k > n_experts:
            raise ValueError("top_k cannot exceed n_experts")
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [_Expert(d_model, expert_hidden_mult) for _ in range(n_experts)]
        )

    def forward(
        self,
        hidden: torch.Tensor,  # (B, T, D)
        query_embed: torch.Tensor | None = None,  # (B, D); defaults to mean-pool
    ) -> torch.Tensor:
        b, t, d = hidden.shape
        if query_embed is None:
            query_embed = hidden.mean(dim=1)  # (B, D)

        # Per-sequence routing weights.
        logits = self.router(query_embed)  # (B, n_experts)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # (B, K)
        topk_w = F.softmax(topk_vals, dim=-1)  # (B, K)

        # Build a (B, n_experts) gate weight vector; zero outside top-k.
        gate = torch.zeros_like(logits)
        gate.scatter_(1, topk_idx, topk_w)

        # Run all experts (small n_experts so this is cheap and stays
        # ONNX-friendly), then mix.
        expert_out = torch.stack([e(hidden) for e in self.experts], dim=1)  # (B, E, T, D)
        gate_b = gate.unsqueeze(-1).unsqueeze(-1)  # (B, E, 1, 1)
        mixed = (expert_out * gate_b).sum(dim=1)  # (B, T, D)
        return mixed
