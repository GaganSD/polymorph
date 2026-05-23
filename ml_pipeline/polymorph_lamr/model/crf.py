"""Linear-chain CRF with weighted-emission NLL training and Viterbi decode.

Why hand-rolled instead of pytorch-crf:
  - pytorch-crf doesn't accept per-token soft weights, which we need for the
    AST hop-decay labels.
  - We export emissions to ONNX without the CRF layer (Viterbi runs in Rust),
    so we want clean separation between training-time NLL and inference-time
    decode.

Tag set is fixed to size 2: {0 = keep, 1 = drop}. Generalizing is trivial but
explicit shapes catch label bugs faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NUM_TAGS = 2


class LinearChainCRF(nn.Module):
    """Linear-chain CRF over 2 tags.

    Parameters:
        transitions: learnable (NUM_TAGS, NUM_TAGS) — transitions[i, j] is the
            score of moving from tag i to tag j.
        start_transitions: (NUM_TAGS,) initial-tag scores.
        end_transitions: (NUM_TAGS,) final-tag scores.
    """

    def __init__(self):
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros(NUM_TAGS, NUM_TAGS))
        self.start_transitions = nn.Parameter(torch.zeros(NUM_TAGS))
        self.end_transitions = nn.Parameter(torch.zeros(NUM_TAGS))

    def nll(
        self,
        emissions: torch.Tensor,  # (B, T, NUM_TAGS)
        tags: torch.Tensor,       # (B, T) int64 in [0, NUM_TAGS)
        mask: torch.Tensor,       # (B, T) bool — True = valid position
        weights: torch.Tensor | None = None,  # (B, T) float — per-token weight
    ) -> torch.Tensor:
        """Mean per-batch NLL, optionally weighted per token.

        Weights apply to the gold-path *emission* contribution only. Transition
        scores are unweighted — they encode a structural prior over tag
        sequences, not a per-token label statement. Padded positions (mask=0)
        are excluded from both numerator and partition.

        CRF arithmetic is forced to fp32 — `logsumexp` in bf16 loses 2-3 bits
        of precision per step, which compounds badly over T=1024 sequences.
        Emissions are upcast before the forward algorithm runs.
        """
        emissions = emissions.float()
        if weights is not None:
            weights = weights.float()
        score = self._gold_score(emissions, tags, mask, weights)
        partition = self._partition(emissions, mask)
        nll = partition - score  # (B,)
        # All-pad rows produce meaningless score/partition; zero their NLL so
        # they don't poison the batch mean.
        valid = (mask.long().sum(dim=1) > 0).to(nll.dtype)
        nll = nll * valid
        denom = valid.sum().clamp(min=1.0)
        return nll.sum() / denom

    def decode(
        self,
        emissions: torch.Tensor,  # (B, T, NUM_TAGS)
        mask: torch.Tensor,       # (B, T) bool
    ) -> list[list[int]]:
        return self._viterbi(emissions, mask)

    # ------------------- internals -------------------

    def _gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        weights: torch.Tensor | None,
    ) -> torch.Tensor:
        b, t, k = emissions.shape
        if weights is None:
            weights = torch.ones((b, t), dtype=emissions.dtype, device=emissions.device)
        mask_f = mask.to(emissions.dtype)

        # Start transition + emission at position 0.
        score = self.start_transitions[tags[:, 0]]  # (B,)
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1) * weights[:, 0] * mask_f[:, 0]

        for i in range(1, t):
            prev_tag = tags[:, i - 1]
            cur_tag = tags[:, i]
            trans_score = self.transitions[prev_tag, cur_tag]
            emit_score = emissions[:, i].gather(1, cur_tag.unsqueeze(1)).squeeze(1)
            step = (trans_score + emit_score * weights[:, i]) * mask_f[:, i]
            score = score + step

        # End transition at the last valid position per sequence. Clamp at 0
        # so an all-padding row doesn't index `tags[:, -1]` (would be silently
        # wrong rather than crash).
        seq_lens = (mask.long().sum(dim=1) - 1).clamp(min=0)  # (B,)
        last_tag = tags.gather(1, seq_lens.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tag]
        return score

    def _partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        b, t, k = emissions.shape
        # alpha[i, j] = log Σ paths ending at tag j up to position i.
        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, K)

        for i in range(1, t):
            emit = emissions[:, i].unsqueeze(1)              # (B, 1, K)
            trans = self.transitions.unsqueeze(0)            # (1, K, K)
            prev = alpha.unsqueeze(2)                        # (B, K, 1)
            next_alpha = torch.logsumexp(prev + trans + emit, dim=1)  # (B, K)

            # Mask: where the position is invalid, keep alpha unchanged.
            m = mask[:, i].unsqueeze(1).to(alpha.dtype)
            alpha = m * next_alpha + (1.0 - m) * alpha

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def _viterbi(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[list[int]]:
        b, t, k = emissions.shape
        history: list[torch.Tensor] = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # (B, K)

        for i in range(1, t):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, K, K)
            best_prev = broadcast.argmax(dim=1)                              # (B, K)
            best_score = broadcast.gather(1, best_prev.unsqueeze(1)).squeeze(1)
            next_score = best_score + emissions[:, i]                        # (B, K)

            m = mask[:, i].unsqueeze(1)
            score = torch.where(m, next_score, score)
            history.append(best_prev)

        score = score + self.end_transitions.unsqueeze(0)
        best_last = score.argmax(dim=1)  # (B,)

        # Backtrack per-sequence to honour per-sequence lengths.
        seq_lens = mask.long().sum(dim=1)
        results: list[list[int]] = []
        for batch_idx in range(b):
            length = int(seq_lens[batch_idx].item())
            if length == 0:
                results.append([])
                continue
            tag = int(best_last[batch_idx].item())
            tags_rev = [tag]
            # history has t-1 entries; honour `length` (could be shorter than t).
            for i in range(length - 2, -1, -1):
                tag = int(history[i][batch_idx, tag].item())
                tags_rev.append(tag)
            results.append(list(reversed(tags_rev)))
        return results
