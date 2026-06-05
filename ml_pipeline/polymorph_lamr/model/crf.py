"""Linear-chain CRF with weighted NLL training and Viterbi decode.

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
        reduction: str = "mean",
    ) -> torch.Tensor:
        """CRF NLL, optionally weighted per token.

        Per-token weights scale emission potentials before both the gold-path
        score and the partition function are computed. Transition scores remain
        unweighted: they encode a structural prior over tag sequences rather
        than a per-token label statement.

        CRF arithmetic is forced to fp32 — `logsumexp` in bf16 loses 2-3 bits
        of precision per step, which compounds badly over T=1024 sequences.
        Emissions are upcast before the forward algorithm runs.
        """
        emissions = emissions.float()
        if weights is not None:
            weights = weights.float()
            emissions = emissions * weights.unsqueeze(-1)
        score = self._gold_score(emissions, tags, mask)
        partition = self._partition(emissions, mask)
        nll = partition - score  # (B,)
        # All-pad rows produce meaningless score/partition; zero their NLL so
        # they don't poison the batch mean.
        valid = (mask.long().sum(dim=1) > 0).to(nll.dtype)
        nll = nll * valid
        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            denom = valid.sum().clamp(min=1.0)
            return nll.sum() / denom
        if reduction == "token_mean":
            # Per-token NLL (nats/token): length-invariant and smooth, and weights
            # every token equally rather than every sequence (a long sequence no
            # longer dominates the loss just by being long).
            return nll.sum() / mask.long().sum().clamp(min=1).to(nll.dtype)
        raise ValueError(f"unknown reduction: {reduction}")

    def nll_with_params(
        self,
        emissions: torch.Tensor,           # (B, T, NUM_TAGS)
        tags: torch.Tensor,                # (B, T) int64 in [0, NUM_TAGS)
        mask: torch.Tensor,                # (B, T) bool — True = valid position
        transitions: torch.Tensor,         # (NUM_TAGS, NUM_TAGS) or (B, NUM_TAGS, NUM_TAGS)
        start_transitions: torch.Tensor,   # (NUM_TAGS,) or (B, NUM_TAGS)
        end_transitions: torch.Tensor,     # (NUM_TAGS,) or (B, NUM_TAGS)
        reduction: str = "mean",
    ) -> torch.Tensor:
        """CRF NLL using *explicit* (optionally per-sequence batched) transition
        params instead of this module's own ``self.transitions``.

        A general utility for decoding/scoring with per-sequence CRF params. fp32
        like ``nll`` (logsumexp precision over long T). The default LaMR objective
        uses ``nll`` directly on the single CRF; this stays available for batched-
        param scoring and is exercised by the tests.
        """
        emissions = emissions.float()
        transitions = transitions.float()
        start_transitions = start_transitions.float()
        end_transitions = end_transitions.float()
        score = self._gold_score(emissions, tags, mask, transitions, start_transitions, end_transitions)
        partition = self._partition(emissions, mask, transitions, start_transitions, end_transitions)
        nll = partition - score
        valid = (mask.long().sum(dim=1) > 0).to(nll.dtype)
        nll = nll * valid
        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        if reduction == "mean":
            return nll.sum() / valid.sum().clamp(min=1.0)
        if reduction == "token_mean":
            return nll.sum() / mask.long().sum().clamp(min=1).to(nll.dtype)
        raise ValueError(f"unknown reduction: {reduction}")

    def decode(
        self,
        emissions: torch.Tensor,  # (B, T, NUM_TAGS)
        mask: torch.Tensor,       # (B, T) bool
    ) -> list[list[int]]:
        return self._viterbi(
            emissions,
            mask,
            transitions=self.transitions,
            start_transitions=self.start_transitions,
            end_transitions=self.end_transitions,
        )

    @staticmethod
    def decode_with_params(
        emissions: torch.Tensor,
        mask: torch.Tensor,
        transitions: torch.Tensor,
        start_transitions: torch.Tensor,
        end_transitions: torch.Tensor,
    ) -> list[list[int]]:
        crf = LinearChainCRF()
        return crf._viterbi(
            emissions,
            mask,
            transitions=transitions,
            start_transitions=start_transitions,
            end_transitions=end_transitions,
        )

    # ------------------- internals -------------------

    def _gold_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        transitions: torch.Tensor | None = None,
        start_transitions: torch.Tensor | None = None,
        end_transitions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, k = emissions.shape
        mask_f = mask.to(emissions.dtype)
        trans = self._batch_matrix(transitions if transitions is not None else self.transitions, b)
        start = self._batch_vector(start_transitions if start_transitions is not None else self.start_transitions, b)
        end = self._batch_vector(end_transitions if end_transitions is not None else self.end_transitions, b)

        # Start transition + emission at position 0.
        batch_idx = torch.arange(b, device=emissions.device)
        score = start[batch_idx, tags[:, 0]]  # (B,)
        score = score + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1) * mask_f[:, 0]

        for i in range(1, t):
            prev_tag = tags[:, i - 1]
            cur_tag = tags[:, i]
            trans_score = trans[batch_idx, prev_tag, cur_tag]
            emit_score = emissions[:, i].gather(1, cur_tag.unsqueeze(1)).squeeze(1)
            step = (trans_score + emit_score) * mask_f[:, i]
            score = score + step

        # End transition at the last valid position per sequence. Clamp at 0
        # so an all-padding row doesn't index `tags[:, -1]` (would be silently
        # wrong rather than crash).
        seq_lens = (mask.long().sum(dim=1) - 1).clamp(min=0)  # (B,)
        last_tag = tags.gather(1, seq_lens.unsqueeze(1)).squeeze(1)
        score = score + end[batch_idx, last_tag] * (mask.long().sum(dim=1) > 0).to(score.dtype)
        return score

    def _partition(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        transitions: torch.Tensor | None = None,
        start_transitions: torch.Tensor | None = None,
        end_transitions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, k = emissions.shape
        trans = self._batch_matrix(transitions if transitions is not None else self.transitions, b)
        start = self._batch_vector(start_transitions if start_transitions is not None else self.start_transitions, b)
        end = self._batch_vector(end_transitions if end_transitions is not None else self.end_transitions, b)
        # alpha[i, j] = log Σ paths ending at tag j up to position i.
        alpha = start + emissions[:, 0]  # (B, K)

        for i in range(1, t):
            emit = emissions[:, i].unsqueeze(1)              # (B, 1, K)
            prev = alpha.unsqueeze(2)                        # (B, K, 1)
            next_alpha = torch.logsumexp(prev + trans + emit, dim=1)  # (B, K)

            # Mask: where the position is invalid, keep alpha unchanged.
            m = mask[:, i].unsqueeze(1).to(alpha.dtype)
            alpha = m * next_alpha + (1.0 - m) * alpha

        alpha = alpha + end
        return torch.logsumexp(alpha, dim=1)  # (B,)

    def _viterbi(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        transitions: torch.Tensor,
        start_transitions: torch.Tensor,
        end_transitions: torch.Tensor,
    ) -> list[list[int]]:
        b, t, k = emissions.shape
        trans = self._batch_matrix(transitions, b)
        start = self._batch_vector(start_transitions, b)
        end = self._batch_vector(end_transitions, b)
        history: list[torch.Tensor] = []
        score = start + emissions[:, 0]  # (B, K)

        for i in range(1, t):
            broadcast = score.unsqueeze(2) + trans  # (B, K, K)
            best_prev = broadcast.argmax(dim=1)                              # (B, K)
            best_score = broadcast.gather(1, best_prev.unsqueeze(1)).squeeze(1)
            next_score = best_score + emissions[:, i]                        # (B, K)

            m = mask[:, i].unsqueeze(1)
            score = torch.where(m, next_score, score)
            history.append(best_prev)

        score = score + end
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

    @staticmethod
    def _batch_vector(param: torch.Tensor, batch_size: int) -> torch.Tensor:
        if param.dim() == 1:
            return param.unsqueeze(0).expand(batch_size, -1)
        if param.dim() == 2 and param.shape[0] == batch_size:
            return param
        raise ValueError(f"cannot batch CRF vector with shape {tuple(param.shape)}")

    @staticmethod
    def _batch_matrix(param: torch.Tensor, batch_size: int) -> torch.Tensor:
        if param.dim() == 2:
            return param.unsqueeze(0).expand(batch_size, -1, -1)
        if param.dim() == 3 and param.shape[0] == batch_size:
            return param
        raise ValueError(f"cannot batch CRF matrix with shape {tuple(param.shape)}")
