"""Backbone encoders for LaMR.

Two implementations share one interface — ``(B, T)`` token ids -> ``(B, T, D)``
hidden states, optional ``(B, T)`` bool padding mask (True = valid), ONNX-
exportable with a dynamic ``T`` axis:

* ``TransformerEncoderBackbone`` (default) — a real bidirectional pre-norm
  Transformer encoder. Bidirectional self-attention is the correct inductive
  bias here: LaMR tags every token keep/drop over a *fully observed* log chunk
  (not autoregressive generation), then a per-token drop head scores each
  token — the textbook bidirectional-encoder sequence-labelling stack. At
  T<=1024 full O(T^2) attention is cheap, and the graph is plain
  MatMul/Softmax/LayerNorm so ONNX export stays clean (no custom recurrent scan
  to go brittle).

* ``GatedDeltaNet2Stub`` — the original feed-forward placeholder, kept for the
  ``backbone: deltanet_stub`` config path and as a fallback. It does no token
  mixing, so it cannot learn context-dependent keep/drop; it exists only to keep
  the interface pinned.

Why a Transformer encoder rather than a literal Gated DeltaNet-2 (the old
``_TODO_REAL_DELTANET`` marker): Gated DeltaNet is a *causal* linear-attention
RNN aimed at long-context generation. Using it for this task would mean a
bidirectional two-pass wrapper plus exporting a custom scan to ONNX — trading
robustness for exoticness with no quality upside at this sequence length. The
encoder is the defensible SOTA choice for per-token sequence labelling.
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


class _MultiHeadSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with key-padding masking.

    Pre-norm + residual. Padded *keys* are filled with ``finfo(dtype).min`` (a
    dtype-aware large-negative, NOT a hardcoded -1e9 — that overflows to -inf
    under fp16 autocast and turns an all-padding row's softmax into NaN). With
    the dtype-aware fill, valid queries put ~zero weight on padding so every
    valid position's output is independent of padding content (the property
    ``test_valid_positions_independent_of_padding_content`` relies on), and an all-padding row
    yields a finite uniform softmax instead of NaN. No causal mask: the task is
    bidirectional sequence labelling, so each token may attend both ways.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.norm = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,  # (B, T, D)
        key_pad_mask: torch.Tensor | None = None,  # (B, 1, 1, T) bool, True = padded key
    ) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        k = self.k_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        if key_pad_mask is not None:
            # dtype-aware fill: safe under fp16/bf16/fp32 (finfo.min never -inf).
            scores = scores.masked_fill(key_pad_mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)  # (B,H,T,hd)
        ctx = ctx.transpose(1, 2).reshape(b, t, d)
        return x + self.dropout(self.out_proj(ctx))


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float):
        super().__init__()
        self.attn = _MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = _FeedForwardBlock(d_model, ff_mult, dropout)

    def forward(self, x: torch.Tensor, key_pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attn(x, key_pad_mask)
        x = self.ff(x)
        return x


class TransformerEncoderBackbone(nn.Module):
    """Real bidirectional pre-norm Transformer encoder.

    ``(B, T)`` token ids -> ``(B, T, D)`` hidden states. Supports a ``(B, T)``
    bool padding mask (True = valid) and exports to ONNX with a dynamic ``T``
    axis. Drop-in replacement for ``GatedDeltaNet2Stub``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        **_kwargs,  # tolerate backbone-specific kwargs (e.g. encoder_name) from the factory
    ):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = _SinusoidalPosEnc(d_model)
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_EncoderBlock(d_model, n_heads, ff_mult, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.pos(x)
        x = self.embed_dropout(x)

        key_pad_mask = None
        if attention_mask is not None:
            # (B, 1, 1, T) bool, True where the key position is padding. Broadcasts
            # over heads and query positions inside attention.
            key_pad_mask = (~attention_mask.bool()).unsqueeze(1).unsqueeze(1)

        for block in self.blocks:
            x = block(x, key_pad_mask)
        x = self.norm(x)

        if attention_mask is not None:
            # Scrub padded positions to 0 (cosmetic; the loss/decode mask them anyway).
            x = torch.where(attention_mask.bool().unsqueeze(-1), x, torch.zeros_like(x))
        return x


class GatedDeltaNet2Stub(nn.Module):
    """Feed-forward placeholder kept for the ``deltanet_stub`` backbone path.

    Does NO token mixing — it cannot learn context-dependent keep/drop. Use the
    real ``TransformerEncoderBackbone`` for training; this exists only to pin the
    interface and as a trivial fallback.

    The replacement contract (which ``TransformerEncoderBackbone`` satisfies):
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
        **_kwargs,  # tolerate backbone-specific kwargs from the factory
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


class ModernBertBackbone(nn.Module):
    """Pretrained ModernBERT encoder as the backbone (``backbone: modernbert``).

    The SOTA lever for #33: replace the from-scratch 100k-row cl100k embedding
    (25.7M of 28.8M params, the dominant capacity sink) with a pretrained
    bidirectional encoder that already understands token structure. The drop head
    sits on top of ``last_hidden_state`` unchanged.

    Interface contract (same as the other backbones): ``(B, T)`` token ids ->
    ``(B, T, hidden_size)`` hidden states, optional ``(B, T)`` mask (True/1 =
    valid). NOTE the tokenizer caveat: ModernBERT does NOT use cl100k — input_ids
    must be ModernBERT-tokenizer ids, so the training shards and the Rust runtime
    tokenization both switch with this backbone (tracked in #33).

    Exportability: loaded with ``attn_implementation="eager"`` and
    ``reference_compile=False`` so ONNX tracing produces a tract-compatible graph.
    The dynamic-seq export hits a Flatten shape-analysis failure in tract — export
    at a FIXED window length (see to_onnx + the #33 de-risk finding).
    """

    def __init__(
        self,
        vocab_size: int | None = None,
        d_model: int = 768,
        n_layers: int | None = None,
        n_heads: int | None = None,
        ff_mult: int | None = None,
        dropout: float = 0.0,
        encoder_name: str = "answerdotai/ModernBERT-base",
        **_kwargs,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "backbone='modernbert' needs `transformers` (pip install "
                "'polymorph-lamr[encoder]' or `uv pip install transformers`)."
            ) from e
        self.encoder = AutoModel.from_pretrained(encoder_name, attn_implementation="eager")
        if hasattr(self.encoder.config, "reference_compile"):
            # torch.compile of submodules breaks ONNX tracing; off for export parity.
            self.encoder.config.reference_compile = False
        hidden = int(self.encoder.config.hidden_size)
        if hidden != d_model:
            raise ValueError(
                f"config d_model={d_model} must equal the {encoder_name} hidden_size={hidden}; "
                f"set model.d_model: {hidden} for this encoder."
            )
        self.d_model = hidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is not None:
            attention_mask = attention_mask.to(input_ids.dtype)
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        if attention_mask is not None:
            # Scrub padded positions (cosmetic; the loss/decode mask them anyway).
            h = torch.where(attention_mask.bool().unsqueeze(-1), h, torch.zeros_like(h))
        return h
