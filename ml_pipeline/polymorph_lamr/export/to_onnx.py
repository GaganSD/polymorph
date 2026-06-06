"""Export the trained LaMR pruner to ONNX.

Output layout:
    <out_dir>/
        model.onnx          # backbone + drop head -> per-token drop logits (B, T)
        decode.json         # decode contract: sigmoid + target-rate threshold
        config.yaml         # mirrors the training config (for reproducibility)
        README.md           # how the Rust side loads this artifact
        parity.json         # max-abs diff between torch and onnxruntime

There is NO CRF and NO transitions side-car: the model emits one drop logit per
token, and the runtime decodes by thresholding ``sigmoid(logit)`` at the cutoff
that drops a target fraction of the (unlocked) tokens — see ``src/lamr.rs`` and
``decode.json``.

The Rust runtime is expected to:
    1. Load model.onnx via tract or ort.
    2. Run the model over the FULL token sequence (matching how training tags
       every token of the chunk) to get the `logits` tensor (B, T).
    3. ``p_drop = sigmoid(logits)``; among UNLOCKED tokens, drop the top
       ``round(target_rate * n_unlocked)`` by ``p_drop`` (or threshold at a fixed
       cutoff). target_rate is a runtime knob (see decode.json default).
    4. Force-keep locked positions (see enforce_lock_invariant in src/lamr.rs).
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch
import yaml

from ..model.lamr import LaMRConfig, LaMRModel

DEFAULT_TARGET_RATE = 0.30


def _load_checkpoint(path: Path) -> tuple[LaMRModel, LaMRConfig]:
    # weights_only=True blocks arbitrary pickle code execution. We allowlist
    # LaMRConfig since the checkpoint dict includes a `cfg` field as a plain
    # dict (we never pickled the dataclass itself), but be explicit so future
    # code that does pickle the dataclass still loads.
    try:
        torch.serialization.add_safe_globals([LaMRConfig])
    except AttributeError:
        # Older torch without safe_globals; fall through to legacy load.
        pass
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Legacy / dataclass-pickled checkpoints: explicit opt-in.
        blob = torch.load(path, map_location="cpu", weights_only=False)
    cfg_keys = {f.name for f in fields(LaMRConfig)}
    cfg = LaMRConfig(**{k: v for k, v in blob["cfg"].items() if k in cfg_keys})
    model = LaMRModel(cfg)
    model.load_state_dict(blob["model_state"])
    model.eval()
    return model, cfg


def export(
    checkpoint: Path,
    out_dir: Path,
    config_path: Path | None = None,
    opset: int = 17,
    parity_seq_len: int = 64,
    target_rate: float = DEFAULT_TARGET_RATE,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = _load_checkpoint(checkpoint)
    core = model.export_core
    core.eval()

    # If the config carries an eval.target_rate, prefer it for the decode default.
    if config_path is not None and config_path.exists():
        try:
            cfg_yaml = yaml.safe_load(config_path.read_text())
            target_rate = float(cfg_yaml.get("eval", {}).get("target_rate", target_rate))
        except Exception:
            pass

    # Dummy inputs.
    dummy_ids = torch.randint(0, cfg.vocab_size, (1, parity_seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, parity_seq_len), dtype=torch.bool)

    onnx_path = out_dir / "model.onnx"
    _export_static_graph(core, dummy_ids, dummy_mask, onnx_path, opset, prefer_dynamo=True)

    # Decode contract side-car (replaces the old transitions.json). The runtime
    # needs no learned params beyond the ONNX weights: decode is sigmoid + a
    # threshold calibrated to `default_target_rate` (a runtime knob).
    (out_dir / "decode.json").write_text(
        json.dumps(
            {
                "decode": "sigmoid_target_rate_threshold",
                "default_target_rate": float(target_rate),
                "logit_output": "logits",
                "note": (
                    "p_drop = sigmoid(logits[token]); among unlocked tokens drop the "
                    "top round(target_rate * n_unlocked) by p_drop. target_rate is a "
                    "runtime knob; locked tokens are always kept."
                ),
            },
            indent=2,
        )
    )

    # Mirror config for downstream reproducibility.
    if config_path is not None and config_path.exists():
        shutil.copy(config_path, out_dir / "config.yaml")
    else:
        (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))

    # Parity check across alternate dynamic shapes.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    try:
        parity = _check_parity(core, sess, cfg, parity_seq_len)
    except Exception:
        _export_static_graph(core, dummy_ids, dummy_mask, onnx_path, opset, prefer_dynamo=False)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        parity = _check_parity(core, sess, cfg, parity_seq_len)
    (out_dir / "parity.json").write_text(json.dumps(parity, indent=2))

    _write_readme(out_dir, cfg, target_rate)
    return parity


def _export_static_graph(
    core: torch.nn.Module,
    dummy_ids: torch.Tensor,
    dummy_mask: torch.Tensor,
    onnx_path: Path,
    opset: int,
    prefer_dynamo: bool,
) -> None:
    names = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
        "dynamic_axes": {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch", 1: "seq"},
        },
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if prefer_dynamo:
        try:
            torch.onnx.export(core, (dummy_ids, dummy_mask), str(onnx_path), dynamo=True, **names)
            return
        except Exception:
            pass

    mha_backend = getattr(torch.backends, "mha", None)
    set_fastpath = getattr(mha_backend, "set_fastpath_enabled", None)
    previous_fastpath = getattr(mha_backend, "get_fastpath_enabled", lambda: None)()
    try:
        if set_fastpath is not None:
            set_fastpath(False)
        torch.onnx.export(core, (dummy_ids, dummy_mask), str(onnx_path), dynamo=False, **names)
    finally:
        if set_fastpath is not None and previous_fastpath is not None:
            set_fastpath(previous_fastpath)


def _check_parity(core: torch.nn.Module, sess, cfg: LaMRConfig, parity_seq_len: int) -> dict[str, float]:
    shapes = [
        (1, parity_seq_len),
        (1, max(1, parity_seq_len // 2 + 1)),
        (2, max(2, min(parity_seq_len, 7))),
    ]
    max_logit = 0.0
    for b, t in shapes:
        ids = torch.randint(0, cfg.vocab_size, (b, t), dtype=torch.long)
        mask = torch.ones((b, t), dtype=torch.bool)
        if t > 2:
            mask[-1, -1] = False
        with torch.no_grad():
            torch_logits = core(ids, mask)
        (onnx_logits,) = sess.run(
            ["logits"],
            {"input_ids": ids.numpy(), "attention_mask": mask.numpy()},
        )
        max_logit = max(max_logit, float(np.max(np.abs(torch_logits.numpy() - onnx_logits))))
    return {
        "max_abs_diff_logits": max_logit,
        "checked_shapes": len(shapes),
    }


def _write_readme(out_dir: Path, cfg: LaMRConfig, target_rate: float) -> None:
    readme = f"""# LaMR ONNX Artifact

Inference pipeline for Polymorph's Rust MCP runtime.

## Files
- `model.onnx` — backbone + drop head -> per-token drop logits.
- `decode.json` — decode contract (sigmoid + target-rate threshold, default {target_rate}).
- `config.yaml` — model architecture (must match the training run).
- `parity.json` — max-abs diff between torch and onnxruntime at export time.

## Inputs / outputs
- input_ids: (B, T) int64, cl100k_base token ids
- attention_mask: (B, T) bool
- logits: (B, T) float, per-token drop logit (sigmoid -> P(drop))

## Decode (Rust side)
0. Run the model over the FULL token sequence (matching training; locked tokens
   are NOT removed before the model — they are force-kept afterwards).
1. Run ONNX session → the `logits` tensor (B, T).
2. `p_drop = sigmoid(logits)`. Among UNLOCKED tokens, drop the top
   `round(target_rate * n_unlocked)` by `p_drop` (target_rate is a runtime knob;
   see `decode.json`). No CRF, no Viterbi, no transitions.
3. Force-keep locked positions (always `false`) via `enforce_lock_invariant`
   (see `src/lamr.rs`).

## Model config
- vocab_size: {cfg.vocab_size}
- d_model: {cfg.d_model}
- n_layers: {cfg.n_layers}
"""
    (out_dir / "README.md").write_text(readme)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export LaMR to ONNX (drop logits + decode contract).")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--parity-seq-len", type=int, default=64)
    p.add_argument("--target-rate", type=float, default=DEFAULT_TARGET_RATE,
                   help="default decode target drop rate written to decode.json")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parity = export(
        checkpoint=args.ckpt,
        out_dir=args.out,
        config_path=args.config,
        opset=args.opset,
        parity_seq_len=args.parity_seq_len,
        target_rate=args.target_rate,
    )
    print(f"exported to {args.out}; parity={parity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
