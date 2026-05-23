"""Export the trained LaMR model to ONNX + side-car transition matrices.

Output layout:
    <out_dir>/
        model.onnx          # backbone + MoE + two emission heads
        transitions.npz     # {sem_trans, sem_start, sem_end, dep_trans, dep_start, dep_end}
        config.yaml         # mirrors the training config (for reproducibility)
        README.md           # how the Rust side loads this artifact
        parity.json         # max-abs diff between torch and onnxruntime

The Rust runtime is expected to:
    1. Load model.onnx via tract or ort.
    2. Concatenate emissions into one (T, 2) per head.
    3. Run Viterbi using the transitions in transitions.npz.
    4. Map decoded tag-1 positions back to the unlocked-token indices and
       return a drop_mask parallel to the lock_mask (see src/lamr.rs).
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import yaml

from ..model.lamr import LaMRConfig, LaMRModel


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
    cfg = LaMRConfig(**blob["cfg"])
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
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = _load_checkpoint(checkpoint)
    core = model.export_core
    core.eval()

    # Dummy inputs.
    dummy_ids = torch.randint(0, cfg.vocab_size, (1, parity_seq_len), dtype=torch.long)
    dummy_mask = torch.ones((1, parity_seq_len), dtype=torch.bool)

    onnx_path = out_dir / "model.onnx"
    torch.onnx.export(
        core,
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["emissions_sem", "emissions_dep"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "emissions_sem": {0: "batch", 1: "seq"},
            "emissions_dep": {0: "batch", 1: "seq"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Transitions side-car.
    sem = model.crf_semantic
    dep = model.crf_dependency
    np.savez(
        out_dir / "transitions.npz",
        sem_trans=sem.transitions.detach().cpu().numpy(),
        sem_start=sem.start_transitions.detach().cpu().numpy(),
        sem_end=sem.end_transitions.detach().cpu().numpy(),
        dep_trans=dep.transitions.detach().cpu().numpy(),
        dep_start=dep.start_transitions.detach().cpu().numpy(),
        dep_end=dep.end_transitions.detach().cpu().numpy(),
    )

    # Mirror config for downstream reproducibility.
    if config_path is not None and config_path.exists():
        shutil.copy(config_path, out_dir / "config.yaml")
    else:
        (out_dir / "config.yaml").write_text(yaml.safe_dump(asdict(cfg)))

    # Parity check.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_sem, torch_dep = core(dummy_ids, dummy_mask)
    onnx_sem, onnx_dep = sess.run(
        ["emissions_sem", "emissions_dep"],
        {
            "input_ids": dummy_ids.numpy(),
            "attention_mask": dummy_mask.numpy(),
        },
    )
    diff_sem = float(np.max(np.abs(torch_sem.numpy() - onnx_sem)))
    diff_dep = float(np.max(np.abs(torch_dep.numpy() - onnx_dep)))
    parity = {"max_abs_diff_sem": diff_sem, "max_abs_diff_dep": diff_dep}
    (out_dir / "parity.json").write_text(json.dumps(parity, indent=2))

    _write_readme(out_dir, cfg)
    return parity


def _write_readme(out_dir: Path, cfg: LaMRConfig) -> None:
    readme = f"""# LaMR ONNX Artifact

Inference pipeline for Polymorph's Rust MCP runtime.

## Files
- `model.onnx` — backbone + MoE gate + 2 emission heads.
- `transitions.npz` — CRF transitions per head (`{{sem,dep}}_{{trans,start,end}}`).
- `config.yaml` — model architecture (must match the training run).
- `parity.json` — max-abs diff between torch and onnxruntime at export time.

## Inputs / outputs
- input_ids: (B, T) int64, cl100k_base token ids
- attention_mask: (B, T) bool
- emissions_sem / emissions_dep: (B, T, 2) float, log-emissions for tags {{0: keep, 1: drop}}

## Decode (Rust side)
1. Run ONNX session → 2 emission tensors.
2. For each head, run Viterbi with the corresponding transition / start / end vectors.
3. Combine: a token is dropped iff *both* heads agree on tag=1 (conservative; tune later).
4. Map back to the lock_mask: locked tokens are never dropped (see `src/lamr.rs`).

## Model config
- vocab_size: {cfg.vocab_size}
- d_model: {cfg.d_model}
- n_layers: {cfg.n_layers}
- experts: {cfg.n_experts} (top-{cfg.top_k})
"""
    (out_dir / "README.md").write_text(readme)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Export LaMR to ONNX + transitions.")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--parity-seq-len", type=int, default=64)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    parity = export(
        checkpoint=args.ckpt,
        out_dir=args.out,
        config_path=args.config,
        opset=args.opset,
        parity_seq_len=args.parity_seq_len,
    )
    print(f"exported to {args.out}; parity={parity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
