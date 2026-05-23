"""End-to-end ONNX export + parity check on a tiny model."""

import json
from pathlib import Path

import pytest
import torch

from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel


def _tiny_cfg() -> LaMRConfig:
    return LaMRConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
        n_experts=2,
        top_k=2,
    )


def test_export_and_parity(tmp_path: Path):
    ort = pytest.importorskip("onnxruntime")
    cfg = _tiny_cfg()
    torch.manual_seed(0)
    model = LaMRModel(cfg)
    model.eval()

    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "step": 0, "cfg": cfg.__dict__}, ckpt)

    from polymorph_lamr.export.to_onnx import export

    out_dir = tmp_path / "art"
    parity = export(checkpoint=ckpt, out_dir=out_dir, parity_seq_len=16)
    assert parity["max_abs_diff_sem"] < 1e-3
    assert parity["max_abs_diff_dep"] < 1e-3

    # Side-car & docs exist.
    assert (out_dir / "model.onnx").exists()
    assert (out_dir / "transitions.npz").exists()
    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "README.md").exists()
    assert json.loads((out_dir / "parity.json").read_text())["max_abs_diff_sem"] < 1e-3
