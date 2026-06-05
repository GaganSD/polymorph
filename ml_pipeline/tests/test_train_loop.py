"""Training loop: one-step training on a tiny model, plus train.py CLI."""

import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from polymorph_lamr.model.lamr import LaMRConfig, LaMRModel
from polymorph_lamr.train.dataset import LabeledShardDataset, collate
from polymorph_lamr.train.loop import (
    _amp_dtype,
    _cosine_lr,
    _pick_device,
    save_checkpoint,
    train,
    TrainState,
)
from polymorph_lamr.train.loss import joint_loss
from polymorph_lamr.train.train import build_model, load_config, main


def _tiny_cfg_dict(tmp_path):
    return {
        "model": {
            "vocab_size": 64,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "ff_mult": 2,
            "dropout": 0.0,
        },
        "train": {
            "batch_size": 1,
            "grad_accum": 1,
            "max_seq_len": 8,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "max_steps": 1,
            "amp_dtype": "fp32",
            "ckpt_every": 1,
            "log_every": 1,
            "lambda_sem": 1.0,
            "lambda_dep": 1.0,
            "seed": 7,
        },
    }


def _make_shard(tmp_path: Path):
    sample = {
        "input_ids": [1, 2, 3, 4, 5],
        "tags": [0, 1, 0, 1, 0],
        "w_semantic": [1.0, 0.5, 0.5, 0.5, 1.0],
        "w_dependency": [0.0, 0.5, 0.5, 0.5, 0.0],
        "is_code": True,
        "src_path": "fixture",
    }
    p = tmp_path / "shard.jsonl"
    p.write_text(json.dumps(sample) + "\n")
    return p


def test_amp_dtype_dispatch():
    assert _amp_dtype("bf16") == torch.bfloat16
    assert _amp_dtype("fp32") == torch.float32


def test_cosine_lr_curve():
    assert _cosine_lr(0, warmup=10, max_steps=100, base_lr=1.0) > 0
    # Past max_steps, LR clamps to ~0.
    final = _cosine_lr(200, warmup=10, max_steps=100, base_lr=1.0)
    assert final < 1e-3


def test_pick_device_returns_device():
    dev = _pick_device()
    assert isinstance(dev, torch.device)


def test_joint_loss_wrapper_returns_blended_loss():
    # Post-C1: the wrapper returns the model's single trained objective (the
    # blended-CRF NLL in out["loss"]), NOT a lambda-weighted sum of the per-head
    # NLLs. The lambda knobs are vestigial and must not change the result.
    out = {"loss": torch.tensor(2.5), "nll_sem": torch.tensor(2.0), "nll_dep": torch.tensor(3.0)}
    assert torch.equal(joint_loss(out, lambda_sem=1.0, lambda_dep=1.0), out["loss"])
    assert torch.equal(joint_loss(out, lambda_sem=2.0, lambda_dep=0.5), out["loss"])


def test_train_one_step_writes_checkpoint(tmp_path, monkeypatch):
    # Force CPU to avoid MPS-specific dtype issues in CI / dev laptops.
    monkeypatch.setattr(
        "polymorph_lamr.train.loop._pick_device",
        lambda: torch.device("cpu"),
    )

    cfg_dict = _tiny_cfg_dict(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    shard = _make_shard(tmp_path)

    cfg = load_config(cfg_path)
    model = build_model(cfg)
    dataset = LabeledShardDataset([shard], max_seq_len=8)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    out_dir = tmp_path / "ckpts"
    state = train(
        model=model,
        loader=loader,
        out_dir=out_dir,
        max_steps=1,
        grad_accum=1,
        lr=1e-3,
        weight_decay=0.0,
        warmup_steps=0,
        amp_dtype="fp32",
        ckpt_every=1,
        log_every=1,
    )
    assert state.step >= 1
    assert (out_dir / "ckpt-final.pt").exists()


def test_save_checkpoint_serializes(tmp_path):
    cfg = LaMRConfig(vocab_size=32, d_model=8, n_layers=1, n_heads=2, ff_mult=1)
    model = LaMRModel(cfg)
    p = tmp_path / "ckpt.pt"
    save_checkpoint(model, p, TrainState(step=42))
    blob = torch.load(p, map_location="cpu", weights_only=False)
    assert blob["step"] == 42
    assert blob["cfg"]["vocab_size"] == 32


def test_train_cli_dry_run(tmp_path, capsys):
    cfg_dict = _tiny_cfg_dict(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    rc = main(["--config", str(cfg_path), "--dry-run"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "params=" in captured
    assert "device=" in captured


def test_train_cli_requires_shards_when_not_dry(tmp_path, capsys):
    cfg_dict = _tiny_cfg_dict(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg_dict))
    rc = main(["--config", str(cfg_path)])
    assert rc != 0
    assert "shards" in capsys.readouterr().err.lower()
