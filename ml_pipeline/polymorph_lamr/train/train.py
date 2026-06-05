"""CLI entrypoint: read config + shard list, build model, run training."""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from ..model.lamr import LaMRConfig, LaMRModel
from .dataset import LabeledShardDataset, collate
from .loop import _pick_device, train


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def build_model(cfg: dict) -> LaMRModel:
    mcfg = cfg["model"]
    lcfg = LaMRConfig(
        vocab_size=int(mcfg["vocab_size"]),
        d_model=int(mcfg["d_model"]),
        n_layers=int(mcfg["n_layers"]),
        n_heads=int(mcfg["n_heads"]),
        ff_mult=int(mcfg["ff_mult"]),
        dropout=float(mcfg["dropout"]),
        backbone=str(mcfg.get("backbone", "transformer")),
    )
    return LaMRModel(lcfg)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train LaMR.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--shards", nargs="+", required=False, help="JSONL labeled shards")
    p.add_argument("--out", type=Path, default=Path("artifacts/checkpoints"))
    p.add_argument("--dry-run", action="store_true", help="report device + param count + memory and exit")
    p.add_argument("--max-steps", type=int, default=None, help="override config max_steps")
    return p


def _dry_run(model: LaMRModel, cfg: dict) -> int:
    device = _pick_device()
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    b = int(cfg["train"]["batch_size"])
    t = int(cfg["train"]["max_seq_len"])
    print(f"device={device}")
    print(f"params={n_params:,} trainable={n_train:,}")
    print(f"batch_size={b} max_seq_len={t}")
    # Rough peak-memory estimate: 4 bytes/param for fp32 weights + 4 bytes/param for AdamW state pair (×2) + activations.
    bytes_per_param = 4 + 4 + 4  # weight + m + v
    static = n_params * bytes_per_param
    act_bytes = b * t * cfg["model"]["d_model"] * cfg["model"]["n_layers"] * 4 * 4  # very rough
    print(f"estimated_static_mem={static / 1e9:.2f}GB")
    print(f"estimated_activation_mem~{act_bytes / 1e9:.2f}GB (rough)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    torch.manual_seed(int(cfg["train"]["seed"]))
    model = build_model(cfg)

    if args.dry_run:
        return _dry_run(model, cfg)

    if not args.shards:
        print("--shards is required unless --dry-run is set", file=sys.stderr)
        return 2

    dataset = LabeledShardDataset(
        shard_paths=[Path(p) for p in args.shards],
        max_seq_len=int(cfg["train"]["max_seq_len"]),
        seed=int(cfg["train"]["seed"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        collate_fn=partial(collate, pad_id=0),
        num_workers=0,  # IterableDataset; bump for production
    )

    max_steps = args.max_steps if args.max_steps is not None else int(cfg["train"]["max_steps"])
    train(
        model=model,
        loader=loader,
        out_dir=args.out,
        max_steps=max_steps,
        grad_accum=int(cfg["train"]["grad_accum"]),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        warmup_steps=int(cfg["train"]["warmup_steps"]),
        amp_dtype=str(cfg["train"]["amp_dtype"]),
        ckpt_every=int(cfg["train"]["ckpt_every"]),
        log_every=int(cfg["train"]["log_every"]),
        # Reserved/inert under the blended-CRF objective (see LaMRModel.joint_nll);
        # .get() so pruning these config keys never breaks training.
        lambda_sem=float(cfg["train"].get("lambda_sem", 1.0)),
        lambda_dep=float(cfg["train"].get("lambda_dep", 1.0)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
