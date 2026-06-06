"""Threshold-independent probe of a LaMR checkpoint's drop-ranking quality.

This is now a thin CLI over ``polymorph_lamr.eval.evaluate`` (the ranking math
lives there, used by both the val loop and ``lamr-eval``). It separates two
questions about a checkpoint:

  1. Can the model RANK which tokens to drop?  -> PR-AUC / ROC-AUC / best-F1
     (threshold-independent: depends only on the ordering of drop-probabilities).
  2. Is its absolute decision bias calibrated? -> argmax-F1 (fixed 0.5) vs F1 at
     the threshold calibrated to a target drop-rate ("survival-at-target-rate").

If ranking is good (high PR-AUC) but argmax-F1 is low, the model is usable today
and the fix is purely decode-side (calibrate a threshold to a target compression
rate) — which is exactly what the runtime does.

Usage:
  python -m scripts.probe_threshold --ckpt <ckpt.pt> --val-shards <val.jsonl> --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    import yaml

    from polymorph_lamr.eval.evaluate import _build_loader, collect_drop_probs, ranking_metrics
    from polymorph_lamr.export.to_onnx import _load_checkpoint
    from polymorph_lamr.train.loop import _pick_device

    p = argparse.ArgumentParser(description="Probe drop-ranking quality (threshold-free).")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--val-shards", nargs="+", required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--target-rate", type=float, default=None, help="default = gold drop rate")
    args = p.parse_args(argv)

    max_seq_len = 1024
    if args.config and args.config.exists():
        max_seq_len = int(yaml.safe_load(args.config.read_text())["train"]["max_seq_len"])

    model, _ = _load_checkpoint(args.ckpt)
    device = _pick_device()
    model.to(device)
    loader = _build_loader([Path(s) for s in args.val_shards], max_seq_len, args.batch_size)
    drop_prob, gold = collect_drop_probs(model, loader, device)
    r = ranking_metrics(drop_prob, gold, args.target_rate)

    print(f"probe: {args.ckpt}")
    print(f"  tokens={r['tokens']:,}  gold_drop_rate={r['gold_rate']:.3f}")
    print(f"  RANKING (threshold-free):  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  best-F1={r['best_f1']:.4f} @thr={r['best_f1_thr']:.3f}")
    print(f"  @target-rate {r['target_rate']:.3f}:  F1={r['f1_at_target']:.4f}  (P{r['prec_at_target']:.3f}/R{r['rec_at_target']:.3f})  thr={r['thr_at_target']:.3f}")
    print(f"  argmax(0.5):  F1={r['argmax_f1']:.4f}  (P{r['argmax_prec']:.3f}/R{r['argmax_rec']:.3f})  drop_rate={r['argmax_drop_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
