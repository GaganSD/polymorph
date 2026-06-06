"""Threshold-independent probe of a LaMR checkpoint's drop-ranking quality.

The training/val loop scores argmax (or CRF-Viterbi) decode, which is hostage to
the model's *global* keep/drop bias — and that bias oscillates across steps even
with the CRF removed. This probe separates the two questions:

  1. Can the model RANK which tokens to drop?  -> PR-AUC / ROC-AUC / best-F1
     (threshold-independent: depends only on the ordering of drop-probabilities).
  2. Is its absolute decision bias calibrated? -> argmax-F1 vs F1 at a threshold
     calibrated so pred drop-rate == gold drop-rate ("survival-at-target-rate").

If ranking is good (high PR-AUC, high best-F1) but argmax-F1 is low, the model is
usable today and the fix is purely decode-side (calibrate a threshold to a target
compression rate) — exactly the consultant's prescription.

Usage:
  python -m scripts.probe_threshold --ckpt <ckpt.pt> --val-shards <val.jsonl> --config configs/default.yaml
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def collect_drop_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (drop_prob, gold) flat arrays over all valid (non-pad) tokens."""
    model.eval()
    probs: list[np.ndarray] = []
    golds: list[np.ndarray] = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tags = batch["tags"].to(device)
        emissions = model(ids, mask)                       # (B, T, 2)
        p_drop = torch.softmax(emissions.float(), dim=-1)[..., 1]  # (B, T)
        m = mask.bool()
        probs.append(p_drop[m].cpu().numpy())
        golds.append(tags[m].cpu().numpy())
    return np.concatenate(probs), np.concatenate(golds)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def ranking_report(drop_prob: np.ndarray, gold: np.ndarray, target_rate: float | None = None) -> dict:
    n = len(gold)
    pos = int(gold.sum())                 # gold drops
    gold_rate = pos / n if n else 0.0
    target_rate = gold_rate if target_rate is None else target_rate

    order = np.argsort(-drop_prob)        # high drop-prob first
    g = gold[order].astype(np.int64)
    cum_tp = np.cumsum(g)                 # tp if we drop the top-k
    k_idx = np.arange(1, n + 1)
    precision = cum_tp / k_idx
    recall = cum_tp / pos if pos else np.zeros(n)

    # Average precision (area under precision-recall, the threshold-free ranking
    # quality for the drop class): sum P_k over the positions where a true drop is
    # newly retrieved, divided by total positives.
    ap = float((precision * g).sum() / pos) if pos else 0.0

    # ROC-AUC via Mann-Whitney U on the ranks (ties broken by argsort order; fine
    # for a diagnostic). rank of positives among all scores.
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(drop_prob)] = np.arange(1, n + 1)  # ascending ranks
    neg = n - pos
    auc = float((ranks[gold == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else 0.0

    # Best F1 across all cutoffs (the calibration ceiling).
    f1_curve = np.divide(2 * cum_tp, k_idx + pos, out=np.zeros(n), where=(k_idx + pos) > 0)
    best_i = int(np.argmax(f1_curve))
    best_f1 = float(f1_curve[best_i])
    best_thr = float(drop_prob[order][best_i])

    # F1 at a threshold calibrated so pred drop-rate == target_rate.
    k = max(1, min(n, int(round(target_rate * n))))
    tp = int(cum_tp[k - 1]); fp = k - tp; fn = pos - tp
    prec_t, rec_t, f1_t = _prf(tp, fp, fn)
    thr_at_target = float(drop_prob[order][k - 1])

    # argmax (threshold 0.5) for reference.
    pred05 = drop_prob >= 0.5
    tp5 = int(((pred05) & (gold == 1)).sum()); fp5 = int(((pred05) & (gold == 0)).sum())
    fn5 = int(((~pred05) & (gold == 1)).sum())
    prec5, rec5, f15 = _prf(tp5, fp5, fn5)

    return {
        "tokens": n, "gold_rate": gold_rate, "target_rate": target_rate,
        "pr_auc": ap, "roc_auc": auc,
        "best_f1": best_f1, "best_f1_thr": best_thr,
        "f1_at_target": f1_t, "prec_at_target": prec_t, "rec_at_target": rec_t,
        "thr_at_target": thr_at_target,
        "argmax_f1": f15, "argmax_prec": prec5, "argmax_rec": rec5,
        "argmax_drop_rate": float(pred05.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    import yaml

    from polymorph_lamr.eval.evaluate import _build_loader
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
    r = ranking_report(drop_prob, gold, args.target_rate)

    print(f"probe: {args.ckpt}")
    print(f"  tokens={r['tokens']:,}  gold_drop_rate={r['gold_rate']:.3f}")
    print(f"  RANKING (threshold-free):  PR-AUC={r['pr_auc']:.4f}  ROC-AUC={r['roc_auc']:.4f}  best-F1={r['best_f1']:.4f} @thr={r['best_f1_thr']:.3f}")
    print(f"  @target-rate {r['target_rate']:.3f}:  F1={r['f1_at_target']:.4f}  (P{r['prec_at_target']:.3f}/R{r['rec_at_target']:.3f})  thr={r['thr_at_target']:.3f}")
    print(f"  argmax(0.5):  F1={r['argmax_f1']:.4f}  (P{r['argmax_prec']:.3f}/R{r['argmax_rec']:.3f})  drop_rate={r['argmax_drop_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
