"""Evaluate a trained LaMR pruner on held-out val shards.

The model is a per-token binary classifier (``sigmoid(logit) = P(drop)``); there
is no CRF and no Viterbi. Decode is a threshold on the drop probability,
**calibrated to a target compression rate** R — sort the tokens by drop-prob and
cut at the threshold that drops an R fraction. That makes the compression ratio a
runtime knob and sidesteps the global-bias oscillation that made argmax (0.5)
decode look like it "collapsed."

So the metrics here are threshold-free ranking quality plus calibrated-decode
quality, NOT argmax-F1:

  * ``pr_auc`` / ``roc_auc`` — does the model RANK the right tokens to drop?
    (the primary selection metric; threshold-independent)
  * ``f1_at_target`` (+ prec/rec/accuracy) — quality of the decode at the
    threshold calibrated to drop exactly the target rate.
  * ``best_f1`` — the calibration ceiling (best F1 over all cutoffs).
  * ``argmax_f1`` — F1 at the fixed 0.5 threshold, for reference only.

Reused two ways: the ``lamr-eval`` CLI (this file) and the periodic val pass in
``train.loop`` (imported as ``evaluate``; it selects ckpt-best by ``pr_auc``).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..model.lamr import LaMRModel

# Default target drop rate used to calibrate the decode threshold when none is
# given. Matches the Rust runtime's DEFAULT_DROP_RATE and the v0 corpus mean.
DEFAULT_TARGET_RATE = 0.30


@torch.no_grad()
def collect_drop_probs(
    model: LaMRModel, loader: DataLoader, device: torch.device, max_batches: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (drop_prob, gold) flat arrays over all valid (non-pad) tokens."""
    was_training = model.training
    model.eval()
    probs: list[np.ndarray] = []
    golds: list[np.ndarray] = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tags = batch["tags"].to(device)
        p_drop = torch.sigmoid(model(ids, mask).float())  # (B, T)
        m = mask.bool()
        probs.append(p_drop[m].detach().cpu().numpy())
        golds.append(tags[m].detach().cpu().numpy())
    if was_training:
        model.train()
    if not probs:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.int64)
    return np.concatenate(probs), np.concatenate(golds)


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def ranking_metrics(drop_prob: np.ndarray, gold: np.ndarray, target_rate: float | None = None) -> dict:
    """Threshold-free ranking quality + calibrated-decode quality at ``target_rate``.

    ``target_rate`` is the fraction of tokens to drop; the decode threshold is the
    drop-prob at the top-``round(target_rate * N)`` cut. ``None`` => the gold drop
    rate (so the calibrated decode matches the teacher's compression).
    """
    drop_prob = np.asarray(drop_prob, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.int64)
    n = len(gold)
    if n == 0:
        return {
            "tokens": 0, "gold_rate": 0.0, "target_rate": target_rate or 0.0,
            "pr_auc": 0.0, "roc_auc": 0.0, "best_f1": 0.0, "best_f1_thr": 0.0,
            "f1_at_target": 0.0, "prec_at_target": 0.0, "rec_at_target": 0.0,
            "accuracy_at_target": 0.0, "thr_at_target": 0.0, "pred_drop_rate": 0.0,
            "argmax_f1": 0.0, "argmax_prec": 0.0, "argmax_rec": 0.0,
            "argmax_drop_rate": 0.0, "per_token_bce": 0.0,
        }
    pos = int(gold.sum())                 # gold drops
    gold_rate = pos / n
    target_rate = gold_rate if target_rate is None else float(target_rate)

    order = np.argsort(-drop_prob)        # high drop-prob first
    g = gold[order].astype(np.int64)
    cum_tp = np.cumsum(g)                 # tp if we drop the top-k
    k_idx = np.arange(1, n + 1)
    precision = cum_tp / k_idx

    # Average precision (area under precision-recall for the drop class): the
    # threshold-free ranking quality. Sum P_k where a true drop is newly retrieved.
    pr_auc = float((precision * g).sum() / pos) if pos else 0.0

    # ROC-AUC via Mann-Whitney U on the ranks.
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(drop_prob)] = np.arange(1, n + 1)  # ascending ranks
    neg = n - pos
    roc_auc = float((ranks[gold == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else 0.0

    # Best F1 across all cutoffs (calibration ceiling).
    f1_curve = np.divide(2 * cum_tp, k_idx + pos, out=np.zeros(n, dtype=np.float64), where=(k_idx + pos) > 0)
    best_i = int(np.argmax(f1_curve))
    best_f1 = float(f1_curve[best_i])
    best_thr = float(drop_prob[order][best_i])

    # Calibrated decode: threshold so pred drop-rate == target_rate.
    k = max(1, min(n, int(round(target_rate * n))))
    tp = int(cum_tp[k - 1]); fp = k - tp; fn = pos - tp
    prec_t, rec_t, f1_t = _prf(tp, fp, fn)
    thr_at_target = float(drop_prob[order][k - 1])
    acc_at_target = (tp + (neg - fp)) / n  # correct keeps + correct drops

    # argmax (threshold 0.5) for reference.
    pred05 = drop_prob >= 0.5
    tp5 = int(((pred05) & (gold == 1)).sum()); fp5 = int(((pred05) & (gold == 0)).sum())
    fn5 = int(((~pred05) & (gold == 1)).sum())
    prec5, rec5, f15 = _prf(tp5, fp5, fn5)

    # Unweighted per-token BCE (a clean, comparable scalar loss).
    p = np.clip(drop_prob, 1e-7, 1 - 1e-7)
    bce = float(-(gold * np.log(p) + (1 - gold) * np.log(1 - p)).mean())

    return {
        "tokens": n, "gold_rate": gold_rate, "target_rate": target_rate,
        "pr_auc": pr_auc, "roc_auc": roc_auc,
        "best_f1": best_f1, "best_f1_thr": best_thr,
        "f1_at_target": f1_t, "prec_at_target": prec_t, "rec_at_target": rec_t,
        "accuracy_at_target": acc_at_target, "thr_at_target": thr_at_target,
        "pred_drop_rate": k / n,
        "argmax_f1": f15, "argmax_prec": prec5, "argmax_rec": rec5,
        "argmax_drop_rate": float(pred05.mean()),
        "per_token_bce": bce,
    }


@torch.no_grad()
def evaluate(
    model: LaMRModel,
    loader: DataLoader,
    device: torch.device,
    target_rate: float | None = DEFAULT_TARGET_RATE,
    max_batches: int | None = None,
) -> dict:
    """Score the pruner's drop-ranking over ``loader`` against gold tags.

    Returns the ranking + calibrated-decode metrics dict (see ``ranking_metrics``).
    """
    drop_prob, gold = collect_drop_probs(model, loader, device, max_batches=max_batches)
    return ranking_metrics(drop_prob, gold, target_rate)


def _build_loader(val_shards: list[Path], max_seq_len: int, batch_size: int) -> DataLoader:
    from ..train.dataset import LabeledShardDataset, collate

    ds = LabeledShardDataset(val_shards, max_seq_len=max_seq_len, shuffle_files=False)
    return DataLoader(ds, batch_size=batch_size, collate_fn=partial(collate, pad_id=0), num_workers=0)


def main(argv: list[str] | None = None) -> int:
    import yaml

    from ..export.to_onnx import _load_checkpoint
    from ..train.loop import _pick_device

    p = argparse.ArgumentParser(description="Evaluate a trained LaMR checkpoint on val shards.")
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--val-shards", nargs="+", required=True)
    p.add_argument("--config", type=Path, default=None, help="for max_seq_len (else 1024)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--target-rate", type=float, default=None,
                   help="target drop rate to calibrate the decode threshold (default = config or gold rate)")
    args = p.parse_args(argv)

    max_seq_len = 1024
    target_rate = args.target_rate
    if args.config and args.config.exists():
        cfg = yaml.safe_load(args.config.read_text())
        max_seq_len = int(cfg["train"]["max_seq_len"])
        if target_rate is None:
            target_rate = cfg.get("eval", {}).get("target_rate", DEFAULT_TARGET_RATE)
    if target_rate is None:
        target_rate = DEFAULT_TARGET_RATE

    model, _cfg = _load_checkpoint(args.ckpt)
    device = _pick_device()
    model.to(device)
    loader = _build_loader([Path(s) for s in args.val_shards], max_seq_len, args.batch_size)

    m = evaluate(model, loader, device, target_rate=target_rate)
    print("LaMR eval on val:")
    print(f"  tokens          : {m['tokens']:,}  gold_drop_rate={m['gold_rate']:.3f}")
    print(f"  RANKING         : PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  best-F1={m['best_f1']:.4f} @thr={m['best_f1_thr']:.3f}")
    print(f"  @target {m['target_rate']:.3f}  : F1={m['f1_at_target']:.4f}  (P{m['prec_at_target']:.3f}/R{m['rec_at_target']:.3f})  acc={m['accuracy_at_target']:.4f}  thr={m['thr_at_target']:.3f}")
    print(f"  argmax(0.5)     : F1={m['argmax_f1']:.4f}  (P{m['argmax_prec']:.3f}/R{m['argmax_rec']:.3f})  drop_rate={m['argmax_drop_rate']:.3f}")
    print(f"  per-token BCE   : {m['per_token_bce']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
