"""Evaluate a trained LaMR model on held-out val shards.

Answers the question the training loss can't: does the model actually keep the
right tokens and drop the rest? Runs the SAME blended-CRF Viterbi decode the
Rust runtime uses, compares the predicted keep/drop tags against the gold
(teacher) labels, and reports token accuracy, drop-class precision/recall/F1,
the achieved vs gold drop rate, and a clean per-token NLL.

Reused two ways: the ``lamr-eval`` CLI (this file) and the periodic val pass in
``train.loop`` (imported as ``evaluate``).
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..model.lamr import LaMRModel


@torch.no_grad()
def evaluate(model: LaMRModel, loader: DataLoader, device: torch.device, max_batches: int | None = None) -> dict:
    """Decode the blended CRF over ``loader`` and score against gold tags.

    Returns a metrics dict. Tag 1 = drop, tag 0 = keep; precision/recall/F1 are
    for the *drop* class (the action the pruner takes).
    """
    was_training = model.training
    model.eval()
    tp = fp = fn = tn = 0  # for the "drop" (tag==1) class
    total = correct = pred_drop = gold_drop = 0
    nll_sum = 0.0
    tok_sum = 0

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tags = batch["tags"].to(device)

        emi_sem, emi_dep, head_weights = model(ids, mask)
        params = model.weighted_crf_parameters(emi_sem, emi_dep, head_weights)
        preds = model.crf_semantic.decode_with_params(
            emissions=params["emissions"],
            mask=mask,
            transitions=params["transitions"],
            start_transitions=params["start_transitions"],
            end_transitions=params["end_transitions"],
        )
        # Per-token NLL of the blended CRF (the clean, length-invariant loss).
        nll_vec = model.crf_semantic.nll_with_params(
            params["emissions"], tags, mask,
            params["transitions"], params["start_transitions"], params["end_transitions"],
            reduction="none",
        )
        lens = mask.long().sum(dim=1)
        nll_sum += float(nll_vec.sum().item())
        tok_sum += int(lens.sum().item())

        for b in range(ids.shape[0]):
            n = int(lens[b].item())
            if n == 0:
                continue
            gold = tags[b, :n].tolist()
            pred = preds[b][:n]
            for g, p in zip(gold, pred):
                total += 1
                if p == g:
                    correct += 1
                if g == 1:
                    gold_drop += 1
                if p == 1:
                    pred_drop += 1
                if p == 1 and g == 1:
                    tp += 1
                elif p == 1 and g == 0:
                    fp += 1
                elif p == 0 and g == 1:
                    fn += 1
                else:
                    tn += 1

    if was_training:
        model.train()

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "tokens": total,
        "accuracy": correct / total if total else 0.0,
        "drop_precision": prec,
        "drop_recall": rec,
        "drop_f1": f1,
        "pred_drop_rate": pred_drop / total if total else 0.0,
        "gold_drop_rate": gold_drop / total if total else 0.0,
        "per_token_nll": nll_sum / tok_sum if tok_sum else 0.0,
    }


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
    args = p.parse_args(argv)

    max_seq_len = 1024
    if args.config and args.config.exists():
        max_seq_len = int(yaml.safe_load(args.config.read_text())["train"]["max_seq_len"])

    model, _cfg = _load_checkpoint(args.ckpt)
    device = _pick_device()
    model.to(device)
    loader = _build_loader([Path(s) for s in args.val_shards], max_seq_len, args.batch_size)

    m = evaluate(model, loader, device)
    print("LaMR eval on val:")
    print(f"  tokens          : {m['tokens']:,}")
    print(f"  accuracy        : {m['accuracy']:.4f}")
    print(f"  drop F1         : {m['drop_f1']:.4f}  (precision {m['drop_precision']:.4f} / recall {m['drop_recall']:.4f})")
    print(f"  drop rate       : pred {m['pred_drop_rate']:.4f}  vs gold {m['gold_drop_rate']:.4f}")
    print(f"  per-token NLL   : {m['per_token_nll']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
