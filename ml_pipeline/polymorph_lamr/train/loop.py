"""Training loop with AMP, grad accumulation, and lightweight checkpointing."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..model.lamr import LaMRModel


@dataclass
class TrainState:
    step: int = 0
    best_loss: float = math.inf
    best_val_pr_auc: float = -1.0


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # CPU fallback on Apple silicon. MPS was historically excluded for the CRF's
    # int64 gather ops; the CRF is gone now, but CPU stays the safe default for
    # the small from-scratch model (flip to MPS via env override if desired).
    return torch.device("cpu")


def _amp_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _cosine_lr(step: int, warmup: int, max_steps: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(
    model: LaMRModel,
    loader: DataLoader,
    out_dir: Path,
    max_steps: int = 20_000,
    grad_accum: int = 8,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_steps: int = 200,
    amp_dtype: str = "bf16",
    ckpt_every: int = 1000,
    log_every: int = 50,
    lambda_sem: float = 1.0,
    lambda_dep: float = 1.0,
    drop_class_weight: float = 1.0,
    target_rate: float = 0.30,
    val_loader: DataLoader | None = None,
    eval_every: int = 0,
) -> TrainState:
    device = _pick_device()
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = TrainState()
    dtype = _amp_dtype(amp_dtype)
    autocast_enabled = device.type in ("cuda", "cpu") and dtype != torch.float32

    optim.zero_grad(set_to_none=True)
    t0 = time.time()
    micro_step = 0
    # The dataset yields each example once per pass, so re-iterate the loader
    # across epochs until max_steps — otherwise training would stop after ~1 epoch
    # and never reach max_steps. A pass that yields no batches breaks the loop
    # (guards an empty dataset from spinning forever).
    while state.step < max_steps:
        batches_this_epoch = 0
        for batch in loader:
            batches_this_epoch += 1
            for k, v in batch.items():
                batch[k] = v.to(device)

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=autocast_enabled):
                # Class-weighted per-token BCE on the drop logits. drop_class_weight
                # up-weights the drop (positive) class so the ranking separates;
                # absolute calibration is a decode-time threshold, not a loss knob
                # (see LaMRModel.loss). w_semantic/w_dependency + lambda_* stay
                # RESERVED (inert) — threaded for forward-compat.
                out = model.loss(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    tags=batch["tags"],
                    drop_class_weight=drop_class_weight,
                )
            loss = out["loss"] / grad_accum
            loss.backward()
            micro_step += 1

            if micro_step % grad_accum == 0:
                # cosine LR.
                cur_lr = _cosine_lr(state.step, warmup_steps, max_steps, lr)
                for pg in optim.param_groups:
                    pg["lr"] = cur_lr
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
                state.step += 1

                if state.step % log_every == 0:
                    dt = time.time() - t0
                    print(
                        f"step={state.step:6d} lr={cur_lr:.2e} "
                        f"loss={out['loss'].item():.4f} bce={out['bce'].item():.4f} elapsed={dt:.1f}s"
                    )
                if state.step % ckpt_every == 0:
                    save_checkpoint(model, out_dir / f"ckpt-{state.step:06d}.pt", state)
                if eval_every and val_loader is not None and state.step % eval_every == 0:
                    from ..eval.evaluate import evaluate

                    m = evaluate(model, val_loader, device, target_rate=target_rate)
                    print(
                        f"  [val] step={state.step} PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f} "
                        f"F1@{m['target_rate']:.2f}={m['f1_at_target']:.4f} "
                        f"(P{m['prec_at_target']:.3f}/R{m['rec_at_target']:.3f}) thr={m['thr_at_target']:.3f} "
                        f"gold_rate={m['gold_rate']:.3f} bce/tok={m['per_token_bce']:.4f}"
                    )
                    # Select ckpt-best by PR-AUC: ranking quality is the stable,
                    # threshold-free signal (argmax-F1 oscillates with the global
                    # bias and is not what the calibrated decode optimizes).
                    if m["pr_auc"] > state.best_val_pr_auc:
                        state.best_val_pr_auc = m["pr_auc"]
                        save_checkpoint(model, out_dir / "ckpt-best.pt", state)
                        print(f"  [val] new best PR-AUC={m['pr_auc']:.4f} -> ckpt-best.pt")
                    model.train()
                if state.step >= max_steps:
                    break
        if batches_this_epoch == 0:
            print("[train] dataloader produced no batches; stopping")
            break
    save_checkpoint(model, out_dir / "ckpt-final.pt", state)
    return state


def save_checkpoint(model: LaMRModel, path: Path, state: TrainState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "step": state.step,
            "cfg": model.cfg.__dict__,
        },
        path,
    )
