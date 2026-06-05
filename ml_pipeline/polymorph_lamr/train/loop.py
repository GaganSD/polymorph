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


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    # Note: MPS is deliberately excluded. The CRF's int64 gather/index ops
    # surface non-deterministic garbage on MPS today (observed: out-of-range
    # `prev_tag` values from `tags.to('mps')` post-async-copy). CPU is slow
    # but correct on Apple silicon; flip to MPS via env override when the
    # upstream bug is resolved.
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
                # NOTE: w_semantic/w_dependency and lambda_sem/lambda_dep are currently
                # RESERVED and do not affect the loss — joint_nll optimizes the single
                # blended CRF (see its docstring). Threaded through for forward-compat.
                out = model.joint_nll(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    tags=batch["tags"],
                    w_semantic=batch["w_semantic"],
                    w_dependency=batch["w_dependency"],
                    lambda_sem=lambda_sem,
                    lambda_dep=lambda_dep,
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
                        f"loss={out['loss'].item():.4f} "
                        f"nll_sem={out['nll_sem'].item():.4f} "
                        f"nll_dep={out['nll_dep'].item():.4f} "
                        f"elapsed={dt:.1f}s"
                    )
                if state.step % ckpt_every == 0:
                    save_checkpoint(model, out_dir / f"ckpt-{state.step:06d}.pt", state)
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
