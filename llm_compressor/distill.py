"""Distillation with modular losses (logit KL, next-token CE, hidden MSE).

Distillation is only ever run on real calibration examples (DATASET MODE).
Losses are modular: a loss whose inputs do not exist (e.g. CE for an
encoder-only model, or KL when the student has no LM head) is reported as
``skipped`` with a reason instead of silently contributing zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .utils.batch import causal_labels, forward_tensor


class DistillError(RuntimeError):
    pass


@dataclass
class DistillConfig:
    steps: int = 100
    lr: float = 5e-5
    batch_size: int = 2
    temperature: float = 2.0
    alpha_kl: float = 1.0
    alpha_ce: float = 1.0
    alpha_hidden: float = 1.0
    loss_mode: str = "auto"  # auto | kl | ce | hidden | all
    seed: int = 0
    log_every: int = 25

    def to_dict(self) -> dict:
        return asdict(self)


_LOSS_ALIASES = {
    "kl": "kl", "logit": "kl", "logits": "kl", "logit_kl": "kl",
    "soft": "kl", "softmax": "kl",
    "ce": "ce", "next_token": "ce", "next-token": "ce", "lm": "ce",
    "cross_entropy": "ce", "cross-entropy": "ce", "hard": "ce",
    "hidden": "hidden", "hidden_mse": "hidden", "hidden-state": "hidden",
    "hidden_state_mse": "hidden", "mse": "hidden", "feature": "hidden",
}


def _losses(requested: str) -> list[str]:
    requested = (requested or "auto").lower().strip()
    if requested in ("auto", "all", ""):
        return ["kl", "ce", "hidden"]
    out = []
    for part in requested.split("+"):
        name = _LOSS_ALIASES.get(part.strip())
        if name is None:
            raise DistillError(
                f"unknown distillation loss '{part.strip()}'; expected one of "
                "kl | logit_kl | ce | next_token | hidden_mse (or 'all')"
            )
        if name not in out:
            out.append(name)
    if not out:
        raise DistillError("no distillation losses requested")
    return out


@torch.no_grad()
def _teacher_targets(teacher, batch, need_hidden: bool):
    teacher.eval()
    out = teacher(**batch, use_cache=False,
                  output_hidden_states=need_hidden)
    logits = getattr(out, "logits", None)
    hidden = None
    if need_hidden and getattr(out, "hidden_states", None):
        hidden = out.hidden_states[-1]
    return logits, hidden


def distill(student, teacher, batches: Sequence[dict],
            config: Optional[DistillConfig] = None,
            protected_prefixes: Sequence[str] = (),
            progress=None) -> dict:
    """Fine-tune ``student`` against frozen ``teacher`` on real examples."""
    cfg = config or DistillConfig()
    if not batches:
        raise DistillError("distillation requires real calibration batches "
                           "(DATASET-FREE mode cannot distil)")
    torch.manual_seed(cfg.seed)
    wanted = _losses(cfg.loss_mode)

    trainable = []
    frozen = []
    for name, param in student.named_parameters():
        if any(name == p or name.startswith(p + ".") for p in protected_prefixes):
            param.requires_grad_(False)
            frozen.append(name)
            continue
        if not param.is_floating_point():
            param.requires_grad_(False)
            frozen.append(name)
            continue
        param.requires_grad_(True)
        trainable.append(name)
    if not trainable:
        raise DistillError("every student parameter is protected or non-float; "
                           "nothing to distil")

    # keep the student in float32 for stable training
    student.float()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    student.train()
    history: list[dict] = []
    applied: dict[str, str] = {}
    step = 0
    while step < cfg.steps:
        for batch in batches:
            if step >= cfg.steps:
                break
            need_hidden = "hidden" in wanted
            t_logits, t_hidden = _teacher_targets(teacher, batch, need_hidden)
            out = student(**batch, use_cache=False,
                          output_hidden_states=need_hidden)
            s_logits = getattr(out, "logits", None)
            s_hidden = (out.hidden_states[-1]
                        if getattr(out, "hidden_states", None) else None)
            total = torch.zeros((), dtype=torch.float32)
            parts: dict[str, float] = {}

            if "kl" in wanted:
                if s_logits is None or t_logits is None:
                    applied["kl"] = "skipped: model has no LM logits"
                else:
                    T = cfg.temperature
                    sl = s_logits[:, :t_logits.shape[1]].float()
                    tl = t_logits.float()
                    kl = F.kl_div(F.log_softmax(sl / T, dim=-1),
                                  F.softmax(tl / T, dim=-1),
                                  reduction="batchmean") * (T * T)
                    total = total + cfg.alpha_kl * kl
                    parts["kl"] = float(kl)
                    applied["kl"] = "applied"
            if "ce" in wanted:
                if s_logits is None:
                    applied["ce"] = "skipped: model has no LM logits"
                else:
                    ids = batch.get("decoder_input_ids", batch["input_ids"])
                    logits = s_logits[:, :-1].float()
                    labels = causal_labels(ids)[:, 1:]
                    # teacher/student sequence lengths can differ by one token;
                    # score the common prefix rather than crashing on a shape error
                    steps = min(logits.shape[1], labels.shape[1])
                    n = min(logits.shape[0], labels.shape[0])
                    ce = F.cross_entropy(
                        logits[:n, :steps].reshape(-1, logits.shape[-1]),
                        labels[:n, :steps].reshape(-1), ignore_index=-100)
                    total = total + cfg.alpha_ce * ce
                    parts["ce"] = float(ce)
                    applied["ce"] = "applied"
            if "hidden" in wanted:
                if s_hidden is None or t_hidden is None:
                    applied["hidden"] = "skipped: model has no hidden states"
                elif s_hidden.shape != t_hidden.shape:
                    applied["hidden"] = (f"skipped: hidden shape mismatch "
                                         f"{tuple(s_hidden.shape)} vs "
                                         f"{tuple(t_hidden.shape)}")
                else:
                    mse = F.mse_loss(s_hidden.float(), t_hidden.float())
                    total = total + cfg.alpha_hidden * mse
                    parts["hidden"] = float(mse)
                    applied["hidden"] = "applied"
            if not parts:
                raise DistillError(
                    "none of the requested distillation losses could be "
                    f"applied: {applied}")
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            if cfg.log_every and (step % cfg.log_every == 0 or step == 1):
                history.append({"step": step, "total": float(total.detach()),
                                **parts})
                if progress is not None:
                    progress(step, cfg.steps, float(total.detach()))
    student.eval()
    return {"steps": step, "history": history, "losses": applied,
            "trainable_parameters": len(trainable),
            "frozen_parameters": len(frozen), "config": cfg.to_dict()}
