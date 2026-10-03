"""Neuron scoring.

Every method declares whether it *requires* representative data. Dataset-free
methods use only weights; activation-aware methods genuinely need calibration
activations (never random tensors).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn

from ..model.architecture import MLPInfo

__all__ = ["ScoringMethod", "REGISTRY", "register", "resolve_method",
           "neuron_weight_scores", "activation_scores", "normalize_scores"]


@dataclass
class ScoringMethod:
    name: str
    requires_data: bool
    fn: Callable[..., torch.Tensor]
    description: str
    requires_grad: bool = False


REGISTRY: dict[str, ScoringMethod] = {}


def register(method: ScoringMethod) -> None:
    REGISTRY[method.name] = method


def resolve_method(name: str) -> ScoringMethod:
    if name not in REGISTRY:
        raise KeyError(
            f"unknown scoring method '{name}'; available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


# --------------------------------------------------------------------------- #
# data-free: weight magnitude
# --------------------------------------------------------------------------- #
def _rows(p, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    w = p.weight_matrix.detach().float()
    return w if keep is None else w.index_select(0, keep)


def neuron_weight_scores(mlp: MLPInfo, mode: str = "combined") -> torch.Tensor:
    """Per-neuron importance from weights alone.

    ``l1``/``l2`` use incoming (gate/up) rows only; ``combined`` also weights by
    the outgoing (down) columns, which is a better proxy for output impact.
    """
    inter = mlp.intermediate_size
    if inter is None or mlp.down_proj is None:
        raise ValueError(f"{mlp.name}: MLP is not resolved; cannot score")

    if mlp.gate_up_fused:
        w = mlp.up_proj.weight_matrix.detach().float()
        gate_in = w[:inter]
        up_in = w[inter:]
    else:
        gate_in = _rows(mlp.gate_proj) if mlp.gate_proj is not None else None
        up_in = _rows(mlp.up_proj) if mlp.up_proj is not None else None

    incoming = [t for t in (gate_in, up_in) if t is not None]
    in_score = torch.stack(
        [t.abs().sum(dim=1) if mode == "l1" else t.pow(2).sum(dim=1)
         for t in incoming]
    ).mean(dim=0)

    if mode == "combined":
        down = mlp.down_proj.weight_matrix.detach().float().index_select(1, _all(inter))
        out_score = down.pow(2).sum(dim=0)
        return (in_score * out_score).sqrt()
    return in_score


def _all(n: int) -> torch.Tensor:
    return torch.arange(n, dtype=torch.long)


register(ScoringMethod("weight_l1", False,
                       lambda mlp: neuron_weight_scores(mlp, "l1"),
                       "sum |w| of incoming gate/up rows"))
register(ScoringMethod("weight_l2", False,
                       lambda mlp: neuron_weight_scores(mlp, "l2"),
                       "sum w^2 of incoming gate/up rows"))
register(ScoringMethod("weight_combined", False,
                       lambda mlp: neuron_weight_scores(mlp, "combined"),
                       "incoming magnitude x outgoing column norm (data-free)"))


# --------------------------------------------------------------------------- #
# dataset-based: activation magnitude
# --------------------------------------------------------------------------- #
class ActivationCollector:
    """Records per-channel mean |activation| at every FFN intermediate input."""

    def __init__(self, modules: dict[str, nn.Module]):
        self.modules = modules
        self._handles = []
        self._sum: dict[str, torch.Tensor] = {}
        self._count = 0
        self.grad_sum: dict[str, torch.Tensor] = {}

    def __enter__(self) -> "ActivationCollector":
        for key, mod in self.modules.items():
            self._handles.append(mod.register_forward_hook(self._make_hook(key)))
        return self

    def _make_hook(self, key: str):
        def hook(module, inputs, output):
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1]).abs().mean(dim=0)
            self._sum[key] = self._sum.get(key, torch.zeros_like(x)) + x
        return hook

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def bump(self, n: int = 1) -> None:
        self._count += n

    def activation(self, key: str) -> torch.Tensor:
        if key not in self._sum or self._count == 0:
            raise RuntimeError(f"no activations captured for {key}")
        return self._sum[key] / self._count


def activation_scores(mlp: MLPInfo, act: torch.Tensor,
                      weighting: str = "combined") -> torch.Tensor:
    """Scale captured activation importance by outgoing column magnitude."""
    if weighting == "activation":
        return act
    if mlp.down_proj is None:
        raise ValueError(f"{mlp.name}: missing down projection")
    inter = act.numel()
    down = mlp.down_proj.weight_matrix.detach().float().index_select(1, _all(inter))
    return act * down.pow(2).sum(dim=0).sqrt()


register(ScoringMethod("activation", True,
                       lambda mlp, act=None, **k: activation_scores(mlp, _require(act), "activation"),
                       "mean |activation| at FFN intermediate (needs data)"))
register(ScoringMethod("activation_weighted", True,
                       lambda mlp, act=None, **k: activation_scores(mlp, _require(act), "combined"),
                       "activation importance x outgoing column norm (needs data)"))


def _require(act):
    if act is None:
        raise ValueError("this scoring method requires calibration activations")
    return act


# --------------------------------------------------------------------------- #
# combination / normalisation
# --------------------------------------------------------------------------- #
def normalize_scores(scores: torch.Tensor, mode: str = "mass") -> torch.Tensor:
    """Normalise a per-neuron score vector to a stable [0,1] range.

    ``mass`` divides by the sum so scores are interpretable as importance mass
    (this is what makes cross-layer global comparison legitimate).
    ``rank`` maps to percentile rank (robust to outlier scales).
    """
    s = scores.detach().float().clamp_min(0)
    if s.numel() == 0:
        return s
    if mode == "rank":
        order = torch.argsort(s)
        ranks = torch.empty_like(s)
        ranks[order] = torch.arange(s.numel(), dtype=s.dtype)
        return ranks / max(1, s.numel() - 1)
    total = s.sum()
    if total <= 0:
        return torch.zeros_like(s)
    return s / total


def combine_scores(weight_scores: torch.Tensor, act_scores: torch.Tensor,
                   alpha: float = 0.5) -> torch.Tensor:
    """Blend two normalised score vectors (alpha = weight on activations)."""
    a = normalize_scores(act_scores, "rank")
    w = normalize_scores(weight_scores, "rank")
    return (1.0 - alpha) * w + alpha * a


def select_by_budget(scores: torch.Tensor, keep: int) -> torch.Tensor:
    """Indices of the ``keep`` highest-scoring neurons (stable order)."""
    n = scores.numel()
    keep = max(0, min(int(keep), n))
    if keep == n:
        return torch.arange(n, dtype=torch.long)
    return torch.topk(scores, keep, largest=True, sorted=False).indices.sort().values


def removed_mass(scores_norm: torch.Tensor, keep: int) -> float:
    """Fraction of importance mass lost when keeping ``keep`` neurons."""
    n = scores_norm.numel()
    if n == 0:
        return 0.0
    keep_idx = select_by_budget(scores_norm, keep)
    return float(1.0 - scores_norm[keep_idx].sum())
