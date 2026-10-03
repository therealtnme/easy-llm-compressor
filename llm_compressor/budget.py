"""Neuron / head budget arithmetic.

Quantities are always kept separate: this module only ever allocates *one*
kind of unit (MLP intermediate channels by default). Percentage and exact
counts are both supported, and allocation can be global (rank all neurons in
the model by score), uniform (equal per layer) or hybrid (global ranking with
per-layer removal caps).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from .scoring import normalize_scores, select_by_budget


class BudgetError(ValueError):
    """The requested budget cannot be met without breaking a hard guard."""


SCOPES = ("global", "uniform", "hybrid")


def resolve_target(total: int, percent: Optional[float] = None,
                   count: Optional[int] = None,
                   min_keep_ratio: float = 0.05) -> int:
    """Number of units to remove, from either a percentage or an exact count."""
    if total <= 0:
        raise BudgetError("no removable units found")
    if (percent is None) == (count is None):
        raise BudgetError("specify exactly one of percent or count")
    if percent is not None:
        if not 0.0 < float(percent) < 100.0:
            raise BudgetError("percent must be between 0 and 100 (exclusive)")
        removed = int(math.floor(total * float(percent) / 100.0))
    else:
        removed = int(count)
        if removed < 0:
            raise BudgetError("count must be non-negative")
    floor = int(total * float(min_keep_ratio))
    if removed > total - floor:
        raise BudgetError(
            f"budget removes {removed} of {total} units, which would keep fewer "
            f"than the minimum {floor} units ({min_keep_ratio:.0%})")
    return removed


@dataclass
class LayerSpec:
    path: str
    scores: torch.Tensor
    protected: bool = False
    forced_keep: Optional[int] = None  # exact number of units to keep


def _caps(width: int, max_remove_ratio: float) -> int:
    return max(0, min(width - 1, int(width * max_remove_ratio)))


def allocate(widths: dict[str, int], scores: dict[str, torch.Tensor],
             target_removed: int, scope: str = "global",
             max_remove_ratio: float = 0.9,
             protected: Sequence[str] = (),
             forced_keep: Optional[dict[str, int]] = None) -> dict:
    """Return ``{"keep": {path: [idx]}, "removed": n, "mass": float,
    "per_layer_removed": {path: n}}``.

    ``scope='uniform'`` removes (approximately) the same number of units per
    layer; ``global`` ranks every unit of every layer together so removal is
    uneven by construction; ``hybrid`` is global ranking with a per-layer cap
    applied first (it falls back to a second pass if the cap makes the budget
    unreachable).
    """
    if scope not in SCOPES:
        raise BudgetError(f"unknown allocation scope '{scope}'")
    protected = set(protected)
    forced_keep = dict(forced_keep or {})
    paths = [p for p in widths if p not in protected]
    if not paths:
        raise BudgetError("every candidate layer is protected")
    if forced_keep:
        paths = [p for p in paths if p in forced_keep] or paths

    caps = {p: _caps(widths[p], max_remove_ratio) for p in paths}
    forced = {p: int(forced_keep[p]) for p in paths if p in forced_keep}
    total_cap = sum(caps[p] for p in paths)
    for p, keep in forced.items():
        if keep > widths[p] or keep < 1:
            raise BudgetError(f"{p}: forced keep {keep} out of range")
    if target_removed > total_cap:
        raise BudgetError(
            f"cannot remove {target_removed} units: at most {total_cap} are "
            f"removable under max_remove_ratio={max_remove_ratio} and the "
            "protection policy")

    keep: dict[str, list[int]] = {}
    per_layer: dict[str, int] = {}

    if forced:
        for p in paths:
            k = forced[p]
            keep[p] = select_by_budget(scores[p], k).tolist()
            per_layer[p] = widths[p] - k
        target_removed -= sum(per_layer.values())
        paths = [p for p in paths if p not in forced]
        if target_removed < 0:
            raise BudgetError("forced widths already exceed the neuron budget")
        if target_removed == 0 or not paths:
            return _finish(widths, scores, keep, per_layer, paths, forced)

    if scope == "uniform":
        shares = _even_shares([caps[p] for p in paths], target_removed)
        for p, share in zip(paths, shares):
            keep[p] = select_by_budget(scores[p], widths[p] - share).tolist()
            per_layer[p] = share
    else:
        ranked = _global_ranking(paths, widths, scores, caps)
        chosen = ranked[:target_removed]
        counts = {p: 0 for p in paths}
        for p, _idx in chosen:
            counts[p] += 1
        for p in paths:
            keep[p] = select_by_budget(scores[p], widths[p] - counts[p]).tolist()
            per_layer[p] = counts[p]
    return _finish(widths, scores, keep, per_layer, paths, forced)


def _finish(widths, scores, keep, per_layer, paths, forced):
    for p in paths:
        if p not in keep:
            keep[p] = select_by_budget(scores[p], widths[p] - per_layer.get(p, 0)).tolist()
    mass = 0.0
    for p, idx in keep.items():
        n = normalize_scores(scores[p], "mass")
        mass += float(1.0 - n[torch.tensor(idx, dtype=torch.long)].sum())
    removed = sum(widths[p] - len(keep[p]) for p in keep)
    return {"keep": keep, "removed": removed, "mass": mass,
            "per_layer_removed": per_layer}


def _global_ranking(paths, widths, scores, caps):
    """Rank every unit model-wide. Per-layer scores are first normalised to
    comparable 'importance share' so a wide layer cannot dominate."""
    ranked = []
    for p in paths:
        s = scores[p].detach().float().clamp_min(0)
        share = s / s.sum() if float(s.sum()) > 0 else torch.zeros_like(s)
        n = widths[p]
        if caps[p] < n - 1:
            keep = select_by_budget(share, n - caps[p])
            removable = torch.ones(n, dtype=torch.bool)
            removable[keep] = False
            idxs = torch.nonzero(removable).flatten().tolist()
        else:
            idxs = list(range(n))
        ranked += [(p, int(i), float(share[i])) for i in idxs]
    ranked.sort(key=lambda t: t[2])
    return [(p, i) for p, i, _ in ranked]


def _even_shares(caps: list[int], target: int) -> list[int]:
    n = len(caps)
    if n == 0:
        return []
    base = target // n
    shares = [min(base, c) for c in caps]
    left = target - sum(shares)
    order = sorted(range(n), key=lambda i: -(caps[i] - shares[i]))
    while left > 0:
        progressed = False
        for i in order:
            if left == 0:
                break
            if shares[i] < caps[i]:
                shares[i] += 1
                left -= 1
                progressed = True
        if not progressed:
            raise BudgetError("uniform allocation cannot reach the budget")
    return shares
