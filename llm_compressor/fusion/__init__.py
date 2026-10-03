"""Affine fusion algebra and structural safety analysis.

Exact fusion is exact algebra, not an approximation:

    y = W2 (W1 x + b1) + b2  ==  (W2 W1) x + (W2 b1 + b2)

It is only *legitimate* when the two projections really are adjacent in the
computation graph. ``EXACT_AFFINE`` (the algebra applies) is therefore separate
from ``EXACT_FUSION_ALLOWED`` (the graph permits merging): branching, shared
parameters, an interleaved nonlinearity/normalisation, protection, or a
boundary that cannot be collapsed without breaking the graph all veto it.

The check is symbolic/structural only. Nothing here uses random tensors to
decide whether a graph is affine.

"Exact" also does not mean "cheaper": composing ``in->mid->out`` yields
``in*out`` weights instead of ``in*mid + mid*out``, which is a *saving* only
when ``mid`` is larger than the smaller of the two ends. :func:`chain_cost`
computes that and callers must consult it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from ..model.projection import Projection, is_projection

AFFINE = "EXACT_AFFINE"
NONLINEAR = "NONLINEAR"
UNKNOWN = "UNKNOWN"

_NONLINEAR_TYPES: tuple[type, ...] = (
    nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Mish, nn.ELU, nn.SELU, nn.CELU,
    nn.Tanh, nn.Sigmoid, nn.Softplus, nn.Softsign, nn.Hardswish, nn.Hardsigmoid,
    nn.Hardtanh, nn.LeakyReLU, nn.PReLU, nn.LogSigmoid, nn.Softmax, nn.LogSoftmax,
    nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.LocalResponseNorm,
    nn.GLU,
)

_TRANSPARENT_TYPES: tuple[type, ...] = (nn.Identity, nn.Dropout, nn.Dropout1d,
                                        nn.Dropout2d)


def classify_module(module: nn.Module) -> str:
    """Symbolic classification of one module: AFFINE / NONLINEAR / UNKNOWN."""
    if is_projection(module):
        return AFFINE
    if isinstance(module, _NONLINEAR_TYPES):
        return NONLINEAR
    name = type(module).__name__.lower()
    if any(tok in name for tok in ("norm", "relu", "gelu", "silu", "swish",
                                   "sigmoid", "tanh", "softmax", "activation")):
        return NONLINEAR
    if isinstance(module, _TRANSPARENT_TYPES):
        # Dropout is only exactly the identity in eval mode.
        return AFFINE if not module.training else UNKNOWN
    return UNKNOWN


# --------------------------------------------------------------------------- #
# algebra
# --------------------------------------------------------------------------- #
@dataclass
class FusionMeta:
    path_a: str = ""
    path_b: str = ""
    classification: str = UNKNOWN          # EXACT_AFFINE | NONLINEAR | UNKNOWN
    affine_algebra: bool = False           # W = W2@W1, b = W2@b1 + b2
    has_branch: bool = False               # residual / multi-consumer
    has_shared_parameters: bool = False
    has_nonlinearity: bool = False
    has_normalization: bool = False
    protected: bool = False
    collapsible_without_breaking_graph: bool = False
    shape_ok: bool = False
    cost_before: int = 0
    cost_after: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def cheaper(self) -> bool:
        return self.cost_before > 0 and self.cost_after < self.cost_before

    @property
    def exact_fusion_allowed(self) -> bool:
        return (self.affine_algebra and self.shape_ok
                and not self.has_branch and not self.has_shared_parameters
                and not self.has_nonlinearity and not self.has_normalization
                and not self.protected
                and self.collapsible_without_breaking_graph)

    def to_dict(self) -> dict:
        return {
            "path_a": self.path_a, "path_b": self.path_b,
            "classification": self.classification,
            "EXACT_AFFINE": self.affine_algebra,
            "EXACT_FUSION_ALLOWED": self.exact_fusion_allowed,
            "has_branch": self.has_branch,
            "has_shared_parameters": self.has_shared_parameters,
            "has_nonlinearity": self.has_nonlinearity,
            "has_normalization": self.has_normalization,
            "protected": self.protected,
            "collapsible_without_breaking_graph":
                self.collapsible_without_breaking_graph,
            "shape_ok": self.shape_ok,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "cheaper": self.cheaper,
            "reasons": list(self.reasons),
        }


class FusionError(RuntimeError):
    pass


def fuse_affine(a: Projection, b: Projection) -> dict:
    """Exact composition ``b(a(x))`` for two dense projections."""
    if a.out_features != b.in_features:
        raise FusionError(
            f"cannot fuse {a.name} ({a.out_features} outputs) with {b.name} "
            f"({b.in_features} inputs)")
    wa = a.weight_matrix.detach().to(torch.float32)
    wb = b.weight_matrix.detach().to(torch.float32)
    if not (torch.isfinite(wa).all() and torch.isfinite(wb).all()):
        raise FusionError("non-finite weight encountered; refusing to fuse")
    new_w = wb @ wa
    if a.has_bias:
        b1 = a.bias.detach().to(torch.float32)
    else:
        b1 = torch.zeros(wa.shape[0], dtype=torch.float32)
    if b.has_bias:
        new_b = wb @ b1 + b.bias.detach().to(torch.float32)
    else:
        new_b = wb @ b1
    return {
        "weight": new_w.to(wa.dtype if wa.dtype.is_floating_point else torch.float32),
        "bias": new_b,
        "in_features": a.in_features,
        "out_features": b.out_features,
        "detail": {
            "rule": "W = W2 @ W1 ; b = W2 @ b1 + b2",
            "bias_a_present": a.has_bias, "bias_b_present": b.has_bias,
            "layout_a": a.layout, "layout_b": b.layout,
        },
    }


def compose_chain(projections: Sequence[Projection]) -> dict:
    """Exact composition of ``len(projections) >= 1`` adjacent projections.

    Works for ``nn.Linear`` and ``Conv1D``-style projections because
    :class:`Projection` normalises both to ``(out, in)``.
    """
    if not projections:
        raise FusionError("empty chain")
    for p, q in zip(projections, projections[1:]):
        if p.out_features != q.in_features:
            raise FusionError(
                f"shape mismatch: {p.name} ({p.out_features}) -> "
                f"{q.name} ({q.in_features})")
    w = projections[0].weight_matrix.detach().to(torch.float32)
    if not torch.isfinite(w).all():
        raise FusionError("non-finite weight encountered; refusing to fuse")
    if projections[0].has_bias:
        b = projections[0].bias.detach().to(torch.float32)
    else:
        b = torch.zeros(w.shape[0], dtype=torch.float32)
    for p in projections[1:]:
        wp = p.weight_matrix.detach().to(torch.float32)
        if not torch.isfinite(wp).all():
            raise FusionError("non-finite weight encountered; refusing to fuse")
        w = wp @ w
        if p.has_bias:
            b = wp @ b + p.bias.detach().to(torch.float32)
        else:
            b = wp @ b
    return {
        "weight": w,
        "bias": b,
        "in_features": projections[0].in_features,
        "out_features": projections[-1].out_features,
        "detail": {
            "rule": "W = W_n @ ... @ W_1 ; b = (prod W) b_1 + ... + b_n",
            "n_projections": len(projections),
            "paths": [p.name for p in projections],
        },
    }


def chain_cost(projections: Sequence[Projection]) -> dict:
    """Parameter cost of the chain before and after exact fusion."""
    before = sum(p.n_parameters() for p in projections)
    first, last = projections[0], projections[-1]
    after = first.in_features * last.out_features
    if last.has_bias:
        after += last.out_features
    return {"before": before, "after": after, "saving": before - after,
            "cheaper": after < before}


# --------------------------------------------------------------------------- #
# structural discovery
# --------------------------------------------------------------------------- #
def model_owner(container: nn.Module):
    """Best-effort ``[(name, module)]`` of a container's owner, for paths."""
    parent = getattr(container, "_llmc_owner", None)
    if parent is None:
        return []
    return [(n, m) for n, m in parent.named_modules()]


def _child_path(container: str, child: str) -> str:
    return f"{container}.{child}".strip(".") if container else child


def _other_consumers(container: nn.Module, projections: Sequence[Projection]) -> bool:
    """True when one of the chain's weights is also used outside the chain."""
    own = {id(p.weight) for p in projections}
    paths = {p.name for p in projections}
    for name, par in container.named_parameters(remove_duplicate=False):
        if id(par) in own and name.rsplit(".", 1)[0] not in paths:
            return True
    return False


def scan_sequential_chains(model: nn.Module) -> list[dict]:
    """Find adjacent dense pairs inside ``nn.Sequential`` chains.

    Sequential order is explicit in the graph, so a run of affine modules with
    nothing in between is provably fusible. Any non-affine module breaks the
    run, which is exactly why transformer blocks are refused.
    """
    found: list[dict] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        run: list[tuple[str, nn.Module]] = []
        runs: list[list[tuple[str, nn.Module]]] = []
        for child in children:
            if classify_module(child[1]) == AFFINE:
                run.append(child)
            else:
                if run:
                    runs.append(run)
                run = []
        if run:
            runs.append(run)
        for group in runs:
            projs = [Projection.of(_child_path(name, n), m) for n, m in group
                     if is_projection(m)]
            if len(projs) < 2:
                continue
            cost = chain_cost(projs)
            meta = FusionMeta(
                path_a=projs[0].name, path_b=projs[-1].name,
                classification=AFFINE, affine_algebra=True,
                shape_ok=all(p.out_features == q.in_features
                             for p, q in zip(projs, projs[1:])),
                collapsible_without_breaking_graph=True,
                cost_before=cost["before"], cost_after=cost["after"],
            )
            if not meta.shape_ok:
                meta.reasons.append("chain shapes do not compose")
            if not cost["cheaper"]:
                meta.reasons.append(
                    f"fusion is not cheaper ({cost['before']} -> "
                    f"{cost['after']} parameters)")
            meta.has_shared_parameters = _other_consumers(model, projs)
            if meta.has_shared_parameters:
                meta.reasons.append("a projection shares its weights")
            found.append({
                "container": name,
                "pair": (projs[0].name, projs[-1].name),
                "names": [p.name for p in projs],
                "start": [n for n, _ in children].index(group[0][0]),
                "length": len(group),
                "projections": projs,
                "meta": meta,
                "cost": cost,
            })
    return found


def analyze_layer_pair(arch, stack_name: str, index: int) -> FusionMeta:
    """Why layer-granularity exact fusion is (almost always) refused."""
    stack = next((s for s in arch.stacks if s.name == stack_name), None)
    if stack is None:
        raise FusionError(f"unknown stack '{stack_name}'")
    blocks = stack.blocks()
    if index + 1 >= len(blocks):
        raise FusionError("no next layer to fuse with")
    layers = [L for L in arch.layers if L.stack == stack_name]
    a = layers[index]
    b = layers[index + 1]
    meta = FusionMeta(path_a=a.name, path_b=b.name, classification=UNKNOWN)
    meta.shape_ok = bool(arch.hidden_size)
    if a.has_residual or b.has_residual:
        meta.has_branch = True
        meta.reasons.append("layer output feeds a residual branch "
                            f"({a.residual_evidence or 'residual evidence'})")
    if getattr(a, "mlp", None) is not None and a.mlp.kind in ("gated", "standard"):
        meta.has_nonlinearity = True
        meta.reasons.append("block contains an MLP activation")
    if any(getattr(n, "kind", "") in ("layernorm", "rmsnorm") for n in a.norms):
        meta.has_normalization = True
        meta.reasons.append("block contains a normalisation applied between the "
                            "two candidate projections")
    protected = {p.name for p in arch.protected}
    if a.name in protected or b.name in protected:
        meta.protected = True
        meta.reasons.append("layers are protected")
    meta.collapsible_without_breaking_graph = not (
        meta.has_branch or meta.has_nonlinearity or meta.has_normalization)
    if not meta.exact_fusion_allowed and not meta.reasons:
        meta.reasons.append("no proof available that the merge is faithful")
    return meta


def detect_exact_fusion(model: nn.Module, arch) -> dict:
    """Model-level exact-fusion capability, with the structural reasons."""
    chains = scan_sequential_chains(model)
    allowed = [c for c in chains if c["meta"].exact_fusion_allowed]
    layer_pairs = []
    stack = arch.primary_stack()
    if stack is not None and len(stack.blocks()) >= 2:
        for i in range(min(len(stack.blocks()) - 1, 3)):
            layer_pairs.append(analyze_layer_pair(arch, stack.name, i))
    classified = [{"name": n, "classification": classify_module(m)}
                  for n, m in list(model.named_modules())[:0]]  # reserved
    return {
        "sequential_candidates": len(chains),
        "allowed": len(allowed),
        "candidates": [{"names": c["names"], "container": c["container"],
                        "cost": c["cost"], "meta": c["meta"].to_dict()}
                       for c in chains[:8]],
        "layer_pairs": [m.to_dict() for m in layer_pairs],
        "classified_probes": classified,
        "reason": ("no adjacent affine chain was found: every candidate pair has "
                   "an intervening nonlinearity, branch or normalisation"
                   if not allowed else f"{len(allowed)} fusible affine chain(s)"),
    }


# --------------------------------------------------------------------------- #
# physical fusion
# --------------------------------------------------------------------------- #
def fuse_chain_in_sequential(container: nn.Sequential, start: int, length: int,
                             *, require_cheaper: bool = True) -> dict:
    """Physically replace ``container[start:start+length]`` with one Linear.

    The absorbed modules are removed from the container (their parameters no
    longer exist). Refuses when the run is not entirely affine, when the shapes
    do not compose, when a weight is shared, or (optionally) when the result
    would not be cheaper.
    """
    children = list(container.named_children())
    span = children[start:start + length]
    if len(span) != length:
        raise FusionError(f"container has no slice [{start}:{start+length}]")
    container_name = next((n for n, m in model_owner(container) if m is container),
                          "")
    projs: list[Projection] = []
    for name, module in span:
        kind = classify_module(module)
        if kind != AFFINE:
            raise FusionError(f"{name} is {kind}, not affine; refusing to fuse")
        if is_projection(module):
            projs.append(Projection.of(_child_path(container_name, name), module))
    if len(projs) < 2:
        raise FusionError("a chain needs at least two dense projections")
    if _other_consumers(container, projs):
        raise FusionError("a projection in the chain shares its weights")
    cost = chain_cost(projs)
    if require_cheaper and not cost["cheaper"]:
        raise FusionError(
            f"exact fusion is not cheaper here: {cost['before']} -> "
            f"{cost['after']} parameters")
    composed = compose_chain(projs)
    first, last = projs[0], projs[-1]
    lin = nn.Linear(first.in_features, last.out_features, bias=last.has_bias,
                    dtype=last.weight.dtype, device=last.weight.device)
    with torch.no_grad():
        lin.weight.copy_(composed["weight"].to(lin.weight.dtype))
        if lin.bias is not None:
            lin.bias.copy_(composed["bias"].to(lin.bias.dtype))
        lin.weight.requires_grad = last.weight.requires_grad
    rebuilt: dict[str, nn.Module] = {}
    for idx, (name, module) in enumerate(children):
        if idx == start:
            rebuilt[span[0][0]] = lin
        elif start < idx < start + length:
            continue
        else:
            rebuilt[name] = module
    container._modules = nn.modules.container.OrderedDict(rebuilt)
    return {
        "container": "", "absorbed": [p.name for p in projs],
        "replaced": span[0][0], "in_features": first.in_features,
        "out_features": last.out_features, "cost": cost,
        "detail": composed["detail"],
    }


def fuse_pair_in_model(model: nn.Module, target: Projection,
                       source: Projection, *, require_cheaper: bool = False) -> dict:
    """Fuse two adjacent projections that are siblings in an ``nn.Sequential``."""
    container = None
    start = length = -1
    for _, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        kids = [m for _, m in module.named_children()]
        if target.module in kids and source.module in kids:
            i, j = kids.index(source.module), kids.index(target.module)
            if j == i + 1:
                container, start, length = module, i, 2
                break
    if container is None:
        raise FusionError(
            "fuse_pair_in_model only supports adjacent siblings of the same "
            "nn.Sequential; use fuse_chain_in_sequential otherwise")
    info = fuse_chain_in_sequential(container, start, length,
                                    require_cheaper=require_cheaper)
    return info["detail"]


@torch.no_grad()
def verify_fusion(model: nn.Module, container: nn.Sequential, start: int,
                  length: int, probe: dict, atol: float = 1e-5,
                  rtol: float = 1e-4) -> dict:
    """Fuse for real, confirm the model's output is unchanged, then restore."""
    from ..utils.batch import forward_tensor

    before = forward_tensor(model, probe).float()
    saved = nn.modules.container.OrderedDict(container._modules)
    try:
        info = fuse_chain_in_sequential(container, start, length,
                                        require_cheaper=False)
        after = forward_tensor(model, probe).float()
    finally:
        container._modules = saved
    diff = float((before - after).abs().max())
    return {"verified": bool(torch.allclose(before, after, atol=atol, rtol=rtol)),
            "max_abs_diff": diff, "fusion": info, "probe": "structural_probe"}


def _dense(module: Any) -> bool:
    return is_projection(module)
