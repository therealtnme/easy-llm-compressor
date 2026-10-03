"""Affine fusion algebra and structural safety analysis.

Exact fusion is exact algebra, not an approximation:

    y = W2 (W1 x + b1) + b2  ==  (W2 W1) x + (W2 b1 + b2)

It is only *legitimate* when the two projections really are adjacent in the
computation graph. ``EXACT_AFFINE`` (the algebra applies) is therefore separate
from ``EXACT_FUSION_ALLOWED`` (the graph permits merging): branching, shared
parameters, an interleaved nonlinearity/normalisation, protection, or a
boundary that cannot be collapsed without breaking the graph all veto it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import torch

from ..model.projection import Projection, is_projection
from ..model.names import tokenize_name


@dataclass
class FusionMeta:
    path_a: str = ""
    path_b: str = ""
    affine_algebra: bool = False          # W = W2@W1, b = W2@b1 + b2
    has_branch: bool = False              # residual / multi-consumer
    has_shared_parameters: bool = False
    has_nonlinearity: bool = False
    has_normalization: bool = False
    protected: bool = False
    collapsible_without_breaking_graph: bool = False
    shape_ok: bool = False
    reasons: list[str] = field(default_factory=list)

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


def fuse_pair_in_model(target: Projection, source: Projection) -> dict:
    """Replace ``target`` (the later projection) with the fused matrix and
    neutralise ``source`` (set to the identity). Returns the algebra detail."""
    fused = fuse_affine(source, target)
    result = target.resize(fused["in_features"], fused["out_features"])
    result = None  # resize() normalises shapes; weights set below
    w = fused["weight"]
    if target.layout == "transposed":
        w = w.t()
    requires_grad = target.weight.requires_grad
    with torch.no_grad():
        target.module.weight.data = w.to(target.module.weight.dtype)
        if getattr(target.module, "bias", None) is not None:
            target.module.bias.data = fused["bias"].to(target.module.bias.dtype)
    from ..model.projection import _set_weight_matrix_if_needed  # noqa: F401

    _neutralise_as_identity(source)
    target._update_shapes()
    return fused["detail"]


def _neutralise_as_identity(p: Projection) -> None:
    """Make ``p`` pass its input through unchanged (identity), so removing the
    *later* fused layer's input dependence on it keeps the graph faithful."""
    n = min(p.in_features, p.out_features)
    w = torch.zeros((p.out_features, p.in_features) if p.layout == "linear"
                    else (p.in_features, p.out_features), dtype=p.weight.dtype)
    diag = torch.arange(n)
    if p.layout == "linear":
        w[diag, diag] = 1.0
    else:
        w[diag, diag] = 1.0
    with torch.no_grad():
        p.module.weight.data = w
        if getattr(p.module, "bias", None) is not None:
            p.module.bias.data.zero_()
    p._update_shapes()


def scan_sequential_chains(model) -> list[dict]:
    """Find genuinely adjacent dense pairs inside ``nn.Sequential`` chains.

    Sequential order is explicit in the graph, so a pair of consecutive dense
    projections with nothing between them is provably fusible.
    """
    import torch.nn as nn

    found = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential):
            continue
        children = list(module.named_children())
        for i in range(len(children) - 1):
            (n1, m1), (n2, m2) = children[i], children[i + 1]
            if not (is_projection(m1) and is_projection(m2)):
                continue
            p1 = Projection.of(f"{name}.{n1}", m1)
            p2 = Projection.of(f"{name}.{n2}", m2)
            meta = FusionMeta(path_a=p1.name, path_b=p2.name,
                              affine_algebra=True,
                              shape_ok=p1.out_features == p2.in_features,
                              collapsible_without_breaking_graph=True)
            if not meta.shape_ok:
                meta.reasons.append(
                    f"shape mismatch: {p1.out_features} -> {p2.in_features}")
            p1_id = id(m1.weight)
            p2_id = id(m2.weight)
            shared = any(
                id(par) in (p1_id, p2_id)
                for other, par in model.named_parameters(remove_duplicate=False)
                if other not in (p1.name, p2.name))
            meta.has_shared_parameters = shared
            if shared:
                meta.reasons.append("one of the projections shares its weights")
            found.append({"container": name, "pair": (p1.name, p2.name),
                          "meta": meta, "projections": (p1, p2)})
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
    meta = FusionMeta(path_a=a.name, path_b=b.name, affine_algebra=False)
    meta.shape_ok = bool(arch.hidden_size)
    if a.has_residual or b.has_residual:
        meta.has_branch = True
        meta.reasons.append("layer output feeds a residual branch "
                            f"({a.residual_evidence or 'residual evidence'})")
    if a.mlp is not None and a.mlp.kind in ("gated", "standard"):
        meta.has_nonlinearity = True
        meta.reasons.append("block contains an MLP activation")
    if any(n.kind in ("layernorm", "rmsnorm") for n in a.norms):
        meta.has_normalization = True
        meta.reasons.append("block contains a normalisation that is applied "
                            "between the two candidate projections")
    protected = {p.name for p in arch.protected}
    if a.name in protected or b.name in protected:
        meta.protected = True
        meta.reasons.append("layers are protected")
    meta.collapsible_without_breaking_graph = not (
        meta.has_branch or meta.has_nonlinearity or meta.has_normalization)
    if not meta.exact_fusion_allowed and not meta.reasons:
        meta.reasons.append("no proof available that the merge is faithful")
    return meta


def detect_exact_fusion(model, arch) -> dict:
    """Model-level exact-fusion capability, with the structural reasons."""
    chains = scan_sequential_chains(model)
    allowed = [c for c in chains if c["meta"].exact_fusion_allowed]
    layer_pairs = []
    stack = arch.primary_stack()
    if stack is not None and len(stack.blocks()) >= 2:
        for i in range(min(len(stack.blocks()) - 1, 3)):
            layer_pairs.append(analyze_layer_pair(arch, stack.name, i))
    return {
        "sequential_candidates": len(chains),
        "allowed": len(allowed),
        "candidates": [{"pair": c["pair"], "meta": c["meta"].to_dict()}
                       for c in chains[:8]],
        "layer_pairs": [m.to_dict() for m in layer_pairs],
        "reason": ("no adjacent dense pair without an intervening nonlinearity, "
                   "branch or normalisation was found"
                   if not allowed else f"{len(allowed)} fusible dense pair(s)"),
    }


@torch.no_grad()
def verify_fusion(model, pair: Sequence[Projection], probe: dict) -> dict:
    """Numerically confirm the fused chain reproduces the original outputs."""
    from ..utils.batch import forward_tensor

    a, b = pair
    x = probe["input_ids"]
    before = forward_tensor(model, probe)
    w = a.weight_matrix.detach().clone()
    bias = a.bias.detach().clone() if a.has_bias else None
    _neutralise_as_identity(a)
    fused = fuse_affine(Projection.of(a.name, a.module), b)
    with torch.no_grad():
        b.module.weight.data = (fused["weight"].t() if b.layout == "transposed"
                                else fused["weight"]).to(b.module.weight.dtype)
        if getattr(b.module, "bias", None) is not None:
            b.module.bias.data = fused["bias"].to(b.module.bias.dtype)
    b._update_shapes()
    after = forward_tensor(model, probe)
    diff = float((before.float() - after.float()).abs().max())
    return {"verified": bool(torch.allclose(before.float(), after.float(),
                                            atol=1e-4, rtol=1e-3)),
            "max_abs_diff": diff, "probe": "structural_probe"}


def _dense(module: Any) -> bool:
    return is_projection(module)
