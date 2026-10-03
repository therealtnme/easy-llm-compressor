"""Joint layer-depth and neuron-width search (beam search, no additive
assumption about *cost*).

Candidates are described by explicit fields:

    region_start / region_end : teacher layer span covered by the candidate
    operation                 : KEEP | DELETE | EXACT_FUSE | DISTILL_FUSE
    student_depth             : number of student layers the span maps to
    student_widths            : per surviving layer neuron counts

Hard constraints: the student depth equals the requested target
(``teacher_depth > target_student_depth`` is validated by the pipeline), the
global neuron budget is respected exactly, and protected components never
appear in a candidate. Beam search keeps the best ``beam_width`` partial
segmentations per (position, depth) and scores complete plans with the joint
layer + neuron cost, so depth and width are decided together rather than in
sequential prune -> fuse -> prune passes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import torch

from .. import budget as budget_mod

OPS = ("KEEP", "DELETE", "EXACT_FUSE", "DISTILL_FUSE")


@dataclass
class Region:
    region_start: int
    region_end: int          # exclusive
    operation: str
    student_depth: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    regions: list[Region]
    survivors: list[int]
    target_student_layers: int
    teacher_depth: int
    layer_cost: float = 0.0
    neuron_cost: float = 0.0
    total_cost: float = 0.0
    allocation: dict = field(default_factory=dict)
    removed_neurons: int = 0
    removed_mass: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


def layer_saliency(model, arch, stack_name: str,
                   batches: Optional[Sequence[dict]] = None,
                   stack_saliency: Optional[dict] = None) -> dict:
    """Per-layer importance in [0, 1] (higher = more costly to delete).

    With real batches this measures how much each block changes its input
    (``1 - cosine(input, output)``) on actual data. Without data it falls back
    to a weight-norm proxy and says so, rather than inventing activations.
    """
    from ..model.names import is_prefix

    stack = next((s for s in arch.stacks if s.name == stack_name), None)
    if stack is None:
        raise ValueError(f"unknown stack '{stack_name}'")
    blocks = stack.blocks()
    if batches:
        acc: dict[int, list[float]] = {i: [] for i in range(len(blocks))}
        handles = []

        def hook(i):
            def fn(module, inputs, output):
                x = inputs[0]
                y = output[0] if isinstance(output, tuple) else output
                if not torch.is_tensor(x) or not torch.is_tensor(y):
                    return
                if x.shape != y.shape:
                    return
                xf = x.detach().float().reshape(-1, x.shape[-1])
                yf = y.detach().float().reshape(-1, y.shape[-1])
                cos = torch.nn.functional.cosine_similarity(xf, yf, dim=-1).mean()
                acc[i].append(float(1.0 - cos))
            return fn

        for i, b in enumerate(blocks):
            handles.append(b.register_forward_hook(hook(i)))
        try:
            with torch.no_grad():
                for batch in batches:
                    probe = {k: v for k, v in batch.items() if k != "provenance"}
                    model(**probe, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        raw = {i: (sum(v) / len(v) if v else 0.0) for i, v in acc.items()}
        method = "activation (1 - cos(h_in, h_out)) on real calibration data"
    else:
        raw = {}
        for i, b in enumerate(blocks):
            total = 0.0
            for name, p in b.named_parameters():
                toks = set()
                leaf = name.split(".")[-1]
                if leaf == "weight" and ("down" in name or "o_proj" in name
                                          or "out_proj" in name or "c_proj" in name):
                    total += float(p.detach().float().norm())
            raw[i] = total
        method = ("weight-norm proxy on output projections "
                  "(DATASET-FREE MODE: no activations are invented)")
    if stack_saliency is not None:
        lo = stack_saliency.get("min", 0.0)
        hi = stack_saliency.get("max", 1.0)
    else:
        lo, hi = None, None
    values = list(raw.values())
    vmin, vmax = min(values), max(values)
    span = (vmax - vmin) or 1.0
    norm = {i: (v - vmin) / span for i, v in raw.items()}
    return {"values": norm, "raw": raw, "method": method,
            "stack": stack_name, "layers": len(blocks)}


def _region_cost(region: Region, sal: dict) -> float:
    span = region.region_end - region.region_start
    if region.operation == "KEEP":
        return 0.0
    total = sum(sal[i] for i in range(region.region_start, region.region_end))
    if region.operation == "DELETE":
        return total
    if region.operation == "EXACT_FUSE":
        return 0.0
    if region.operation == "DISTILL_FUSE":
        removed = span - region.student_depth
        return total * (removed / span) * 0.6
    raise ValueError(f"unknown operation {region.operation}")


def joint_search(arch, stack_name: str, sal: dict, widths: dict, scores: dict,
                 remove_target: int, target_student_layers: int, *,
                 beam_width: int = 16, max_span: int = 4,
                 max_remove_ratio: float = 0.9,
                 protected: Sequence[str] = (),
                 allow_exact_fuse: bool = True,
                 allow_distill_fuse: bool = True,
                 exact_fuse_reason: str = "",
                 distill_available: bool = True,
                 top_k: int = 3) -> list[Plan]:
    stack = next((s for s in arch.stacks if s.name == stack_name), None)
    if stack is None:
        raise ValueError(f"unknown stack '{stack_name}'")
    depth = len(stack.blocks())
    if not 0 < target_student_layers < depth:
        raise ValueError(
            f"target student depth {target_student_layers} is not < teacher "
            f"depth {depth}: refusing to 'compress' {depth} -> "
            f"{target_student_layers}")

    sal_values = sal["values"]
    # state: (cost, position, depth_so_far, regions)
    states = [(0.0, 0, 0, [])]
    for pos in range(depth):
        nxt = []
        for cost, _p, d, regions in states:
            if d >= target_student_layers:
                continue
            moves = [Region(pos, pos + 1, "KEEP", 1)]
            if d < target_student_layers:
                moves.append(Region(pos, pos + 1, "DELETE", 0))
            if allow_exact_fuse and pos + 2 <= depth and (
                    target_student_layers - d) <= (depth - pos - 1):
                moves.append(Region(pos, pos + 2, "EXACT_FUSE", 1,
                                    reasons=["exact affine algebra available"]))
            if allow_distill_fuse and distill_available:
                for span in range(2, min(max_span, depth - pos) + 1):
                    for q in range(1, span):
                        if target_student_layers - d < q:
                            continue
                        if (depth - (pos + span)) < (target_student_layers - d - q):
                            continue
                        moves.append(Region(pos, pos + span, "DISTILL_FUSE", q))
            for mv in moves:
                nd = d + mv.student_depth
                end = mv.region_end
                if nd > target_student_layers:
                    continue
                if nd + (depth - end) < target_student_layers:
                    continue
                nxt.append((cost + _region_cost(mv, sal_values), end, nd,
                            regions + [mv]))
        if not nxt:
            return []
        nxt.sort(key=lambda s: s[0])
        # keep the best partial segmentations per (position, depth)
        bucket: dict[tuple[int, int], list] = {}
        for state in nxt:
            bucket.setdefault((state[1], state[2]), []).append(state)
        states = []
        for group in bucket.values():
            states.extend(group[:beam_width])

    complete = [s for s in states if s[1] == depth
                and s[2] == target_student_layers]
    plans: list[Plan] = []
    notes: list[str] = []
    for cost, _p, _d, regions in complete:
        survivors = [L for L in range(depth)
                     if not any(r.region_start <= L < r.region_end
                                and r.operation == "DELETE" for r in regions)]
        kept_region_layers = set(survivors)
        if allow_exact_fuse and any(r.operation == "EXACT_FUSE" for r in regions):
            for r in regions:
                if r.operation == "EXACT_FUSE":
                    kept_region_layers.add(r.region_start)
            survivors = sorted(kept_region_layers)
        names = [f"{stack_name}.{L}" for L in survivors]
        try:
            alloc = budget_mod.allocate(widths, scores, remove_target,
                                        scope="global",
                                        max_remove_ratio=max_remove_ratio,
                                        protected=protected)
        except budget_mod.BudgetError as exc:
            notes.append(str(exc))
            continue
        neuron_cost = alloc["mass"]
        plan = Plan(
            regions=regions, survivors=survivors,
            target_student_layers=target_student_layers, teacher_depth=depth,
            layer_cost=cost, neuron_cost=neuron_cost,
            total_cost=cost + neuron_cost, allocation=alloc,
            removed_neurons=alloc["removed"], removed_mass=alloc["mass"],
            notes=[])
        plan.notes.append(f"layer paths: {', '.join(names[:6])}"
                          + (" ..." if len(names) > 6 else ""))
        plans.append(plan)
    plans.sort(key=lambda p: p.total_cost)
    seen = set()
    unique: list[Plan] = []
    for p in plans:
        key = tuple((r.region_start, r.region_end, r.operation,
                     r.student_depth) for r in p.regions)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    out: list[Plan] = []
    for p in unique[:top_k]:
        if not any(r.operation in ("DELETE", "DISTILL_FUSE", "EXACT_FUSE")
                   for r in p.regions):
            p.notes.append("no depth compression in this candidate")
        else:
            if allow_exact_fuse and not exact_fuse_reason:
                p.notes.append("EXACT_FUSE was not structurally provable for "
                              "any region, so only deletion/distillation regions "
                              "were used")
        out.append(p)
    return out
