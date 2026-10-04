"""Physical structural rewrites: FFN neurons, attention heads, layers.

Rules enforced here:
* tensors are actually re-created smaller (no masks, no zeroing);
* a rewrite refuses to touch protected modules or shared (tied/multi-referenced)
  modules;
* after rewriting, an equivalence guard can compare the rewritten model against
  the original function evaluated with the removed channels zeroed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn

from ..model.architecture import ModelArchitecture, SupportLevel
from ..model.names import is_prefix

__all__ = [
    "RewriteError", "PruneRecord", "PruneResult",
    "prune_mlp_neurons", "prune_attention_heads", "delete_layers",
    "mlp_reference_guard", "reference_logits", "compare_to_reference",
    "verify_pruned_equivalence", "sync_config",
]


class RewriteError(RuntimeError):
    """Raised when a rewrite cannot be proven safe. Never silently ignored."""


@dataclass
class PruneRecord:
    path: str
    kind: str            # mlp_neurons | attention_heads | kv_heads | layer_delete
    layer: str = ""
    removed: int = 0
    kept: int = 0
    detail: dict = field(default_factory=dict)


@dataclass
class PruneResult:
    records: list[PruneRecord] = field(default_factory=list)
    removed_neurons: int = 0
    removed_heads: int = 0
    removed_kv_heads: int = 0
    removed_layers: int = 0
    parameters_before: int = 0
    parameters_after: int = 0


# --------------------------------------------------------------------------- #
# FFN neurons
# --------------------------------------------------------------------------- #
def _fused_layout(mlp) -> Optional[tuple[int, int]]:
    """(gate_offset, up_offset) inside a fused gate/up projection."""
    if not mlp.gate_up_fused:
        return None
    inter = mlp.intermediate_size
    return 0, inter  # convention: [gate ; up] concatenated along rows


def prune_mlp_neurons(model: nn.Module, arch: ModelArchitecture,
                      keep: dict[str, Sequence[int]],
                      protected_check=None) -> PruneResult:
    """Physically remove FFN intermediate channels.

    ``keep`` maps an FFN path (``MLPInfo.name``) to the neuron indices to keep.
    """
    if not arch.cap_prune_mlp_neurons.is_supported():
        raise RewriteError(
            "MLP neuron pruning is not supported for this model: "
            f"{arch.cap_prune_mlp_neurons.reason}"
        )
    result = PruneResult()
    result.parameters_before = arch.total_parameters
    by_name = {L.mlp.name: L.mlp for L in arch.layers if L.mlp is not None}

    for path, keep_idx in keep.items():
        mlp = by_name.get(path)
        if mlp is None:
            raise RewriteError(f"unknown FFN path '{path}'")
        if not mlp.is_prunable():
            raise RewriteError(
                f"FFN '{path}' is not safely pruneable: {mlp.reason}"
            )
        if protected_check is not None and protected_check(path):
            raise RewriteError(f"FFN '{path}' is protected")

        inter = mlp.intermediate_size
        keep_list = sorted({int(i) for i in keep_idx})
        if not keep_list:
            raise RewriteError(f"FFN '{path}': refusing to remove every neuron")
        if keep_list[0] < 0 or keep_list[-1] >= inter:
            raise RewriteError(f"FFN '{path}': neuron index out of range")
        if len(keep_list) == inter:
            continue
        idx = torch.tensor(keep_list, dtype=torch.long)

        fused = _fused_layout(mlp)
        if fused is not None:
            gate_off, up_off = fused
            fused_rows = torch.cat([idx + gate_off, idx + up_off])
            mlp.up_proj.slice_outputs(fused_rows.tolist())
        else:
            if mlp.gate_proj is not None:
                mlp.gate_proj.slice_outputs(keep_list)
            if mlp.up_proj is not None:
                mlp.up_proj.slice_outputs(keep_list)
        mlp.down_proj.slice_inputs(keep_list)
        mlp.intermediate_size = len(keep_list)

        result.records.append(PruneRecord(
            path=path, kind="mlp_neurons", layer=path,
            removed=inter - len(keep_list), kept=len(keep_list),
            detail={"original_width": inter, "kept_indices_head": keep_list[:8],
                    "kept_indices_tail": keep_list[-8:]},
        ))
        result.removed_neurons += inter - len(keep_list)

    result.parameters_after = sum(p.numel() for p in model.parameters())
    return result


# --------------------------------------------------------------------------- #
# attention heads
# --------------------------------------------------------------------------- #
def prune_attention_heads(model: nn.Module, arch: ModelArchitecture,
                          keep_q_heads: dict[str, Sequence[int]],
                          keep_kv_heads: Optional[dict[str, Sequence[int]]] = None,
                          protected_check=None) -> PruneResult:
    """Remove attention heads (and whole KV groups) from a model.

    Query heads are grouped: for grouped-query attention a KV head owns
    ``num_key_value_groups`` query heads, so query heads are removed in whole
    groups so that the resulting head layout stays expressible by the config.
    """
    if not arch.cap_prune_attention_heads.is_supported():
        raise RewriteError(
            "attention head pruning is not supported: "
            f"{arch.cap_prune_attention_heads.reason}"
        )
    result = PruneResult()
    result.parameters_before = arch.total_parameters
    attn_by_name = {}
    for L in arch.layers:
        for a in L.attentions:
            attn_by_name[a.name] = a

    for path, want in keep_q_heads.items():
        a = attn_by_name.get(path)
        if a is None or not a.is_prunable():
            raise RewriteError(f"attention '{path}' is not safely pruneable")
        if protected_check is not None and protected_check(path):
            raise RewriteError(f"attention '{path}' is protected")
        groups = a.num_key_value_groups or 1
        keep = sorted({int(i) for i in want})
        if not keep:
            raise RewriteError(f"attention '{path}': refusing to remove all heads")
        by_group: dict[int, list[int]] = {}
        for h in keep:
            by_group.setdefault(h // groups, []).append(h % groups)
        if any(len(v) != groups for v in by_group.values()):
            raise RewriteError(
                f"attention '{path}': query heads must be kept in whole KV groups"
            )
        kv_keep = sorted(by_group.keys())
        hd = a.head_dim
        q_rows = [h * hd + d for h in keep for d in range(hd)]
        o_cols = q_rows
        kv_rows = [h * hd + d for h in kv_keep for d in range(hd)]

        a.q_proj.slice_outputs(q_rows)
        a.k_proj.slice_outputs(kv_rows)
        a.v_proj.slice_outputs(kv_rows)
        a.o_proj.slice_inputs(o_cols)
        new_heads = len(keep)
        new_kv = len(kv_keep)
        _set_attention_attrs(a.module, new_heads, new_kv,
                             groups if new_kv else 1)
        removed_q = (a.num_heads or 0) - new_heads
        removed_kv = (a.num_kv_heads or 0) - new_kv
        a.num_heads, a.num_kv_heads = new_heads, new_kv
        a.num_key_value_groups = groups if new_kv else 1
        if a.q_proj.out_features != new_heads * hd:
            raise RewriteError(f"attention '{path}': query resize failed")
        result.records.append(PruneRecord(
            path=path, kind="attention_heads", layer=path, removed=removed_q,
            kept=new_heads, detail={"kv_removed": removed_kv, "head_dim": hd,
                                    "num_key_value_groups": a.num_key_value_groups},
        ))
        result.removed_heads += removed_q
        result.removed_kv_heads += removed_kv

    result.parameters_after = sum(p.numel() for p in model.parameters())
    return result


def _set_attention_attrs(module: nn.Module, heads: int, kv_heads: int,
                         groups: int) -> None:
    for name, value in (("num_heads", heads), ("num_key_value_heads", kv_heads),
                        ("num_kv_heads", kv_heads), ("num_key_value_groups", groups)):
        if hasattr(module, name):
            try:
                setattr(module, name, value)
            except Exception:  # pragma: no cover - defensive
                pass


# --------------------------------------------------------------------------- #
# layers
# --------------------------------------------------------------------------- #
def _provided(block: nn.Module) -> set[str]:
    """Parameter/buffer paths a block contributes, relative to itself."""
    return ({name for name, _ in block.named_parameters()} |
            {name for name, _ in block.named_buffers()})


def unsafe_layer_indices(arch: ModelArchitecture, stack) -> set[int]:
    """Indices of blocks that must not be deleted.

    Some blocks own a parameter that every other block relies on without holding
    a reference to it (T5's first decoder block carries the relative-attention
    bias used by the whole decoder stack). Removing such a block leaves a model
    whose config can no longer rebuild the surviving structure, so the manifest
    could not be filled on reload. Detected structurally, not per architecture.
    """
    module = getattr(stack, "module", None)
    if module is None:
        return set()
    provided = [_provided(b) for b in module]
    unsafe: set[int] = set()
    for i in range(len(provided)):
        survivors: set[str] = set()
        for j, p in enumerate(provided):
            if j != i:
                survivors |= p
        if not provided[i] <= survivors:
            unsafe.add(i)
    return unsafe


def delete_layers(model: nn.Module, arch: ModelArchitecture, stack_name: str,
                  indices: Iterable[int]) -> PruneResult:
    """Remove whole blocks from a layer stack and keep the config consistent."""
    stack = next((s for s in arch.stacks if s.name == stack_name), None)
    if stack is None:
        raise RewriteError(f"unknown stack '{stack_name}'")
    if arch.cap_prune_layers.level != SupportLevel.SUPPORTED:
        raise RewriteError(
            f"layer deletion is not supported: {arch.cap_prune_layers.reason}"
        )
    drop = sorted({int(i) for i in indices})
    n = len(stack.module)  # type: ignore[arg-type]
    if any(i < 0 or i >= n for i in drop):
        raise RewriteError(f"layer index out of range for '{stack_name}'")
    if len(drop) >= n:
        raise RewriteError("refusing to delete every layer of a stack")

    # A layer record that is not positionally addressable cannot be renumbered
    # after the ModuleList shifts, and a stale record produces a manifest that
    # references modules which no longer exist (an unloadable checkpoint).
    # Verify this *before* mutating anything and refuse instead of guessing.
    misaddressed = [L.name for L in arch.layers
                    if L.stack == stack_name
                    and (L.index is None or L.name != f"{stack_name}.{L.index}")]
    if misaddressed:
        raise RewriteError(
            "layer deletion is not supported for this stack: layer records are "
            f"not positionally addressable (first: {misaddressed[0]}); refusing "
            "to produce a checkpoint whose structure cannot be described "
            "unambiguously")

    # Detect, before any mutation, a stack whose depth cannot be changed
    # independently: if one of this stack's depth keys also describes a sibling
    # stack (e.g. a shared depth attribute), shrinking this stack would silently
    # resize that sibling, so the config would disagree with the module list and
    # the reloaded skeleton could not be filled from the manifest.
    own_keys = set(stack.config_keys) | set(stack.list_config_keys)
    shared: list[tuple[str, str]] = []
    for other in arch.stacks:
        if other.name == stack.name:
            continue
        overlap = own_keys & (set(other.config_keys) | set(other.list_config_keys))
        if overlap:
            shared.append((other.name, sorted(overlap)[0]))
    if shared:
        other_name, key = shared[0]
        raise RewriteError(
            f"layer deletion is not supported for stack '{stack_name}': its "
            f"depth is described by config key '{key}', which also describes "
            f"stack '{other_name}'; shrinking this stack would change that "
            "stack's depth, so a saved checkpoint could not be rebuilt")

    # A block may own a parameter that the remaining blocks depend on but do not
    # hold a reference to. Deleting the owner would leave a structure the config
    # cannot rebuild, so refuse before mutating anything.
    kept_preview = [b for i, b in enumerate(stack.module)  # type: ignore[union-attr]
                    if i not in set(drop)]
    survived: set[str] = set()
    for b in kept_preview:
        survived |= _provided(b)
    lost: set[str] = set()
    for i in drop:
        lost |= _provided(stack.module[i])  # type: ignore[index]
    missing = sorted(lost - survived)
    if missing:
        raise RewriteError(
            f"layer deletion is not supported for stack '{stack_name}': block(s) "
            f"{drop} contribute parameter(s) that no surviving block provides "
            f"(first: {missing[0]}); deleting them would leave a structure the "
            "config cannot rebuild")

    refs: dict[int, int] = {}
    for b in stack.module:  # type: ignore[union-attr]
        refs[id(b)] = refs.get(id(b), 0) + 1
    for i in drop:
        if refs[id(stack.module[i])] > 1:  # type: ignore[index]
            raise RewriteError(
                f"layer {stack_name}.{i} is a shared module instance; "
                "deleting it would corrupt other references"
            )

    keep_blocks = [b for i, b in enumerate(stack.module)  # type: ignore[union-attr]
                   if i not in set(drop)]
    stack.module._modules = {str(i): b for i, b in enumerate(keep_blocks)}  # type: ignore[union-attr]

    # re-index bookkeeping attributes and config
    for new_i, block in enumerate(keep_blocks):
        for mod in block.modules():
            if hasattr(mod, "layer_idx") and isinstance(
                getattr(mod, "layer_idx"), int
            ):
                setattr(mod, "layer_idx", new_i)
    sync_config(arch, stack, new_len=len(keep_blocks), old_len=n)

    # The config must now describe exactly the surviving structure for *every*
    # stack. If it does not, a reload builds a skeleton with the wrong number of
    # blocks and the manifest cannot be filled -- refuse rather than write a
    # checkpoint that cannot be loaded.
    from ..model.introspect import _ConfigView

    cfg_view = _ConfigView(arch.config)
    for s2 in arch.stacks:
        depth = len(s2.module)
        for key in s2.config_keys:
            value = cfg_view.get(key)
            if isinstance(value, int) and value != depth:
                raise RewriteError(
                    f"config key '{key}' now describes {value} blocks for stack "
                    f"'{s2.name}' but that stack has {depth}; refusing to "
                    "produce a checkpoint whose config disagrees with its modules")

    # The architecture snapshot must not keep describing deleted blocks, or a
    # later manifest/report would reference modules that no longer exist. The
    # surviving blocks also shift position inside the ModuleList, so their
    # recorded dotted paths (layer, MLP and attention projections, norms) have
    # to be renumbered -- a reloaded skeleton is built from the *new* config and
    # is matched against the manifest by path.
    remap = {old_i: new_i for new_i, old_i in
             enumerate(i for i in range(n) if i not in set(drop))}

    def _set_name(obj, value: str) -> None:
        params = getattr(type(obj), "__dataclass_params__", None)
        if params is not None and getattr(params, "frozen", False):
            object.__setattr__(obj, "name", value)
        else:
            obj.name = value

    def _rename(obj, old_pref: str, new_pref: str) -> None:
        nm = getattr(obj, "name", None)
        if isinstance(nm, str) and nm.startswith(old_pref):
            _set_name(obj, new_pref + nm[len(old_pref):])

    def _rename_proj(p, old_pref: str, new_pref: str) -> None:
        _rename(p, old_pref, new_pref)

    def _in_stack(L) -> bool:
        nm = getattr(L, "name", None)
        return isinstance(nm, str) and nm.startswith(stack_name + ".")

    surviving = []
    for L in arch.layers:
        if _in_stack(L) and L.index in drop:
            continue                      # physically removed: drop the record
        if _in_stack(L) and L.index in remap:
            old_pref = f"{stack_name}.{L.index}"
            new_pref = f"{stack_name}.{remap[L.index]}"
            L.index = remap[L.index]
            _rename(L, old_pref, new_pref)
            m, a_ = L.mlp, L.attention
            if m is not None:
                _rename(m, old_pref, new_pref)
                for proj in (m.gate_proj, m.up_proj, m.down_proj):
                    _rename_proj(proj, old_pref, new_pref)
            if a_ is not None:
                _rename(a_, old_pref, new_pref)
                for proj in (a_.q_proj, a_.k_proj, a_.v_proj, a_.o_proj):
                    _rename_proj(proj, old_pref, new_pref)
            for extra in list(L.attentions) + list(L.norms):
                _rename(extra, old_pref, new_pref)
        surviving.append(L)
    arch.layers = surviving

    stack.num_blocks = len(keep_blocks)
    arch.num_layers = sum(len(s2.module) for s2 in arch.stacks)  # type: ignore[arg-type]
    arch.total_parameters = sum(p.numel() for p in model.parameters())
    arch.trainable_parameters = sum(p.numel() for p in model.parameters()
                                    if p.requires_grad)

    result = PruneResult()
    result.removed_layers = len(drop)
    for i in drop:
        result.records.append(PruneRecord(
            path=f"{stack_name}.{i}", kind="layer_delete", layer=f"{stack_name}.{i}",
            removed=1, kept=len(keep_blocks)))
    return result


def sync_config(arch: ModelArchitecture, stack, new_len: int,
                old_len: Optional[int] = None) -> None:
    """Update config attributes so ``from_pretrained`` builds the right skeleton."""
    from ..model.introspect import _ConfigView

    old_len = int(old_len if old_len is not None else stack.num_blocks)
    cfg = _ConfigView(arch.config)
    touched = 0
    for key in stack.config_keys:
        cfg.set(key, new_len)
        touched += 1
    for key in stack.list_config_keys:
        value = cfg.get(key)
        if isinstance(value, (list, tuple)) and len(value) > new_len:
            cfg.set(key, list(value[:new_len]))

    # Depth keys that describe a *different* stack must keep their value: deleting
    # a layer from one stack while a base key silently resizes a sibling rebuilds a
    # skeleton of the wrong shape, and the saved manifest then references blocks
    # that no longer exist.
    owned = set(stack.config_keys) | set(stack.list_config_keys)
    foreign: dict[str, int] = {}
    for other in getattr(arch, "stacks", None) or []:
        if other.name == stack.name:
            continue
        for key in list(other.config_keys) + list(other.list_config_keys):
            if key not in owned:
                foreign[key] = other.num_blocks

    # Robust fallback: any config leaf that still claims the old depth and looks
    # like a layer count must be updated, otherwise a saved checkpoint rebuilds
    # the wrong number of blocks (this is what broke reload).
    layer_ish = ("layer", "depth", "n_layer", "num_hidden", "num_decoder",
                 "num_encoder", "num_layers")
    try:
        leaves = list(cfg.iter_values())
    except Exception:  # pragma: no cover - defensive
        leaves = []
    if not leaves:
        for name in dir(arch.config):
            if name.startswith("_"):
                continue
            try:
                value = getattr(arch.config, name)
            except Exception:
                continue
            leaves.append((name, value))
    for dotted, value in leaves:
        leaf = dotted.rsplit(".", 1)[-1]
        if not any(t in leaf for t in layer_ish):
            continue
        if isinstance(value, bool):
            continue
        if dotted in foreign:
            continue
        if isinstance(value, int) and value == old_len and value != new_len:
            cfg.set(dotted, new_len)
            touched += 1
        elif isinstance(value, (list, tuple)) and len(value) == old_len \
                and old_len != new_len:
            cfg.set(dotted, list(value[:new_len]))
            touched += 1
    drifted = [key for key, depth in foreign.items()
               if isinstance(cfg.get(key), int) and cfg.get(key) != depth]
    if drifted:
        for key in drifted:
            cfg.set(key, foreign[key])
        raise RuntimeError(
            f"refusing to resize '{stack.name}': config key(s) {drifted} "
            "describe the depth of another stack and would rebuild a skeleton "
            "of the wrong shape")

    if touched == 0:
        raise RuntimeError(
            f"could not find a config attribute describing the depth of "
            f"'{stack.name}' ({old_len} -> {new_len}); refusing to save a "
            "checkpoint whose config disagrees with its module list")


# --------------------------------------------------------------------------- #
# equivalence guard
# --------------------------------------------------------------------------- #
def mlp_reference_guard(arch: ModelArchitecture,
                        keep: dict[str, Sequence[int]]):
    """Context manager zeroing the *removed* FFN channels in place.

    Restores the original weights on exit. Used to compute the reference output
    of the masked (not structurally pruned) model.
    """
    by_name = {L.mlp.name: L.mlp for L in arch.layers if L.mlp is not None}
    saved = []

    def save(param: nn.Parameter, tensor: torch.Tensor) -> None:
        saved.append((param, tensor))

    for path, keep_idx in keep.items():
        mlp = by_name[path]
        inter = mlp.intermediate_size
        drop_q = sorted(set(range(inter)) - set(int(i) for i in keep_idx))
        if not drop_q:
            continue
        rows = torch.tensor(drop_q, dtype=torch.long)
        fused = _fused_layout(mlp)
        if fused is not None:
            gate_off, up_off = fused
            w = mlp.up_proj.weight_matrix.clone()
            w.index_fill_(0, rows + gate_off, 0.0)
            w.index_fill_(0, rows + up_off, 0.0)
            saved.append((mlp.up_proj.module.weight, mlp.up_proj.module.weight.data.clone()))
            mlp.up_proj._set_weight_matrix(w, mlp.up_proj.weight.requires_grad)
        else:
            for proj in (mlp.gate_proj, mlp.up_proj):
                if proj is None:
                    continue
                w = proj.weight_matrix.clone()
                w.index_fill_(0, rows, 0.0)
                saved.append((proj.module.weight, proj.module.weight.data.clone()))
                proj._set_weight_matrix(w, proj.weight.requires_grad)
        if mlp.down_proj.has_bias:
            pass
        w = mlp.down_proj.weight_matrix.clone()
        w.index_fill_(1, rows, 0.0)
        saved.append((mlp.down_proj.module.weight,
                      mlp.down_proj.module.weight.data.clone()))
        mlp.down_proj._set_weight_matrix(w, mlp.down_proj.weight.requires_grad)
    return _RestoreGuard(saved)


class _RestoreGuard:
    def __init__(self, saved):
        self.saved = saved

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for param, data in self.saved:
            with torch.no_grad():
                param.data.copy_(data)


@torch.no_grad()
def reference_logits(model: nn.Module, arch: ModelArchitecture,
                     keep: dict[str, Sequence[int]], batch: dict) -> torch.Tensor:
    """Reference output of the ORIGINAL model with removed channels zeroed.

    Must be called *before* the structural rewrite (the masked channels still
    exist at that point).
    """
    with mlp_reference_guard(arch, keep):
        ref = model(**batch, use_cache=False)
    logits = ref.logits if hasattr(ref, "logits") else ref.last_hidden_state
    return logits.detach().float()


@torch.no_grad()
def compare_to_reference(model: nn.Module, reference: torch.Tensor, batch: dict,
                         atol: float = 1e-4, rtol: float = 1e-3) -> dict:
    out = model(**batch, use_cache=False)
    logits = out.logits if hasattr(out, "logits") else out.last_hidden_state
    logits = logits.detach().float()
    if logits.shape != reference.shape:
        return {"ok": False, "max_abs_diff": float("inf"),
                "reason": f"shape mismatch {tuple(logits.shape)} vs {tuple(reference.shape)}",
                "atol": atol, "rtol": rtol}
    diff = (logits - reference).abs().max().item()
    ok = bool(torch.allclose(logits, reference, atol=atol, rtol=rtol))
    return {"max_abs_diff": diff, "ok": ok, "atol": atol, "rtol": rtol}


@torch.no_grad()
def verify_pruned_equivalence(model: nn.Module, arch: ModelArchitecture,
                              keep: dict[str, Sequence[int]], batch: dict,
                              atol: float = 1e-4, rtol: float = 1e-3) -> dict:
    """Deprecated one-shot helper; only valid on an un-pruned model."""
    ref = reference_logits(model, arch, keep, batch)
    return compare_to_reference(model, ref, batch, atol=atol, rtol=rtol)
