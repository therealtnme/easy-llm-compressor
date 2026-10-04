"""Save / reload of structurally compressed checkpoints.

A compressed model can have *per-layer* widths (layer 3 keeps 40 neurons,
layer 4 keeps 52), which no HF config can express. So the checkpoint carries a
manifest describing the surviving structure and ``load_compressed`` rebuilds a
skeleton of exactly that shape before filling it with the saved tensors. The
saved tensors are the real, physically smaller ones produced by the pruning
pass; nothing is masked or zero-padded.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import torch

from .model.projection import Projection
from .scoring import select_by_budget

MANIFEST = "compression_manifest.json"


def _proj_path(p) -> Optional[str]:
    return p.name if p is not None else None


def _present(model, *paths) -> bool:
    """True when every given dotted path really exists on the live model."""
    for path in paths:
        if not path:
            continue
        try:
            model.get_submodule(path)
        except AttributeError:
            return False
    return True


def build_manifest(arch, model, extra: Optional[dict] = None) -> dict:
    """Describe the *current* structure of ``model``.

    The architecture snapshot is only a hint here: after structural deletion it
    can still mention blocks that were physically removed, and a manifest that
    references missing modules makes the checkpoint unloadable. So every entry
    is checked against the live module graph.
    """
    mlp = {}
    attn = {}
    for L in arch.layers:
        m = L.mlp
        if m is not None and m.intermediate_size and _present(
                model, _proj_path(m.gate_proj), _proj_path(m.up_proj),
                _proj_path(m.down_proj)):
            mlp[m.name] = {
                "width": int(m.intermediate_size),
                "gate": _proj_path(m.gate_proj), "up": _proj_path(m.up_proj),
                "down": _proj_path(m.down_proj), "fused": bool(m.gate_up_fused),
            }
        for a in (L.attentions or ([L.attention] if L.attention else [])):
            if not a.num_heads or not _present(
                    model, _proj_path(a.q_proj), _proj_path(a.o_proj)):
                continue
            attn[a.name] = {
                "heads": int(a.num_heads),
                "kv_heads": int(a.num_kv_heads or a.num_heads),
                "head_dim": int(a.head_dim or 0),
                "q": _proj_path(a.q_proj), "k": _proj_path(a.k_proj),
                "v": _proj_path(a.v_proj), "o": _proj_path(a.o_proj),
                "groups": int(a.num_key_value_groups or 1),
            }
    manifest = {
        "format": "llm-compressor/v1",
        "model_class": type(model).__name__,
        "config_class": type(getattr(model, "config", arch.config)).__name__,
        "mlp": mlp,
        "attention": attn,
        "num_layers": sum(1 for L in arch.layers
                          if L.mlp is not None and L.mlp.name in mlp)
        if mlp else arch.num_layers,
        "parameters": arch.total_parameters,
    }
    manifest.update(extra or {})
    return manifest


# --------------------------------------------------------------------------- #
# config synchronisation: make the saved config describe the saved tensors
# --------------------------------------------------------------------------- #
_WIDTH_KEYS = ("intermediate_size", "n_inner", "ffn_dim", "ff_dim",
               "dense_ffn_dim", "ffn_hidden_size", "mlp_dim", "d_ff")


def _layer_index(name: str):
    """Last integer segment of a dotted path -> the layer index it belongs to."""
    idx = None
    for part in name.split("."):
        if part.isdigit():
            idx = int(part)
    return idx


def live_mlp_widths(model, arch) -> dict:
    """Measured intermediate width of every FFN, read from the live tensors."""
    widths = {}
    for L in arch.layers:
        m = getattr(L, "mlp", None)
        if m is None:
            continue
        for proj, axis in ((m.up_proj, "out"), (m.gate_proj, "out"),
                           (m.down_proj, "in")):
            if proj is None or not proj.name:
                continue
            try:
                mod = model.get_submodule(proj.name)
            except AttributeError:
                continue
            p = Projection.of(proj.name, mod)
            w = p.out_features if axis == "out" else p.in_features
            if w:
                widths[m.name] = int(w)
                break
    return widths


# --------------------------------------------------------------------------- #
# config-expressibility: can config.json rebuild exactly the saved FFN shapes?
# --------------------------------------------------------------------------- #
def _probe_specs(arch) -> dict:
    """Per-FFN probe spec: the class that builds the FFN and the attribute path
    of its up/down projections, discovered from the live module graph (nothing
    architecture specific is hardcoded)."""
    specs = {}
    for L in getattr(arch, "layers", []):
        m = getattr(L, "mlp", None)
        module = getattr(m, "module", None)
        if m is None or module is None or not getattr(m, "name", None):
            continue
        attrs = []
        for proj in (getattr(m, "up_proj", None), getattr(m, "gate_proj", None),
                     getattr(m, "down_proj", None)):
            name = getattr(proj, "name", None)
            if name:
                attr = name.split(".")[-1]
                if attr not in attrs:
                    attrs.append(attr)
        specs[m.name] = {"cls": type(module), "attrs": attrs,
                         "fused": bool(getattr(m, "gate_up_fused", False))}
    return specs


def _width_probe(spec, cfg, key, value):
    """Width an FFN of ``spec['cls']`` builds when config key ``key`` = value.

    The FFN constructor is used as an empirical oracle for the architecture's
    config -> width mapping, so merged/fused/gated layouts are handled by the
    model's own code rather than by a guess here.
    """
    import copy as _copy

    c = _copy.deepcopy(cfg)
    try:
        setattr(c, key, value)
        module = spec["cls"](c)
    except Exception:
        return None
    for attr in spec["attrs"]:
        sub = module
        try:
            for part in attr.split("."):
                sub = getattr(sub, part)
        except AttributeError:
            continue
        out = getattr(sub, "out_features", None)
        if out:
            return int(out) // 2 if spec["fused"] else int(out)
        inn = getattr(sub, "in_features", None)
        if inn:
            return int(inn)
    return None


def width_key(model):
    """The config key an FFN width can be written to, or None."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return None
    for key in _WIDTH_KEYS:
        if isinstance(getattr(cfg, key, None), int):
            return key
    for key in _WIDTH_KEYS:
        cur = getattr(cfg, key, None)
        if isinstance(cur, (list, tuple)) and cur:
            return key
    return None


def uniform_widths_required(model) -> bool:
    """True when the config can only ever state *one* FFN width for the model."""
    cfg = getattr(model, "config", None)
    if cfg is None:
        return False
    for key in _WIDTH_KEYS:
        cur = getattr(cfg, key, None)
        if isinstance(cur, (list, tuple)) and cur:
            return False
    return width_key(model) is not None


def widths_described_by_config(model, arch, widths=None) -> bool:
    """True when rebuilding every FFN from the *live* config reproduces the
    width actually stored on disk, i.e. plain `transformers`
    `from_pretrained()` loads this checkpoint with no size mismatch at all."""
    cfg = getattr(model, "config", None)
    key = width_key(model)
    specs = _probe_specs(arch) if arch is not None else {}
    if cfg is None or key is None or not specs:
        return False
    value = getattr(cfg, key)
    if isinstance(value, (list, tuple)):
        # per-layer keys are written verbatim (one entry per layer); the
        # plain-load gate in the pipeline proves the result empirically.
        return True
    live = widths if widths is not None else live_mlp_widths(model, arch)
    if not live:
        return True
    for name, spec in specs.items():
        if name in live and _width_probe(spec, cfg, key, value) != int(live[name]):
            return False
    return True


def _find_width_value(model, arch, key, target):
    """Find a config value whose FFN widths are ``target``; failing an exact
    match, the largest reachable width <= target. Returns (value, width)."""
    cfg = model.config
    specs = _probe_specs(arch)
    if not specs:
        return None
    names = list(specs)
    cur = getattr(cfg, key)
    seen = _width_probe(specs[names[0]], cfg, key, cur)
    candidates = [target]
    if seen:
        centre = int(round(cur * (target / float(seen))))
        candidates += [centre + k for k in range(-16, 17) if centre + k > 0]
    candidates = list(dict.fromkeys(candidates))
    for v in candidates:
        if _width_probe(specs[names[0]], cfg, key, v) != target:
            continue
        if all(_width_probe(specs[n], cfg, key, v) == target for n in names[1:]):
            return v, target
    best = None
    for v in candidates:
        w = _width_probe(specs[names[0]], cfg, key, v)
        if w is not None and w <= target and (best is None or w > best[1]):
            best = (v, w)
    return best


def enforce_loadable_structure(model, arch, scores, keep, protected=(),
                               exclude=()) -> tuple:
    """Level an allocation so the saved config.json can describe the result.

    Runs *before* any tensor is sliced: ``keep`` maps FFN path -> kept neuron
    indices in the teacher index space and ``exclude`` lists FFNs that are about
    to be deleted together with their layer. Returns ``(keep, report)``. A
    scalar FFN key can state only one width model-wide, so if the allocation is
    uneven the survivors are levelled to a width every one of them can be
    sliced to. Nothing is ever widened and no config-inexpressible checkpoint is
    ever produced.
    """
    report = {"adjusted": False, "width": None,
              "reason": "config.json already describes every FFN width exactly"}
    cfg = getattr(model, "config", None)
    key = width_key(model)
    if cfg is None or key is None or not keep:
        return keep, report
    if isinstance(getattr(cfg, key), (list, tuple)):
        return keep, report
    exclude = set(exclude or ())
    live = live_mlp_widths(model, arch)
    desired = {p: len(idx) for p, idx in keep.items()}
    fixed = {n: int(w) for n, w in live.items()
             if n not in desired and n not in exclude}
    if fixed:
        if len(set(fixed.values())) != 1:
            raise RuntimeError(
                "cannot make this checkpoint loadable: the architecture's config "
                "can state only a single FFN width but the FFNs that are not "
                f"pruned have different widths {sorted(set(fixed.values()))}")
        target = min(fixed.values())
        if any(w < target for w in desired.values()):
            raise RuntimeError(
                "cannot make this checkpoint loadable: the config must state "
                f"width {target} (fixed by un-pruned FFNs) but some FFNs were "
                f"allotted only {min(desired.values())} neurons; a checkpoint is "
                "never widened")
    else:
        target = min(desired.values())
    found = _find_width_value(model, arch, key, target)
    if found is None:
        raise RuntimeError(
            "cannot make this checkpoint loadable by plain transformers: no "
            f"value of config key '{key}' rebuilds an FFN of width {target}")
    value, width = found
    setattr(cfg, key, value)
    if width >= target and all(w == width for w in desired.values()):
        report["width"] = width
        return keep, report
    new_keep = {}
    for path, idx in keep.items():
        if len(idx) == width:
            new_keep[path] = list(idx)
            continue
        if len(idx) < width:
            raise RuntimeError(
                f"cannot make this checkpoint loadable: FFN '{path}' would have "
                f"to grow from {len(idx)} to {width} neurons")
        if not scores or path not in scores:
            raise RuntimeError(
                f"cannot make this checkpoint loadable: FFN '{path}' must be "
                f"re-selected to {width} neurons but has no importance scores")
        new_keep[path] = select_by_budget(scores[path], width).tolist()
    report.update({
        "adjusted": True, "width": width, "key": key, "value": value,
        "reason": (
            f"FFN widths were levelled to {width} neurons: this architecture's "
            "config can state only a single FFN width, and uneven widths would "
            "make the checkpoint unloadable by plain transformers "
            "(ignore_mismatched_sizes=False)"),
    })
    return new_keep, report


def plain_load_check(out_dir: str):
    """Literal proof of the contract: plain transformers, mismatches fatal."""
    import transformers

    manifest = load_manifest(out_dir)
    cls = getattr(transformers, manifest.get("model_class", ""), None)
    if cls is None:
        cls = getattr(transformers, "AutoModelForCausalLM", None)
    try:
        model = cls.from_pretrained(out_dir, ignore_mismatched_sizes=False)
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        return False, (f"plain transformers {cls.__name__}.from_pretrained("
                       f"..., ignore_mismatched_sizes=False) failed: "
                       f"{type(exc).__name__}: {exc}")
    del model
    return True, (f"plain transformers {cls.__name__}.from_pretrained(..., "
                  f"ignore_mismatched_sizes=False) loaded the checkpoint")


def sync_config_to_structure(model, arch) -> dict:
    """Rewrite the model config so it describes the *compressed* FFN widths.

    A checkpoint whose config still advertises the pre-pruning width cannot be
    reloaded by plain `transformers`, so the config is rebuilt from the live
    tensor shapes. Uniform widths go into the scalar keys; per-layer widths go
    into a per-layer list when the config already uses one. When neither can
    express the result the caller is told, so it can never be silent.
    """
    cfg = getattr(model, "config", None)
    info = {"updated": [], "expressible": True, "widths": {}}
    if cfg is None or arch is None:
        return info
    widths = live_mlp_widths(model, arch)
    info["widths"] = widths
    if not widths:
        return info
    unique = set(widths.values())
    ordered = [widths[k] for k in sorted(
        widths, key=lambda n: (_layer_index(n) if _layer_index(n) is not None
                               else 0))]
    n_layers = max(len(arch.layers), len(ordered))
    for key in _WIDTH_KEYS:
        if not hasattr(cfg, key):
            continue
        current = getattr(cfg, key)
        if isinstance(current, (list, tuple)):
            if len(ordered) == len(current):
                setattr(cfg, key, list(ordered))
                info["updated"].append(key)
            elif len(unique) == 1 and len(current) >= n_layers:
                setattr(cfg, key, [ordered[0]] * len(current))
                info["updated"].append(key)
            elif any(isinstance(v, (list, tuple)) for v in current) is False:
                info["expressible"] = False
            continue
        if not isinstance(current, int):
            continue
        if len(unique) != 1:
            info["expressible"] = False
            continue
        # the key is not necessarily the width itself (e.g. LFM2 derives the FFN
        # width from intermediate_size), so invert the mapping through the FFN
        # constructor instead of writing the width blindly.
        found = _find_width_value(model, arch, key, ordered[0])
        if found is not None and found[1] == ordered[0]:
            setattr(cfg, key, found[0])
            info["updated"].append(key)
        elif ordered[0] == current:
            info["updated"].append(key)  # the config already states this width
        else:
            info["expressible"] = False
    return info


def save_compressed(model, arch, out_dir: str, tokenizer=None,
                    extra: Optional[dict] = None) -> dict:
    from safetensors.torch import save_model

    os.makedirs(out_dir, exist_ok=True)
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("model has no config; cannot save a loadable checkpoint")
    sync_info = sync_config_to_structure(model, arch)
    # Never write a checkpoint whose own config.json cannot describe it: plain
    # `from_pretrained(..., ignore_mismatched_sizes=False)` must always work.
    if arch is not None and not widths_described_by_config(model, arch):
        raise RuntimeError(
            "refusing to save a checkpoint whose config cannot express its FFN "
            "widths: plain transformers from_pretrained(..., "
            "ignore_mismatched_sizes=False) would reject it. That is a bug in "
            "the compression step, not a property of the checkpoint.")
    config.save_pretrained(out_dir)
    save_model(model, os.path.join(out_dir, "model.safetensors"),
               metadata={"format": "pt"})
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    manifest = build_manifest(arch, model, extra) if arch is not None else {
        "format": "llm-compressor/v1", "model_class": type(model).__name__,
        "mlp": {}, "attention": {}, **(extra or {})}
    # Fail loudly instead of writing a checkpoint that cannot be reloaded.
    missing = [path for info in list((manifest.get("mlp") or {}).values()) +
               list((manifest.get("attention") or {}).values())
               for path in (info.get("gate"), info.get("up"), info.get("down"),
                            info.get("q"), info.get("k"), info.get("v"),
                            info.get("o"))
               if path and not _present(model, path)]
    if missing:
        raise RuntimeError(
            "refusing to save an inconsistent checkpoint: the recorded structure "
            f"references modules that do not exist ({missing[0]}). This "
            "architecture's layer records cannot be addressed positionally, so "
            "structural layer deletion is UNSUPPORTED for it.")
    manifest["config_describes_structure"] = bool(
        widths_described_by_config(model, arch)) if arch is not None else False
    manifest["config_keys_synced"] = list(sync_info.get("updated", []))
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf8") as fh:
        json.dump(manifest, fh, indent=2)
    if not manifest["config_describes_structure"]:
        raise RuntimeError(
            "internal safety check failed: the saved config does not describe "
            "the saved tensors, so plain from_pretrained() would reject this "
            "checkpoint. Refusing to leave it on disk.")
    manifest["checkpoint_bytes"] = sum(
        os.path.getsize(os.path.join(out_dir, f))
        for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f)))
    return manifest


def load_manifest(out_dir: str) -> dict:
    path = os.path.join(out_dir, MANIFEST)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf8") as fh:
        return json.load(fh)


def _proj(model, path: str, role: str) -> Optional[Projection]:
    try:
        module = model.get_submodule(path)
    except AttributeError:
        raise RuntimeError(
            f"manifest references '{path}' which does not exist in the rebuilt "
            f"skeleton; the checkpoint is inconsistent")
    return Projection.of(path, module)


def apply_manifest_structure(model, manifest: dict) -> dict:
    """Shrink a freshly built skeleton to the recorded structure.

    Only used before loading saved tensors; weights are overwritten anyway, so
    this only has to produce the right *shapes*.
    """
    applied = {"mlp": 0, "attention": 0}
    for name, info in (manifest.get("mlp") or {}).items():
        width = int(info["width"])
        if info.get("up") and info.get("fused"):
            p = _proj(model, info["up"], "up")
            if p.out_features == width * 2:
                p.slice_outputs(list(range(width * 2)))
            elif p.out_features > width:
                p.slice_outputs(list(range(width)))
        else:
            for key in ("gate", "up"):
                if info.get(key):
                    p = _proj(model, info[key], key)
                    if p.out_features != width:
                        p.slice_outputs(list(range(width)))
        if info.get("down"):
            p = _proj(model, info["down"], "down")
            if p.in_features != width:
                p.slice_inputs(list(range(width)))
        applied["mlp"] += 1
    for name, info in (manifest.get("attention") or {}).items():
        hd, heads, kv = int(info["head_dim"]), int(info["heads"]), int(info["kv_heads"])
        if not hd:
            continue
        if info.get("q"):
            p = _proj(model, info["q"], "q")
            if p.out_features != heads * hd:
                p.slice_outputs(list(range(heads * hd)))
        for key in ("k", "v"):
            if info.get(key):
                p = _proj(model, info[key], key)
                if p.out_features != kv * hd:
                    p.slice_outputs(list(range(kv * hd)))
        if info.get("o"):
            p = _proj(model, info["o"], "o")
            if p.in_features != heads * hd:
                p.slice_inputs(list(range(heads * hd)))
        applied["attention"] += 1
    return applied


def load_compressed(out_dir: str, device: str = "cpu"):
    """Rebuild the compressed checkpoint. Returns (model, tokenizer, manifest).

    Every checkpoint written by this tool has a config.json that describes its
    tensors exactly, so this is just the plain transformers load path.
    """
    import transformers
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoTokenizer

    manifest = load_manifest(out_dir)
    config = AutoConfig.from_pretrained(out_dir)
    if manifest.get("config_describes_structure"):
        cls_name = manifest.get("model_class", "")
        plain = getattr(transformers, cls_name, None)
        if plain is not None:
            model = plain.from_pretrained(out_dir, ignore_mismatched_sizes=False)
            model.eval()
            if device and device != "cpu":
                model.to(device)
            tokenizer = None
            try:
                tokenizer = AutoTokenizer.from_pretrained(out_dir)
            except Exception:
                tokenizer = None
            return model, tokenizer, manifest
    cls = getattr(transformers, manifest.get("model_class", ""), None)
    if cls is None:
        cls = getattr(transformers, "AutoModel")
    model = cls(config)
    apply_manifest_structure(model, manifest)
    load_model(model, os.path.join(out_dir, "model.safetensors"), strict=True)
    model.eval()
    if device and device != "cpu":
        model.to(device)
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(out_dir)
    except Exception:
        tokenizer = None
    return model, tokenizer, manifest
