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
        if len(unique) == 1:
            setattr(cfg, key, ordered[0])
            info["updated"].append(key)
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
    manifest["config_describes_structure"] = bool(sync_info.get("expressible", True))
    manifest["config_keys_synced"] = list(sync_info.get("updated", []))
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf8") as fh:
        json.dump(manifest, fh, indent=2)
    if not manifest["config_describes_structure"]:
        print(
            "warning: this checkpoint has per-layer widths that its config cannot "
            "express. Load it with llm_compressor.checkpoint.load_compressed() or "
            "llm_compressor.load_model_for_inspection(); plain transformers "
            "from_pretrained() will reject it rather than silently mismatch.")
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
    """Rebuild the compressed checkpoint. Returns (model, tokenizer, manifest)."""
    import transformers
    from safetensors.torch import load_model
    from transformers import AutoConfig, AutoTokenizer

    manifest = load_manifest(out_dir)
    config = AutoConfig.from_pretrained(out_dir)
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
