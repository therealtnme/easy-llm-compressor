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


def build_manifest(arch, model, extra: Optional[dict] = None) -> dict:
    mlp = {}
    attn = {}
    for L in arch.layers:
        m = L.mlp
        if m is not None and m.intermediate_size:
            mlp[m.name] = {
                "width": int(m.intermediate_size),
                "gate": _proj_path(m.gate_proj), "up": _proj_path(m.up_proj),
                "down": _proj_path(m.down_proj), "fused": bool(m.gate_up_fused),
            }
        for a in (L.attentions or ([L.attention] if L.attention else [])):
            if not a.num_heads:
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
        "num_layers": arch.num_layers,
        "parameters": arch.total_parameters,
    }
    manifest.update(extra or {})
    return manifest


def save_compressed(model, arch, out_dir: str, tokenizer=None,
                    extra: Optional[dict] = None) -> dict:
    from safetensors.torch import save_model

    os.makedirs(out_dir, exist_ok=True)
    config = getattr(model, "config", None)
    if config is None:
        raise RuntimeError("model has no config; cannot save a loadable checkpoint")
    config.save_pretrained(out_dir)
    save_model(model, os.path.join(out_dir, "model.safetensors"),
               metadata={"format": "pt"})
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    manifest = build_manifest(arch, model, extra) if arch is not None else {
        "format": "llm-compressor/v1", "model_class": type(model).__name__,
        "mlp": {}, "attention": {}, **(extra or {})}
    with open(os.path.join(out_dir, MANIFEST), "w", encoding="utf8") as fh:
        json.dump(manifest, fh, indent=2)
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
