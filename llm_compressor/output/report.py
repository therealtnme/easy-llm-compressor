"""Machine-readable and human-readable compression reports."""

from __future__ import annotations

import json
import os
from typing import Any

from ..utils.formatting import format_number, format_params


def build_report(arch, options: dict, *, initial: dict, structural: dict,
                 plan: dict, execution: dict, validation: dict,
                 dataset_info: dict, fusion: dict, distillation: dict,
                 extra: dict | None = None) -> dict:
    report = {
        "tool": "llm-compressor",
        "format": "llm-compressor/report/v1",
        "initial_model_report": initial,
        "options": options,
        "dataset": dataset_info,
        "search": plan,
        "execution": execution,
        "structural_validation": structural,
        "fusion": fusion,
        "distillation": distillation,
        "validation": validation,
        "protected": [{"name": p.name, "category": p.category, "reason": p.reason}
                      for p in arch.protected],
        "capabilities": {
            "inspect": arch.cap_inspect.level.value,
            "prune_mlp_neurons": arch.cap_prune_mlp_neurons.level.value,
            "prune_attention_heads": arch.cap_prune_attention_heads.level.value,
            "prune_kv_heads": arch.cap_prune_kv_heads.level.value,
            "prune_layers": arch.cap_prune_layers.level.value,
            "fuse_affine": arch.cap_fuse_affine.level.value,
            "fuse_learned": arch.cap_fuse_learned.level.value,
        },
        "capability_reasons": {
            "prune_mlp_neurons": arch.cap_prune_mlp_neurons.reason,
            "prune_attention_heads": arch.cap_prune_attention_heads.reason,
            "prune_kv_heads": arch.cap_prune_kv_heads.reason,
            "prune_layers": arch.cap_prune_layers.reason,
            "fuse_affine": arch.cap_fuse_affine.reason,
            "fuse_learned": arch.cap_fuse_learned.reason,
        },
        "unsupported_operations": execution.get("unsupported", []),
        "unknown": execution.get("unknown", []),
        "final_model": arch.summarize(),
    }
    report.update(extra or {})
    return report


def write_report(report: dict, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "compression_report.json")
    txt_path = os.path.join(out_dir, "compression_report.txt")
    with open(json_path, "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2, default=str)
    with open(txt_path, "w", encoding="utf8") as fh:
        fh.write(render_text(report))
    return {"json": json_path, "text": txt_path}


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, int):
        return format_number(value)
    return str(value)


def render_text(report: dict) -> str:
    lines: list[str] = []
    add = lines.append
    init = report.get("initial_model_report", {})
    add("=" * 72)
    add("llm-compressor compression report")
    add("=" * 72)
    add("")
    add("[initial model]")
    for key in ("model_name", "model_class", "parameters", "trainable_parameters",
                "dtype", "device", "hidden_size", "layers", "mlp_intermediate_size",
                "num_attention_heads", "num_kv_heads", "head_dim", "vocab_size",
                "mlp_neuron_count", "attention_head_count", "kv_head_count"):
        if key in init:
            add(f"  {key:26s} {_fmt(init[key])}")
    add(f"  {'protected_components':26s} {len(report.get('protected', []))}")
    add("")

    add("[options]")
    for key, value in (report.get("options") or {}).items():
        add(f"  {key:26s} {_fmt(value)}")
    add("")

    data = report.get("dataset") or {}
    add("[calibration data]")
    add(f"  {data.get('mode', 'dataset-free')}")
    for key in ("source", "examples", "sequences", "seq_len", "packing",
                "cache", "cached", "tokens", "note"):
        if data.get(key) is not None:
            add(f"  {key:26s} {_fmt(data[key])}")
    add("")

    search = report.get("search") or {}
    add("[search]")
    add(f"  {'algorithm':26s} {search.get('algorithm', 'beam')}")
    add(f"  {'teacher_depth':26s} {_fmt(search.get('teacher_depth'))}")
    add(f"  {'target_student_layers':26s} "
        f"{_fmt(search.get('target_student_layers'))}")
    add(f"  {'candidates_considered':26s} {_fmt(search.get('candidates'))}")
    add(f"  {'layer_cost':26s} {_fmt(search.get('layer_cost'))}")
    add(f"  {'neuron_cost':26s} {_fmt(search.get('neuron_cost'))}")
    add(f"  {'total_cost':26s} {_fmt(search.get('total_cost'))}")
    add(f"  {'scoring_method':26s} {search.get('scoring_method', '')}")
    add(f"  {'saliency_method':26s} {search.get('saliency_method', '')}")
    for region in search.get("regions", []):
        add(f"    region {region['region_start']:>3}-{region['region_end']:<3} "
            f"{region['operation']:<14} student_depth={region['student_depth']}")
    add("")

    add("[structural changes]")
    for rec in (report.get("execution") or {}).get("records", []):
        add(f"  {rec.get('kind'):<16} {rec.get('path')} "
            f"removed={rec.get('removed')} kept={rec.get('kept')}")
    for note in (report.get("execution") or {}).get("notes", []):
        add(f"  note: {note}")
    add(f"  {'equivalence_check':26s} "
        f"{(report.get('execution') or {}).get('equivalence')}")
    add("")

    fusion = report.get("fusion") or {}
    add("[fusion]")
    add(f"  {'exact_affine_allowed':26s} {_fmt(fusion.get('exact_fusion_allowed'))}")
    add(f"  {'sequential_candidates':26s} "
        f"{_fmt(fusion.get('sequential_candidates'))}")
    add(f"  {'applied':26s} {_fmt(fusion.get('applied'))}")
    if fusion.get("reason"):
        add(f"  reason: {fusion['reason']}")
    for reason in fusion.get("exact_fusion_reasons", [])[:6]:
        add(f"    - {reason}")
    add("")

    dist = report.get("distillation") or {}
    add("[learned layer fusion / distillation]")
    if dist.get("used"):
        for key in ("steps", "losses", "trainable_parameters"): 
            add(f"  {key:26s} {_fmt(dist.get(key))}")
    else:
        add(f"  not used: {dist.get('reason', 'not requested')}")
    add("")

    add("[validation]")
    for key, value in (report.get("validation") or {}).items():
        if isinstance(value, dict):
            add(f"  {key}:")
            for k2, v2 in value.items():
                if k2 in ("history", "problem_list"):
                    continue
                add(f"    {k2:24s} {_fmt(v2)}")
        elif isinstance(value, list):
            continue
        else:
            add(f"  {key:26s} {_fmt(value)}")
    add("")

    struct = report.get("structural_validation") or {}
    add("[final structural counts]")
    for key in ("parameters_before", "parameters", "parameters_saved",
                "parameter_change_percent", "mlp_neurons", "attention_heads",
                "kv_heads", "layers", "checkpoint_bytes"):
        if struct.get(key) is not None:
            add(f"  {key:26s} {_fmt(struct[key])}")
    add(f"  {'structurally_ok':26s} {struct.get('structurally_ok')}")
    for problem in struct.get("problems", []):
        add(f"  PROBLEM: {problem}")
    add("")

    add("[capabilities]")
    for key, value in (report.get("capabilities") or {}).items():
        add(f"  {key:26s} {value}")
    for key, reason in (report.get("capability_reasons") or {}).items():
        if reason:
            add(f"    {key}: {reason}")
    add("")
    for entry in report.get("unsupported_operations", []):
        add(f"UNSUPPORTED: {entry}")
    for entry in report.get("unknown", []):
        add(f"UNKNOWN: {entry}")
    add("")
    add("=" * 72)
    return "\n".join(lines) + "\n"
