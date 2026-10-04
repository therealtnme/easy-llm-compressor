"""End-to-end compression pipeline (the thing the CLI calls)."""

from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch

from . import budget as budget_mod
from . import checkpoint as checkpoint_mod
from . import evaluation as eval_mod
from . import fusion as fusion_mod
from . import pruning
from .data import CalibrationData, DataError, benchmark_batch, prepare
from .distill import DistillConfig, DistillError, distill
from .model.introspect import ModelIntrospector
from .model.loader import load_model_for_inspection
from .model.protection import (ProtectionPolicy, identify_protected_components,
                               summarize_protected)
from .output.report import build_report, write_report
from .scoring import (ActivationCollector, activation_scores, combine_scores,
                      neuron_weight_scores, normalize_scores, select_by_budget)
from .search import joint_search, layer_saliency


class CompressionError(RuntimeError):
    pass


@dataclass
class CompressionOptions:
    model: str = ""
    output: str = "compressed_model"
    device: str = "cpu"
    dtype: str = "float32"
    trust_remote_code: bool = False
    revision: Optional[str] = None
    seed: int = 0

    # data
    dataset: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    dataset_field: str = "text"
    dataset_mode: str = "balanced"       # fast | balanced | accurate | custom
    num_samples: Optional[int] = None
    seq_len: Optional[int] = None
    packing: Optional[bool] = None
    batch_size: int = 4
    dataset_free: bool = False

    # neuron budget
    remove_percent: Optional[float] = None
    remove_count: Optional[int] = None
    remove_scope: str = "mlp"            # mlp | mlp+attention
    allocation: str = "global"           # global | uniform | hybrid
    scoring: str = "weight_combined"     # weight_* | activation | activation_weighted
    min_keep_ratio: float = 0.05
    max_remove_ratio: float = 0.9

    # attention / KV
    remove_attention_percent: Optional[float] = None
    remove_kv_percent: Optional[float] = None

    # depth
    layer_mode: str = "search"           # search | none
    target_student_layers: int = 2
    beam_width: int = 16
    max_span: int = 4
    top_candidates: int = 3
    distill_recovery_estimate: float = 0.6
    fuse: str = "learned"                # none | exact | learned

    # distillation
    distill_steps: int = 100
    distill_lr: float = 5e-5
    distill_losses: str = "auto"
    temperature: float = 2.0

    # protection
    protect_categories: tuple = ()
    unprotect_categories: tuple = ()
    protect: tuple = ()
    unprotect: tuple = ()

    validate: bool = True
    report_name: str = "compression_report"


def _policy(opts: CompressionOptions) -> ProtectionPolicy:
    policy = ProtectionPolicy()
    if opts.protect_categories:
        policy.categories |= set(opts.protect_categories)
    if opts.unprotect_categories:
        policy.categories -= set(opts.unprotect_categories)
        policy.unprotect.extend(opts.unprotect_categories)
    policy.extra.extend(opts.protect)
    policy.unprotect.extend(opts.unprotect)
    return policy


def _initial_report(model, arch) -> dict:
    summary = dict(arch.summarize())
    summary["protected_components"] = summarize_protected(arch.protected)
    summary["tied_weights"] = len(arch.tied_groups)
    summary["notes"] = list(arch.notes)
    return summary


def _target_stack(arch, opts: CompressionOptions):
    stack = arch.primary_stack()
    if stack is None:
        raise CompressionError("no transformer layer stack was found")
    if opts.layer_mode == "search":
        for s in arch.target_stacks():
            if s.role in ("decoder", "encoder") and len(s.blocks()) > 1:
                return s
    return stack


def _attention_scores(arch) -> dict:
    scores = {}
    for L in arch.layers:
        for a in (L.attentions or ([L.attention] if L.attention else [])):
            if not a.is_prunable():
                continue
            mat = a.o_proj.weight_matrix.detach().float()
            hd = a.head_dim or (mat.shape[0] // max(1, a.num_heads or 1))
            per_head = mat.reshape(mat.shape[0], -1)[:, :].abs().sum(dim=1)
            heads = a.num_heads or 0
            if heads and hd and mat.shape[0] >= heads * hd:
                per_head = (mat[: heads * hd].reshape(heads, hd, -1)
                            .abs().sum(dim=(1, 2)))
            scores[a.name] = per_head
    return scores


def _neuron_scores(arch, data: Optional[CalibrationData], opts: CompressionOptions):
    """Per-FFN neuron importance. Data-free never fabricates activations."""
    pruneable = [L.mlp for L in arch.layers if L.mlp is not None and L.mlp.is_prunable()]
    if not pruneable:
        return {}, "none"
    want_activation = opts.scoring in ("activation", "activation_weighted")
    if want_activation and data is None:
        raise CompressionError(
            f"scoring method '{opts.scoring}' requires calibration data; run in "
            "DATASET MODE (--dataset ...) or use a weight-based method "
            "(weight_l1 | weight_l2 | weight_combined) in DATASET-FREE MODE")
    if not want_activation:
        return {m.name: neuron_weight_scores(m, opts.scoring.replace("weight_", ""))
                for m in pruneable}, opts.scoring
    if arch._model is None:
        raise CompressionError(
            "internal wiring error: architecture is missing its model "
            "reference for activation capture")
    modules = {m.name: m.module for m in pruneable}
    collector = ActivationCollector(modules)
    with collector, torch.no_grad():
        for batch in data.train:
            probe = {k: v for k, v in batch.items() if k != "provenance"}
            arch._model(**probe, use_cache=False)
            collector.bump(1)
    out = {}
    for m in pruneable:
        act = collector.activation(m.name)
        if opts.scoring == "activation":
            out[m.name] = activation_scores(m, act, "activation")
        else:
            out[m.name] = combine_scores(neuron_weight_scores(m, "combined"), act,
                                         alpha=0.5)
    return out, opts.scoring


def _apply_attention_pruning(model, arch, opts, batch) -> dict:
    pct = opts.remove_attention_percent
    kv_pct = opts.remove_kv_percent
    if pct is None and kv_pct is None:
        return {"applied": False, "reason": "not requested"}
    if not arch.cap_prune_attention_heads.is_supported():
        return {"applied": False,
                "reason": f"attention pruning unsupported: "
                          f"{arch.cap_prune_attention_heads.reason}"}
    scores = _attention_scores(arch)
    keep_q, keep_kv = {}, {}
    for L in arch.layers:
        for a in (L.attentions or ([L.attention] if L.attention else [])):
            if a.name not in scores or not a.is_prunable():
                continue
            groups = a.num_key_value_groups or 1
            heads = a.num_heads or 0
            s = scores[a.name]
            if pct is not None and heads:
                n_remove = min(int(heads * float(pct) / 100.0), heads - groups)
                if n_remove <= 0:
                    continue
                n_remove -= n_remove % groups
                if n_remove <= 0:
                    continue
                order = torch.argsort(s, descending=False).tolist()
                drop_groups = set()
                for h in order:
                    g = h // groups
                    if g in drop_groups:
                        continue
                    if len(drop_groups) * groups >= n_remove:
                        break
                    drop_groups.add(g)
                keep_q[a.name] = [h for h in range(heads)
                                  if (h // groups) not in drop_groups]
                kept_groups = sorted({h // groups for h in keep_q[a.name]})
                keep_kv[a.name] = kept_groups
    if not keep_q:
        return {"applied": False, "reason": "no attention heads removable "
                                           "without breaking KV groups"}
    ref = pruning.reference_logits(model, arch, {}, batch)
    res = pruning.prune_attention_heads(model, arch, keep_q, keep_kv)
    guard = pruning.compare_to_reference(model, ref, batch, atol=1e-3, rtol=1e-2)
    return {"applied": True, "removed_heads": res.removed_heads,
            "removed_kv_heads": res.removed_kv_heads,
            "equivalence": guard,
            "records": [asdict(r) for r in res.records]}


def run_compression(model_or_id: Any, opts: CompressionOptions,
                    progress=None) -> dict:
    """Run the full pipeline and return the report dict (also written to disk)."""
    torch.manual_seed(opts.seed)
    if isinstance(model_or_id, str):
        model, config = load_model_for_inspection(
            model_or_id, device=opts.device, dtype_str=opts.dtype,
            trust_remote_code=opts.trust_remote_code, revision=opts.revision)
    else:
        model, config = model_or_id, model_or_id.config

    config = model.config  # the object that save_pretrained will serialise
    policy = _policy(opts)
    arch = ModelIntrospector(model, opts.model or type(model).__name__, config).analyze()
    arch.protected = identify_protected_components(
        model, arch.embeddings, arch.output_heads, arch.tied_groups, policy)
    initial = _initial_report(model, arch)
    model_parameters_before = arch.total_parameters
    protected_names = {p.name for p in arch.protected}
    execution: dict = {"records": [], "notes": [], "unsupported": [], "unknown": []}
    tokenizer = None
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            opts.model, trust_remote_code=opts.trust_remote_code)
    except Exception:
        tokenizer = None

    # ---- calibration data -------------------------------------------------- #
    data: Optional[CalibrationData] = None
    if opts.dataset and not opts.dataset_free:
        if progress:
            progress("loading calibration data")
        if tokenizer is None:
            raise DataError(
                f"DATASET MODE needs a tokenizer for '{opts.model}'")
        data = prepare(model, tokenizer, opts.dataset, mode=opts.dataset_mode,
                       seq_len=opts.seq_len, samples=opts.num_samples,
                       packing=opts.packing, field=opts.dataset_field,
                       split=opts.dataset_split, batch_size=opts.batch_size)
    elif not opts.dataset and not opts.dataset_free:
        opts.dataset_free = True

    dataset_info = data.info if data is not None else {
        "mode": "dataset-free",
        "note": ("no calibration dataset supplied: only data-free scoring "
                 "(weight norms / structure) is used and no activations are "
                 "invented"),
    }

    # ---- neuron scores ----------------------------------------------------- #
    if progress:
        progress("scoring FFN neurons")
    arch._model = model
    scores, scoring_used = _neuron_scores(arch, data, opts)

    widths = {L.mlp.name: int(L.mlp.intermediate_size)
              for L in arch.layers
              if L.mlp is not None and L.mlp.is_prunable()
              and L.mlp.name not in protected_names}
    total_neurons = sum(widths.values())
    remove_target = 0
    if opts.remove_percent is not None or opts.remove_count is not None:
        remove_target = budget_mod.resolve_target(
            total_neurons, opts.remove_percent, opts.remove_count,
            min_keep_ratio=opts.min_keep_ratio)

    # ---- depth / width search --------------------------------------------- #
    stack = _target_stack(arch, opts)
    teacher_depth = len(stack.blocks())
    sal_info = layer_saliency(model, arch, stack.name,
                              batches=data.train if data else None)
    depth_plan: Optional[dict] = None
    allocation = {}
    unsupported: list[str] = execution["unsupported"]

    if opts.layer_mode == "search":
        if teacher_depth <= opts.target_student_layers:
            raise CompressionError(
                f"target student depth {opts.target_student_layers} is not "
                f"smaller than teacher depth {teacher_depth}; layer compression "
                "would not reduce depth (refusing to generate "
                f"{teacher_depth} -> {opts.target_student_layers})")
        fusion_info = fusion_mod.detect_exact_fusion(model, arch)
        allow_exact = opts.fuse in ("exact", "learned") and fusion_info["allowed"] > 0
        if opts.fuse == "exact" and fusion_info["allowed"] == 0:
            unsupported.append(
                "EXACT_FUSE: " + fusion_info["reason"] +
                " (structural proof unavailable, so no layer was fused)")
        # Layers whose block owns a parameter no other block provides cannot be
        # deleted structurally (e.g. T5's first decoder block). Treat them as
        # protected so the search keeps them instead of proposing a plan whose
        # deletion would produce an unloadable checkpoint.
        unsafe_layers = pruning.unsafe_layer_indices(arch, stack)
        if unsafe_layers:
            protected_names = set(protected_names) | {
                f"{stack.name}.{i}" for i in unsafe_layers}
            unsupported.append(
                f"layer deletion: indices {sorted(unsafe_layers)} of "
                f"'{stack.name}' own parameters that no other block provides, "
                "so they are kept (deleting them would not be reloadable)")
        plans = joint_search(
            arch, stack.name, sal_info, widths, scores, remove_target,
            opts.target_student_layers, beam_width=opts.beam_width,
            max_span=opts.max_span, max_remove_ratio=opts.max_remove_ratio,
            protected=protected_names,
            allow_exact_fuse=allow_exact, allow_distill_fuse=opts.fuse != "none",
            exact_fuse_reason=fusion_info["reason"],
            distill_available=data is not None or opts.fuse == "learned",
            top_k=opts.top_candidates)
        if not plans:
            raise CompressionError(
                "no valid depth/width plan satisfies the requested target "
                f"student depth {opts.target_student_layers} with a neuron "
                f"budget of {remove_target} removals")
        plan = plans[0]
        allocation = plan.allocation
        depth_plan = plan.to_dict()
        depth_plan["algorithm"] = "beam"
        depth_plan["candidates"] = len(plans)
        depth_plan["scoring_method"] = scoring_used
        depth_plan["saliency_method"] = sal_info["method"]
        depth_plan["teacher_depth"] = teacher_depth
        depth_plan["target_student_layers"] = opts.target_student_layers
        depth_plan["beam_width"] = opts.beam_width
    else:
        if opts.remove_percent is not None or opts.remove_count is not None:
            allocation = budget_mod.allocate(
                widths, scores, remove_target, scope=opts.allocation,
                max_remove_ratio=opts.max_remove_ratio,
                protected=protected_names)
        depth_plan = {"algorithm": "none", "regions": [], "teacher_depth":
                      teacher_depth, "target_student_layers": teacher_depth,
                      "layer_cost": 0.0, "neuron_cost": allocation.get("mass", 0.0),
                      "total_cost": allocation.get("mass", 0.0),
                      "scoring_method": scoring_used,
                      "saliency_method": sal_info["method"]}
        execution["notes"].append("layer_mode=none: no layer was deleted or fused")

    # ---- structural neuron pruning (physical slice) ------------------------ #
    teacher_copy = None
    if depth_plan.get("regions") and any(
            r["operation"] == "DISTILL_FUSE" for r in depth_plan["regions"]) \
            and opts.fuse == "learned":
        teacher_copy = copy.deepcopy(model).eval()

    keep = {p: idx for p, idx in allocation.get("keep", {}).items()}
    doomed = set()
    if depth_plan.get("survivors") is not None and depth_plan.get("regions"):
        survivors = set(depth_plan.get("survivors", []))
        layer_of = {L.mlp.name: (L.stack, L.index) for L in arch.layers
                    if L.mlp is not None}
        for path in list(keep):
            stack_i = layer_of.get(path)
            if stack_i and stack_i[0] == stack.name and stack_i[1] not in survivors:
                doomed.add(path)
        for path in doomed:
            keep.pop(path)
        if doomed:
            execution["notes"].append(
                f"{len(doomed)} FFN(s) belong to layers removed by the depth plan; "
                "their neurons are counted in the global budget arithmetic but are "
                "removed by layer deletion rather than by slicing")
    probe = benchmark_batch(model, seq_len=8)
    mlp_records = []
    equivalence = None
    if keep:
        if progress:
            progress("pruning FFN neurons (physical slice)")
        ref = pruning.reference_logits(model, arch, keep, probe)
        res = pruning.prune_mlp_neurons(model, arch, keep)
        equivalence = pruning.compare_to_reference(model, ref, probe)
        mlp_records = [asdict(r) for r in res.records]
        if not equivalence.get("ok"):
            raise CompressionError(
                "structural neuron pruning did not reproduce the zero-masked "
                f"reference outputs (max diff "
                f"{equivalence.get('max_abs_diff')}); refusing to save. "
                "This indicates a slicing bug, not an approximation.")
        arch = ModelIntrospector(model, opts.model, config).analyze()
        arch._model = model
    else:
        equivalence = {"ok": True, "note": "no neuron pruning requested"}

    attn_result = _apply_attention_pruning(model, arch, opts, probe)
    if attn_result.get("applied"):
        arch = ModelIntrospector(model, opts.model, config).analyze()
        arch._model = model

    # ---- layer deletion / learned fusion ---------------------------------- #
    dropped_layers = 0
    distill_result: dict = {"used": False, "reason": "no DISTILL_FUSE region "
                                                     "in the selected plan"}
    if depth_plan.get("regions"):
        drop = sorted({i for i in range(teacher_depth)} -
                      set(depth_plan.get("survivors", [])))
        if drop:
            if progress:
                progress(f"deleting {len(drop)} layer(s)")
            try:
                pruning.delete_layers(model, arch, stack.name, drop)
            except pruning.RewriteError as exc:
                unsupported.append(f"layer deletion skipped: {exc}")
                dropped_layers = 0
            else:
                dropped_layers = len(drop)
            arch = ModelIntrospector(model, opts.model, config).analyze()
            arch._model = model

    fuse_regions = [r for r in depth_plan.get("regions", [])
                    if r["operation"] == "DISTILL_FUSE"]
    if fuse_regions and opts.fuse == "learned":
        if data is None:
            raise CompressionError(
                "learned layer fusion requires calibration data; run in "
                "DATASET MODE or choose --fuse none")
        if teacher_copy is None:
            raise CompressionError("internal error: teacher copy missing")
        if progress:
            progress("distilling fused layers")
        cfg = DistillConfig(steps=opts.distill_steps, lr=opts.distill_lr,
                            temperature=opts.temperature,
                            loss_mode=opts.distill_losses,
                            batch_size=opts.batch_size)
        result = distill(model, teacher_copy, data.train, cfg,
                         protected_prefixes=tuple(p.name for p in arch.protected))
        distill_result = {"used": True, **result}
        arch = ModelIntrospector(model, opts.model, config).analyze()
        arch._model = model
    elif fuse_regions:
        distill_result = {
            "used": False,
            "reason": (f"{len(fuse_regions)} DISTILL_FUSE region(s) were reduced "
                       "by deletion without fine-tuning (--fuse none); this is "
                       "reported as an approximation, not an exact fusion"),
        }

    # ---- exact fusion ------------------------------------------------------ #
    fusion_report = {"requested": opts.fuse}
    if opts.fuse in ("exact", "learned"):
        info = fusion_mod.detect_exact_fusion(model, arch)
        fusion_report.update(info)
        fusion_report["exact_fusion_allowed"] = info["allowed"]
        reasons = [c["meta"]["reasons"] for c in info["candidates"]]
        fusion_report["exact_fusion_reasons"] = [
            "; ".join(r) for r in reasons if r][:6]
        if info["allowed"] == 0 and opts.fuse == "exact":
            execution["unsupported"].append(
                "EXACT_FUSE unavailable: " + info["reason"])
        else:
            execution["unknown"].append(
                "exact affine fusion: candidate detection only proves merges "
                "inside nn.Sequential chains; block-level merges remain "
                "unsupported because a residual/nonlinearity intervenes")
        fusion_report["applied"] = 0
    else:
        fusion_report["applied"] = 0
        fusion_report["reason"] = "fusion not requested (--fuse none)"

    # ---- save + reload ---------------------------------------------------- #
    if progress:
        progress("saving checkpoint")
    manifest_extra = {
        "compression": {
            "neuron_budget": {"percent": opts.remove_percent,
                              "count": opts.remove_count,
                              "target_removed": remove_target,
                              "removed": allocation.get("removed", 0),
                              "scope": opts.allocation,
                              "unit": "mlp_intermediate_channel"},
            "layers": {"teacher_depth": teacher_depth,
                       "target_student_layers": opts.target_student_layers,
                       "student_depth": arch.num_layers,
                       "deleted": dropped_layers,
                       "operations": depth_plan.get("regions", [])},
            "fusion": {"mode": opts.fuse, "applied": fusion_report.get("applied", 0)},
            "distillation": {"used": distill_result.get("used", False)},
        },
        "protected": [p.name for p in arch.protected],
    }
    manifest = checkpoint_mod.save_compressed(
        model, arch, opts.output, tokenizer=tokenizer, extra=manifest_extra)

    reloaded, _, _ = checkpoint_mod.load_compressed(opts.output, device="cpu")
    reload_ok = True
    reload_reason = "reloaded and state dict matched exactly"
    try:
        with torch.no_grad():
            reloaded(**probe, use_cache=False)
    except Exception as exc:
        reload_ok = False
        reload_reason = f"forward pass after reload failed: {exc}"

    structural = eval_mod.structural_validation(
        reloaded, arch, model_parameters_before, manifest.get("checkpoint_bytes"))
    if not reload_ok:
        structural["problems"].append(reload_reason)
        structural["structurally_ok"] = False
    if not structural["structurally_ok"]:
        raise CompressionError(
            "structural validation failed; refusing to report success: "
            + "; ".join(structural["problems"]))

    # ---- validation ------------------------------------------------------- #
    validation: dict = {"reload": {"ok": reload_ok, "reason": reload_reason}}
    if opts.validate:
        if progress:
            progress("validating")
        validation["latency"] = {"compressed": eval_mod.measure_latency(reloaded, probe)}
        validation["memory"] = eval_mod.peak_memory_mb()
        if data is not None:
            validation["next_token_agreement"] = eval_mod.next_token_agreement(
                teacher_copy if teacher_copy is not None else model, reloaded,
                data.valid)
            validation["loss"] = eval_mod.evaluation_loss(reloaded, data.valid)
        else:
            validation["next_token_agreement"] = {
                "available": False,
                "reason": "DATASET-FREE MODE: agreement requires real examples"}
            validation["loss"] = {
                "available": False,
                "reason": "DATASET-FREE MODE: held-out loss requires real data"}
        if data is not None and tokenizer is not None:
            try:
                prompts = [tokenizer.decode(data.valid[0]["input_ids"][0][:12],
                                            skip_special_tokens=True)]
                validation["generation"] = eval_mod.generation_compare(
                    teacher_copy if teacher_copy is not None else model,
                    reloaded, tokenizer, [p for p in prompts if p])
            except Exception as exc:
                validation["generation"] = [{"available": False,
                                             "reason": str(exc)}]

    execution["records"] = mlp_records + list(attn_result.get("records", []))
    execution["equivalence"] = equivalence
    execution["attention"] = {k: v for k, v in attn_result.items() if k != "records"}
    execution["layers_deleted"] = dropped_layers
    execution["distillation"] = distill_result
    execution["neuron_budget"] = {
        "pool_neurons": total_neurons, "target_removed": remove_target,
        "actual_removed": allocation.get("removed", 0),
        "removed_mass": allocation.get("mass", 0.0),
        "unit": "mlp_intermediate_channel",
    }

    report = build_report(
        arch, asdict(opts), initial=initial, structural=structural,
        plan=depth_plan, execution=execution, validation=validation,
        dataset_info=dataset_info, fusion=fusion_report,
        distillation=distill_result,
        extra={"output_dir": opts.output,
               "checkpoint_manifest": manifest,
               "report_files": {}})
    files = write_report(report, opts.output)
    report["report_files"] = files
    write_report(report, opts.output)
    return report
