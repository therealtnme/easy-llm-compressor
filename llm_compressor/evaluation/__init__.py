"""Validation: structure, size, latency, memory, agreement, loss, generation.

Every measurement is taken on real inputs. Latency is measured, never inferred
from parameter counts. Structural validation is mandatory before a checkpoint
is considered compressed.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Optional, Sequence

import torch

from ..utils.batch import forward_tensor, model_kind


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"parameters": total, "trainable_parameters": trainable}


def structural_validation(model, arch, original_params: int,
                          checkpoint_bytes: Optional[int] = None) -> dict:
    """Verify the model really is structurally smaller."""
    now = count_parameters(model)["parameters"]
    problems: list[str] = []
    neurons = arch.count_mlp_neurons()
    heads = arch.count_attention_heads()
    kv = arch.count_kv_heads()
    layers = arch.num_layers or 0
    if now <= 0:
        problems.append("model reports zero parameters")
    if not getattr(arch, "cap_prune_mlp_neurons", None):
        problems.append("architecture metadata missing")
    for L in arch.layers:
        if L.mlp is None or L.mlp.intermediate_size is None:
            continue
        expected = int(L.mlp.intermediate_size)
        if L.mlp.up_proj is not None and not L.mlp.gate_up_fused:
            if L.mlp.up_proj.out_features != expected:
                problems.append(
                    f"{L.mlp.name}: up projection width "
                    f"{L.mlp.up_proj.out_features} != recorded {expected}")
        if L.mlp.down_proj is not None and L.mlp.down_proj.in_features != expected:
            problems.append(
                f"{L.mlp.name}: down projection input "
                f"{L.mlp.down_proj.in_features} != recorded {expected}")
    return {
        "parameters": now,
        "parameters_before": original_params,
        "parameters_saved": original_params - now,
        "parameter_change_percent": (100.0 * (original_params - now)
                                    / original_params if original_params else 0.0),
        "mlp_neurons": neurons,
        "attention_heads": heads,
        "kv_heads": kv,
        "layers": layers,
        "checkpoint_bytes": checkpoint_bytes,
        "problems": problems,
        "structurally_ok": not problems and now < original_params,
    }


def checkpoint_size(out_dir: str) -> int:
    return sum(os.path.getsize(os.path.join(out_dir, f))
               for f in os.listdir(out_dir)
               if os.path.isfile(os.path.join(out_dir, f)))


def measure_latency(model, batch: dict, warmup: int = 2, iters: int = 5) -> dict:
    probe = {k: v for k, v in batch.items() if k != "provenance"}
    was_training = model.training
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(**probe, use_cache=False)
        for _ in range(iters):
            start = time.perf_counter()
            model(**probe, use_cache=False)
            times.append((time.perf_counter() - start) * 1000.0)
    if was_training:
        model.train()
    times.sort()
    return {"iterations": iters, "warmup": warmup,
            "median_ms": times[len(times) // 2], "min_ms": times[0],
            "max_ms": times[-1],
            "batch_shape": list(next(iter(probe.values())).shape),
            "provenance": batch.get("provenance", "model_input")}


def peak_memory_mb() -> dict:
    out = {}
    try:
        import psutil

        out["process_rss_mb"] = psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    if torch.cuda.is_available():
        out["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 1e6
    return out


@torch.no_grad()
def next_token_agreement(teacher, student, batches: Sequence[dict]) -> dict:
    same = total = top5 = 0
    for batch in batches:
        probe = {k: v for k, v in batch.items() if k != "provenance"}
        tl = forward_tensor(teacher, probe).float()
        sl = forward_tensor(student, probe).float()
        if tl.shape != sl.shape:
            return {"available": False,
                    "reason": f"logit shapes differ {tuple(tl.shape)} vs "
                              f"{tuple(sl.shape)}"}
        width = min(tl.shape[1], sl.shape[1]) - 1
        if width <= 0:
            continue
        t_ids = tl[:, :width].argmax(-1)
        s_ids = sl[:, :width].argmax(-1)
        same += int((t_ids == s_ids).sum())
        top5 += int((sl[:, :width].topk(min(5, sl.shape[-1]), dim=-1).indices
                     == t_ids.unsqueeze(-1)).any(-1).sum())
        total += t_ids.numel()
    if not total:
        return {"available": False, "reason": "no comparable tokens"}
    return {"available": True, "next_token_agreement": same / total,
            "top5_agreement": top5 / total, "tokens": total}


@torch.no_grad()
def evaluation_loss(model, batches: Sequence[dict]) -> dict:
    """Next-token cross-entropy (labels from the input ids) and perplexity."""
    import torch.nn.functional as F

    total_loss, total_tokens = 0.0, 0
    for batch in batches:
        probe = {k: v for k, v in batch.items() if k != "provenance"}
        if model_kind(model) != "causal":
            out = model(**probe, use_cache=False)
            logits = getattr(out, "logits", None)
            if logits is None:
                return {"available": False,
                        "reason": "model has no causal LM logits"}
        logits = forward_tensor(model, probe)
        labels = probe.get("decoder_input_ids", probe["input_ids"])
        shift_logits = logits[:, :-1].float()
        shift_labels = labels[:, 1:]
        # teacher/student sequence lengths may differ by a token; score the
        # common prefix instead of raising a shape error
        n = min(shift_logits.shape[0], shift_labels.shape[0])
        steps = min(shift_logits.shape[1], shift_labels.shape[1])
        loss = F.cross_entropy(shift_logits[:n, :steps].reshape(
            -1, shift_logits.shape[-1]), shift_labels[:n, :steps].reshape(-1),
            ignore_index=-100, reduction="sum")
        tokens = int((shift_labels[:n, :steps] != -100).sum())
        if tokens:
            total_loss += float(loss)
            total_tokens += tokens
    if not total_tokens:
        return {"available": False, "reason": "no label tokens"}
    mean = total_loss / total_tokens
    return {"available": True, "loss": mean,
            "perplexity": float(torch.exp(torch.tensor(mean))),
            "tokens": total_tokens}


@torch.no_grad()
def generation_compare(teacher, student, tokenizer, prompts: Sequence[str],
                       max_new_tokens: int = 12) -> list[dict]:
    out = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v for k, v in enc.items() if k in ("input_ids",
                                                     "attention_mask")}
        if model_kind(teacher) == "seq2seq":
            enc["decoder_input_ids"] = enc["input_ids"][:, :1]
        try:
            t = teacher.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False)
            s = student.generate(**enc, max_new_tokens=max_new_tokens,
                                 do_sample=False)
        except Exception as exc:  # pragma: no cover - model-specific
            out.append({"prompt": prompt, "available": False,
                        "reason": f"generation failed: {exc}"})
            continue
        out.append({
            "prompt": prompt,
            "teacher": tokenizer.decode(t[0], skip_special_tokens=True),
            "student": tokenizer.decode(s[0], skip_special_tokens=True),
            "match": bool(t.shape == s.shape and torch.equal(t, s)),
            "available": True,
        })
    return out
