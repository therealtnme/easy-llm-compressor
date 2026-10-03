"""Protection policy.

Default-protect anything whose semantics cannot be preserved by structural
rewriting: input/output embeddings, tied weights, normalisation, rotary/
positional structures, MoE routing, adapters and auxiliary/vision towers.
Everything is configurable, but only *explicitly*.
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Iterable, Optional

import torch.nn as nn

from .architecture import (
    EmbeddingInfo,
    OutputHeadInfo,
    ProtectedComponent,
    TiedGroup,
)
from .names import is_prefix, tokenize_name

NORM_HINTS = ("norm", "layernorm", "rmsnorm", "batchnorm", "groupnorm")
ROTARY_HINTS = ("rotary", "rotaryembedding", "positional", "position_embedding",
                "alibi", "rope")
MOE_HINTS = ("router", "moe", "experts", "expert", "topk", "gate")
ADAPTER_HINTS = ("lora", "adapter", "prefix", "promptencoder", "ia3")
AUX_HINTS = ("pooler", "classification", "classifier", "score", "qa_outputs",
             "lm_head", "lmhead", "tokentype", "seq_relationship", "answer")

DEFAULT_CATEGORIES = {
    "embeddings", "lm_head", "norm", "rotary", "moe", "adapter",
    "tied_weights", "auxiliary", "vision", "audio",
}


@dataclass
class ProtectionPolicy:
    """Decides whether a module may be rewritten."""

    categories: set[str] = field(default_factory=lambda: set(DEFAULT_CATEGORIES))
    extra: list[str] = field(default_factory=list)          # globs to protect
    unprotect: list[str] = field(default_factory=list)      # globs to unprotect

    def is_protected(self, name: str, category: Optional[str] = None) -> bool:
        for pat in self.unprotect:
            if fnmatch.fnmatch(name, pat):
                return False
        if category is not None and category in self.categories:
            return True
        for pat in self.extra:
            if fnmatch.fnmatch(name, pat) or is_prefix(pat, name):
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "categories": sorted(self.categories),
            "extra": list(self.extra),
            "unprotect": list(self.unprotect),
        }


def categorize(name: str, module: Optional[nn.Module]) -> Optional[str]:
    cname = type(module).__name__.lower() if module is not None else ""
    toks = tokenize_name(name)
    lname = name.lower()

    if any(h in cname for h in ADAPTER_HINTS) or any(h in lname for h in ADAPTER_HINTS):
        return "adapter"
    if isinstance(module, nn.Embedding):
        return "embeddings"
    if any(h in cname for h in ROTARY_HINTS) or (toks & set(ROTARY_HINTS)):
        return "rotary"
    if any(h in cname for h in NORM_HINTS) or (toks & {"norm", "ln", "layernorm", "rmsnorm"}):
        return "norm"
    if toks & {"router", "moe", "experts"}:
        return "moe"
    if toks & set(AUX_HINTS) and not toks & {"lm"}:
        return "auxiliary"
    if toks & set(MOE_HINTS) and "gate" in toks and "proj" not in toks:
        # e.g. `gate` (a router) as opposed to `gate_proj` (a gated-MLP matrix)
        return "moe"
    return None


def identify_protected_components(
    model: nn.Module,
    embeddings: list[EmbeddingInfo],
    output_heads: list[OutputHeadInfo],
    tied_groups: list[TiedGroup],
    policy: Optional[ProtectionPolicy] = None,
) -> list[ProtectedComponent]:
    policy = policy or ProtectionPolicy()
    protected: list[ProtectedComponent] = []
    seen: set[str] = set()

    def add(name: str, module, reason: str, category: str) -> None:
        if name in seen:
            return
        seen.add(name)
        protected.append(ProtectedComponent(name, module, reason, category))

    for emb in embeddings:
        if policy.is_protected(emb.name, "embeddings"):
            add(emb.name, emb.module, "input embedding", "embeddings")
    for head in output_heads:
        if policy.is_protected(head.name, "lm_head"):
            add(head.name, head.module, "output head / LM head", "lm_head")

    for name, module in model.named_modules():
        if not name:
            continue
        cat = categorize(name, module)
        if cat in ("adapter", "rotary", "moe"):
            if policy.is_protected(name, cat):
                add(name, module, f"{cat} component", cat)
            continue
        if isinstance(module, nn.Module) and not list(module.children()) \
                and not list(module.parameters(recurse=False)):
            continue
        if cat is None and isinstance(module, (nn.Embedding,)):
            cat = "embeddings"
        if cat is not None and policy.is_protected(name, cat):
            add(name, module, f"{cat} component", cat)

    for grp in tied_groups:
        if not policy.is_protected(grp.names[0], "tied_weights"):
            continue
        for n in grp.names:
            add(n, None, f"tied parameter (shared with {grp.names[0]})", "tied_weights")

    # vision / audio towers
    for name, module in model.named_modules():
        toks = tokenize_name(name)
        if toks & {"vision", "visual", "vit", "audio", "speech"}:
            if policy.is_protected(name, "vision"):
                add(name, module, "non-text tower (protected by default)", "vision")
            continue
    return protected


def summarize_protected(protected: Iterable[ProtectedComponent]) -> dict:
    out: dict[str, int] = {}
    for p in protected:
        out[p.category] = out.get(p.category, 0) + 1
    return out
