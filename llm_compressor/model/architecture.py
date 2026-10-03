"""Normalised structural representation of an instantiated Transformers model.

Nothing in here is architecture-name driven. A `ModelArchitecture` describes what
was *actually found* in the module graph, and every capability is tri-state:
SUPPORTED / UNKNOWN / UNSUPPORTED with a reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import torch.nn as nn

from .projection import Projection


class SupportLevel(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class Capability:
    level: SupportLevel
    reason: str = ""

    def is_supported(self) -> bool:
        return self.level == SupportLevel.SUPPORTED


# --------------------------------------------------------------------------- #
# component refs
# --------------------------------------------------------------------------- #
@dataclass
class NormRef:
    name: str
    module: nn.Module
    kind: str = "unknown"  # layernorm | rmsnorm | batchnorm | unknown


@dataclass
class AttentionInfo:
    name: str
    module: nn.Module
    q_proj: Optional[Projection] = None
    k_proj: Optional[Projection] = None
    v_proj: Optional[Projection] = None
    o_proj: Optional[Projection] = None
    fused_proj: Optional[Projection] = None
    num_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None
    num_key_value_groups: Optional[int] = None
    is_grouped_query: bool = False
    is_fused_qkv: bool = False
    is_cross_attention: bool = False
    has_qk_norm: bool = False
    status: str = "unknown"  # ok | unknown
    reason: str = ""

    def is_prunable(self) -> bool:
        return (
            self.status == "ok"
            and not self.is_cross_attention
            and self.q_proj is not None
            and self.k_proj is not None
            and self.v_proj is not None
            and self.o_proj is not None
            and self.num_heads
            and self.head_dim
        )


@dataclass
class MLPInfo:
    name: str
    module: nn.Module
    gate_proj: Optional[Projection] = None
    up_proj: Optional[Projection] = None
    down_proj: Optional[Projection] = None
    gate_up_fused: bool = False
    intermediate_size: Optional[int] = None
    kind: str = "unknown"  # gated | standard | unknown
    # neurons can be physically sliced iff:
    #   safe=True  -> structure proven compatible
    #   hidden_coupled=True -> an activation couples the up/down chains in a
    #                          non-elementwise way (fused gate/up slicing).
    safe: bool = False
    hidden_coupled: bool = False
    status: str = "unknown"  # ok | unknown
    reason: str = ""

    def is_prunable(self) -> bool:
        return self.status == "ok" and self.safe and self.down_proj is not None and bool(
            self.up_proj or self.gate_proj
        )


@dataclass
class LayerInfo:
    index: int
    name: str
    module: nn.Module
    attention: Optional[AttentionInfo] = None
    attentions: list[AttentionInfo] = field(default_factory=list)
    mlp: Optional[MLPInfo] = None
    norms: list[NormRef] = field(default_factory=list)
    has_residual: bool = False
    residual_evidence: str = ""
    stack: str = ""
    role: str = "unknown"  # decoder | encoder | vision | audio | unknown
    status: str = "ok"
    reason: str = ""


@dataclass
class EmbeddingInfo:
    name: str
    module: nn.Module
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None


@dataclass
class OutputHeadInfo:
    name: str
    module: nn.Module
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None


@dataclass
class TiedGroup:
    names: list[str]
    shape: tuple[int, ...]


@dataclass
class ProtectedComponent:
    name: str
    module: Optional[nn.Module]
    reason: str
    category: str


@dataclass
class LayerStack:
    """A homogeneous ``nn.ModuleList`` of transformer-ish blocks."""

    name: str
    module: nn.Module
    role: str
    num_blocks: int
    config_keys: list[str] = field(default_factory=list)
    list_config_keys: list[str] = field(default_factory=list)

    def blocks(self) -> list[nn.Module]:
        return list(self.module)  # type: ignore[arg-type]


@dataclass
class ModelArchitecture:
    model_name: str
    model_class: str
    config: Any
    total_parameters: int
    trainable_parameters: int
    dtype: Any
    device: str

    hidden_size: Optional[int] = None
    vocab_size: Optional[int] = None
    num_layers: Optional[int] = None
    mlp_intermediate_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_kv_heads: Optional[int] = None
    head_dim: Optional[int] = None

    embeddings: list[EmbeddingInfo] = field(default_factory=list)
    output_heads: list[OutputHeadInfo] = field(default_factory=list)
    stacks: list[LayerStack] = field(default_factory=list)
    layers: list[LayerInfo] = field(default_factory=list)
    tied_groups: list[TiedGroup] = field(default_factory=list)
    protected: list[ProtectedComponent] = field(default_factory=list)

    adapter: str = "generic"
    notes: list[str] = field(default_factory=list)
    # the live nn.Module this architecture describes (set by the pipeline so
    # activation hooks and config sync can reach it without globals)
    _model: Any = None

    cap_inspect: Capability = field(default_factory=lambda: Capability(SupportLevel.SUPPORTED))
    cap_prune_mlp_neurons: Capability = field(default_factory=lambda: Capability(SupportLevel.UNKNOWN))
    cap_prune_attention_heads: Capability = field(default_factory=lambda: Capability(SupportLevel.UNKNOWN))
    cap_prune_kv_heads: Capability = field(default_factory=lambda: Capability(SupportLevel.UNKNOWN))
    cap_prune_layers: Capability = field(default_factory=lambda: Capability(SupportLevel.UNKNOWN))
    cap_fuse_affine: Capability = field(default_factory=lambda: Capability(SupportLevel.UNKNOWN))
    cap_fuse_learned: Capability = field(default_factory=lambda: Capability(SupportLevel.UNKNOWN))

    # ---- derived ---------------------------------------------------------- #
    def all_layers(self) -> list[LayerInfo]:
        return self.layers

    def primary_stack(self) -> Optional[LayerStack]:
        if not self.stacks:
            return None
        for s in self.stacks:
            if s.role == "decoder":
                return s
        for s in self.stacks:
            if s.role == "encoder":
                return s
        return self.stacks[0]

    def target_stacks(self, include_protected: bool = False) -> list[LayerStack]:
        """Stacks eligible for compression by default (text stacks)."""
        out = []
        for s in self.stacks:
            if s.role in ("vision", "audio") and not include_protected:
                continue
            out.append(s)
        return out or list(self.stacks)

    def count_mlp_neurons(self, stacks: Optional[list[LayerStack]] = None) -> int:
        names = {s.name for s in stacks} if stacks else None
        total = 0
        for L in self.layers:
            if names is not None and L.stack not in names:
                continue
            if L.mlp and L.mlp.intermediate_size:
                total += L.mlp.intermediate_size
        return total

    def count_attention_heads(self) -> int:
        return sum(
            a.num_heads or 0
            for L in self.layers
            for a in (L.attentions or ([L.attention] if L.attention else []))
            if not a.is_cross_attention
        )

    def count_kv_heads(self) -> int:
        return sum(
            a.num_kv_heads or 0
            for L in self.layers
            for a in (L.attentions or ([L.attention] if L.attention else []))
            if not a.is_cross_attention
        )

    def summarize(self) -> dict:
        return {
            "model_name": self.model_name,
            "model_class": self.model_class,
            "adapter": self.adapter,
            "parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "dtype": str(self.dtype),
            "device": self.device,
            "hidden_size": self.hidden_size,
            "layers": self.num_layers,
            "mlp_intermediate_size": self.mlp_intermediate_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "mlp_neuron_count": self.count_mlp_neurons(),
            "attention_head_count": self.count_attention_heads(),
            "kv_head_count": self.count_kv_heads(),
            "stacks": [
                {
                    "name": s.name,
                    "role": s.role,
                    "num_blocks": s.num_blocks,
                    "config_keys": s.config_keys,
                    "list_config_keys": s.list_config_keys,
                }
                for s in self.stacks
            ],
        }
