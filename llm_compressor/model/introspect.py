"""Generic, module-graph-driven introspection.

Strategy (no architecture-name dispatch):

1. find candidate block stacks (homogeneous ``nn.ModuleList``s of transformer-ish
   blocks) from the instantiated module graph;
2. inside each block, classify projections by *name tokens* **and** *shape*,
   validating candidate attention/MLP groupings structurally;
3. refuse to claim support for anything not proven (UNKNOWN + reason).
"""
from __future__ import annotations

import inspect
import re
from typing import Any, Optional

import torch.nn as nn

from .architecture import (
    AttentionInfo,
    Capability,
    EmbeddingInfo,
    LayerInfo,
    LayerStack,
    MLPInfo,
    ModelArchitecture,
    NormRef,
    OutputHeadInfo,
    SupportLevel,
    TiedGroup,
)
from .names import is_prefix, tokenize_name
from .projection import Projection, is_projection
from .protection import identify_protected_components

# --------------------------------------------------------------------------- #
# config hints
# --------------------------------------------------------------------------- #
_ALIASES: dict[str, list[str]] = {
    "hidden": ["hidden_size", "d_model", "n_embd", "dim", "model_dim", "d_model_t5"],
    "layers": ["num_hidden_layers", "n_layer", "num_layers", "encoder_layers",
               "decoder_layers", "num_decoder_layers", "num_encoder_layers"],
    "inter": ["intermediate_size", "n_inner", "ffn_dim", "ff_dim", "mlp_dim",
              "d_ff", "dense_ffn_dim", "ffn_hidden_size"],
    "heads": ["num_attention_heads", "n_head", "num_heads",
              "encoder_attention_heads", "decoder_attention_heads"],
    "kv": ["num_key_value_heads", "num_kv_heads", "n_kv_head"],
    "head_dim": ["head_dim", "d_kv", "d_head"],
    "vocab": ["vocab_size"],
}


class _ConfigView:
    """Read-only view over config + one level of nested sub-configs."""

    def __init__(self, config: Any):
        self.root = config
        self._scopes: list[tuple[str, Any]] = [("", config)]
        for name in ("text_config", "decoder", "encoder", "vision_config",
                     "audio_config", "language_config"):
            sub = getattr(config, name, None)
            if sub is not None and hasattr(sub, "__dict__"):
                self._scopes.append((name, sub))

    def iter_values(self) -> list[tuple[str, Any]]:
        out: list[tuple[str, Any]] = []
        for prefix, scope in self._scopes:
            for k, v in vars(scope).items():
                if k.startswith("_") or callable(v):
                    continue
                dotted = f"{prefix}.{k}" if prefix else k
                out.append((dotted, v))
        return out

    def get(self, dotted: str) -> Any:
        obj: Any = self.root
        for part in dotted.split("."):
            if obj is None or not hasattr(obj, part):
                return None
            obj = getattr(obj, part)
        return obj

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        obj: Any = self.root
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def hint(self, kind: str, prefer: Optional[tuple[str, ...]] = None):
        """First alias hit, preferring scopes whose path matches `prefer`."""
        hits: list[tuple[str, Any]] = []
        for alias in _ALIASES[kind]:
            for dotted, value in self.iter_values():
                if dotted.rsplit(".", 1)[-1] == alias and isinstance(
                    value, (int, float)
                ):
                    hits.append((dotted, value))
        if not hits:
            return None, None
        if prefer:
            for prefix in prefer:
                for dotted, value in hits:
                    if prefix in dotted.lower():
                        return dotted, value
        return hits[0]


# --------------------------------------------------------------------------- #
# name vocabularies (token based)
# --------------------------------------------------------------------------- #
ATTN_TOK = {"attn", "attention", "mha", "selfattention", "self"}
CROSS_TOK = {"cross"}
Q_TOK = {"q", "query", "wq"}
K_TOK = {"k", "key", "wk"}
V_TOK = {"v", "value", "wv"}
O_TOK = {"o", "out", "output", "dense", "wo", "outproj"}
FUSED_QKV_LEAVES = {"c_attn", "wqkv", "qkv_proj", "query_key_value", "qkv",
                    "in_proj", "qkv_linear"}
GATE_TOK = {"gate", "w1", "wi"}
UP_LEAVES = {"c_fc", "fc1", "fc_in", "dense_h_to_4h", "up_proj", "wi_1",
             "gate_up_proj", "h_to_4h", "w3", "up"}
DOWN_LEAVES = {"c_proj", "fc2", "fc_out", "dense_4h_to_h", "down_proj", "wo",
               "4h_to_h", "w2", "down", "output"}
MLP_CONTAINER_TOK = {"mlp", "ffn", "ff", "feedforward", "feed", "forward",
                     "densereludense", "densegatedactdense", "intermediate",
                     "output", "dense"}
VISION_TOK = {"vision", "visual", "vit", "image", "patch", "resampler",
              "merger", "connector", "multimodalprojector"}
AUDIO_TOK = {"audio", "speech", "wav"}
ROUTER_TOK = {"router", "gate", "moe", "experts", "expert"}
NORM_TOK = {"norm", "layernorm", "rmsnorm", "batchnorm", "groupnorm", "ln"}
RESIDUAL_RE = re.compile(
    r"(\+\s*residual|residual\s*\+|hidden_states\s*\+|\+\s*hidden_states"
    r"|torch\.add|\.add_\(|=\s*\w+\s*\+\s*\w+|attention_output\s*\+)"
)


def _leaf(name: str) -> str:
    return str(name).rsplit(".", 1)[-1].lower()


def _norm_kind(module: nn.Module, name: str) -> str:
    cname = type(module).__name__.lower()
    lname = name.lower()
    if "rms" in cname or "rms" in lname:
        return "rmsnorm"
    if "layer" in cname or "layernorm" in lname:
        return "layernorm"
    if "batch" in cname:
        return "batchnorm"
    if "group" in cname:
        return "groupnorm"
    if tokenize_name(name) & NORM_TOK:
        return "layernorm"
    return "unknown"


def _is_norm(module: nn.Module, name: str) -> bool:
    return _norm_kind(module, name) != "unknown"


class BlockAnalysis:
    """Roles assigned inside one block."""

    def __init__(self) -> None:
        self.attention: Optional[AttentionInfo] = None
        self.attentions: list[AttentionInfo] = []
        self.mlp: Optional[MLPInfo] = None
        self.norms: list[NormRef] = []


class ModelIntrospector:
    def __init__(self, model: nn.Module, model_name: str, config: Any):
        self.model = model
        self.model_name = model_name
        self.config = config
        self.cfg = _ConfigView(config)

        self.hidden_path, self.hidden = self.cfg.hint("hidden")
        self.inter_path, self.inter_cfg = self.cfg.hint("inter")
        self.heads_path, self.heads_cfg = self.cfg.hint("heads")
        self.kv_path, self.kv_cfg = self.cfg.hint("kv")
        self.hdim_path, self.head_dim_cfg = self.cfg.hint("head_dim")
        self.vocab_path, self.vocab = self.cfg.hint("vocab")
        if self.head_dim_cfg is None and self.hidden and self.heads_cfg:
            if int(self.hidden) % int(self.heads_cfg) == 0:
                self.head_dim_cfg = int(self.hidden) // int(self.heads_cfg)

    # ------------------------------------------------------------------ #
    def analyze(self) -> ModelArchitecture:
        params = list(self.model.parameters())
        total = sum(p.numel() for p in params)
        trainable = sum(p.numel() for p in params if p.requires_grad)
        dtype = params[0].dtype if params else None
        device = str(params[0].device) if params else "unknown"

        stacks = self._find_stacks()
        layers: list[LayerInfo] = []
        for stack in stacks:
            for i, block in enumerate(stack.blocks()):
                li = self._analyze_block(stack, i, block)
                layers.append(li)

        hidden = self.hidden or self._infer_hidden(layers)
        embeddings = self._find_embeddings()
        output_heads = self._find_output_heads()
        tied = self._find_tied_parameters()
        protected = identify_protected_components(
            self.model, embeddings, output_heads, tied
        )

        primary = next((s for s in stacks if s.role == "decoder"), None)
        if primary is None:
            primary = stacks[0] if stacks else None
        primary_layers = [L for L in layers if primary and L.stack == primary.name]

        intermediate = self.inter_cfg
        if intermediate is None:
            for L in primary_layers:
                if L.mlp and L.mlp.intermediate_size:
                    intermediate = L.mlp.intermediate_size
                    break

        arch = ModelArchitecture(
            model_name=self.model_name,
            model_class=type(self.model).__name__,
            config=self.config,
            total_parameters=total,
            trainable_parameters=trainable,
            dtype=dtype,
            device=device,
            hidden_size=int(hidden) if hidden else None,
            vocab_size=int(self.vocab) if self.vocab else None,
            num_layers=len(primary_layers) if primary_layers else None,
            mlp_intermediate_size=int(intermediate) if intermediate else None,
            num_attention_heads=int(self.heads_cfg) if self.heads_cfg else None,
            num_kv_heads=int(self.kv_cfg) if self.kv_cfg else (
                int(self.heads_cfg) if self.heads_cfg else None
            ),
            head_dim=int(self.head_dim_cfg) if self.head_dim_cfg else None,
            embeddings=embeddings,
            output_heads=output_heads,
            stacks=stacks,
            layers=layers,
            tied_groups=tied,
            protected=protected,
        )
        self._compute_capabilities(arch)
        return arch

    # ------------------------------------------------------------------ #
    # stacks
    # ------------------------------------------------------------------ #
    def _role_of(self, name: str) -> str:
        toks = tokenize_name(name)
        if toks & VISION_TOK:
            return "vision"
        if toks & AUDIO_TOK:
            return "audio"
        if "decoder" in toks:
            return "decoder"
        if "encoder" in toks:
            return "encoder"
        # fall back on config semantics
        if getattr(self.config, "is_encoder_decoder", False):
            return "unknown"
        if getattr(self.config, "is_decoder", False) or self._has_lm_head():
            return "decoder"
        return "encoder"

    def _has_lm_head(self) -> bool:
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) and self.vocab and mod.out_features == self.vocab:
                return True
        return False

    def _find_stacks(self) -> list[LayerStack]:
        candidates: list[tuple[int, LayerStack]] = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.ModuleList) or len(module) < 2:
                continue
            blocks = list(module)
            if len(blocks) < 2:
                continue
            if len({type(b).__name__ for b in blocks}) != 1:
                continue
            n_proj = sum(1 for b in blocks[:1] for _, m in b.named_modules()
                         if is_projection(m))
            if n_proj < 2:
                continue
            stack = LayerStack(
                name=name, module=module, role=self._role_of(name),
                num_blocks=len(blocks),
            )
            stack.config_keys, stack.list_config_keys = self._config_keys_for(stack)
            candidates.append((self._score_stack(name, blocks), stack))

        if not candidates:
            return []
        candidates.sort(key=lambda x: (-x[0], x[1].name))

        # keep only outermost stacks (drop stacks nested inside another)
        selected: list[LayerStack] = []
        for _, stack in candidates:
            if any(is_prefix(s.name, stack.name) for s in selected):
                continue
            if any(is_prefix(stack.name, s.name) for s in selected):
                continue
            selected.append(stack)
        return selected

    def _score_stack(self, name: str, blocks: list[nn.Module]) -> int:
        score = min(len(blocks), 64)
        toks = tokenize_name(name)
        if toks & {"layers", "layer", "blocks", "block", "h", "stack"}:
            score += 25
        if toks & (VISION_TOK | AUDIO_TOK):
            score += 5
        if self.role_layers_matches(blocks):
            score += 100
        return score

    def role_layers_matches(self, blocks: list[nn.Module]) -> bool:
        for kind in ("layers",):
            for alias in _ALIASES[kind]:
                for dotted, value in self.cfg.iter_values():
                    if dotted.rsplit(".", 1)[-1] == alias and isinstance(value, int):
                        if value == len(blocks):
                            return True
        return False

    def _config_keys_for(self, stack: LayerStack) -> tuple[list[str], list[str]]:
        """Config attributes that must be kept in sync with the stack length."""
        ints: list[str] = []
        lists: list[str] = []
        role_tok = stack.role if stack.role in ("encoder", "decoder") else None
        for dotted, value in self.cfg.iter_values():
            leaf = dotted.rsplit(".", 1)[-1]
            scope = dotted.rsplit(".", 1)[0].lower() if "." in dotted else ""
            if isinstance(value, int) and value == stack.num_blocks:
                if any(leaf == a for a in _ALIASES["layers"]):
                    if role_tok:
                        named_role = ("encoder" in leaf) or ("decoder" in leaf)
                        if named_role and role_tok not in leaf:
                            continue
                        if named_role and role_tok not in scope + leaf:
                            continue
                    ints.append(dotted)
            elif isinstance(value, (list, tuple)) and len(value) == stack.num_blocks:
                if any(t in leaf for t in ("layer_types", "layer_type",
                                           "sliding_window", "attention_types")):
                    lists.append(dotted)
        # prefer role-specific keys when both exist
        if role_tok and len(ints) > 1:
            specific = [k for k in ints if role_tok in k.lower()]
            if specific:
                ints = specific
        return ints, lists

    # ------------------------------------------------------------------ #
    # block analysis
    # ------------------------------------------------------------------ #
    def _analyze_block(self, stack: LayerStack, index: int, block: nn.Module) -> LayerInfo:
        name = f"{stack.name}.{index}"
        out = BlockAnalysis()

        prefix = name
        projs: list[Projection] = []
        for rel, mod in block.named_modules():
            if rel and is_projection(mod):
                projs.append(Projection.of(f"{prefix}.{rel}", mod))

        for rel, mod in block.named_modules():
            if rel and _is_norm(mod, rel):
                out.norms.append(NormRef(f"{name}.{rel}", mod, _norm_kind(mod, rel)))

        hidden = self.hidden or self._infer_hidden_from_projs(projs)
        out.attentions = self._find_attentions(projs, hidden, prefix)
        used_ids = {id(p.module) for a in out.attentions
                    for p in (a.q_proj, a.k_proj, a.v_proj, a.o_proj, a.fused_proj)
                    if p is not None}
        remaining = [p for p in projs if id(p.module) not in used_ids]
        out.mlp = self._find_mlp(remaining, hidden, used_ids, prefix)
        out.attention = next(
            (a for a in out.attentions if not a.is_cross_attention), None
        )

        info = LayerInfo(
            index=index, name=name, module=block, stack=stack.name,
            role=stack.role, attention=out.attention, attentions=out.attentions,
            mlp=out.mlp, norms=out.norms,
        )
        evidence = self._residual_evidence(block)
        info.has_residual = bool(evidence)
        info.residual_evidence = evidence
        if not info.attentions and not info.mlp:
            info.status = "unknown"
            info.reason = "no attention or FFN structure identified"
        return info

    def _infer_hidden_from_projs(self, projs: list[Projection]) -> Optional[int]:
        counts: dict[int, int] = {}
        for p in projs:
            counts[p.in_features] = counts.get(p.in_features, 0) + 1
            counts[p.out_features] = counts.get(p.out_features, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]

    def _infer_hidden(self, layers: list[LayerInfo]) -> Optional[int]:
        for L in layers:
            if L.mlp and L.mlp.down_proj:
                return L.mlp.down_proj.out_features
            if L.attention and L.attention.o_proj:
                return L.attention.o_proj.out_features
        return None

    # ---------------------------------------------------------------- #
    def _containers(self, projs: list[Projection], prefix: str) -> list[str]:
        """Candidate container paths (absolute), deepest first."""
        depth = len(prefix.split("."))
        paths = {prefix}
        for p in projs:
            parts = p.name.split(".")
            for i in range(depth, len(parts)):
                paths.add(".".join(parts[:i]))
        return sorted(paths, key=lambda s: (-s.count("."), s))

    def _find_attentions(self, projs: list[Projection],
                         hidden: Optional[int], prefix: str) -> list[AttentionInfo]:
        best: Optional[AttentionInfo] = None
        best_score = -1
        for container in self._containers(projs, prefix):
            group = [p for p in projs if is_prefix(container, p.name)]
            if len(group) < 2:
                continue
            info = self._try_attention(container, group, hidden)
            if info is None:
                continue
            score = self._attn_score(container, info, group)
            if score > best_score:
                best, best_score = info, score
        if best is None and hidden is not None:
            # no complete group proven
            return []
        return [best] if best is not None else []

    def _try_attention(self, container: str, group: list[Projection],
                       hidden: Optional[int]) -> Optional[AttentionInfo]:
        info = AttentionInfo(name=container, module=self._module_of(container))
        extras: list[Projection] = []
        for p in group:
            leaf = _leaf(p.name)
            toks = tokenize_name(p.name)
            if leaf in FUSED_QKV_LEAVES or "qkv" in toks:
                info.fused_proj = p
            elif toks & Q_TOK and info.q_proj is None:
                info.q_proj = p
            elif toks & K_TOK and info.k_proj is None:
                info.k_proj = p
            elif toks & V_TOK and info.v_proj is None:
                info.v_proj = p
            elif toks & O_TOK and info.o_proj is None:
                info.o_proj = p
            else:
                extras.append(p)

        # shape-based role completion
        hd = self.head_dim_cfg
        if hidden:
            eq = [p for p in extras if p.in_features == hidden
                  and p.out_features == hidden]
            wide = [p for p in extras if p.in_features == hidden
                    and p.out_features != hidden]
            if info.q_proj is None and len(eq) >= 3:
                info.q_proj, info.k_proj, info.v_proj = eq[0], eq[1], eq[2]
                eq = eq[3:]
            if info.k_proj is None and info.v_proj is None and wide:
                # GQA: equal-width k/v narrower than hidden
                by_out: dict[int, list[Projection]] = {}
                for p in wide:
                    by_out.setdefault(p.out_features, []).append(p)
                for out_features, ps in sorted(by_out.items(), reverse=True):
                    if len(ps) >= 2:
                        info.k_proj, info.v_proj = ps[0], ps[1]
                        break
            if info.q_proj is None:
                q_cands = [p for p in wide if hd and p.out_features % hd == 0
                           and p.out_features == max(x.out_features for x in wide)]
                if q_cands:
                    info.q_proj = q_cands[0]
            if info.k_proj is None and info.q_proj is not None:
                same = [p for p in wide if p is not info.q_proj
                        and p.in_features == hidden]
                if len(same) >= 2:
                    info.k_proj, info.v_proj = same[0], same[1]
                elif len(same) == 1:
                    info.k_proj = info.v_proj = same[0]

        # validation
        if info.fused_proj is not None:
            info.is_fused_qkv = True
            info.status = "unknown"
            info.reason = "fused QKV projection cannot be split safely"
            return info

        if not (info.q_proj and info.k_proj and info.v_proj):
            return None

        info.head_dim = int(hd) if hd else None
        if info.head_dim is None and self.heads_cfg and info.q_proj.out_features % int(self.heads_cfg) == 0:
            info.head_dim = info.q_proj.out_features // int(self.heads_cfg)

        is_cross = bool(tokenize_name(container) & CROSS_TOK)
        if hidden is not None:
            if info.q_proj.in_features != hidden:
                return None
            if info.k_proj.in_features != hidden or info.v_proj.in_features != hidden:
                info.is_cross_attention = True
                info.status = "unknown"
                info.reason = "key/value projections take a different input width (cross-attention)"
                if not is_cross:
                    return None

        if info.o_proj is not None and hidden is not None:
            if info.o_proj.out_features != hidden:
                info.o_proj = None

        if info.head_dim and info.head_dim > 0:
            if info.q_proj.out_features % info.head_dim == 0:
                info.num_heads = info.q_proj.out_features // info.head_dim
            if info.k_proj.out_features % info.head_dim == 0:
                info.num_kv_heads = info.k_proj.out_features // info.head_dim
        if info.num_kv_heads is None:
            info.num_kv_heads = info.num_heads
        if info.num_heads and info.num_kv_heads:
            if info.num_heads % info.num_kv_heads == 0:
                info.num_key_value_groups = info.num_heads // info.num_kv_heads
            elif info.num_heads != info.num_kv_heads:
                info.status = "unknown"
                info.reason = "query heads not divisible by KV heads"
                return info
            info.is_grouped_query = info.num_kv_heads != info.num_heads
        if info.o_proj is None:
            info.status = "unknown"
            info.reason = "output projection not identified"
        else:
            info.status = "ok"
        return info

    def _module_of(self, path: str) -> nn.Module:
        if not path:
            return self.model
        mod = self.model
        for part in path.split("."):
            mod = getattr(mod, part)
        return mod

    def _attn_score(self, container: str, info: AttentionInfo,
                    group: list[Projection]) -> int:
        score = 0
        if not container:
            score -= 5  # prefer specific containers
        score += 2 * container.count(".")
        if tokenize_name(container) & ATTN_TOK:
            score += 12
        module = self._module_of(container) if container else None
        if module is not None and any(isinstance(m, nn.Softmax) for m in module.modules()):
            score += 6
        if info.o_proj is not None:
            score += 6
        if info.q_proj is not None and info.q_proj.out_features == info.q_proj.in_features:
            score += 4
        if info.is_cross_attention:
            score -= 40
        extra_projs = len(group) - sum(
            x is not None for x in (info.q_proj, info.k_proj, info.v_proj,
                                    info.o_proj, info.fused_proj)
        )
        score -= 3 * max(0, extra_projs)
        return score

    # ---------------------------------------------------------------- #
    def _find_mlp(self, projs: list[Projection], hidden: Optional[int],
                  used_ids: set[int], prefix: str) -> Optional[MLPInfo]:
        best: Optional[MLPInfo] = None
        best_score = -1
        for container in self._containers(projs, prefix):
            group = [p for p in projs if is_prefix(container, p.name)]
            if len(group) < 1:
                continue
            info = self._try_mlp(container, group, hidden, used_ids)
            if info is None:
                continue
            score = self._mlp_score(container, info, group)
            if score > best_score:
                best, best_score = info, score
        return best

    def _try_mlp(self, container: str, group: list[Projection],
                 hidden: Optional[int], used_ids: set[int]) -> Optional[MLPInfo]:
        info = MLPInfo(name=container, module=self._module_of(container))
        extras: list[Projection] = []
        ups: list[Projection] = []
        for p in group:
            leaf = _leaf(p.name)
            toks = tokenize_name(p.name)
            if leaf in GATE_TOK or "gate" in toks:
                if info.gate_proj is None:
                    info.gate_proj = p
                    continue
            if leaf in DOWN_LEAVES or ("down" in toks):
                if info.down_proj is None:
                    info.down_proj = p
                    continue
            if leaf in UP_LEAVES or up_tokens(toks):
                ups.append(p)
                continue
            extras.append(p)

        pool = list(group)
        # 1) locate the down projection (hidden <- intermediate)
        if info.down_proj is None:
            cands = [p for p in pool if hidden is not None
                     and p.out_features == hidden and p.in_features > hidden]
            if cands:
                info.down_proj = max(cands, key=lambda p: p.in_features)
        if info.down_proj is None:
            return None
        inter = info.down_proj.in_features

        # 2) locate up / gate projections (hidden -> intermediate)
        up_like = [p for p in pool
                   if p is not info.down_proj and hidden is not None
                   and p.in_features == hidden and p.out_features == inter]
        fused = [p for p in pool
                 if p is not info.down_proj and hidden is not None
                 and p.in_features == hidden and p.out_features == 2 * inter]
        assigned = {id(p.module) for p in (info.up_proj, info.gate_proj) if p}
        broke = False
        for p in sorted(fused + up_like, key=lambda p: (p.out_features != 2 * inter, p.name)):
            if id(p.module) in assigned:
                continue
            if p.out_features == 2 * inter and p.out_features != inter:
                info.up_proj = p
                info.gate_up_fused = True
                assigned.add(id(p.module))
                broke = True
                break
        if not broke:
            for p in sorted(up_like, key=lambda p: p.name):
                if id(p.module) in assigned:
                    continue
                if info.up_proj is None:
                    info.up_proj = p
                elif info.gate_proj is None:
                    info.gate_proj = p
                assigned.add(id(p.module))
        if info.up_proj is None and info.gate_proj is None:
            return None
        return self._finalize_mlp(container, info, inter, used_ids, hidden)

    def _finalize_mlp(self, container: str, info: MLPInfo, inter: int,
                      used_ids: set[int], hidden: Optional[int]) -> Optional[MLPInfo]:
        if info.gate_up_fused:
            if info.up_proj is None or info.up_proj.out_features != 2 * inter:
                return None
        if info.down_proj is None or info.down_proj.in_features != inter:
            return None
        if info.up_proj is not None and info.up_proj.out_features not in (inter, 2 * inter):
            return None
        if info.gate_proj is not None and info.gate_proj.out_features != inter:
            return None
        info.intermediate_size = int(inter)
        info.intermediate_size = int(inter)
        info.kind = "gated" if (info.gate_proj is not None or info.gate_up_fused) \
            else "standard"
        info.hidden_coupled = bool(info.gate_up_fused)

        # structural safety: nothing parameterised between the projections that
        # changes the intermediate width, and no MoE-style siblings.
        unsafe = self._mlp_structural_risk(container, info, used_ids, hidden)
        if unsafe:
            info.status = "unknown"
            info.reason = unsafe
            info.safe = False
        else:
            info.status = "ok"
            info.safe = True
        return info

    def _mlp_structural_risk(self, container: str, info: MLPInfo,
                            used_ids: set[int], hidden: Optional[int]) -> Optional[str]:
        chosen = [p for p in (info.gate_proj, info.up_proj, info.down_proj)
                  if p is not None]
        chosen_ids = {id(p.module) for p in chosen}
        # multiple down-like projections in one container => MoE / expert bank
        down_like = [
            Projection.of(f"{container}.{n}", m)
            for n, m in info.module.named_modules()
            if n and is_projection(m)
            and id(m) not in chosen_ids
            and hidden is not None
            and Projection.of(f"{container}.{n}", m).out_features == hidden
            and Projection.of(f"{container}.{n}", m).in_features == info.intermediate_size
        ]
        if down_like:
            return ("multiple FFN down-projections share a container "
                    "(mixture-of-experts or split FFN); not safely pruneable")
        for rel, mod in info.module.named_modules():
            if not rel or id(mod) in chosen_ids or id(mod) in used_ids:
                continue
            if _is_norm(mod, rel):
                continue
            names = {n for n, _ in mod.named_parameters(recurse=False)}
            if not names:
                continue
            if is_projection(mod):
                p = Projection.of(rel, mod)
                if id(mod) in chosen_ids or id(mod) in used_ids:
                    continue
                if hidden is not None and info.intermediate_size and (
                    p.in_features == info.intermediate_size
                    or p.out_features == info.intermediate_size
                    or p.out_features == 2 * info.intermediate_size
                ):
                    return (f"unexpected projection '{rel}' inside FFN chain "
                            f"({p.in_features}->{p.out_features})")
                continue
            return f"parameterised module '{rel}' ({type(mod).__name__}) inside FFN chain"
        return None

    def _mlp_score(self, container: str, info: MLPInfo,
                   group: list[Projection]) -> int:
        score = 0
        score += 2 * container.count(".")
        toks = tokenize_name(container)
        if toks & MLP_CONTAINER_TOK:
            score += 10
        if info.status == "ok":
            score += 8
        if info.kind == "gated":
            score += 2
        score -= 3 * max(0, len(group) - (3 if info.gate_proj else 2))
        return score

    # ---------------------------------------------------------------- #
    def _residual_evidence(self, block: nn.Module) -> str:
        """Look for a residual add in the block's forward or, for blocks that
        delegate (e.g. BERT's ``*Output`` sub-modules), in descendant forwards."""
        seen_types = set()
        for mod in block.modules():
            t = type(mod)
            if t in seen_types or t is nn.Sequential or t is nn.ModuleList:
                continue
            seen_types.add(t)
            try:
                src = inspect.getsource(t.forward)
            except (OSError, TypeError):
                continue
            m = RESIDUAL_RE.search(src)
            if m:
                start = max(0, m.start() - 24)
                where = "block" if mod is block else t.__name__
                return f"forward source ({where}): {src[start:m.end() + 16].strip()!r}"
        return ""

    # ------------------------------------------------------------------ #
    def _find_embeddings(self) -> list[EmbeddingInfo]:
        out = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Embedding):
                out.append(EmbeddingInfo(name, mod, mod.num_embeddings,
                                         mod.embedding_dim))
        return out

    def _find_output_heads(self) -> list[OutputHeadInfo]:
        out = []
        for name, mod in self.model.named_modules():
            if not is_projection(mod):
                continue
            try:
                p = Projection.of(name, mod)
            except TypeError:
                continue
            if self.vocab and p.out_features == int(self.vocab) \
                    and p.in_features != p.out_features:
                out.append(OutputHeadInfo(name, mod, p.out_features, p.in_features))
        return out

    def _find_tied_parameters(self) -> list[TiedGroup]:
        buckets: dict[int, list[str]] = {}
        shapes: dict[int, tuple] = {}
        for name, param in self.model.named_parameters(remove_duplicate=False):
            key = id(param)
            buckets.setdefault(key, []).append(name)
            shapes[key] = tuple(param.shape)
        return [TiedGroup(sorted(names), shapes[k])
                for k, names in buckets.items() if len(names) > 1]

    # ------------------------------------------------------------------ #
    def _compute_capabilities(self, arch: ModelArchitecture) -> None:
        targets = arch.target_stacks()
        target_names = {s.name for s in targets}
        tl = [L for L in arch.layers if L.stack in target_names]

        mlp_ok = bool(tl) and all(
            L.mlp is not None and L.mlp.is_prunable() for L in tl
        )
        bad = [L.name for L in tl if not (L.mlp and L.mlp.is_prunable())]
        arch.cap_prune_mlp_neurons = Capability(
            SupportLevel.SUPPORTED if mlp_ok else (
                SupportLevel.UNKNOWN if tl else SupportLevel.UNSUPPORTED),
            "" if mlp_ok else (
                f"FFN not safely resolvable on {len(bad)} layer(s); first: "
                f"{bad[0]}" if bad else "no target layers"),
        )

        attn_ok = bool(tl) and all(
            L.attention is not None and L.attention.is_prunable() for L in tl
        )
        bad_a = [L.name for L in tl
                 if not (L.attention and L.attention.is_prunable())]
        arch.cap_prune_attention_heads = Capability(
            SupportLevel.SUPPORTED if attn_ok else SupportLevel.UNKNOWN,
            "" if attn_ok else (
                f"attention not safely resolvable on {len(bad_a)} layer(s); "
                f"first: {bad_a[0]}" if bad_a else "no target layers"),
        )

        kv_layers = [L for L in tl if L.attention and L.attention.k_proj
                     and L.attention.v_proj and L.attention.num_kv_heads]
        kv_ok = bool(tl) and len(kv_layers) == len(tl) and all(
            L.attention and L.attention.status == "ok" and not L.attention.is_cross_attention
            for L in tl
        )
        arch.cap_prune_kv_heads = Capability(
            SupportLevel.SUPPORTED if kv_ok else SupportLevel.UNKNOWN,
            "" if kv_ok else "KV projections not safely resolvable on all target layers",
        )

        stacks_ok = [s for s in targets if s.config_keys]
        layer_ok = bool(stacks_ok) and all(
            all(L.has_residual or True for L in arch.layers if L.stack == s.name)
            for s in stacks_ok
        )
        reasons = []
        if not stacks_ok:
            reasons.append("no layer-count config attribute could be resolved")
        missing_res = [L.name for L in tl if not L.has_residual]
        if missing_res:
            reasons.append(
                f"residual structure not evidenced on {len(missing_res)} layer(s) "
                f"(first: {missing_res[0]})"
            )
        arch.cap_prune_layers = Capability(
            SupportLevel.SUPPORTED if (stacks_ok and not missing_res)
            else SupportLevel.UNKNOWN,
            "; ".join(reasons),
        )

        arch.cap_fuse_affine = Capability(
            SupportLevel.UNKNOWN, "evaluated by the graph analyzer at plan time"
        )
        arch.cap_fuse_learned = Capability(
            SupportLevel.UNKNOWN, "requires calibration data; evaluated at plan time"
        )


def up_tokens(toks: set[str]) -> bool:
    return bool(toks & {"up", "fc1", "cfc", "h4h", "w3", "wi"} or "h_to_4h" in toks)
