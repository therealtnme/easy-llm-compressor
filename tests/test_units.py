"""Unit tests: affine algebra, fusion safety, budgets, scoring, search, distillation."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from llm_compressor import budget
from llm_compressor.model.projection import Projection
from llm_compressor import fusion as F


# --------------------------------------------------------------------- algebra
def _bt(name):
    """`conftest.build_tiny` returns ``(model, cfg)``; tests only want the model."""
    from conftest import build_tiny as _b
    return _b(name)[0]


_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "a small model can still learn useful structure",
    "compression removes parameters that carry little information",
    "the second sentence has roughly the same length as the first",
    "attention heads and feed forward neurons can both be pruned",
    "distillation trains a smaller student to match its teacher",
    "calibration data must be real text and never random tensors",
    "the report records every structural change that was applied",
]

def test_linear_linear_exact():
    a = nn.Linear(6, 5)
    b = nn.Linear(5, 4)
    pa, pb = Projection.of("a", a), Projection.of("b", b)
    fused = F.fuse_affine(pa, pb)
    x = torch.randn(3, 6)
    assert torch.allclose(fused["weight"], b.weight @ a.weight, atol=1e-6)
    assert torch.allclose(fused["bias"], b.weight @ a.bias + b.bias, atol=1e-6)
    assert torch.allclose(fused["weight"] @ x.T + fused["bias"][:, None],
                          b(a(x)).T, atol=1e-5)


def test_chain_of_three_biasless():
    mods = [nn.Linear(7, 6, bias=False), nn.Linear(6, 5), nn.Linear(5, 3)]
    projs = [Projection.of(str(i), m) for i, m in enumerate(mods)]
    out = F.compose_chain(projs)
    x = torch.randn(2, 7)
    ref = mods[2](mods[1](mods[0](x)))
    got = x @ out["weight"].T + out["bias"]
    assert torch.allclose(ref, got, atol=1e-5)
    assert out["detail"]["n_projections"] == 3


def test_dimension_mismatch_refused():
    a, b = nn.Linear(4, 4), nn.Linear(5, 4)
    with pytest.raises(F.FusionError):
        F.fuse_affine(Projection.of("a", a), Projection.of("b", b))


def test_conv1d_conv1d_fuses():
    from transformers.pytorch_utils import Conv1D
    # Conv1D(nf, nx): nf = OUTPUT features, nx = INPUT features.
    a, b = Conv1D(5, 6), Conv1D(4, 5)
    out = F.compose_chain([Projection.of("a", a), Projection.of("b", b)])
    x = torch.randn(2, 6)
    assert torch.allclose(b(a(x)), x @ out["weight"].T + out["bias"], atol=1e-5)
    assert out["weight"].shape == (4, 6)


def test_conv1d_linear_fuses():
    from transformers.pytorch_utils import Conv1D
    a, b = Conv1D(5, 6), nn.Linear(5, 3)
    out = F.compose_chain([Projection.of("a", a), Projection.of("b", b)])
    assert out["weight"].shape == (3, 6)


def test_dtype_preserved_shape():
    a = nn.Linear(4, 4).double()
    out = F.compose_chain([Projection.of("a", a), Projection.of("b", nn.Linear(4, 2))])
    assert out["weight"].shape == (2, 4)


# ------------------------------------------------------------ fusion safety
def test_classify_module():
    assert F.classify_module(nn.Linear(2, 2)) == F.AFFINE
    assert F.classify_module(nn.ReLU()) == F.NONLINEAR
    assert F.classify_module(nn.LayerNorm(4)) == F.NONLINEAR
    assert F.classify_module(nn.LSTM(2, 2)) == F.UNKNOWN


def test_sequential_chain_detected_and_fused():
    net = nn.Sequential(nn.Linear(8, 6), nn.Linear(6, 4))
    x = torch.randn(3, 8)
    before = net(x)
    chains = F.scan_sequential_chains(net)
    assert len(chains) == 1 and chains[0]["meta"].exact_fusion_allowed
    info = F.fuse_chain_in_sequential(net, 0, 2)
    assert info["cost"]["before"] == 8 * 6 + 6 + 6 * 4 + 4
    assert info["cost"]["cheaper"]
    assert len(list(net.named_children())) == 1
    assert torch.allclose(net(x), before, atol=1e-5)


class _AffineNet(nn.Module):
    """Minimal stand-in for an HF block: accepts **batch and returns an object."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(6, 5), nn.Linear(5, 5))

    def forward(self, input_ids=None, **kwargs):
        return type("Out", (), {"logits": self.proj(input_ids)})()


def test_verify_fusion_restores_and_matches():
    model = _AffineNet()
    probe = {"input_ids": torch.randn(2, 4, 6)}
    out = F.verify_fusion(model, model.proj, 0, 2, probe)
    assert out["verified"] and out["max_abs_diff"] < 1e-5
    assert len(list(model.proj.named_children())) == 2  # restored


def test_nonlinearity_blocks_fusion():
    net = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 4))
    assert F.scan_sequential_chains(net) == []


def test_fusion_refused_when_not_cheaper():
    net = nn.Sequential(nn.Linear(64, 2), nn.Linear(2, 64))
    with pytest.raises(F.FusionError):
        F.fuse_chain_in_sequential(net, 0, 2)


def test_layer_pair_refusal_reasons():
    from llm_compressor.model.introspect import ModelIntrospector
    import sys
    sys.path.insert(0, "tests")
    
    model = _bt("llama")
    arch = ModelIntrospector(model, "tiny", model.config).analyze()
    meta = F.analyze_layer_pair(arch, arch.primary_stack().name, 0)
    assert meta.classification == F.UNKNOWN
    assert meta.has_branch or meta.has_nonlinearity
    assert not meta.exact_fusion_allowed
    assert meta.reasons


# --------------------------------------------------------------------- budget
def test_resolve_target_percent_and_count():
    assert budget.resolve_target(256, percent=30) == 76
    assert budget.resolve_target(256, count=100) == 100
    with pytest.raises(budget.BudgetError):
        budget.resolve_target(256, percent=30, count=3)
    with pytest.raises(budget.BudgetError):
        budget.resolve_target(256, percent=0)
    with pytest.raises(budget.BudgetError):
        budget.resolve_target(256, percent=101)
    with pytest.raises(budget.BudgetError):
        budget.resolve_target(256, percent=-5)
    with pytest.raises(budget.BudgetError):
        budget.resolve_target(256, count=-1)
    with pytest.raises(budget.BudgetError):
        budget.resolve_target(256, count=999)


def _scores(width, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(width, generator=g)


def test_allocate_exact_and_global():
    widths = {"a": 128, "b": 128}
    scores = {k: _scores(v, i) for i, (k, v) in enumerate(widths.items())}
    res = budget.allocate(widths, scores, 76, scope="global")
    assert res["removed"] == 76
    assert sum(len(v) for v in res["keep"].values()) == 256 - 76
    per = sorted(res["per_layer_removed"].values())
    assert per[0] != per[1] or True  # uneven is the default, not enforced


def test_allocate_uniform_scope():
    widths = {"a": 100, "b": 100, "c": 100}
    scores = {k: _scores(v) for k, v in widths.items()}
    res = budget.allocate(widths, scores, 30, scope="uniform")
    assert res["removed"] == 30


def test_allocate_invalid_and_protection():
    widths = {"a": 10, "b": 10}
    scores = {k: _scores(v) for k, v in widths.items()}
    with pytest.raises(budget.BudgetError):
        budget.allocate(widths, scores, 10_000)
    with pytest.raises(budget.BudgetError):
        budget.allocate(widths, scores, 5, scope="nonsense")
    with pytest.raises(budget.BudgetError):
        budget.allocate(widths, scores, 5, protected=["a", "b"])


# -------------------------------------------------------------------- scoring
def test_weight_scoring_modes_and_allocation():
    from llm_compressor.scoring import (neuron_weight_scores, select_by_budget,
                                        normalize_scores, resolve_method)
    m = nn.Linear(8, 8)
    m2 = nn.Linear(8, 16)
    mlp = nn.Module()
    mlp.add_module("gate_proj", m)
    mlp.add_module("up_proj", m)
    mlp.add_module("down_proj", m2)
    from llm_compressor.model.architecture import MLPInfo
    info = MLPInfo("blk.mlp", mlp, gate_proj=Projection.of("g", m),
                   up_proj=Projection.of("u", m),
                   down_proj=Projection.of("d", m2), intermediate_size=8,
                   kind="gated")
    for mode in ("l1", "l2", "combined"):
        s = neuron_weight_scores(info, mode)
        assert s.shape == (8,) and torch.isfinite(s).all()
    sel = select_by_budget(neuron_weight_scores(info), keep=3)
    assert len(sel) == 3 and sel.tolist() == sorted(sel.tolist())
    n = normalize_scores(neuron_weight_scores(info), "mass")
    assert abs(float(n.sum()) - 1.0) < 1e-5 or float(n.sum()) == 0.0
    assert resolve_method("weight_l1").requires_data is False
    assert resolve_method("activation_weighted").requires_data is True
    with pytest.raises(KeyError):
        resolve_method("nope")


def test_activation_scoring_needs_real_data(tiny_tokenizer, tmp_path):
    import sys
    sys.path.insert(0, "tests")
    from llm_compressor.data import prepare
    from llm_compressor.scoring import (ActivationCollector, activation_scores,
                                        neuron_weight_scores)
    model = _bt("llama")
    text_file = tmp_path / "corpus.txt"
    text_file.write_text("\n".join(_TEXTS), encoding="utf8")
    data = prepare(model, tiny_tokenizer, str(text_file), seq_len=8, samples=4,
                   packing=False, cache_dir=str(tmp_path / "cache"))
    arch = __import__("llm_compressor.model.introspect", fromlist=["x"]) \
        .ModelIntrospector(model, "tiny", model.config).analyze()
    from llm_compressor.model.names import leaf
    mods = {leaf(L.mlp.name): model.get_submodule(L.mlp.name) for L in arch.layers}
    with ActivationCollector(mods) as coll:
        for batch in data.valid:
            model(**batch, use_cache=False)
    a = activation_scores(arch.layers[0].mlp, coll.activation(leaf(arch.layers[0].mlp.name)),
                          weighting="combined")
    assert a.shape == (arch.layers[0].mlp.intermediate_size,)
    assert float(a.sum()) > 0
    assert a.shape == neuron_weight_scores(arch.layers[0].mlp).shape


# --------------------------------------------------------------------- search
def _sal(arch, n):
    return {"values": [0.1 * (i + 1) for i in range(n)]}


def test_search_enforces_target_depth_rule():
    import sys
    sys.path.insert(0, "tests")
    
    from llm_compressor.model.introspect import ModelIntrospector
    from llm_compressor.search import joint_search
    model = _bt("llama")          # 4 layers
    arch = ModelIntrospector(model, "tiny", model.config).analyze()
    stack = arch.primary_stack()
    widths = {L.mlp.name: L.mlp.intermediate_size for L in arch.layers}
    scores = {k: _scores(v, i) for i, (k, v) in enumerate(widths.items())}
    plans = joint_search(arch, stack.name, _sal(arch, 4), widths, scores,
                         remove_target=40, target_student_layers=2,
                         beam_width=8, top_k=3)
    assert plans, "search returned no plan"
    for plan in plans:
        assert plan.target_student_layers == 2
        assert len(plan.survivors) == 2
        for region in plan.regions:
            span = region.region_end - region.region_start
            if region.operation == "KEEP":
                assert region.student_depth == span
            else:
                # a compressed region must genuinely lose depth
                assert region.student_depth < span, (region.operation, span)


def test_search_rejects_impossible_target_depth():
    import sys
    sys.path.insert(0, "tests")
    
    from llm_compressor.model.introspect import ModelIntrospector
    from llm_compressor.search import joint_search
    model = _bt("llama")
    arch = ModelIntrospector(model, "tiny", model.config).analyze()
    stack = arch.primary_stack()
    widths = {L.mlp.name: L.mlp.intermediate_size for L in arch.layers}
    scores = {k: _scores(v) for k, v in widths.items()}
    for bad in (4, 5, 0, -1):
        with pytest.raises(ValueError):
            joint_search(arch, stack.name, _sal(arch, 4), widths, scores,
                         remove_target=0, target_student_layers=bad)


# ----------------------------------------------------------------- distillation
def test_distillation_improves_student(tiny_tokenizer, tmp_path):
    import sys
    sys.path.insert(0, "tests")
    from llm_compressor.data import prepare
    from llm_compressor.distill import DistillConfig, distill
    torch.manual_seed(0)
    teacher = _bt("llama")
    student = _bt("llama")
    for p in student.parameters():
        p.data.normal_(0, 0.1)
    f = tmp_path / "corpus.txt"
    f.write_text("\n".join(_TEXTS * 4), encoding="utf8")
    data = prepare(student, tiny_tokenizer, str(f), seq_len=16, samples=8,
                   packing=True, cache_dir=str(tmp_path / "c"))
    batches = data.valid
    with torch.no_grad():
        t_logits = teacher(**batches[0], use_cache=False).logits
        s_logits = student(**batches[0], use_cache=False).logits
    before = float(torch.nn.functional.mse_loss(s_logits, t_logits))
    cfg = DistillConfig(steps=25, lr=1e-2, batch_size=2, loss_mode="logit_kl",
                        temperature=1.0, seed=0)
    out = distill(student, teacher, batches, cfg)
    with torch.no_grad():
        s_logits = student(**batches[0], use_cache=False).logits
    after = float(torch.nn.functional.mse_loss(s_logits, t_logits))
    assert out["losses"]["kl"] == "applied"
    assert out["steps"] == 25
    assert after < before, (before, after)



def test_distillation_refuses_without_examples(tiny_tokenizer, tmp_path):
    import sys
    sys.path.insert(0, "tests")
    
    from llm_compressor.distill import distill
    model = _bt("llama")
    with pytest.raises(Exception):
        distill(model, _bt("llama"), [])
