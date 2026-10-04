"""Model-level tests: introspection, physical pruning, layer deletion, save/reload."""
from __future__ import annotations

import pytest
import torch

from llm_compressor.model.introspect import ModelIntrospector
from llm_compressor.model.protection import (ProtectionPolicy,
                                             identify_protected_components)
import llm_compressor.pruning as pruning
from llm_compressor.utils.batch import make_batch, model_kind, output_tensor
from llm_compressor.checkpoint import load_compressed, save_compressed


def _reload_model(out_dir):
    """`load_compressed` returns ``(model, tokenizer, manifest)``."""
    model, tokenizer, manifest = load_compressed(out_dir)
    return model, tokenizer

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

FAMILIES = ["llama", "qwen2", "mistral", "gpt2", "opt", "bert", "t5"]

EXPECTED_NEURONS = {"llama": 4, "qwen2": 3, "mistral": 3, "gpt2": 3,
                    "opt": 3, "bert": 3, "t5": 3}


def arch_of(model, name="tiny"):
    return ModelIntrospector(model, name, model.config).analyze()


@pytest.mark.parametrize("family", FAMILIES)
def test_introspection(family):
    import sys
    sys.path.insert(0, "tests")
    
    model = _bt(family)
    arch = arch_of(model, family)
    stacks = {s.role for s in arch.stacks}
    if family == "t5":
        assert {"encoder", "decoder"} <= stacks
        assert len(arch.stacks) >= 2
    else:
        assert stacks
    assert arch.total_parameters > 0
    assert arch.num_layers == EXPECTED_NEURONS[family] or family == "t5"
    mlps = [L.mlp for L in arch.layers if L.mlp is not None]
    assert mlps, f"{family}: no FFN discovered"
    assert all(m.intermediate_size > 0 for m in mlps)
    attns = [L.attention for L in arch.layers if L.attention is not None]
    assert attns, f"{family}: no attention discovered"
    if family != "gpt2":            # GPT-2 fuses QKV; head count is UNKNOWN
        assert all(a.num_heads and a.num_heads > 0 for a in attns)
    assert arch.hidden_size > 0
    # protection: embeddings + norm must be protected by default
    cats = {p.category for p in arch.protected}
    assert "embeddings" in cats
    assert arch.cap_inspect.level.name == "SUPPORTED"
    assert arch.summarize()["layers"] > 0


def test_gpt2_fused_qkv_is_not_head_prunable():
    import sys
    sys.path.insert(0, "tests")
    
    arch = arch_of(_bt("gpt2"), "gpt2")
    attns = [L.attention for L in arch.layers if L.attention is not None]
    assert attns and all(a.is_fused_qkv for a in attns)
    assert all(not a.is_prunable() for a in attns)
    assert arch.cap_prune_attention_heads.level.name in ("UNKNOWN", "UNSUPPORTED")
    assert arch.cap_prune_attention_heads.reason


def test_tied_weights_detected():
    import sys
    sys.path.insert(0, "tests")
    
    arch = arch_of(_bt("qwen2"), "qwen2")     # tie_word_embeddings=True
    assert arch.tied_groups
    assert any(p.category == "tied_weights" for p in arch.protected)


# --------------------------------------------------------------------- pruning
@pytest.mark.parametrize("family", ["llama", "qwen2", "gpt2", "opt", "bert"])
def test_mlp_pruning_matches_zero_masked_reference(family):
    import sys
    sys.path.insert(0, "tests")
    
    from llm_compressor.scoring import neuron_weight_scores, select_by_budget
    model = _bt(family)
    arch = arch_of(model, family)
    ids = torch.randint(0, 100, (2, 8))
    ids[:, -1] = 1
    batch = make_batch(model, ids)
    keep = {}
    for L in arch.layers:
        if L.mlp is None:
            continue
        scores = neuron_weight_scores(L.mlp)
        n_keep = max(1, int(L.mlp.intermediate_size * 0.7))
        keep[L.mlp.name] = select_by_budget(scores, n_keep).tolist()
    ref = pruning.reference_logits(model, arch, keep, batch)
    before = sum(p.numel() for p in model.parameters())
    res = pruning.prune_mlp_neurons(model, arch, keep)
    after = sum(p.numel() for p in model.parameters())
    assert after < before and res.removed_neurons > 0
    cmp = pruning.compare_to_reference(model, ref, batch)
    assert cmp["ok"], cmp
    widths = {L.mlp.name: L.mlp.intermediate_size for L in arch.layers
              if L.mlp is not None}
    assert all(w == len(keep[k]) for k, w in widths.items())
    out = model(**batch, use_cache=False)
    assert output_tensor(out).shape[:2] == ids.shape


def test_mlp_pruning_refuses_all_and_out_of_range():
    import sys
    sys.path.insert(0, "tests")
    
    model = _bt("llama")
    arch = arch_of(model, "llama")
    path = arch.layers[0].mlp.name
    with pytest.raises(pruning.RewriteError):
        pruning.prune_mlp_neurons(model, arch, {path: []})
    with pytest.raises(pruning.RewriteError):
        pruning.prune_mlp_neurons(model, arch, {path: [10 ** 6]})
    with pytest.raises(pruning.RewriteError):
        pruning.prune_mlp_neurons(model, arch, {"nope.mlp": [0]})


def test_attention_head_pruning_llama_and_kv():
    import sys
    sys.path.insert(0, "tests")
    
    model = _bt("qwen2")        # 4 heads, 2 kv heads -> 2 groups
    arch = arch_of(model, "qwen2")
    attn = arch.layers[0].attention
    assert attn.num_heads == 4 and attn.num_kv_heads == 2
    keep_q = {attn.name: [0, 1, 2, 3]}
    keep_kv = {attn.name: [0, 1]}
    ids = torch.randint(0, 100, (1, 6))
    batch = make_batch(model, ids)
    q_before = arch.layers[0].attention.q_proj.out_features
    v_before = arch.layers[0].attention.v_proj.out_features
    # drop one KV group -> that group's query heads go too (whole groups only)
    res = pruning.prune_attention_heads(
        model, arch, {attn.name: [0, 1]}, {attn.name: [0]})
    assert res.removed_heads == 2 and res.removed_kv_heads == 1
    assert arch.layers[0].attention.q_proj.out_features < q_before
    assert arch.layers[0].attention.v_proj.out_features < v_before
    assert output_tensor(model(**batch, use_cache=False)).shape[1] == 6


def test_attention_pruning_refuses_partial_kv_group():
    import sys
    sys.path.insert(0, "tests")
    
    model = _bt("qwen2")
    arch = arch_of(model, "qwen2")
    a = arch.layers[0].attention
    with pytest.raises(pruning.RewriteError):
        pruning.prune_attention_heads(model, arch, {a.name: [0, 3]},
                                      {a.name: [0, 1]})
    with pytest.raises(pruning.RewriteError):
        pruning.prune_attention_heads(model, arch, {a.name: []})


# --------------------------------------------------------------- layer deletion
@pytest.mark.parametrize("family,stack_role", [("llama", "decoder"), ("bert", "encoder")])
def test_layer_deletion_updates_stack_and_config(family, stack_role):
    import sys
    sys.path.insert(0, "tests")
    
    model = _bt(family)
    arch = arch_of(model, family)
    stack = next(s for s in arch.stacks if s.role == stack_role)
    n_before = len(stack.blocks())
    res = pruning.delete_layers(model, arch, stack.name, [0])
    assert res.removed_layers == 1
    assert len(stack.module) == n_before - 1
    depth = getattr(model.config, "num_hidden_layers",
                    getattr(model.config, "num_layers", None))
    if depth is not None:
        assert depth == n_before - 1
    ids = torch.randint(0, 100, (2, 6))
    batch = make_batch(model, ids)
    assert output_tensor(model(**batch, use_cache=False)).shape[1] == 6
    # layer indices must be renumbered
    idxs = sorted(getattr(sub, "layer_idx") for blk in stack.module
                  for sub in blk.modules() if hasattr(sub, "layer_idx"))
    assert idxs and idxs == list(range(len(stack.module)))


def test_layer_deletion_refuses_shared_modules():
    import sys
    sys.path.insert(0, "tests")
    
    model = _bt("llama")
    arch = arch_of(model, "llama")
    stack = arch.primary_stack()
    stack.module[1] = stack.module[0]        # share the instance
    with pytest.raises(pruning.RewriteError):
        pruning.delete_layers(model, arch, stack.name, [0, 1])




# ------------------------------------------------------------------ save/reload
def test_save_and_reload_pruned_checkpoint(tmp_path):
    import sys
    sys.path.insert(0, "tests")
    
    from llm_compressor.scoring import neuron_weight_scores, select_by_budget
    model = _bt("llama")
    arch = arch_of(model, "llama")
    keep = {L.mlp.name: select_by_budget(neuron_weight_scores(L.mlp),
                                         max(1, int(L.mlp.intermediate_size * 0.5))
                                         ).tolist()
            for L in arch.layers if L.mlp is not None}
    pruning.prune_mlp_neurons(model, arch, keep)
    pruning.delete_layers(model, arch, arch.primary_stack().name, [0])
    ids = torch.randint(0, 100, (2, 8))
    batch = make_batch(model, ids)
    with torch.no_grad():
        expected = model(**batch, use_cache=False).logits
    save_compressed(model, arch, str(tmp_path / "out"))
    reloaded, tokenizer = _reload_model(str(tmp_path / "out"))
    got = reloaded(**batch, use_cache=False).logits
    assert got.shape == expected.shape
    assert torch.allclose(expected, got, atol=1e-5)
    widths = [m.mlp.up_proj.out_features for m in reloaded.model.layers]
    assert widths == [m.mlp.up_proj.out_features for m in model.model.layers]


# ------------------------------------------- T5 encoder/decoder layer coupling
@pytest.mark.parametrize("stack_role,sibling_role", [("decoder", "encoder"),
                                                     ("encoder", "decoder")])
def test_t5_layer_deletion_does_not_resize_sibling_stack(stack_role, sibling_role):
    """Deleting a layer from one T5 stack must not resize the other one.

    Encoder and decoder depth use different config keys (``num_layers`` vs
    ``num_decoder_layers``); a generic sync that touches the sibling's key
    rebuilds a skeleton with the wrong number of blocks and the saved manifest
    can no longer be filled. Index 1 is used because block 0 is not deletable.
    """
    import sys
    sys.path.insert(0, "tests")

    model = _bt("t5")
    arch = arch_of(model, "t5")
    stack = next(s for s in arch.stacks if s.role == stack_role)
    sibling = next(s for s in arch.stacks if s.role == sibling_role)
    sibling_depth = len(sibling.blocks())

    res = pruning.delete_layers(model, arch, stack.name, [1])
    assert res.removed_layers == 1
    assert len(stack.module) == sibling_depth - 1
    assert len(sibling.module) == sibling_depth, "sibling stack was resized"

    ids = torch.randint(0, 100, (2, 6))
    batch = make_batch(model, ids)
    assert output_tensor(model(**batch, use_cache=False)).shape[0] == 2

    # every stack's config depth must agree with its module list, otherwise a
    # reload would rebuild the wrong skeleton
    for s in arch.stacks:
        for key in s.config_keys:
            assert getattr(model.config, key.rsplit(".", 1)[-1]) == len(s.module)


def test_t5_first_block_is_refused_not_corrupted():
    """T5 block 0 owns the relative-attention bias shared by the whole stack.

    Deleting it cannot be expressed in the config, so it must be refused up
    front (UNSUPPORTED) rather than turned into an unloadable checkpoint.
    """
    import sys
    sys.path.insert(0, "tests")

    model = _bt("t5")
    arch = arch_of(model, "t5")
    decoder = next(s for s in arch.stacks if s.role == "decoder")
    assert 0 in pruning.unsafe_layer_indices(arch, decoder)
    with pytest.raises(pruning.RewriteError, match="no surviving block provides"):
        pruning.delete_layers(model, arch, decoder.name, [0])
    # the refusal must not have mutated the model
    assert len(decoder.module) == len(model.decoder.block)
    assert 1 not in pruning.unsafe_layer_indices(arch, decoder)


def test_t5_layer_deletion_survives_save_and_reload(tmp_path):
    import sys
    sys.path.insert(0, "tests")

    model = _bt("t5")
    arch = arch_of(model, "t5")
    stack = next(s for s in arch.stacks if s.role == "decoder")
    pruning.delete_layers(model, arch, stack.name, [1])

    ids = torch.randint(0, 100, (2, 6))
    batch = make_batch(model, ids)
    with torch.no_grad():
        expected = output_tensor(model(**batch, use_cache=False))
    save_compressed(model, arch, str(tmp_path / "t5out"))
    reloaded, tokenizer = _reload_model(str(tmp_path / "t5out"))
    got = output_tensor(reloaded(**batch, use_cache=False))
    assert got.shape == expected.shape
    assert torch.allclose(expected, got, atol=1e-5)


def test_layer_deletion_refuses_shared_depth_key():
    """If a depth key is shared between stacks, deletion is refused up front."""
    import sys
    sys.path.insert(0, "tests")

    model = _bt("t5")
    arch = arch_of(model, "t5")
    decoder = next(s for s in arch.stacks if s.role == "decoder")
    encoder = next(s for s in arch.stacks if s.role == "encoder")
    encoder.config_keys = list(decoder.config_keys)   # pretend the key is shared
    with pytest.raises(pruning.RewriteError, match="not supported"):
        pruning.delete_layers(model, arch, decoder.name, [1])
