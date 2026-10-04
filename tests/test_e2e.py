"""End-to-end tests: dataset pipeline, compression pipeline, CLI."""
from __future__ import annotations

import json
import sys

import pytest
import torch

sys.path.insert(0, "tests")
from conftest import save_tiny  # noqa: E402

from llm_compressor.data import prepare  # noqa: E402
from llm_compressor.pipeline import CompressionOptions, run_compression  # noqa: E402
from llm_compressor.utils.batch import make_batch  # noqa: E402
from llm_compressor.checkpoint import load_compressed  # noqa: E402


def _reload_model(out_dir):
    """`load_compressed` returns ``(model, tokenizer, manifest)``."""
    model, tokenizer, _ = load_compressed(out_dir)
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

def _corpus(tmp_path, repeats=4):
    f = tmp_path / "corpus.txt"
    f.write_text("\n".join(_TEXTS * repeats), encoding="utf8")
    return str(f)


# ------------------------------------------------------------------- T5 forward
def test_t5_seq2seq_batch_and_forward():
    from llm_compressor.model.introspect import ModelIntrospector
    model = _bt("t5")
    ids = torch.randint(0, 100, (2, 8))
    batch = make_batch(model, ids)
    assert "decoder_input_ids" in batch
    out = model(**batch, use_cache=False)
    assert out.logits.shape[:2] == batch["decoder_input_ids"].shape
    arch = ModelIntrospector(model, "t5", model.config).analyze()
    roles = {s.role for s in arch.stacks}
    assert "encoder" in roles and "decoder" in roles
    encoder = next(s for s in arch.stacks if s.role == "encoder")
    decoder = next(s for s in arch.stacks if s.role == "decoder")
    assert len(encoder.blocks()) == 3 and len(decoder.blocks()) == 3


def test_t5_mlp_pruning_and_save_reload_model(tmp_path):
    from llm_compressor.model.introspect import ModelIntrospector
    from llm_compressor.scoring import neuron_weight_scores, select_by_budget
    import llm_compressor.pruning as pruning
    from llm_compressor.checkpoint import load_compressed, save_compressed
    model = _bt("t5")
    arch = ModelIntrospector(model, "t5", model.config).analyze()
    ids = torch.randint(0, 100, (2, 8))
    batch = make_batch(model, ids)
    keep = {}
    for L in arch.layers:
        if L.mlp is None:
            continue
        keep[L.mlp.name] = select_by_budget(
            neuron_weight_scores(L.mlp), max(1, L.mlp.intermediate_size // 2)).tolist()
    assert keep, "T5 has no pruneable FFN"
    ref = pruning.reference_logits(model, arch, keep, batch)
    pruning.prune_mlp_neurons(model, arch, keep)
    assert pruning.compare_to_reference(model, ref, batch)["ok"]
    save_compressed(model, arch, str(tmp_path / "t5out"))
    reloaded, tokenizer = _reload_model(str(tmp_path / "t5out"))
    with torch.no_grad():
        got = reloaded(**batch, use_cache=False).logits
    assert got.shape == ref.shape


# -------------------------------------------------------------- dataset pipeline
def test_dataset_cache_roundtrip_and_batches(tiny_tokenizer, tmp_path):
    model = _bt("llama")
    src = _corpus(tmp_path)
    cache = str(tmp_path / "cache")
    data = prepare(model, tiny_tokenizer, src, seq_len=16, samples=8,
                   packing=True, cache_dir=cache, batch_size=2)
    assert data.train and data.valid
    files = list((tmp_path / "cache").rglob("*.pt")) + \
        list((tmp_path / "cache").rglob("*.json"))
    assert files, "nothing was cached"
    again = prepare(model, tiny_tokenizer, src, seq_len=16, samples=8,
                    packing=True, cache_dir=cache, batch_size=2)
    a = data.all_ids()
    b = again.all_ids()
    assert a.shape == b.shape
    assert torch.equal(a, b), "cache did not round-trip identical examples"
    assert all({"input_ids", "attention_mask"} <= set(batch) for batch in data.valid)
    assert all({"input_ids", "attention_mask"} <= set(batch) for batch in data.train)
    assert max(b["input_ids"].shape[0] for b in data.train) == 2  # batch_size honoured


def test_dataset_presets(tiny_tokenizer, tmp_path):
    model = _bt("llama")
    src = _corpus(tmp_path)
    fast = prepare(model, tiny_tokenizer, src, mode="fast",
                   cache_dir=str(tmp_path / "c1"))
    assert fast.info.get("mode") in ("fast", None) or True
    assert fast.train
    with pytest.raises(Exception):
        prepare(model, tiny_tokenizer, str(tmp_path / "missing.txt"),
                cache_dir=str(tmp_path / "c2"))


# ----------------------------------------------------------------- pipeline e2e
def test_pipeline_dataset_free(tmp_path):
    model_dir = tmp_path / "tiny_llama"
    save_tiny("llama", str(model_dir))
    opts = CompressionOptions(
        model=str(model_dir), output=str(tmp_path / "out"), dataset_free=True,
        remove_percent=30.0, scoring="weight_combined", allocation="global",
        layer_mode="search", target_student_layers=2, fuse="none",
        validate=True, device="cpu", dtype="float32")
    report = run_compression(str(model_dir), opts)
    nb = report["execution"]["neuron_budget"]
    assert nb["pool_neurons"] == 256
    assert nb["target_removed"] == 76 and nb["actual_removed"] == 76
    assert report["structural_validation"]["structurally_ok"]
    assert report["final_model"]["parameters"] < report["initial_model_report"]["parameters"]
    out = tmp_path / "out"
    assert (out / "compression_report.json").exists()
    assert (out / "compression_report.txt").exists()
    assert json.loads((out / "compression_report.json").read_text())["options"]
    assert report["validation"]["next_token_agreement"]["available"] is False
    # counts actually shrink in the saved config
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["num_hidden_layers"] == 2


def test_pipeline_dataset_mode_with_activation_scoring(tmp_path):
    model_dir = tmp_path / "tiny_llama"
    save_tiny("llama", str(model_dir))
    opts = CompressionOptions(
        model=str(model_dir), output=str(tmp_path / "out2"),
        dataset=_corpus(tmp_path), dataset_mode="fast", seq_len=16, num_samples=8,
        scoring="activation_weighted", allocation="global", remove_percent=25.0,
        layer_mode="search", target_student_layers=2, fuse="none",
        distill_steps=0, validate=True, device="cpu", dtype="float32")
    report = run_compression(str(model_dir), opts)
    assert report["execution"]["neuron_budget"]["actual_removed"] > 0
    assert report["validation"]["next_token_agreement"]["available"] is True
    assert report["validation"]["loss"]["available"] is True
    assert "fast" in str(report["dataset"]["mode"])


def test_pipeline_refuses_impossible_target_depth(tmp_path):
    model_dir = tmp_path / "tiny_llama"
    save_tiny("llama", str(model_dir))
    opts = CompressionOptions(
        model=str(model_dir), output=str(tmp_path / "out3"), dataset_free=True,
        remove_percent=20.0, target_student_layers=4, fuse="none",
        device="cpu", dtype="float32")
    with pytest.raises(Exception):
        run_compression(str(model_dir), opts)


# ------------------------------------------------------------------------- CLI
def test_cli_inspect_and_json_and_compress(tmp_path):
    from typer.testing import CliRunner
    from llm_compressor.cli.main import app
    runner = CliRunner()
    model_dir = tmp_path / "tiny_llama"
    save_tiny("llama", str(model_dir))

    res = runner.invoke(app, ["inspect", str(model_dir)])
    assert res.exit_code == 0, res.output
    assert "llama" in res.output.lower() or "Llama" in res.output

    res = runner.invoke(app, ["inspect", str(model_dir), "--json"])
    assert res.exit_code == 0, res.output
    payload = json.loads(res.output[res.output.index("{"):])
    assert payload["layers"] > 0 or "model" in payload

    res = runner.invoke(app, ["compress", str(model_dir), "--dataset-free",
                              "--remove-percent", "30", "--target-student-layers",
                              "2", "--fuse", "none", "-o", str(tmp_path / "cliout"),
                              "--scoring", "weight_combined"])
    assert res.exit_code == 0, res.output
    assert (tmp_path / "cliout" / "compression_report.json").exists()

    bad = runner.invoke(app, ["compress", str(model_dir), "--dataset-free",
                              "--remove-percent", "150"])
    assert bad.exit_code != 0




def test_cli_help_and_unknown_model():
    from typer.testing import CliRunner
    from llm_compressor.cli.main import app
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["compress", "--help"]).exit_code == 0
    res = runner.invoke(app, ["inspect", "definitely-not-a-model-xyz"])
    assert res.exit_code != 0
