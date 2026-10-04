"""Dataset layout support: auto-detection, chat rendering, and the ask path."""

from __future__ import annotations

import json
import sys
import types

import pytest

from llm_compressor.data import DataError, read_texts
from llm_compressor.data.formats import (FormatError, detect_format,
                                         format_from_answer, looks_like_chat,
                                         render_value, resolve_format)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("columns, expected_kind, expected_columns", [
    (["instruction", "input", "output"], "template",
     ["instruction", "input", "output"]),
    (["input", "output"], "template", ["input", "output"]),
    (["prompt", "response"], "template", ["prompt", "response"]),
    (["question", "answer"], "template", ["question", "answer"]),
    (["in", "out"], "template", ["in", "out"]),
    (["instruction", "output"], "template", ["instruction", "output"]),
    (["source", "target"], "template", ["source", "target"]),
    (["messages"], "chat", ["messages"]),
    (["conversations"], "chat", ["conversations"]),
    (["conversation"], "chat", ["conversation"]),
    (["text"], "column", ["text"]),
    (["content"], "column", ["content"]),
])
def test_detect_format_recognises_common_schemas(columns, expected_kind,
                                                 expected_columns):
    fmt = detect_format(columns)
    assert fmt is not None, columns
    assert fmt.kind == expected_kind
    assert list(fmt.columns) == expected_columns
    assert fmt.confidence in ("high", "medium")


def test_detect_format_ignores_metadata_columns():
    fmt = detect_format(["id", "text"])
    assert fmt is not None and fmt.kind == "column"
    assert fmt.columns == ("text",)


def test_detect_format_prefers_chat_column():
    fmt = detect_format(["id", "messages", "text"])
    assert fmt is not None and fmt.kind == "chat" and fmt.columns == ("messages",)


def test_detect_format_returns_none_when_hopeless():
    assert detect_format(["id", "idx"], sample={"id": 1, "idx": 2}) is None


def test_detect_format_guesses_unknown_pair_with_low_confidence():
    fmt = detect_format(["foo", "bar"], sample={"foo": "a", "bar": "b"})
    assert fmt is not None
    assert fmt.confidence == "low"
    assert list(fmt.columns) == ["foo", "bar"]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def test_template_render_joins_and_skips_empty_slots():
    fmt = detect_format(["instruction", "input", "output"])
    text = fmt.render({"instruction": "Add.", "input": "", "output": "3"})
    assert text == "Add.\n\n3"
    text = fmt.render({"instruction": "Add.", "input": "1+2", "output": "3"})
    assert text == "Add.\n\n1+2\n\n3"


def test_pair_render_uses_both_columns():
    fmt = detect_format(["in", "out"])
    assert fmt.render({"in": "hello", "out": "world"}) == "hello\n\nworld"


def test_chat_render_role_content_and_sharegpt():
    messages = [{"role": "system", "content": "be nice"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"}]
    text = render_value(messages)
    assert text == "System: be nice\nUser: hi\nAssistant: hello"
    sharegpt = [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]
    assert render_value(sharegpt) == "User: hi\nAssistant: hello"


def test_chat_render_from_json_string_and_nested_dict():
    raw = json.dumps([{"role": "user", "content": "hi"},
                      {"role": "assistant", "content": "yo"}])
    assert render_value(raw) == "User: hi\nAssistant: yo"
    assert looks_like_chat({"conversations": raw})
    fmt = detect_format(["conversations"])
    assert fmt.render({"conversations": [{"role": "user", "content": "q"}]}) == "User: q"


def test_local_jsonl_detects_pair(tmp_path):
    path = tmp_path / "cal.jsonl"
    path.write_text("\n".join(json.dumps({"in": f"q{i}", "out": f"a{i}"})
                              for i in range(4)), encoding="utf8")
    texts = read_texts(str(path), interactive=False)
    assert texts == ["q0\n\na0", "q1\n\na1", "q2\n\na2", "q3\n\na3"]


def test_local_json_list_is_supported(tmp_path):
    path = tmp_path / "cal.json"
    path.write_text(json.dumps([{"prompt": "p", "response": "r"}]), encoding="utf8")
    assert read_texts(str(path), interactive=False) == ["p\n\nr"]


# --------------------------------------------------------------------------- #
# User-supplied formats and the ask path
# --------------------------------------------------------------------------- #

def test_format_from_answer_accepts_column_name_order_and_template():
    assert format_from_answer("out", ["in", "out"], {"in": "a", "out": "b"}).kind \
        == "column"
    # a role word ('output', 'prompt', ...) is mapped onto the matching column
    assert format_from_answer("output", ["in", "out"],
                              {"in": "a", "out": "b"}).columns == ("out",)
    assert format_from_answer("prompt", ["question", "answer"],
                              {"question": "q", "answer": "a"}).columns \
        == ("question",)
    fmt = format_from_answer("in,out", ["in", "out"], {"in": "a", "out": "b"})
    assert fmt.render({"in": "a", "out": "b"}) == "a\n\nb"
    fmt = format_from_answer("Q: {in}\nA: {out}", ["in", "out"],
                             {"in": "a", "out": "b"})
    assert fmt.render({"in": "a", "out": "b"}) == "Q: a\nA: b"
    with pytest.raises(FormatError):
        format_from_answer("{nope}", ["in", "out"], {"in": "a", "out": "b"})


def test_resolve_format_prefers_explicit_field_and_format():
    fmt = resolve_format(["in", "out"], field="out", sample={"in": "a", "out": "b"})
    assert fmt.kind == "column" and fmt.columns == ("out",)
    fmt = resolve_format(["in", "out"], format_spec="{out} then {in}",
                         sample={"in": "a", "out": "b"})
    assert fmt.render({"in": "a", "out": "b"}) == "b then a"
    with pytest.raises(FormatError):
        resolve_format(["in", "out"], field="text", sample={"in": "a", "out": "b"})


def test_resolve_format_asks_the_user_when_ambiguous(monkeypatch):
    import typer
    asked = {}

    def fake_prompt(message, default=None):
        asked["default"] = default
        return "{foo} -> {bar}"

    monkeypatch.setattr(typer, "prompt", fake_prompt)
    notes = []
    fmt = resolve_format(["foo", "bar"], sample={"foo": "a", "bar": "b"},
                         extra_samples=[{"foo": "c", "bar": "d"}],
                         interactive=True, notify=notes.append)
    assert fmt.render({"foo": "a", "bar": "b"}) == "a -> b"
    assert asked["default"] == "{foo}\n\n{bar}"
    assert any("could not auto-detect" in n for n in notes)


def test_resolve_format_warns_but_proceeds_offline():
    notes = []
    fmt = resolve_format(["foo", "bar"], sample={"foo": "a", "bar": "b"},
                         interactive=False, notify=notes.append)
    assert fmt.confidence == "low"
    assert any("warning" in n for n in notes)


def test_resolve_format_raises_when_nothing_to_go_on():
    with pytest.raises(FormatError) as exc:
        resolve_format(["id", "idx"], sample={"id": 1, "idx": 2},
                       interactive=False)
    assert "dataset-format" in str(exc.value)


# --------------------------------------------------------------------------- #
# read_texts over a stubbed HF datasets module
# --------------------------------------------------------------------------- #

class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)
        self.column_names = list(dict.fromkeys(k for r in self.rows for k in r))

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, key):
        return [r[key] for r in self.rows]

    def select(self, indices):
        return _FakeDataset([self.rows[i] for i in indices])


@pytest.fixture
def fake_datasets(monkeypatch):
    def install(rows):
        module = types.ModuleType("datasets")
        module.load_dataset = lambda *a, **k: _FakeDataset(rows)
        monkeypatch.setitem(sys.modules, "datasets", module)
        return module
    return install


def test_read_texts_detects_pair_columns(fake_datasets):
    fake_datasets([{"in": "q1", "out": "a1"}, {"in": "q2", "out": "a2"}])
    assert read_texts("some/dataset", interactive=False) == ["q1\n\na1", "q2\n\na2"]


def test_read_texts_detects_instruction_triple_with_limit(fake_datasets):
    fake_datasets([{"instruction": "i", "input": "x", "output": "o"},
                   {"instruction": "j", "input": "", "output": "p"}])
    assert read_texts("some/dataset", limit=1, interactive=False) == ["i\n\nx\n\no"]


def test_read_texts_renders_chat_column(fake_datasets):
    fake_datasets([{"messages": [{"role": "user", "content": "hi"},
                                 {"role": "assistant", "content": "hey"}]}])
    assert read_texts("some/dataset", interactive=False) == ["User: hi\nAssistant: hey"]


def test_read_texts_explicit_field_still_works(fake_datasets):
    fake_datasets([{"text": "plain"}])
    assert read_texts("some/dataset", field="text", interactive=False) == ["plain"]
    with pytest.raises(FormatError):
        read_texts("some/dataset", field="missing", interactive=False)


def test_read_texts_reports_unusable_dataset(fake_datasets):
    fake_datasets([{"id": 1, "idx": 2}])
    with pytest.raises(FormatError):
        read_texts("some/dataset", interactive=False)


def test_prepare_records_detected_format(fake_datasets, model_factory,
                                        tiny_tokenizer):
    fake_datasets([{"in": f"what is {i}", "out": str(i)} for i in range(20)])
    from llm_compressor.data import prepare

    model = model_factory("llama")
    data = prepare(model, tiny_tokenizer, "some/dataset", mode="fast",
                   samples=8, seq_len=16, interactive=False)
    assert data.info["dataset_format_name"] == "in/out"
    assert data.info["dataset_format_confidence"] == "high"
    assert data.info["dataset_format_columns"] == ["in", "out"]
    assert data.info["examples"] == 8
    assert data.train, "calibration batches should be produced"
