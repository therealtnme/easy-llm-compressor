"""Calibration data: DATASET MODE vs DATASET-FREE MODE.

DATASET MODE  -> real tokenized examples are needed (activation scoring and
                 distillation). Examples are encoded once, cached on disk keyed
                 by (tokenizer, seq_len, packing, sample count), and replayed in
                 batches. No random tensors are ever substituted for text.
                 The column layout of a dataset is auto-detected (see
                 ``llm_compressor.data.formats``); if it cannot be recognised the
                 user is asked how the dataset is formatted.
DATASET-FREE  -> no examples exist at all; only genuinely data-free methods may
                 be used (weight-norm scoring, structural analysis). The dataset
                 helpers are never called in that mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

import torch

from .formats import DatasetFormat, FormatError, resolve_format

MODES = {
    "fast": {"samples": 8, "seq_len": 64, "packing": False},
    "balanced": {"samples": 32, "seq_len": 128, "packing": True},
    "accurate": {"samples": 128, "seq_len": 256, "packing": True},
}


class DataError(RuntimeError):
    pass


@dataclass
class CalibrationData:
    """Encoded, cached calibration batches plus provenance metadata."""

    train: list[dict] = field(default_factory=list)
    valid: list[dict] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def batches(self) -> list[dict]:
        return self.train

    def all_ids(self) -> torch.Tensor:
        if not self.train:
            raise DataError("no calibration data")
        return torch.cat([b["input_ids"] for b in self.train], dim=0)


# --------------------------------------------------------------------------- #
# Reading datasets in any shape
# --------------------------------------------------------------------------- #

def _render_rows(rows: Iterable[Any], fmt: DatasetFormat,
                 limit: Optional[int] = None) -> list[str]:
    from .formats import render_value

    texts: list[str] = []
    for row in rows:
        text = fmt.render(row) if isinstance(row, dict) else render_value(row)
        if text.strip():
            texts.append(text)
        if limit and len(texts) >= limit:
            break
    return texts


def _read_local(source: str, field: str, limit: Optional[int],
                format_spec: Optional[str], interactive: Optional[bool],
                notify: Optional[Callable[[str], None]],
                meta: Optional[dict] = None) -> list[str]:
    if source.endswith((".txt", ".text")):
        with open(source, encoding="utf8") as fh:
            texts = [ln.strip() for ln in fh if ln.strip()]
        if notify:
            notify("calibration data: local plain-text file, one example per line")
        return texts[:limit] if limit else texts
    if not source.endswith((".jsonl", ".json", ".ndjson")):
        raise DataError(
            f"unsupported local dataset file '{source}' "
            "(expected .txt, .jsonl or .json)")

    with open(source, encoding="utf8") as fh:
        blob = fh.read().strip()
    rows: list[Any] = []
    try:
        parsed = json.loads(blob) if blob else None
    except ValueError:
        parsed = None
    if isinstance(parsed, list):
        rows = parsed
    elif isinstance(parsed, dict):
        nested = next((v for v in parsed.values()
                       if isinstance(v, list) and v and
                       isinstance(v[0], (dict, str))), None)
        rows = list(nested) if nested is not None else [parsed]
    else:  # jsonl / ndjson: one JSON object per line
        for line in blob.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise DataError(f"no examples found in '{source}'")
    if all(isinstance(r, str) for r in rows):
        return rows[:limit] if limit else rows

    dicts = [r for r in rows if isinstance(r, dict)]
    if not dicts:
        raise DataError(f"no usable examples found in '{source}'")
    columns = list(dict.fromkeys(k for r in dicts[:5] for k in r))
    fmt = resolve_format(columns, field=field, format_spec=format_spec,
                         sample=dicts[0], extra_samples=dicts[1:5],
                         interactive=interactive, notify=notify)
    _record_format(meta, fmt)
    if notify:
        notify("calibration data: " + fmt.summary())
    return _render_rows(dicts, fmt, limit)


def read_texts(source: str, split: str = "train", field: str = "auto",
               limit: Optional[int] = None,
               local_files_only: bool = False,
               dataset_config: Optional[str] = None,
               format_spec: Optional[str] = None,
               interactive: Optional[bool] = None,
               notify: Optional[Callable[[str], None]] = None,
               meta: Optional[dict] = None) -> list[str]:
    """Read real text from a local file (.txt/.jsonl/.json) or a HF dataset.

    ``field`` may be a column name, or ``"auto"`` (default) to detect the
    layout. ``format_spec`` accepts a template such as
    ``"{instruction}\\n\\n{input}\\n\\n{output}"``, a comma separated column
    order, or a single column name. When the layout cannot be detected, the
    user is asked (interactive terminals) or ``FormatError`` explains what to
    pass instead.
    """
    if os.path.exists(source):
        return _read_local(source, field, limit, format_spec, interactive,
                           notify, meta)

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise DataError("the 'datasets' package is required to load "
                        f"'{source}'") from exc
    if dataset_config:
        ds = load_dataset(source, dataset_config, split=split,
                          trust_remote_code=False)
    else:
        ds = load_dataset(source, split=split, trust_remote_code=False)

    preview: list[dict] = []
    try:
        n = min(5, len(ds))
        if n:
            preview = [dict(r) for r in ds.select(range(n))]
    except Exception:  # iterable datasets without select()
        preview = [dict(r) for _, r in zip(range(5), iter(ds))]

    fmt = resolve_format(list(ds.column_names), field=field,
                         format_spec=format_spec,
                         sample=preview[0] if preview else None,
                         extra_samples=preview[1:], interactive=interactive,
                         notify=notify)
    _record_format(meta, fmt)
    if notify:
        notify("calibration data: " + fmt.summary())
    if fmt.kind in ("column", "chat"):
        column = fmt.columns[0]
        return _render_rows(({column: v} for v in ds[column]), fmt, limit)
    return _render_rows(ds, fmt, limit)


def _record_format(meta: Optional[dict], fmt: DatasetFormat) -> None:
    """Record how the dataset was interpreted, for the run report."""
    if meta is None:
        return
    meta["dataset_format_name"] = fmt.name
    meta["dataset_format"] = fmt.summary()
    meta["dataset_format_confidence"] = fmt.confidence
    meta["dataset_format_columns"] = list(fmt.columns)


def _cache_key(tokenizer, seq_len: int, packing: bool, texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update((getattr(tokenizer, "name_or_path", "") or type(tokenizer).__name__).encode())
    h.update(f"|{seq_len}|{packing}|".encode())
    for t in texts[:64]:
        h.update(t.encode("utf8", "ignore"))
    return h.hexdigest()[:20]


def build_sequences(tokenizer, texts: Sequence[str], seq_len: int,
                    packing: bool = True, mode: str = "custom") -> list[list[int]]:
    if not texts:
        raise DataError("no calibration text available")
    eos = getattr(tokenizer, "eos_token_id", None) or 0
    flat: list[int] = []
    for text in texts:
        ids = tokenizer(text).get("input_ids") or []
        if not ids:
            continue
        flat.extend(list(ids) + [eos])
    if not flat:
        raise DataError("tokenizer produced no tokens for the supplied text")
    if packing:
        if len(flat) < seq_len:
            raise DataError(
                f"only {len(flat)} tokens available but seq_len={seq_len}; "
                "supply more data or use a smaller --seq-len")
        n = (len(flat) // seq_len) * seq_len
        return [flat[i:i + seq_len] for i in range(0, n, seq_len)]
    out = []
    for text in texts:
        ids = list(tokenizer(text).get("input_ids") or [])
        if not ids:
            continue
        if len(ids) < 2:
            continue
        out.append(ids[:seq_len])
    if not out:
        raise DataError("no usable examples after tokenization")
    return out


def make_batches(model, sequences: Sequence[Sequence[int]], pad_id: int = 0,
                 batch_size: int = 4) -> list[dict]:
    from ..utils.batch import make_batch

    batches = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        width = max(len(s) for s in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        for r, seq in enumerate(chunk):
            ids[r, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        batches.append(make_batch(model, ids, pad_id=pad_id))
    return batches


def prepare(model, tokenizer, source: str, *, mode: str = "custom",
            seq_len: Optional[int] = None, samples: Optional[int] = None,
            packing: Optional[bool] = None, field: str = "auto",
            split: str = "train", batch_size: int = 4,
            cache_dir: str = ".cache/calibration",
            local_files_only: bool = False,
            dataset_config: Optional[str] = None,
            format_spec: Optional[str] = None,
            interactive: Optional[bool] = None,
            notify: Optional[Callable[[str], None]] = None) -> CalibrationData:
    """Encode-once-and-cache calibration data (DATASET MODE only)."""
    preset = MODES.get(mode, {})
    seq_len = int(seq_len or preset.get("seq_len") or 128)
    samples = int(samples or preset.get("samples") or 32)
    packing = bool(preset.get("packing", True)) if packing is None else packing

    meta: dict = {}
    try:
        texts = read_texts(source, split=split, field=field, limit=samples,
                           local_files_only=local_files_only,
                           dataset_config=dataset_config, format_spec=format_spec,
                           interactive=interactive, notify=notify, meta=meta)
    except FormatError as exc:
        raise DataError(str(exc)) from exc
    if len(texts) < 2:
        raise DataError(f"need at least 2 examples, got {len(texts)} from '{source}'")
    key = _cache_key(tokenizer, seq_len, packing, texts)
    cache_file = os.path.join(cache_dir, f"{key}.pt")
    if os.path.exists(cache_file):
        blob = torch.load(cache_file, weights_only=False)
        seqs = blob["sequences"]
    else:
        seqs = build_sequences(tokenizer, texts, seq_len, packing=packing, mode=mode)
        os.makedirs(cache_dir, exist_ok=True)
        torch.save({"sequences": seqs, "seq_len": seq_len, "packing": packing},
                   cache_file)

    holdout = max(1, len(seqs) // 10) if len(seqs) > 4 else 0
    valid_seqs = seqs[-holdout:] if holdout else seqs[:1]
    train_seqs = seqs[:-holdout] if holdout else seqs
    pad_id = getattr(tokenizer, "pad_token_id", None) or 0
    data = CalibrationData(
        train=make_batches(model, train_seqs, pad_id, batch_size),
        valid=make_batches(model, valid_seqs, pad_id, batch_size),
        info={
            "mode": f"dataset:{mode}",
            "source": source, "split": split, "field": field,
            "examples": len(texts), "sequences": len(seqs),
            "seq_len": seq_len, "packing": packing,
            "train_sequences": len(train_seqs),
            "valid_sequences": len(valid_seqs),
            "cache": cache_file, "cached": os.path.exists(cache_file),
            "tokens": int(sum(len(s) for s in seqs)),
            **meta,
        },
    )
    return data


def benchmark_batch(model, seq_len: int = 16, batch: int = 1) -> dict:
    """Deterministic fixed-token input used ONLY for latency/equivalence checks.

    This is deliberately not called calibration data: it carries no text
    information and is used purely as an invariant/benchmark probe.
    """
    from ..utils.batch import make_batch

    ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).repeat(batch, 1)
    ids = ids % max(1, int(getattr(model.config, "vocab_size", 128) or 128))
    # make_batch adds decoder_input_ids for encoder-decoder models, which are
    # required for a seq2seq forward pass to be well defined
    out = dict(make_batch(model, ids, pad_id=0))
    out["provenance"] = "structural_probe"
    return out
