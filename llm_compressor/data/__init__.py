"""Calibration data: DATASET MODE vs DATASET-FREE MODE.

DATASET MODE  -> real tokenized examples are needed (activation scoring and
                 distillation). Examples are encoded once, cached on disk keyed
                 by (tokenizer, seq_len, packing, sample count), and replayed in
                 batches. No random tensors are ever substituted for text.
DATASET-FREE  -> no examples exist at all; only genuinely data-free methods may
                 be used (weight-norm scoring, structural analysis). The dataset
                 helpers are never called in that mode.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import torch

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


def read_texts(source: str, split: str = "train", field: str = "text",
               limit: Optional[int] = None,
               local_files_only: bool = False) -> list[str]:
    """Read real text from a local file (.txt/.jsonl) or a HF datasets dataset."""
    if os.path.exists(source):
        if source.endswith((".txt", ".text")):
            with open(source, encoding="utf8") as fh:
                texts = [ln.strip() for ln in fh if ln.strip()]
        elif source.endswith((".jsonl", ".json")):
            texts = []
            with open(source, encoding="utf8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    texts.append(str(row[field] if isinstance(row, dict) else row))
        else:
            raise DataError(
                f"unsupported local dataset file '{source}' "
                "(expected .txt or .jsonl)")
        return texts[:limit] if limit else texts

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise DataError("the 'datasets' package is required to load "
                        f"'{source}'") from exc
    ds = load_dataset(source, split=split, trust_remote_code=False)
    if field not in ds.column_names:
        raise DataError(
            f"field '{field}' not in dataset columns {ds.column_names}")
    texts = [str(t) for t in ds[field]]
    return texts[:limit] if limit else texts


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
            packing: Optional[bool] = None, field: str = "text",
            split: str = "train", batch_size: int = 4,
            cache_dir: str = ".cache/calibration",
            local_files_only: bool = False) -> CalibrationData:
    """Encode-once-and-cache calibration data (DATASET MODE only)."""
    preset = MODES.get(mode, {})
    seq_len = int(seq_len or preset.get("seq_len") or 128)
    samples = int(samples or preset.get("samples") or 32)
    packing = bool(preset.get("packing", True)) if packing is None else packing

    texts = read_texts(source, split=split, field=field, limit=samples,
                       local_files_only=local_files_only)
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
