"""Forward helpers that work across causal, encoder-only and seq2seq models.

Nothing here produces "fake calibration": these helpers only shape already
tokenized example ids into a batch and read logits / hidden states back out.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


def model_kind(model: Any) -> str:
    """'seq2seq' | 'causal' | 'encoder'."""
    cfg = getattr(model, "config", None)
    if getattr(cfg, "is_encoder_decoder", False) or hasattr(model, "get_encoder"):
        return "seq2seq"
    if getattr(cfg, "is_decoder", False) or "Causal" in type(model).__name__:
        return "causal"
    if "LMHead" in type(model).__name__:
        return "causal"
    return "encoder"


def make_batch(model: Any, ids: torch.Tensor, pad_id: int = 0,
               attention_mask: Optional[torch.Tensor] = None,
               kind: Optional[str] = None) -> dict:
    """Build a batch of already-tokenized examples for ``model``."""
    kind = kind or model_kind(model)
    if attention_mask is None:
        attention_mask = (ids != pad_id).long()
    batch: dict = {"input_ids": ids, "attention_mask": attention_mask}
    if kind == "seq2seq":
        # teacher-forced decoder input (BOS/prepended pad), no labels -> no loss
        dec = ids[:, :-1].clone()
        dec[:, 0] = model.config.decoder_start_token_id if getattr(
            model.config, "decoder_start_token_id", None) is not None else pad_id
        batch["decoder_input_ids"] = dec
    return batch


def output_tensor(out: Any) -> torch.Tensor:
    """Logits when the model has them, else hidden states."""
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        return out.last_hidden_state
    raise RuntimeError("model output contains neither logits nor hidden states")


@torch.no_grad()
def forward_tensor(model: Any, batch: dict, **kw) -> torch.Tensor:
    return output_tensor(model(**batch, use_cache=False, **kw))


def causal_labels(ids: torch.Tensor, pad_id: int = 0) -> torch.Tensor:
    labels = ids.clone()
    labels[labels == pad_id] = -100
    return labels
