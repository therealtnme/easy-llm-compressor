from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


_DTYPE_MAP = {
    "auto": "auto",
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_dtype(dtype_str: str):
    key = dtype_str.lower()
    if key not in _DTYPE_MAP:
        raise ValueError(
            f"Unknown dtype '{dtype_str}'. "
            f"Expected one of: {sorted(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[key]


def load_model_for_inspection(
    model_id: str,
    device: str = "cpu",
    dtype_str: str = "auto",
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
) -> tuple[nn.Module, Any]:
    """Load a HF model for inspection.

    Tries AutoModelForCausalLM first (so the LM head is a real module),
    then falls back to AutoModel. Uses dtype= (not torch_dtype=).
    Never modifies the loaded model beyond eval() and optional device move.
    """
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    dtype = resolve_dtype(dtype_str)

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )

    common_kwargs = dict(
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        revision=revision,
        low_cpu_mem_usage=True,
    )

    model: Optional[nn.Module] = None
    last_err: Optional[Exception] = None

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
    except Exception as e:  # noqa: BLE001
        last_err = e
        model = None

    if model is None:
        try:
            model = AutoModel.from_pretrained(model_id, **common_kwargs)
        except Exception:
            if last_err is not None:
                raise last_err
            raise

    model.eval()
    if device and device != "cpu":
        try:
            model.to(device)
        except Exception:
            # Best-effort: leave on CPU if device move fails.
            pass

    return model, config