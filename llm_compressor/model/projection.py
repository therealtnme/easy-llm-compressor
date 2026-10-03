"""Projection abstraction.

`nn.Linear` is not the only dense projection in Transformers models. GPT-2 (and
several other older families) use `transformers.pytorch_utils.Conv1D`, whose
weight is stored transposed as `(in_features, out_features)`. Custom architectures
use their own classes.

Everything downstream (scoring, pruning, fusion, rebuild) works through
`Projection`, which normalises the layout to `(out_features, in_features)` and
knows how to physically slice and resize the underlying module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

try:  # pragma: no cover - import guard
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover
    Conv1D = None  # type: ignore[assignment]


__all__ = ["Projection", "as_projection", "is_projection"]


def _is_conv1d(module: nn.Module) -> bool:
    return Conv1D is not None and isinstance(module, Conv1D)


def _is_plain_linear(module: nn.Module) -> bool:
    """`nn.Linear` and subclasses, but not exotic quantised variants.

    Quantised linear layers (bitsandbytes `Linear8bitLt`, `Linear4bit`, torchao,
    etc.) carry extra parameters/buffers and cannot be structurally sliced. We
    refuse them rather than silently corrupting the checkpoint.
    """
    if not isinstance(module, nn.Linear):
        return False
    names = {n for n, _ in module.named_parameters(recurse=False)}
    return names <= {"weight", "bias"}


def _generic_dense_layout(module: nn.Module) -> Optional[tuple[int, int, str]]:
    """Fallback for custom dense modules: (in_features, out_features, layout)."""
    w = getattr(module, "weight", None)
    if not isinstance(w, nn.Parameter) or w.ndim != 2:
        return None
    inf = getattr(module, "in_features", None)
    outf = getattr(module, "out_features", None)
    if inf is None or outf is None:
        return None
    if tuple(w.shape) == (outf, inf):
        return int(inf), int(outf), "linear"
    if tuple(w.shape) == (inf, outf):
        return int(inf), int(outf), "transposed"
    return None


def is_projection(module: nn.Module) -> bool:
    if _is_conv1d(module) or _is_plain_linear(module):
        return True
    # Exclude the *same* object being seen as both; generic fallback requires a
    # rank-2 weight parameter and declared feature counts.
    return _generic_dense_layout(module) is not None


@dataclass
class Projection:
    """Layout-normalised view over a dense projection module."""

    name: str
    module: nn.Module
    in_features: int
    out_features: int
    layout: str  # "linear" (weight = out x in) | "transposed" (weight = in x out)
    kind: str = "linear"  # "linear" | "conv1d" | "custom"

    # ------------------------------------------------------------------ #
    @classmethod
    def of(cls, name: str, module: nn.Module) -> "Projection":
        if _is_conv1d(module):
            nf = int(module.nf)
            wshape = tuple(module.weight.shape)
            nx = wshape[0] if wshape[1] == nf else wshape[1]
            return cls(
                name=name,
                module=module,
                in_features=int(nx),
                out_features=nf,
                layout="transposed" if wshape == (nx, nf) else "linear",
                kind="conv1d",
            )
        if _is_plain_linear(module):
            return cls(
                name=name,
                module=module,
                in_features=int(module.in_features),
                out_features=int(module.out_features),
                layout="linear",
                kind="linear",
            )
        layout = _generic_dense_layout(module)
        if layout is None:
            raise TypeError(f"{type(module).__name__} is not a dense projection")
        inf, outf, lay = layout
        return cls(name=name, module=module, in_features=inf, out_features=outf,
                   layout=lay, kind="custom")

    # ------------------------------------------------------------------ #
    @property
    def has_bias(self) -> bool:
        return getattr(self.module, "bias", None) is not None

    @property
    def weight(self) -> nn.Parameter:
        return self.module.weight  # type: ignore[attr-defined]

    @property
    def weight_matrix(self) -> torch.Tensor:
        """A read-only ``(out_features, in_features)`` view."""
        w = self.weight
        return w if self.layout == "linear" else w.t()

    def n_parameters(self) -> int:
        n = self.in_features * self.out_features
        if self.has_bias:
            n += self.out_features
        return n

    # ------------------------------------------------------------------ #
    def _set_weight_matrix(self, new_w: torch.Tensor, requires_grad: bool) -> None:
        assert new_w.shape == (self.out_features, self.in_features), (
            f"{self.name}: expected {(self.out_features, self.in_features)}, "
            f"got {tuple(new_w.shape)}"
        )
        stored = new_w if self.layout == "linear" else new_w.t().contiguous()
        self.module.weight = nn.Parameter(
            stored.contiguous().to(self.weight.dtype), requires_grad=requires_grad
        )

    def _set_bias(self, keep: Optional[torch.Tensor]) -> None:
        if not self.has_bias:
            if keep is not None:
                # Module had no bias; keep it that way (a bias cannot be conjured
                # without changing the learned function).
                raise ValueError(f"{self.name}: module has no bias to set")
            return
        bias = self.module.bias  # type: ignore[attr-defined]
        if keep is None:
            self.module.bias = None  # type: ignore[assignment]
            return
        self.module.bias = nn.Parameter(
            keep.contiguous().to(bias.dtype), requires_grad=bias.requires_grad
        )

    def _update_shapes(self) -> None:
        """Refresh the module's declared feature counts (and Conv1D `nf`)."""
        if _is_conv1d(self.module):
            self.module.nf = self.out_features  # type: ignore[attr-defined]
            if hasattr(self.module, "nx"):
                self.module.nx = self.in_features  # type: ignore[attr-defined]
            return
        for attr, val in (("in_features", self.in_features),
                          ("out_features", self.out_features)):
            if hasattr(self.module, attr):
                setattr(self.module, attr, val)

    # ------------------------------------------------------------------ #
    def slice_outputs(self, keep: Sequence[int]) -> None:
        """Physically keep only the listed output channels (rows)."""
        keep_idx = torch.as_tensor(list(keep), dtype=torch.long)
        w = self.weight_matrix.index_select(0, keep_idx)
        new_out = len(keep)
        bias = None
        if self.has_bias:
            bias = self.module.bias.index_select(0, keep_idx)  # type: ignore[attr-defined]
        rg = self.weight.requires_grad
        self.out_features = new_out
        self._set_weight_matrix(w.contiguous(), rg)
        self._set_bias(bias)
        self._update_shapes()

    def slice_inputs(self, keep: Sequence[int]) -> None:
        """Physically keep only the listed input channels (columns)."""
        keep_idx = torch.as_tensor(list(keep), dtype=torch.long)
        w = self.weight_matrix.index_select(1, keep_idx)
        new_in = len(keep)
        rg = self.weight.requires_grad
        self.in_features = new_in
        self._set_weight_matrix(w.contiguous(), rg)
        self._update_shapes()

    def resize(self, in_features: int, out_features: int) -> None:
        """Re-declare shapes with fresh (empty) parameters of the right size."""
        dtype = self.weight.dtype
        dev = self.weight.device
        rg = self.weight.requires_grad
        w = torch.zeros((out_features, in_features), dtype=dtype, device=dev)
        self.in_features = in_features
        self.out_features = out_features
        self._set_weight_matrix(w, rg)
        if self.has_bias:
            self.module.bias = nn.Parameter(  # type: ignore[assignment]
                torch.zeros(out_features, dtype=dtype, device=dev), requires_grad=rg
            )
        elif "bias" in getattr(self.module, "_parameters", {}):
            self.module.bias = None
        self._update_shapes()

    def describe(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "has_bias": self.has_bias,
            "parameters": self.n_parameters(),
        }


def as_projection(name: str, module: nn.Module) -> Optional[Projection]:
    try:
        return Projection.of(name, module)
    except TypeError:
        return None
