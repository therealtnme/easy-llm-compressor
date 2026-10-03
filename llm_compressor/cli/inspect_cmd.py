from __future__ import annotations

import json
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..model.architecture import ModelArchitecture, SupportLevel
from ..model.introspect import ModelIntrospector
from ..model.loader import load_model_for_inspection
from ..utils.formatting import format_number, format_params, shorten

console = Console()


def _level_style(level: SupportLevel) -> str:
    return {
        SupportLevel.SUPPORTED: "bold green",
        SupportLevel.UNKNOWN: "bold yellow",
        SupportLevel.UNSUPPORTED: "bold red",
    }[level]


def inspect_command(
    model: str = typer.Argument(..., help="Hugging Face model id or local path"),
    device: str = typer.Option("cpu", "--device", "-d", help="Device to load on"),
    dtype: str = typer.Option(
        "auto", "--dtype", help="auto | float32 | float16 | bfloat16"
    ),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code", help="Allow custom model code"
    ),
    revision: Optional[str] = typer.Option(None, "--revision"),
    json_output: bool = typer.Option(
        False, "--json", help="Emit JSON instead of the Rich report"
    ),
) -> None:
    """Load a model and print a structural report. Never modifies the model."""

    with console.status(f"[cyan]Loading {model}…[/cyan]"):
        try:
            loaded, config = load_model_for_inspection(
                model_id=model,
                device=device,
                dtype_str=dtype,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
        except Exception as e:  # noqa: BLE001
            console.print(f"[bold red]Failed to load model:[/bold red] {e}")
            raise typer.Exit(code=2)

    arch = ModelIntrospector(loaded, model_name=model, config=config).analyze()

    if json_output:
        console.print_json(json.dumps(_to_jsonable(arch)))
        return

    _print_header(arch)
    _print_embeddings(arch)
    _print_output_heads(arch)
    _print_tied_groups(arch)
    _print_layer_summary(arch)
    _print_protected(arch)
    _print_capabilities(arch)
    console.print(
        "\n[dim]Note: inspection only. No tensors were modified or saved.[/dim]"
    )


# ---------- rendering ---------------------------------------------------------


def _print_header(arch: ModelArchitecture) -> None:
    lines = [
        f"[bold]Model:[/bold]                 {arch.model_name}",
        f"[bold]Architecture:[/bold]          {arch.model_class}",
        f"[bold]Parameters:[/bold]            {format_params(arch.total_parameters)} "
        f"({format_number(arch.total_parameters)})",
        f"[bold]Trainable:[/bold]             {format_params(arch.trainable_parameters)}",
        f"[bold]Dtype:[/bold]                 {arch.dtype}",
        f"[bold]Device:[/bold]                {arch.device}",
        "",
        f"[bold]Hidden size:[/bold]           {arch.hidden_size}",
        f"[bold]Layers:[/bold]                {arch.num_layers}",
        f"[bold]MLP intermediate size:[/bold] {arch.mlp_intermediate_size}",
        f"[bold]Attention heads:[/bold]       {arch.num_attention_heads}",
        f"[bold]KV heads:[/bold]              {arch.num_kv_heads}",
        f"[bold]Head dim:[/bold]              {arch.head_dim}",
        f"[bold]Vocab size:[/bold]            {arch.vocab_size}",
    ]
    console.print(Panel("\n".join(lines), title="Model", border_style="cyan"))


def _print_neuron_definition(arch: ModelArchitecture) -> None:
    n = arch.count_mlp_neurons()
    console.print(
        f"[bold]Estimated neurons:[/bold] {format_number(n)}  "
        f"[dim](definition: MLP intermediate channels)[/dim]"
    )
    if arch.num_attention_heads:
        console.print(
            f"[bold]Attention heads:[/bold]   {format_number(arch.count_attention_heads())}  "
            f"[dim]({arch.num_layers} layers × {arch.num_attention_heads} heads)[/dim]"
        )
    if arch.num_kv_heads:
        console.print(
            f"[bold]KV heads:[/bold]          {format_number(arch.count_kv_heads())}  "
            f"[dim]({arch.num_layers} layers × {arch.num_kv_heads} kv-heads)[/dim]"
        )


def _print_embeddings(arch: ModelArchitecture) -> None:
    if not arch.embeddings:
        return
    t = Table(title="Embeddings", show_lines=False, header_style="bold")
    t.add_column("Name")
    t.add_column("Shape", justify="right")
    for e in arch.embeddings:
        t.add_row(shorten(e.name), f"({e.vocab_size}, {e.hidden_size})")
    console.print(t)


def _print_output_heads(arch: ModelArchitecture) -> None:
    if not arch.output_heads:
        return
    t = Table(title="Output heads", header_style="bold")
    t.add_column("Name")
    t.add_column("Shape", justify="right")
    for h in arch.output_heads:
        t.add_row(shorten(h.name), f"({h.vocab_size}, {h.hidden_size})")
    console.print(t)


def _print_tied_groups(arch: ModelArchitecture) -> None:
    if not arch.tied_groups:
        return
    t = Table(title="Tied parameter groups", header_style="bold")
    t.add_column("Names")
    t.add_column("Shape", justify="right")
    for g in arch.tied_groups:
        t.add_row(" ↔ ".join(shorten(n, 32) for n in g.names), str(g.shape))
    console.print(t)


def _print_layer_summary(arch: ModelArchitecture) -> None:
    if not arch.layers:
        console.print(
            "[yellow]No transformer block stack was identified. "
            "Structural compression operations will be reported as UNKNOWN.[/yellow]"
        )
        return

    t = Table(title=f"Layers ({len(arch.layers)})", header_style="bold")
    t.add_column("#", justify="right")
    t.add_column("Attention")
    t.add_column("MLP kind")
    t.add_column("MLP width", justify="right")
    t.add_column("Norms", justify="right")

    for L in arch.layers[:8]:
        attn = "—"
        if L.attention:
            if L.attention.is_fused_qkv:
                attn = "fused QKV"
            elif L.attention.is_grouped_query:
                attn = f"GQA ({L.attention.num_heads}/{L.attention.num_kv_heads})"
            elif L.attention.num_heads:
                attn = f"MHA ({L.attention.num_heads})"
            else:
                attn = "attention"
        mlp_kind = L.mlp.kind if L.mlp else "—"
        mlp_w = str(L.mlp.intermediate_size) if (L.mlp and L.mlp.intermediate_size) else "—"
        t.add_row(str(L.index), attn, mlp_kind, mlp_w, str(len(L.norms)))

    if len(arch.layers) > 8:
        t.add_row("…", "…", "…", "…", "…")
        last = arch.layers[-1]
        attn = "—"
        if last.attention:
            if last.attention.is_grouped_query:
                attn = f"GQA ({last.attention.num_heads}/{last.attention.num_kv_heads})"
            elif last.attention.num_heads:
                attn = f"MHA ({last.attention.num_heads})"
        mlp_kind = last.mlp.kind if last.mlp else "—"
        mlp_w = str(last.mlp.intermediate_size) if (last.mlp and last.mlp.intermediate_size) else "—"
        t.add_row(str(last.index), attn, mlp_kind, mlp_w, str(len(last.norms)))

    console.print(t)


def _print_protected(arch: ModelArchitecture) -> None:
    if not arch.protected:
        return
    t = Table(
        title=f"Protected components ({len(arch.protected)})",
        header_style="bold",
    )
    t.add_column("Category")
    t.add_column("Name")
    t.add_column("Reason", overflow="fold")

    # Group by category for readability.
    by_cat: dict[str, list] = {}
    for p in arch.protected:
        by_cat.setdefault(p.category, []).append(p)

    for cat, items in by_cat.items():
        for i, p in enumerate(items):
            t.add_row(
                cat if i == 0 else "",
                shorten(p.name, 46),
                p.reason if i == 0 else "",
            )
    console.print(t)


def _print_capabilities(arch: ModelArchitecture) -> None:
    t = Table(title="Structural compression capabilities", header_style="bold")
    t.add_column("Operation")
    t.add_column("Status")
    t.add_column("Notes", overflow="fold")

    rows = [
        ("inspect", arch.cap_inspect),
        ("prune MLP neurons", arch.cap_prune_mlp_neurons),
        ("prune attention heads", arch.cap_prune_attention_heads),
        ("prune KV heads", arch.cap_prune_kv_heads),
        ("delete layers", arch.cap_prune_layers),
        ("exact affine fusion", arch.cap_fuse_affine),
        ("learned layer fusion", arch.cap_fuse_learned),
    ]
    for label, cap in rows:
        t.add_row(
            label,
            Text(cap.level.value, style=_level_style(cap.level)),
            cap.reason or "",
        )
    console.print(t)


# ---------- json output -------------------------------------------------------


def _to_jsonable(arch: ModelArchitecture) -> dict:
    def cap(c) -> dict:
        return {"level": c.level.value, "reason": c.reason}

    return {
        "model_name": arch.model_name,
        "model_class": arch.model_class,
        "parameters": arch.total_parameters,
        "trainable_parameters": arch.trainable_parameters,
        "dtype": str(arch.dtype),
        "device": arch.device,
        "hidden_size": arch.hidden_size,
        "layers": arch.num_layers,
        "mlp_intermediate_size": arch.mlp_intermediate_size,
        "num_attention_heads": arch.num_attention_heads,
        "num_kv_heads": arch.num_kv_heads,
        "head_dim": arch.head_dim,
        "vocab_size": arch.vocab_size,
        "mlp_neuron_count": arch.count_mlp_neurons(),
        "attention_head_count": arch.count_attention_heads(),
        "kv_head_count": arch.count_kv_heads(),
        "embeddings": [
            {"name": e.name, "vocab_size": e.vocab_size, "hidden_size": e.hidden_size}
            for e in arch.embeddings
        ],
        "output_heads": [
            {"name": h.name, "vocab_size": h.vocab_size, "hidden_size": h.hidden_size}
            for h in arch.output_heads
        ],
        "tied_groups": [
            {"names": g.names, "shape": list(g.shape)} for g in arch.tied_groups
        ],
        "protected": [
            {"name": p.name, "category": p.category, "reason": p.reason}
            for p in arch.protected
        ],
        "capabilities": {
            "inspect": cap(arch.cap_inspect),
            "prune_mlp_neurons": cap(arch.cap_prune_mlp_neurons),
            "prune_attention_heads": cap(arch.cap_prune_attention_heads),
            "prune_kv_heads": cap(arch.cap_prune_kv_heads),
            "prune_layers": cap(arch.cap_prune_layers),
            "fuse_affine": cap(arch.cap_fuse_affine),
            "fuse_learned": cap(arch.cap_fuse_learned),
        },
    }