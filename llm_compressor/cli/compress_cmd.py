"""``llm-compress compress`` — interactive wizard + full non-interactive CLI."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import typer

from ..pipeline import CompressionError, CompressionOptions, run_compression


def _banner(text: str) -> None:
    typer.echo("")
    typer.echo("=" * 66)
    typer.echo(text)
    typer.echo("=" * 66)


def _wizard(base: CompressionOptions) -> CompressionOptions:
    """Interactive wizard covering every option that changes the result."""
    from ..pipeline import CompressionOptions as O

    _banner("llm-compress interactive wizard")
    out = replace(base)
    out.output = typer.prompt("output directory", default=base.output)

    _banner("1) calibration data")
    if typer.confirm(
            "Use real calibration data (DATASET MODE)?\n"
            "  Dataset-free mode only uses weight/structure analysis and never "
            "invents activations.", default=bool(base.dataset)):
        out.dataset_free = False
        out.dataset = typer.prompt("dataset id or local .txt/.jsonl path",
                                   default=base.dataset or "")
        out.dataset_mode = typer.prompt("mode (fast/balanced/accurate/custom)",
                                        default=base.dataset_mode)
        out.dataset_field = typer.prompt("text field", default=base.dataset_field)
        if out.dataset_mode == "custom":
            out.num_samples = typer.prompt("number of examples", type=int,
                                           default=base.num_samples or 32)
            out.seq_len = typer.prompt("sequence length", type=int,
                                       default=base.seq_len or 128)
    else:
        out.dataset_free = True
        out.dataset = None

    _banner("2) neuron budget (one MLP intermediate channel = one neuron)")
    if typer.confirm("Set an explicit neuron budget or output-size goal?",
                     default=True):
        how = typer.prompt("specify as (percent/count/goal-neurons/goal-params)",
                           default="percent")
        out.remove_percent = None
        out.remove_count = None
        out.goal_neurons = None
        out.goal_params = None
        if how == "percent":
            out.remove_percent = typer.prompt("percent of MLP neurons to remove",
                                              type=float, default=30.0)
        elif how == "count":
            out.remove_count = typer.prompt("exact number of neurons to remove",
                                            type=int, default=500000)
        elif how == "goal-neurons":
            out.goal_neurons = typer.prompt(
                "MLP neurons wanted in the OUTPUT model", type=int, default=50000)
        elif how == "goal-params":
            out.goal_params = typer.prompt(
                "parameters wanted in the OUTPUT model", type=int,
                default=175000000)
        else:
            typer.secho(f"unknown budget kind '{how}': leaving the budget unset",
                        fg=typer.colors.YELLOW)
    out.allocation = typer.prompt("allocation (global/uniform/hybrid)",
                                  default=base.allocation)
    out.scoring = typer.prompt(
        "neuron scoring (weight_l1/weight_l2/weight_combined/"
        "activation/activation_weighted)", default=base.scoring)

    _banner("3) depth (target student layers)")
    out.layer_mode = typer.prompt("layer mode (search/none)", default=base.layer_mode)
    out.target_student_layers = typer.prompt(
        "target student layers (must be < teacher depth)",
        type=int, default=base.target_student_layers)
    out.fuse = typer.prompt("fusion (none/exact/learned)", default=base.fuse)
    if out.fuse == "learned":
        out.distill_steps = typer.prompt("distillation steps", type=int,
                                         default=base.distill_steps)
        out.distill_lr = typer.prompt("distillation learning rate", type=float,
                                      default=base.distill_lr)

    _banner("4) attention / KV")
    if typer.confirm("Remove attention heads as well?", default=False):
        out.remove_attention_percent = typer.prompt(
            "percent of query heads (whole KV groups)", type=float, default=25.0)

    _banner("5) protection overrides")
    if typer.confirm(
            "Keep the default protection (embeddings, LM head, norms, rotary, "
            "MoE routers, adapters, tied weights)?", default=True):
        out.protect_categories = ()
        out.unprotect_categories = ()
    else:
        raw = typer.prompt("categories to UNPROTECT (comma separated)",
                           default="")
        out.unprotect_categories = tuple(x.strip() for x in raw.split(",") if x.strip())
        raw = typer.prompt("glob patterns to PROTECT (comma separated)", default="")
        out.protect = tuple(x.strip() for x in raw.split(",") if x.strip())
    return out


def compress_command(
    model: str = typer.Argument(..., help="HF model id or local checkpoint path"),
    output: str = typer.Option("compressed_model", "--output", "-o",
                               help="directory for the compressed checkpoint"),
    dataset: Optional[str] = typer.Option(
        None, "--dataset", help="DATASET MODE: HF dataset id or local .txt/.jsonl "
                                "with real calibration text"),
    dataset_mode: str = typer.Option("balanced", "--dataset-mode",
                                     help="fast | balanced | accurate | custom"),
    dataset_split: str = typer.Option("train", "--dataset-split"),
    dataset_field: str = typer.Option("text", "--dataset-field"),
    num_samples: Optional[int] = typer.Option(None, "--num-samples"),
    seq_len: Optional[int] = typer.Option(None, "--seq-len"),
    dataset_free: bool = typer.Option(
        False, "--dataset-free",
        help="DATASET-FREE MODE: use only data-free methods (no activations are "
             "invented)"),
    remove_percent: Optional[float] = typer.Option(
        None, "--remove-percent", help="remove this %% of all MLP neurons"),
    remove_count: Optional[int] = typer.Option(
        None, "--remove-count", help="remove exactly this many MLP neurons"),
    goal_neurons: Optional[int] = typer.Option(
        None, "--goal-neurons",
        help="aim for this many MLP neurons in the OUTPUT model"),
    goal_params: Optional[int] = typer.Option(
        None, "--goal-params",
        help="aim for this many parameters in the OUTPUT model"),
    allocation: str = typer.Option("global", "--allocation",
                                   help="global | uniform | hybrid"),
    scoring: str = typer.Option("weight_combined", "--scoring",
                                help="weight_l1|weight_l2|weight_combined|"
                                     "activation|activation_weighted"),
    remove_attention_percent: Optional[float] = typer.Option(
        None, "--remove-attention-percent"),
    layer_mode: str = typer.Option("search", "--layer-mode",
                                   help="search | none"),
    target_student_layers: int = typer.Option(
        2, "--target-student-layers",
        help="target student layer count; must be smaller than the teacher depth"),
    fuse: str = typer.Option("learned", "--fuse", help="none | exact | learned"),
    distill_steps: int = typer.Option(100, "--distill-steps"),
    distill_lr: float = typer.Option(5e-5, "--distill-lr"),
    distill_losses: str = typer.Option("auto", "--distill-losses",
                                       help="auto|all|kl|ce|hidden|kl+hidden"),
    beam_width: int = typer.Option(16, "--beam-width"),
    device: str = typer.Option("cpu", "--device"),
    dtype: str = typer.Option("float32", "--dtype",
                              help="auto|float32|float16|bfloat16"),
    trust_remote_code: bool = typer.Option(False, "--trust-remote-code"),
    revision: Optional[str] = typer.Option(None, "--revision"),
    protect: Optional[str] = typer.Option(None, "--protect",
                                          help="extra glob patterns to protect"),
    unprotect: Optional[str] = typer.Option(
        None, "--unprotect", help="category names or globs to unprotect"),
    no_validate: bool = typer.Option(False, "--no-validate"),
    interactive: bool = typer.Option(False, "--interactive", "-i",
                                     help="run the interactive wizard"),
) -> None:
    """Structurally compress a model: physically delete neurons, heads and
    layers, fusing or distilling where that is provable/useful."""
    given = [name for name, value in (("--remove-percent", remove_percent),
                                      ("--remove-count", remove_count),
                                      ("--goal-neurons", goal_neurons),
                                      ("--goal-params", goal_params))
             if value is not None]
    if len(given) > 1:
        typer.secho("error: choose exactly one of " + ", ".join(given),
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    opts = CompressionOptions(
        model=model, output=output, dataset=dataset or None,
        dataset_mode=dataset_mode, dataset_split=dataset_split,
        dataset_field=dataset_field, num_samples=num_samples, seq_len=seq_len,
        dataset_free=dataset_free, remove_percent=remove_percent,
        remove_count=remove_count, goal_neurons=goal_neurons,
        goal_params=goal_params, allocation=allocation, scoring=scoring,
        remove_attention_percent=remove_attention_percent,
        layer_mode=layer_mode, target_student_layers=target_student_layers,
        fuse=fuse, distill_steps=distill_steps, distill_lr=distill_lr,
        distill_losses=distill_losses, beam_width=beam_width, device=device,
        dtype=dtype, trust_remote_code=trust_remote_code, revision=revision,
        protect=tuple(x.strip() for x in (protect or "").split(",") if x.strip()),
        unprotect=tuple(x.strip() for x in (unprotect or "").split(",") if x.strip()),
        validate=not no_validate,
    )
    if interactive:
        opts = _wizard(opts)
    if not opts.dataset and not opts.dataset_free:
        typer.echo("No --dataset given: running in DATASET-FREE MODE "
                   "(weight/structure scoring only).")
        opts.dataset_free = True

    try:
        report = run_compression(model, opts, progress=typer.echo)
    except (CompressionError, ValueError, RuntimeError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    init = report["initial_model_report"]
    struct = report["structural_validation"]
    typer.echo("")
    typer.echo(f"parameters      {init.get('parameters'):,} -> "
               f"{struct.get('parameters'):,} "
               f"({struct.get('parameter_change_percent'):.2f}% smaller)")
    typer.echo(f"mlp neurons     {init.get('mlp_neuron_count'):,} -> "
               f"{struct.get('mlp_neurons'):,}")
    typer.echo(f"layers          {init.get('layers')} -> {struct.get('layers')}")
    goal = report.get("goal") or {}
    if goal.get("kind"):
        noun = "neurons" if goal["kind"] == "neurons" else "parameters"
        verdict = "at or under goal" if goal.get("within_goal") else "above goal"
        typer.echo(f"goal            {noun} {int(goal.get('goal', 0)):,} -> "
                   f"{int(goal.get('achieved', 0)):,} ({verdict}, "
                   f"{goal.get('relative_delta', 0.0) * 100:+.2f}%)")
    typer.echo(f"checkpoint      {report.get('output_dir')}")
    typer.echo(f"report          {report['report_files']['json']}")
    typer.echo(f"report          {report['report_files']['text']}")
