from __future__ import annotations

import typer

from .compress_cmd import compress_command
from .inspect_cmd import inspect_command

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Structural compression for Hugging Face language models.",
)

app.command(
    "inspect",
    help="Load and inspect a model's architecture without modifying it.",
)(inspect_command)

app.command(
    "compress",
    help="Physically compress a model (neurons, heads, layers) and report it.",
)(compress_command)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
