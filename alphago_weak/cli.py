# -*- coding: utf-8 -*-

import shutil
from pathlib import Path
from typing import Optional

import click
import typer

from alphago_weak.dataset import GameArchive
from alphago_weak.gtp import GTPClientBase
from alphago_weak.gtp.gtp_alphago_weak import GTPAlphaGoWeakV0
from alphago_weak.model.alpha_go_weak import AlphaGoWeak

VERSION = "AlphaGo Weak v1.0"
DEFAULT_ROOT = Path(".data")
DOWNLOAD_TYPES = tuple(GameArchive.FACTORY_DICT.keys())
BOT_TYPES = tuple(GTPClientBase.FACTORY_DICT.keys())

app = typer.Typer(
    help="AlphaGo Weak command-line tools.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool):
    if value:
        typer.echo(VERSION)
        raise typer.Exit()


def _choice(value: str, choices: tuple[str, ...], field: str) -> str:
    normalized = value.replace("-", "_")
    if normalized not in choices:
        raise click.BadParameter(f"{field} must be one of: {', '.join(choices)}")
    return normalized


def _root(ctx: typer.Context) -> Path:
    return ctx.obj["root"] if ctx.obj else DEFAULT_ROOT


@app.callback()
def main(
    ctx: typer.Context,
    root: Path = typer.Option(DEFAULT_ROOT, "--root", "-r", help="Root directory for datasets, caches, and model weights."),
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        callback=_version_callback,
        is_eager=True,
        help="Show the AlphaGo Weak version and exit.",
    ),
):
    ctx.obj = {"root": root}


@app.command()
def download(
    ctx: typer.Context,
    type_: str = typer.Option(DOWNLOAD_TYPES[0], "--type", "-t", help=f"Dataset source. Choices: {', '.join(DOWNLOAD_TYPES)}."),
    force: bool = typer.Option(False, "--force", "-f", help="Download and unpack archives even when local files exist."),
):
    """Download and unpack training datasets."""
    archive_type = _choice(type_, DOWNLOAD_TYPES, "type")
    archive = GameArchive.FACTORY_DICT[archive_type](str(_root(ctx)))
    try:
        archive.download(force)
    except Exception as exc:
        raise click.ClickException(f"download failed: {exc}") from exc


@app.command()
def clean(ctx: typer.Context):
    """Remove the configured dataset/cache/model root directory."""
    root = _root(ctx)
    if not root.exists():
        typer.echo(f"{root} does not exist.")
        return
    shutil.rmtree(root)
    typer.echo(f"Removed {root}.")


@app.command()
def train(
    ctx: typer.Context,
    init: int = typer.Option(0, "--init", help="Preprocess this many cache shards before training."),
    batch_size: int = typer.Option(512, "--batch-size", "--batch_size", help="Training batch size."),
    epochs: int = typer.Option(100, "--epochs", help="Maximum training epochs."),
):
    """Train AlphaGo Weak from preprocessed SGF game data."""
    root = str(_root(ctx))
    archive = GameArchive(root)
    if init > 0:
        AlphaGoWeak.preprocess_archive(archive, init)
    dataset, steps = AlphaGoWeak.dataset_from_preprocess(archive)
    model = AlphaGoWeak(root=root)
    model.fit_from_dataset(dataset, steps, epochs, batch_size)


@app.command("self-play")
@app.command("self_play", hidden=True)
def self_play(
    ctx: typer.Context,
    steps: int = typer.Option(4, "--steps", help="Self-play matches per epoch."),
    epochs: int = typer.Option(100, "--epochs", help="Maximum self-play epochs."),
    board_size: int = typer.Option(19, "--board-size", "--board_size", help="Go board size."),
    komi: float = typer.Option(6.5, "--komi", help="Extra score given to white."),
):
    """Run the experimental MCTS self-play fine-tuning loop."""
    bot = GTPAlphaGoWeakV0(board_size, komi, root=str(_root(ctx)))
    for epoch in range(epochs):
        inputs, policies, values = bot.self_play(steps)
        dataset = bot.model.dataset_from_arrays(inputs, policies, values)
        loss, acc = bot.model.fit_step_from_dataset(dataset)
        bot.model.save()
        typer.echo(f"epoch {epoch}: Total Loss: {loss:.04f}, Policy Accuracy: {acc:.3%}")


@app.command()
def evaluate(
    black: str = typer.Option(..., "--black", "-b", help=f"Black bot. Choices: {', '.join(BOT_TYPES)}."),
    white: str = typer.Option(..., "--white", "-w", help=f"White bot. Choices: {', '.join(BOT_TYPES)}."),
    board_size: int = typer.Option(19, "--board-size", "--board_size", help="Go board size."),
    komi: float = typer.Option(6.5, "--komi", help="Extra score given to white."),
    count: int = typer.Option(100, "--count", "-c", help="Number of matches to evaluate."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional SGF output directory."),
    multiprocessing: bool = typer.Option(False, "--multiprocessing", help="Use multiprocessing for match evaluation."),
):
    """Evaluate two GTP bots against each other."""
    black_type = _choice(black, BOT_TYPES, "black")
    white_type = _choice(white, BOT_TYPES, "white")
    output_dir = str(output) if output is not None else None
    black_v = GTPClientBase.evaluate(black_type, white_type, board_size, count, komi, output_dir, multiprocessing)
    typer.echo("the winning rate of black bot (%s): %0.3f" % (black_type, black_v))
    typer.echo("the winning rate of white bot (%s): %0.3f" % (white_type, 1 - black_v))


@app.command()
def play(
    ctx: typer.Context,
    type_: str = typer.Option(..., "--type", "-t", help=f"Bot type. Choices: {', '.join(BOT_TYPES)}."),
    board_size: int = typer.Option(19, "--board-size", "--board_size", help="Go board size."),
    komi: float = typer.Option(6.5, "--komi", help="Extra score given to white."),
):
    """Start a GTP command loop for a bot."""
    bot_type = _choice(type_, BOT_TYPES, "type")
    bot = GTPClientBase.FACTORY_DICT[bot_type](board_size, komi, {"root": str(_root(ctx))})
    bot.cmdloop()


if __name__ == "__main__":
    app()
