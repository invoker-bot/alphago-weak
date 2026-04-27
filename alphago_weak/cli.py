# -*- coding: utf-8 -*-

import shutil
import math
from collections import deque
from pathlib import Path
from typing import Optional

import click
import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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
    simulations: int = typer.Option(1600, "--simulations", help="MCTS simulations per self-play move."),
    max_moves: Optional[int] = typer.Option(None, "--max-moves", "--max_moves", help="Maximum moves per self-play game."),
    record_games: int = typer.Option(0, "--record-games", "--record_games", help="Self-play games to save as SGF per epoch."),
    record_dir: Optional[Path] = typer.Option(None, "--record-dir", "--record_dir", help="Directory for sampled self-play SGF records."),
    replay_buffer_size: int = typer.Option(8192, "--replay-buffer-size", "--replay_buffer_size", help="Maximum self-play positions retained for training."),
    train_batch_size: int = typer.Option(64, "--train-batch-size", "--train_batch_size", help="Self-play training batch size."),
    train_steps: Optional[int] = typer.Option(None, "--train-steps", "--train_steps", help="Maximum training batches per epoch."),
    learning_rate: float = typer.Option(0.01, "--learning-rate", "--learning_rate", help="Self-play optimizer learning rate."),
    board_size: int = typer.Option(19, "--board-size", "--board_size", help="Go board size."),
    komi: float = typer.Option(6.5, "--komi", help="Extra score given to white."),
):
    """Run the experimental AlphaGo Zero-style self-play loop."""
    if replay_buffer_size <= 0:
        raise click.BadParameter("replay-buffer-size must be greater than zero")
    if train_batch_size <= 0:
        raise click.BadParameter("train-batch-size must be greater than zero")
    if train_steps is not None and train_steps <= 0:
        raise click.BadParameter("train-steps must be greater than zero")
    bot = GTPAlphaGoWeakV0(board_size, komi, root=str(_root(ctx)))
    optimizer = bot.model.make_optimizer(learning_rate)
    replay_buffer = deque(maxlen=replay_buffer_size)
    output_dir = record_dir if record_dir is not None else _root(ctx) / "self-play"
    if record_games > 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    move_limit = max_moves if max_moves is not None else board_size * board_size * 2
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        epoch_task = progress.add_task("epochs", total=epochs)
        game_task = progress.add_task("games", total=steps)
        move_task = progress.add_task("moves", total=move_limit, visible=False)
        simulation_task = progress.add_task("simulations", total=simulations, visible=False)
        training_task = progress.add_task("training", total=1, visible=False)

        for epoch in range(epochs):
            progress.update(game_task, completed=0, total=steps, description=f"epoch {epoch} games")

            def record_game(game_idx, game_data):
                if game_idx < record_games:
                    game_data.to_sgf(str(output_dir / f"epoch-{epoch:04d}-game-{game_idx:04d}.sgf"))
                progress.advance(game_task, 1)

            def on_move(game_idx, move_idx, player, phase, pos):
                if phase == "start":
                    progress.update(
                        move_task,
                        completed=move_idx,
                        total=move_limit,
                        description=f"game {game_idx} moves",
                        visible=True,
                    )
                    progress.update(
                        simulation_task,
                        completed=0,
                        total=simulations,
                        description=f"move {move_idx} simulations",
                        visible=True,
                    )
                else:
                    progress.update(move_task, completed=move_idx + 1)
                    progress.update(simulation_task, visible=False)

            def on_simulation(advance):
                progress.advance(simulation_task, advance)

            inputs, policies, values = bot.self_play(
                steps,
                simulations,
                max_moves,
                game_recorder=record_game,
                move_callback=on_move,
                simulation_callback=on_simulation,
            )
            replay_buffer.extend(zip(inputs, policies, values))
            replay_samples = list(replay_buffer)
            train_inputs = [sample[0] for sample in replay_samples]
            train_policies = [sample[1] for sample in replay_samples]
            train_values = [sample[2] for sample in replay_samples]
            progress.update(move_task, visible=False)
            progress.update(simulation_task, visible=False)
            training_batches = max(math.ceil(len(train_inputs) / train_batch_size), 1)
            if train_steps is not None:
                training_batches = min(training_batches, train_steps)
            progress.update(
                training_task,
                completed=0,
                total=training_batches,
                description=f"training {len(train_inputs)} replay samples",
                visible=True,
            )
            dataset = bot.model.dataset_from_arrays(train_inputs, train_policies, train_values)
            loss, acc = bot.model.fit_step_from_dataset(
                dataset,
                batch_size=train_batch_size,
                max_batches=train_steps,
                optimizer=optimizer,
                learning_rate=learning_rate,
                progress_callback=lambda advance: progress.advance(training_task, advance),
            )
            progress.update(training_task, visible=False)
            bot.model.save()
            progress.advance(epoch_task, 1)
            progress.console.print(f"epoch {epoch}: Total Loss: {loss:.04f}, Policy Accuracy: {acc:.3%}")


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
