# -*- coding: utf-8 -*-

import copy
import html
import json
import math
import random
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from alphago_weak.board import GoBoard, GoPlayer
from alphago_weak.dataset import GameData
from alphago_weak.gtp.gtp_alphago_weak import GTPAlphaGoWeakV0
from alphago_weak.model.alpha_go_weak import AlphaGoWeakV0
from alphago_weak.model.mcts import GoMCTSTree
from alphago_weak.zero_report import (
    ZeroReportConfig,
    augment_samples,
    json_default,
    load_snapshot,
    play_random_opening,
    resolve_torch_device,
    run_parallel_self_play,
    set_seed,
    snapshot_state,
)

REPLAY_BUFFER_FILENAME = "replay_buffers.npz"
IN_PROGRESS_CANDIDATE_FILENAME = "candidate-in-progress.pt"


@dataclass
class ZeroLadderConfig:
    root: Path
    output: Path
    board_size: int = 7
    komi: float = 2.5
    generations: int = 20
    max_attempts: Optional[int] = None
    candidate_epochs: int = 2
    gate_training_rounds: int = 4
    games_per_epoch: int = 4
    simulations: int = 12
    max_moves: int = 98
    eval_games: int = 20
    eval_simulations: int = 8
    eval_opening_moves: int = 4
    gate_recheck_games: int = 20
    gate_win_rate: float = 0.55
    calibrate_komi: bool = True
    komi_window: int = 2
    komi_step: float = 1.0
    komi_eval_games: int = 6
    train_batch_size: int = 32
    train_steps: int = 96
    replay_buffer_size: int = 8192
    learning_rate: float = 0.001
    optimizer_name: str = "adam"
    augment_training: bool = True
    num_filters: int = 16
    workers: int = 4
    inference_batch_size: int = 16
    self_play_device: str = "auto"
    shared_inference: bool = True
    seed: int = 7
    force: bool = False
    resume: bool = False


def run_zero_ladder(config: ZeroLadderConfig, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    set_seed(config.seed)
    work_root = config.root / f"zero-ladder-{config.board_size}x{config.board_size}"
    if config.force and config.resume:
        raise ValueError("force and resume cannot be used together")
    if work_root.exists():
        if config.resume:
            pass
        elif not config.force:
            raise FileExistsError(f"{work_root} already exists; pass --force to rerun from scratch")
        else:
            shutil.rmtree(work_root)
    elif config.resume:
        raise FileNotFoundError(f"{work_root} does not exist; cannot resume")
    work_root.mkdir(parents=True, exist_ok=True)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    model_dir = config.output.parent / "models"
    sgf_dir = config.output.parent / "sgf"
    model_dir.mkdir(parents=True, exist_ok=True)
    sgf_dir.mkdir(parents=True, exist_ok=True)
    replay_path = config.output.parent / REPLAY_BUFFER_FILENAME
    in_progress_candidate_path = model_dir / IN_PROGRESS_CANDIDATE_FILENAME

    device = resolve_torch_device(config.self_play_device)
    if config.resume:
        metrics_path = config.output.parent / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"{metrics_path} does not exist; cannot resume")
        previous_report = json.loads(metrics_path.read_text(encoding="utf-8"))
        records = list(previous_report.get("attempts", []))
        generation_paths = list(previous_report.get("generations", []))
        if not generation_paths:
            raise ValueError("resume report does not contain generation checkpoints")
        summary = previous_report.get("summary", {})
        accepted = int(summary.get("accepted_generations", 0))
        attempt = int(summary.get("attempts", len(records)))
        cumulative_elo = float(summary.get("final_elo", 0.0))
        champion_state = load_snapshot_file(Path(generation_paths[-1]["path"]))
        replay_buffer, candidate_buffer, replay_restored = load_replay_buffers(replay_path, config.replay_buffer_size)
        if not replay_restored:
            report_progress(progress_callback, {"event": "resume_replay_missing", "path": str(replay_path)})
        if records and not records[-1].get("accepted") and records[-1].get("candidate_path"):
            candidate_state = load_snapshot_file(Path(records[-1]["candidate_path"]))
        else:
            candidate_state = champion_state
            candidate_buffer = deque(replay_buffer, maxlen=config.replay_buffer_size)
        if in_progress_candidate_path.exists():
            candidate_state = load_snapshot_file(in_progress_candidate_path)
            report_progress(progress_callback, {"event": "resume_in_progress_candidate", "path": str(in_progress_candidate_path)})
    else:
        bot = GTPAlphaGoWeakV0(
            config.board_size,
            config.komi,
            root=str(work_root),
            num_filters=config.num_filters,
            device=device,
            inference_batch_size=config.inference_batch_size,
        )
        champion_state = snapshot_state(bot.model)
        save_snapshot_file(model_dir / "generation-000.pt", champion_state, config, generation=0, elo=0.0)
        accepted = 0
        attempt = 0
        cumulative_elo = 0.0
        records: List[Dict[str, Any]] = []
        generation_paths = [{"generation": 0, "elo": 0.0, "path": str(model_dir / "generation-000.pt")}]
        replay_buffer = deque(maxlen=config.replay_buffer_size)
        candidate_state = champion_state
        candidate_buffer = deque(maxlen=config.replay_buffer_size)
    max_attempts = config.max_attempts if config.max_attempts is not None else max(config.generations * 3, config.generations)
    worker_count = min(max(config.workers, 1), max(config.games_per_epoch, 1))
    epochs_before_gate = config.candidate_epochs * config.gate_training_rounds

    while accepted < config.generations and attempt < max_attempts:
        attempt += 1
        source_generation = accepted
        report_progress(
            progress_callback,
            {
                "event": "attempt_start",
                "attempt": attempt,
                "accepted": accepted,
                "target": config.generations,
                "epochs_before_gate": epochs_before_gate,
            },
        )
        calibration = calibrate_relative_komi(champion_state, config, work_root, config.seed + attempt * 100000)
        report_progress(progress_callback, {"event": "komi", "attempt": attempt, "komi": calibration["komi"]})
        generation_config = to_zero_report_config(config, calibration["komi"])
        candidate = GTPAlphaGoWeakV0(
            config.board_size,
            calibration["komi"],
            root=str(work_root),
            num_filters=config.num_filters,
            device=device,
            inference_batch_size=config.inference_batch_size,
        )
        load_snapshot(candidate.model, candidate_state)
        optimizer = candidate.model.make_optimizer(config.learning_rate, optimizer=config.optimizer_name)
        attempt_buffer = deque(candidate_buffer, maxlen=config.replay_buffer_size)
        training_metrics: List[Dict[str, Any]] = []

        for epoch in range(1, epochs_before_gate + 1):
            epoch_seed = config.seed + attempt * 1000 + epoch
            inputs, policies, values, games = run_ladder_self_play_epoch(
                candidate,
                generation_config,
                work_root,
                epoch_seed,
                worker_count,
                progress_callback,
                attempt,
                epoch,
            )
            attempt_buffer.extend(zip(inputs, policies, values))
            train_inputs, train_policies, train_values = samples_to_arrays(attempt_buffer)
            if config.augment_training:
                train_inputs, train_policies, train_values = augment_samples(train_inputs, train_policies, train_values, config.board_size)
            dataset = candidate.model.dataset_from_arrays(train_inputs, train_policies, train_values)
            loss, accuracy = candidate.model.fit_step_from_dataset(
                dataset,
                batch_size=config.train_batch_size,
                max_batches=config.train_steps,
                optimizer=optimizer,
                learning_rate=config.learning_rate,
            )
            if games:
                games[0].to_sgf(str(sgf_dir / f"attempt-{attempt:03d}-epoch-{epoch:03d}.sgf"))
            training_metrics.append(
                {
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": accuracy,
                    "samples": len(inputs),
                    "replay_size": len(attempt_buffer),
                    "train_size": len(train_inputs),
                    "avg_moves": float(np.mean([len(game.sequence) for game in games])) if games else 0.0,
                }
            )
            report_progress(
                progress_callback,
                {
                    "event": "epoch",
                    "attempt": attempt,
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": accuracy,
                    "samples": len(inputs),
                },
            )
            epoch_candidate_state = snapshot_state(candidate.model)
            save_snapshot_file(
                in_progress_candidate_path,
                epoch_candidate_state,
                config,
                generation=accepted + 1,
                elo=cumulative_elo,
            )
            save_replay_buffers(replay_path, replay_buffer, attempt_buffer, accepted, attempt)

        candidate_state = snapshot_state(candidate.model)
        match = evaluate_generation_match(
            candidate_state,
            champion_state,
            generation_config,
            work_root,
            config.seed + attempt * 100000 + 50000,
            progress_callback=lambda played, total, attempt=attempt: report_progress(
                progress_callback,
                {"event": "gate_progress", "attempt": attempt, "played": played, "total": total},
            ),
        )
        match["gate_assessment"] = assess_gate_result(
            match["candidate_wins"],
            match["games"],
            config.gate_win_rate,
        )
        if match["gate_assessment"]["label"] == "borderline" and config.gate_recheck_games > 0:
            report_progress(
                progress_callback,
                {
                    "event": "gate_recheck_start",
                    "attempt": attempt,
                    "games": config.gate_recheck_games,
                    "ci_low": match["gate_assessment"]["ci_low"],
                    "ci_high": match["gate_assessment"]["ci_high"],
                },
            )
            recheck_config = copy.copy(generation_config)
            recheck_config.eval_games = config.gate_recheck_games
            recheck = evaluate_generation_match(
                candidate_state,
                champion_state,
                recheck_config,
                work_root,
                config.seed + attempt * 100000 + 70000,
                progress_callback=lambda played, total, attempt=attempt: report_progress(
                    progress_callback,
                    {"event": "gate_recheck_progress", "attempt": attempt, "played": played, "total": total},
                ),
            )
            match = combine_match_results(match, recheck)
            match["gate_assessment"] = assess_gate_result(
                match["candidate_wins"],
                match["games"],
                config.gate_win_rate,
            )
            match["recheck"] = {
                "games": config.gate_recheck_games,
                "trigger": "borderline_initial_gate",
            }
        elo_delta = elo_delta_from_win_rate(match["candidate_win_rate"])
        accepted_candidate = match["candidate_win_rate"] >= config.gate_win_rate
        candidate_path = model_dir / f"candidate-attempt-{attempt:03d}.pt"
        save_snapshot_file(candidate_path, candidate_state, config, generation=accepted + 1, elo=cumulative_elo + elo_delta)

        if accepted_candidate:
            accepted += 1
            cumulative_elo += elo_delta
            champion_state = candidate_state
            replay_buffer = attempt_buffer
            candidate_state = champion_state
            candidate_buffer = deque(replay_buffer, maxlen=config.replay_buffer_size)
            champion_path = model_dir / f"generation-{accepted:03d}.pt"
            save_snapshot_file(champion_path, champion_state, config, generation=accepted, elo=cumulative_elo)
            generation_paths.append({"generation": accepted, "elo": cumulative_elo, "path": str(champion_path)})
        else:
            candidate_buffer = attempt_buffer
            champion_path = None

        records.append(
            {
                "attempt": attempt,
                "from_generation": source_generation,
                "to_generation": accepted if accepted_candidate else None,
                "accepted": accepted_candidate,
                "komi": calibration["komi"],
                "calibration": calibration,
                "epochs_before_gate": epochs_before_gate,
                "training": training_metrics,
                "match": match,
                "elo_delta": elo_delta,
                "cumulative_elo": cumulative_elo,
                "candidate_path": str(candidate_path),
                "champion_path": str(champion_path) if champion_path is not None else None,
            }
        )
        report_progress(
            progress_callback,
            {
                "event": "attempt_result",
                "attempt": attempt,
                "accepted": accepted_candidate,
                "accepted_generations": accepted,
                "candidate_win_rate": match["candidate_win_rate"],
                "elo_delta": elo_delta,
                "cumulative_elo": cumulative_elo,
            },
        )
        save_replay_buffers(replay_path, replay_buffer, candidate_buffer, accepted, attempt)
        write_ladder_report(config, work_root, generation_paths, records, accepted, attempt, cumulative_elo)

    return write_ladder_report(config, work_root, generation_paths, records, accepted, attempt, cumulative_elo)


def write_ladder_report(config: ZeroLadderConfig, work_root: Path, generation_paths: Sequence[Dict[str, Any]],
                        records: Sequence[Dict[str, Any]], accepted: int, attempt: int,
                        cumulative_elo: float) -> Dict[str, Any]:
    report = {
        "config": config_to_dict(config, work_root),
        "generations": list(generation_paths),
        "attempts": list(records),
        "summary": {
            "accepted_generations": accepted,
            "attempts": attempt,
            "target_generations": config.generations,
            "final_elo": cumulative_elo,
            "gate_win_rate": config.gate_win_rate,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (config.output.parent / "metrics.json").write_text(json.dumps(report, default=json_default, indent=2), encoding="utf-8")
    config.output.write_text(render_ladder_report(report), encoding="utf-8")
    return report


def run_ladder_self_play_epoch(candidate: GTPAlphaGoWeakV0, generation_config: ZeroReportConfig, work_root: Path,
                               epoch_seed: int, worker_count: int,
                               progress_callback: Optional[Callable[[Dict[str, Any]], None]],
                               attempt: int, epoch: int):
    if worker_count <= 1:
        return run_ladder_self_play_sequential(candidate, generation_config)
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            return run_parallel_self_play(
                generation_config,
                work_root,
                snapshot_state(candidate.model),
                epoch_seed,
                executor,
                worker_count,
            )
    except Exception as exc:
        report_progress(
            progress_callback,
            {
                "event": "parallel_fallback",
                "attempt": attempt,
                "epoch": epoch,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return run_ladder_self_play_sequential(candidate, generation_config)


def run_ladder_self_play_sequential(candidate: GTPAlphaGoWeakV0, generation_config: ZeroReportConfig):
    games: List[GameData] = []
    inputs, policies, values = candidate.self_play(
        count=generation_config.games_per_epoch,
        simulations=generation_config.simulations,
        max_moves=generation_config.max_moves,
        inference_batch_size=generation_config.inference_batch_size,
        game_recorder=lambda _idx, game: games.append(game),
    )
    return inputs, policies, values, games


def samples_to_arrays(samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
    return [sample[0] for sample in samples], [sample[1] for sample in samples], [sample[2] for sample in samples]


def save_replay_buffers(path: Path, replay_buffer: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                        candidate_buffer: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                        accepted: int, attempt: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    arrays: Dict[str, Any] = {
        "format_version": np.array([1], dtype=np.int64),
        "accepted": np.array([accepted], dtype=np.int64),
        "attempt": np.array([attempt], dtype=np.int64),
    }
    arrays.update(buffer_to_npz_arrays("replay", replay_buffer))
    arrays.update(buffer_to_npz_arrays("candidate", candidate_buffer))
    with temp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp_path.replace(path)


def load_replay_buffers(path: Path, maxlen: int):
    if not path.exists():
        return deque(maxlen=maxlen), deque(maxlen=maxlen), False
    with np.load(path, allow_pickle=False) as data:
        replay_buffer = npz_arrays_to_buffer(data["replay_inputs"], data["replay_policies"], data["replay_values"], maxlen)
        candidate_buffer = npz_arrays_to_buffer(data["candidate_inputs"], data["candidate_policies"], data["candidate_values"], maxlen)
    return replay_buffer, candidate_buffer, True


def buffer_to_npz_arrays(prefix: str, samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    inputs, policies, values = samples_to_arrays(samples)
    return {
        f"{prefix}_inputs": np.asarray(inputs, dtype=np.float32),
        f"{prefix}_policies": np.asarray(policies, dtype=np.float32),
        f"{prefix}_values": np.asarray(values, dtype=np.float32),
    }


def npz_arrays_to_buffer(inputs: np.ndarray, policies: np.ndarray, values: np.ndarray, maxlen: int):
    restored = deque(maxlen=maxlen)
    for sample in zip(inputs[-maxlen:], policies[-maxlen:], values[-maxlen:]):
        restored.append(sample)
    return restored


def report_progress(progress_callback: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]):
    if progress_callback is not None:
        progress_callback(event)


def to_zero_report_config(config: ZeroLadderConfig, komi: float) -> ZeroReportConfig:
    return ZeroReportConfig(
        root=config.root,
        output=config.output,
        board_size=config.board_size,
        komi=komi,
        epochs=config.candidate_epochs * config.gate_training_rounds,
        games_per_epoch=config.games_per_epoch,
        simulations=config.simulations,
        max_moves=config.max_moves,
        eval_games=config.eval_games,
        eval_simulations=config.eval_simulations,
        eval_opening_moves=config.eval_opening_moves,
        train_batch_size=config.train_batch_size,
        train_steps=config.train_steps,
        replay_buffer_size=config.replay_buffer_size,
        learning_rate=config.learning_rate,
        optimizer_name=config.optimizer_name,
        augment_training=config.augment_training,
        num_filters=config.num_filters,
        workers=config.workers,
        inference_batch_size=config.inference_batch_size,
        self_play_device=config.self_play_device,
        shared_inference=config.shared_inference,
        seed=config.seed,
        force=config.force,
    )


def config_to_dict(config: ZeroLadderConfig, work_root: Path) -> Dict[str, Any]:
    data = {key: value for key, value in config.__dict__.items() if key not in {"root", "output"}}
    data["root"] = str(config.root)
    data["work_root"] = str(work_root)
    data["output"] = str(config.output)
    data["replay_buffer_path"] = str(config.output.parent / REPLAY_BUFFER_FILENAME)
    data["in_progress_candidate_path"] = str(config.output.parent / "models" / IN_PROGRESS_CANDIDATE_FILENAME)
    data["device"] = resolve_torch_device(config.self_play_device)
    return data


def save_snapshot_file(path: Path, state: Dict[str, torch.Tensor], config: ZeroLadderConfig, generation: int, elo: float):
    torch.save(
        {
            "model_state_dict": copy.deepcopy(state),
            "size": config.board_size,
            "num_filters": config.num_filters,
            "generation": generation,
            "elo": elo,
        },
        path,
    )


def load_snapshot_file(path: Path) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"{path} is not a zero ladder checkpoint")
    return checkpoint["model_state_dict"]


def elo_delta_from_win_rate(win_rate: float) -> float:
    p = min(max(float(win_rate), 1e-6), 1.0 - 1e-6)
    return 400.0 * math.log10(p / (1.0 - p))


def wilson_score_interval(wins: int, games: int, z: float = 1.96) -> Tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    p = wins / games
    denominator = 1.0 + z * z / games
    center = p + z * z / (2.0 * games)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * games)) / games)
    return max(0.0, (center - margin) / denominator), min(1.0, (center + margin) / denominator)


def assess_gate_result(wins: int, games: int, threshold: float) -> Dict[str, Any]:
    low, high = wilson_score_interval(wins, games)
    win_rate = wins / games if games else 0.0
    if low >= threshold:
        label = "clear_pass"
    elif high < threshold:
        label = "clear_reject"
    else:
        label = "borderline"
    return {
        "win_rate": win_rate,
        "threshold": threshold,
        "ci_low": low,
        "ci_high": high,
        "label": label,
    }


def combine_match_results(first: Dict[str, Any], second: Dict[str, Any]) -> Dict[str, Any]:
    games = int(first["games"]) + int(second["games"])
    if games <= 0:
        return empty_match_result()
    candidate_black_games = int(first["candidate_black_games"]) + int(second["candidate_black_games"])
    candidate_white_games = int(first["candidate_white_games"]) + int(second["candidate_white_games"])
    candidate_wins = int(first["candidate_wins"]) + int(second["candidate_wins"])
    candidate_black_wins = int(first["candidate_black_wins"]) + int(second["candidate_black_wins"])
    candidate_white_wins = int(first["candidate_white_wins"]) + int(second["candidate_white_wins"])
    black_wins = int(first["black_wins"]) + int(second["black_wins"])
    return {
        "games": games,
        "candidate_wins": candidate_wins,
        "candidate_win_rate": candidate_wins / games,
        "candidate_avg_margin": weighted_average(first, second, "candidate_avg_margin"),
        "candidate_black_games": candidate_black_games,
        "candidate_black_wins": candidate_black_wins,
        "candidate_black_win_rate": candidate_black_wins / candidate_black_games if candidate_black_games else 0.0,
        "candidate_white_games": candidate_white_games,
        "candidate_white_wins": candidate_white_wins,
        "candidate_white_win_rate": candidate_white_wins / candidate_white_games if candidate_white_games else 0.0,
        "black_wins": black_wins,
        "black_win_rate": black_wins / games,
        "avg_black_margin": weighted_average(first, second, "avg_black_margin"),
        "avg_moves": weighted_average(first, second, "avg_moves"),
    }


def weighted_average(first: Dict[str, Any], second: Dict[str, Any], key: str) -> float:
    total_games = int(first["games"]) + int(second["games"])
    if total_games <= 0:
        return 0.0
    return (float(first[key]) * int(first["games"]) + float(second[key]) * int(second["games"])) / total_games


def calibrate_relative_komi(state: Dict[str, torch.Tensor], config: ZeroLadderConfig, work_root: Path, seed: int) -> Dict[str, Any]:
    if not config.calibrate_komi or config.komi_eval_games <= 0:
        return {"komi": config.komi, "enabled": False, "candidates": []}
    candidates = []
    for idx, offset in enumerate(range(-config.komi_window, config.komi_window + 1)):
        komi = config.komi + offset * config.komi_step
        eval_config = to_zero_report_config(config, komi)
        eval_config.eval_games = config.komi_eval_games
        result = evaluate_generation_match(state, state, eval_config, work_root, seed + idx)
        candidates.append(
            {
                "komi": komi,
                "black_win_rate": result["black_win_rate"],
                "avg_black_margin": result["avg_black_margin"],
                "balance_error": abs(result["black_win_rate"] - 0.5),
            }
        )
    best = min(candidates, key=lambda item: (item["balance_error"], abs(item["avg_black_margin"])))
    return {"komi": best["komi"], "enabled": True, "candidates": candidates}


def evaluate_generation_match(candidate_state: Dict[str, torch.Tensor], baseline_state: Dict[str, torch.Tensor],
                              config: ZeroReportConfig, work_root: Path, seed: int,
                              progress_callback: Optional[Callable[[int, int], None]] = None) -> Dict[str, Any]:
    if config.eval_games <= 0:
        return empty_match_result()
    device = resolve_torch_device(config.self_play_device)
    candidate = AlphaGoWeakV0(
        name="zero_ladder_candidate",
        root=str(work_root),
        size=config.board_size,
        num_filters=config.num_filters,
        device=device,
    )
    baseline = AlphaGoWeakV0(
        name="zero_ladder_baseline",
        root=str(work_root),
        size=config.board_size,
        num_filters=config.num_filters,
        device=device,
    )
    load_snapshot(candidate, candidate_state)
    load_snapshot(baseline, baseline_state)
    set_seed(seed)
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    candidate_wins = 0
    candidate_black_games = 0
    candidate_black_wins = 0
    candidate_white_games = 0
    candidate_white_wins = 0
    black_wins = 0
    candidate_margins: List[float] = []
    black_margins: List[float] = []
    lengths: List[int] = []
    played_games = 0

    while played_games < config.eval_games:
        opening_board = GoBoard(config.board_size)
        next_player, opening_passes, opening_length = play_random_opening(
            opening_board,
            config.eval_opening_moves,
            GoPlayer.black,
            py_rng,
        )
        for candidate_player in shuffled_players(rng):
            if played_games >= config.eval_games:
                break
            board = opening_board.copy()
            player = next_player
            consecutive_passes = opening_passes
            moves = opening_length
            candidate_tree = GoMCTSTree(
                board,
                candidate.state_calculator,
                candidate.policy_evaluator,
                candidate.value_evaluator,
                player=player,
                komi=config.komi,
                batch_state_calculator=candidate.state_calculator_batch,
            )
            baseline_tree = GoMCTSTree(
                board,
                baseline.state_calculator,
                baseline.policy_evaluator,
                baseline.value_evaluator,
                player=player,
                komi=config.komi,
                batch_state_calculator=baseline.state_calculator_batch,
            )
            while consecutive_passes < 2 and moves < config.max_moves:
                if player == candidate_player:
                    pos = candidate_tree.evaluate(
                        player,
                        num=config.eval_simulations,
                        temperature=0.0,
                        dirichlet_alpha=0.0,
                        inference_batch_size=config.inference_batch_size,
                    )
                else:
                    pos = baseline_tree.evaluate(
                        player,
                        num=config.eval_simulations,
                        temperature=0.0,
                        dirichlet_alpha=0.0,
                        inference_batch_size=config.inference_batch_size,
                    )
                board.play(player, pos)
                candidate_tree.play(player, pos)
                baseline_tree.play(player, pos)
                consecutive_passes = consecutive_passes + 1 if pos is None else 0
                player = player.other
                moves += 1
            black_margin = board.score(GoPlayer.black, config.komi)
            winner = GoPlayer.black if black_margin > 0 else GoPlayer.white
            candidate_won = winner == candidate_player
            candidate_wins += int(candidate_won)
            black_wins += int(winner == GoPlayer.black)
            if candidate_player == GoPlayer.black:
                candidate_black_games += 1
                candidate_black_wins += int(candidate_won)
                candidate_margins.append(black_margin)
            else:
                candidate_white_games += 1
                candidate_white_wins += int(candidate_won)
                candidate_margins.append(-black_margin)
            black_margins.append(black_margin)
            lengths.append(moves)
            played_games += 1
            if progress_callback is not None:
                progress_callback(played_games, config.eval_games)

    return {
        "games": config.eval_games,
        "candidate_wins": candidate_wins,
        "candidate_win_rate": candidate_wins / config.eval_games,
        "candidate_avg_margin": float(np.mean(candidate_margins)),
        "candidate_black_games": candidate_black_games,
        "candidate_black_wins": candidate_black_wins,
        "candidate_black_win_rate": candidate_black_wins / candidate_black_games if candidate_black_games else 0.0,
        "candidate_white_games": candidate_white_games,
        "candidate_white_wins": candidate_white_wins,
        "candidate_white_win_rate": candidate_white_wins / candidate_white_games if candidate_white_games else 0.0,
        "black_wins": black_wins,
        "black_win_rate": black_wins / config.eval_games,
        "avg_black_margin": float(np.mean(black_margins)),
        "avg_moves": float(np.mean(lengths)),
    }


def shuffled_players(rng: np.random.Generator) -> Tuple[GoPlayer, GoPlayer]:
    return (GoPlayer.black, GoPlayer.white) if bool(rng.integers(0, 2)) else (GoPlayer.white, GoPlayer.black)


def empty_match_result() -> Dict[str, Any]:
    return {
        "games": 0,
        "candidate_wins": 0,
        "candidate_win_rate": 0.0,
        "candidate_avg_margin": 0.0,
        "candidate_black_games": 0,
        "candidate_black_wins": 0,
        "candidate_black_win_rate": 0.0,
        "candidate_white_games": 0,
        "candidate_white_wins": 0,
        "candidate_white_win_rate": 0.0,
        "black_wins": 0,
        "black_win_rate": 0.0,
        "avg_black_margin": 0.0,
        "avg_moves": 0.0,
    }


def render_ladder_report(report: Dict[str, Any]) -> str:
    config = report["config"]
    summary = report["summary"]
    rows = "\n".join(render_attempt_row(attempt) for attempt in report["attempts"])
    chart = render_elo_chart(report["generations"])
    analysis = render_progress_analysis(report)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AlphaGo Weak Zero Ladder</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f7f8fa; }}
header {{ padding: 28px 36px; background: #111827; color: white; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 26px 18px 48px; }}
h1, h2 {{ margin: 0 0 12px; }}
p {{ line-height: 1.55; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
.metric {{ font-size: 28px; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e2e8f0; }}
th, td {{ padding: 9px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
th {{ background: #eef2f7; }}
.accepted {{ color: #12783a; font-weight: 700; }}
.rejected {{ color: #a33a2f; font-weight: 700; }}
.note {{ color: #526071; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
.lang-switch {{ display: flex; gap: 8px; margin-top: 16px; }}
.lang-switch button {{ border: 1px solid #475569; border-radius: 6px; background: #1f2937; color: white; padding: 7px 12px; cursor: pointer; }}
.lang-switch button.active {{ background: #ffffff; color: #111827; border-color: #ffffff; }}
</style>
<script>
function setLanguage(lang) {{
  const key = lang === "en" ? "en" : "zh";
  document.documentElement.lang = key === "en" ? "en" : "zh-CN";
  document.querySelectorAll("[data-zh][data-en]").forEach((node) => {{
    node.textContent = key === "en" ? node.dataset.en : node.dataset.zh;
  }});
  document.querySelectorAll(".lang-switch button").forEach((button) => {{
    button.classList.toggle("active", button.dataset.lang === key);
  }});
  localStorage.setItem("zeroLadderLanguage", key);
}}
document.addEventListener("DOMContentLoaded", () => {{
  setLanguage(localStorage.getItem("zeroLadderLanguage") || "en");
}});
</script>
</head>
<body>
<header>
<h1 data-zh="AlphaGo Weak Zero Ladder 代际训练" data-en="AlphaGo Weak Zero Ladder">AlphaGo Weak Zero Ladder</h1>
<p data-zh="代际晋级训练报告。生成时间：{html.escape(report['generated_at'])}" data-en="Generation-gated training report. Generated at: {html.escape(report['generated_at'])}">Generation-gated training report. Generated at: {html.escape(report['generated_at'])}</p>
<div class="lang-switch" aria-label="Language">
<button type="button" data-lang="en" onclick="setLanguage('en')">English</button>
<button type="button" data-lang="zh" onclick="setLanguage('zh')">中文</button>
</div>
</header>
<main>
<section class="grid">
{metric_card("棋盘", "Board", f"{config['board_size']}x{config['board_size']}")}
{metric_card("目标晋级", "Target Advances", str(summary['target_generations']))}
{metric_card("实际晋级", "Accepted Advances", str(summary['accepted_generations']))}
{metric_card("最终 Elo", "Final Elo", f"{summary['final_elo']:.1f}")}
{metric_card("晋级阈值", "Gate Threshold", f"{summary['gate_win_rate']:.0%}")}
{metric_card("每次 Gate 训练", "Epochs per Gate", str(config['candidate_epochs'] * config['gate_training_rounds']))}
{metric_card("设备", "Device", html.escape(config['device']))}
</section>
<section class="card" style="margin-top:16px">
<h2 data-zh="方法" data-en="Method">Method</h2>
<p data-zh="候选模型轨道从当前 champion 权重初始化，使用自身 self-play 生成样本并持续训练；被拒后继续训练同一个候选，直到某次配对换先评估达到阈值才晋级为下一代。胜率会换算为 Elo 增量。报告不使用随机 bot 作为主指标。" data-en="The candidate track starts from the current champion weights and keeps training on its own self-play samples. If a candidate is rejected, the same candidate continues training; it advances only when a paired color-swapped gate match reaches the threshold. The win rate is converted into an Elo delta. Random-bot games are not used as the main strength metric.">The candidate track starts from the current champion weights and keeps training on its own self-play samples. If a candidate is rejected, the same candidate continues training; it advances only when a paired color-swapped gate match reaches the threshold. The win rate is converted into an Elo delta. Random-bot games are not used as the main strength metric.</p>
<p data-zh="相对贴目通过 champion 对自身的同代评估选择，使黑棋胜率尽量接近 50%。这样候选胜率主要反映模型差异，而不是固定贴目带来的颜色偏差。" data-en="Relative komi is selected by same-generation champion mirror matches so that black's win rate is as close to 50% as the sampled grid allows. Candidate win rate therefore reflects model differences more than fixed-komi color bias.">Relative komi is selected by same-generation champion mirror matches so that black's win rate is as close to 50% as the sampled grid allows. Candidate win rate therefore reflects model differences more than fixed-komi color bias.</p>
</section>
<section class="card" style="margin-top:16px">
<h2 data-zh="进步分析" data-en="Progress Analysis">Progress Analysis</h2>
{analysis}
</section>
<section class="card" style="margin-top:16px">
<h2 data-zh="Elo 曲线" data-en="Elo Curve">Elo Curve</h2>
{chart}
</section>
<section class="card" style="margin-top:16px">
<h2 data-zh="代际尝试" data-en="Generation Attempts">Generation Attempts</h2>
<table><thead><tr>{table_header()}</tr></thead><tbody>{rows}</tbody></table>
</section>
</main>
</body>
</html>
"""


def i18n_text(zh: str, en: str) -> str:
    zh_attr = html.escape(zh, quote=True)
    en_attr = html.escape(en, quote=True)
    return f'<span data-zh="{zh_attr}" data-en="{en_attr}">{html.escape(en)}</span>'


def metric_card(zh: str, en: str, value: str) -> str:
    return f'<div class="card"><div class="note">{i18n_text(zh, en)}</div><div class="metric">{value}</div></div>'


def table_header() -> str:
    headers = [
        ("Attempt", "Attempt"),
        ("结果", "Result"),
        ("晋级", "Advance"),
        ("Komi", "Komi"),
        ("候选胜率", "Candidate Win Rate"),
        ("Gate 可信度", "Gate Confidence"),
        ("候选黑/白", "Candidate B/W"),
        ("同代黑胜率", "Same-Gen Black Win"),
        ("Elo Δ", "Elo Δ"),
        ("累计 Elo", "Cumulative Elo"),
        ("Loss / Acc", "Loss / Acc"),
    ]
    return "".join(f"<th>{i18n_text(zh, en)}</th>" for zh, en in headers)


def render_progress_analysis(report: Dict[str, Any]) -> str:
    attempts = report["attempts"]
    if not attempts:
        return f"<p>{i18n_text('还没有完成 candidate attempt。', 'No candidate attempt has completed yet.')}</p>"
    first_train = next((metric for attempt in attempts for metric in attempt["training"]), None)
    last_train = next((metric for attempt in reversed(attempts) for metric in reversed(attempt["training"])), None)
    accepted = sum(1 for attempt in attempts if attempt["accepted"])
    rejected = len(attempts) - accepted
    summary = report["summary"]
    config = report["config"]
    if first_train is not None and last_train is not None:
        learning_zh = (
            f"训练指标从 loss {first_train['loss']:.4f} / policy accuracy {first_train['accuracy']:.1%} "
            f"改善到 loss {last_train['loss']:.4f} / policy accuracy {last_train['accuracy']:.1%}。"
        )
        learning_en = (
            f"Training moved from loss {first_train['loss']:.4f} / policy accuracy {first_train['accuracy']:.1%} "
            f"to loss {last_train['loss']:.4f} / policy accuracy {last_train['accuracy']:.1%}."
        )
    else:
        learning_zh = "没有可用的训练指标。"
        learning_en = "No training metrics are available."
    gate_zh = (
        f"本次运行完成 {summary['accepted_generations']}/{summary['target_generations']} 代晋级，"
        f"共 {len(attempts)} 次 attempt，其中 {accepted} 次通过、{rejected} 次拒绝。"
    )
    gate_en = (
        f"This run advanced {summary['accepted_generations']}/{summary['target_generations']} generations "
        f"in {len(attempts)} attempts: {accepted} accepted and {rejected} rejected."
    )
    cadence_zh = (
        f"每次正式 gate 前训练 {config['candidate_epochs']} x {config['gate_training_rounds']} = "
        f"{config['candidate_epochs'] * config['gate_training_rounds']} 个 self-play epoch；"
        f"如果仍出现过多拒绝，应继续增大 --gate-training-rounds 或 --candidate-epochs。"
    )
    cadence_en = (
        f"Before each formal gate, the candidate trains for {config['candidate_epochs']} x "
        f"{config['gate_training_rounds']} = {config['candidate_epochs'] * config['gate_training_rounds']} "
        f"self-play epochs; if rejections are still high, increase --gate-training-rounds or --candidate-epochs."
    )
    noise_zh = (
        f"评估每次 gate 使用 {config['eval_games']} 局，报告中的 95% Wilson 区间用于标记 clear/borderline；"
        f"更可靠的强度估计需要增加 --eval-games 和 --eval-simulations。"
    )
    noise_en = (
        f"Each gate used {config['eval_games']} evaluation games. The 95% Wilson interval marks clear vs borderline gates; "
        f"use higher --eval-games and --eval-simulations for stronger estimates."
    )
    return (
        f"<p>{i18n_text(gate_zh, gate_en)}</p>"
        f"<p>{i18n_text(cadence_zh, cadence_en)}</p>"
        f"<p>{i18n_text(learning_zh, learning_en)}</p>"
        f"<p>{i18n_text(noise_zh, noise_en)}</p>"
    )


def render_attempt_row(attempt: Dict[str, Any]) -> str:
    match = attempt["match"]
    last_train = attempt["training"][-1] if attempt["training"] else {"loss": 0.0, "accuracy": 0.0}
    status = i18n_text("通过", "Accepted") if attempt["accepted"] else i18n_text("拒绝", "Rejected")
    status_class = "accepted" if attempt["accepted"] else "rejected"
    to_generation = attempt["to_generation"] if attempt["to_generation"] is not None else "-"
    same_gen_black = attempt["calibration"]["candidates"]
    same_gen_black_rate = min(same_gen_black, key=lambda item: abs(item["komi"] - attempt["komi"]))["black_win_rate"] if same_gen_black else 0.0
    gate_assessment = render_gate_assessment(match)
    return (
        f"<tr><td>{attempt['attempt']}</td><td class=\"{status_class}\">{status}</td><td>{to_generation}</td>"
        f"<td>{attempt['komi']:.2f}</td><td>{match['candidate_win_rate']:.0%}</td>"
        f"<td>{gate_assessment}</td>"
        f"<td>{match['candidate_black_win_rate']:.0%} / {match['candidate_white_win_rate']:.0%}</td>"
        f"<td>{same_gen_black_rate:.0%}</td><td>{attempt['elo_delta']:+.1f}</td>"
        f"<td>{attempt['cumulative_elo']:.1f}</td><td>{last_train['loss']:.4f} / {last_train['accuracy']:.1%}</td></tr>"
    )


def render_gate_assessment(match: Dict[str, Any]) -> str:
    assessment = match.get("gate_assessment")
    if not assessment:
        return i18n_text("未记录", "Not recorded")
    labels = {
        "clear_pass": ("明确通过", "Clear pass"),
        "clear_reject": ("明确拒绝", "Clear reject"),
        "borderline": ("边界结果", "Borderline"),
    }
    zh, en = labels.get(assessment["label"], ("未知", "Unknown"))
    interval = f"{assessment['ci_low']:.0%}-{assessment['ci_high']:.0%}"
    return f"{i18n_text(zh, en)}<br/><span class=\"note\">95% CI {interval}</span>"


def render_elo_chart(generations: Sequence[Dict[str, Any]]) -> str:
    if not generations:
        return f"<p>{i18n_text('暂无晋级数据。', 'No accepted-generation data yet.')}</p>"
    width, height, pad = 720, 220, 34
    elos = [row["elo"] for row in generations]
    min_elo = min(min(elos), 0.0)
    max_elo = max(max(elos), 1.0)
    span = max(max_elo - min_elo, 1.0)

    def point(idx: int, value: float) -> Tuple[float, float]:
        x = pad + idx * (width - pad * 2) / max(len(generations) - 1, 1)
        y = height - pad - ((value - min_elo) / span) * (height - pad * 2)
        return x, y

    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(idx, row["elo"]) for idx, row in enumerate(generations)))
    labels = "".join(
        f'<text x="{point(idx, row["elo"])[0]:.1f}" y="{height - 8}" font-size="11" text-anchor="middle">{row["generation"]}</text>'
        for idx, row in enumerate(generations)
    )
    return f"""<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#9aa4b2"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#9aa4b2"/>
<polyline points="{points}" fill="none" stroke="#1d7f5f" stroke-width="3"/>
{labels}
<text x="{pad}" y="20" font-size="13" fill="#1d7f5f" data-zh="已晋级代 Elo" data-en="accepted generation Elo">已晋级代 Elo</text>
</svg>"""
