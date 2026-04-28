#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from collections import deque
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from alphago_weak.cli import app
from alphago_weak.zero_ladder import (
    assess_gate_result,
    combine_match_results,
    elo_delta_from_win_rate,
    load_replay_buffers,
    save_replay_buffers,
    shuffled_players,
    wilson_score_interval,
)


def test_elo_delta_from_win_rate():
    assert np.isclose(elo_delta_from_win_rate(0.5), 0.0)
    assert elo_delta_from_win_rate(0.55) > 0
    assert elo_delta_from_win_rate(0.45) < 0


def test_shuffled_players_returns_both_colors():
    rng = np.random.default_rng(0)

    players = shuffled_players(rng)

    assert {player.name for player in players} == {"black", "white"}


def test_wilson_gate_assessment_marks_borderline_and_clear_results():
    low, high = wilson_score_interval(11, 20)

    assert low < 0.55 < high
    assert assess_gate_result(11, 20, 0.55)["label"] == "borderline"
    assert assess_gate_result(20, 20, 0.55)["label"] == "clear_pass"
    assert assess_gate_result(1, 20, 0.55)["label"] == "clear_reject"


def test_replay_buffers_round_trip(tmp_path):
    replay = deque(maxlen=3)
    candidate = deque(maxlen=3)
    for idx in range(4):
        sample = (
            np.full((2, 3, 3), idx, dtype=np.float32),
            np.eye(10, dtype=np.float32)[idx],
            np.array([idx % 2], dtype=np.float32),
        )
        replay.append(sample)
        candidate.append(sample)

    replay_path = tmp_path / "replay_buffers.npz"
    save_replay_buffers(replay_path, replay, candidate, accepted=2, attempt=5)
    restored_replay, restored_candidate, restored = load_replay_buffers(replay_path, maxlen=2)

    assert restored
    assert len(restored_replay) == 2
    assert len(restored_candidate) == 2
    assert np.allclose(restored_replay[0][0], np.full((2, 3, 3), 2, dtype=np.float32))
    assert np.allclose(restored_replay[-1][1], np.eye(10, dtype=np.float32)[3])


def test_match_results_combine_counts_and_weighted_averages():
    first = {
        "games": 20,
        "candidate_wins": 11,
        "candidate_avg_margin": 1.0,
        "candidate_black_games": 10,
        "candidate_black_wins": 6,
        "candidate_white_games": 10,
        "candidate_white_wins": 5,
        "black_wins": 9,
        "avg_black_margin": -0.5,
        "avg_moves": 60.0,
    }
    second = {
        "games": 10,
        "candidate_wins": 7,
        "candidate_avg_margin": 4.0,
        "candidate_black_games": 5,
        "candidate_black_wins": 4,
        "candidate_white_games": 5,
        "candidate_white_wins": 3,
        "black_wins": 5,
        "avg_black_margin": 1.0,
        "avg_moves": 75.0,
    }

    combined = combine_match_results(first, second)

    assert combined["games"] == 30
    assert combined["candidate_wins"] == 18
    assert np.isclose(combined["candidate_win_rate"], 0.6)
    assert np.isclose(combined["candidate_black_win_rate"], 10 / 15)
    assert np.isclose(combined["candidate_white_win_rate"], 8 / 15)
    assert np.isclose(combined["candidate_avg_margin"], 2.0)
    assert np.isclose(combined["avg_moves"], 65.0)


def test_zero_ladder_cli_writes_report_and_initial_model():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "--root", "data",
                "zero-ladder",
                "--output", "reports/ladder/index.html",
                "--board-size", "3",
                "--generations", "0",
                "--gate-training-rounds", "3",
                "--num-filters", "4",
                "--self-play-device", "cpu",
                "--force",
            ],
        )

        assert result.exit_code == 0
        html = Path("reports/ladder/index.html").read_text(encoding="utf-8")
        metrics = json.loads(Path("reports/ladder/metrics.json").read_text(encoding="utf-8"))
        model_path = Path("reports/ladder/models/generation-000.pt")
        assert "Zero Ladder" in html
        assert "English" in html
        assert '<html lang="en">' in html
        assert '|| "en"' in html
        assert "data-en=\"Generation-gated training report" in html
        assert "Candidate Win Rate" in html
        assert "Gate Confidence" in html
        assert "Progress Analysis" in html
        assert metrics["summary"]["accepted_generations"] == 0
        assert metrics["config"]["board_size"] == 3
        assert metrics["config"]["gate_training_rounds"] == 3
        assert model_path.exists()

        resumed = runner.invoke(
            app,
            [
                "--root", "data",
                "zero-ladder",
                "--output", "reports/ladder/index.html",
                "--board-size", "3",
                "--generations", "0",
                "--gate-training-rounds", "3",
                "--num-filters", "4",
                "--self-play-device", "cpu",
                "--resume",
            ],
        )

        assert resumed.exit_code == 0
        resumed_metrics = json.loads(Path("reports/ladder/metrics.json").read_text(encoding="utf-8"))
        assert resumed_metrics["summary"]["accepted_generations"] == 0
        assert resumed_metrics["config"]["resume"] is True


def test_zero_ladder_cli_requires_even_gate_games():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "zero-ladder",
                "--board-size", "3",
                "--generations", "1",
                "--eval-games", "3",
                "--num-filters", "4",
                "--self-play-device", "cpu",
                "--force",
            ],
        )

        assert result.exit_code != 0
        assert "eval-games must be even" in result.output
