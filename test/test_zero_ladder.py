#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from alphago_weak.cli import app
from alphago_weak.zero_ladder import elo_delta_from_win_rate, shuffled_players


def test_elo_delta_from_win_rate():
    assert np.isclose(elo_delta_from_win_rate(0.5), 0.0)
    assert elo_delta_from_win_rate(0.55) > 0
    assert elo_delta_from_win_rate(0.45) < 0


def test_shuffled_players_returns_both_colors():
    rng = np.random.default_rng(0)

    players = shuffled_players(rng)

    assert {player.name for player in players} == {"black", "white"}


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
