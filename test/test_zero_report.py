#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from alphago_weak.cli import app
from alphago_weak.model.alpha_go_weak import AlphaGoWeakV0
from alphago_weak.zero_report import (
    ZeroReportConfig,
    augment_samples,
    evaluate_against_model,
    should_use_shared_inference,
    snapshot_state,
)


def test_zero_report_cli_writes_html_and_metrics():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "--root", "data",
                "zero-report",
                "--output", "reports/zero/index.html",
                "--board-size", "5",
                "--epochs", "0",
                "--steps", "1",
                "--simulations", "1",
                "--eval-games", "0",
                "--num-filters", "4",
                "--workers", "1",
                "--inference-batch-size", "2",
                "--self-play-device", "cpu",
                "--no-shared-inference",
                "--checkpoints", "0",
                "--force",
            ],
        )

        assert result.exit_code == 0
        html = Path("reports/zero/index.html").read_text(encoding="utf-8")
        metrics = json.loads(Path("reports/zero/metrics.json").read_text(encoding="utf-8"))
        assert "AlphaGo Weak 7x7 Zero" in html
        assert "固定局面策略分析" in html
        assert metrics["config"]["board_size"] == 5
        assert metrics["config"]["inference_batch_size"] == 2
        assert metrics["config"]["device"] == "cpu"
        assert metrics["config"]["shared_inference"] is False
        assert metrics["config"]["shared_inference_active"] is False
        assert "black_win_rate" in metrics["stages"][0]["eval"]
        assert "white_win_rate" in metrics["stages"][0]["vs_initial"]


def test_augment_samples_rotates_policy_and_keeps_pass_value():
    input_ = np.zeros((1, 2, 2), dtype=np.float32)
    input_[0, 0, 1] = 1.0
    policy = np.array([0.0, 1.0, 0.0, 0.0, 0.5], dtype=np.float32)
    value = np.array(1.0, dtype=np.float32)

    inputs, policies, values = augment_samples([input_], [policy], [value], 2)

    assert len(inputs) == 8
    assert len(policies) == 8
    assert len(values) == 8
    assert all(item[-1] == 0.5 for item in policies)
    assert any(item[2] == 1.0 for item in policies)
    assert all(np.array_equal(item, np.array(1.0, dtype=np.float32)) for item in values)


def test_paired_model_evaluation_cancels_same_model_color_bias(tmp_path):
    model = AlphaGoWeakV0(size=3, root=str(tmp_path), num_filters=4, device="cpu")
    state = snapshot_state(model)
    config = ZeroReportConfig(
        root=tmp_path,
        output=tmp_path / "report.html",
        board_size=3,
        komi=2.5,
        eval_games=4,
        eval_simulations=1,
        eval_opening_moves=1,
        max_moves=8,
        num_filters=4,
        workers=1,
        self_play_device="cpu",
    )

    result = evaluate_against_model(state, state, config, tmp_path, seed=123)

    assert result["win_rate"] == 0.5
    assert result["black_win_rate"] + result["white_win_rate"] == 1.0
    assert result["avg_margin"] == 0.0


def test_shared_inference_only_activates_for_cuda_multi_worker(tmp_path, monkeypatch):
    monkeypatch.setattr("alphago_weak.zero_report.torch.cuda.is_available", lambda: True)
    config = ZeroReportConfig(
        root=tmp_path,
        output=tmp_path / "report.html",
        self_play_device="auto",
        workers=2,
        games_per_epoch=2,
    )

    assert should_use_shared_inference(config, worker_count=2)
    assert not should_use_shared_inference(config, worker_count=1)
    config.self_play_device = "cpu"
    assert not should_use_shared_inference(config, worker_count=2)
    config.self_play_device = "cuda"
    config.shared_inference = False
    assert not should_use_shared_inference(config, worker_count=2)
