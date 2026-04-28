#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from typer.testing import CliRunner

from alphago_weak.cli import app


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
                "--checkpoints", "0",
                "--force",
            ],
        )

        assert result.exit_code == 0
        html = Path("reports/zero/index.html").read_text(encoding="utf-8")
        metrics = Path("reports/zero/metrics.json").read_text(encoding="utf-8")
        assert "AlphaGo Weak 7x7 Zero" in html
        assert "固定局面策略分析" in html
        assert '"board_size": 5' in metrics
