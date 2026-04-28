<h1 align="center">AlphaGo Weak</h1>

<p align="center">
  <strong>A compact AlphaGo Zero-style Go lab in PyTorch.</strong><br/>
  Small-board self-play, MCTS, GTP bots, SGF data tools, and bilingual HTML reports.
</p>

AlphaGo Weak 是一个用于学习和实验的围棋程序。它不是论文级复现，而是一个能在本地跑通的研究脚手架：先在 7x7/9x9 小棋盘验证规则、搜索、训练和测评，再逐步放大实验规模。

## What It Does

| Area | Status |
| --- | --- |
| Rules engine | Go legality, captures, passes, flood-fill area scoring |
| Models | PyTorch policy/value network with CPU/CUDA support |
| Search | MCTS self-play and batched neural-network inference |
| Training | Supervised learning plus experimental AlphaGo Zero-style replay training |
| Evaluation | Champion/candidate gate, paired color-swap games, Elo-style ladder |
| Reports | Local bilingual HTML reports with Elo, gate confidence, policy entropy, value targets, and sample SGF records |

## Latest 7x7 Result

Latest completed ladder run, produced on a CUDA RTX 4090 environment, is summarized below. Generated reports are ignored by Git; rerun the command in the reproduction section to recreate `reports/7x7-ladder-10/index.html` locally.

| Metric | Value |
| --- | ---: |
| Board | 7x7 |
| Target advances | 10 |
| Accepted generations | 10/10 |
| Candidate attempts | 12 |
| Rejections | 2 |
| Rejection rate | 16.7% |
| Gate games | 20 per attempt |
| Color split | 10 candidate-black / 10 candidate-white |
| Gate threshold | 55% candidate win rate |
| Final Elo estimate | +1104.0 |
| Policy accuracy | 4.0% -> 70.6% |
| Loss | 3.1689 -> 0.8541 |

The ladder uses paired evaluation: for each sampled opening, the candidate and champion play both colors. This keeps the main win-rate signal from being dominated by first-player or komi bias.

<details>
<summary>Attempt timeline</summary>

| Attempt | Result | Advance | Win Rate | Komi | Elo Δ | Cumulative Elo |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Rejected | - | 40% | 4.5 | -70.4 | 0.0 |
| 2 | Accepted | 1 | 60% | 0.5 | +70.4 | 70.4 |
| 3 | Accepted | 2 | 70% | 0.5 | +147.2 | 217.6 |
| 4 | Accepted | 3 | 85% | 4.5 | +301.3 | 519.0 |
| 5 | Accepted | 4 | 70% | 0.5 | +147.2 | 666.1 |
| 6 | Accepted | 5 | 55% | 0.5 | +34.9 | 701.0 |
| 7 | Rejected | - | 50% | 1.5 | +0.0 | 701.0 |
| 8 | Accepted | 6 | 55% | 0.5 | +34.9 | 735.9 |
| 9 | Accepted | 7 | 55% | 4.5 | +34.9 | 770.7 |
| 10 | Accepted | 8 | 65% | 0.5 | +107.5 | 878.3 |
| 11 | Accepted | 9 | 55% | 1.5 | +34.9 | 913.1 |
| 12 | Accepted | 10 | 75% | 4.5 | +190.8 | 1104.0 |

</details>

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
alphago --help
```

Confirm CUDA support:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Run tests:

```powershell
pytest -q
```

## Reproduce The 10-Generation Ladder

This command trains longer before each formal gate: `3 x 4 = 12` self-play epochs per candidate evaluation. It uses 20 games for both gate evaluation and relative-komi calibration. The exact summary above was generated before automatic borderline rechecks, so the command pins `--gate-recheck-games 0`; omit that flag for stricter new runs.

```powershell
alphago --root .data-ladder-10-strong-gate zero-ladder `
  --output reports/7x7-ladder-10/index.html `
  --board-size 7 --komi 2.5 `
  --generations 10 --max-attempts 30 `
  --candidate-epochs 3 --gate-training-rounds 4 `
  --steps 4 --simulations 6 --max-moves 98 `
  --eval-games 20 --eval-simulations 3 --eval-opening-moves 4 `
  --gate-recheck-games 0 `
  --gate-win-rate 0.55 --komi-eval-games 20 --komi-window 2 `
  --train-batch-size 32 --train-steps 64 `
  --replay-buffer-size 8192 --learning-rate 0.001 `
  --optimizer adam --augment --workers 4 `
  --inference-batch-size 8 --self-play-device cuda `
  --num-filters 16 --seed 7 --force
```

If a long CUDA run stops midway, rerun the same command with `--resume` instead of `--force`. The report is refreshed after every complete attempt.
`zero-ladder` also persists replay buffers and an in-progress candidate checkpoint during training, so a resumed run keeps the candidate training window instead of falling back to champion checkpoints only.
New runs recheck borderline gates by default with extra paired games; use `--gate-recheck-games 0` only when you need the older fixed-20-game behavior.
Fresh reports also include MCTS policy-target entropy and value-target statistics, which help diagnose whether self-play is producing sharper search targets or drifting into one-sided outcomes.

## Other Useful Commands

```powershell
alphago download --type u-go
alphago train --init 4096 --epochs 1
alphago self-play --board-size 9 --steps 4 --epochs 10 --simulations 32
alphago zero-report --force
alphago zero-ladder --generations 10 --force
```

`self-play` continues from the same `--root` weights and saves after every epoch. `--inference-batch-size` batches MCTS leaf positions before neural-network inference, which is important for CUDA utilization.

## Project Layout

- `alphago_weak/board/`: Go rules, captures, legality, scoring, and encodings.
- `alphago_weak/model/`: PyTorch networks, training helpers, and MCTS.
- `alphago_weak/gtp/`: GTP clients and playable bots.
- `alphago_weak/dataset/`: SGF download, extraction, parsing, and preprocessing.
- `alphago_weak/zero_report.py`: From-scratch 7x7 training report.
- `alphago_weak/zero_ladder.py`: Champion/candidate gating, Elo ladder, and bilingual report generation.
- `test/`: pytest regression coverage.
- `reports/`: local generated reports, ignored by Git.
- `.data*/`: local datasets, caches, replay workdirs, and checkpoints, ignored by Git.

## Current Limits

- The Go engine and MCTS are still mostly Python, so GPU is used mainly for batched network inference and training.
- Elo values are useful for relative progress, but they remain noisy with 20-game gates.
- `--resume` restores the latest champion, candidate path, in-progress candidate checkpoint, report state, and persisted replay buffers.
- The 7x7 result demonstrates measurable local improvement, not strong general Go skill.

## References

- Leela Zero: <http://zero.sjeng.org/>
- KataGo: <https://github.com/lightvector/KataGo>
- GNU Go: <http://www.gnu.org/software/gnugo>
- Pachi: <https://github.com/pasky/pachi>
- Sabaki: <https://github.com/SabakiHQ/Sabaki>
