# AlphaGo Weak

AlphaGo Weak 是一个用于学习和实验的围棋程序。当前代码已经迁移到 PyTorch，包含棋盘规则、SGF 数据处理、GTP bot、MCTS、监督学习入口，以及一个实验性的 AlphaGo Zero-style 自我对弈训练流程。

这不是论文级复现。仓库现在更适合作为可运行的研究脚手架：先在 7x7/9x9 小棋盘上验证规则、搜索、训练和报告流程，再逐步扩大训练规模。

## 7x7 实际训练报告

已生成一份从随机初始化开始的 7x7 小规模训练报告：

- [打开 HTML 报告](reports/7x7-zero/index.html)
- [查看原始指标 JSON](reports/7x7-zero/metrics.json)
- 样例 SGF: [reports/7x7-zero/sgf](reports/7x7-zero/sgf)

本次报告参数：7x7，komi 2.5，40 epoch，每 epoch 4 局 self-play，每手 12 次 MCTS simulation，16 filters 小网络，4 worker 多进程采样，CPU 运行。它记录了 checkpoint 0/10/20/30/40 的随机对手评估、配对换先的初始模型头对头评估、训练曲线、固定局面策略热力图和样例终局。

关键结果：

| Checkpoint | Random Win | Random Margin | Vs Initial Win | Vs Initial Margin | Avg Moves |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 65% | 5.40 | 50% | 0.00 | 57.4 |
| 10 | 65% | 0.90 | 45% | -6.00 | 57.1 |
| 20 | 65% | 11.20 | 55% | 4.80 | 64.5 |
| 30 | 80% | 8.70 | 60% | 4.50 | 61.2 |
| 40 | 60% | 5.70 | 80% | 14.10 | 60.7 |

训练指标从 loss 3.1471 / policy accuracy 6.0% 改善到 loss 2.0525 / policy accuracy 28.0%，replay buffer 达到 8192 样本。旧报告无法体现变强的主要原因是评估没有配对换先，且 7x7 使用 komi 0.5 会让结果被先手优势主导；现在的头对头评估会对同一随机开局分别交换候选模型执黑/执白。结论是：当前小规模流程已经能观察到相对初始模型的提升，但仍不是论文级棋力复现。

## 7x7 代际 Ladder 报告

新增的 `zero-ladder` 会保存每一代 champion，并使用候选模型对上一代 champion 的配对换先胜率作为晋级标准。候选胜率达到 55% 才记为前进一代，并把胜率换算成 Elo 增量。

- [打开 20 代中英 Ladder HTML 报告](reports/7x7-ladder-20-bilingual-v6/index.html)
- [查看 20 代 Ladder 指标 JSON](reports/7x7-ladder-20-bilingual-v6/metrics.json)
- 样例 SGF: [reports/7x7-ladder-20-bilingual-v6/sgf](reports/7x7-ladder-20-bilingual-v6/sgf)

当前 20 代示例从随机初始化开始，目标前进 20 代，实际在 38 次 candidate attempt 后完成 20/20 次晋级；其中 20 次通过、18 次拒绝，最终累计 Elo 为 +4760.6。训练指标从首轮 loss 3.1402 / policy accuracy 5.1% 改善到末轮 loss 2.1221 / policy accuracy 30.3%。相对贴目校准让同代黑棋胜率尽量接近 50%，因此表格中的候选胜率主要反映代际模型差异，而不是固定贴目带来的颜色偏差。

报告支持中文/English 切换，并记录每次 attempt 的 gate 结果、Elo 变化、loss/accuracy、相对贴目和样例 SGF。为保证本地可快速复现，本次报告使用 `eval-games=5` 和较小 MCTS simulation，Elo 曲线方向可用于观察进步，但数值会比长评估更噪。模型文件保存在本地 `reports/7x7-ladder-20-bilingual-v6/models/generation-*.pt`，默认不纳入 Git。

复现 20 代 ladder 实验：

```powershell
alphago --root .data-ladder-20-bilingual-v6 zero-ladder `
  --output reports/7x7-ladder-20-bilingual-v6/index.html `
  --board-size 7 --komi 2.5 `
  --generations 20 --max-attempts 80 `
  --candidate-epochs 2 --steps 2 --simulations 6 --max-moves 98 `
  --eval-games 5 --eval-simulations 3 --eval-opening-moves 4 `
  --gate-win-rate 0.55 --komi-eval-games 4 --komi-window 2 `
  --train-batch-size 32 --train-steps 32 `
  --replay-buffer-size 4096 --learning-rate 0.001 `
  --optimizer adam --augment --workers 1 `
  --inference-batch-size 8 --self-play-device cuda `
  --num-filters 16 --seed 7 --force
```

复现基础 zero-report 实验：

```powershell
alphago --root .data-report-final zero-report `
  --output reports/7x7-zero/index.html `
  --board-size 7 --komi 2.5 `
  --epochs 40 --steps 4 --simulations 12 --max-moves 98 `
  --eval-games 20 --eval-simulations 8 --eval-opening-moves 4 `
  --train-batch-size 32 --train-steps 96 `
  --replay-buffer-size 8192 --learning-rate 0.001 `
  --optimizer adam --augment --workers 4 `
  --inference-batch-size 16 --shared-inference `
  --num-filters 16 --checkpoints 0,10,20,30,40 `
  --seed 7 --force
```

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
alphago --help
```

PyTorch 会在安装了兼容 CUDA build 时自动使用 GPU；否则使用 CPU。

## 常用命令

```powershell
pytest -q
alphago download --type u-go
alphago train --init 4096 --epochs 1
alphago self-play --board-size 9 --steps 4 --epochs 10 --simulations 32
alphago zero-report --force
alphago zero-ladder --generations 20 --force
```

`self-play` 会自动从同一 `--root` 下的历史权重继续训练，并在每个 epoch 后保存模型。`--inference-batch-size` 会把 MCTS leaf 局面合并后批量送入模型，在 CUDA 版 PyTorch 下能减少单样本 GPU 调用开销。`zero-report` 在多 worker + CUDA 时默认启用共享推理服务；可用 `--no-shared-inference` 退回每个 worker 独立推理。`zero-report` 专门用于从零开始生成可浏览报告；它会使用 root 下的 `zero-report-{board_size}x{board_size}` 工作目录。

`zero-ladder` 会按 AlphaGo Zero 的 gating 思路训练候选模型：候选从当前 champion 权重继续训练，和上一代配对换先评估，胜率达到 `--gate-win-rate`（默认 55%）才晋级为下一代，并保存 `reports/7x7-ladder/models/generation-*.pt`。`--generations 20` 表示目标前进 20 代；默认会用相对贴目校准同代黑白胜率，降低颜色偏差。

## 项目结构

- `alphago_weak/board/`: 围棋规则、合法性、提子、计分和编码。
- `alphago_weak/model/`: PyTorch 网络、训练循环和 MCTS。
- `alphago_weak/gtp/`: GTP 客户端和 bot。
- `alphago_weak/dataset/`: SGF 下载、解压和解析。
- `alphago_weak/zero_report.py`: 7x7 AlphaGo Zero-style 实验和 HTML 报告生成。
- `alphago_weak/zero_ladder.py`: 代际 champion/candidate gating、Elo ladder 和中英 HTML 报告。
- `test/`: pytest 回归测试。
- `reports/7x7-zero/`: 当前提交包含的基础 self-play 训练报告。
- `reports/7x7-ladder-20-bilingual-v6/`: 当前提交包含的 20 代中英 ladder 报告。

## 当前限制

- MCTS 搜索和规则引擎仍主要是 Python 代码，GPU 只负责批量网络推理和训练。
- 已有 AlphaGo Zero-style 新旧模型胜率 gating，但评估局数较少时 Elo 噪声很大。
- replay buffer 只在单次命令运行内存在，暂未持久化为可恢复训练队列。
- 当前 7x7 报告只证明小规模相对提升，不代表模型已经具备可靠围棋棋力。
- 需要继续改进批量推理稳定性、value 校准、持久化 replay 和更长训练。

## 参考

- Leela Zero: <http://zero.sjeng.org/>
- KataGo: <https://github.com/lightvector/KataGo>
- GNU Go: <http://www.gnu.org/software/gnugo>
- Pachi: <https://github.com/pasky/pachi>
- Sabaki: <https://github.com/SabakiHQ/Sabaki>
