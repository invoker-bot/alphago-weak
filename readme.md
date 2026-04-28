# AlphaGo Weak

AlphaGo Weak 是一个用于学习和实验的围棋程序。当前代码已经迁移到 PyTorch，包含棋盘规则、SGF 数据处理、GTP bot、MCTS、监督学习入口，以及一个实验性的 AlphaGo Zero-style 自我对弈训练流程。

这不是论文级复现。仓库现在更适合作为可运行的研究脚手架：先在 7x7/9x9 小棋盘上验证规则、搜索、训练和报告流程，再逐步扩大训练规模。

## 7x7 实际训练报告

已生成一份从随机初始化开始的 7x7 小规模训练报告：

- [打开 HTML 报告](reports/7x7-zero/index.html)
- [查看原始指标 JSON](reports/7x7-zero/metrics.json)
- 样例 SGF: [reports/7x7-zero/sgf](reports/7x7-zero/sgf)

本次报告参数：7x7，komi 0.5，20 epoch，每 epoch 4 局 self-play，每手 8 次 MCTS simulation，16 filters 小网络，CPU 运行。它记录了 checkpoint 0/5/10/20 的随机对手评估、训练曲线、固定局面策略热力图和样例终局。

关键结果：

| Checkpoint | Random Eval Win Rate | Avg Margin | Avg Moves |
| --- | ---: | ---: | ---: |
| 0 | 33% | -4.00 | 55.5 |
| 5 | 33% | -8.50 | 54.7 |
| 10 | 33% | -3.83 | 57.7 |
| 20 | 0% | -15.00 | 59.7 |

训练指标有轻微变化：loss 从 3.1879 降到 3.1701，policy accuracy 从 2.5% 到 5.5%，replay buffer 达到 4096 样本。但固定局面的策略熵仍接近 1.0，value 也接近 50%。结论是：当前流程能真实训练和生成报告，但这个小规模配置没有产生稳定棋力提升，离论文中的有效强化学习还差 replay/gating、更多 simulation、批量推理和更强评估。

复现实验：

```powershell
alphago --root .data-report-final zero-report `
  --output reports/7x7-zero/index.html `
  --board-size 7 --komi 0.5 `
  --epochs 20 --steps 4 --simulations 8 --max-moves 60 `
  --eval-games 6 --eval-simulations 4 `
  --train-batch-size 32 --train-steps 8 `
  --replay-buffer-size 4096 --learning-rate 0.001 `
  --num-filters 16 --checkpoints 0,5,10,20 `
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
```

`self-play` 会自动从同一 `--root` 下的历史权重继续训练，并在每个 epoch 后保存模型。`zero-report` 专门用于从零开始生成可浏览报告；它会使用 root 下的 `zero-report-{board_size}x{board_size}` 工作目录。

## 项目结构

- `alphago_weak/board/`: 围棋规则、合法性、提子、计分和编码。
- `alphago_weak/model/`: PyTorch 网络、训练循环和 MCTS。
- `alphago_weak/gtp/`: GTP 客户端和 bot。
- `alphago_weak/dataset/`: SGF 下载、解压和解析。
- `alphago_weak/zero_report.py`: 7x7 AlphaGo Zero-style 实验和 HTML 报告生成。
- `test/`: pytest 回归测试。
- `reports/7x7-zero/`: 当前提交包含的实际训练报告。

## 当前限制

- MCTS 仍是 Python 单样本推理，GPU 利用率低。
- 没有完整 AlphaGo Zero 的新旧模型胜率 gating。
- replay buffer 只在单次命令运行内存在，暂未持久化。
- 当前 7x7 报告显示训练流程可运行，但棋力没有稳定提升。
- 需要继续改进批量推理、评估赛制、value 校准和更长训练。

## 参考

- Leela Zero: <http://zero.sjeng.org/>
- KataGo: <https://github.com/lightvector/KataGo>
- GNU Go: <http://www.gnu.org/software/gnugo>
- Pachi: <https://github.com/pasky/pachi>
- Sabaki: <https://github.com/SabakiHQ/Sabaki>
