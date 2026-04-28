# AlphaGo Weak

AlphaGo Weak 是一个用于学习和实验的围棋程序。当前代码已迁移到 PyTorch，包含棋盘规则、SGF 数据处理、GTP bot、MCTS、监督学习入口，以及实验性的 AlphaGo Zero-style 自我对弈训练流程。

这不是论文级复现。仓库更适合作为可运行的研究脚手架：先在 7x7/9x9 小棋盘上验证规则、搜索、训练和报告流程，再逐步扩大训练规模。

## 安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
alphago --help
```

PyTorch 会在安装了兼容 CUDA build 时自动使用 GPU；否则使用 CPU。可用下面的命令确认：

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 常用命令

```powershell
pytest -q
alphago download --type u-go
alphago train --init 4096 --epochs 1
alphago self-play --board-size 9 --steps 4 --epochs 10 --simulations 32
alphago zero-report --force
alphago zero-ladder --generations 20 --force
```

`self-play` 会自动从同一 `--root` 下的历史权重继续训练，并在每个 epoch 后保存模型。`--inference-batch-size` 会把 MCTS leaf 局面合并后批量送入模型，在 CUDA 版 PyTorch 下能减少单样本 GPU 调用开销。

## 7x7 Zero Report

`zero-report` 从随机初始化开始做小规模 self-play 训练，并生成可浏览 HTML。报告输出到本地 `reports/`，该目录默认被 Git 忽略。

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

运行后打开 `reports/7x7-zero/index.html` 查看训练曲线、评估表、策略热力图和样例终局。

## 7x7 Generation Ladder

`zero-ladder` 会保存每一代 champion，并使用候选模型对上一代 champion 的配对换先胜率作为晋级标准。候选胜率达到 `--gate-win-rate`（默认 55%）才前进一代，并把胜率换算成 Elo 增量。报告默认显示英文，支持中文/English 切换。

为了降低过早评估导致的拒绝率，默认每次正式 gate 前会训练 `--candidate-epochs 2` x `--gate-training-rounds 4`，也就是 8 个 self-play epoch。拒绝仍然可能发生；如果拒绝过多，优先增大 `--gate-training-rounds` 或 `--candidate-epochs`，而不是降低 gate 阈值。

推荐的 20 代 7x7 本地实验：

```powershell
alphago --root .data-ladder-20-strong-gate zero-ladder `
  --output reports/7x7-ladder-20/index.html `
  --board-size 7 --komi 2.5 `
  --generations 20 --max-attempts 60 `
  --candidate-epochs 3 --gate-training-rounds 4 `
  --steps 4 --simulations 6 --max-moves 98 `
  --eval-games 20 --eval-simulations 3 --eval-opening-moves 4 `
  --gate-win-rate 0.55 --komi-eval-games 20 --komi-window 2 `
  --train-batch-size 32 --train-steps 64 `
  --replay-buffer-size 8192 --learning-rate 0.001 `
  --optimizer adam --augment --workers 4 `
  --inference-batch-size 8 --self-play-device cuda `
  --num-filters 16 --seed 7 --force
```

长时间 CUDA 运行如果中断，可以去掉 `--force` 并加上 `--resume`，命令会从已写入的 `metrics.json` 和最新 champion checkpoint 继续；报告会在每个完整 attempt 后刷新。

## 项目结构

- `alphago_weak/board/`: 围棋规则、合法性、提子、计分和编码。
- `alphago_weak/model/`: PyTorch 网络、训练循环和 MCTS。
- `alphago_weak/gtp/`: GTP 客户端和 bot。
- `alphago_weak/dataset/`: SGF 下载、解压和解析。
- `alphago_weak/zero_report.py`: 7x7 AlphaGo Zero-style 实验和 HTML 报告生成。
- `alphago_weak/zero_ladder.py`: 代际 champion/candidate gating、Elo ladder 和中英 HTML 报告。
- `test/`: pytest 回归测试。
- `reports/`: 本地生成报告，默认不纳入 Git。
- `.data*/`: 本地数据、缓存和实验工作目录，默认不纳入 Git。

## 当前限制

- MCTS 搜索和规则引擎仍主要是 Python 代码，GPU 只负责批量网络推理和训练。
- 已有 AlphaGo Zero-style 新旧模型胜率 gating；更高精度 Elo 仍需要更多评估局和更高 MCTS simulation。
- replay buffer 只在单次命令运行内存在，暂未持久化为可恢复训练队列。
- 当前 7x7 报告只证明小规模相对提升，不代表模型已经具备可靠围棋棋力。
- 需要继续改进批量推理稳定性、value 校准、持久化 replay 和更长训练。

## 参考

- Leela Zero: <http://zero.sjeng.org/>
- KataGo: <https://github.com/lightvector/KataGo>
- GNU Go: <http://www.gnu.org/software/gnugo>
- Pachi: <https://github.com/pasky/pachi>
- Sabaki: <https://github.com/SabakiHQ/Sabaki>
