# -*- coding: utf-8 -*-

import copy
import html
import json
import math
import random
import shutil
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from alphago_weak.board import GoBoard, GoBoardBase, GoEyeType, GoPlayer, GoPoint
from alphago_weak.dataset import GameData
from alphago_weak.gtp.gtp_alphago_weak import GTPAlphaGoWeakV0
from alphago_weak.model.alpha_go_weak import AlphaGoWeakV0
from alphago_weak.model.mcts import GoMCTSTree


@dataclass
class ZeroReportConfig:
    root: Path
    output: Path
    board_size: int = 7
    komi: float = 0.5
    epochs: int = 20
    games_per_epoch: int = 4
    simulations: int = 8
    max_moves: int = 60
    eval_games: int = 6
    eval_simulations: int = 4
    train_batch_size: int = 32
    train_steps: int = 8
    replay_buffer_size: int = 4096
    learning_rate: float = 0.001
    num_filters: int = 16
    seed: int = 7
    checkpoint_epochs: Tuple[int, ...] = (0, 5, 10, 20)
    force: bool = False
    temperature: float = 1.0
    temperature_drop: int = 10
    temperature_after_drop: float = 0.0
    dirichlet_alpha: float = 0.15
    dirichlet_epsilon: float = 0.25


def run_zero_report(config: ZeroReportConfig) -> Dict[str, Any]:
    set_seed(config.seed)
    work_root = config.root / f"zero-report-{config.board_size}x{config.board_size}"
    if work_root.exists():
        if not config.force:
            raise FileExistsError(f"{work_root} already exists; pass --force to rerun from scratch")
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    config.output.parent.mkdir(parents=True, exist_ok=True)

    bot = GTPAlphaGoWeakV0(
        config.board_size,
        config.komi,
        root=str(work_root),
        num_filters=config.num_filters,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    optimizer = bot.model.make_optimizer(config.learning_rate)
    replay_buffer = deque(maxlen=config.replay_buffer_size)
    checkpoints = {0: snapshot_state(bot.model)}
    checkpoint_epochs = set(config.checkpoint_epochs)
    checkpoint_epochs.add(0)
    checkpoint_epochs.add(config.epochs)
    training_metrics: List[Dict[str, Any]] = []
    sample_games: Dict[int, GameData] = {}
    sgf_dir = config.output.parent / "sgf"
    sgf_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        games: List[GameData] = []
        inputs, policies, values = bot.self_play(
            count=config.games_per_epoch,
            simulations=config.simulations,
            max_moves=config.max_moves,
            temperature=config.temperature,
            temperature_drop=config.temperature_drop,
            temperature_after_drop=config.temperature_after_drop,
            dirichlet_alpha=config.dirichlet_alpha,
            dirichlet_epsilon=config.dirichlet_epsilon,
            game_recorder=lambda _idx, game: games.append(game),
        )
        replay_buffer.extend(zip(inputs, policies, values))
        replay_samples = list(replay_buffer)
        dataset = bot.model.dataset_from_arrays(
            [sample[0] for sample in replay_samples],
            [sample[1] for sample in replay_samples],
            [sample[2] for sample in replay_samples],
        )
        loss, accuracy = bot.model.fit_step_from_dataset(
            dataset,
            batch_size=config.train_batch_size,
            max_batches=config.train_steps,
            optimizer=optimizer,
            learning_rate=config.learning_rate,
        )
        if games:
            sample_games[epoch] = games[0]
            games[0].to_sgf(str(sgf_dir / f"epoch-{epoch:03d}.sgf"))
        training_metrics.append(
            {
                "epoch": epoch,
                "loss": loss,
                "accuracy": accuracy,
                "samples": len(inputs),
                "replay_size": len(replay_buffer),
                "avg_moves": float(np.mean([len(game.sequence) for game in games])) if games else 0.0,
                "black_win_rate": float(np.mean([game.winner == GoPlayer.black for game in games])) if games else 0.0,
            }
        )
        if epoch in checkpoint_epochs:
            checkpoints[epoch] = snapshot_state(bot.model)

    stages = []
    for epoch in sorted(checkpoints):
        load_snapshot(bot.model, checkpoints[epoch])
        eval_result = evaluate_against_random(
            bot.model,
            config.eval_games,
            config.eval_simulations,
            config.max_moves,
            config.komi,
            config.seed + epoch,
        )
        probes = analyze_probes(bot.model, config.board_size)
        stages.append(
            {
                "epoch": epoch,
                "eval": eval_result,
                "probes": probes,
                "sample_game": sample_games.get(epoch),
            }
        )

    report = {
        "config": config_to_dict(config, work_root),
        "training": training_metrics,
        "stages": stages,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (config.output.parent / "metrics.json").write_text(json.dumps(report, default=json_default, indent=2), encoding="utf-8")
    config.output.write_text(render_report(report), encoding="utf-8")
    return report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def snapshot_state(model: AlphaGoWeakV0) -> Dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.model.state_dict().items()}


def load_snapshot(model: AlphaGoWeakV0, state: Dict[str, torch.Tensor]):
    model.model.load_state_dict(copy.deepcopy(state))
    model.model.to(model.device)


def config_to_dict(config: ZeroReportConfig, work_root: Path) -> Dict[str, Any]:
    data = {key: value for key, value in config.__dict__.items() if key not in {"root", "output"}}
    data["root"] = str(config.root)
    data["work_root"] = str(work_root)
    data["output"] = str(config.output)
    data["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    data["checkpoint_epochs"] = list(config.checkpoint_epochs)
    return data


def json_default(value: Any):
    if isinstance(value, set):
        return list(value)
    if isinstance(value, GoPlayer):
        return value.name
    if isinstance(value, GoPoint):
        return tuple(value)
    if isinstance(value, GameData):
        return {
            "size": value.size,
            "winner": value.winner.name,
            "komi": value.komi,
            "moves": [(player.name if player is not None else None, tuple(pos) if pos is not None else None) for player, pos in value.sequence],
        }
    raise TypeError(f"unsupported json value: {type(value)!r}")


def evaluate_against_random(model: AlphaGoWeakV0, games: int, simulations: int, max_moves: int,
                            komi: float, seed: int) -> Dict[str, Any]:
    if games <= 0:
        return {"games": 0, "wins": 0, "win_rate": 0.0, "avg_margin": 0.0, "avg_moves": 0.0}
    rng = random.Random(seed)
    wins = 0
    margins: List[float] = []
    lengths: List[int] = []
    for game_idx in range(games):
        board = GoBoard(model.size)
        tree = GoMCTSTree(board, model.state_calculator, model.policy_evaluator, model.value_evaluator, komi=komi)
        model_player = GoPlayer.black if game_idx % 2 == 0 else GoPlayer.white
        player = GoPlayer.black
        consecutive_passes = 0
        moves = 0
        while consecutive_passes < 2 and moves < max_moves:
            if player == model_player:
                pos = tree.evaluate(player, num=simulations, temperature=0.0, dirichlet_alpha=0.0)
            else:
                actions = [pos for pos in board.valid_points(player) if board.eye_type(player, pos) < GoEyeType.unknown]
                pos = rng.choice(actions) if actions else None
            board.play(player, pos)
            tree.play(player, pos)
            consecutive_passes = consecutive_passes + 1 if pos is None else 0
            player = player.other
            moves += 1
        black_margin = board.score(GoPlayer.black, komi)
        winner = GoPlayer.black if black_margin > 0 else GoPlayer.white
        wins += int(winner == model_player)
        margins.append(black_margin if model_player == GoPlayer.black else -black_margin)
        lengths.append(moves)
    return {
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "avg_margin": float(np.mean(margins)),
        "avg_moves": float(np.mean(lengths)),
    }


def analyze_probes(model: AlphaGoWeakV0, size: int) -> List[Dict[str, Any]]:
    probes = probe_positions(size)
    results = []
    for name, note, player, board in probes:
        policy, value = model.predict(board, player)
        legal_probs: Dict[GoPoint, float] = {}
        for pos in board.valid_points(player):
            if board.eye_type(player, pos) < GoEyeType.unknown:
                legal_probs[pos] = float(policy[pos.x * size + pos.y])
        total = sum(legal_probs.values())
        normalized = {pos: prob / total for pos, prob in legal_probs.items()} if total > 0 else legal_probs
        ranked = sorted(normalized.items(), key=lambda item: item[1], reverse=True)[:5]
        pass_prob = float(policy[-1])
        entropy = policy_entropy(policy)
        results.append(
            {
                "name": name,
                "note": note,
                "player": player,
                "value": float(value),
                "pass_prob": pass_prob,
                "entropy": entropy,
                "top_moves": [(format_point(pos), prob) for pos, prob in ranked],
                "board_svg": render_board_svg(board, normalized, ranked, f"{name}: {player.name} to play"),
            }
        )
    return results


def probe_positions(size: int) -> List[Tuple[str, str, GoPlayer, GoBoardBase]]:
    empty = GoBoard(size)

    corner = GoBoard(size)
    corner.setup_stones(
        {GoPoint(0, 0), GoPoint(1, 0), GoPoint(0, 2)},
        {GoPoint(0, 1), GoPoint(2, 1)},
        set(),
    )

    capture = GoBoard(size)
    capture.setup_stones(
        {GoPoint(2, 1), GoPoint(1, 2), GoPoint(3, 2)},
        {GoPoint(2, 2)},
        set(),
    )

    territory = GoBoard(size)
    territory.setup_stones(
        {GoPoint(0, 0), GoPoint(0, 1), GoPoint(1, 0), GoPoint(size - 1, size - 1)},
        {GoPoint(size - 1, 0), GoPoint(size - 2, 0), GoPoint(size - 1, 1)},
        set(),
    )
    return [
        ("空棋盘", "观察开局偏好是否从近似均匀变得更集中。", GoPlayer.black, empty),
        ("角部接触", "黑棋需要判断角部战斗和外势方向。", GoPlayer.black, corner),
        ("吃子提示", "白棋一子处于低气，模型是否提高相邻关键点概率。", GoPlayer.black, capture),
        ("地域选择", "双方已有小角，观察模型更偏补强还是抢空。", GoPlayer.white, territory),
    ]


def policy_entropy(policy: np.ndarray) -> float:
    clipped = np.clip(policy.astype(np.float64), 1e-12, 1.0)
    return float(-(clipped * np.log(clipped)).sum() / math.log(len(policy)))


def format_point(pos: Optional[GoPoint]) -> str:
    if pos is None:
        return "pass"
    return f"({pos.x + 1},{pos.y + 1})"


def render_board_svg(board: GoBoardBase, probabilities: Dict[GoPoint, float],
                     ranked: Sequence[Tuple[GoPoint, float]], title: str) -> str:
    size = board.size
    cell = 34
    margin = 28
    extent = margin * 2 + cell * (size - 1)
    max_prob = max(probabilities.values(), default=0.0)
    top_rank = {pos: idx + 1 for idx, (pos, _prob) in enumerate(ranked)}
    parts = [
        f'<svg class="board" viewBox="0 0 {extent} {extent}" role="img" aria-label="{html.escape(title)}">',
        f'<rect x="0" y="0" width="{extent}" height="{extent}" rx="8" fill="#d7a85d"/>',
    ]
    for idx in range(size):
        pos = margin + idx * cell
        parts.append(f'<line x1="{margin}" y1="{pos}" x2="{extent - margin}" y2="{pos}" stroke="#312315" stroke-width="1"/>')
        parts.append(f'<line x1="{pos}" y1="{margin}" x2="{pos}" y2="{extent - margin}" stroke="#312315" stroke-width="1"/>')
    for pos, prob in probabilities.items():
        if board[pos] != GoPlayer.none or max_prob <= 0:
            continue
        cx = margin + pos.y * cell
        cy = margin + pos.x * cell
        radius = 5 + 13 * math.sqrt(prob / max_prob)
        opacity = 0.2 + 0.55 * (prob / max_prob)
        parts.append(f'<circle cx="{cx}" cy="{cy}" r="{radius:.2f}" fill="#1266d6" opacity="{opacity:.3f}"><title>{format_point(pos)} {prob:.3f}</title></circle>')
        if pos in top_rank:
            parts.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" font-size="11" font-weight="700" fill="#ffffff">{top_rank[pos]}</text>')
    for point, player in board.items():
        if player == GoPlayer.none:
            continue
        cx = margin + point.y * cell
        cy = margin + point.x * cell
        if player == GoPlayer.black:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="13" fill="#111111" stroke="#000" stroke-width="1"/>')
        else:
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="13" fill="#f8f8f8" stroke="#111" stroke-width="1"/>')
    parts.append("</svg>")
    return "".join(parts)


def render_final_board(game: Optional[GameData]) -> str:
    if game is None:
        return "<p>该阶段未保存样例对局。</p>"
    board = GoBoard(game.size)
    board.setup_stones(*game.setup_stones)
    for player, pos in game.sequence:
        board.play(player, pos)
    return render_board_svg(board, {}, [], f"sample game winner: {game.winner.name}")


def render_report(report: Dict[str, Any]) -> str:
    config = report["config"]
    training = report["training"]
    stages = report["stages"]
    stage_rows = "\n".join(
        f"<tr><td>{stage['epoch']}</td><td>{stage['eval']['games']}</td><td>{stage['eval']['win_rate']:.0%}</td>"
        f"<td>{stage['eval']['avg_margin']:.2f}</td><td>{stage['eval']['avg_moves']:.1f}</td></tr>"
        for stage in stages
    )
    training_rows = "\n".join(
        f"<tr><td>{row['epoch']}</td><td>{row['loss']:.4f}</td><td>{row['accuracy']:.1%}</td>"
        f"<td>{row['replay_size']}</td><td>{row['avg_moves']:.1f}</td><td>{row['black_win_rate']:.0%}</td></tr>"
        for row in training
    )
    stage_sections = "\n".join(render_stage(stage) for stage in stages)
    chart = render_training_chart(training)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AlphaGo Weak 7x7 Zero Report</title>
<style>
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f7f8fa; }}
header {{ padding: 28px 36px; background: #111827; color: white; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 26px 18px 48px; }}
h1, h2, h3 {{ margin: 0 0 12px; }}
p {{ line-height: 1.55; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
.card {{ background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; }}
.metric {{ font-size: 28px; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #e2e8f0; }}
th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
th {{ background: #eef2f7; }}
.board {{ width: 100%; max-width: 310px; display: block; }}
.probe {{ display: grid; grid-template-columns: minmax(220px, 320px) 1fr; gap: 14px; align-items: start; }}
.stage {{ margin-top: 22px; }}
.topmoves {{ margin: 0; padding-left: 18px; }}
.note {{ color: #526071; }}
code {{ background: #eef2f7; padding: 2px 5px; border-radius: 4px; }}
@media (max-width: 720px) {{ .probe {{ grid-template-columns: 1fr; }} header {{ padding: 22px 18px; }} }}
</style>
</head>
<body>
<header>
<h1>AlphaGo Weak 7x7 Zero 训练报告</h1>
<p>从随机初始化开始的小规模 AlphaGo Zero-style 自我对弈实验。生成时间：{html.escape(report['generated_at'])}</p>
</header>
<main>
<section class="grid">
<div class="card"><div class="note">棋盘</div><div class="metric">{config['board_size']}x{config['board_size']}</div></div>
<div class="card"><div class="note">训练 epoch</div><div class="metric">{config['epochs']}</div></div>
<div class="card"><div class="note">每手 MCTS simulations</div><div class="metric">{config['simulations']}</div></div>
<div class="card"><div class="note">设备</div><div class="metric">{html.escape(config['device'])}</div></div>
</section>
<section class="card" style="margin-top:16px">
<h2>实验说明</h2>
<p>这是本仓库当前实现的一次真实本地运行，不是论文规模复现。它用于观察 7x7 小棋盘上，模型从随机策略经过少量 self-play 后，策略分布、value 估计和对随机对手胜率的变化。训练使用 replay buffer、根节点 Dirichlet 噪声和 visit-count 温度退火。</p>
<p>复现命令见 README。完整参数保存在 <code>metrics.json</code>。</p>
</section>
<section class="card" style="margin-top:16px">
<h2>训练曲线</h2>
{chart}
<table><thead><tr><th>Epoch</th><th>Loss</th><th>Policy Acc.</th><th>Replay</th><th>Avg Moves</th><th>Black Win</th></tr></thead><tbody>{training_rows}</tbody></table>
</section>
<section class="card" style="margin-top:16px">
<h2>阶段评估：模型 MCTS vs 随机落子</h2>
<table><thead><tr><th>Checkpoint Epoch</th><th>Games</th><th>Win Rate</th><th>Avg Margin</th><th>Avg Moves</th></tr></thead><tbody>{stage_rows}</tbody></table>
</section>
{stage_sections}
</main>
</body>
</html>
"""


def render_training_chart(training: Sequence[Dict[str, Any]]) -> str:
    if not training:
        return "<p>没有训练 epoch。</p>"
    width, height, pad = 720, 220, 34
    losses = [row["loss"] for row in training]
    accs = [row["accuracy"] for row in training]
    max_loss = max(max(losses), 1e-6)

    def point(idx: int, value: float, max_value: float) -> Tuple[float, float]:
        x = pad + idx * (width - pad * 2) / max(len(training) - 1, 1)
        y = height - pad - (value / max_value) * (height - pad * 2)
        return x, y

    loss_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(idx, value, max_loss) for idx, value in enumerate(losses)))
    acc_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (point(idx, value, 1.0) for idx, value in enumerate(accs)))
    return f"""<svg viewBox="0 0 {width} {height}" style="width:100%;max-width:{width}px">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#9aa4b2"/>
<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#9aa4b2"/>
<polyline points="{loss_points}" fill="none" stroke="#d64545" stroke-width="3"/>
<polyline points="{acc_points}" fill="none" stroke="#1d7f5f" stroke-width="3"/>
<text x="{pad}" y="20" font-size="13" fill="#d64545">loss</text>
<text x="{pad+60}" y="20" font-size="13" fill="#1d7f5f">policy accuracy</text>
</svg>"""


def render_stage(stage: Dict[str, Any]) -> str:
    probes = "\n".join(render_probe(probe) for probe in stage["probes"])
    sample_board = render_final_board(stage.get("sample_game"))
    return f"""<section class="stage card">
<h2>Checkpoint epoch {stage['epoch']}</h2>
<p class="note">随机对手评估：胜率 {stage['eval']['win_rate']:.0%}，平均目差 {stage['eval']['avg_margin']:.2f}，平均手数 {stage['eval']['avg_moves']:.1f}。</p>
<h3>样例对局终局</h3>
{sample_board}
<h3>固定局面策略分析</h3>
{probes}
</section>"""


def render_probe(probe: Dict[str, Any]) -> str:
    top_moves = "".join(f"<li>{html.escape(move)}: {prob:.1%}</li>" for move, prob in probe["top_moves"])
    return f"""<div class="probe" style="margin-top:14px">
<div>{probe['board_svg']}</div>
<div>
<h3>{html.escape(probe['name'])}</h3>
<p class="note">{html.escape(probe['note'])}</p>
<p>当前方：<strong>{probe['player'].name}</strong>；value 估计：<strong>{probe['value']:.1%}</strong>；pass 概率：<strong>{probe['pass_prob']:.1%}</strong>；策略熵：<strong>{probe['entropy']:.3f}</strong></p>
<ol class="topmoves">{top_moves}</ol>
</div>
</div>"""
