#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typer.testing import CliRunner

from alphago_weak.board import GoBoard, GoPlayer, GoPoint
from alphago_weak.cli import app
from alphago_weak.gtp.gtp_alphago_weak import GTPAlphaGoWeakV0
from alphago_weak.gtp.gtp_random_bot_mcts import GTPRandomBotMCTS
from alphago_weak.model.alpha_go_weak import AlphaGoWeakV0
from alphago_weak.model.mcts import GoMCTSNode, GoMCTSTree


b, w = GoPlayer.black, GoPlayer.white
P = GoPoint


def test_random_mcts_bot_passes_player_to_tree_evaluate():
    bot = GTPRandomBotMCTS(3)
    calls = []
    bot.should_resign = lambda _player: False

    def evaluate(player, num=1600):
        calls.append((player, num))
        return P(0, 0)

    bot.mcts.evaluate = evaluate

    assert bot.genmove(b) == P(0, 0)
    assert calls == [(b, 400)]


def test_gtp_genmove_pass_advances_mcts_tree_player():
    bot = GTPRandomBotMCTS(3)
    bot.stdout = StringIO()
    bot.should_resign = lambda _player: False
    bot.mcts.evaluate = lambda _player, num=1600: None

    bot.do_genmove("b")

    assert bot.mcts.player == w


def test_gtp_play_pass_advances_mcts_tree_player():
    bot = GTPRandomBotMCTS(3)
    bot.stdout = StringIO()

    bot.do_play("b pass")

    assert bot.mcts.player == w


def test_model_state_calculator_keeps_pass_as_policy_action(tmp_path):
    model = AlphaGoWeakV0(size=3, root=str(tmp_path), num_filters=4, device="cpu")
    board = GoBoard(3)
    policy = np.zeros(10, dtype=np.float32)
    policy[0] = 0.25
    policy[-1] = 0.75

    model.predict = lambda _board, _player: (policy, 0.5)

    actions, weights = model.state_calculator(board, b)[0]
    weighted_actions = dict(zip(map(lambda action: None if action is None else tuple(action), actions), weights))

    assert weighted_actions[None] == 0.75
    assert weighted_actions[(0, 0)] == 0.25


def test_mcts_expansion_filters_illegal_policy_actions_and_keeps_pass():
    board = GoBoard(3)
    board.play(b, P(0, 0))

    def state(board, player):
        return board, player

    def policy(_state, komi=6.5):
        return [P(0, 0), P(1, 1), None], [0.8, 0.1, 0.1]

    tree = GoMCTSTree(board, state, policy, lambda _state, komi=6.5: 0.5)
    tree.evaluate(b, num=1)

    actions = {None if child.pos is None else tuple(child.pos) for child in tree.node.children}
    assert (0, 0) not in actions
    assert (1, 1) in actions
    assert None in actions


def test_mcts_choose_uses_legal_play_and_does_not_overwrite_illegal_child():
    board = GoBoard(3)
    board.play(b, P(0, 0))
    root = GoMCTSNode()
    root.children = [GoMCTSNode(root, P(0, 0), 1.0)]

    player, node = root.choose(board, w)

    assert player == w
    assert node is root
    assert root.children == []
    assert board[P(0, 0)] == b


def test_random_mcts_rollout_passes_once_before_terminal_scoring(monkeypatch):
    board = GoBoard(3)
    calls = []

    def valid_points(_board, player):
        calls.append(player)
        if len(calls) == 2:
            return [P(0, 0)]
        return []

    monkeypatch.setattr(GTPRandomBotMCTS, "valid_points", staticmethod(valid_points))

    GTPRandomBotMCTS.value_evaluator((board, b), 0)

    assert calls[:2] == [b, w]
    assert board[P(0, 0)] == w


def test_random_mcts_rollout_does_not_overwrite_illegal_action(monkeypatch):
    board = GoBoard(3)
    board.play(b, P(0, 0))
    calls = []

    def valid_points(_board, player):
        calls.append(player)
        if len(calls) == 1:
            return [P(0, 0)]
        return []

    monkeypatch.setattr(GTPRandomBotMCTS, "valid_points", staticmethod(valid_points))

    GTPRandomBotMCTS.value_evaluator((board, w), 0)

    assert board[P(0, 0)] == b


def test_alphago_policy_encoding_handles_pass_child():
    bot = GTPAlphaGoWeakV0.__new__(GTPAlphaGoWeakV0)
    bot._size = 3
    root = GoMCTSNode()
    point_child = GoMCTSNode(root, P(0, 0), 1.0)
    pass_child = GoMCTSNode(root, None, 1.0)
    point_child.n = 2
    pass_child.n = 3
    root.children = [point_child, pass_child]
    bot.mcts = SimpleNamespace(node=root)

    encoded = bot.encode_policy_output()

    assert encoded[0] > 0
    assert encoded[-1] > 0
    assert np.isclose(encoded.sum(), 1.0)


def test_alphago_zero_self_play_uses_final_winner_for_value_targets():
    bot = GTPAlphaGoWeakV0.__new__(GTPAlphaGoWeakV0)
    bot._size = 3
    bot.komi = 0.5
    bot.board = GoBoard(3)

    def encode_input(player, board):
        return np.array([player.value, board.score(player, bot.komi)], dtype=np.float32)

    bot.model = SimpleNamespace(size=3, root=".", encode_input=encode_input)

    root = GoMCTSNode()
    pass_child = GoMCTSNode(root, None, 1.0)
    pass_child.n = 5
    root.children = [pass_child]
    root.w = 9.0
    root.n = 10

    class FakeMCTS:
        def __init__(self):
            self.node = root
            self.moves = [None, None]

        def evaluate(self, player, num=1600, **_kwargs):
            return self.moves.pop(0)

        def play(self, player, pos):
            return None

    bot.mcts = FakeMCTS()
    bot.clear_board = lambda: (setattr(bot, "board", GoBoard(3)), setattr(bot, "mcts", FakeMCTS()))

    inputs, policies, values = bot.self_play(count=1, simulations=1, max_moves=10)

    assert len(inputs) == 2
    assert np.allclose(policies[0], np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32))
    assert values[0] == np.array(0.0, dtype=np.float32)
    assert values[1] == np.array(1.0, dtype=np.float32)


def test_alphago_zero_self_play_records_game_data():
    bot = GTPAlphaGoWeakV0.__new__(GTPAlphaGoWeakV0)
    bot._size = 3
    bot.komi = 0.5
    bot.board = GoBoard(3)
    bot.model = SimpleNamespace(size=3, root=".", encode_input=lambda _player, _board: np.zeros(1, dtype=np.float32))

    root = GoMCTSNode()
    pass_child = GoMCTSNode(root, None, 1.0)
    pass_child.n = 1
    root.children = [pass_child]

    class FakeMCTS:
        def __init__(self):
            self.node = root
            self.moves = [P(0, 0), None, None]

        def evaluate(self, player, num=1600, **_kwargs):
            return self.moves.pop(0)

        def play(self, player, pos):
            return None

    bot.clear_board = lambda: (setattr(bot, "board", GoBoard(3)), setattr(bot, "mcts", FakeMCTS()))
    records = []

    bot.self_play(count=1, simulations=1, max_moves=10, game_recorder=lambda idx, game: records.append((idx, game)))

    idx, game = records[0]
    assert idx == 0
    assert game.size == 3
    assert game.komi == 0.5
    assert game.winner == b
    assert game.sequence == [(b, P(0, 0)), (w, None), (b, None)]


def test_self_play_cli_records_sample_sgf():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(
            app,
            [
                "self-play",
                "--steps", "1",
                "--epochs", "1",
                "--board-size", "3",
                "--komi", "0.5",
                "--simulations", "1",
                "--max-moves", "2",
                "--record-games", "1",
                "--record-dir", "records",
            ],
        )

        assert result.exit_code == 0
        sgf_files = list(Path("records").glob("*.sgf"))
        assert len(sgf_files) == 1
        assert "KM[0.5]" in sgf_files[0].read_text(encoding="utf-8")


def test_alphago_zero_self_play_reports_move_and_simulation_progress():
    bot = GTPAlphaGoWeakV0.__new__(GTPAlphaGoWeakV0)
    bot._size = 3
    bot.komi = 0.5
    bot.board = GoBoard(3)
    bot.model = SimpleNamespace(size=3, root=".", encode_input=lambda _player, _board: np.zeros(1, dtype=np.float32))

    root = GoMCTSNode()
    pass_child = GoMCTSNode(root, None, 1.0)
    pass_child.n = 1
    root.children = [pass_child]

    class FakeMCTS:
        def __init__(self):
            self.node = root
            self.moves = [None, None]

        def evaluate(self, player, num=1600, progress_callback=None):
            if progress_callback is not None:
                for _ in range(num):
                    progress_callback(1)
            return self.moves.pop(0)

        def play(self, player, pos):
            return None

    bot.clear_board = lambda: (setattr(bot, "board", GoBoard(3)), setattr(bot, "mcts", FakeMCTS()))
    moves = []
    simulations = []

    bot.self_play(
        count=1,
        simulations=3,
        max_moves=10,
        move_callback=lambda *event: moves.append(event),
        simulation_callback=lambda advance: simulations.append(advance),
    )

    assert [event[3] for event in moves] == ["start", "end", "start", "end"]
    assert simulations == [1, 1, 1, 1, 1, 1]


def test_fit_step_reports_training_progress(tmp_path):
    model = AlphaGoWeakV0(size=3, root=str(tmp_path), num_filters=4, device="cpu")
    inputs = [np.zeros((model.FEATURES["length"], 3, 3), dtype=np.float32) for _ in range(65)]
    policies = [AlphaGoWeakV0.encode_policy_output(3, None) for _ in inputs]
    values = [np.array(0.5, dtype=np.float32) for _ in inputs]
    dataset = model.dataset_from_arrays(inputs, policies, values)
    progress = []

    model.fit_step_from_dataset(dataset, progress_callback=lambda advance: progress.append(advance))

    assert progress == [1, 1]
