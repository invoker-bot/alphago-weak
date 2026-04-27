# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/3/26
@Version : 1.0
"""

import numpy as np
from typing import *
from .basic import *
from ..board import *
from ..dataset import GameData
from ..model.alpha_go_weak import *
from ..model.mcts import *

__all__ = ["GTPAlphaGoWeakV0"]


class GTPAlphaGoWeakV0(GTPClientBase):
    name = "alphago_weak_v0"
    __version__ = "0.0"

    def __init__(self, size=19, komi=6.5, **kwargs):
        super().__init__(size, komi)
        self.board = GoBoard(size)
        self.model = AlphaGoWeakV0(size=size, root=kwargs.get("root", "."))
        self.mcts = GoMCTSTree(self.board, self.model.state_calculator, self.model.policy_evaluator, self.model.value_evaluator, komi=komi)

    def genmove(self, player):
        pos = self.mcts.evaluate(player, 1600)
        if pos is not None:
            return pos
        return "pass"

    def encode_policy_output(self):
        size = self.size
        node = self.mcts.node
        arr = np.zeros((size * size + 1,), dtype=np.float32)
        if node.children is None or len(node.children) == 0:
            arr[-1] = 1.0
        else:
            for child in node.children:
                pos = child.pos
                weight = child.n ** (1 / node.t)
                if pos is None:
                    arr[-1] = weight
                else:
                    arr[pos.x * size + pos.y] = weight
            total = np.sum(arr)
            if total > 0:
                arr /= total
            else:
                arr[-1] = 1.0
        return arr

    def _winner_by_score(self) -> GoPlayer:
        return GoPlayer.black if self.board.score(GoPlayer.black, self.komi) > 0 else GoPlayer.white

    def self_play(self, count=4, simulations=1600, max_moves: Optional[int] = None,
                  game_recorder: Optional[Callable[[int, GameData], None]] = None,
                  move_callback: Optional[Callable[[int, int, GoPlayer, str, Optional[GoPoint]], None]] = None,
                  simulation_callback: Optional[Callable[[int], None]] = None):
        I, P, V = [], [], []
        for game_idx in range(count):
            self.clear_board()
            player = GoPlayer.black
            game_inputs, game_policies, game_players = [], [], []
            sequence: List[Tuple[GoPlayer, Optional[GoPoint]]] = []
            consecutive_passes = 0
            moves = 0
            move_limit = max_moves if max_moves is not None else self.size * self.size * 2
            while consecutive_passes < 2 and moves < move_limit:
                if move_callback is not None:
                    move_callback(game_idx, moves, player, "start", None)
                pos = self.mcts.evaluate(player, simulations, progress_callback=simulation_callback)
                game_inputs.append(self.model.encode_input(player, self.board))
                game_policies.append(self.encode_policy_output())
                game_players.append(player)
                sequence.append((player, pos))
                self.play(player, pos)
                if move_callback is not None:
                    move_callback(game_idx, moves, player, "end", pos)
                if pos is None:
                    consecutive_passes += 1
                else:
                    consecutive_passes = 0
                player = player.other
                moves += 1
            winner = self._winner_by_score()
            I.extend(game_inputs)
            P.extend(game_policies)
            V.extend(AlphaGoWeakV0.encode_value_output(player, winner) for player in game_players)
            if game_recorder is not None:
                game_recorder(game_idx, GameData(self.size, winner, sequence, self.komi))
        return I, P, V

    def play(self, player, pos):
        result = super().play(player, pos)
        if result:
            self.mcts.play(player, pos)
        return result

    def clear_board(self):
        super().clear_board()
        if self.model.size != self.size:
            self.model = AlphaGoWeakV0(size=self.size, root=self.model.root)
        self.mcts = GoMCTSTree(self.board, self.model.state_calculator, self.model.policy_evaluator, self.model.value_evaluator, komi=self.komi)
