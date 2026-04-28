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
        self._model_name = kwargs.get("name", self.name)
        self._model_num_filters = kwargs.get("num_filters", 128)
        self._model_device = kwargs.get("device", None)
        self._inference_batch_size = kwargs.get("inference_batch_size", 1)
        self.board = GoBoard(size)
        self.model = AlphaGoWeakV0(
            name=self._model_name,
            size=size,
            root=kwargs.get("root", "."),
            num_filters=self._model_num_filters,
            device=self._model_device,
        )
        self.mcts = GoMCTSTree(
            self.board,
            self.model.state_calculator,
            self.model.policy_evaluator,
            self.model.value_evaluator,
            komi=komi,
            batch_state_calculator=self.model.state_calculator_batch,
        )

    def genmove(self, player):
        pos = self.mcts.evaluate(player, 1600, inference_batch_size=self._inference_batch_size)
        if pos is not None:
            return pos
        return "pass"

    def encode_policy_output(self, temperature: Optional[float] = None):
        size = self.size
        node = self.mcts.node
        arr = np.zeros((size * size + 1,), dtype=np.float32)
        if node.children is None or len(node.children) == 0:
            arr[-1] = 1.0
        else:
            actions, weights = node.visit_policy(temperature)
            for pos, weight in zip(actions, weights):
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
                  temperature: float = 1.0,
                  temperature_drop: Optional[int] = 30,
                  temperature_after_drop: float = 0.0,
                  dirichlet_alpha: Optional[float] = 0.03,
                  dirichlet_epsilon: float = 0.25,
                  inference_batch_size: Optional[int] = None,
                  game_recorder: Optional[Callable[[int, GameData], None]] = None,
                  move_callback: Optional[Callable[[int, int, GoPlayer, str, Optional[GoPoint]], None]] = None,
                  simulation_callback: Optional[Callable[[int], None]] = None):
        I, P, V = [], [], []
        batch_size = getattr(self, "_inference_batch_size", 1) if inference_batch_size is None else inference_batch_size
        for game_idx in range(count):
            self.clear_board()
            player = GoPlayer.black
            game_inputs, game_policies, game_players = [], [], []
            sequence: List[Tuple[GoPlayer, Optional[GoPoint]]] = []
            consecutive_passes = 0
            moves = 0
            move_limit = max_moves if max_moves is not None else self.size * self.size * 2
            while consecutive_passes < 2 and moves < move_limit:
                move_temperature = temperature if temperature_drop is None or moves < temperature_drop else temperature_after_drop
                if move_callback is not None:
                    move_callback(game_idx, moves, player, "start", None)
                pos = self.mcts.evaluate(
                    player,
                    simulations,
                    progress_callback=simulation_callback,
                    temperature=move_temperature,
                    dirichlet_alpha=dirichlet_alpha,
                    exploration_fraction=dirichlet_epsilon,
                    inference_batch_size=batch_size,
                )
                game_inputs.append(self.model.encode_input(player, self.board))
                game_policies.append(self.encode_policy_output(move_temperature))
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
            self.model = AlphaGoWeakV0(
                name=self._model_name,
                size=self.size,
                root=self.model.root,
                num_filters=self._model_num_filters,
                device=self._model_device,
            )
        self.mcts = GoMCTSTree(
            self.board,
            self.model.state_calculator,
            self.model.policy_evaluator,
            self.model.value_evaluator,
            komi=self.komi,
            batch_state_calculator=self.model.state_calculator_batch,
        )
