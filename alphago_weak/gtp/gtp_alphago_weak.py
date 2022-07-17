# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    : 
@Data    : 2021/3/26
@Version : 1.0
"""

import tqdm
import numpy as np
from typing import *
from .basic import *
from ..board import *
from ..model.alpha_go_weak import *
from ..model.mcts import *

__all__ = ["GTPAlphaGoWeakV0"]


class GTPAlphaGoWeakV0(GTPClientBase):
    name = "alphago_weak_v0"
    __version__ = "0.0"

    def __init__(self, size=19, komi=6.5):
        super().__init__(size, komi)
        self.board = GoBoard(size)
        self.model = AlphaGoWeakV0(size=size)
        self.mcts = GoMCTSTree(self.board, self.model.state_calculator, self.model.policy_evaluator, self.model.value_evaluator, komi=komi)

    def genmove(self, player):
        pos = self.mcts.evaluate(400)
        if pos is not None:
            return pos
        return "pass"

    def encode_policy_output(self, pos: Union[GoPoint, str]):
        size = self.size
        node = self.mcts.node
        arr = np.zeros((size * size + 1,), dtype=np.float32)
        if node.win_rate < 0.2 or not isinstance(pos, GoPoint):
            arr[-1] = 1.0
        else:
            for child in node.children:
                pos = child.pos
                arr.itemset(pos.x * size + pos.y, child.n)
            arr **= 1 / node.t
            arr /= np.sum(arr)
        return arr

    def self_play(self, count=4):
        I, P, V = [], [], []
        for _ in tqdm.tqdm(range(count), total=count, desc="Self Playing..."):
            self.clear_board()
            player = GoPlayer.black
            while True:
                pos = self.genmove(player)
                I.append(self.model.encode_input(player, self.board))
                P.append(self.encode_policy_output(pos))
                V.append(np.array(1.0 - self.mcts.node.win_rate, dtype=np.float32))
                if isinstance(pos, GoPoint):
                    self.play(player, pos)
                    player = player.other
                else:
                    break
        return I, P, V

    def play(self, player, pos):
        super().play(player, pos)
        self.mcts.play(player, pos)

    def clear_board(self):
        super().clear_board()
        if self.model.size != self.size:
            self.model = AlphaGoWeakV0(size=self.size)
        self.mcts = GoMCTSTree(self.board, self.model.state_calculator, self.model.policy_evaluator, self.model.value_evaluator, komi=self.komi)
