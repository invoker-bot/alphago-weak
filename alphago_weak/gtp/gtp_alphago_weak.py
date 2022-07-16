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
from .basic import *
from ..board import *
from ..model.alpha_go_weak import AlphaGoWeakV0
from ..model.mcts import MCTSNode

__all__ = ["GTPAlphaGoWeakV0"]


class GTPAlphaGoWeakV0(GTPClientBase):
    name = "alphago_weak_v0"
    __version__ = "0.0"

    def __init__(self, board: GoBoardBase = None, komi=6.5):
        super().__init__(board, komi)
        if self.board is None:
            self.board = GoBoard()
        self.model = AlphaGoWeakV0(size=self.board.size)

    def genmove(self, player):
        mcts = MCTSNode()
        point = mcts.evaluate(self.board, player, self.model.policy_evaluator, self.model.value_evaluator, num=40, komi=self.komi)
        return point

    def boardsize(self, size: int) -> bool:
        self.model = AlphaGoWeakV0(size=size)
        return super().boardsize(size)

    def self_play(self, board_size=19, count=16):
        I, P, V = [], [], []
        for _ in tqdm.tqdm(range(count), total=count, desc="Self Playing..."):
            player = GoPlayer.black
            self.boardsize(board_size)
            while True:
                mcts = MCTSNode()
                pos = mcts.evaluate(self.board, player, self.model.policy_evaluator, self.model.value_evaluator, num=40, komi=self.komi)
                I.append(self.model.encode_input(player, self.board))
                P.append(self.model.encode_policy_output(board_size, pos))
                V.append(np.array(mcts.win_rate, dtype=np.float32))
                if pos is not None:
                    self.play(player, pos)
                    player = player.other
                else:
                    break
        print("length:", len(I))
        self.model.cache_train_data(I, P, V)
