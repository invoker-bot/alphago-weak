# -*- coding: utf-8 -*-
"""
@Author  : Invoker Bot
@Email   : invoker-bot@outlook.com
@Site    :
@Data    : 2021/3/25
@Version : 1.0
"""
import sys
import math
import copy
from typing import *

from ..board import *
from ..model.mcts import MCTSNode
from .gtp_random_bot import *

__all__ = ["GTPRandomBotMCTS"]


class GTPRandomBotMCTS(GTPRandomBot):
    name = "random_bot_plus"
    __version__ = "1.0"

    def __init__(self, board: GoBoardBase = None, komi=6.5):
        super().__init__(board, komi)
        self.mcts = MCTSNode()

    @staticmethod
    def policy_evaluator(b_: GoBoardBase, player: GoPlayer):
        actions = [pos for pos in b_.valid_points(player) if b_.eye_type(player, pos) < GoEyeType.unknown]
        weights = [1.0 for _ in range(len(actions))]
        return actions, weights

    def genmove(self, player: GoPlayer) -> Union[GoPoint, str]:
        if self.should_resign(player):
            return "resign"
        mcts = MCTSNode()
        point = mcts.evaluate(self.board, player, self.policy_evaluator, num=800, komi=self.komi)
        if point is not None:
            return point
        else:
            return "pass"

    def play(self, player, pos):
        super().play(player, pos)
        self.mcts = self.mcts.next(pos)

    def boardsize(self, size):
        super().boardsize(size)
        self.mcts = MCTSNode()

    def clear_board(self):
        super().clear_board()
        self.mcts = MCTSNode()
